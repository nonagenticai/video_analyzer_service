"""Background sweeper: fail analysis jobs whose owning pod is provably gone.

Before this existed, reaping was LAZY — ``JobRepository.reap_if_stranded`` ran only
inside the tenant-scoped GET handler, so **a job nobody polls was never reaped**.
Migration 004 hand-failed 23 such rows (oldest ~11 DAYS) and the count was back to 2
within seven hours, because a one-off UPDATE cannot stop re-accumulation.

The lazy design was not laziness for its own sake: under ``TENANT_RLS_MODE=enforce`` the
runtime connects as a non-bypass role, so a cross-tenant UPDATE with no
``app.current_tenant`` bound fail-closes to 0 rows. This sweeper does not fight that —
it calls ``fail_stranded_analysis_jobs``, a ``SECURITY DEFINER`` function owned by the
table owner (see ``src/database/rls_resolvers.py``), which is the same mechanism
``video_compiler_service`` uses for its reaper.

Safe on both replicas. The SQL takes ``FOR UPDATE SKIP LOCKED``, and a job with a live
lease is invisible to the victim predicate, so a concurrent sweep on the sibling pod is
a 0-row no-op rather than a lock fight or a cross-replica kill.

RE-CLAIM BEFORE FAIL (migration 007). Each sweep first drains
``reclaim_next_analysis_job``: a lapsed-lease job with attempts left is handed a fresh
lease owned by THIS pod and re-run through ``reclaim`` (``PipelineRunner.resume``); one
that has exhausted them is failed there, with the stranded message. Only then does the
fail function run, and it now leaves re-claimable rows alone — so a rollout no longer
throws away the work of the pod it replaced.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.config import PipelineSettings

logger = logging.getLogger(__name__)

_SWEEP_SQL = text(
    "SELECT fail_stranded_analysis_jobs("
    ":stranded_seconds, :max_rows, CAST(:active_job_ids AS text[]), :max_reclaims)"
)

_RECLAIM_SQL = text(
    "SELECT r_job_id, r_tenant_id, r_identity, r_attempts FROM reclaim_next_analysis_job("
    ":worker_id, :lease_seconds, :max_reclaims, CAST(:active_job_ids AS text[]))"
)


class StrandedJobSweeper:
    """Periodically fail jobs whose lease lapsed (or which went stale unclaimed)."""

    def __init__(
        self,
        settings: PipelineSettings,
        session_maker_factory: Callable[[], Awaitable[async_sessionmaker[AsyncSession]]],
        active_job_ids: Callable[[], list[str]],
        stranded_after_seconds: int,
        *,
        reclaim: Callable[[str, str, Mapping[str, object] | None], Awaitable[None]] | None = None,
        instance_id: str = "",
    ) -> None:
        self._settings = settings
        self._session_maker_factory = session_maker_factory
        self._active_job_ids = active_job_ids
        self._stranded_after_seconds = stranded_after_seconds
        self._reclaim = reclaim
        self._instance_id = instance_id
        self._task: asyncio.Task[None] | None = None

    @property
    def _max_reclaims(self) -> int:
        """Re-claims allowed per job; 0 when this sweeper cannot re-run anything.

        Without a ``reclaim`` callback there is no runner to hand a job to, so the fail
        function must keep the pre-007 fail-on-lapse rule or lapsed jobs would never end.
        """
        return self._settings.job_max_reclaims if self._reclaim is not None else 0

    @property
    def running(self) -> bool:
        """True when the sweep loop task exists and has not ended.

        A dead sweeper is invisible by construction — nothing polls it, nothing
        errors, jobs whose owning pod died simply stay ``processing`` forever
        (measured on the live DB before this loop existed: 23 stuck rows, oldest
        ~11 days). ``/health`` is the only place that can say so.
        """
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self._settings.sweeper_enabled:
            logger.info("Stranded-job sweeper disabled by config")
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        # Sweep FIRST, then sleep: a pod that boots after its predecessor crashed should
        # re-claim the orphans now, not one interval later.
        while True:
            try:
                failed = await self.sweep_once()
                if failed:
                    logger.warning("Stranded-job sweep failed %d abandoned job(s)", failed)
            except asyncio.CancelledError:
                raise
            # A single bad sweep — a DB blip, a lock timeout — must NOT kill the loop.
            # This service previously had NO sweeper at all; a loop that dies silently on
            # the first transient error is the same outcome with more moving parts.
            except Exception:
                logger.exception("Stranded-job sweep failed; continuing")
            await asyncio.sleep(self._settings.sweeper_interval_seconds)

    async def sweep_once(self) -> int:
        """Run one sweep: re-claim what can be re-run, then fail what cannot.

        Returns how many rows were FAILED (re-claims are logged, not counted), so the
        loop's "failed N abandoned job(s)" warning keeps meaning what it says.
        """
        failed = await self.reclaim_once()
        session_maker = await self._session_maker_factory()
        async with session_maker() as session:
            # NO bind_session_tenant here, deliberately: the whole point is a
            # cross-tenant sweep, and the SECURITY DEFINER function is what makes that
            # legal. Binding a tenant would silently scope the sweep to one tenant.
            result = await session.execute(
                _SWEEP_SQL,
                {
                    "stranded_seconds": self._stranded_after_seconds,
                    "max_rows": self._settings.sweeper_max_per_sweep,
                    "active_job_ids": list(self._active_job_ids()),
                    "max_reclaims": self._max_reclaims,
                },
            )
            await session.commit()
            return failed + int(result.scalar_one() or 0)

    async def reclaim_once(self) -> int:
        """Re-claim lapsed-lease jobs one at a time. Returns how many were poison-failed.

        One row per transaction: the claim (``FOR UPDATE SKIP LOCKED`` + the lease
        stamp) commits BEFORE the job is started, so a sibling pod sweeping at the same
        moment can never take the same row, and a crash between claim and start just
        leaves a row with a lease that will lapse and be re-claimed again.
        """
        reclaim = self._reclaim
        if reclaim is None or self._max_reclaims <= 0:
            return 0
        session_maker = await self._session_maker_factory()
        poisoned = 0
        for _ in range(self._settings.sweeper_max_per_sweep):
            async with session_maker() as session:
                row = (
                    await session.execute(
                        _RECLAIM_SQL,
                        {
                            "worker_id": self._instance_id,
                            "lease_seconds": self._settings.job_lease_seconds,
                            "max_reclaims": self._max_reclaims,
                            "active_job_ids": list(self._active_job_ids()),
                        },
                    )
                ).first()
                await session.commit()
            if row is None:
                break
            job_id, tenant_id, identity, attempts = row
            if job_id is None:
                # Sentinel: a job out of attempts was failed in-function. Keep draining —
                # healthy orphans may be queued behind it.
                poisoned += 1
                logger.warning(
                    "Stranded job exhausted %s re-claim(s); failed as stranded", attempts
                )
                continue
            logger.warning(
                "Re-claimed job %s (tenant %s) whose owner is gone; re-running (attempt %s)",
                job_id,
                tenant_id,
                attempts,
            )
            try:
                await reclaim(str(job_id), str(tenant_id), identity)
            except Exception:
                # The row holds OUR lease but nothing runs it; it lapses and is
                # re-claimed (spending an attempt), so this degrades, never wedges.
                logger.exception("Could not start re-claimed job %s", job_id)
        return poisoned
