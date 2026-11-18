"""Repository for AnalysisJob CRUD + progress updates.

All read/write methods that look up a row by ``job_id`` require a
``tenant_id`` keyword argument; the tenant filter is enforced inside the
SQL ``WHERE`` clause. A wrong-tenant lookup returns None / a wrong-tenant
write affects zero rows.

Under Phase D RLS the session's ``app.current_tenant`` GUC must be bound
(``bind_session_tenant``) before any access — the caller owns binding. The
background pipeline receives its ``tenant_id`` from the enqueue call (the
request handler that created the job knows it), so there is no unchecked
cross-tenant entry point: every pipeline session binds the GUC and uses
the tenant-scoped lookups, same as the request path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import AnalysisJob


def _tenant(tenant_id: str | uuid.UUID) -> uuid.UUID:
    """Coerce the caller's tenant to ``uuid.UUID`` at the repository boundary.

    ``AnalysisJob.tenant_id`` became a native ``uuid`` in migration 006, but
    production still hands this layer a ``str`` from the auth middleware. The
    generated SQL is fine either way -- psycopg casts it -- so this looks
    harmless. It is not: SQLAlchemy's ORM-enabled ``update()`` synchronises the
    session with ``synchronize_session="evaluate"``, which compares IN PYTHON.
    ``uuid.UUID(x) == "x"`` is ``False``, so the in-session object is never
    updated and every subsequent ORM read returns STALE state -- the row is
    correct in the database while the API keeps reporting the old status
    (a poller told "pending" forever). Coerce once, here.
    """
    return tenant_id if isinstance(tenant_id, uuid.UUID) else uuid.UUID(tenant_id)


# The complete terminal vocabulary of ``analysis_jobs.status``. The live
# vocabulary is exactly {pending, completed, failed} (see ``create`` /
# ``mark_completed`` / ``mark_failed`` — there is no ``cancelled`` state in this
# service), and the two below are the states a job never leaves.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed"})

# Deterministically ordered form for use in a SQL ``NOT IN (...)`` clause; a
# frozenset has no stable iteration order, which would churn the rendered SQL
# (and any statement cache keyed on it) between processes.
_TERMINAL_STATUS_TUPLE: tuple[str, ...] = tuple(sorted(TERMINAL_STATUSES))


# How long a NON-terminal row may sit without any write before it is treated as
# abandoned. A healthy job writes continuously — ``update_progress`` stamps
# ``updated_at`` on every stage transition — so staleness, not age, is the signal.
#
# Sized from the live corpus (7 days to 2026-08-04): 210 completed jobs, p50 16.7s,
# p90 108.6s, max 176.2s. One hour is ~20x the slowest job that has ever finished
# here, so it cannot reap work that is merely slow, while still resolving the
# zombies measured the same day: 23 rows stuck non-terminal, the oldest ~11 DAYS.
STRANDED_AFTER_SECONDS: int = 3600


class JobRepository:
    """Encapsulates all DB access for AnalysisJob rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: str | uuid.UUID,
        video_path: str | None,
        video_url: str | None,
        upload_filename: str | None,
        request_config: dict[str, Any],
        verification_criteria: dict[str, Any] | None,
        dispatch_identity: dict[str, str] | None = None,
    ) -> AnalysisJob:
        """Insert a new pending analysis job.

        ``dispatch_identity`` is the submitting request's identity snapshot
        (``services.job_identity.capture_dispatch_identity``); a re-claim on another pod
        needs it to make metered calls as the original caller.
        """
        job = AnalysisJob(
            dispatch_identity=dispatch_identity,
            tenant_id=_tenant(tenant_id),
            video_path=video_path,
            video_url=video_url,
            upload_filename=upload_filename,
            request_config=request_config,
            verification_criteria=verification_criteria,
            status="pending",
            stage="queued",
            progress=0.0,
        )
        self.session.add(job)
        # flush+refresh BEFORE commit: the RLS GUC is transaction-scoped
        # (set_config is_local=true), so a refresh after commit runs in a new
        # txn with no tenant bound and fail-closes ("could not refresh
        # instance"). Materialize defaults inside the bound transaction.
        await self.session.flush()
        await self.session.refresh(job)
        await self.session.commit()
        return job

    async def get_by_job_id(self, job_id: str, *, tenant_id: str | uuid.UUID) -> AnalysisJob | None:
        """Fetch a job by its public job_id UUID, scoped to the caller's tenant."""
        stmt = select(AnalysisJob).where(
            AnalysisJob.job_id == job_id,
            AnalysisJob.tenant_id == _tenant(tenant_id),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_progress(
        self,
        job_id: str,
        *,
        tenant_id: str | uuid.UUID,
        stage: str,
        progress: float,
    ) -> None:
        """Advance stage + progress without touching other fields."""
        await self.session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.job_id == job_id,
                AnalysisJob.tenant_id == _tenant(tenant_id),
            )
            .values(stage=stage, progress=progress, updated_at=datetime.now(UTC))
        )
        await self.session.commit()

    async def update_fields(
        self, job_id: str, *, tenant_id: str | uuid.UUID, **fields: object
    ) -> None:
        """Apply arbitrary field updates to a job row (tenant-scoped)."""
        await self.session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.job_id == job_id,
                AnalysisJob.tenant_id == _tenant(tenant_id),
            )
            .values(updated_at=datetime.now(UTC), **fields)
        )
        await self.session.commit()

    async def claim(
        self, job_id: str, *, tenant_id: str | uuid.UUID, instance_id: str, lease_seconds: int
    ) -> bool:
        """Take ownership of a non-terminal job for this pod. Returns True if claimed.

        Stamps ``instance_id`` + ``lease_expires_at`` so the sweeper and
        ``reap_if_stranded`` can tell "a live pod is running this" from "its worker is
        gone". Guarded on non-terminal status so a claim can never revive a finished row.
        """
        now = datetime.now(UTC)
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                update(AnalysisJob)
                .where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == _tenant(tenant_id),
                    AnalysisJob.status.not_in(_TERMINAL_STATUS_TUPLE),
                )
                .values(
                    instance_id=instance_id,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            ),
        )
        await self.session.commit()
        return result.rowcount == 1

    async def renew_lease(
        self, job_id: str, *, tenant_id: str | uuid.UUID, instance_id: str, lease_seconds: int
    ) -> bool:
        """Push this pod's lease on *job_id* forward (heartbeat). True if still owned.

        ``instance_id`` is in the WHERE clause deliberately: a pod that has LOST the claim
        — because the sweeper reaped it and a resubmit re-claimed it elsewhere — must stop
        renewing rather than extend a lease it no longer holds. The caller uses the False
        return to stop heartbeating.

        Note this does NOT touch ``updated_at``: a heartbeat is liveness, not progress.
        Bumping ``updated_at`` here would make every job look permanently fresh and
        silently disable the legacy staleness rule for pre-005 rows.
        """
        now = datetime.now(UTC)
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                update(AnalysisJob)
                .where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == _tenant(tenant_id),
                    AnalysisJob.instance_id == instance_id,
                    AnalysisJob.status.not_in(_TERMINAL_STATUS_TUPLE),
                )
                .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
            ),
        )
        await self.session.commit()
        return result.rowcount == 1

    async def release_lease(
        self, job_id: str, *, tenant_id: str | uuid.UUID, instance_id: str
    ) -> bool:
        """Give back this pod's lease on graceful shutdown so a live pod re-claims NOW.

        Without it a SIGTERM'd pod's jobs sit until the lease TTL runs out (300s by
        default) before anyone re-runs them. ``lease_released`` tells the re-claim that
        this was a drain, not a crash, so it does not spend one of the job's attempts.
        Owner-fenced on ``instance_id`` like ``renew_lease``: a pod can only give back a
        lease it still holds.
        """
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                update(AnalysisJob)
                .where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == _tenant(tenant_id),
                    AnalysisJob.instance_id == instance_id,
                    AnalysisJob.status.not_in(_TERMINAL_STATUS_TUPLE),
                )
                # DATABASE clock, and a second in the past: the re-claim compares against
                # the DB's ``now()``, so a pod clock running ahead of the DB would
                # otherwise stamp a lease that is still "live" to the next sweep.
                .values(
                    lease_expires_at=func.now() - timedelta(seconds=1),
                    lease_released=True,
                )
            ),
        )
        await self.session.commit()
        return result.rowcount == 1

    async def mark_completed(self, job_id: str, *, tenant_id: str | uuid.UUID) -> bool:
        """Flip job to completed with completed_at set. Returns True if THIS call won.

        GUARDED, exactly like its sibling ``mark_failed`` — and it was not, which was a
        live double-bill. The old implementation delegated to ``update_fields``: an
        unconditional ``UPDATE ... SET status='completed'`` with only job_id + tenant_id
        predicates. Concrete failure it allowed:

        a large video stalls past ``STRANDED_AFTER_SECONDS`` inside the vision stage
        (which writes no progress between frames) → a client poll reaps it to ``failed``
        → the client resubmits (a second PAID analysis) → the still-live original then
        finishes and unconditionally flips the row back to ``completed``/``progress=1.0``
        with the strand error still attached — and ``pipeline._emit_cost`` billed a
        SECOND time for the same job_id, because it had no won-boolean to gate on.

        ``video_compiler_service`` gets this right (``api/routes/compile.py`` does
        ``won = await _stamp_completed(...); if won: _emit_cost(...)``); this mirrors it.

        Like ``mark_failed`` this is a SINGLE conditional statement, not a read-then-write:
        Postgres evaluates ``status NOT IN (...)`` against the row it locks, so exactly one
        concurrent writer can win. It also NULLs the lease — a terminal row owns nothing.
        """
        now = datetime.now(UTC)
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                update(AnalysisJob)
                .where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == _tenant(tenant_id),
                    AnalysisJob.status.not_in(_TERMINAL_STATUS_TUPLE),
                )
                .values(
                    status="completed",
                    stage="completed",
                    progress=1.0,
                    completed_at=now,
                    updated_at=now,
                    lease_expires_at=None,
                )
            ),
        )
        await self.session.commit()
        return result.rowcount == 1

    async def mark_failed(
        self, job_id: str, error_message: str, *, tenant_id: str | uuid.UUID
    ) -> bool:
        """Flip job to failed with error message captured, and STAMP ``completed_at``.

        GUARDED: a row that is already in a terminal state (``TERMINAL_STATUSES``)
        is left completely untouched and ``False`` is returned. Without the guard
        an unconditional ``UPDATE ... SET status='failed'`` can DEMOTE a finished
        job — e.g. a late cleanup path firing after the pipeline already wrote
        ``completed`` would rewrite a good result as a failure, and re-marking an
        already-failed job would clobber the first (real) error message with a
        generic one. Returns True only when this call moved the row.

        The guard is a SINGLE conditional statement, not a read-then-write. An
        earlier ``get_by_job_id`` + ``update_fields`` pair was a TOCTOU: two
        concurrent ``mark_failed`` calls both read ``pending`` and both UPDATEd
        (the real error message clobbered by the generic one), and a
        ``mark_completed`` landing between the SELECT and the UPDATE was still
        demoted. Postgres evaluates the ``status NOT IN (...)`` predicate against
        the row it locks, so exactly one concurrent writer can win.

        ``failed`` is terminal exactly like ``completed``, so it must carry a completion
        timestamp for the same reason. Without it the row is indistinguishable from a job
        that is still running: measured on the live DB 2026-08-04, 22 rows sat at
        ``status='failed'`` with ``completed_at IS NULL``, the oldest ~25 days old.

        Two consequences, both observed: every duration statistic keyed on
        ``completed_at`` silently EXCLUDES failures (so the analyzer's timing profile was
        built from successes only), and any "is it finished?" check written as
        ``completed_at IS NULL`` treats a long-dead job as forever in-flight.
        """
        now = datetime.now(UTC)
        # Tenant scoping is identical to every other write here: an explicit
        # ``tenant_id`` predicate in the WHERE clause, on top of RLS.
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                update(AnalysisJob)
                .where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == _tenant(tenant_id),
                    AnalysisJob.status.not_in(_TERMINAL_STATUS_TUPLE),
                )
                .values(
                    status="failed",
                    error_message=error_message,
                    completed_at=now,
                    updated_at=now,
                )
            ),
        )
        await self.session.commit()
        return result.rowcount == 1

    async def reap_if_stranded(
        self,
        job: AnalysisJob,
        *,
        tenant_id: str | uuid.UUID,
        stranded_after_seconds: int = STRANDED_AFTER_SECONDS,
        max_reclaims: int = 0,
    ) -> bool:
        """Fail a non-terminal job that stopped being written to. Returns True if reaped.

        RE-CLAIM FIRST (migration 007). With ``max_reclaims > 0`` a lapsed lease is NOT a
        death sentence: while the job has attempts left (or its lease was released on a
        graceful drain) the background re-claim on a live pod will pick it up and re-run
        it, so this poll leaves it alone and the caller keeps seeing ``pending``. Failing
        it here would race that re-claim and throw the work away — which is exactly what
        a rollout did to job 01a1642c on 2026-10-22. ``max_reclaims=0`` is the pre-007
        behaviour (fail on lapse), used when the sweeper/re-claim is disabled.

        A job is owned by an in-process background task. If the POD dies mid-run — a
        rollout, an eviction, an OOM — no handler runs, so the row keeps whatever
        non-terminal status it had, forever. ``_run_job``'s broad ``except`` guarantees a
        terminal state only when the process survives to execute it.

        Measured on the live DB 2026-08-04: 23 rows stuck non-terminal, the oldest ~11
        DAYS, across stages ``scene_detection`` / ``vision`` / ``probe`` / ``quality``.
        A client polling any of those waits forever on a job nobody is running.

        OWNERSHIP FENCE (migration 005). A CLAIMED job is judged on its LEASE, not on
        wall-clock staleness — because staleness is a weak signal here: ``updated_at``
        only moves on a stage transition and the vision stage runs for minutes writing
        nothing between frames. At ``replicas: 2`` that weakness is a correctness bug:
        pod B's poll could fail a job pod A is actively running. A lease cannot be
        confused that way — the owner renews it every ``lease/3``, so three consecutive
        beats must be missed before the row looks abandoned.

        An UNCLAIMED job (``lease_expires_at IS NULL`` — a row created before 005, or one
        that died between INSERT and the claim inside ``_execute``) keeps the ORIGINAL
        1-hour staleness rule, so nothing regresses and no pre-existing row becomes
        un-reapable.

        Reaps through ``mark_failed``, so it inherits that method's terminal guard: a job
        that reached ``completed`` between the read and this write is left untouched and
        this returns False. This stays the LAZY path (the tenant-scoped status route);
        the background sweep that covers jobs nobody polls now lives in
        ``services/stranded_sweeper.py`` and runs through a SECURITY DEFINER function,
        which is how it reads across tenants without the privilege this role correctly
        lacks.
        """
        if job.status in TERMINAL_STATUSES:
            return False

        lease = job.lease_expires_at
        if lease is not None:
            if lease.tzinfo is None:
                lease = lease.replace(tzinfo=UTC)
            if lease >= datetime.now(UTC):
                # Claimed and heartbeating — a LIVE worker owns this, possibly on the
                # sibling replica. Wall-clock staleness is irrelevant and must not be
                # consulted, or a long silent vision stage gets killed under itself.
                return False
            if max_reclaims > 0 and (job.lease_released or job.reclaim_attempts < max_reclaims):
                return False
            reason = (
                f"stranded: lease lapsed at {lease.isoformat()} at stage "
                f"{job.stage or 'unknown'!s} — the worker that owned this job is gone "
                "(pod restart/eviction). Resubmit the analysis."
            )
            return await self.mark_failed(job.job_id, reason, tenant_id=tenant_id)

        last_write = job.updated_at or job.created_at
        if last_write is None:
            return False
        if last_write.tzinfo is None:
            last_write = last_write.replace(tzinfo=UTC)
        idle_seconds = (datetime.now(UTC) - last_write).total_seconds()
        if idle_seconds < stranded_after_seconds:
            return False
        return await self.mark_failed(
            job.job_id,
            (
                f"stranded: no progress for {int(idle_seconds)}s at stage "
                f"{job.stage or 'unknown'!s} — the worker that owned this job is gone "
                "(pod restart/eviction). Resubmit the analysis."
            ),
            tenant_id=_tenant(tenant_id),
        )

    async def delete(self, job_id: str, *, tenant_id: str | uuid.UUID) -> bool:
        """Delete a job row for the given tenant; returns True if a row was removed."""
        job = await self.get_by_job_id(job_id, tenant_id=tenant_id)
        if job is None:
            return False
        await self.session.delete(job)
        await self.session.commit()
        return True
