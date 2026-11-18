"""The background sweeper: cross-tenant, lease-aware, and it must not die on one error.

Why this exists at all: before migration 005 this service had NO reaper loop.
``reap_if_stranded`` ran only inside the tenant-scoped GET handler, so **a job nobody
polls was never reaped** — measured on the live DB 2026-08-04 as 23 rows stuck
non-terminal, oldest ~11 DAYS. Migration 004 hand-failed them and the count was back to 2
within seven hours.

The reason a sweeper did not already exist was real: under ``TENANT_RLS_MODE=enforce``
the runtime is a non-bypass role, so a cross-tenant UPDATE with no ``app.current_tenant``
bound fail-closes to 0 rows. The answer is the ``SECURITY DEFINER`` function these tests
exercise, which is the same mechanism ``video_compiler_service``'s reaper uses.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import PipelineSettings
from src.repository.job_repository import STRANDED_AFTER_SECONDS, JobRepository
from src.services.stranded_sweeper import StrandedJobSweeper
from tests.conftest import TENANT_A, TENANT_B

pytestmark = pytest.mark.asyncio


async def _job(session, tenant: str, **overrides) -> str:
    job = await JobRepository(session).create(
        tenant_id=tenant,
        video_path=None,
        video_url="https://example.com/v.mp4",
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    if overrides:
        assignments = ", ".join(f"{col} = :{col}" for col in overrides)
        await session.execute(
            text(f"UPDATE analysis_jobs SET {assignments} WHERE job_id = :job_id"),  # noqa: S608
            {**overrides, "job_id": job.job_id},
        )
        await session.commit()
    return job.job_id


async def _status(session, job_id: str) -> str:
    row = await session.execute(
        text("SELECT status FROM analysis_jobs WHERE job_id = :j"), {"j": job_id}
    )
    return str(row.scalar_one())


async def test_sweeper_reaps_across_tenants_and_spares_live_leases(
    db_session, db_session_factory
) -> None:
    """The whole point, in one test: cross-tenant reach, and the lease as the fence.

    Four rows across TWO tenants with NO ``app.current_tenant`` bound on the sweeping
    session. A plain UPDATE would be RLS-scoped (and under enforce would touch zero rows);
    the SECURITY DEFINER function reaches all of them — but must still spare the one whose
    owner is heartbeating, and the one that is merely young.
    """
    now = datetime.now(UTC)
    lapsed_a = await _job(
        db_session, TENANT_A, lease_expires_at=now - timedelta(seconds=1), instance_id="dead"
    )
    lapsed_b = await _job(
        db_session, TENANT_B, lease_expires_at=now - timedelta(seconds=1), instance_id="dead"
    )
    live = await _job(
        db_session, TENANT_A, lease_expires_at=now + timedelta(seconds=300), instance_id="alive"
    )
    young = await _job(db_session, TENANT_B)  # unclaimed, fresh updated_at

    sweeper = StrandedJobSweeper(
        PipelineSettings(),
        AsyncMock(return_value=db_session_factory),
        lambda: [],
        STRANDED_AFTER_SECONDS,
    )

    assert await sweeper.sweep_once() == 2

    assert await _status(db_session, lapsed_a) == "failed"
    assert await _status(db_session, lapsed_b) == "failed"
    assert await _status(db_session, live) == "pending"
    assert await _status(db_session, young) == "pending"


async def test_sweeper_never_touches_a_job_running_on_this_pod(
    db_session, db_session_factory
) -> None:
    """``active_job_ids`` closes the INSERT-to-claim window.

    A job this process is running but has not claimed yet (or whose claim failed) has no
    lease, so only ``active_job_ids`` protects it. Without that argument the sweeper could
    fail a job whose asyncio task is running right beside it.
    """
    stale = await _job(
        db_session,
        TENANT_A,
        updated_at=datetime.now(UTC) - timedelta(seconds=STRANDED_AFTER_SECONDS + 60),
    )

    sweeper = StrandedJobSweeper(
        PipelineSettings(),
        AsyncMock(return_value=db_session_factory),
        lambda: [stale],
        STRANDED_AFTER_SECONDS,
    )

    assert await sweeper.sweep_once() == 0
    assert await _status(db_session, stale) == "pending"


async def test_sweeper_reaps_a_stale_unclaimed_row(db_session, db_session_factory) -> None:
    """Pre-005 rows (no lease) are still caught, on the original 1-hour rule.

    This is the class migration 004 had to mop by hand. If the sweeper only understood
    leases, every row written before this deploy would be permanently un-reapable.
    """
    stale = await _job(
        db_session,
        TENANT_A,
        updated_at=datetime.now(UTC) - timedelta(seconds=STRANDED_AFTER_SECONDS + 60),
    )

    sweeper = StrandedJobSweeper(
        PipelineSettings(),
        AsyncMock(return_value=db_session_factory),
        lambda: [],
        STRANDED_AFTER_SECONDS,
    )

    assert await sweeper.sweep_once() == 1
    assert await _status(db_session, stale) == "failed"


async def test_deployed_function_is_security_definer_with_a_pinned_search_path(
    db_session,
) -> None:
    """Assert the SECURITY PROPERTIES of the function actually in the database.

    HONEST SCOPE: the cross-tenant test above runs as the table OWNER, which bypasses
    ENABLE'd RLS policies anyway — so it proves the function reaps across tenants, but it
    can NOT prove that ``SECURITY DEFINER`` is what makes that legal in prod, where the
    runtime is the non-bypass ``w2v_video_analyzer_app`` role. Reproducing enforce mode
    would need that role, which CI deliberately does not create.

    So this asserts the two properties directly instead:

    * ``prosecdef`` — without it, the prod sweep would fail-closed to 0 rows under
      ``TENANT_RLS_MODE=enforce`` and the service would silently have no reaper again,
      with a green test suite.
    * a pinned ``search_path`` in ``proconfig`` — a SECURITY DEFINER function with a
      mutable search_path is a privilege-escalation vector, and it is also what makes the
      function resolve ``analysis_jobs`` correctly under CI's ``postgres_isolation:
      schema``.
    """
    row = await db_session.execute(
        text(
            "SELECT prosecdef, proconfig FROM pg_proc WHERE proname = 'fail_stranded_analysis_jobs'"
        )
    )
    secdef, proconfig = row.one()
    assert secdef is True, "sweeper function lost SECURITY DEFINER — prod sweeps would 0-row"
    assert any(str(c).startswith("search_path=") for c in (proconfig or [])), (
        f"sweeper function has no pinned search_path (proconfig={proconfig})"
    )


async def test_loop_survives_a_failing_sweep() -> None:
    """One bad sweep must not kill the loop.

    A loop that dies on the first transient DB error leaves the service in exactly the
    state this whole change is fixing — no reaper — but with more moving parts and a
    silent failure mode. Both calls must happen.
    """
    calls: list[int] = []

    # The failure is injected through the sweeper's real DEPENDENCY rather than by
    # monkeypatching ``sweep_once`` itself — so the loop's own error handling is what is
    # under test, not a stub standing in for it.
    async def _flaky_factory() -> async_sessionmaker[AsyncSession]:
        calls.append(1)
        if len(calls) == 1:
            raise SQLAlchemyError("transient")
        raise asyncio.CancelledError

    sweeper = StrandedJobSweeper(
        PipelineSettings(sweeper_interval_seconds=0.01),
        _flaky_factory,
        lambda: [],
        STRANDED_AFTER_SECONDS,
    )
    sweeper.start()
    for _ in range(200):
        if len(calls) >= 2:
            break
        await asyncio.sleep(0.01)
    await sweeper.stop()

    assert len(calls) >= 2, "the loop stopped after the first failing sweep"


async def test_sweeper_disabled_by_config_starts_no_task() -> None:
    """The kill switch must actually be a kill switch."""
    sweeper = StrandedJobSweeper(
        PipelineSettings(sweeper_enabled=False),
        AsyncMock(),
        lambda: [],
        STRANDED_AFTER_SECONDS,
    )
    sweeper.start()
    assert sweeper._task is None
    await sweeper.stop()
