"""Ownership lease: the fence, the terminal guard, and the billing gate (migration 005).

Three defects this covers, all measured or traced on ``origin/main`` before the fix:

1. ``reap_if_stranded`` had NO ownership fence — it judged purely on ``updated_at``
   staleness. At ``replicas: 2`` (live-verified) pod B's poll could therefore fail a job
   pod A was actively running, because the vision stage writes no progress between frames.

2. ``mark_completed`` delegated to ``update_fields`` — an UNCONDITIONAL
   ``UPDATE ... SET status='completed'`` — while its sibling ``mark_failed`` carefully
   guarded ``status NOT IN ('completed','failed')``. A reaped job whose original task
   later finished was resurrected to ``completed``.

3. ``pipeline._emit_cost`` fired unconditionally after that unguarded write, so the
   resurrection also BILLED A SECOND TIME for the same job_id.

These are DB tests: they run against a real Postgres, because the guard being asserted is
a SQL ``WHERE`` predicate. A mocked session would assert the mock, not the behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from src.repository.job_repository import STRANDED_AFTER_SECONDS, JobRepository

pytestmark = pytest.mark.asyncio

TENANT = "11111111-1111-1111-1111-111111111111"


async def _job(session, **overrides):
    """Create a job row, optionally backdating/leasing it.

    Overrides go through raw SQL rather than ``update_fields``: that helper stamps
    ``updated_at`` itself (correctly — it is a progress write), so it cannot be used to
    BACKDATE a row, which is exactly what the staleness branch needs to be tested.
    """
    repo = JobRepository(session)
    job = await repo.create(
        tenant_id=TENANT,
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
    return repo, job.job_id


async def test_live_lease_blocks_the_reap_even_when_updated_at_is_ancient(db_session) -> None:
    """THE FENCE. A heartbeating job must survive an arbitrarily stale ``updated_at``.

    This is the exact production shape: a long vision stage writes nothing between frames,
    so ``updated_at`` goes hours stale while the job is perfectly healthy. Without the
    lease branch, any client poll (or the sibling replica's) reaps it and the client pays
    for a second analysis.
    """
    repo, job_id = await _job(
        db_session,
        updated_at=datetime.now(UTC) - timedelta(seconds=STRANDED_AFTER_SECONDS * 5),
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=300),
        instance_id="pod-a",
    )
    job = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert job is not None

    assert await repo.reap_if_stranded(job, tenant_id=TENANT) is False

    refreshed = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert refreshed is not None
    assert refreshed.status == "pending"


async def test_lapsed_lease_is_reaped_even_when_updated_at_is_fresh(db_session) -> None:
    """The other half: a lapsed lease means the pod is gone, whatever ``updated_at`` says.

    A dead pod stops heartbeating, so this fires within ``job_lease_seconds`` instead of
    the hour the wall-clock rule needs.
    """
    repo, job_id = await _job(
        db_session,
        updated_at=datetime.now(UTC),
        lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
        instance_id="dead-pod",
    )
    job = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert job is not None

    assert await repo.reap_if_stranded(job, tenant_id=TENANT) is True

    refreshed = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert "lease lapsed" in (refreshed.error_message or "")


async def test_unclaimed_row_keeps_the_legacy_staleness_rule(db_session) -> None:
    """Rows written before migration 005 must not become un-reapable.

    ``lease_expires_at IS NULL`` falls through to the original 1-hour wall-clock branch,
    so the 23 pre-existing zombies (oldest ~11 days) are still caught.
    """
    repo, job_id = await _job(
        db_session,
        updated_at=datetime.now(UTC) - timedelta(seconds=STRANDED_AFTER_SECONDS + 60),
    )
    job = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert job is not None

    assert await repo.reap_if_stranded(job, tenant_id=TENANT) is True

    refreshed = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert "no progress" in (refreshed.error_message or "")


async def test_mark_completed_cannot_resurrect_a_reaped_job(db_session) -> None:
    """THE TERMINAL GUARD. A late completion must not overwrite a terminal row.

    Sequence, straight from the failure trace: job strands -> poll reaps it to ``failed``
    -> client resubmits (second paid analysis) -> the ORIGINAL task finally finishes and
    calls ``mark_completed``. Before the guard that flipped the row back to ``completed``
    with the strand error still attached.
    """
    repo, job_id = await _job(db_session)
    assert await repo.mark_failed(job_id, "stranded: reaped", tenant_id=TENANT) is True

    assert await repo.mark_completed(job_id, tenant_id=TENANT) is False

    refreshed = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_message == "stranded: reaped"


async def test_mark_completed_returns_true_on_the_happy_path(db_session) -> None:
    """Polarity guard. If this boolean were inverted, billing would silently STOP.

    Asserting only the False case would let an inverted return look correct, so the happy
    path is asserted explicitly alongside it.
    """
    repo, job_id = await _job(db_session)

    assert await repo.mark_completed(job_id, tenant_id=TENANT) is True

    refreshed = await repo.get_by_job_id(job_id, tenant_id=TENANT)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.progress == 1.0
    assert refreshed.completed_at is not None
    # A terminal row owns nothing — the lease must be released.
    assert refreshed.lease_expires_at is None


async def test_claim_and_renew_are_owner_fenced(db_session) -> None:
    """A pod that does not own the row cannot renew its lease.

    Without ``instance_id`` in the renew WHERE clause, a pod that lost its claim would
    keep pushing the lease forward and make the row permanently un-reapable.
    """
    repo, job_id = await _job(db_session)

    assert (
        await repo.claim(job_id, tenant_id=TENANT, instance_id="pod-a", lease_seconds=300) is True
    )
    assert (
        await repo.renew_lease(job_id, tenant_id=TENANT, instance_id="pod-a", lease_seconds=300)
        is True
    )
    assert (
        await repo.renew_lease(job_id, tenant_id=TENANT, instance_id="pod-b", lease_seconds=300)
        is False
    )


async def test_claim_cannot_revive_a_terminal_row(db_session) -> None:
    """Claiming is guarded on non-terminal status, same as every other write here."""
    repo, job_id = await _job(db_session)
    assert await repo.mark_completed(job_id, tenant_id=TENANT) is True

    assert (
        await repo.claim(job_id, tenant_id=TENANT, instance_id="pod-a", lease_seconds=300) is False
    )
