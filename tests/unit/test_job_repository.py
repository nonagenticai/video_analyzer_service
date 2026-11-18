"""JobRepository CRUD tests against the postgres CI database fixture."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.job_repository import JobRepository
from tests.conftest import TENANT_A


async def test_create_and_fetch(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path="/tmp/video.mp4",
        video_url=None,
        upload_filename=None,
        request_config={"generate_walkthrough": True},
        verification_criteria=None,
    )
    assert job.job_id
    assert str(job.tenant_id) == TENANT_A  # native uuid column -> UUID on read
    assert job.status == "pending"
    assert job.stage == "queued"

    fetched = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert fetched is not None
    assert fetched.id == job.id


async def test_update_progress(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path="/tmp/video.mp4",
        video_url=None,
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    await repo.update_progress(job.job_id, tenant_id=TENANT_A, stage="frame_extract", progress=0.2)
    refreshed = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert refreshed is not None
    assert refreshed.stage == "frame_extract"
    assert refreshed.progress == 0.2


async def test_update_progress_missing_job(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    await repo.update_progress(
        "does-not-exist", tenant_id=TENANT_A, stage="x", progress=0.0
    )  # no-op


async def test_update_fields_sets_arbitrary_values(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path=None,
        video_url="https://example.com/v.mp4",
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    await repo.update_fields(
        job.job_id,
        tenant_id=TENANT_A,
        duration_seconds=42.5,
        width=1920,
        height=1080,
        has_audio=1,
        transcript_data={"text": "hello"},
    )
    refreshed = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert refreshed is not None
    assert refreshed.duration_seconds == 42.5
    assert refreshed.width == 1920
    assert refreshed.has_audio == 1
    assert refreshed.transcript_data == {"text": "hello"}


async def test_mark_completed(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path="/tmp/a.mp4",
        video_url=None,
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    await repo.mark_completed(job.job_id, tenant_id=TENANT_A)
    refreshed = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.progress == 1.0
    assert refreshed.completed_at is not None


async def test_mark_failed(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path="/tmp/a.mp4",
        video_url=None,
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    await repo.mark_failed(job.job_id, "ffmpeg crashed", tenant_id=TENANT_A)
    refreshed = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_message == "ffmpeg crashed"
    # `failed` is terminal, so it must be timestamped like `completed`. This
    # assertion was ABSENT while 22 live rows sat at status='failed' with a NULL
    # completed_at (oldest ~25 days) — the test passed the whole time.
    assert refreshed.completed_at is not None


async def test_both_terminal_states_are_timestamped(db_session: AsyncSession) -> None:
    """Symmetry, asserted directly: a reader cannot tell a dead job from a running one
    unless BOTH terminal transitions stamp completed_at.

    Guards the two concrete consequences of the asymmetry: duration statistics keyed on
    completed_at silently exclude failures, and `completed_at IS NULL` — the obvious way
    to ask "still running?" — reports a long-dead job as in-flight forever.
    """
    repo = JobRepository(db_session)

    async def _job() -> str:
        job = await repo.create(
            tenant_id=TENANT_A,
            video_path="/tmp/a.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        return job.job_id

    ok_id, bad_id = await _job(), await _job()
    await repo.mark_completed(ok_id, tenant_id=TENANT_A)
    await repo.mark_failed(bad_id, "ffmpeg crashed", tenant_id=TENANT_A)

    ok = await repo.get_by_job_id(ok_id, tenant_id=TENANT_A)
    bad = await repo.get_by_job_id(bad_id, tenant_id=TENANT_A)
    assert ok is not None and bad is not None
    assert {ok.status, bad.status} == {"completed", "failed"}
    # The invariant, stated as the query a consumer would actually write.
    assert [j for j in (ok, bad) if j.completed_at is None] == []


async def test_delete_returns_bool(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    job = await repo.create(
        tenant_id=TENANT_A,
        video_path="/tmp/a.mp4",
        video_url=None,
        upload_filename=None,
        request_config={},
        verification_criteria=None,
    )
    assert await repo.delete(job.job_id, tenant_id=TENANT_A) is True
    assert await repo.delete(job.job_id, tenant_id=TENANT_A) is False


async def test_update_fields_missing_noop(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    await repo.update_fields("nope", tenant_id=TENANT_A, status="failed")  # no error


async def test_mark_failed_missing_noop(db_session: AsyncSession) -> None:
    repo = JobRepository(db_session)
    await repo.mark_failed("nope", "gone", tenant_id=TENANT_A)


async def test_migration_003_backfills_a_stranded_failed_row(db_session: AsyncSession) -> None:
    """Executes migration 003's REAL statement against a row in the exact broken shape.

    A migration asserted only by "alembic upgrade head didn't error" proves nothing about
    whether it repaired anything. This plants the defect (terminal `failed`, NULL
    `completed_at` — the shape of the 22 live rows found 2026-08-04), runs the statement,
    and asserts the repair, the idempotency, and that non-terminal rows are left alone.
    """
    from pathlib import Path

    from sqlalchemy import text

    repo = JobRepository(db_session)

    async def _job() -> str:
        job = await repo.create(
            tenant_id=TENANT_A,
            video_path="/tmp/a.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        return job.job_id

    broken_id, pending_id = await _job(), await _job()
    # Recreate the pre-fix write exactly: status flips, completed_at does not.
    await db_session.execute(
        text(
            "UPDATE analysis_jobs SET status='failed', error_message='x', completed_at=NULL "
            "WHERE job_id = :j"
        ),
        {"j": broken_id},
    )
    await db_session.commit()

    # The statement under test, read from the migration file itself so the test cannot
    # drift into asserting a copy that no longer matches what ships.
    src = Path("src/database/migrations/versions/003_backfill_failed_completed_at.py").read_text()
    stmt = src.split('op.execute(\n        """', 1)[1].split('"""', 1)[0]
    assert "UPDATE analysis_jobs" in stmt and "status = 'failed'" in stmt

    await db_session.execute(text(stmt))
    await db_session.commit()

    fixed = await repo.get_by_job_id(broken_id, tenant_id=TENANT_A)
    still_pending = await repo.get_by_job_id(pending_id, tenant_id=TENANT_A)
    assert fixed is not None and still_pending is not None
    assert fixed.completed_at is not None, "the stranded failed row must be repaired"
    # A non-terminal row must NOT be stamped: inventing a completion would hide a
    # genuinely stranded job rather than surface it.
    assert still_pending.status != "failed"
    assert still_pending.completed_at is None

    # Idempotent: a replay must not rewrite an already-correct timestamp.
    first = fixed.completed_at
    await db_session.execute(text(stmt))
    await db_session.commit()
    replayed = await repo.get_by_job_id(broken_id, tenant_id=TENANT_A)
    assert replayed is not None and replayed.completed_at == first


class TestReapStrandedJobs:
    """A job is owned by an in-process task; when the pod dies mid-run no handler runs,
    so the row keeps its non-terminal status forever and a poller waits on work nobody
    is doing. Measured live 2026-08-04: 23 rows stuck non-terminal, oldest ~11 DAYS."""

    @staticmethod
    async def _job(repo: JobRepository) -> str:
        job = await repo.create(
            tenant_id=TENANT_A,
            video_path="/tmp/a.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        return job.job_id

    @staticmethod
    async def _age(session: AsyncSession, job_id: str, *, seconds: int) -> None:
        from sqlalchemy import text

        await session.execute(
            text(
                "UPDATE analysis_jobs SET updated_at = now() - make_interval(secs => :s) "
                "WHERE job_id = :j"
            ),
            {"s": seconds, "j": job_id},
        )
        await session.commit()

    async def test_a_fresh_job_is_never_reaped(self, db_session: AsyncSession) -> None:
        # The whole risk of a reaper is killing live work. A just-written job must be
        # untouchable.
        repo = JobRepository(db_session)
        jid = await self._job(repo)
        job = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert job is not None
        assert await repo.reap_if_stranded(job, tenant_id=TENANT_A) is False
        refreshed = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert refreshed is not None and refreshed.status != "failed"

    async def test_a_long_but_still_progressing_job_is_never_reaped(
        self, db_session: AsyncSession
    ) -> None:
        # Staleness, not age, is the signal: a job running for hours that is still
        # writing progress must survive. This is what makes the reaper safe for a
        # genuinely large video.
        repo = JobRepository(db_session)
        jid = await self._job(repo)
        await self._age(db_session, jid, seconds=10 * 3600)  # started long ago...
        await repo.update_progress(jid, tenant_id=TENANT_A, stage="vision", progress=0.5)
        job = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert job is not None
        assert await repo.reap_if_stranded(job, tenant_id=TENANT_A) is False

    async def test_a_stale_job_is_reaped_with_an_attributable_reason(
        self, db_session: AsyncSession
    ) -> None:
        repo = JobRepository(db_session)
        jid = await self._job(repo)
        await repo.update_progress(jid, tenant_id=TENANT_A, stage="scene_detection", progress=0.3)
        await self._age(db_session, jid, seconds=2 * 3600)
        job = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert job is not None

        assert await repo.reap_if_stranded(job, tenant_id=TENANT_A) is True

        reaped = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert reaped is not None
        assert reaped.status == "failed"
        # Terminal ⇒ timestamped, same invariant as 003.
        assert reaped.completed_at is not None
        # The reason must name the stage and the cause, or the operator learns nothing.
        assert "stranded" in (reaped.error_message or "")
        assert "scene_detection" in (reaped.error_message or "")

    async def test_a_completed_job_is_never_demoted(self, db_session: AsyncSession) -> None:
        # The race that matters: the job finishes between the read and the reap. The
        # reaper must inherit mark_failed's terminal guard and leave it alone.
        repo = JobRepository(db_session)
        jid = await self._job(repo)
        await self._age(db_session, jid, seconds=5 * 3600)
        stale_view = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert stale_view is not None
        await repo.mark_completed(jid, tenant_id=TENANT_A)  # ...finishes underneath us

        assert await repo.reap_if_stranded(stale_view, tenant_id=TENANT_A) is False
        final = await repo.get_by_job_id(jid, tenant_id=TENANT_A)
        assert final is not None and final.status == "completed"

    async def test_migration_004_fails_the_long_stranded_backlog(
        self, db_session: AsyncSession
    ) -> None:
        """Executes migration 004's REAL statement against the exact live shapes."""
        from pathlib import Path

        from sqlalchemy import text

        repo = JobRepository(db_session)
        zombie, recent, done = (
            await self._job(repo),
            await self._job(repo),
            await self._job(repo),
        )
        await self._age(db_session, zombie, seconds=11 * 24 * 3600)  # the ~11-day row
        await self._age(db_session, recent, seconds=2 * 3600)  # stale but < 24h
        await repo.mark_completed(done, tenant_id=TENANT_A)
        await self._age(db_session, done, seconds=30 * 24 * 3600)  # old AND terminal

        src = Path("src/database/migrations/versions/004_fail_long_stranded_jobs.py").read_text()
        stmt = src.split('op.execute(\n        """', 1)[1].split('"""', 1)[0]
        assert "UPDATE analysis_jobs" in stmt and "24 hours" in stmt

        await db_session.execute(text(stmt))
        await db_session.commit()

        z = await repo.get_by_job_id(zombie, tenant_id=TENANT_A)
        r = await repo.get_by_job_id(recent, tenant_id=TENANT_A)
        d = await repo.get_by_job_id(done, tenant_id=TENANT_A)
        assert z is not None and r is not None and d is not None
        assert z.status == "failed" and z.completed_at is not None
        assert "stranded" in (z.error_message or "")
        # Under the cutoff: the runtime reaper owns this one, not the migration.
        assert r.status != "failed"
        # Already terminal: an old COMPLETED job must never be rewritten as failed.
        assert d.status == "completed"

        # Idempotent: replaying selects nothing new.
        before = z.completed_at
        await db_session.execute(text(stmt))
        await db_session.commit()
        again = await repo.get_by_job_id(zombie, tenant_id=TENANT_A)
        assert again is not None and again.completed_at == before
