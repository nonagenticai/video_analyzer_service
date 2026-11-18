"""Durable jobs: a job whose pod died is RE-CLAIMED and re-run, not thrown away.

Live 2026-10-22 (run 01a16423, job 01a1642c): a rollout replaced the single analyzer pod
3s after a job was submitted. Two defects followed, both pinned here:

1. The caller's poll answered **404 for its own job**. ``get_job_status`` reaped the
   stranded row (``mark_failed`` COMMITS), and the re-read ran in a fresh transaction
   with no tenant GUC bound (``set_config(..., is_local=true)``), so RLS hid the row.
   Only visible when connected as the RUNTIME non-owner role — the table owner bypasses
   the ENABLE'd policy, which is why the owner-connected suite never saw it.
2. The work was simply lost: a lapsed lease could only be FAILED ("Resubmit").
   Migration 007 makes a live pod re-claim it (atomically, bounded by an attempt
   counter) and re-run it under the job's own tenant + identity; a graceful SIGTERM
   gives the lease back so the re-claim is immediate and free.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import urlsplit, urlunsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import PipelineSettings
from src.repository.job_repository import STRANDED_AFTER_SECONDS, JobRepository
from src.services.artifact_generator import ArtifactGenerator
from src.services.pipeline import PipelineDependencies, PipelineRunner
from src.services.stranded_sweeper import StrandedJobSweeper
from tests.conftest import _TEST_DATABASE_URL, TENANT_A, TENANT_APP, TENANT_B, attach_test_tenant
from tests.unit.test_pipeline import (
    _fake_probe,
    _FakeAudioExtractor,
    _FakeDownloader,
    _FakeOCR,
    _FakeQualityAnalyzer,
    _FakeSceneDetector,
    _FakeStorage,
    _FakeTranscriptionClient,
    _test_settings,
)
from tests.unit.test_pipeline_resilience import (
    _HangingVisionProvider,
    _NFrameExtractor,
    _RecordingVisionProvider,
    _request,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _job(factory: async_sessionmaker[AsyncSession], tenant: str, **overrides: Any) -> str:
    async with factory() as session:
        job = await JobRepository(session).create(
            tenant_id=tenant,
            video_path=None,
            video_url="https://example.com/input.mp4",
            upload_filename=None,
            request_config=_request().model_dump(mode="json"),
            verification_criteria=None,
            dispatch_identity={"license_id": "lic-1", "tenant_source": "license-token"},
        )
        if overrides:
            assignments = ", ".join(f"{col} = :{col}" for col in overrides)
            await session.execute(
                text(f"UPDATE analysis_jobs SET {assignments} WHERE job_id = :job_id"),  # noqa: S608
                {**overrides, "job_id": job.job_id},
            )
            await session.commit()
        return job.job_id


async def _row(factory: async_sessionmaker[AsyncSession], job_id: str) -> dict[str, Any]:
    async with factory() as session:
        row = await session.execute(
            text(
                "SELECT status, stage, instance_id, reclaim_attempts, lease_released, "
                "lease_expires_at, error_message FROM analysis_jobs WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        return dict(row.mappings().one())


def _lapsed() -> datetime:
    return datetime.now(UTC) - timedelta(seconds=5)


def _sweeper(
    factory: async_sessionmaker[AsyncSession],
    *,
    reclaim: Any = None,
    instance_id: str = "pod-b",
    active: Any = None,
    settings: PipelineSettings | None = None,
) -> StrandedJobSweeper:
    return StrandedJobSweeper(
        settings or PipelineSettings(),
        AsyncMock(return_value=factory),
        active or (lambda: []),
        STRANDED_AFTER_SECONDS,
        reclaim=reclaim,
        instance_id=instance_id,
    )


def _pod(tmp_path: Path, instance_id: str, provider: Any) -> PipelineRunner:
    settings = _test_settings(tmp_path / instance_id)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / instance_id / "storage"),
        settings=settings,
        frame_extractor=_NFrameExtractor(3),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=provider,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    return PipelineRunner(dependencies=deps, instance_id=instance_id)


@pytest.fixture
def _pipeline_db(db_session_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)


async def _wait_for_stage(factory, job_id: str, stage: str, timeout: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if (await _row(factory, job_id))["stage"] == stage:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"job never reached stage {stage!r}: {await _row(factory, job_id)}")


# ---------------------------------------------------------------------------
# 1. The SQL: atomic, bounded, drain-aware
# ---------------------------------------------------------------------------


async def test_reclaim_takes_a_lapsed_job_and_hands_it_a_fresh_lease(db_session_factory) -> None:
    job_id = await _job(
        db_session_factory,
        TENANT_B,
        instance_id="dead-pod",
        lease_expires_at=_lapsed(),
        stage="vision",
        progress=0.5,
    )
    seen: list[tuple[str, str, Any]] = []

    async def _reclaim(jid: str, tenant: str, identity: Any) -> None:
        seen.append((jid, tenant, identity))

    assert await _sweeper(db_session_factory, reclaim=_reclaim).sweep_once() == 0

    assert seen == [
        (job_id, TENANT_B, {"license_id": "lic-1", "tenant_source": "license-token"})
    ], "the job's OWN tenant + identity must come back from the row, not the sweeper"
    row = await _row(db_session_factory, job_id)
    assert row["status"] == "pending"
    assert row["stage"] == "queued"
    assert row["instance_id"] == "pod-b"
    assert row["reclaim_attempts"] == 1
    assert row["lease_expires_at"] > datetime.now(UTC)


async def test_reclaim_never_touches_a_live_lease_or_this_pods_own_jobs(
    db_session_factory,
) -> None:
    live = await _job(
        db_session_factory,
        TENANT_A,
        instance_id="alive",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=300),
    )
    mine = await _job(db_session_factory, TENANT_A, instance_id="pod-b", lease_expires_at=_lapsed())
    called = AsyncMock()

    await _sweeper(db_session_factory, reclaim=called, active=lambda: [mine]).sweep_once()

    called.assert_not_awaited()
    assert (await _row(db_session_factory, live))["status"] == "pending"
    assert (await _row(db_session_factory, mine))["reclaim_attempts"] == 0


async def test_attempts_exhausted_fails_with_the_stranded_message(db_session_factory) -> None:
    job_id = await _job(
        db_session_factory,
        TENANT_A,
        instance_id="dead",
        lease_expires_at=_lapsed(),
        reclaim_attempts=2,
        stage="vision",
    )
    healthy = await _job(
        db_session_factory, TENANT_B, instance_id="dead", lease_expires_at=_lapsed()
    )
    called = AsyncMock()

    # The poison row is failed IN the function and the sweep keeps going past it.
    assert await _sweeper(db_session_factory, reclaim=called).sweep_once() == 1

    row = await _row(db_session_factory, job_id)
    assert row["status"] == "failed"
    assert "stranded" in row["error_message"]
    assert "re-run 2 time(s)" in row["error_message"]
    assert "vision" in row["error_message"]
    called.assert_awaited_once()
    assert called.await_args is not None
    assert called.await_args.args[0] == healthy


async def test_a_released_lease_is_reclaimed_without_spending_an_attempt(
    db_session_factory,
) -> None:
    """A drain on SIGTERM is not the job's fault — three rollouts must not fail it."""
    job_id = await _job(
        db_session_factory,
        TENANT_A,
        instance_id="drained",
        lease_expires_at=_lapsed(),
        reclaim_attempts=2,
        lease_released=True,
    )
    called = AsyncMock()

    assert await _sweeper(db_session_factory, reclaim=called).sweep_once() == 0

    called.assert_awaited_once()
    row = await _row(db_session_factory, job_id)
    assert row["status"] == "pending"
    assert row["reclaim_attempts"] == 2
    assert row["lease_released"] is False


async def test_concurrent_reclaim_hands_the_job_to_exactly_one_pod(db_session_factory) -> None:
    job_id = await _job(
        db_session_factory, TENANT_A, instance_id="dead", lease_expires_at=_lapsed()
    )

    # Deterministic half: pod A holds the row lock mid-claim; pod B must SKIP it.
    sql = text("SELECT r_job_id FROM reclaim_next_analysis_job(:w, 300, 2, CAST(:a AS text[]))")
    async with db_session_factory() as a, db_session_factory() as b:
        got_a = (await a.execute(sql, {"w": "pod-a", "a": []})).scalar_one_or_none()
        got_b = (await b.execute(sql, {"w": "pod-b", "a": []})).first()
        await a.commit()
        await b.commit()
    assert got_a == job_id
    assert got_b is None, "a second pod took a row another pod was mid-claiming"
    assert (await _row(db_session_factory, job_id))["instance_id"] == "pod-a"

    # Racing half: many sweepers at once, one fresh orphan, exactly one winner.
    other = await _job(db_session_factory, TENANT_B, instance_id="dead", lease_expires_at=_lapsed())
    winners: list[str] = []

    def _recorder(pod: str) -> Any:
        async def _r(jid: str, tenant: str, identity: Any) -> None:
            winners.append(pod)

        return _r

    sweepers = [
        _sweeper(db_session_factory, reclaim=_recorder(f"pod-{i}"), instance_id=f"pod-{i}")
        for i in range(6)
    ]
    await asyncio.gather(*(s.reclaim_once() for s in sweepers))
    assert len(winners) == 1, winners
    assert (await _row(db_session_factory, other))["instance_id"] == winners[0]
    assert (await _row(db_session_factory, other))["reclaim_attempts"] == 1


async def test_fail_sweep_leaves_reclaimable_rows_alone_unless_reclaim_is_off(
    db_session_factory,
) -> None:
    job_id = await _job(
        db_session_factory, TENANT_A, instance_id="dead", lease_expires_at=_lapsed()
    )
    sql = text("SELECT fail_stranded_analysis_jobs(3600, 50, CAST(:a AS text[]), :m)")
    async with db_session_factory() as s:
        assert (await s.execute(sql, {"a": [], "m": 2})).scalar_one() == 0
        await s.commit()
    assert (await _row(db_session_factory, job_id))["status"] == "pending"

    # A sweeper with no runner to hand the job to keeps the pre-007 fail-on-lapse rule.
    assert await _sweeper(db_session_factory, reclaim=None).sweep_once() == 1
    assert (await _row(db_session_factory, job_id))["status"] == "failed"


# ---------------------------------------------------------------------------
# 2. End to end on two "pods"
# ---------------------------------------------------------------------------


async def test_a_job_killed_mid_run_completes_on_a_second_worker(
    tmp_path: Path, db_session_factory, _pipeline_db: None
) -> None:
    pod_a = _pod(tmp_path, "pod-a", _HangingVisionProvider())
    job_id = await _job(db_session_factory, TENANT_A)

    await pod_a.enqueue(job_id, TENANT_A)
    await _wait_for_stage(db_session_factory, job_id, "scene_detection")
    # Pod A dies: its task stops, nothing is released, the lease just runs out.
    task = pod_a._tasks[job_id]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with db_session_factory() as s:
        await s.execute(
            text("UPDATE analysis_jobs SET lease_expires_at = :t WHERE job_id = :j"),
            {"t": _lapsed(), "j": job_id},
        )
        await s.commit()
    assert (await _row(db_session_factory, job_id))["status"] == "pending"

    pod_b = _pod(tmp_path, "pod-b", _RecordingVisionProvider())
    sweeper = _sweeper(
        db_session_factory, reclaim=pod_b.resume, instance_id="pod-b", active=pod_b.active_job_ids
    )
    assert await sweeper.sweep_once() == 0
    await asyncio.wait_for(pod_b.wait_for(job_id), timeout=30)

    row = await _row(db_session_factory, job_id)
    assert row["status"] == "completed", row
    assert row["instance_id"] == "pod-b"
    assert row["reclaim_attempts"] == 1


async def test_graceful_drain_releases_the_lease_for_an_immediate_free_reclaim(
    tmp_path: Path, db_session_factory, _pipeline_db: None
) -> None:
    pod_a = _pod(tmp_path, "pod-a", _HangingVisionProvider())
    job_id = await _job(db_session_factory, TENANT_A)
    await pod_a.enqueue(job_id, TENANT_A)
    await _wait_for_stage(db_session_factory, job_id, "scene_detection")

    assert await pod_a.drain() == 1
    assert pod_a.active_job_ids() == []
    row = await _row(db_session_factory, job_id)
    assert row["lease_released"] is True
    assert row["lease_expires_at"] is not None

    pod_b = _pod(tmp_path, "pod-b", _RecordingVisionProvider())
    await _sweeper(
        db_session_factory, reclaim=pod_b.resume, active=pod_b.active_job_ids
    ).sweep_once()
    await asyncio.wait_for(pod_b.wait_for(job_id), timeout=30)
    row = await _row(db_session_factory, job_id)
    assert row["status"] == "completed"
    assert row["reclaim_attempts"] == 0, "a drain must not spend one of the job's attempts"


async def test_a_copy_whose_job_was_reclaimed_elsewhere_stops_itself(
    tmp_path: Path, db_session_factory, _pipeline_db: None
) -> None:
    """Two pods must never both run (and pay for) the same job."""
    pod_a = _pod(tmp_path, "pod-a", _HangingVisionProvider())
    pod_a._lease_seconds = 3  # heartbeat every 1s
    job_id = await _job(db_session_factory, TENANT_A)
    await pod_a.enqueue(job_id, TENANT_A)
    await _wait_for_stage(db_session_factory, job_id, "scene_detection")

    async with db_session_factory() as s:
        await s.execute(
            text("UPDATE analysis_jobs SET instance_id = 'pod-b' WHERE job_id = :j"),
            {"j": job_id},
        )
        await s.commit()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pod_a._tasks[job_id], timeout=10)
    # The new owner's row is untouched by the stopped copy.
    row = await _row(db_session_factory, job_id)
    assert row["instance_id"] == "pod-b"
    assert row["status"] == "pending"


async def test_reclaimed_task_runs_under_the_submitters_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        get_license_id_from_context,
        get_tenant_id_from_context,
        get_tenant_source_from_context,
    )

    pod = _pod(tmp_path, "pod-b", _RecordingVisionProvider())
    seen: dict[str, Any] = {}

    async def _fake_run_job(job_id: str, tenant_id: str) -> None:
        seen["tenant"] = str(get_tenant_id_from_context())
        seen["license"] = get_license_id_from_context()
        src = get_tenant_source_from_context()
        seen["source"] = src.value if src is not None else None

    monkeypatch.setattr(pod, "_run_job", _fake_run_job)
    clear_identity_context()
    await pod.resume("job-x", TENANT_B, {"license_id": "lic-9", "tenant_source": "license-token"})
    await pod.wait_for("job-x")

    assert seen == {"tenant": TENANT_B, "license": "lic-9", "source": "license-token"}
    assert get_tenant_id_from_context() is None, "identity bled into the sweeper's context"


# ---------------------------------------------------------------------------
# 3. RLS: the 404, reproduced as the RUNTIME (non-owner) role
# ---------------------------------------------------------------------------


@pytest.fixture
async def runtime_factory(db_engine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Sessions connected as a throwaway NOSUPERUSER NOBYPASSRLS LOGIN role.

    The table owner bypasses ENABLE'd (not FORCE'd) policies, so an owner-connected test
    cannot see RLS hide anything. Needs CREATEROLE; skips loudly where the DB user lacks it.
    """
    role = f"va_rt_{uuid.uuid4().hex[:10]}"
    password = uuid.uuid4().hex
    async with db_engine.begin() as conn:
        try:
            await conn.execute(
                text(
                    f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}' "
                    "NOSUPERUSER NOBYPASSRLS NOINHERIT"
                )
            )
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
        db = (await conn.execute(text("SELECT current_database()"))).scalar_one()
        await conn.execute(text(f'GRANT CONNECT ON DATABASE "{db}" TO "{role}"'))
        await conn.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await conn.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_jobs TO "{role}"')
        )
        await conn.execute(text(f'GRANT USAGE ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{role}"'))
    parts = urlsplit(_TEST_DATABASE_URL)
    host = parts.netloc.rsplit("@", 1)[-1]
    url = urlunsplit(
        (parts.scheme, f"{role}:{password}@{host}", parts.path, parts.query, parts.fragment)
    )
    engine = create_async_engine(
        url, connect_args={"options": f"-csearch_path={schema}"}, echo=False
    )
    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with db_engine.begin() as conn:
            await conn.execute(text(f'DROP OWNED BY "{role}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{role}"'))


def _status_app(
    monkeypatch: pytest.MonkeyPatch, factory: async_sessionmaker[AsyncSession]
) -> FastAPI:
    from src.api.routes import analysis as analysis_module

    async def _sm() -> async_sessionmaker[AsyncSession]:
        return factory

    monkeypatch.setattr(analysis_module, "get_session_maker", _sm)
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    app = FastAPI()
    app.include_router(analysis_module.router)
    return attach_test_tenant(app, TENANT_APP)


async def test_poll_that_reaps_a_stranded_job_returns_it_not_404(
    db_session_factory, runtime_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = await _job(
        db_session_factory,
        TENANT_APP,
        instance_id="dead",
        lease_expires_at=_lapsed(),
        reclaim_attempts=2,
    )
    client = TestClient(_status_app(monkeypatch, runtime_factory))

    resp = client.get(f"/api/analyze/{job_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "failed"
    assert "stranded" in (resp.json()["error_message"] or "")


async def test_poll_leaves_a_reclaimable_job_pending(
    db_session_factory, runtime_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = await _job(
        db_session_factory, TENANT_APP, instance_id="dead", lease_expires_at=_lapsed()
    )
    client = TestClient(_status_app(monkeypatch, runtime_factory))

    resp = client.get(f"/api/analyze/{job_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "pending"
    assert (await _row(db_session_factory, job_id))["status"] == "pending"


async def test_rebind_listener_heals_a_read_after_commit(
    db_session_factory, runtime_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asgi-installed listener alone keeps a post-commit read visible."""
    from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
    from w2v_common.database.rls import bind_session_tenant, install_rls_tenant_rebind

    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    job_id = await _job(db_session_factory, TENANT_A)
    install_rls_tenant_rebind()
    set_identity_context(TENANT_A, None, None)
    try:
        async with runtime_factory() as session:
            await bind_session_tenant(session, uuid.UUID(TENANT_A))
            repo = JobRepository(session)
            assert await repo.get_by_job_id(job_id, tenant_id=TENANT_A) is not None
            await session.commit()
            assert await repo.get_by_job_id(job_id, tenant_id=TENANT_A) is not None
    finally:
        clear_identity_context()


async def test_a_released_lease_is_still_failed_when_reclaim_is_off(db_session_factory) -> None:
    """With re-claim off nothing picks a released lease up — the sweeper must fail it."""
    job_id = await _job(
        db_session_factory,
        TENANT_A,
        instance_id="drained",
        lease_expires_at=_lapsed(),
        lease_released=True,
    )

    assert await _sweeper(db_session_factory, reclaim=None).sweep_once() == 1
    assert (await _row(db_session_factory, job_id))["status"] == "failed"


async def test_a_job_whose_claim_failed_is_adopted_not_cancelled(
    tmp_path: Path, db_session_factory, _pipeline_db: None
) -> None:
    """An unowned row (best-effort ``_claim`` hit a DB blip) is not "stolen".

    The heartbeat must take the lease and keep the job running — cancelling it would
    leave an unleased row nobody re-claims, failed only an hour later.
    """
    pod_a = _pod(tmp_path, "pod-a", _HangingVisionProvider())
    pod_a._lease_seconds = 3  # heartbeat every 1s
    job_id = await _job(db_session_factory, TENANT_A)
    await pod_a.enqueue(job_id, TENANT_A)
    await _wait_for_stage(db_session_factory, job_id, "scene_detection")

    async with db_session_factory() as s:
        await s.execute(
            text(
                "UPDATE analysis_jobs SET instance_id = NULL, lease_expires_at = NULL "
                "WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        await s.commit()

    await asyncio.sleep(2.5)
    task = pod_a._tasks[job_id]
    assert not task.done(), "a healthy unowned job was cancelled by its own heartbeat"
    row = await _row(db_session_factory, job_id)
    assert row["instance_id"] == "pod-a"
    assert row["lease_expires_at"] is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_drain_releases_nothing_when_reclaim_is_off(
    tmp_path: Path, db_session_factory, _pipeline_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import clear_settings_cache

    monkeypatch.setenv("PIPELINE_JOB_MAX_RECLAIMS", "0")
    clear_settings_cache()
    pod_a = _pod(tmp_path, "pod-a", _HangingVisionProvider())
    job_id = await _job(db_session_factory, TENANT_A)
    await pod_a.enqueue(job_id, TENANT_A)
    await _wait_for_stage(db_session_factory, job_id, "scene_detection")

    assert await pod_a.drain() == 0
    assert (await _row(db_session_factory, job_id))["lease_released"] is False
