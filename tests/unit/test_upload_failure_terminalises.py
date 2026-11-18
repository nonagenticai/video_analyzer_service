"""Failure paths of ``POST /api/analyze/upload`` must terminalise the job row.

The job row is INSERTed before a single byte is streamed, and this service has
no reaper (migration ``003_backfill_failed_completed_at`` says so explicitly),
so any abort after the INSERT that leaves ``status='pending'`` strands the row
forever — and leaves the partial upload on the pod's disk.

Every stub here implements the REAL method names of the collaborators
(``store_upload`` / ``enqueue``); no ``AsyncMock``, whose auto-attributes would
happily let a renamed production call pass.

The cancellation test drives a REAL ``asyncio.Task`` cancellation, not a stub
that raises a ``CancelledError`` object. That distinction is the whole test: a
raised-object stub leaves no cancellation pending, so every subsequent ``await``
in the ``except`` block succeeds and the test passes against code that is broken
in production, where the request's scope is genuinely cancelled and each
``await`` re-raises immediately.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import ClientDisconnect, Request

from src.api.routes import analysis as analysis_module
from src.api.routes.analysis import router as analysis_router
from src.api.routes.analysis import upload_and_analyze
from src.config import clear_settings_cache, get_settings
from src.database.models import AnalysisJob
from src.repository.job_repository import JobRepository
from tests.conftest import TENANT_A, TENANT_APP, attach_test_tenant

# The tenant the test app attributes requests to (see attach_test_tenant).
TENANT = TENANT_APP


class _StubRunner:
    def __init__(self, fail_with: BaseException | None = None) -> None:
        self.calls: list[str] = []
        self.fail_with = fail_with

    async def enqueue(self, job_id: str, tenant_id: str) -> None:
        self.calls.append(job_id)
        if self.fail_with is not None:
            raise self.fail_with


class _Storage:
    """Storage stub whose ``store_upload`` raises ``fail_with`` when set."""

    def __init__(self, fail_with: BaseException | None = None) -> None:
        self.fail_with = fail_with
        self.uploads: list[tuple[str, str, str]] = []

    async def store_upload(
        self, tenant_id: str, job_id: str, file_path: Path, filename: str
    ) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.uploads.append((tenant_id, job_id, filename))
        return f"uploads/{tenant_id}/{job_id}/{filename}"


class _BlockingStorage:
    """Storage stub that parks inside ``store_upload`` until released.

    Used to hold the endpoint mid-upload so the test can cancel its task for
    real — the closest in-process analogue of a client vanishing while uvicorn
    is still feeding the request body.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def store_upload(
        self, tenant_id: str, job_id: str, file_path: Path, filename: str
    ) -> str:
        self.entered.set()
        await self.release.wait()  # never set: the test cancels instead
        return f"uploads/{tenant_id}/{job_id}/{filename}"


@pytest.fixture
def uploads_root(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def make_app(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    uploads_root: Path,
):
    """Return a builder for a minimal app wired to the analysis router."""

    def _make(storage: object, runner: _StubRunner | None = None) -> FastAPI:
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "jobs"))
        monkeypatch.setenv("STORAGE_UPLOAD_DIR", str(uploads_root))
        clear_settings_cache()

        async def _fake_session_maker() -> async_sessionmaker[AsyncSession]:
            return db_session_factory

        monkeypatch.setattr(analysis_module, "get_session_maker", _fake_session_maker)

        app = FastAPI()
        app.state.settings = get_settings()
        app.state.pipeline_runner = runner if runner is not None else _StubRunner()
        app.state.storage_service = storage
        app.include_router(analysis_router)
        return attach_test_tenant(app)

    return _make


async def _drain_shielded_aborts() -> None:
    """Wait for every shielded terminalisation task to finish.

    Under a real cancellation the DB write deliberately OUTLIVES the request
    task, so a test that asserts DB state immediately after the cancellation
    would be racing the fix it is verifying.
    """
    for _ in range(50):
        pending = set(analysis_module._PENDING_ABORTS)
        if not pending:
            return
        await asyncio.wait(pending, timeout=5)
    raise AssertionError("shielded abort tasks never completed")


async def _only_job(factory: async_sessionmaker[AsyncSession]) -> AnalysisJob:
    """The single job row written by the request under test."""
    async with factory() as session:
        rows = (await session.execute(select(AnalysisJob))).scalars().all()
    assert len(rows) == 1
    return rows[0]


async def _job_row(factory: async_sessionmaker[AsyncSession], job_id: str) -> AnalysisJob:
    """Re-read a row through a FRESH session (no identity-map staleness)."""
    async with factory() as session:
        return (
            await session.execute(select(AnalysisJob).where(AnalysisJob.job_id == job_id))
        ).scalar_one()


def _post_upload(app: FastAPI, payload: bytes = b"FAKE-MP4-DATA"):
    client = TestClient(app)
    return client.post(
        "/api/analyze/upload",
        files={"file": ("demo.mp4", io.BytesIO(payload), "video/mp4")},
    )


def _bare_request(app: FastAPI) -> Request:
    """A Request good enough for a direct call to the endpoint coroutine."""
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/api/analyze/upload",
            "query_string": b"",
            "headers": [],
            "app": app,
            # AuthMiddleware puts the tenant here in production; the routes now
            # 401 without it rather than substituting the nil UUID.
            "state": {"tenant_id": TENANT},
        }
    )


async def test_store_upload_oserror_terminalises_job_and_removes_dir(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    app = make_app(_Storage(OSError("object store unreachable")))
    resp = _post_upload(app)
    assert resp.status_code == 500

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.completed_at is not None
    assert job.error_message
    assert not (uploads_root / job.job_id).exists()


async def test_write_path_oserror_terminalises_job_and_removes_dir(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = make_app(_Storage())
    real_open = Path.open

    def _failing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if str(self).startswith(str(uploads_root)):
            raise OSError("No space left on device")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _failing_open)

    resp = _post_upload(app)
    assert resp.status_code == 500

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.completed_at is not None
    assert not (uploads_root / job.job_id).exists()


async def test_real_task_cancellation_terminalises_and_propagates(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    """A GENUINE cancellation, not a stub raising a ``CancelledError`` object.

    The endpoint runs as its own task and is cancelled while parked inside
    ``store_upload``. Without ``asyncio.shield`` around the DB leg the row stays
    ``pending`` forever: every ``await`` in the ``except`` block re-raises and
    ``except Exception`` cannot even log a ``BaseException``.
    """
    storage = _BlockingStorage()
    app = make_app(storage)
    upload = UploadFile(file=io.BytesIO(b"FAKE-MP4-DATA"), filename="demo.mp4")

    task = asyncio.create_task(upload_and_analyze(_bare_request(app), upload, None))
    await asyncio.wait_for(storage.entered.wait(), timeout=10)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await _drain_shielded_aborts()

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.error_message == "upload aborted: client disconnected"
    assert job.completed_at is not None
    assert not (uploads_root / job.job_id).exists()


async def test_client_disconnect_exception_terminalises_job(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    """Uvicorn surfaces a half-sent body as ``ClientDisconnect`` — a plain
    ``Exception``, not a ``CancelledError``. It must still terminalise, and with
    a truthful reason rather than the generic ``upload failed: …`` branch."""
    app = make_app(_Storage(ClientDisconnect()))

    with pytest.raises(ClientDisconnect):
        _post_upload(app)

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.error_message == "upload aborted: client disconnected"
    assert job.completed_at is not None
    assert not (uploads_root / job.job_id).exists()


async def test_enqueue_failure_terminalises_job_and_removes_dir(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    """``_kickoff_pipeline`` used to run OUTSIDE the guard: a Redis outage left
    the job ``pending`` forever with ``upload_dir`` on disk."""
    storage = _Storage()
    runner = _StubRunner(fail_with=RuntimeError("redis connection refused"))
    app = make_app(storage, runner)

    with pytest.raises(RuntimeError, match="redis connection refused"):
        _post_upload(app)

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.error_message == "upload failed: RuntimeError: redis connection refused"
    assert job.completed_at is not None
    # The object-store copy is deliberately NOT deleted — only the pod-local
    # scratch dir, which _resolve_source recreates by re-downloading.
    assert storage.uploads == [(TENANT, job.job_id, "demo.mp4")]
    assert not (uploads_root / job.job_id).exists()


async def test_missing_storage_service_terminalises_job(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    """``request.app.state.storage_service`` used to be read before the ``try``."""
    app = make_app(_Storage())
    del app.state.storage_service

    with pytest.raises(AttributeError):
        _post_upload(app)

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.completed_at is not None
    assert not (uploads_root / job.job_id).exists()


async def test_too_large_upload_terminalises_and_removes_dir(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    app = make_app(_Storage())
    app.state.settings.storage.max_upload_size_bytes = 4

    resp = _post_upload(app, payload=b"hello world")
    assert resp.status_code == 413

    job = await _only_job(db_session_factory)
    assert job.status == "failed"
    assert job.error_message == "upload exceeded MAX_UPLOAD_SIZE_BYTES"
    assert job.completed_at is not None
    assert not (uploads_root / job.job_id).exists()


async def test_mark_failed_does_not_demote_a_completed_job(db_session: AsyncSession) -> None:
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

    moved = await repo.mark_failed(job.job_id, "late cleanup lost the race", tenant_id=TENANT_A)
    assert moved is False

    refreshed = await repo.get_by_job_id(job.job_id, tenant_id=TENANT_A)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.progress == 1.0
    assert refreshed.error_message is None


async def test_completion_landing_after_a_stale_read_is_not_demoted(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Deterministic reproduction of the old read-then-write TOCTOU window.

    The loser reads ``pending`` (exactly what the old ``get_by_job_id`` guard
    cached), the winner completes the job, and only THEN does the loser attempt
    its UPDATE. The old two-round-trip guard demoted the finished row; the
    single conditional statement cannot.
    """
    async with db_session_factory() as winner_session, db_session_factory() as loser_session:
        winner = JobRepository(winner_session)
        loser = JobRepository(loser_session)
        job = await winner.create(
            tenant_id=TENANT_A,
            video_path="/tmp/a.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )

        stale = await loser.get_by_job_id(job.job_id, tenant_id=TENANT_A)
        assert stale is not None
        assert stale.status == "pending"

        await winner.mark_completed(job.job_id, tenant_id=TENANT_A)

        moved = await loser.mark_failed(job.job_id, "generic cleanup", tenant_id=TENANT_A)
        assert moved is False

    row = await _job_row(db_session_factory, job.job_id)
    assert row.status == "completed"
    assert row.error_message is None


async def test_concurrent_mark_failed_only_one_wins_and_keeps_the_real_message(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A TRUE race, on separate sessions/connections against the real DB.

    Postgres serialises the two UPDATEs on the row lock and re-evaluates the
    ``status NOT IN (...)`` predicate for the second, so exactly one call moves
    the row and the loser cannot clobber the winner's error message.
    """
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_A,
            video_path="/tmp/a.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )

    async def _fail(message: str) -> bool:
        async with db_session_factory() as session:
            return await JobRepository(session).mark_failed(job.job_id, message, tenant_id=TENANT_A)

    first, second = await asyncio.gather(_fail("real ffmpeg error"), _fail("generic cleanup"))
    assert [first, second].count(True) == 1

    row = await _job_row(db_session_factory, job.job_id)
    assert row.status == "failed"
    assert row.completed_at is not None
    assert row.error_message == ("real ffmpeg error" if first else "generic cleanup")


async def test_happy_path_still_202_and_job_left_pending(
    make_app: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    uploads_root: Path,
) -> None:
    storage = _Storage()
    app = make_app(storage)
    resp = _post_upload(app)
    assert resp.status_code == 202

    job = await _only_job(db_session_factory)
    assert job.status == "pending"
    assert job.stage == "queued"
    assert job.completed_at is None
    assert storage.uploads == [(TENANT, job.job_id, "demo.mp4")]
    # The temp copy is gone once the object store owns the bytes.
    assert not (uploads_root / job.job_id / "demo.mp4").exists()
    assert app.state.pipeline_runner.calls == [job.job_id]
