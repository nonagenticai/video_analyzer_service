"""Analysis API route tests against the postgres CI database fixture + fake pipeline runner."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.routes.analysis import router as analysis_router
from src.config import get_settings
from src.repository.job_repository import JobRepository
from tests.conftest import TENANT_APP, TENANT_B, attach_test_tenant


class _StubPipelineRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def enqueue(self, job_id: str, tenant_id: str) -> None:
        self.calls.append(job_id)


class _StubStorage:
    def __init__(self) -> None:
        self._files: dict[str, Path] = {}
        # Records (tenant_id, job_id, filename, streamed_bytes) for each upload
        # so tests can assert the source went to the object store, not disk.
        self.uploads: list[tuple[str, str, str, bytes]] = []
        self.deleted_uploads: list[tuple[str, str]] = []

    def get_local_path(self, path: str) -> Path | None:
        return self._files.get(path)

    async def retrieve(self, path: str) -> bytes | None:
        local = self._files.get(path)
        if local and local.exists():
            return local.read_bytes()
        return None

    async def delete_job_dir(self, job_id: str) -> None:
        return None

    async def store_upload(
        self, tenant_id: str, job_id: str, file_path: Path, filename: str
    ) -> str:
        self.uploads.append((tenant_id, job_id, filename, Path(file_path).read_bytes()))
        return f"uploads/{tenant_id}/{job_id}/{filename}"

    async def delete_upload(self, tenant_id: str, job_id: str) -> None:
        self.deleted_uploads.append((tenant_id, job_id))

    def register(self, path: str, file_path: Path) -> None:
        self._files[path] = file_path


@pytest.fixture
def stub_storage() -> _StubStorage:
    return _StubStorage()


@pytest.fixture
def stub_runner() -> _StubPipelineRunner:
    return _StubPipelineRunner()


@pytest.fixture
async def routes_app(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: async_sessionmaker[AsyncSession],
    stub_runner: _StubPipelineRunner,
    stub_storage: _StubStorage,
    tmp_path: Path,
) -> FastAPI:
    """Build a minimal FastAPI app with the analysis router wired up."""
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "jobs"))
    monkeypatch.setenv("STORAGE_UPLOAD_DIR", str(tmp_path / "uploads"))
    from src.config import clear_settings_cache

    clear_settings_cache()

    from src.api.routes import analysis as analysis_module

    async def _fake_session_maker() -> async_sessionmaker[AsyncSession]:
        return db_session_factory

    monkeypatch.setattr(analysis_module, "get_session_maker", _fake_session_maker)

    app = FastAPI()
    app.state.settings = get_settings()
    app.state.pipeline_runner = stub_runner
    app.state.storage_service = stub_storage
    app.include_router(analysis_router)
    return attach_test_tenant(app)


@pytest.fixture
def routes_app_without_tenant(routes_app: FastAPI) -> FastAPI:
    """The same router with NO tenant middleware attached."""
    app = FastAPI()
    app.state.settings = routes_app.state.settings
    app.state.pipeline_runner = routes_app.state.pipeline_runner
    app.state.storage_service = routes_app.state.storage_service
    app.include_router(analysis_router)
    return app


def test_create_analysis_job_requires_source(routes_app: FastAPI) -> None:
    """422, not the old hand-rolled 400: the invariant moved onto the model, so
    it is now enforced during body validation rather than inside the handler."""
    client = TestClient(routes_app)
    resp = client.post("/api/analyze", json={})
    assert resp.status_code == 422
    assert "video_url" in resp.text


def test_create_analysis_job_rejects_reference_image_path(routes_app: FastAPI) -> None:
    """The unconfined read is reachable from the BODY, via verification criteria."""
    client = TestClient(routes_app)
    resp = client.post(
        "/api/analyze",
        json={
            "video_url": "https://example.com/v.mp4",
            "verification": {
                "required_screenshots": [{"name": "x", "path": "/etc/passwd"}],
            },
        },
    )
    assert resp.status_code == 422
    assert "no longer accepted" in resp.text


async def test_create_analysis_job_inserts_row(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
    stub_runner: _StubPipelineRunner,
    tmp_path: Path,
) -> None:
    client = TestClient(routes_app)
    resp = client.post("/api/analyze", json={"video_url": "https://example.com/v.mp4"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["stage"] == "queued"
    assert stub_runner.calls == [body["job_id"]]

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(body["job_id"], tenant_id=TENANT_APP)
        assert job is not None
        assert job.video_url == "https://example.com/v.mp4"
        # No caller-supplied pod-local path is ever recorded.
        assert job.video_path is None


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "/proc/self/environ",
        "../../etc/shadow",
        "/app/data/uploads/00000000-0000-0000-0000-000000000000evil/x.mp4",
        "file:///etc/passwd",
    ],
)
def test_create_analysis_job_rejects_video_path(
    routes_app: FastAPI, stub_runner: _StubPipelineRunner, hostile: str
) -> None:
    """``video_path`` is removed: any request carrying it is refused outright.

    Refused — not ignored — so a caller cannot get a 202 for a job that silently
    analyses nothing, and so the arbitrary-pod-file read has no entry point.
    """
    client = TestClient(routes_app)
    resp = client.post("/api/analyze", json={"video_path": hostile})
    assert resp.status_code == 422
    assert "video_path" in resp.text
    assert stub_runner.calls == []


def test_create_analysis_job_rejects_video_path_even_with_video_url(
    routes_app: FastAPI, stub_runner: _StubPipelineRunner
) -> None:
    """A valid ``video_url`` does not launder a ``video_path`` smuggled alongside."""
    client = TestClient(routes_app)
    resp = client.post(
        "/api/analyze",
        json={"video_url": "https://example.com/v.mp4", "video_path": "/etc/passwd"},
    )
    assert resp.status_code == 422
    assert stub_runner.calls == []


async def test_upload_endpoint_stores_source_in_object_store(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
    stub_runner: _StubPipelineRunner,
    stub_storage: _StubStorage,
    tmp_path: Path,
) -> None:
    client = TestClient(routes_app)
    contents = b"FAKE-MP4-DATA"
    files = {"file": ("demo.mp4", io.BytesIO(contents), "video/mp4")}
    data = {
        "config": json.dumps(
            {
                "generate_walkthrough": True,
                "extract_frame_descriptions": False,
            }
        )
    }
    resp = client.post("/api/analyze/upload", files=files, data=data)
    assert resp.status_code == 202
    body = resp.json()
    assert stub_runner.calls == [body["job_id"]]

    # The source is streamed into the shared object store under a tenant-scoped
    # key — NOT persisted as a pod-local video_path. This is the multi-replica
    # fix: an upload accepted on pod A is resolvable when analyze runs on pod B.
    assert len(stub_storage.uploads) == 1
    tenant, job_id, filename, streamed = stub_storage.uploads[0]
    assert job_id == body["job_id"]
    assert tenant == TENANT_APP
    assert filename == "demo.mp4"
    assert streamed == contents

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(body["job_id"], tenant_id=TENANT_APP)
        assert job is not None
        # No pod-local path is recorded; the pipeline resolves via upload_filename.
        assert job.video_path is None
        assert job.upload_filename == "demo.mp4"
    # The temp copy on the accepting pod is cleaned up once it is in the store.
    assert not (tmp_path / "uploads" / body["job_id"] / "demo.mp4").exists()


def test_upload_rejects_invalid_config(routes_app: FastAPI) -> None:
    client = TestClient(routes_app)
    files = {"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")}
    data = {"config": "not-json{"}
    resp = client.post("/api/analyze/upload", files=files, data=data)
    assert resp.status_code == 400


def test_upload_rejects_video_path_in_config(routes_app: FastAPI) -> None:
    client = TestClient(routes_app)
    files = {"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")}
    data = {"config": json.dumps({"video_path": "/tmp/x"})}
    resp = client.post("/api/analyze/upload", files=files, data=data)
    assert resp.status_code == 400


def test_upload_enforces_max_size(routes_app: FastAPI) -> None:
    routes_app.state.settings.storage.max_upload_size_bytes = 4
    client = TestClient(routes_app)
    files = {"file": ("a.mp4", io.BytesIO(b"hello world"), "video/mp4")}
    resp = client.post("/api/analyze/upload", files=files)
    assert resp.status_code == 413


async def test_get_job_status_404(routes_app: FastAPI) -> None:
    client = TestClient(routes_app)
    resp = client.get("/api/analyze/missing")
    assert resp.status_code == 404


async def test_get_job_result_returns_full_payload(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_APP,
            video_path="/tmp/v.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        await JobRepository(session).update_fields(
            job.job_id,
            tenant_id=TENANT_APP,
            transcript_data={"text": "hello"},
            scene_boundaries=[0.0, 3.0],
            artifacts={"walkthrough.md": "/tmp/walk.md"},
            # Same shape the pipeline stores: asdict(QualityMetrics).
            quality_metrics={
                "mean_sharpness": 110.0,
                "mean_brightness": 0.5,
                "motion_score": 0.3,
                "dominant_colors": ["#101010"],
                "overall_score": 0.75,
                "motion_timeline": [{"t": 3.0, "motion": 0.3}],
                "visual_timeline": [
                    {
                        "t": 3.0,
                        "index": 1,
                        "sharpness": 110.0,
                        "brightness": 0.5,
                        "dominant_rgb": [16, 16, 16],
                    }
                ],
            },
            verification_result={
                "passed": True,
                "criteria_total": 1,
                "criteria_passed": 1,
                "failures": [],
            },
        )
    client = TestClient(routes_app)
    resp = client.get(f"/api/analyze/{job.job_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript_data"]["text"] == "hello"
    assert body["scene_boundaries"] == [0.0, 3.0]
    assert body["verification"]["passed"] is True
    assert body["verification_passed"] is True
    # quality_metrics survives the raw-JSONB → typed-schema validation.
    assert body["quality_metrics"]["overall_score"] == 0.75
    assert body["quality_metrics"]["motion_timeline"] == [{"t": 3.0, "motion": 0.3}]
    # ...including the per-frame visual curve, which is what makes shot-to-shot
    # continuity/colour drift measurable off the wire.
    assert body["quality_metrics"]["visual_timeline"] == [
        {"t": 3.0, "index": 1, "sharpness": 110.0, "brightness": 0.5, "dominant_rgb": [16, 16, 16]}
    ]


async def test_verification_endpoint_404_without_data(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_APP,
            video_path="/tmp/v.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
    client = TestClient(routes_app)
    resp = client.get(f"/api/analyze/{job.job_id}/verification")
    assert resp.status_code == 404


async def test_artifact_download_streams_local_file(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
    stub_storage: _StubStorage,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "walk.md"
    artifact_path.write_text("# walkthrough")

    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_APP,
            video_path=None,
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        await JobRepository(session).update_fields(
            job.job_id,
            tenant_id=TENANT_APP,
            artifacts={"walkthrough.md": "/storage/walk.md"},
        )
    stub_storage.register("/storage/walk.md", artifact_path)

    client = TestClient(routes_app)
    resp = client.get(f"/api/analyze/{job.job_id}/artifact/walkthrough.md")
    assert resp.status_code == 200
    assert "walkthrough" in resp.text


async def test_artifact_download_404_missing_artifact(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_APP,
            video_path=None,
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
    client = TestClient(routes_app)
    resp = client.get(f"/api/analyze/{job.job_id}/artifact/missing.md")
    assert resp.status_code == 404


async def test_delete_job_removes_row(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_APP,
            video_path=None,
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
    client = TestClient(routes_app)
    resp = client.delete(f"/api/analyze/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    async with db_session_factory() as session:
        assert await JobRepository(session).get_by_job_id(job.job_id, tenant_id=TENANT_APP) is None


def test_delete_job_404(routes_app: FastAPI) -> None:
    client = TestClient(routes_app)
    resp = client.delete("/api/analyze/missing")
    assert resp.status_code == 404


def test_create_with_verification_persists_criteria(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    client = TestClient(routes_app)
    payload: dict[str, Any] = {
        "video_url": "https://example.com/v.mp4",
        "verification": {
            "required_phrases": ["welcome"],
            "min_duration_seconds": 1.0,
        },
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    async def _check() -> None:
        async with db_session_factory() as session:
            job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_APP)
            assert job is not None
            assert job.verification_criteria is not None
            assert "welcome" in job.verification_criteria["required_phrases"]

    import asyncio

    asyncio.get_event_loop().run_until_complete(_check())


async def test_cross_tenant_get_returns_404(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression — BOLA: a tenant must not be able to read another tenant's job.

    Originally fixed at the route layer via an ``_enforce_job_tenant`` helper
    called after every ``get_by_job_id``. That defense was correct but lived
    in the wrong layer: any new caller of the repo had to remember to wrap
    the result. The fix is now at the SQL ``WHERE`` clause —
    ``JobRepository.get_by_job_id(job_id, *, tenant_id=...)`` returns None
    for a wrong-tenant lookup, and the route maps None → 404. The test still
    proves the route returns 404 across every read/delete shape.
    """
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_B,  # NOT the request's tenant
            video_path="/tmp/v.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )

    client = TestClient(routes_app)
    # Caller defaults to the unattributed nil-tenant sentinel (no AuthMiddleware
    # in test app); the job belongs to TENANT_B so every read/delete
    # must 404.
    for path in (
        f"/api/analyze/{job.job_id}",
        f"/api/analyze/{job.job_id}/result",
        f"/api/analyze/{job.job_id}/verification",
        f"/api/analyze/{job.job_id}/artifact/anything.md",
    ):
        resp = client.get(path)
        assert resp.status_code == 404, f"BOLA: {path} returned {resp.status_code}, expected 404"
    resp = client.delete(f"/api/analyze/{job.job_id}")
    assert resp.status_code == 404, "BOLA: DELETE returned {resp.status_code}, expected 404"

    # And the job is still there from the OWNER's view (cross-tenant DELETE
    # did not actually remove it).
    async with db_session_factory() as session:
        assert (
            await JobRepository(session).get_by_job_id(job.job_id, tenant_id=TENANT_B) is not None
        )


async def test_status_poll_reaps_a_stranded_job(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The reaper must be REACHABLE through the route, not merely callable.

    A repository method nobody calls fixes nothing — the 23 rows measured stuck live on
    2026-08-04 (oldest ~11 days) were reachable only by polling. This drives the real
    HTTP path and asserts the poller stops being told `pending` forever.
    """
    tenant = TENANT_APP
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=tenant,
            video_path="/tmp/v.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        job_id = job.job_id
        # Owning pod died two hours ago: no writes since.
        await session.execute(
            text(
                "UPDATE analysis_jobs SET updated_at = now() - interval '2 hours' WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        await session.commit()

    resp = TestClient(routes_app).get(f"/api/analyze/{job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed", "a poller must not be told `pending` forever"
    # The response reflects the REAPED row, not the stale pre-reap read.
    assert "stranded" in (body.get("error_message") or body.get("error") or "")


async def test_status_poll_leaves_a_live_job_alone(
    routes_app: FastAPI,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Control: polling must never reap work that is still progressing."""
    tenant = TENANT_APP
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=tenant,
            video_path="/tmp/v.mp4",
            video_url=None,
            upload_filename=None,
            request_config={},
            verification_criteria=None,
        )
        job_id = job.job_id

    resp = TestClient(routes_app).get(f"/api/analyze/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] != "failed"


# ---------------------------------------------------------------------------
# Tenant refusal (replaces the old nil-UUID substitution)
# ---------------------------------------------------------------------------
# Before w2v-common 0.67.0 a request with no tenant was silently attributed to
# the nil UUID (registered as `default-anon`), so it type-checked, passed dirty-
# row audits and mis-attributed cost. These prove the refusal instead.


@pytest.mark.parametrize(
    "nil_spelling",
    [
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
    ],
)
def test_nil_tenant_is_refused_in_every_spelling(
    routes_app_without_tenant: FastAPI, nil_spelling: str
) -> None:
    """The nil tenant is rejected BY VALUE, so no spelling of it gets through."""
    attach_test_tenant(routes_app_without_tenant, nil_spelling)
    client = TestClient(routes_app_without_tenant)
    resp = client.get("/api/analyze/does-not-matter")
    assert resp.status_code == 401, resp.text


def test_absent_tenant_is_refused_not_substituted(
    routes_app_without_tenant: FastAPI,
) -> None:
    """No tenant at all must FAIL LOUD — never fall back to a sentinel."""
    client = TestClient(routes_app_without_tenant)
    resp = client.get("/api/analyze/does-not-matter")
    assert resp.status_code == 401, resp.text
