"""Health endpoint tests via TestClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import health as health_module
from src.api.health import router as health_router


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build a FastAPI app with health router and monkeypatched DB."""

    class _FakeConn:
        async def execute(self, _query: Any) -> Any:
            return None

        async def __aenter__(self) -> _FakeConn:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    fake_engine = MagicMock()
    fake_engine.connect = MagicMock(return_value=_FakeConn())

    async def _fake_get_engine() -> Any:
        return fake_engine

    monkeypatch.setattr(health_module, "get_engine", _fake_get_engine)
    app = FastAPI()
    app.include_router(health_router)
    _make_fully_started(app)
    return app


class _FakeRunner:
    """Stands in for PipelineRunner: only ``enqueue`` is on the health path."""

    async def enqueue(self, job_id: str, tenant_id: str) -> None:
        return None


class _FakeSweeper:
    """Stands in for StrandedJobSweeper: only ``running`` is on the health path."""

    def __init__(self, *, running: bool) -> None:
        self.running = running


def _make_fully_started(app: FastAPI) -> FastAPI:
    """Put the app in the state a COMPLETED ``_startup`` leaves it in.

    Every one of these attributes now defaults to not-healthy when absent, so a
    test that wants a green pod has to say so — which is the point.
    """
    app.state.beat_detector_ready = True
    app.state.pipeline_runner = _FakeRunner()
    app.state.stranded_sweeper = _FakeSweeper(running=True)
    return app


def test_health_check_healthy(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["components"]["database"] == "healthy"


def test_health_reports_beat_detector_degraded_but_ready_stays_green(app: FastAPI) -> None:
    """A dead beat detector must be VISIBLE on /health and INVISIBLE to readiness.

    Failing readiness would evict a pod whose video-analysis pipeline is fine.
    """
    app.state.beat_detector_ready = False
    client = TestClient(app)

    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["beat_detector"].startswith("degraded")
    assert body["components"]["database"] == "healthy"

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_health_live(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_ready(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True


def test_health_check_database_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _broken_engine() -> Any:
        raise OSError("database is down")

    monkeypatch.setattr(health_module, "get_engine", _broken_engine)
    app = _make_fully_started(FastAPI())
    app.include_router(health_router)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "database is down" in body["components"]["database"]


def test_health_ready_database_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _broken_engine() -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(health_module, "get_engine", _broken_engine)
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False


# ---------------------------------------------------------------------------
# The component that EXECUTES the work. Each state below is one in which the old
# payload said "healthy" while submitted jobs piled up as `pending` forever.
# ---------------------------------------------------------------------------


def test_health_unhealthy_when_pipeline_runner_is_absent(app: FastAPI) -> None:
    """The ImportError branch in _startup ran: jobs are accepted and never executed."""
    del app.state.pipeline_runner
    body = TestClient(app).get("/health").json()

    assert body["status"] == "unhealthy"
    assert "stay pending forever" in body["components"]["pipeline_runner"]
    # The dependency is fine; the fault must be attributed to the executor.
    assert body["components"]["database"] == "healthy"


def test_health_unhealthy_when_pipeline_runner_cannot_enqueue(app: FastAPI) -> None:
    """A capability check, not an attribute check -- 'something is set' is not a signal."""

    class _Useless:
        pass

    app.state.pipeline_runner = _Useless()
    body = TestClient(app).get("/health").json()

    assert body["status"] == "unhealthy"
    assert body["components"]["pipeline_runner"].startswith("unhealthy")


def test_health_degraded_when_the_stranded_sweeper_loop_has_exited(app: FastAPI) -> None:
    """A dead sweeper leaves jobs from a crashed pod stuck at 'processing' forever."""
    app.state.stranded_sweeper = _FakeSweeper(running=False)
    body = TestClient(app).get("/health").json()

    assert body["status"] == "degraded"
    assert "never reaped" in body["components"]["stranded_sweeper"]


def test_health_degraded_when_the_sweeper_was_never_started(app: FastAPI) -> None:
    """Startup never reached the sweeper wiring -- no attribute at all."""
    del app.state.stranded_sweeper
    body = TestClient(app).get("/health").json()

    assert body["status"] == "degraded"
    assert "not started" in body["components"]["stranded_sweeper"]


def test_unset_beat_detector_attribute_is_not_healthy(app: FastAPI) -> None:
    """An unset attribute means startup never warmed it -- that is not 'healthy'."""
    del app.state.beat_detector_ready
    body = TestClient(app).get("/health").json()

    assert body["status"] == "degraded"
    assert body["components"]["beat_detector"].startswith("degraded")


def test_readiness_stays_green_when_nothing_executes_jobs(app: FastAPI) -> None:
    """Deliberate: a missing runner is an image-level fault, so it is shared.

    Evicting every replica at once would also delete status polling, results and
    artifact download for jobs already in flight.
    """
    del app.state.pipeline_runner
    del app.state.stranded_sweeper
    resp = TestClient(app).get("/readyz")

    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_health_is_green_only_when_every_component_reports_itself(app: FastAPI) -> None:
    body = TestClient(app).get("/health").json()
    assert body["status"] == "healthy"
    assert body["components"]["pipeline_runner"] == "healthy"
    assert body["components"]["stranded_sweeper"] == "healthy"
