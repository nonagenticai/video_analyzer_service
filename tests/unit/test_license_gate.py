"""The entitlement gate on /api/analyze actually refuses an unlicensed tenant.

Until w2v_common 0.75.0 this service authenticated the licence token for
tenant IDENTITY and stopped there: ``AuthMiddleware`` answers "who are you",
never "may you". A tenant whose licence was ``expired`` or ``revoked`` still
presented a syntactically valid token, so every ``/api/analyze`` route served
them. ``LICENSE_GATE`` is mounted router-wide in ``src/api/routes/analysis.py``
to close that.

Every test here drives a REAL HTTP request through the REAL ``analysis_router``
(and the REAL health router), because the interesting question is not whether
``require_license`` works — w2v_common owns that — but whether THIS service's
routes actually carry it. This FastAPI build includes sub-routers lazily
(``_IncludedRouter``), so the parent router's ``dependencies=`` are applied
when a request is matched, not baked into the route objects at import time:
static introspection of ``route.dependant`` reports the sub-router routes as
ungated and would be a *false negative*. A request is the only honest probe.

The paired 200s (entitled, flag-off) are what stop this file from passing
vacuously: if the gate were mounted wrongly (or the fixture broken) they would
not be 200s either.

No postgres CI database is touched here on purpose: the gate is a router-level
FastAPI dependency, so on the 402/503 paths it runs and refuses BEFORE the
handler (and its DB session) ever executes. On the pass-through paths (entitled,
flag unset, flag false) ``get_job_status`` does run, so ``get_session_maker``,
``bind_session_tenant`` and ``JobRepository`` are monkeypatched to a cheap
in-memory stub — the point of those tests is "the gate let the request through
to the handler", not "the handler's DB integration works" (that is
``tests/unit/test_analysis_routes.py``'s job).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.health import router as health_router
from src.api.routes import analysis as analysis_module
from src.api.routes.analysis import router as analysis_router

TENANT = "33333333-3333-3333-3333-333333333333"
SUBJECT = "user-under-test"
JOB_ID = "job-under-test"

#: The cheapest genuinely tenant-scoped route on the gated router: its only
#: work past the gate is a JobRepository lookup, which is stubbed below. So a
#: non-200 here can only be the gate — which is the whole point.
GATED_PATH = f"/api/analyze/{JOB_ID}"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness
#: probe CrashLoops the pod, and an unlicensed tenant cannot be billed by
#: killing our own deployment.
PROBE_PATH = "/health/live"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TENANT,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


@dataclass
class _StubJob:
    """Just enough of ``AnalysisJob`` for ``_job_to_status`` to serialize."""

    job_id: str = JOB_ID
    status: str = "completed"
    stage: str = "done"
    progress: float = 1.0
    error_message: str | None = None
    artifacts: dict[str, str] | None = None
    duration_seconds: float | None = 12.5
    width: int | None = 1920
    height: int | None = 1080
    has_audio: bool | None = True
    vision_cost_usd: float = 0.0
    verification_result: dict[str, Any] | None = None


class _StubJobRepository:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def get_by_job_id(self, job_id: str, *, tenant_id: str) -> _StubJob:
        return _StubJob(job_id=job_id)

    async def reap_if_stranded(self, job: Any, *, tenant_id: str, max_reclaims: int = 0) -> bool:
        return False


class _StubSession:
    async def __aenter__(self) -> _StubSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class _StubSessionMaker:
    def __call__(self) -> _StubSession:
        return _StubSession()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Make the gated handler's own work free, so only the gate can fail it.

    ``get_job_status`` calls these module-level names directly (not via
    FastAPI ``Depends``), so they are patched on the ``analysis`` module
    rather than overridden through ``app.dependency_overrides``.
    """

    async def _fake_session_maker() -> _StubSessionMaker:
        return _StubSessionMaker()

    async def _noop_bind(session: Any, tenant_id: Any) -> None:
        return None

    monkeypatch.setattr(analysis_module, "get_session_maker", _fake_session_maker)
    monkeypatch.setattr(analysis_module, "bind_session_tenant", _noop_bind)
    monkeypatch.setattr(analysis_module, "JobRepository", _StubJobRepository)


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health_router)
    application.include_router(analysis_router)
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.asyncio
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == JOB_ID


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_is_served_for_a_revoked_tenant(app, monkeypatch):
    """A gated probe path CrashLoops the pod. /health/live must stay 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, PROBE_PATH)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
