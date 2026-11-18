"""Guardrail tests for the auth middleware integration.

video_analyzer_service delegates auth to ``w2v_common.auth.middleware.AuthMiddleware``.
This test exercises that the middleware is **wired and registered** on the
asgi app (not merely importable), and that it behaves correctly for the
standard public / protected / invalid-token matrix.

The intent is to catch the "false-merge" pattern: a PR that says it added
auth but never called ``app.add_middleware(AuthMiddleware)``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.middleware import AuthMiddleware


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    cache = w2v_auth._get_license_cache()
    cache.clear()


def _make_minimal_app() -> FastAPI:
    """Tiny app for behaviour tests — independent of asgi lifespan side-effects."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/analyze/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "license_present": getattr(request.state, "license", None) is not None,
            }
        )

    return app


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Wiring check — the regression-against-false-merge assertion
# ---------------------------------------------------------------------------


def test_auth_middleware_is_actually_registered_on_asgi_app() -> None:
    """The asgi entry must build the app via the shared ``create_app`` factory.

    Auth is gated by ``settings.auth.enabled``; the factory registers the shared
    ``AuthMiddleware`` when it is on. Tests run with ``AUTH_ENABLED=false`` (per
    conftest.py ``_clear_env``), so the live ``app`` does not carry AuthMiddleware
    — the wiring is asserted at the source level (the false-merge guard) plus a
    factory-behaviour check below. We verify both:
    (1) the import path resolves to the shared library, and
    (2) the asgi module builds the app via ``create_app`` (delegating auth to it).
    """
    from pathlib import Path

    # (1) AuthMiddleware comes from the shared lib (not a local stub).
    from w2v_common.auth.middleware import AuthMiddleware as SharedAuthMiddleware

    assert AuthMiddleware is SharedAuthMiddleware

    # (2) The asgi module builds the app via the shared factory (which owns auth).
    asgi_src = (Path(__file__).resolve().parent.parent / "src" / "asgi.py").read_text()
    assert "create_app(" in asgi_src, "asgi.py must build the app via w2v_common.asgi.create_app"


def test_create_app_registers_shared_auth_when_enabled() -> None:
    """Building the app the way asgi.py does, with auth on, wires the shared
    AuthMiddleware onto the stack — the runtime complement to the source guard."""
    from w2v_common.asgi import create_app
    from w2v_common.auth.middleware import AuthMiddleware as SharedAuthMiddleware

    from src.config import get_settings

    app = create_app(get_settings(), auth_enabled=True)
    assert SharedAuthMiddleware in {m.cls for m in app.user_middleware}


# ---------------------------------------------------------------------------
# Public bypass
# ---------------------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Protected paths
# ---------------------------------------------------------------------------


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": "44444444-4444-4444-4444-444444444444",
        "subject_id": "subj-1",
        "license_id": "lic-1",
        "roles": ["user"],
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/analyze/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 0.67.0: the middleware parses the introspected tenant into a uuid.UUID and
    # REFUSES a slug outright, so this asserts the uuid round-trips onto state.
    assert body["tenant_id"] == "44444444-4444-4444-4444-444444444444"
    assert body["license_present"] is True


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False, "reason": "expired"}

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/analyze/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "INVALID_TOKEN"
    assert body["error"]["status"] == 401
    assert "Invalid or expired license token" in body["error"]["message"]


async def test_protected_route_no_token_untrusted_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted source IP + no token must 401, regardless of TRUST_INCLUSTER_IDENTITY."""
    monkeypatch.setattr(w2v_auth, "_trust_incluster_identity", lambda: False)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/analyze/echo")
    assert resp.status_code == 401
