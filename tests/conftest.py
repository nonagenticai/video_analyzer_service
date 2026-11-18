"""Shared fixtures for video_analyzer_service tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import clear_settings_cache

# Capture the test DB URL BEFORE the autouse _clear_env fixture
# monkeypatches DATABASE_URL away. CI sets DATABASE_URL to its
# video_analyzer_service_ci postgres; locally falls back to the default
# below (developer must run a local postgres for db_engine tests).
_TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5432/video_analyzer_service_ci",  # pragma: allowlist secret
)

# Seed env BEFORE any test module imports ``src.asgi`` (e.g. ``test_openapi``
# imports the app at module-load to assert the OpenAPI surface). The
# per-test autouse ``_clear_env`` fixture below still wins for per-test
# overrides — these defaults only fill the void during collection.
os.environ.setdefault("DATABASE_URL", _TEST_DATABASE_URL)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("TRANSCRIPTION_CLIENT_URL", "http://localhost:6318")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove env vars that would leak into Settings() during tests."""
    for key in list(os.environ):
        if key.startswith(
            (
                "API_",
                "AUTH_",
                "DATABASE_",
                "REDIS_",
                "STORAGE_",
                "PIPELINE_",
                "VISION_",
                "TRANSCRIPTION_CLIENT_",
                "LOG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ANALYZER_PROVIDER", raising=False)
    # w2v_common 0.15.0+ strips localhost fallbacks on DB/Redis/Storage and
    # the service itself strips it on TRANSCRIPTION_CLIENT_URL. Tests need
    # *some* value to construct settings.
    monkeypatch.setenv("DATABASE_URL", _TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("TRANSCRIPTION_CLIENT_URL", "http://localhost:6318")
    clear_settings_cache()


@pytest.fixture
def tmp_storage_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point STORAGE_LOCAL_PATH + STORAGE_UPLOAD_DIR at a temp dir."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        upload = root_path / "uploads"
        upload.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(root_path / "jobs"))
        monkeypatch.setenv("STORAGE_UPLOAD_DIR", str(upload))
        clear_settings_cache()
        yield root_path


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine() -> AsyncIterator[Any]:
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost:5434 locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine: Any) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with db_session_factory() as session:
        yield session


@pytest.fixture
def mock_storage_service() -> MagicMock:
    """Minimal MagicMock replacement for StorageService."""
    storage = MagicMock()
    storage.get_local_path.return_value = None

    async def _noop_async(*args: Any, **kwargs: Any) -> Any:
        return None

    storage.retrieve = _noop_async
    storage.delete_job_dir = _noop_async
    storage.health_check = _noop_async
    return storage


# ---------------------------------------------------------------------------
# Tenant identities
# ---------------------------------------------------------------------------
# ``analysis_jobs.tenant_id`` is a native postgres ``uuid`` column, so every
# fixture/seed/parametrisation must supply a real uuid — a bare label like
# "acme" raises InvalidTextRepresentation at insert time. A and B are DISTINCT
# on purpose: the tenant-isolation tests need two tenants to prove scoping.
TENANT_A = "11111111-1111-1111-1111-111111111111"
TENANT_B = "22222222-2222-2222-2222-222222222222"
# The tenant the auth-disabled test app attributes requests to. It is a REAL
# uuid, not the nil sentinel: w2v-common 0.67.0 removed the nil-UUID
# substitution and the routes now refuse a request that carries no real tenant,
# so a test app must supply one the way AuthMiddleware does in production.
TENANT_APP = "33333333-3333-3333-3333-333333333333"


def attach_test_tenant(app: Any, tenant_id: str = TENANT_APP) -> Any:
    """Stand in for ``AuthMiddleware``: put a real tenant on ``request.state``.

    The routes read ``request.state.tenant_id`` and 401 when it is absent. In
    production AuthMiddleware sets it from license introspection; a bare test
    app has no middleware, so without this every route test would assert the
    refusal instead of the behaviour it is actually about.
    """

    @app.middleware("http")
    async def _set_tenant(request: Any, call_next: Any) -> Any:
        request.state.tenant_id = tenant_id
        return await call_next(request)

    return app
