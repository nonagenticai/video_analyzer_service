"""``alembic upgrade head`` must actually COMMIT — and two of them must not race.

This test exists because of a bug introduced (and caught) while adding the advisory lock
in ``src/database/migrations/env.py``. The first version ran the lock statement without
committing it, which under SQLAlchemy 2.0 leaves an implicit transaction open on the
connection. Alembic's own ``begin_transaction()`` then NESTED inside it instead of owning
the outer transaction, so nothing was ever committed:

    $ alembic upgrade head
    INFO  Running upgrade  -> 001_initial
    ...
    INFO  Running upgrade 005_... -> 006_tenant_id_uuid
    $ echo $?
    0
    $ psql -c 'select * from alembic_version'
    ERROR:  relation "alembic_version" does not exist

Exit code 0, full "Running upgrade" log, and an empty database. On a service whose
initContainer gates the rollout, a silently-no-op migration is the worst possible shape of
this failure: the deploy goes green and the schema never moves. Nothing in the suite would
have caught it, because every other DB test builds its schema with ``create_all`` and
never runs alembic at all.

The concurrency half pins the reason the lock is there: this service runs ``replicas: 2``
with RollingUpdate, so two migrate initContainers can start at once. The compiler
CrashLoopBackOff'd a pod on exactly that.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _admin_url(url: str, dbname: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{dbname}", parts.query, parts.fragment))


def _sync(url: str) -> str:
    return url.replace("+psycopg", "").replace("postgresql://", "postgresql+psycopg://")


@pytest.fixture
def scratch_database() -> Iterator[str]:
    """A throwaway database, because alembic must run against a genuinely empty one."""
    if not _TEST_DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    name = f"alembic_persist_{uuid.uuid4().hex[:12]}"
    admin = create_engine(
        _sync(_admin_url(_TEST_DATABASE_URL, "postgres")), isolation_level="AUTOCOMMIT"
    )
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cannot create a scratch database: {exc}")
    try:
        yield _admin_url(_TEST_DATABASE_URL, name)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _upgrade(url: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=_REPO_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


@pytest.mark.integration
def test_upgrade_head_commits_and_survives_a_concurrent_run(scratch_database: str) -> None:
    """Both halves in one run: the schema PERSISTS, and two racers both succeed."""
    first, second = _upgrade(scratch_database), _upgrade(scratch_database)
    out1, out2 = first.communicate()[0], second.communicate()[0]

    assert first.returncode == 0, out1.decode()
    assert second.returncode == 0, out2.decode()
    combined = (out1 + out2).decode().lower()
    assert "duplicate" not in combined, combined
    assert "traceback" not in combined, combined

    engine = create_engine(_sync(scratch_database))
    try:
        with engine.connect() as conn:
            # THE ASSERTION THAT WOULD HAVE CAUGHT THE SILENT NO-OP.
            versions = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
            assert versions == ["007_durable_reclaim"], versions

            columns = set(
                conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'analysis_jobs'"
                    )
                ).scalars()
            )
            assert {"instance_id", "lease_expires_at"} <= columns

            secdef, proconfig = conn.execute(
                text(
                    "SELECT prosecdef, proconfig FROM pg_proc "
                    "WHERE proname = 'fail_stranded_analysis_jobs'"
                )
            ).one()
            assert secdef is True
            assert any(str(c).startswith("search_path=") for c in (proconfig or []))
    finally:
        engine.dispose()
