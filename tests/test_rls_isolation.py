"""Proves the Phase D RLS policies ISOLATE tenants (not just that they exist).

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role — the posture the runtime
``w2v_video_analyzer_app`` role has under enforce — and asserts the shipped
policies (applied by ``register_rls_ddl`` on ``create_all``) actually scope
reads/writes. Needs a superuser test DB (``CREATE ROLE`` + ``SET ROLE``); skips
loudly in CI where the role lacks that privilege.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from tests.conftest import TENANT_A, TENANT_B

_ROLE = "w2v_video_analyzer_app_rls_test"
# create_all (used by the db_engine fixture) renders NO server defaults — those
# live in the migrations — so the raw seed must supply every NOT NULL column.
_INS = (
    "INSERT INTO analysis_jobs "
    "(job_id, tenant_id, status, stage, progress, vision_cost_usd, total_cost_usd, "
    "created_at, updated_at) "
    "VALUES (:job_id, :tenant, 'pending', 'queued', 0, 0, 0, now(), now())"
)


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        # skip-gate FIRST (before the RLS-affected seed): a non-superuser CI role
        # can't CREATE ROLE and its seed would hit the policy with no GUC.
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        await conn.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_jobs TO "{_ROLE}"')
        )

        await conn.execute(text(_INS), {"job_id": "job-a", "tenant": TENANT_A})
        await conn.execute(text(_INS), {"job_id": "job-b", "tenant": TENANT_B})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            n = (await conn.execute(text("SELECT count(*) FROM analysis_jobs"))).scalar()
            assert n == 0, f"leaked {n} rows with no GUC"

            await conn.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": TENANT_A}
            )
            rows = (await conn.execute(text("SELECT tenant_id FROM analysis_jobs"))).scalars().all()
            # native uuid column -> psycopg returns uuid.UUID; compare as text.
            seen = [str(r) for r in rows]
            assert seen == [TENANT_A], f"not isolated under GUC=A: {seen}"

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS), {"job_id": "job-x", "tenant": TENANT_B})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))
