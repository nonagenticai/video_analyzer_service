"""Fleet ruling: ``analysis_jobs.tenant_id`` becomes a native ``uuid``.

Measured on the live standby 2026-08-26: 1605 rows, **0 dirty** — every value already
parses as a uuid, so ``USING tenant_id::uuid`` cannot fail and no data cleanup precedes
this. Purely a type flip.

⚠️ The RLS policy is NOT incidental here, it is the whole risk:

  * ``analysis_jobs`` carries a ``tenant_isolation`` policy (migration 002) whose
    predicate is rendered by ``w2v_common.database.rls._predicate``, which chooses the
    GUC cast from the column type: ``isinstance(col.type, String)`` → text-to-text,
    otherwise ``…::uuid``. The policy live in prod today was rendered against
    ``String(128)``, so it compares ``tenant_id = NULLIF(current_setting(…), '')``.
    Once the column is ``uuid`` that predicate is ``uuid = text`` → ``operator does not
    exist``. Leaving it in place would break every RLS-bound query.
  * Independently, Postgres **refuses** ``ALTER TABLE … ALTER COLUMN … TYPE`` while the
    column is referenced by a policy USING/WITH CHECK expression.

Both force the same order, which is what this migration does in one transaction:
DROP policy → ALTER TYPE → re-CREATE policy. The re-CREATE renders from
``upgrade_statements(Base.metadata)``, i.e. the same single source 002 and the tests'
``create_all`` event use, so the migrated policy and the test-built policy cannot drift.

The ``downgrade`` cannot use that helper — it renders from the CURRENT code's metadata,
which is uuid — so it spells the legacy text predicate out literally.

Not touched, deliberately:
  * ``fail_stranded_analysis_jobs`` (migration 005) never references ``tenant_id``, so
    the SECURITY DEFINER sweeper needs no recreate.
  * Indexes ``ix_analysis_jobs_tenant_id`` / ``ix_analysis_jobs_tenant_status`` are
    rebuilt automatically by the ALTER; nullability (NOT NULL) is unchanged.

Revision ID: 006_tenant_id_uuid
Revises: 005_job_ownership_lease
"""

from alembic import op
from w2v_common.database.rls import upgrade_statements

from src.database.models import Base

revision = "006_tenant_id_uuid"
down_revision = "005_job_ownership_lease"
branch_labels = None
depends_on = None

# The legacy (varchar) predicate, spelled out for the downgrade only. Must stay
# byte-compatible with what ``_predicate`` renders for a ``String`` column.
_LEGACY_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
_LEGACY_PREDICATE = f"tenant_id = {_LEGACY_GUC}"


def upgrade() -> None:
    # 1. The policy references tenant_id — the ALTER below is refused until it is gone.
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON analysis_jobs")

    # 2. 0 dirty rows measured live, so this USING clause is total.
    op.execute("ALTER TABLE analysis_jobs ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")

    # 3. Re-create from the single source. Base.metadata now carries PgUUID, so
    #    ``_predicate`` renders the GUC with a ``::uuid`` cast. This also re-runs the
    #    idempotent ``ENABLE ROW LEVEL SECURITY`` (ENABLE, never FORCE — owner bypass).
    for stmt in upgrade_statements(Base.metadata):
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON analysis_jobs")
    op.execute(
        "ALTER TABLE analysis_jobs ALTER COLUMN tenant_id TYPE varchar(128) USING tenant_id::text"
    )
    op.execute("ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON analysis_jobs "
        f"USING ({_LEGACY_PREDICATE}) WITH CHECK ({_LEGACY_PREDICATE})"
    )
