"""Phase D RLS — enable row-level security on the tenant-scoped tables.

⚠️ PINNED LITERAL DDL — DO NOT "simplify" THIS BACK TO ``upgrade_statements(Base.metadata)``.

This migration used to render its policy from ``w2v_common.database.rls`` against the
LIVE ``Base.metadata``. That makes history mutable: the predicate the migration emits
changes whenever the model changes. When ``analysis_jobs.tenant_id`` became a native
``uuid`` (migration 006), an ``alembic upgrade head`` on an EMPTY database died right
here — 001 creates the column as ``varchar(128)``, but the helper rendered the uuid-era
predicate against it::

    (psycopg.errors.UndefinedFunction) operator does not exist: character varying = uuid

So the DDL below is spelled out verbatim as it was correct at THIS revision: one table
(``analysis_jobs``), ``tenant_id varchar(128) NOT NULL`` → the strict, text-comparison
predicate. Migration 006 is the revision that flips it to ``::uuid``; that one may keep
using the helper because at 006 the live metadata IS the correct era.

ENABLE-not-FORCE → inert until the runtime connects as the restricted non-owner role
``w2v_video_analyzer_app`` with ``TENANT_RLS_MODE=enforce``. The background pipeline
receives its tenant_id from the enqueue call and binds it on every session, so there is
no cross-tenant background loop and no SECURITY DEFINER resolver is needed. Fully
reversible.

Revision ID: 002_rls
Revises: 001_initial
"""

from alembic import op

revision = "002_rls"
down_revision = "001_initial"
branch_labels = None
depends_on = None

# The era-correct predicate: at 001_initial ``tenant_id`` is ``varchar(128)``, so the
# GUC is compared text-to-text with NO ``::uuid`` cast. Verbatim as Postgres reports it
# back after this migration runs:
#   ((tenant_id)::text = NULLIF(current_setting('app.current_tenant'::text, true), ''::text))
_GUC_TEXT = "NULLIF(current_setting('app.current_tenant', true), '')"
_PREDICATE = f"tenant_id = {_GUC_TEXT}"

# Every policy name the helper could ever emit for this table, so the downgrade stays a
# total reverse of the upgrade (strict + the nullable "global-or-own" variants).
_ALL_POLICY_NAMES = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)


def upgrade() -> None:
    op.execute("ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON analysis_jobs")
    op.execute(
        f"CREATE POLICY tenant_isolation ON analysis_jobs "
        f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
    )


def downgrade() -> None:
    for policy in _ALL_POLICY_NAMES:
        op.execute(f"DROP POLICY IF EXISTS {policy} ON analysis_jobs")
    op.execute("ALTER TABLE analysis_jobs DISABLE ROW LEVEL SECURITY")
