"""Give analysis jobs a real ownership lease, and a sweeper that can actually run.

Migration 004 failed 23 jobs that had sat non-terminal for over a day — the oldest ~11
DAYS. It was a MOP, not a fix: ``reap_if_stranded`` only fires when someone polls that
specific job, and nothing polls an 11-day-old zombie. Re-accumulation was measured the
same day it was cleaned: 0 stranded at 06:07Z, **2 by ~12:55Z** on 2026-08-05.

This adds the two things needed to close the loop for good.

**1. A lease (``instance_id`` + ``lease_expires_at``).**
Staleness of ``updated_at`` is a weak liveness signal — it only advances on a stage
transition, and the vision stage runs for minutes writing nothing between frames. So the
1-hour threshold had to be generous enough to never kill a slow job, which made it far
too slow to catch a dead one. A heartbeat renewed every ``lease/3`` by the owning pod is
an unambiguous signal instead: lease in the future ⇒ alive, lease lapsed ⇒ the pod is
gone. It is also the OWNERSHIP FENCE this service needs at ``replicas: 2`` — without it,
pod B's poll could fail a job pod A is actively running.

**2. A ``SECURITY DEFINER`` sweeper function.**
The old code documented why a background sweep was impossible: under
``TENANT_RLS_MODE=enforce`` the runtime is a non-bypass role, so a cross-tenant UPDATE
with no ``app.current_tenant`` bound fail-closes to 0 rows. That is true of a PLAIN
statement. ``video_compiler_service`` already solved it — see its ``rls_resolvers.py``
— by routing the cross-tenant sweep through a function owned by the table owner, which
bypasses the ENABLE'd policies. The DDL is single-sourced in
``src/database/rls_resolvers.py`` so this migration and the tests' ``create_all`` run an
identical sequence.

RACE-SAFE BY CONSTRUCTION. This service runs ``replicas: 2`` with RollingUpdate and each
pod has its own migrate initContainer, so two alembic processes can execute this
concurrently. Every statement here is idempotent (``ADD COLUMN IF NOT EXISTS``,
``CREATE INDEX IF NOT EXISTS``, ``CREATE OR REPLACE FUNCTION``), AND ``env.py`` now takes
a session-level ``pg_advisory_lock`` around the whole run so the loser blocks instead of
dying. Belt and braces on purpose: the compiler's migration 004 crashlooped a pod on
exactly this, because it assumed the old pod was torn down first.

Backfill is deliberately EMPTY. Existing non-terminal rows keep ``lease_expires_at IS
NULL``, which the sweeper's second victim class handles under the original 1-hour
wall-clock rule — so nothing pre-005 becomes un-reapable, and nothing is reaped sooner
than it would have been before.

Revision ID: 005_job_ownership_lease
Revises: 004_fail_long_stranded_jobs
"""

from __future__ import annotations

from alembic import op

from src.database.rls_resolvers import v005_downgrade_statements, v005_upgrade_statements

revision = "005_job_ownership_lease"
down_revision = "004_fail_long_stranded_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS instance_id varchar(64)")
    op.execute("ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz")
    # Not CONCURRENTLY: that cannot run inside alembic's transaction, and the table is
    # small (816 rows measured 2026-08-05). The partial predicate matches the ORM's
    # postgresql_where so create_all and the migration produce the same index.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_analysis_jobs_sweep "
        "ON analysis_jobs (lease_expires_at) "
        "WHERE status NOT IN ('completed', 'failed')"
    )
    for stmt in v005_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in v005_downgrade_statements():
        op.execute(stmt)
    op.execute("DROP INDEX IF EXISTS ix_analysis_jobs_sweep")
    op.execute("ALTER TABLE analysis_jobs DROP COLUMN IF EXISTS lease_expires_at")
    op.execute("ALTER TABLE analysis_jobs DROP COLUMN IF EXISTS instance_id")
