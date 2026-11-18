"""Re-claim a job whose pod died instead of failing it.

Live 2026-10-22 (run 01a16423, job 01a1642c): a rollout replaced the single analyzer pod
3s after a job was submitted. The job's lease lapsed, and the only thing this service
could do with a lapsed lease was FAIL it ("stranded ... Resubmit the analysis"), so the
caller's work was thrown away by a deploy.

This adds what a live pod needs to pick the job up and run it again:

* ``reclaim_attempts`` — bounds the re-runs (a job that kills its pod must not loop);
* ``lease_released`` — a lease given back on graceful shutdown does not spend an attempt;
* ``dispatch_identity`` — the submitting request's identity snapshot, so the re-run's
  metered calls carry the original caller's provenance instead of failing closed;
* ``reclaim_next_analysis_job`` (SECURITY DEFINER) — the atomic cross-tenant re-claim,
  and a 4-arg ``fail_stranded_analysis_jobs`` that leaves re-claimable rows alone.

Idempotent statements only, for the same reason as 005: two migrate initContainers can
run at once during a RollingUpdate, and ``env.py``'s advisory lock is the second belt.

Revision ID: 007_durable_reclaim
Revises: 006_tenant_id_uuid
"""

from __future__ import annotations

from alembic import op

from src.database.rls_resolvers import downgrade_statements, upgrade_statements

revision = "007_durable_reclaim"
down_revision = "006_tenant_id_uuid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE analysis_jobs "
        "ADD COLUMN IF NOT EXISTS reclaim_attempts integer NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE analysis_jobs "
        "ADD COLUMN IF NOT EXISTS lease_released boolean NOT NULL DEFAULT false"
    )
    op.execute("ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS dispatch_identity jsonb")
    for stmt in upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in downgrade_statements():
        op.execute(stmt)
    op.execute("ALTER TABLE analysis_jobs DROP COLUMN IF EXISTS dispatch_identity")
    op.execute("ALTER TABLE analysis_jobs DROP COLUMN IF EXISTS lease_released")
    op.execute("ALTER TABLE analysis_jobs DROP COLUMN IF EXISTS reclaim_attempts")
