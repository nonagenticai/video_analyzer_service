"""Backfill completed_at on terminal `failed` rows that never got one.

`JobRepository.mark_failed` stamped `status='failed'` but not `completed_at`, while
`mark_completed` stamped both. So every failure ever recorded looks — to any reader
keyed on `completed_at` — like a job that is still running. Measured on the live
database 2026-08-04: 22 such rows, the oldest ~25 days old.

The code defect is fixed alongside this migration; this repairs the history it left,
because a fix that only helps future jobs leaves the existing rows permanently
ambiguous.

`updated_at` is the backfill value and is the RIGHT one, not a guess: `update_fields`
sets `updated_at=now()` on the same statement that writes `status='failed'`, so for a
row whose last write was the failure it IS the failure time. A row updated after
failing would be slightly late, which is still strictly better than NULL and is why
this is a one-shot repair rather than a recurring job.

Deliberately NOT touched: rows in a non-terminal status (`pending`/`processing`) with
a NULL `completed_at`. Those are correctly NULL — some are genuinely stranded (23 were
live at the time of writing, the oldest ~11 days) but stamping them would invent a
completion that never happened and hide the stranding. That needs a reaper, not a
backfill.

Revision ID: 003_backfill_failed_completed_at
Revises: 002_rls
"""

from __future__ import annotations

from alembic import op

revision = "003_backfill_failed_completed_at"
down_revision = "002_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent by construction: the WHERE clause excludes anything already stamped,
    # so a replay is a no-op rather than a rewrite of correct timestamps.
    op.execute(
        """
        UPDATE analysis_jobs
        SET completed_at = updated_at
        WHERE status = 'failed'
          AND completed_at IS NULL
          AND updated_at IS NOT NULL
        """
    )


def downgrade() -> None:
    # Intentionally not reversible. Clearing completed_at for every failed row would
    # also clear the ones written correctly by the fixed `mark_failed` after this
    # migration ran — the backfill is indistinguishable from a real stamp, which is
    # exactly the point. Re-introducing the ambiguity is never the desired outcome.
    pass
