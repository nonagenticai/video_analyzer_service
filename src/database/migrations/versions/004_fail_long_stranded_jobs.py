"""Fail jobs stranded in a non-terminal status for over a day.

A job is owned by an in-process background task. When the pod dies mid-run — rollout,
eviction, OOM — no handler runs, so the row keeps its non-terminal status forever and a
client polling it waits on work nobody is doing.

Measured on the live database 2026-08-04: 23 rows stuck non-terminal, the oldest ~11
DAYS, across stages ``scene_detection`` / ``vision`` / ``probe`` / ``quality``.

``reap_if_stranded`` stops NEW ones accumulating, but it only fires when someone polls
that job, and nothing will ever poll an 11-day-old zombie. This repairs the backlog the
same way 003 repaired the untimestamped failures.

CUTOFF is 24 hours of no writes — deliberately far above the runtime reaper's 1 hour.
The live timing profile (7 days: 210 completed jobs, p50 16.7s, p90 108.6s, max 176.2s)
puts a day at ~490x the slowest job that has ever finished here, so this cannot touch
work that is merely slow, even allowing for a much larger video than any seen so far.

Staleness, not age: ``updated_at`` advances on every stage transition, so a genuinely
progressing job is never selected no matter how long it has been running.

Revision ID: 004_fail_long_stranded_jobs
Revises: 003_backfill_failed_completed_at
"""

from __future__ import annotations

from alembic import op

revision = "004_fail_long_stranded_jobs"
down_revision = "003_backfill_failed_completed_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `completed_at` is stamped here for the same reason 003 exists: `failed` is
    # terminal, and a terminal row without a completion time is indistinguishable from
    # one still running. `updated_at` is used as the completion time — it is when the
    # job actually stopped being worked on, which is the honest answer.
    #
    # Idempotent: the status predicate excludes anything already terminal, so a replay
    # selects nothing.
    op.execute(
        """
        UPDATE analysis_jobs
        SET status = 'failed',
            completed_at = COALESCE(completed_at, updated_at, created_at),
            error_message = COALESCE(error_message, '')
                || 'stranded: no progress for over 24h — the worker that owned this job '
                || 'is gone (pod restart/eviction). Failed by migration 004.'
        WHERE status NOT IN ('completed', 'failed')
          AND COALESCE(updated_at, created_at) < now() - interval '24 hours'
        """
    )


def downgrade() -> None:
    # Not reversible: the original non-terminal status is not recorded anywhere, and
    # restoring rows to `pending` would recreate zombies that poll forever. These jobs
    # are dead either way — the migration only makes that legible.
    pass
