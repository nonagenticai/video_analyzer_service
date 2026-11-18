"""SECURITY DEFINER cross-tenant sweeper for abandoned analysis jobs.

A job is owned by an in-process ``asyncio`` task. If the POD dies mid-run — a rollout,
an eviction, an OOM — no handler runs, so the row keeps whatever non-terminal status it
had, forever. Measured on the live database 2026-08-04: 23 rows stuck non-terminal, the
oldest ~11 DAYS. Migration 004 mopped them; by 2026-08-05 the count had climbed back to
2 within seven hours, because nothing prevents re-accumulation.

Until now reaping was **lazy**: ``JobRepository.reap_if_stranded`` runs only inside the
tenant-scoped GET handler, so *a job nobody polls is never reaped*. The stated reason
for the laziness was correct as far as it went — under ``TENANT_RLS_MODE=enforce`` the
runtime is a non-bypass role, so a plain cross-tenant ``UPDATE`` with no
``app.current_tenant`` bound fail-closes to 0 rows.

This module is the answer the compiler already found (``video_compiler_service``'s
``reclaim_orphaned_compile_jobs`` / ``prune_compile_jobs``): run the sweep through a
``SECURITY DEFINER`` function owned by the table owner, which bypasses the ENABLE'd
policies. ``SET search_path FROM CURRENT`` pins the schema captured at CREATE time,
which is both a privilege-escalation guard (a mutable search_path on a SECURITY DEFINER
function is a known attack) and what makes the function resolve ``analysis_jobs`` in the
schema it was created in under CI's ``postgres_isolation: schema``.

EXECUTE is REVOKEd from PUBLIC then GRANTed to the restricted app role
``w2v_video_analyzer_app`` via a ``pg_roles``-guarded DO block, so the runtime can call
it in prod while CI — where the role is absent — skips the grant cleanly. Single-sourced
so BOTH the migration (prod) and the ``create_all`` ``after_create`` event (tests) run an
identical CREATE + REVOKE + guarded-GRANT sequence.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

# The restricted runtime role the app connects as under TENANT_RLS_MODE=enforce.
# Source of truth: gitops clusters/strato/services/w2v/video-analyzer-service-config.env
# ("Phase D RLS: runtime connects as restricted w2v_video_analyzer_app").
APP_ROLE = "w2v_video_analyzer_app"

# Fail every non-terminal job whose owner is provably gone, ACROSS ALL TENANTS.
#
# Two victim classes, deliberately distinct:
#
#  1. ``lease_expires_at < now()`` — a CLAIMED job whose pod stopped heartbeating. This is
#     the strong signal: a live job renews every ``lease/3``, so three consecutive beats
#     must be missed. It is what lets the sweeper reap a dead worker within seconds
#     instead of an hour, and what stops it from killing a live-but-silent vision stage.
#
#  2. ``lease_expires_at IS NULL`` AND stale ``updated_at`` — a row that predates
#     migration 005, or one that died in the window between INSERT and the claim inside
#     ``_execute``. These keep the ORIGINAL 1-hour wall-clock rule, so nothing regresses
#     and no pre-existing row becomes un-reapable.
#
# ``p_active_job_ids`` excludes jobs live on the CALLING pod. That is belt-and-braces
# only — a live job's lease is in the future, so class 1 already excludes it — but it
# closes the narrow window where this pod's own row is between INSERT and claim.
#
# ``FOR UPDATE SKIP LOCKED`` + ``LIMIT`` bound the work and make a concurrent sweep on
# the sibling replica a clean 0-row no-op rather than a lock fight.
V005_SWEEP_FN_SQL = """
CREATE OR REPLACE FUNCTION fail_stranded_analysis_jobs(
    p_stranded_seconds int,
    p_limit int,
    p_active_job_ids text[]
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_count integer;
  BEGIN
    WITH victims AS (
      SELECT id FROM analysis_jobs
       WHERE status NOT IN ('completed', 'failed')
         AND (p_active_job_ids IS NULL OR job_id <> ALL(p_active_job_ids))
         AND (
              (lease_expires_at IS NOT NULL AND lease_expires_at < now())
           OR (lease_expires_at IS NULL
               AND COALESCE(updated_at, created_at)
                   < now() - make_interval(secs => p_stranded_seconds))
         )
       ORDER BY created_at
       LIMIT p_limit
       FOR UPDATE SKIP LOCKED
    )
    UPDATE analysis_jobs j
       SET status = 'failed',
           completed_at = now(),
           updated_at = now(),
           lease_expires_at = NULL,
           error_message = COALESCE(j.error_message, '')
             || 'stranded: the worker that owned this job is gone (pod restart/eviction). '
             || 'Resubmit the analysis.'
      FROM victims v
     WHERE j.id = v.id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
  END
  $fn$
"""

# ---------------------------------------------------------------------------
# Migration 007: durable re-claim instead of fail-on-lease-lapse.
# ---------------------------------------------------------------------------
#
# Before 007 a job whose pod died was only ever FAILED ("stranded ... Resubmit"): a
# rollout 3s after submission threw the caller's work away (2026-10-22, run 01a16423).
# A live pod now RE-CLAIMS it and re-runs it from the start, bounded by
# ``reclaim_attempts``. The inputs are durable (a ``video_url`` or the tenant-scoped
# upload object) and every stage overwrites its own columns, so a from-scratch re-run is
# safe; stages are not checkpointed, so there is nothing finer to resume from.
#
# ``lease_released`` marks a lease this service gave back on graceful shutdown (SIGTERM).
# A drain is not the job's fault, so the next re-claim does not spend an attempt; an
# unclean death (OOM, eviction, node loss) does. Without the cap a job that kills its pod
# would re-claim forever; without the exemption three rollouts in a row would fail a
# perfectly good job.
#
# Atomic by construction: ``FOR UPDATE SKIP LOCKED`` on ONE row, then the UPDATE that
# hands it a fresh lease in the same transaction, so two pods sweeping at once can never
# both take a job — the loser skips the locked row and sees the next one (or none).
#
# When the cap is reached the row is failed HERE (the stranded message, naming the
# attempts) and one all-NULL sentinel row is returned so the caller can tell "consumed a
# poison job" from "queue empty" and keep going — the compiler's
# ``reclaim_next_compile_job`` shape, which learned that the hard way.
RECLAIM_FN_SQL = """
CREATE OR REPLACE FUNCTION reclaim_next_analysis_job(
    p_worker_id text,
    p_lease_seconds int,
    p_max_reclaims int,
    p_active_job_ids text[]
) RETURNS TABLE(r_job_id varchar, r_tenant_id uuid, r_identity jsonb, r_attempts int)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id integer; v_job_id varchar; v_tenant uuid; v_identity jsonb;
    v_attempts integer; v_released boolean; v_stage varchar; v_lease timestamptz;
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.dispatch_identity, j.reclaim_attempts,
           j.lease_released, j.stage, j.lease_expires_at
      INTO v_id, v_job_id, v_tenant, v_identity, v_attempts, v_released, v_stage, v_lease
      FROM analysis_jobs j
     WHERE j.status NOT IN ('completed', 'failed')
       AND j.lease_expires_at IS NOT NULL
       AND j.lease_expires_at < now()
       AND (p_active_job_ids IS NULL OR j.job_id <> ALL(p_active_job_ids))
     ORDER BY j.created_at
     LIMIT 1
     FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;
    END IF;

    IF NOT v_released AND v_attempts >= p_max_reclaims THEN
      UPDATE analysis_jobs
         SET status = 'failed',
             completed_at = now(),
             updated_at = now(),
             lease_expires_at = NULL,
             error_message = format(
               'stranded: lease lapsed at %s at stage %s — the worker that owned this '
               'job is gone (pod restart/eviction) and it was already re-run %s '
               'time(s). Resubmit the analysis.',
               v_lease, COALESCE(v_stage, 'unknown'), v_attempts)
       WHERE id = v_id;
      RETURN QUERY SELECT NULL::varchar, NULL::uuid, NULL::jsonb, v_attempts;
      RETURN;
    END IF;

    UPDATE analysis_jobs
       SET status = 'pending',
           stage = 'queued',
           progress = 0,
           instance_id = p_worker_id,
           lease_expires_at = now() + make_interval(secs => p_lease_seconds),
           reclaim_attempts = v_attempts + CASE WHEN v_released THEN 0 ELSE 1 END,
           lease_released = false,
           updated_at = now()
     WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_identity,
                        v_attempts + CASE WHEN v_released THEN 0 ELSE 1 END;
  END
  $fn$
"""

# The 007 sweeper. Identical to 005 except victim class 1 (lease lapsed) now only takes
# rows that have EXHAUSTED their re-claims: anything with attempts left belongs to
# ``reclaim_next_analysis_job``, and failing it here would race the re-claim on the
# sibling pod and throw the work away again. ``p_max_reclaims = 0`` restores the 005
# behaviour exactly (fail on lapse), which is what a deployment with re-claim disabled
# gets.
SWEEP_FN_SQL = """
CREATE OR REPLACE FUNCTION fail_stranded_analysis_jobs(
    p_stranded_seconds int,
    p_limit int,
    p_active_job_ids text[],
    p_max_reclaims int
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_count integer;
  BEGIN
    WITH victims AS (
      SELECT id FROM analysis_jobs
       WHERE status NOT IN ('completed', 'failed')
         AND (p_active_job_ids IS NULL OR job_id <> ALL(p_active_job_ids))
         AND (
              (lease_expires_at IS NOT NULL AND lease_expires_at < now()
               AND (p_max_reclaims <= 0
                    OR (NOT lease_released AND reclaim_attempts >= p_max_reclaims)))
           OR (lease_expires_at IS NULL
               AND COALESCE(updated_at, created_at)
                   < now() - make_interval(secs => p_stranded_seconds))
         )
       ORDER BY created_at
       LIMIT p_limit
       FOR UPDATE SKIP LOCKED
    )
    UPDATE analysis_jobs j
       SET status = 'failed',
           completed_at = now(),
           updated_at = now(),
           lease_expires_at = NULL,
           error_message = COALESCE(j.error_message, '')
             || 'stranded: the worker that owned this job is gone (pod restart/eviction). '
             || 'Resubmit the analysis.'
      FROM victims v
     WHERE j.id = v.id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
  END
  $fn$
"""

_V005_FNS = ("fail_stranded_analysis_jobs(int, int, text[])",)
_FNS = (
    "fail_stranded_analysis_jobs(int, int, text[], int)",
    "reclaim_next_analysis_job(text, int, int, text[])",
)


def _guarded_grant(grants: list[str]) -> str:
    """Wrap EXECUTE grants in a ``pg_roles``-guarded DO block.

    Grants to ``APP_ROLE`` only when the role exists — so the statement runs in prod
    (role present) and is a clean no-op in CI/tests (role absent).
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- APP_ROLE/grants are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


def _grant_for(fns: tuple[str, ...]) -> str:
    return _guarded_grant([f"GRANT EXECUTE ON FUNCTION {sig} TO {APP_ROLE};" for sig in fns])


def _revoke_for(fns: tuple[str, ...]) -> list[str]:
    return [f"REVOKE EXECUTE ON FUNCTION {sig} FROM PUBLIC" for sig in fns]


def _drop_for(fns: tuple[str, ...]) -> list[str]:
    return [f"DROP FUNCTION IF EXISTS {sig}" for sig in fns]


def v005_upgrade_statements() -> list[str]:
    """FROZEN: exactly what migration 005 ran. Never edit — 007 supersedes it."""
    return [V005_SWEEP_FN_SQL, *_revoke_for(_V005_FNS), _grant_for(_V005_FNS)]


def v005_downgrade_statements() -> list[str]:
    return _drop_for(_V005_FNS)


def upgrade_statements() -> list[str]:
    """The CURRENT function set: what migration 007 and ``create_all`` both run.

    Drops the 3-arg 005 sweeper first. Leaving it would keep a second overload that
    fails lapsed leases unconditionally — exactly the behaviour 007 removes — and an old
    pod still calling it mid-rollout just gets a logged sweep error, which its loop
    already tolerates.
    """
    return [
        *_drop_for(_V005_FNS),
        RECLAIM_FN_SQL,
        SWEEP_FN_SQL,
        *_revoke_for(_FNS),
        _grant_for(_FNS),
    ]


def downgrade_statements() -> list[str]:
    """Reverse 007: drop the current pair and restore the 005 sweeper."""
    return [*_drop_for(_FNS), *v005_upgrade_statements()]


def register_sweep_fn(metadata: MetaData) -> None:
    """Emit the sweeper functions after ``create_all`` so tests get the prod DDL.

    Without this, every test that builds its schema via ``create_all`` (rather than by
    running alembic) would have the columns but not the functions, and the sweeper's own
    tests would fail for a reason unrelated to the behaviour they assert.
    """

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in upgrade_statements():
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
