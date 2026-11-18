"""SQLAlchemy ORM models for video_analyzer_service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class AnalysisJob(Base):
    """Video analysis job lifecycle + results."""

    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        # uuid7 (native, py3.14) is the fleet standard for persisted ids — time-
        # ordered so job_id sorts by creation. Python-side default: existing rows
        # keep their historical uuid4 values (both are valid UUID strings).
        default=lambda: str(uuid.uuid7()),
    )
    # Native ``uuid``, matching w2v_orchestrator (the in-stack precedent, already fully
    # uuid). NOT a String subclass — which is load-bearing twice over:
    #   * ``w2v_common.database.rls._predicate`` keys off ``isinstance(col.type, String)``,
    #     so this flip is what makes the tenant_isolation policy render the GUC as
    #     ``…::uuid`` instead of a text-to-text compare. Migration 006 drops and re-creates
    #     the live policy accordingly — the type change alone would leave prod on the old
    #     text predicate and return ZERO rows.
    #   * Postgres refuses ``ALTER COLUMN … TYPE`` on a column referenced by an RLS policy,
    #     so 006 must drop the policy BEFORE the ALTER.
    tenant_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Source inputs
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_filename: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Request config + verification criteria (both stored as JSONB)
    request_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    verification_criteria: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Video metadata (populated after probing)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_audio: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Results (accumulated as the pipeline runs)
    transcript_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    frame_analyses: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    scene_boundaries: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    quality_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    verification_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    artifacts: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    # Cost tracking
    vision_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Ownership lease (migration 005) ---
    # Which pod is running this job. Minted per-process in ``asgi._startup`` as
    # ``uuid7().hex``. Load-bearing on the heartbeat's ``WHERE instance_id = :me``: a pod
    # that has LOST the claim stops renewing instead of resurrecting a lease it no longer
    # owns. NULL for rows created before 005 and for rows never claimed.
    instance_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # When the current claim lapses. The owning pod renews it every
    # ``job_lease_seconds // 3``; a non-terminal row past this value lost its worker.
    #
    # This is the LIVENESS signal that ``updated_at`` is not. ``updated_at`` only moves on
    # a stage transition, and the vision stage can run for minutes writing nothing between
    # frames — so wall-clock staleness alone cannot distinguish "slow" from "dead". A
    # heartbeat can. NULL for pre-005 rows, which keep the legacy staleness rule.
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # --- Durable re-claim (migration 007) ---
    # How many times a live pod has re-claimed this job after its owner died. Bounded by
    # ``PIPELINE_JOB_MAX_RECLAIMS``; past it the job fails with the stranded message
    # instead of re-running forever (a job that OOM-kills its pod would otherwise loop).
    reclaim_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    # True when the owning pod GAVE BACK the lease on graceful shutdown (SIGTERM). A drain
    # is not the job's fault, so the next re-claim does not spend an attempt.
    lease_released: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    # The submitting request's identity snapshot (provenance, licence, subject, actor,
    # project/job attribution, correlation). A re-claim runs on a different pod with no
    # request in scope; without this its metered vision/synthesis calls fail closed as
    # provenance 'unrecorded'. Never holds a credential — see ``services/job_identity``.
    dispatch_identity: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_analysis_jobs_tenant_status", "tenant_id", "status"),
        Index("ix_analysis_jobs_created_at", "created_at"),
        # Drives the cross-tenant sweeper's victim scan. Partial: terminal rows are the
        # overwhelming majority (completed 771 / failed 43 / pending 2, measured
        # 2026-08-05) and are never candidates.
        Index(
            "ix_analysis_jobs_sweep",
            "lease_expires_at",
            postgresql_where=text("status NOT IN ('completed', 'failed')"),
        ),
    )


# Phase D RLS: register the ENABLE + tenant_isolation policy DDL as an after_create
# event on the metadata, so tests' create_all applies the exact policies the
# 002_rls migration ships to prod. Inert until the runtime connects as the
# restricted non-owner role with TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

# Migration 005: the SECURITY DEFINER cross-tenant sweeper. Registered the same way and
# for the same reason as the RLS DDL above — tests build their schema with create_all,
# so without this they would have the lease columns but not the function that consumes
# them. Imported here (not at module top) to keep the models module's import graph flat.
from src.database.rls_resolvers import register_sweep_fn  # noqa: E402

register_sweep_fn(Base.metadata)
