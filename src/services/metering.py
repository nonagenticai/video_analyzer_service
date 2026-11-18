"""Cost metering for video_analyzer_service.

A thin service-identity wrapper over the shared cost-bus emitter in
``w2v_common.metering``. The shared helper owns the canonical cost-bus stream
name (``improvement-loop:costs``, overridable via ``METERING_STREAM``), the
EMITTER_GUIDE.md field shape, the never-raise guard and the GC-safe
fire-and-forget task lifecycle. This module only pins this service's fixed
identity (``service_key`` / ``cost_source`` / ``provider`` / ``model``) and
re-exports the Redis connection lifecycle helpers.

The expensive vision-LLM stage already meters through worker_management and
transcription through asr_backend; this emitter prices the service's *local*
non-LLM compute (frame extraction, scene detection, OCR, quality metrics) at a
flat per-minute-of-source-video rate. The billable unit is video duration:
``input_tokens`` = whole seconds of source video, ``output_tokens`` = 0. The
emitted ``cost_usd`` is only an advisory hint — the metering/billing service
recomputes the authoritative cost from the centralised ``provider_pricing``
table (the ``video-analyzer`` provider row, priced per minute).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from w2v_common.auth import W2V_APP_KEY
from w2v_common.metering import (
    close_metering_redis,
    cost_event_key,
    init_metering_redis,
    resolve_stream,
)
from w2v_common.metering import (
    schedule_cost_event as _schedule_cost_event,
)

if TYPE_CHECKING:
    import redis.asyncio as aioredis

__all__ = [
    "APP_ID",
    "COST_SOURCE",
    "FUNDING",
    "PROVIDER",
    "SERVED_MODEL",
    "SERVICE_KEY",
    "analysis_event_key",
    "close_metering_redis",
    "init_metering_redis",
    "resolve_stream",
    "schedule_cost_event",
]

COST_SOURCE = "w2v-video-analyzer"
SERVICE_KEY = "w2v-video-analyzer"  # pragma: allowlist secret -- routing slug, not a credential
PROVIDER = "video-analyzer"
SERVED_MODEL = "local"
#: The app the cost belongs to: the licences registry KEY of w2v (billing's
#: ``split_asserted_app`` resolves a non-uuid app_id through its app-key mirror).
APP_ID = W2V_APP_KEY
#: WHO FUNDS this event. The only thing this emitter prices is the service's own
#: local compute (frames / scenes / OCR / quality) on platform hardware; the
#: vision-LLM spend is metered by worker_management, transcription by asr_backend.
FUNDING = "local_gpu"
#: Step label of the one billable event per job (the terminal completion).
_ANALYSIS_STEP = "analysis-complete"


def analysis_event_key(job_id: str) -> str:
    """Deterministic billing idempotency key for one job's completion event.

    The completion is billed at most once per job (gated on the terminal write in
    ``pipeline``), so ``(service, job_id, step)`` identifies the physical call: a
    re-emit of the same job dedupes at billing, two jobs never collide.
    """
    return cost_event_key(SERVICE_KEY, job_id, _ANALYSIS_STEP)


def schedule_cost_event(
    redis_conn: aioredis.Redis,
    *,
    tenant_id: str,
    job_id: str,
    license_id: str,
    subject_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    cost_usd: float = 0.0,
    provider_request_id: str = "",
    usage_extra: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget a FULLY ATTRIBUTED video-analyzer cost event (GC-safe).

    Injects this service's fixed identity, the w2v app, ``funding`` and the
    job's deterministic ``event_key``, and delegates field-building, stream
    resolution and task lifecycle to ``w2v_common.metering``. Project and actor
    come from w2v_common's ambient chain (cell claims / trusted headers).

    Under ``METERING_REQUIRE_ATTRIBUTION`` a missing licence/subject raises
    ``MissingAttributionError`` (fail loud) instead of shipping a partial event.
    ``input_tokens`` is whole seconds of source video (per-minute pricing);
    ``cost_usd`` is advisory (the consumer recomputes from the pricing table).
    """
    _schedule_cost_event(
        redis_conn,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        provider=PROVIDER,
        model=SERVED_MODEL,
        served_model=SERVED_MODEL,
        tenant_id=tenant_id,
        license_id=license_id,
        subject_id=subject_id,
        app_id=APP_ID,
        funding=FUNDING,
        event_key=analysis_event_key(job_id),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        provider_request_id=provider_request_id,
        usage_extra=usage_extra,
    )
