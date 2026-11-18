"""ASGI app for video_analyzer_service.

Built on the shared ``w2v_common.asgi.create_app`` factory: it wires the
canonical middleware stack (SecurityHeaders → Auth → RequestId), the
license-token BearerAuth/TenantIdHeader OpenAPI customization, and the fleet
error-envelope handlers. Auth (``X-Tenant-Id: Bearer <license_token>``) is gated
by ``settings.auth.enabled`` via the factory's ``auth_enabled`` fallback.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError
from w2v_common.asgi import create_app
from w2v_common.database.rls import install_rls_tenant_rebind
from w2v_common.observability.request_context import RequestIdLogFilter

from src.api.health import router as health_router
from src.api.routes.analysis import router as analysis_router
from src.config import Settings, get_settings
from src.database.connection import close_engine, get_engine

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


_setup_logging()


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress access log noise from health check endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


logging.getLogger("uvicorn.access").addFilter(HealthCheckLoggingFilter())
# Bind request_id onto every log record (the factory registers RequestIdMiddleware,
# which sets the ContextVar; this filter surfaces it in the JSON log formatter).
logging.getLogger().addFilter(RequestIdLogFilter())


settings: Settings = get_settings()


async def _startup(app: FastAPI) -> None:
    """Initialise DB engine, storage, metering, and (optionally) the pipeline runner."""
    logger.info("Starting video_analyzer_service on port %d", settings.api.port)
    app.state.settings = settings

    # Re-bind ``app.current_tenant`` at the start of EVERY transaction, BEFORE any
    # session opens. The tenant GUC is transaction-scoped (``is_local=true``), so any
    # code path that commits and then reads again in the same session ran unbound and
    # RLS hid the row — live 2026-10-22: a poll that reaped a stranded job (commit) then
    # re-read it and answered 404 for the caller's own job. Same call as
    # video_compiler_service's ``asgi._startup``. Inert unless enforce + tenant in context.
    install_rls_tenant_rebind()

    try:
        engine = await get_engine()
        app.state.db_engine = engine
    except (OSError, SQLAlchemyError) as exc:
        logger.error("Failed to connect to database: %s", exc)
        raise

    from src.services.storage_service import StorageService

    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage

    # Cost-metering Redis (separate stream Redis). None when disabled/unreachable;
    # the pipeline emits fire-and-forget so a missing client is a silent no-op.
    from src.services.metering import init_metering_redis

    metering_redis = await init_metering_redis(settings.metering_redis_url)
    app.state.metering_redis = metering_redis

    # Identity of THIS pod for the ownership lease (migration 005). Minted per process,
    # exactly like video_compiler_service's ``app.state.instance_id``. It MUST be the same
    # value the heartbeat renews with and the sweeper excludes on, or a live render is
    # permanently orphaned — which is the trap the compiler's reaper documents.
    app.state.instance_id = uuid.uuid7().hex

    # Pipeline runner is optional at skeleton stage; wired in step 5.
    try:
        from src.services.pipeline import PipelineRunner

        app.state.pipeline_runner = PipelineRunner(
            storage=storage,
            metering_redis=metering_redis,
            instance_id=app.state.instance_id,
        )
    except ImportError, AttributeError:
        logger.info("PipelineRunner not yet available; jobs will queue without execution")

    # Pay librosa/numba's JIT cost HERE, once, instead of charging it to the first
    # caller of POST /api/analyze/audio. Measured: 9.72s on the first call in a
    # process, 0.04s on every call after it. Awaited (not fire-and-forget) so the
    # pod is not marked ready while a request could still race the compile; it is
    # bounded and non-fatal, so a failure degrades latency, never health.
    #
    # The outcome is RECORDED, not just logged: a warm-up failure is (measured in
    # prod) never latency-only — numba's cache-locator error fires on every call,
    # so the beat endpoint is dead for the life of the pod. ``beat_detector_ready``
    # is what lets /api/analyze/audio answer 503 and /health report degraded
    # instead of both of them claiming success. Readiness deliberately ignores it:
    # video analysis (pipeline, OCR, vision) is unaffected.
    from src.services.beat_detector import warm_up as warm_up_beat_detector

    app.state.beat_detector_ready = await warm_up_beat_detector()
    if not app.state.beat_detector_ready:
        logger.error(
            "Beat detector is UNAVAILABLE: warm-up failed, so POST /api/analyze/audio "
            "will answer 503 for the life of this pod. Video analysis is unaffected."
        )

    # Background sweeper for jobs whose owning pod died. Without it, reaping happens ONLY
    # when a client polls that specific job — so a job nobody polls is never reaped
    # (measured on the live DB: 23 rows stuck, oldest ~11 days; and 0 -> 2 in seven hours
    # after migration 004 mopped them).
    runner = getattr(app.state, "pipeline_runner", None)
    if runner is not None:
        from src.database.session import get_session_maker
        from src.repository.job_repository import STRANDED_AFTER_SECONDS
        from src.services.stranded_sweeper import StrandedJobSweeper

        # The sweeper also RE-CLAIMS jobs whose owning pod died (migration 007) and hands
        # them to ``runner.resume`` under THIS pod's instance_id — the same id the
        # heartbeat renews with, or the re-run would lose its own lease on the first beat.
        sweeper = StrandedJobSweeper(
            settings.pipeline,
            get_session_maker,
            runner.active_job_ids,
            STRANDED_AFTER_SECONDS,
            reclaim=runner.resume,
            instance_id=app.state.instance_id,
        )
        sweeper.start()
        app.state.stranded_sweeper = sweeper


async def _shutdown(app: FastAPI) -> None:
    logger.info("Shutting down video_analyzer_service")
    # Close the pipeline's pooled HTTP clients so we don't leak the open TCP
    # connections at shutdown. Tolerant of a missing pipeline (skeleton mode).
    # Stop the sweeper FIRST so it cannot re-claim onto a pod that is going away, then
    # drain: cancel this pod's jobs and give their leases back so the NEXT pod re-claims
    # them on its next sweep instead of after the lease TTL (300s). Best-effort — an
    # unreleased lease still lapses and is re-claimed, it just costs one attempt.
    sweeper = getattr(app.state, "stranded_sweeper", None)
    if sweeper is not None:
        await sweeper.stop()
    runner = getattr(app.state, "pipeline_runner", None)
    if runner is not None:
        try:
            await runner.drain()
        except (OSError, SQLAlchemyError, RuntimeError) as exc:
            logger.warning("Job drain at shutdown failed: %s", exc)
    if runner is not None:
        deps = runner.dependencies
        for client_attr in ("transcription_client", "video_downloader"):
            client = getattr(deps, client_attr, None)
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except (OSError, RuntimeError) as exc:
                    logger.warning("Error closing %s: %s", client_attr, exc)

    from src.services.metering import close_metering_redis

    await close_metering_redis(getattr(app.state, "metering_redis", None))
    try:
        await close_engine()
    except (OSError, SQLAlchemyError) as exc:
        logger.error("Error closing DB: %s", exc)


app = create_app(
    settings,
    routers=[health_router, analysis_router],
    title=settings.app_name,
    version=settings.app_version,
    description="Video analysis and verification microservice",
    on_startup=[_startup],
    on_shutdown=[_shutdown],
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route that
    # forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's default
    # `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "analysis",
            "description": (
                "Video analysis jobs: submission, upload, status, results, "
                "verification, and artifact download."
            ),
        },
    ],
)


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "analyze": "/api/analyze",
            "analyze_audio": "/api/analyze/audio",
            "upload": "/api/analyze/upload",
            "job_status": "/api/analyze/{job_id}",
            "job_result": "/api/analyze/{job_id}/result",
            "verification": "/api/analyze/{job_id}/verification",
            "artifact": "/api/analyze/{job_id}/artifact/{name}",
            "health": "/health",
        },
    }


def main() -> None:
    """Run the ASGI app via uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
