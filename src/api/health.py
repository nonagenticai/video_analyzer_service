"""Health endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def _pipeline_runner_component(request: Request) -> tuple[str, bool]:
    """Report the component that EXECUTES analysis jobs.

    ``_startup`` imports ``PipelineRunner`` inside a ``try`` that swallows
    ImportError/AttributeError and logs "jobs will queue without execution" at
    INFO. That is a survivable skeleton-mode default and a silent production
    catastrophe: ``POST /api/analyze`` keeps answering 202, every job is written
    ``pending``, nothing ever runs it, and the old ``/health`` said ``healthy``
    the whole time.

    The check is a capability check, not an attribute check: an object on
    ``app.state`` that cannot ``enqueue`` is as useless as no object at all.
    """
    runner = getattr(request.app.state, "pipeline_runner", None)
    if runner is None:
        return (
            "unhealthy: PipelineRunner not loaded -- submitted jobs stay pending forever",
            False,
        )
    if not callable(getattr(runner, "enqueue", None)):
        return (
            "unhealthy: pipeline_runner cannot enqueue -- submitted jobs stay pending forever",
            False,
        )
    return "healthy", True


def _stranded_sweeper_component(request: Request) -> tuple[str, bool]:
    """Report the sweeper loop that fails jobs abandoned by a dead pod.

    Default is NOT-running: the attribute only exists once ``_startup`` reached
    the sweeper wiring, so an unset attribute means startup never got there.
    """
    sweeper = getattr(request.app.state, "stranded_sweeper", None)
    if sweeper is None:
        return (
            "degraded: sweeper not started -- jobs orphaned by a dead pod are never reaped",
            False,
        )
    if not getattr(sweeper, "running", False):
        return "degraded: sweeper loop has exited -- orphaned jobs are never reaped", False
    return "healthy", True


@router.get("/health", response_model=None, operation_id="health_check")
async def health_check(request: Request) -> dict[str, Any] | JSONResponse:
    """Return combined DB + API + executor + sweeper + beat-detector health."""
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except (OSError, SQLAlchemyError) as exc:
        logger.warning("Database health check failed: %s", exc)
        db_status = f"unhealthy: {exc}"

    # Beat detection is ONE endpoint of several; when its numba warm-up failed,
    # /api/analyze/audio is dead but the video pipeline is fine. So it lands here
    # as "degraded" and deliberately NOT in /health/ready — failing readiness
    # would take working video analysis out of the load balancer too.
    # Default False, NOT True. The attribute is only set once ``_startup`` reached
    # the warm-up call, so an *unset* attribute means startup never got that far —
    # exactly the case a health check exists to catch. Defaulting to True made an
    # incomplete startup indistinguishable from a warm, working pod.
    beat_ready = getattr(request.app.state, "beat_detector_ready", False)
    beat_status = "healthy" if beat_ready else "degraded: warm-up failed, /api/analyze/audio is 503"

    runner_status, runner_ok = _pipeline_runner_component(request)
    sweeper_status, sweeper_ok = _stranded_sweeper_component(request)

    if not runner_ok:
        # Nothing executes jobs. Distinguished from "degraded" so an alert can
        # tell "one endpoint is sick" from "this pod accepts work it will never
        # do" — the failure the old payload could not express at all.
        overall = "unhealthy"
    elif db_status != "healthy" or not beat_ready or not sweeper_ok:
        overall = "degraded"
    else:
        overall = "healthy"
    return {
        "status": overall,
        "components": {
            "database": db_status,
            "api": "healthy",
            "beat_detector": beat_status,
            "pipeline_runner": runner_status,
            "stranded_sweeper": sweeper_status,
        },
    }


@router.get("/health/live", response_model=None, operation_id="health_live")
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/healthz", operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=None, operation_id="health_ready")
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe.

    The pipeline runner, the sweeper and the beat detector are deliberately NOT
    checked here:

    * **A missing PipelineRunner is a build-time fault, so it is shared.** It
      comes from an import that failed against the image — which means it failed
      identically on every replica. Failing readiness would evict the entire
      Deployment at once, taking down status polling, results, verification and
      artifact download for every job already in flight, on top of an executor
      outage. Blindness is bad; deleting the read surface too is worse.
    * **The sweeper is a background janitor.** It repairs rows; it serves no
      request. Nothing about a request is wrong when it is dead.
    * **The beat detector is one endpoint of several.** Video analysis (pipeline,
      OCR, vision) is unaffected by a failed numba warm-up, and that endpoint
      already answers an honest 503 on its own.

    So all three land on ``/health`` — ``unhealthy`` for the executor, ``degraded``
    for the other two — loud to an operator and to alerting, inert to the
    scheduler. Readiness stays what it can honestly assert: this pod can reach
    its database and serve HTTP.
    """
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except (OSError, SQLAlchemyError) as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "error": str(exc),
            },
        )
    return {"status": "healthy", "ready": True}


@router.get("/readyz", response_model=None, operation_id="health.ready")
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
