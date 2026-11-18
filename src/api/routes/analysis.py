"""Analysis API routes (POST /api/analyze, upload, poll, artifact, delete)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.requests import ClientDisconnect
from w2v_common.auth import LICENSE_GATE
from w2v_common.database.rls import bind_session_tenant
from w2v_common.tenant import TenantError

from src.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AudioBeatGrid,
    AudioBeatGridRequest,
    JobResultResponse,
    JobStatusResponse,
    QualityMetricsResponse,
    VerificationResult,
)
from src.config import get_settings
from src.database.models import AnalysisJob
from src.database.session import get_session_maker
from src.repository.job_repository import JobRepository
from src.services import beat_detector
from src.services.job_identity import capture_dispatch_identity
from src.services.video_downloader import VideoDownloader
from src.tenant import require_tenant

logger = logging.getLogger(__name__)

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted on this
# router. Until now this service authenticated the licence token for tenant
# IDENTITY only (AuthMiddleware answers "who are you") and never asked "may
# you" — a tenant whose licence is expired or revoked still carries a
# syntactically valid token and was served every /api/analyze route.
# LICENSE_GATE closes that: none/expired/revoked -> 402 license_<state>, an
# unestablishable verdict -> 503 license_unavailable + Retry-After,
# active/expiring_soon -> pass.
#
# Every route under this router is tenant-scoped analysis work (create job,
# upload+analyze, audio beat-grid, status/result/verification polling,
# artifact download, delete) so every one of them is correctly gated here —
# there is no internal-callback or peer-to-peer route on this prefix.
#
# DELIBERATELY NOT GATED, each for a stated reason:
#   * the health router (/health, /health/live, /health/ready and aliases),
#     mounted separately in asgi.py. A 402/503 on a probe CrashLoops the pod,
#     and an unlicensed tenant cannot be billed by killing our own
#     deployment. The gate also refuses these paths by name
#     (is_license_exempt_path), so this is belt and braces, not a single
#     point of failure.
#   * /metrics and /docs|/openapi.json, added by w2v_common's create_app: a
#     Prometheus scrape and the schema carry no tenant and do no tenant work.
#
# DORMANT ON MERGE: require_license is a no-op unless
# W2V_LICENSE_GATE_ENABLED is true, read per-call from os.environ. It
# defaults to False, so this change alters no behaviour until gitops arms
# the flag. Preserve that default.
router = APIRouter(prefix="/api", tags=["analysis"], dependencies=list(LICENSE_GATE))

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(name: str) -> str:
    """Strip directory traversal and punctuation from a filename."""
    stem = Path(name).name
    sanitized = _FILENAME_SAFE.sub("_", stem)
    return sanitized or "upload.bin"


# There is NO unattributed / nil-tenant fallback. w2v-common 0.67.0 removed the
# nil-UUID substitution because that value is a *valid* UUID registered as
# `default-anon`, so it passed every type check and every dirty-row audit while
# silently mis-attributing cost. A request without a real registered tenant is
# refused with 401 instead of being written under a sentinel.


def _tenant_from_request(request: Request) -> str:
    state = getattr(request, "state", None)
    tenant = getattr(state, "tenant_id", None) if state is not None else None
    try:
        return str(require_tenant(tenant, source="request.state.tenant_id"))
    except TenantError as exc:
        raise HTTPException(status_code=401, detail="tenant identity required") from exc


def _job_to_status(job: AnalysisJob) -> JobStatusResponse:
    artifacts = job.artifacts if isinstance(job.artifacts, dict) else None
    verification_passed: bool | None = None
    if isinstance(job.verification_result, dict):
        verification_passed = bool(job.verification_result.get("passed"))
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error_message=job.error_message,
        artifacts=artifacts,
        duration_seconds=job.duration_seconds,
        width=job.width,
        height=job.height,
        has_audio=bool(job.has_audio) if job.has_audio is not None else None,
        vision_cost_usd=job.vision_cost_usd,
        verification_passed=verification_passed,
    )


def _job_to_result(job: AnalysisJob) -> JobResultResponse:
    base = _job_to_status(job).model_dump()
    verification: VerificationResult | None = None
    if isinstance(job.verification_result, dict):
        verification = VerificationResult.model_validate(job.verification_result)
    quality_metrics: QualityMetricsResponse | None = None
    if isinstance(job.quality_metrics, dict):
        quality_metrics = QualityMetricsResponse.model_validate(job.quality_metrics)
    return JobResultResponse(
        **base,
        transcript_data=job.transcript_data,
        scene_boundaries=job.scene_boundaries,
        frame_analyses=job.frame_analyses,
        # Derived, not stored: the pipeline checkpoints frame_analyses after every
        # vision batch, so any row that carries frames without having reached
        # 'completed' was cut off (still running, failed, cost-capped, or past the
        # wall-clock deadline). Deriving it needs no migration and cannot drift out
        # of sync with the status column.
        frame_analyses_partial=bool(job.frame_analyses) and job.status != "completed",
        quality_metrics=quality_metrics,
        verification=verification,
    )


def _upload_failure_reason(exc: BaseException) -> str:
    """Terminal ``error_message`` for a job whose upload aborted."""
    # Both flavours of "the client went away": an anyio-cancelled request scope
    # raises ``CancelledError``, while uvicorn's own detection of a half-sent
    # body raises ``starlette.requests.ClientDisconnect`` — a plain ``Exception``
    # that would otherwise fall through to the generic branch and produce a
    # misleading "upload failed: ClientDisconnect: " message with no detail.
    if isinstance(exc, asyncio.CancelledError | ClientDisconnect):
        return "upload aborted: client disconnected"
    if isinstance(exc, HTTPException):
        if exc.status_code == 413:
            return "upload exceeded MAX_UPLOAD_SIZE_BYTES"
        return f"upload rejected: {exc.detail}"
    if isinstance(exc, OSError):
        return f"failed to save upload: {exc}"
    return f"upload failed: {type(exc).__name__}: {exc}"


# Strong references to in-flight shielded terminalisation tasks. Without this the
# event loop only holds a weak reference and a task spawned as the request is
# torn down can be garbage-collected mid-await, which would strand exactly the
# row we are trying to terminalise.
_PENDING_ABORTS: set[asyncio.Task[None]] = set()

#: Strong references to in-flight ``VideoDownloader.aclose()`` tasks spawned by the
#: shielded cleanup in :func:`analyze_audio_beats`. Same purpose as
#: ``_PENDING_ABORTS``: without it the event loop keeps only a weak reference and
#: the close can be collected before it finishes.
_PENDING_CLOSES: set[asyncio.Task[None]] = set()


async def _mark_upload_job_failed(job_id: str, tenant_id: str, reason: str) -> None:
    """The DB leg of an upload abort. Runs shielded; never raises to the caller.

    Deliberately touches ONLY the DB row and never object storage: on the
    kickoff-failure path the upload was already stored durably, and re-downloading
    it is how a retry/resume works.
    """
    try:
        session_maker = await get_session_maker()
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).mark_failed(job_id, reason, tenant_id=tenant_id)
    except asyncio.CancelledError:
        logger.warning("Terminalisation of upload job %s was itself cancelled", job_id)
        raise
    except BaseException:
        logger.exception("Could not terminalise stranded upload job %s (%s)", job_id, reason)


async def _abort_upload_job(job_id: str, tenant_id: str, upload_dir: Path, reason: str) -> None:
    """Terminalise a stranded upload job AND reclaim its pod-local bytes.

    This service has **no reaper** (see migration 003), so a job left at
    ``pending`` by a failed upload stays pending forever and its partial file
    stays on the pod's disk. Runs inside an ``except`` block, therefore it must
    never mask the original exception.

    Ordering and shielding are both load-bearing:

    * ``rmtree`` is FIRST and synchronous — it cannot be interrupted, so the disk
      is reclaimed even if every ``await`` below dies.
    * the DB write runs in its OWN task, awaited through :func:`asyncio.shield`.
      Under a REAL client disconnect this ``except`` block executes inside an
      already-cancelled anyio scope where *every* ``await`` re-raises
      ``CancelledError`` immediately — a plain ``await`` here would never reach
      the DB, and ``except Exception`` could not even log it. The shielded task
      is not a child of that scope, so it completes regardless.
    * ``CancelledError`` is re-raised, never swallowed: the endpoint must still
      report as cancelled.
    """
    shutil.rmtree(upload_dir, ignore_errors=True)
    task = asyncio.ensure_future(_mark_upload_job_failed(job_id, tenant_id, reason))
    _PENDING_ABORTS.add(task)
    task.add_done_callback(_PENDING_ABORTS.discard)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.warning(
            "Upload job %s cancelled (%s); terminalisation continues in a shielded task",
            job_id,
            reason,
        )
        raise
    except BaseException:
        # ``_mark_upload_job_failed`` already logged; guard is only here so a
        # BaseException from the cleanup can never mask the original error.
        logger.exception("Cleanup of stranded upload job %s (%s) failed", job_id, reason)


async def _kickoff_pipeline(request: Request, job_id: str, tenant_id: str) -> None:
    """Dispatch pipeline execution via the app's pipeline runner if present."""
    runner = getattr(request.app.state, "pipeline_runner", None)
    if runner is None:
        logger.warning(
            "No pipeline_runner registered on app state; job %s will stay pending", job_id
        )
        return
    await runner.enqueue(job_id, tenant_id)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=202,
    operation_id="start_video_analysis",
)
async def create_analysis_job(
    request: Request,
    body: AnalysisRequest,
) -> AnalysisResponse:
    """Create a new analysis job from a JSON body (video_url).

    Local filesystem inputs are not accepted here — ``video_path`` was removed
    (arbitrary pod-file read); use this endpoint with ``video_url`` or
    ``POST /analyze/upload`` for a local file.

    The "must carry a source" rule now lives on ``AnalysisRequest`` itself
    (``_require_a_source``) rather than in this handler, so no other caller of
    the model can build a sourceless request. Consequence: a body with no
    ``video_url`` is rejected during body validation and returns **422**, the
    FastAPI-standard code, where this handler used to hand-roll a 400.
    """
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        repo = JobRepository(session)
        job = await repo.create(
            tenant_id=tenant_id,
            video_path=None,
            video_url=body.video_url,
            upload_filename=None,
            request_config=body.model_dump(mode="json"),
            verification_criteria=(
                body.verification.model_dump(mode="json") if body.verification else None
            ),
            dispatch_identity=capture_dispatch_identity(),
        )
    await _kickoff_pipeline(request, job.job_id, tenant_id)
    return AnalysisResponse(
        job_id=job.job_id, status=job.status, stage=job.stage, progress=job.progress
    )


@router.post(
    "/analyze/upload",
    response_model=AnalysisResponse,
    status_code=202,
    operation_id="upload_video_for_analysis",
)
async def upload_and_analyze(
    request: Request,
    file: Annotated[UploadFile, File(..., description="Video file to analyze")],
    config: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    """Accept a multipart upload + optional JSON config and schedule analysis."""
    settings = request.app.state.settings

    if config:
        try:
            payload: dict[str, Any] = json.loads(config)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc
    else:
        payload = {}

    try:
        # The uploaded file IS the source, so this is the one sanctioned
        # sourceless construction (see AnalysisRequest._require_a_source).
        request_body = AnalysisRequest.for_upload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc

    # A ``video_path`` in the config is already rejected by AnalysisRequest's
    # validator above (surfaced as the 400 "Invalid config" below).
    if request_body.video_url:
        raise HTTPException(
            status_code=400,
            detail="upload endpoint cannot also set video_url",
        )

    filename = _sanitize_filename(file.filename or "upload.bin")
    tenant_id = _tenant_from_request(request)

    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        repo = JobRepository(session)
        job = await repo.create(
            tenant_id=tenant_id,
            video_path=None,
            video_url=None,
            upload_filename=filename,
            request_config=request_body.model_dump(mode="json"),
            verification_criteria=(
                request_body.verification.model_dump(mode="json")
                if request_body.verification
                else None
            ),
            dispatch_identity=capture_dispatch_identity(),
        )

    # Stream the upload to a temp file on THIS pod (enforcing the size cap),
    # then stream it up into the shared object store under a tenant-scoped key.
    # The source video lives in object storage — NOT pod-local disk — so the
    # async analyze work can run on a DIFFERENT replica than the one that
    # accepted the upload (that pod's disk is unreachable there). Before this,
    # an upload accepted on pod A and analyzed on pod B failed to resolve its
    # source. See StorageService.store_upload / PipelineRunner._resolve_source.
    upload_dir = Path(settings.storage.upload_dir) / job.job_id
    tmp_path = upload_dir / filename

    max_bytes = settings.storage.max_upload_size_bytes
    total = 0
    # ONE guarded block covering EVERYTHING after the row exists: storage lookup
    # + mkdir + streaming write + store_upload + enqueue. The job row is already
    # INSERTed at this point, so EVERY exit other than success must terminalise
    # the row and reclaim the dir — including the two paths that used to sit
    # outside this block (``app.state.storage_service`` missing, and an
    # ``enqueue`` that raises because Redis is down or the queue is full, which
    # left the job ``pending`` forever with its bytes on disk: the exact defect
    # this guard exists to fix).
    #
    # ``BaseException`` (not ``Exception``, and certainly not the old
    # ``OSError``) because a client disconnect surfaces as
    # ``asyncio.CancelledError``. The original exception is always re-raised;
    # nothing is swallowed.
    try:
        storage = request.app.state.storage_service
        upload_dir.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large")
                fh.write(chunk)
        # Stream (never buffer) the temp file into the object store.
        await storage.store_upload(tenant_id, job.job_id, tmp_path, filename)

        # Success: drop the local copy — its durable home is now the object
        # store, keyed by tenant. ``upload_dir`` itself stays on the happy path:
        # the pipeline reuses it as the download target when it resolves the
        # source back out of the object store
        # (``PipelineRunner._resolve_source`` → ``upload_dir / job_id``).
        tmp_path.unlink(missing_ok=True)

        await _kickoff_pipeline(request, job.job_id, tenant_id)
    except BaseException as exc:
        # ``_abort_upload_job`` removes ``upload_dir`` on EVERY failure path,
        # including a post-``store_upload`` enqueue failure. That is safe and
        # deliberate: ``upload_dir`` is a pod-local scratch/download target that
        # ``PipelineRunner._resolve_source`` recreates by re-downloading from the
        # object store. The durable object-store copy is NOT touched here (only
        # ``DELETE /api/analyze/{job_id}`` calls ``storage.delete_upload``), so a
        # retry or a manual re-enqueue still finds its bytes.
        await _abort_upload_job(job.job_id, tenant_id, upload_dir, _upload_failure_reason(exc))
        if isinstance(exc, OSError):
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
        raise

    return AnalysisResponse(job_id=job.job_id, status="pending", stage="queued", progress=0.0)


# Byte cap for the audio fetch. Independent of the 2 GiB video cap: a music bed
# short enough to pass the duration guard cannot legitimately be larger than
# this even as uncompressed WAV, and the cap is what stops a "audio_url" that
# actually points at a multi-gigabyte video from filling the pod's disk before
# ffprobe ever gets a chance to reject it on duration.
_MAX_AUDIO_BYTES = 512 * 1024 * 1024
_AUDIO_DOWNLOAD_TIMEOUT_S = 120.0


@router.post(
    "/analyze/audio",
    response_model=AudioBeatGrid,
    operation_id="analyze_audio_beat_grid",
)
async def analyze_audio_beat_grid(
    request: Request,
    body: AudioBeatGridRequest,
) -> AudioBeatGrid:
    """Detect a beat grid (bpm, beats, downbeats, onsets) for a music track.

    SYNCHRONOUS — and deliberately unlike every sibling route on this router,
    which answers 202 with a ``job_id`` and makes the caller poll. The reason is
    the shape of the work, not consistency for its own sake:

    * it is seconds, not minutes. A warm process analyses a full track in
      ~0.3s (ffmpeg decode) + ~0.04s (librosa). The 202/poll pattern exists for
      the multi-minute vision+OCR+transcription pipeline; wrapping a sub-second
      computation in a job row, a queue hop, a poll loop and a DB round-trip
      would cost more latency than the analysis itself;
    * the caller is a *composer* — the cut-to-beat product needs the grid in
      hand before it can lay out a single edit, so it blocks either way. Making
      it poll only moves the blocking into a loop;
    * there is no artifact and nothing to resume. The result is a few hundred
      floats returned in the response body, so no job row, no object-store
      write, and nothing to strand if the pod dies mid-request.

    The duration guard is what keeps that choice safe: input longer than
    ``beat_detector.MAX_DURATION_S`` is refused (413) off the ffprobe header,
    before a single sample is decoded, so no request can pin a CPU indefinitely.
    """
    # Cheapest gate first: if the numba-backed pipeline never warmed, EVERY call
    # into librosa raises from inside numba's C layer, and the caller would get a
    # 500 out of a request that is perfectly well-formed. 503 is the honest code —
    # the dependency is unavailable, the request is not wrong, and retrying against
    # a healthy pod works. Default **False**: the attribute is set only once
    # ``_startup`` reached the warm-up, so an unset attribute means the numba path
    # was never warmed — indistinguishable, from in here, from a warm-up that
    # failed, and admitting the request lands the caller in the same 500 from
    # inside numba's C layer. An app that wants this endpoint must record success,
    # exactly as ``_startup`` does; "nobody told me" is not evidence of health.
    if not getattr(request.app.state, "beat_detector_ready", False):
        raise HTTPException(
            status_code=503,
            detail="beat detection unavailable: the audio analysis pipeline failed to initialise",
        )

    settings = request.app.state.settings
    # Derived exactly as the sibling routes derive it. Unlike them there is no
    # ``bind_session_tenant`` here because there is no session: this endpoint
    # writes no row and reads none, so there is no RLS-scoped statement to bind
    # a tenant onto. Binding a session we never use would be theatre.
    tenant_id = _tenant_from_request(request)

    # Reuses the SSRF-gated downloader wholesale: ``download()`` calls
    # ``url_security.validate_public_url`` and pins the connection to the
    # address that was judged — identical posture to ``video_url``, including
    # the per-redirect re-validation that closes the DNS-rebind window.
    downloader = VideoDownloader(
        max_bytes=_MAX_AUDIO_BYTES,
        timeout=_AUDIO_DOWNLOAD_TIMEOUT_S,
        allowed_hosts=frozenset(settings.allowed_url_hosts),
    )
    work_dir = Path(tempfile.mkdtemp(prefix="beatgrid-"))
    try:
        try:
            local_path = await downloader.download(video_url=body.audio_url, output_dir=work_dir)
        except ValueError as exc:
            # Everything the SSRF guard and the size cap raise: a blocked host, a
            # private/reserved address, too many redirects, oversize body. All of
            # them are the caller's input being unacceptable.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Parenthesised ONLY because PEP 758's bare `except A, B:` form is
        # illegal together with an `as` binding — the house style applies to the
        # unbound form (see video_prober's `except TypeError, ValueError:`).
        except (httpx.HTTPError, OSError) as exc:
            raise HTTPException(
                status_code=502, detail=f"could not fetch audio_url: {exc}"
            ) from exc

        try:
            grid = await beat_detector.analyze_file(local_path, meter=body.meter)
        except beat_detector.AudioTooLongError as exc:
            # 413 for the same reason the upload route uses it: the request is
            # well-formed, the payload is simply larger than we will process.
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except beat_detector.BeatGridError as exc:
            # Undecodable / no audio stream / no readable duration — the URL was
            # fetchable but is not analysable audio.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        # ORDER IS LOAD-BEARING, and it is the same rule ``_abort_upload_job``
        # states above. ``rmtree`` is synchronous and cannot be interrupted, so it
        # goes FIRST: under a real client disconnect this ``finally`` runs inside
        # an already-cancelled anyio scope, where the ``aclose()`` await re-raises
        # ``CancelledError`` immediately and every statement after it is skipped.
        # With ``aclose()`` first, that leaked up to ``_MAX_AUDIO_BYTES`` (512 MiB)
        # of downloaded audio into the pod's ephemeral storage on every cancelled
        # request — the eviction shape this fleet has already been bitten by.
        shutil.rmtree(work_dir, ignore_errors=True)
        # Shielded for the same reason the abort path shields its DB write: a bare
        # await here never reaches ``aclose()`` in a cancelled scope, so the httpx
        # connection pool would leak with it. The shielded task is not a child of
        # that scope and completes regardless; a strong reference is held until it
        # does, so it cannot be garbage-collected mid-close.
        _close = asyncio.ensure_future(downloader.aclose())
        _PENDING_CLOSES.add(_close)
        _close.add_done_callback(_PENDING_CLOSES.discard)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(_close)

    logger.info(
        "Beat grid for tenant %s: %.2f bpm, %d beats, confidence %.2f over %.1fs",
        tenant_id,
        grid.bpm,
        len(grid.beats),
        grid.bpm_confidence,
        grid.duration_s,
    )
    return AudioBeatGrid(
        duration_s=grid.duration_s,
        bpm=grid.bpm,
        bpm_confidence=grid.bpm_confidence,
        beats=grid.beats,
        downbeats=grid.downbeats,
        onsets=grid.onsets,
        meter=grid.meter,
        grid_source=grid.grid_source,
    )


def _job_or_404(job: AnalysisJob | None) -> AnalysisJob:
    """Return the job or raise 404 (a tenant-scoped repo miss is the same as a true miss)."""
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get(
    "/analyze/{job_id}",
    response_model=JobStatusResponse,
    operation_id="get_analysis_status",
)
async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
    """Poll the status/progress of an analysis job (tenant-scoped).

    Reaps the job first if it has been abandoned. A job whose owning pod died mid-run
    keeps its non-terminal status forever (no handler runs), so a poller waits on work
    nobody is doing — measured 2026-08-04: 23 such rows, oldest ~11 days. Reaping on the
    POLL is what makes it visible to the one caller who is actually asking; a background
    sweep would need to read across tenants, which RLS correctly forbids this role.
    """
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        repo = JobRepository(session)
        job = _job_or_404(await repo.get_by_job_id(job_id, tenant_id=tenant_id))
        pipeline_settings = get_settings().pipeline
        if await repo.reap_if_stranded(
            job,
            tenant_id=tenant_id,
            # Leave a lapsed job with attempts left to the background RE-CLAIM
            # (migration 007) instead of failing it here; 0 = re-claim is off.
            max_reclaims=(
                pipeline_settings.job_max_reclaims if pipeline_settings.sweeper_enabled else 0
            ),
        ):
            # ``mark_failed`` COMMITTED, and the tenant GUC is transaction-scoped
            # (``set_config(..., is_local=true)``), so the re-read below runs in a NEW
            # transaction. Without re-binding it is unbound, RLS hides the row and the
            # poller gets a 404 for its own job — live 2026-10-22, job 01a1642c, right
            # after a rollout. ``install_rls_tenant_rebind`` (asgi) heals this from the
            # identity ContextVar; binding explicitly keeps the route correct on its own.
            await bind_session_tenant(session, require_tenant(tenant_id))
            job = _job_or_404(await repo.get_by_job_id(job_id, tenant_id=tenant_id))
        return _job_to_status(job)


@router.get(
    "/analyze/{job_id}/result",
    response_model=JobResultResponse,
    operation_id="get_analysis_result",
)
async def get_job_result(request: Request, job_id: str) -> JobResultResponse:
    """Fetch the full result of a completed (or in-progress) job (tenant-scoped)."""
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        job = _job_or_404(await JobRepository(session).get_by_job_id(job_id, tenant_id=tenant_id))
        return _job_to_result(job)


@router.get(
    "/analyze/{job_id}/verification",
    response_model=VerificationResult,
    operation_id="get_analysis_verification",
)
async def get_verification_result(request: Request, job_id: str) -> VerificationResult:
    """Return the verification report for a job that ran with criteria (tenant-scoped)."""
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        job = _job_or_404(await JobRepository(session).get_by_job_id(job_id, tenant_id=tenant_id))
        if not isinstance(job.verification_result, dict):
            raise HTTPException(status_code=404, detail="job has no verification result")
        return VerificationResult.model_validate(job.verification_result)


@router.get(
    "/analyze/{job_id}/artifact/{name}",
    response_model=None,
    operation_id="download_analysis_artifact",
)
async def download_artifact(request: Request, job_id: str, name: str) -> FileResponse | Response:
    """Stream a single artifact file by logical name (tenant-scoped)."""
    sanitized = _sanitize_filename(name)
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        job = _job_or_404(await JobRepository(session).get_by_job_id(job_id, tenant_id=tenant_id))
        artifacts = job.artifacts if isinstance(job.artifacts, dict) else {}
        stored_path = artifacts.get(sanitized)
        if not stored_path:
            raise HTTPException(status_code=404, detail=f"artifact {name} not found")

    storage = request.app.state.storage_service
    local = storage.get_local_path(stored_path)
    if local is not None and local.exists():
        return FileResponse(local)
    data = await storage.retrieve(stored_path)
    if data is None:
        raise HTTPException(status_code=404, detail="artifact missing from storage")
    return Response(content=data)


@router.delete("/analyze/{job_id}", operation_id="delete_analysis_job")
async def delete_job(request: Request, job_id: str) -> dict[str, str]:
    """Delete a job, its DB row, and all artifacts (tenant-scoped).

    The repo's tenant-scoped ``delete()`` returns False for a wrong-tenant
    or missing row, which we map to 404 — same shape as a true miss.
    """
    tenant_id = _tenant_from_request(request)
    session_maker = await get_session_maker()
    async with session_maker() as session:
        await bind_session_tenant(session, require_tenant(tenant_id))
        deleted = await JobRepository(session).delete(job_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="job not found")

    storage = getattr(request.app.state, "storage_service", None)
    if storage is not None:
        try:
            await storage.delete_job_dir(job_id)
        except OSError as exc:
            logger.warning("Storage cleanup for %s failed: %s", job_id, exc)
        # Uploaded source videos live under a separate tenant-scoped prefix
        # (uploads/<tenant>/<job_id>/…), not the artifact job dir — clean both.
        try:
            await storage.delete_upload(tenant_id, job_id)
        except OSError as exc:
            logger.warning("Upload object cleanup for %s failed: %s", job_id, exc)

    upload_dir = Path(request.app.state.settings.storage.upload_dir) / job_id
    if upload_dir.exists():
        try:
            await asyncio.to_thread(lambda: _rmtree_safe(upload_dir))
        except OSError as exc:
            logger.warning("Upload dir cleanup for %s failed: %s", job_id, exc)
    return {"status": "deleted", "job_id": job_id}


def _rmtree_safe(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
