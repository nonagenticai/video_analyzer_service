"""Analysis pipeline orchestrator.

Implements the full stage graph as an async task driven by a background
worker. Stages are gated by ``AnalysisRequest`` flags, so callers can disable
OCR, vision, or quality without changing the pipeline itself.

Weighted progress: each enabled stage contributes equally to the 0→1
progression so callers always see forward motion.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from w2v_common.auth.identity_context import (
    get_attribution_subject_id_from_context,
    get_license_id_from_context,
)
from w2v_common.database.rls import bind_session_tenant

from src.tenant import require_tenant

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas import AnalysisRequest, VerificationCriteria
from src.config import Settings, get_settings
from src.database.session import get_session_maker
from src.domain.analysis import (
    ExtractedFrame,
    FrameAnalysis,
    FrameExtraction,
    QualityMetrics,
)
from src.repository.job_repository import TERMINAL_STATUSES, JobRepository
from src.services.audio_extractor import AudioExtractor
from src.services.frame_extractor import FrameExtractor
from src.services.ocr_service import OCRService
from src.services.protocols import (
    AudioExtractorProto,
    FrameExtractorProto,
    OCRServiceProto,
    QualityAnalyzerProto,
    SceneDetectorProto,
    TranscriptionClientProto,
    VideoDownloaderProto,
)
from src.services.quality_analyzer import QualityAnalyzer
from src.services.scene_detector import SceneDetector
from src.services.storage_service import StorageService
from src.services.transcription_client import (
    TranscriptionClient,
    TranscriptionClientError,
    is_empty_decode,
)
from src.services.verifier import extract_transcript_text, transcript_unavailable_marker
from src.services.video_downloader import VideoDownloader
from src.services.video_prober import VideoMetadata, probe_video
from src.services.vision.base import (
    FrameInput,
    FrameVisionResult,
    VisionProvider,
    VisionProviderError,
    VisionSynthesisResult,
)
from src.services.vision.fake_provider import FakeVisionProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineDependencies:
    """Swappable collaborators so tests can inject fakes.

    Most collaborators are typed by Protocol so production classes and test
    doubles satisfy the dataclass structurally — no subclassing required and
    no ``# type: ignore[arg-type]`` at construction sites. Fields with no
    Protocol (storage, vision_provider, probe, verifier, image_matcher,
    artifact_generator) remain ``Any`` because they are either covered by
    their own ABC (``VisionProvider``) or have shapes too varied to lock down
    here.
    """

    storage: Any
    settings: Settings
    frame_extractor: FrameExtractorProto
    audio_extractor: AudioExtractorProto
    video_downloader: VideoDownloaderProto
    scene_detector: SceneDetectorProto
    ocr_service: OCRServiceProto
    quality_analyzer: QualityAnalyzerProto
    transcription_client: TranscriptionClientProto
    vision_provider: Any
    probe: Any
    verifier: Any | None = None
    image_matcher: Any | None = None
    artifact_generator: Any | None = None


def _enabled_stages(request: AnalysisRequest) -> list[str]:
    """Return the ordered list of stage names that will execute."""
    stages: list[str] = ["probe", "frame_extract", "audio_extract"]
    if _needs_transcript(request):
        stages.append("transcribe")
    stages.append("scene_detection")
    if request.extract_frame_descriptions:
        stages.append("vision")
    if request.extract_ocr:
        stages.append("ocr")
    if request.compute_quality_metrics:
        stages.append("quality")
    # Synthesis is a METERED worker_management LLM call, and its only consumer is
    # the walkthrough artifact (``_generate_artifacts`` renders ``synthesis.walkthrough``
    # under ``generate_walkthrough``; ui_inventory/frame_index build from
    # ``frame_analyses``, not from synthesis). It used to run unconditionally while
    # every sibling optional stage was flag-gated, so an _ANALYZER_OVERRIDES submit
    # with the render flags False paid for one discarded LLM call per clip, folded
    # invisibly into ``vision_cost_usd``. Gate it like its siblings — same flag its
    # consumer reads, and the same flag ``_needs_transcript`` already keys on.
    if request.generate_walkthrough:
        stages.append("synthesis")
    if request.verification is not None:
        stages.append("verification")
    stages.append("artifact_gen")
    return stages


def _effective_vision_cap(request: AnalysisRequest, settings: Settings) -> float | None:
    """Resolve the vision spend cap actually enforced for this job.

    ``VISION_MAX_COST_USD`` (``settings.vision.max_cost_usd``) is the OPERATOR
    cap. It used to be defined and never read: every enforcement site gated on
    the per-request ``max_vision_cost_usd``, which defaults to ``None``, so an
    operator reading a ConfigMap that says ``5.0`` was protected by nothing and
    every caller that omitted the field (i.e. every orchestrator submit) ran
    uncapped.

    RULE — the operator cap is a CEILING, not a default a caller can raise. A
    per-request value may only LOWER it; ``min`` is the whole rule. A cap a
    caller can raise is not a cap, and the caller here is an untrusted request
    body. Absent per-request value ⇒ the operator cap applies on its own.

    ``max_cost_usd <= 0`` means "no operator ceiling" (explicitly disabled), in
    which case the per-request value stands alone — that is the only way to
    exceed the configured number, and it takes an operator to set it.
    """
    operator_cap = settings.vision.max_cost_usd
    ceiling = operator_cap if operator_cap > 0 else None
    requested = request.max_vision_cost_usd
    if ceiling is None:
        return requested
    if requested is None:
        return ceiling
    return min(requested, ceiling)


def _needs_transcript(request: AnalysisRequest) -> bool:
    """Whether the pipeline should transcribe audio."""
    if request.verification is not None:
        if request.verification.expected_transcript is not None:
            return True
        if request.verification.required_phrases:
            return True
        if request.verification.forbidden_phrases:
            return True
    return bool(request.generate_walkthrough)


class PipelineRunner:
    """Owns the background task loop and stage execution for jobs."""

    def __init__(
        self,
        *,
        storage: StorageService | None = None,
        dependencies: PipelineDependencies | None = None,
        metering_redis: aioredis.Redis | None = None,
        instance_id: str = "",
    ) -> None:
        settings = get_settings()
        self._metering_redis = metering_redis
        # Identity of THIS pod, minted once in ``asgi._startup``. Stamped onto every job
        # this runner claims, and required by the heartbeat's ``WHERE instance_id = :me``
        # so a runner that lost a claim stops renewing instead of resurrecting a lease it
        # no longer owns. Empty in unit tests that construct the runner directly, which
        # simply means those runs claim under an empty owner — harmless, and the tests
        # that care pass one explicitly.
        self._instance_id = instance_id
        self._lease_seconds = settings.pipeline.job_lease_seconds
        if dependencies is None:
            if storage is None:
                storage = StorageService()
            from src.services.artifact_generator import ArtifactGenerator
            from src.services.image_matcher import ImageMatcher
            from src.services.verifier import Verifier

            dependencies = PipelineDependencies(
                storage=storage,
                settings=settings,
                frame_extractor=FrameExtractor(),
                audio_extractor=AudioExtractor(),
                video_downloader=VideoDownloader(
                    max_bytes=settings.storage.max_upload_size_bytes,
                    allowed_hosts=frozenset(settings.allowed_url_hosts),
                ),
                scene_detector=SceneDetector(),
                ocr_service=OCRService(),
                quality_analyzer=QualityAnalyzer(),
                transcription_client=TranscriptionClient(
                    base_url=settings.transcription_client.url,
                    timeout=settings.transcription_client.timeout_seconds,
                    license_token=settings.downstream_license_token,
                ),
                vision_provider=_select_vision_provider(settings),
                probe=probe_video,
                verifier=Verifier(),
                image_matcher=ImageMatcher(),
                artifact_generator=ArtifactGenerator(),
            )
        self._deps = dependencies
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # job_id -> owning tenant, for the jobs in ``_tasks``. Graceful shutdown needs it
        # to give each lease back under that job's own tenant (RLS) — see ``drain``.
        self._task_tenants: dict[str, str] = {}
        # PIPELINE_MAX_CONCURRENT_JOBS, finally load-bearing. ``enqueue`` creates one
        # unbounded task per job; without this every accepted job piled onto ffmpeg /
        # tesseract / the vision provider at once, which is a large part of why a
        # 15-clip caller could poll for minutes and get nothing back.
        # Resolved from ``get_settings()`` (like ``_lease_seconds`` above) rather than
        # ``dependencies.settings``: it is a per-POD bound, not a per-job one, so it must
        # not vary with whatever settings object a caller happened to inject.
        self._job_slots = asyncio.Semaphore(max(1, settings.pipeline.max_concurrent_jobs))

    @property
    def dependencies(self) -> PipelineDependencies:
        return self._deps

    def active_job_ids(self) -> list[str]:
        """job_ids currently running on THIS pod.

        Passed to the sweeper so it can never touch a row this process still owns, even
        in the narrow window between the job row's INSERT and the claim inside
        ``_run_job``. The lease already covers the claimed case; this covers the gap.
        """
        return [jid for jid, task in self._tasks.items() if not task.done()]

    async def enqueue(self, job_id: str, tenant_id: str) -> None:
        """Start (or replace) a background task running the pipeline.

        ``tenant_id`` comes from the request handler that created the job —
        it is threaded through every pipeline session so RLS can bind
        ``app.current_tenant`` (no unchecked cross-tenant load).
        """
        existing = self._tasks.get(job_id)
        if existing and not existing.done():
            return
        # A plain ``asyncio.create_task(coro)`` snapshots the CURRENT contextvars
        # Context at creation time, so this detached pipeline task inherits the
        # caller's identity + cost-correlation ContextVars (tenant_id / license_id
        # / subject_id / correlation_id / stage) for free — no ``context=`` needed.
        # That inheritance is load-bearing: it is what makes the pipeline's
        # downstream WM/IR cost events attribute to this run's tenant with the
        # right correlation_id. Do NOT move this work to a thread/process pool or
        # ``run_in_executor``, and do NOT pass a blank ``context=`` — either would
        # drop the snapshot and emit cost events with a NULL correlation_id and
        # the wrong (unattributed) tenant. Regression-guarded in
        # tests/unit/test_pipeline_correlation.py.
        self._track(job_id, tenant_id, asyncio.create_task(self._run_job(job_id, tenant_id)))

    async def resume(
        self, job_id: str, tenant_id: str, identity: Mapping[str, object] | None
    ) -> None:
        """Re-run a job the sweeper RE-CLAIMED from a dead pod (migration 007).

        The row already carries THIS pod's fresh lease (the SECURITY DEFINER claim
        stamped it). The task re-binds the SUBMITTING request's identity from the
        persisted snapshot before running, because there is no request in scope here and
        the pipeline's metered vision/synthesis calls fail closed without provenance.

        Self-reclaim guard: if this pod is somehow still running the job, do not start a
        second copy — that would be pure double spend on vision calls.
        """
        existing = self._tasks.get(job_id)
        if existing and not existing.done():
            logger.warning("Re-claimed job %s is already running on this pod; skipping", job_id)
            return
        self._track(
            job_id,
            tenant_id,
            asyncio.create_task(self._run_reclaimed(job_id, tenant_id, identity)),
        )

    async def _run_reclaimed(
        self, job_id: str, tenant_id: str, identity: Mapping[str, object] | None
    ) -> None:
        # Runs in this task's OWN copy of the context, so binding here can never bleed
        # into the sweeper that spawned it (which claims rows across tenants).
        from src.services.job_identity import bind_dispatch_identity

        bind_dispatch_identity(tenant_id, identity)
        await self._run_job(job_id, tenant_id)

    def _track(self, job_id: str, tenant_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[job_id] = task
        self._task_tenants[job_id] = tenant_id

        # Drop the reference once the job finishes so a long-lived runner does not
        # accumulate one Task per job for the pod's lifetime. The identity check
        # guards against evicting a newer task that re-used the same job_id.
        def _discard(done: asyncio.Task[None], jid: str = job_id) -> None:
            if self._tasks.get(jid) is done:
                del self._tasks[jid]
                self._task_tenants.pop(jid, None)

        task.add_done_callback(_discard)

    async def drain(self) -> int:
        """Graceful shutdown: stop this pod's jobs and GIVE BACK their leases.

        On SIGTERM (a rollout) the jobs this pod is running cannot finish here. Without
        this their leases simply run out — up to ``job_lease_seconds`` (300s) of a caller
        polling a job nobody is running — before a live pod re-claims them. Releasing
        marks the lease lapsed NOW and ``lease_released`` so the re-claim does not spend
        an attempt on a drain that was not the job's fault.

        Cancel FIRST, release SECOND: releasing while the task still runs would let the
        sibling re-claim it and run two copies. Returns how many leases were released.
        """
        live = {jid: t for jid, t in self._tasks.items() if not t.done()}
        if not live:
            return 0
        # With re-claim OFF nothing would ever pick a released lease up, and the sweeper's
        # fail-on-lapse rule must keep applying — so give nothing back; the leases lapse.
        release = get_settings().pipeline.job_max_reclaims > 0
        tenants = {jid: self._task_tenants.get(jid, "") for jid in live}
        for task in live.values():
            task.cancel()
        await asyncio.gather(*live.values(), return_exceptions=True)

        if not release:
            return 0
        released = 0
        session_maker = await get_session_maker()
        for jid, tenant_id in tenants.items():
            if not tenant_id:
                continue
            try:
                async with session_maker() as session:
                    await bind_session_tenant(session, require_tenant(tenant_id))
                    if await JobRepository(session).release_lease(
                        jid, tenant_id=tenant_id, instance_id=self._instance_id
                    ):
                        released += 1
            except Exception:
                # Best-effort: an unreleased lease just lapses on its TTL and is
                # re-claimed then (spending one attempt). Never block shutdown on it.
                logger.exception("Could not release the lease on job %s at shutdown", jid)
        logger.info("Shutdown: released %d of %d in-flight job lease(s)", released, len(live))
        return released

    async def wait_for(self, job_id: str) -> None:
        """Testing helper to await a scheduled pipeline job."""
        task = self._tasks.get(job_id)
        if task is not None:
            await task

    async def _run_job(self, job_id: str, tenant_id: str) -> None:
        """Execute the pipeline end-to-end with progress updates.

        ``tenant_id`` is the owning tenant passed in from ``enqueue``; every
        repo write is gated on it and every session binds it for RLS, so the
        pipeline cannot read or cross-write a sibling tenant's job.
        """
        heartbeat: asyncio.Task[None] | None = None
        finishing = asyncio.Event()
        try:
            await self._claim(job_id, tenant_id)
            # The heartbeat is what turns "no write for an hour" into "this pod is gone".
            # It runs as its OWN task so the vision stage's minutes of silence between
            # frames still produce liveness. Cancelled in the ``finally`` below on every
            # exit path — success, failure, and cooperative shutdown.
            heartbeat = asyncio.create_task(
                self._heartbeat(
                    job_id, tenant_id, owner=asyncio.current_task(), finishing=finishing
                )
            )
            # The lease is already held and heartbeating, so a job that queues here
            # for a slot cannot be reaped as stranded while it waits. Queue time is
            # deliberately OUTSIDE the deadline below: waiting costs nothing, and a
            # deadline that ran while queued would fail jobs for being popular.
            async with self._job_slots:
                await self._run_execute_with_deadline(job_id, tenant_id)
        # A background job must always reach a terminal state. Stages hit I/O
        # (frame/audio extraction), HTTP (OCR/LLM), and DB (SQLAlchemyError)
        # failures — the latter does not subclass any of the OSError/RuntimeError
        # family, so a narrow tuple let a transient DB error escape the task and
        # wedge the job in a non-terminal status forever. Catch broadly and record
        # the failure on the job row instead. (CancelledError is BaseException, so
        # cooperative shutdown still propagates.)
        except Exception as exc:  # terminal-state guarantee for a background job
            logger.exception("Pipeline failed for job %s", job_id)
            # This write itself touches the DB (session_maker / tenant bind / the
            # UPDATE): a blip here must not raise a NEW exception that supersedes
            # ``exc`` and escapes this handler unguarded — that would replace the
            # real stage failure (already logged above) with e.g. a bare
            # ``ConnectionError`` on whatever awaits this task, AND skip the write,
            # leaving the job row stuck in its last non-terminal status. Nothing
            # reconciles that promptly: ``stranded_sweeper`` eventually fails it,
            # but only after ``stranded_seconds`` of silence and with a generic
            # staleness message, not ``str(exc)``.
            try:
                session_maker = await get_session_maker()
                async with session_maker() as session:
                    await bind_session_tenant(session, require_tenant(tenant_id))
                    await JobRepository(session).mark_failed(job_id, str(exc), tenant_id=tenant_id)
            except Exception:
                logger.exception(
                    "Could not persist failed status for job %s (original failure "
                    "logged above); relying on stranded_sweeper to reclaim it",
                    job_id,
                )
        finally:
            finishing.set()
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            # BOTH terminal paths (and cooperative shutdown) land here. Shielded so a
            # cancellation — the deadline below, or pod drain — cannot skip the reclaim
            # the moment it awaits.
            with contextlib.suppress(Exception):
                await asyncio.shield(asyncio.to_thread(self._reclaim_job_scratch, job_id))

    async def _run_execute_with_deadline(self, job_id: str, tenant_id: str) -> None:
        """Run ``_execute`` under a wall-clock ceiling.

        Before this, nothing bounded a job: the heartbeat renewed the lease forever, so
        a caller that gave up at 420s left the job running to completion — still paying
        for vision calls, still emitting a cost event nobody reads. On expiry the job is
        marked failed with a message that names the bound; whatever partial
        ``frame_analyses`` the vision stage persisted stays on the row, so the caller
        gets the frames it already paid for instead of ``null``.
        """
        deadline = self._deps.settings.pipeline.job_deadline_seconds
        if deadline <= 0:
            await self._execute(job_id, tenant_id)
            return
        timeout_cm = asyncio.timeout(deadline)
        try:
            async with timeout_cm:
                await self._execute(job_id, tenant_id)
        except TimeoutError:
            # A TimeoutError raised by a DOWNSTREAM call (httpx, ffmpeg) is
            # indistinguishable at this except site from our own expiry, so ask the
            # context manager which one it was and let the other propagate to the
            # generic handler with its real message.
            if not timeout_cm.expired():
                raise
            msg = (
                f"Analysis exceeded the {deadline}s wall-clock deadline and was "
                "cancelled; partial results (if any) are retained"
            )
            logger.warning("Job %s hit the wall-clock deadline (%ss)", job_id, deadline)
            session_maker = await get_session_maker()
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).mark_failed(job_id, msg, tenant_id=tenant_id)

    def _reclaim_job_scratch(self, job_id: str) -> None:
        """Delete this job's working data now that it has reached a terminal state.

        Reclaims the two things that actually fill the pod: the materialised source
        video under ``upload_dir/<job_id>`` (up to ``max_upload_size_bytes``, 4 GiB by
        default) and the up-to-``max_frames`` extracted JPEGs plus ``audio.wav`` under
        the job dir.

        Deliberately NOT reclaimed:
        * the job dir itself — on the local backend the rendered artifacts live there
          and ``GET /analyze/{job_id}/artifact/{name}`` serves them off disk;
        * the tenant-scoped uploaded-source OBJECT — that is the caller's own input,
          owned by ``DELETE /analyze/{job_id}`` (``storage.delete_upload``), not by the
          end of one run.

        Synchronous on purpose: it is called via ``asyncio.to_thread``. Best-effort —
        a failure to reclaim must never change a job's outcome.
        """
        upload_dir = Path(self._deps.settings.storage.upload_dir) / job_id
        shutil.rmtree(upload_dir, ignore_errors=True)
        try:
            job_dir = self._deps.storage.get_job_dir(job_id)
        except OSError:
            logger.warning("Could not resolve job dir for %s during reclaim", job_id)
            return
        shutil.rmtree(job_dir / "frames", ignore_errors=True)
        with contextlib.suppress(OSError):
            (job_dir / "audio.wav").unlink(missing_ok=True)

    async def _claim(self, job_id: str, tenant_id: str) -> None:
        """Stamp this pod's ownership + first lease on the job row. Best-effort.

        A failed claim (already terminal, or a DB blip) must not stop the analysis — the
        job simply falls back to the pre-005 wall-clock staleness rule, which is exactly
        the behaviour that shipped before this change.
        """
        try:
            session_maker = await get_session_maker()
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).claim(
                    job_id,
                    tenant_id=tenant_id,
                    instance_id=self._instance_id,
                    lease_seconds=self._lease_seconds,
                )
        except Exception:
            logger.exception("Could not claim job %s; continuing unclaimed", job_id)

    async def _heartbeat(
        self,
        job_id: str,
        tenant_id: str,
        *,
        owner: asyncio.Task[Any] | None = None,
        finishing: asyncio.Event | None = None,
    ) -> None:
        """Renew this pod's lease every ``lease//3`` until cancelled or ownership is lost.

        lease//3 means FIVE consecutive missed beats (not three) before the row looks
        abandoned at the default 300s lease — comfortably above one sweeper interval, so
        a healthy job is never reaped under itself.

        A renew that returns False means another writer owns the row now (or it reached a
        terminal state), so this stops rather than fighting for a lease it lost. Transient
        errors are logged and retried on the next tick: one DB blip must not silently end
        the heartbeat and let a live job be reaped.
        """
        interval = max(1.0, self._lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                session_maker = await get_session_maker()
                async with session_maker() as session:
                    await bind_session_tenant(session, require_tenant(tenant_id))
                    repo = JobRepository(session)
                    still_ours = await repo.renew_lease(
                        job_id,
                        tenant_id=tenant_id,
                        instance_id=self._instance_id,
                        lease_seconds=self._lease_seconds,
                    )
                    # Only when the claim moved to ANOTHER live owner — a sibling pod
                    # RE-CLAIMED it after our lease lapsed (migration 007) — is this copy
                    # unwanted. A row that merely went terminal (our own completion
                    # racing this beat, or a reap) must NOT cancel us: the task may be
                    # between ``mark_completed`` and its cost event.
                    stolen = False
                    if not still_ours:
                        await bind_session_tenant(session, require_tenant(tenant_id))
                        row = await repo.get_by_job_id(job_id, tenant_id=tenant_id)
                        live = row is not None and row.status not in TERMINAL_STATUSES
                        if live and row is not None and row.instance_id is None:
                            # Our best-effort ``_claim`` failed (DB blip) and the job ran
                            # unowned. Nobody else holds it: take the lease now instead of
                            # stopping, so it is protected and re-claimable from here on.
                            await bind_session_tenant(session, require_tenant(tenant_id))
                            if await repo.claim(
                                job_id,
                                tenant_id=tenant_id,
                                instance_id=self._instance_id,
                                lease_seconds=self._lease_seconds,
                            ):
                                continue
                        lease = row.lease_expires_at if row is not None else None
                        if lease is not None and lease.tzinfo is None:
                            lease = lease.replace(tzinfo=UTC)
                        stolen = (
                            live
                            and row is not None
                            and row.instance_id not in (None, self._instance_id)
                            and lease is not None
                            and lease > datetime.now(UTC)
                        )
                if not still_ours:
                    if stolen and owner is not None and not (finishing and finishing.is_set()):
                        # Two pods would otherwise run the same job and both pay for
                        # every vision call. ``finishing`` = already in its own
                        # terminal ``finally``; leave that alone.
                        logger.warning(
                            "Job %s was re-claimed by another pod; stopping this copy", job_id
                        )
                        owner.cancel()
                    else:
                        logger.warning("Lost the lease on job %s; stopping heartbeat", job_id)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Heartbeat renew failed for job %s; will retry", job_id)

    async def _load_job_request(
        self, job_id: str, session_maker: async_sessionmaker[AsyncSession], tenant_id: str
    ) -> tuple[AnalysisRequest, VerificationCriteria | None, str | None] | None:
        """Load the persisted request/criteria/upload_filename for *job_id*.

        Tenant-scoped. ``upload_filename`` is non-None only for multipart
        uploads and is what :meth:`_resolve_source` uses to fetch the source
        back out of the shared object store. Returns ``None`` when the job is
        missing for this tenant.
        """
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            job = await JobRepository(session).get_by_job_id(job_id, tenant_id=tenant_id)
            if job is None:
                logger.warning("Job %s not found for tenant; skipping", job_id)
                return None
            request_payload: dict[str, Any] = job.request_config or {}
            criteria_payload = job.verification_criteria
            upload_filename = job.upload_filename

        # An upload job's persisted config carries no source field — the source
        # is the uploaded object, named by ``upload_filename`` — so it must be
        # re-validated in the upload context or the model's at-least-one-source
        # invariant would fail every upload job at execution time.
        request = (
            AnalysisRequest.for_upload(request_payload)
            if upload_filename
            else AnalysisRequest.model_validate(request_payload)
        )
        criteria = (
            VerificationCriteria.model_validate(criteria_payload) if criteria_payload else None
        )
        return request, criteria, upload_filename

    async def _run_extraction_stages(
        self,
        job_id: str,
        request: AnalysisRequest,
        progress: Callable[[str, int], Awaitable[None]],
        session_maker: async_sessionmaker[AsyncSession],
        tenant_id: str,
        upload_filename: str | None,
    ) -> tuple[Path, VideoMetadata, FrameExtraction, Path | None, str | None]:
        """Probe / extract frames / extract audio. Reports progress at each stage.

        The last element is the audio-extraction FAILURE REASON, or ``None`` when
        no failure occurred. It exists because ``audio_path is None`` on its own
        conflates two states that must not be conflated downstream: a video that
        carries NO audio stream (a legitimately empty transcript) and a video whose
        audio stream could not be extracted (an infra measurement GAP). See
        ``_run_transcription_stage``.
        """
        video_path = await self._resolve_source(job_id, request, tenant_id, upload_filename)

        metadata = await self._deps.probe(video_path)
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id,
                tenant_id=tenant_id,
                duration_seconds=metadata.duration_seconds,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                frame_count=metadata.frame_count,
                has_audio=1 if metadata.has_audio else 0,
            )
        await progress("probe", 1)

        job_dir = self._deps.storage.get_job_dir(job_id)
        frames_dir = job_dir / "frames"
        extraction = await self._deps.frame_extractor.extract(
            video_path=video_path,
            output_dir=frames_dir,
            metadata=metadata,
            desired_interval=request.frame_interval_seconds
            or self._deps.settings.pipeline.frame_interval_seconds,
            max_frames=request.max_frames or self._deps.settings.pipeline.max_frames,
        )
        await progress("frame_extract", 2)

        audio_path: Path | None = None
        audio_error: str | None = None
        if metadata.has_audio:
            audio_path = job_dir / "audio.wav"
            try:
                await self._deps.audio_extractor.extract(
                    video_path=video_path, output_path=audio_path
                )
            except (OSError, RuntimeError) as exc:
                logger.warning("Audio extraction failed: %s", exc)
                audio_path = None
                audio_error = f"audio extraction failed: {exc}"
        await progress("audio_extract", 3)

        # The whole FrameExtraction, not just ``frames``: ``planned`` is the
        # denominator the quality means (and the downstream hard-fail gate that
        # votes on them) rest on, and it cannot be recovered from the surviving
        # frames once a sample has been dropped.
        return video_path, metadata, extraction, audio_path, audio_error

    async def _run_transcription_stage(
        self,
        job_id: str,
        request: AnalysisRequest,
        audio_path: Path | None,
        audio_error: str | None = None,
    ) -> dict[str, Any] | None:
        """Run optional transcription. Returns transcript dict, the
        infra-unavailable sentinel, or ``None``.

        ``audio_path is None`` is TWO different states and they must not collapse
        into the same return value:

        * the video carries no audio stream at all -> ``None``. A legitimately
          empty transcript; the verifier evaluates transcript criteria against it.
        * ffmpeg could not extract the audio (``audio_error`` set) -> the
          UNAVAILABLE sentinel, exactly as an ASR/infra failure below.

        A THIRD state hides inside a perfectly healthy HTTP 200: the worker
        answers ``no_speech_reason='empty_decode'``, meaning its decode produced
        nothing usable. That body is shape-identical to a clean transcription
        (every required field present, ``text=""``), so consuming it as-is is
        the same measurement gap as a failed extraction — and it bypassed the
        sentinel entirely, because the sentinel was built only for a
        ``TranscriptionClientError``. It is routed to the SAME sentinel below.
        ``no_speech_reason='silent_audio'`` is NOT routed there: that decode
        succeeded over a measurable duration, so the absence of a phrase is a
        real measurement and the criterion must be evaluated normally.

        WHY it matters: with ``None`` the verifier builds an empty haystack and
        evaluates the criteria anyway, so every ``forbidden_phrases`` criterion
        PASSES vacuously -- a video that does say the forbidden phrase verifies
        clean-green because its audio never got extracted. That is a measurement
        gap scoring identically to a real measurement. The sentinel marks those
        criteria INDETERMINATE instead, which is neither a pass nor a fail.
        """
        if audio_path is None:
            return transcript_unavailable_marker(audio_error) if audio_error else None
        try:
            payload = await self._deps.transcription_client.transcribe_file(
                audio_path=audio_path,
                language=request.language,
                word_timestamps=True,
            )
        except TranscriptionClientError as exc:
            # ASR/infra failure — return a sentinel distinct from ``None`` (no
            # audio) and ``""`` (silent video) so the verifier marks
            # transcript-dependent criteria INDETERMINATE rather than emitting a
            # false content-"failed" verdict for a transient outage.
            logger.warning("Transcription unavailable (infra): %s", exc)
            return transcript_unavailable_marker(str(exc))

        if is_empty_decode(payload):
            # HTTP 200, every field present — but the worker told us its decode
            # produced nothing usable, which is explicitly NOT evidence of
            # silence. Reuse the SAME sentinel so transcript criteria go
            # indeterminate rather than passing vacuously.
            logger.warning(
                "Transcription unusable for job %s: worker reported empty_decode "
                "(no evidence about the audio); marking transcript INDETERMINATE",
                job_id,
            )
            return transcript_unavailable_marker(
                "transcription worker reported no_speech_reason='empty_decode': "
                "the decode produced nothing usable"
            )
        return payload

    async def _run_synthesis_stage(
        self,
        frame_vision: list[FrameVisionResult],
        transcript_text: str | None,
        *,
        enabled: bool,
    ) -> VisionSynthesisResult | None:
        """Run the metered synthesis LLM call, or skip it when nothing consumes it.

        ``enabled`` mirrors ``"synthesis" in _enabled_stages(request)``. Skipping
        returns ``None``, which every downstream consumer already handles (the
        walkthrough template defaults it, and ``_fold_synthesis_cost_and_maybe_halt``
        no-ops on ``None``).
        """
        if not enabled:
            return None
        try:
            return await self._deps.vision_provider.synthesize(
                frame_results=frame_vision,
                transcript_text=transcript_text,
            )
        except VisionProviderError as exc:
            logger.warning("Synthesis failed: %s", exc)
            return None

    async def _synthesis_stage_with_progress(
        self,
        *,
        enabled: bool,
        frame_vision: list[FrameVisionResult],
        transcript_text: str | None,
        stage_index: int,
        progress: Callable[[str, int], Awaitable[None]],
    ) -> tuple[VisionSynthesisResult | None, int]:
        """Run synthesis and advance the progress counter — only when it is enabled.

        A skipped stage must not report progress under its own name, and must not
        consume an index: ``total_stages`` comes from ``_enabled_stages``, which no
        longer counts synthesis when ``generate_walkthrough`` is False.
        """
        synthesis = await self._run_synthesis_stage(frame_vision, transcript_text, enabled=enabled)
        if enabled:
            stage_index += 1
            await progress("synthesis", stage_index)
        return synthesis, stage_index

    async def _run_verification_stage(
        self,
        job_id: str,
        criteria: VerificationCriteria,
        metadata: VideoMetadata,
        transcript_data: dict[str, Any] | None,
        frame_analyses: list[Any],
        frames: list[ExtractedFrame],
        scene_boundaries: list[Any],
        quality: QualityMetrics | None,
        session_maker: async_sessionmaker[AsyncSession],
        tenant_id: str,
    ) -> dict[str, Any]:
        verifier = self._deps.verifier
        if verifier is None:
            raise RuntimeError("Verifier not configured")
        verification_payload = await verifier.verify(
            criteria=criteria,
            metadata=metadata,
            transcript_data=transcript_data,
            frame_analyses=frame_analyses,
            frames=frames,
            scene_boundaries=scene_boundaries,
            quality_metrics=quality,
            image_matcher=self._deps.image_matcher,
            # Reference images are read ONLY from this tenant's upload root. The
            # verifier resolves each `reference_key` under it and refuses
            # anything that lands outside, so verification cannot be aimed at
            # another tenant's uploads or at the pod filesystem.
            reference_root=Path(self._deps.settings.storage.upload_dir) / tenant_id,
        )
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id, tenant_id=tenant_id, verification_result=verification_payload
            )
        return verification_payload

    async def _fold_synthesis_cost_and_maybe_halt(
        self,
        job_id: str,
        tenant_id: str,
        synthesis: VisionSynthesisResult | None,
        vision_cost: float,
        max_vision_cost_usd: float | None,
    ) -> bool:
        """Fold synthesis LLM spend into ``vision_cost`` and re-persist it.

        Previously ``vision_cost`` counted only per-frame descriptions, so both
        the persisted metering total AND the cost cap under-counted total vision
        spend by the synthesis call. Returns ``True`` if the combined total now
        breaches ``max_vision_cost_usd`` (the caller halts before the remaining
        verification/artifact stages). The FakeVisionProvider reports 0.0
        synthesis cost, so this is a no-op there.
        """
        if synthesis is None or not synthesis.cost_usd:
            return False
        vision_cost += synthesis.cost_usd
        session_maker = await get_session_maker()
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id, tenant_id=tenant_id, vision_cost_usd=vision_cost
            )
        if max_vision_cost_usd is not None and vision_cost > max_vision_cost_usd:
            msg = f"Vision cost ${vision_cost:.2f} exceeds cap ${max_vision_cost_usd:.2f}"
            logger.warning("Vision stage halted after synthesis: %s", msg)
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).mark_failed(job_id, msg, tenant_id=tenant_id)
            return True
        return False

    async def _execute(self, job_id: str, tenant_id: str) -> None:
        started = time.monotonic()
        session_maker = await get_session_maker()
        loaded = await self._load_job_request(job_id, session_maker, tenant_id)
        if loaded is None:
            return
        request, criteria, upload_filename = loaded
        settings = self._deps.settings
        stages = _enabled_stages(request)
        total_stages = len(stages)

        async def _progress(name: str, index: int) -> None:
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).update_progress(
                    job_id, tenant_id=tenant_id, stage=name, progress=index / total_stages
                )

        await _progress(stages[0], 0)

        _, metadata, extraction, audio_path, audio_error = await self._run_extraction_stages(
            job_id, request, _progress, session_maker, tenant_id, upload_filename
        )
        frames = extraction.frames

        # --- Optional stage: transcription ---
        transcript_data: dict[str, Any] | None = None
        stage_index = 3
        if "transcribe" in stages:
            stage_index += 1
            transcript_data = await self._run_transcription_stage(
                job_id, request, audio_path, audio_error=audio_error
            )
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).update_fields(
                    job_id, tenant_id=tenant_id, transcript_data=transcript_data
                )
            await _progress("transcribe", stage_index)

        # --- Stage: scene detection ---
        # OpenCV histogram loop over up to ``max_frames`` images is CPU-bound and
        # synchronous; run it off the event loop so concurrent jobs + status polls
        # + health probes are not starved while it runs.
        scene_boundaries = await asyncio.to_thread(self._deps.scene_detector.detect, frames)
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id, tenant_id=tenant_id, scene_boundaries=scene_boundaries
            )
        stage_index += 1
        await _progress("scene_detection", stage_index)

        # --- Optional stage: vision LLM ---
        vision_outcome = await self._run_vision_stage(
            job_id, request, frames, session_maker, tenant_id
        )
        if vision_outcome is None:
            return  # pipeline halted on vision cost cap
        frame_vision, vision_cost, vision_ran = vision_outcome
        if vision_ran:
            stage_index += 1
            await _progress("vision", stage_index)

        # --- Optional stage: OCR ---
        # tesseract shells out once per frame (up to ``max_frames``); keep the
        # blocking subprocess loop off the event loop.
        ocr_text_by_index = await asyncio.to_thread(self._run_ocr_stage, request, frames)
        if request.extract_ocr:
            stage_index += 1
            await _progress("ocr", stage_index)

        frame_analyses = _build_frame_analyses(
            frames=frames,
            frame_vision=frame_vision,
            ocr_by_index=ocr_text_by_index,
        )
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            serialized = [_frame_analysis_to_dict(fa) for fa in frame_analyses]
            await JobRepository(session).update_fields(
                job_id,
                tenant_id=tenant_id,
                frame_analyses=serialized,
                vision_cost_usd=vision_cost,
            )

        # --- Optional stage: quality ---
        quality = await self._run_quality_stage(
            job_id, request, frames, session_maker, tenant_id, frames_planned=extraction.planned
        )
        if request.compute_quality_metrics:
            stage_index += 1
            await _progress("quality", stage_index)

        # --- Optional stage: LLM synthesis ---
        # transcript_text is computed unconditionally: the artifact stage renders it
        # into walkthrough.md independently of whether synthesis ran.
        transcript_text = extract_transcript_text(transcript_data)
        synthesis, stage_index = await self._synthesis_stage_with_progress(
            enabled="synthesis" in stages,
            frame_vision=frame_vision,
            transcript_text=transcript_text,
            stage_index=stage_index,
            progress=_progress,
        )

        # Fold synthesis spend into vision_cost + re-persist; halt if it now
        # breaches the cap (before verification/artifact stages).
        if await self._fold_synthesis_cost_and_maybe_halt(
            job_id, tenant_id, synthesis, vision_cost, _effective_vision_cap(request, settings)
        ):
            return

        # --- Optional stage: verification ---
        verification_payload: dict[str, Any] | None = None
        if "verification" in stages and criteria is not None and self._deps.verifier:
            stage_index += 1
            verification_payload = await self._run_verification_stage(
                job_id,
                criteria,
                metadata,
                transcript_data,
                frame_analyses,
                frames,
                scene_boundaries,
                quality,
                session_maker,
                tenant_id,
            )
            await _progress("verification", stage_index)

        # --- Stage: artifacts ---
        job_dir = self._deps.storage.get_job_dir(job_id)
        artifacts = await self._render_artifacts(
            job_id=job_id,
            job_dir=job_dir,
            metadata=metadata,
            request=request,
            frames=frames,
            frame_analyses=frame_analyses,
            scene_boundaries=scene_boundaries,
            transcript_text=transcript_text,
            synthesis=synthesis,
            verification_payload=verification_payload,
        )
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id, tenant_id=tenant_id, artifacts=artifacts
            )
        stage_index += 1
        await _progress("artifact_gen", stage_index)

        await self._complete_and_bill(
            job_id,
            tenant_id,
            session_maker=session_maker,
            duration_seconds=metadata.duration_seconds,
            started=started,
        )

    async def _complete_and_bill(
        self,
        job_id: str,
        tenant_id: str,
        *,
        session_maker: async_sessionmaker[AsyncSession],
        duration_seconds: float,
        started: float,
    ) -> None:
        """Stamp the terminal completion, and bill for it ONLY if this call won the write.

        Meters the local (non-LLM) analysis compute, priced per minute of source video.
        Vision-LLM is metered by worker_management and transcription by asr_backend, so
        this covers only frame/scene/OCR/quality work. Best-effort.

        GATED ON THE TERMINAL WRITE. ``mark_completed`` is guarded on non-terminal status
        and returns whether it moved the row. False means the job was ALREADY terminal —
        it was reaped as stranded and the client has very likely resubmitted and paid for
        a second analysis — so emitting a cost event here bills the same job_id TWICE.
        That was the live behaviour before this gate. Mirrors video_compiler_service's
        ``api/routes/compile.py`` (``won = await _stamp_completed(...); if won: _emit_cost``).
        """
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            won = await JobRepository(session).mark_completed(job_id, tenant_id=tenant_id)

        if not won:
            logger.warning("Job %s was already terminal at completion; skipping cost event", job_id)
            return
        self._emit_cost(
            tenant_id=tenant_id,
            job_id=job_id,
            duration_seconds=duration_seconds,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _emit_cost(
        self, *, tenant_id: str, job_id: str, duration_seconds: float, latency_ms: int
    ) -> None:
        """Fire-and-forget a per-minute cost event for this completed job.

        Identity (license/subject) is read from the request context the
        background task inherited via ``asyncio.create_task``; ``tenant_id`` is
        the authoritative owner threaded through the pipeline. No-op when
        metering is disabled (no Redis client).
        """
        if self._metering_redis is None:
            return
        from src.services.metering import schedule_cost_event

        seconds = max(0, round(duration_seconds or 0.0))
        cost_hint = seconds / 60.0 * 0.017  # advisory only; consumer recomputes
        # GC-safe fire-and-forget; the shared helper owns the task lifecycle.
        schedule_cost_event(
            self._metering_redis,
            tenant_id=tenant_id,
            job_id=job_id,
            license_id=get_license_id_from_context(),
            subject_id=get_attribution_subject_id_from_context(),
            input_tokens=seconds,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=cost_hint,
            usage_extra={"unit": "video_seconds"},
        )

    async def _run_vision_stage(
        self,
        job_id: str,
        request: AnalysisRequest,
        frames: list[ExtractedFrame],
        session_maker: async_sessionmaker[AsyncSession],
        tenant_id: str,
    ) -> tuple[list[FrameVisionResult], float, bool] | None:
        """Run the optional vision stage, in ``VISION_BATCH_SIZE`` chunks.

        Returns ``(frame_vision, vision_cost, ran)`` on success, or ``None`` if
        the cost cap was exceeded (pipeline must halt; the helper has already
        marked the job failed).

        Two things changed here after the 2026-08-21 ``footage_critique`` run that
        polled for 3m09s and got ``frame_analyses: null``:

        1. The provider is called in batches of ``settings.vision.batch_size``
           instead of once with every frame. That knob existed in ``config.py``
           and was referenced nowhere in ``src/``.
        2. After EACH batch the frames scored so far are persisted to the job row.
           Previously ``frame_analyses`` was written exactly once, after both the
           vision AND OCR passes finished, so a caller that timed out — or a job
           that died — mid-vision saw ``null`` even though N frames had been
           scored and BILLED. The final write at the end of ``_execute`` still
           overwrites this with the full set once OCR has merged in.
        """
        if not request.extract_frame_descriptions:
            return [], 0.0, False
        # The ENFORCED cap: the operator ceiling (VISION_MAX_COST_USD) bounding
        # the optional per-request value. See ``_effective_vision_cap``.
        cap = _effective_vision_cap(request, self._deps.settings)
        try:
            frame_inputs = [
                FrameInput(index=f.index, timestamp=f.timestamp, path=f.path) for f in frames
            ]

            # PRE-FLIGHT: refuse an over-cap job BEFORE paying for it. The cap
            # used to be checked only after analyze_frames had already billed
            # every frame, so a job that could never satisfy its own cap still
            # spent the full amount and then failed — the tenant was charged for
            # a result they never received. Per-frame cost is a fixed provider
            # value, so the total is knowable up front.
            #
            # Providers that cannot predict their spend return 0.0 and are
            # unaffected (see VisionProvider.estimated_cost_for).
            estimated = self._deps.vision_provider.estimated_cost_for(len(frame_inputs))
            if cap is not None and estimated > cap:
                raise VisionProviderError(
                    f"Vision cost ${estimated:.2f} exceeds cap "
                    f"${cap:.2f} "
                    f"(estimated for {len(frame_inputs)} frames; no frames were billed)"
                )

            batch_size = max(1, self._deps.settings.vision.batch_size)
            frame_vision: list[FrameVisionResult] = []
            for start in range(0, len(frame_inputs), batch_size):
                batch = frame_inputs[start : start + batch_size]
                frame_vision.extend(await self._deps.vision_provider.analyze_frames(batch))
                # Persist what we have BEFORE the next (possibly fatal, possibly
                # deadline-cancelled) batch. OCR has not run yet, hence the empty
                # ocr map — the end-of-_execute write merges it in.
                await self._persist_partial_frame_analyses(
                    job_id,
                    tenant_id,
                    session_maker,
                    frames=frames,
                    frame_vision=frame_vision,
                    vision_cost=sum(r.cost_usd for r in frame_vision),
                )
            vision_cost = sum(r.cost_usd for r in frame_vision)
            # Post-hoc check RETAINED, not replaced: actual spend can exceed the
            # estimate (retries, the synthesis fold in _fold_synthesis_cost), and
            # a provider reporting 0.0 up front is caught only here.
            if cap is not None and vision_cost > cap:
                raise VisionProviderError(f"Vision cost ${vision_cost:.2f} exceeds cap ${cap:.2f}")
        except VisionProviderError as exc:
            logger.warning("Vision stage halted: %s", exc)
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).mark_failed(job_id, str(exc), tenant_id=tenant_id)
            return None
        return frame_vision, vision_cost, True

    async def _persist_partial_frame_analyses(
        self,
        job_id: str,
        tenant_id: str,
        session_maker: async_sessionmaker[AsyncSession],
        *,
        frames: list[ExtractedFrame],
        frame_vision: list[FrameVisionResult],
        vision_cost: float,
    ) -> None:
        """Write the frames scored SO FAR onto the job row.

        Best-effort: a checkpoint write that fails must not abort a run that is
        otherwise fine. The authoritative write still happens at the end of
        ``_execute``; this only ensures a caller who is cut off mid-vision can read
        the work it already paid for instead of ``null``.

        Rows written here are partial by construction — every job that is not
        ``completed`` and has a non-null ``frame_analyses`` was cut off, which is
        exactly what ``JobResultResponse.frame_analyses_partial`` reports.
        """
        try:
            serialized = [
                _frame_analysis_to_dict(fa)
                for fa in _build_frame_analyses(
                    frames=frames, frame_vision=frame_vision, ocr_by_index={}
                )
            ]
            async with session_maker() as session:
                await bind_session_tenant(session, require_tenant(tenant_id))
                await JobRepository(session).update_fields(
                    job_id,
                    tenant_id=tenant_id,
                    frame_analyses=serialized,
                    vision_cost_usd=vision_cost,
                )
        except Exception:  # noqa: BLE001 — a checkpoint must never fail the run
            logger.warning("Partial frame_analyses checkpoint failed for job %s", job_id)

    def _run_ocr_stage(
        self,
        request: AnalysisRequest,
        frames: list[ExtractedFrame],
    ) -> dict[int, str]:
        """Extract OCR text per frame when enabled. Returns mapping by frame index."""
        if not request.extract_ocr:
            return {}
        ocr_text_by_index: dict[int, str] = {}
        for frame in frames:
            text = self._deps.ocr_service.extract_text(frame.path)
            if text:
                ocr_text_by_index[frame.index] = text
        return ocr_text_by_index

    async def _run_quality_stage(
        self,
        job_id: str,
        request: AnalysisRequest,
        frames: list[ExtractedFrame],
        session_maker: async_sessionmaker[AsyncSession],
        tenant_id: str,
        *,
        frames_planned: int,
    ) -> QualityMetrics | None:
        """Compute quality metrics and persist them when enabled.

        ``frames_planned`` comes from ``FrameExtraction.planned``, NOT from
        ``len(frames)`` — the gap between the two is exactly what
        ``verifier.check_quality`` needs to refuse a verdict built on the
        survivors of a botched extraction.
        """
        if not request.compute_quality_metrics:
            return None
        # OpenCV sharpness/brightness/k-means palette over up to ``max_frames``
        # images is CPU-bound and synchronous; run it off the event loop.
        quality = await asyncio.to_thread(
            partial(self._deps.quality_analyzer.analyze, frames, frames_planned=frames_planned)
        )
        async with session_maker() as session:
            await bind_session_tenant(session, require_tenant(tenant_id))
            await JobRepository(session).update_fields(
                job_id, tenant_id=tenant_id, quality_metrics=asdict(quality)
            )
        return quality

    async def _resolve_source(
        self,
        job_id: str,
        request: AnalysisRequest,
        tenant_id: str,
        upload_filename: str | None,
    ) -> Path:
        """Return a local Path to the input video (materialised on THIS pod).

        There is deliberately NO caller-supplied local-path branch: every source
        is either downloaded through the SSRF-gated downloader or fetched from
        the tenant-scoped upload key. See ``AnalysisRequest`` for why
        ``video_path`` was removed.
        """
        if request.video_url:
            output_dir = Path(self._deps.settings.storage.upload_dir) / job_id
            return await self._deps.video_downloader.download(
                video_url=request.video_url, output_dir=output_dir
            )
        # Multipart upload: /analyze/upload streamed the source into the shared
        # object store under a TENANT-SCOPED key (video_url unset). Resolve it
        # from the object store — never from pod-local disk —
        # so a >1-replica / HPA deployment can run analyze on a different pod
        # than the one that accepted the upload. On the local backend the object
        # already sits on this pod's disk; on minio we presign it and stream it
        # down via the SSRF-gated downloader (#64 allowlists the internal store
        # host), which writes to disk chunk-by-chunk and never buffers the whole
        # video in memory.
        if upload_filename:
            storage = self._deps.storage
            local = storage.local_upload_path(tenant_id, job_id, upload_filename)
            if local is not None:
                return local
            url = await storage.presign_upload(tenant_id, job_id, upload_filename)
            if url:
                output_dir = Path(self._deps.settings.storage.upload_dir) / job_id
                return await self._deps.video_downloader.download(
                    video_url=url, output_dir=output_dir
                )
        raise ValueError("Neither video_url nor an uploaded file provided")

    async def _render_artifacts(
        self,
        *,
        job_id: str,
        job_dir: Path,
        metadata: VideoMetadata,
        request: AnalysisRequest,
        frames: list[ExtractedFrame],
        frame_analyses: list[FrameAnalysis],
        scene_boundaries: list[float],
        transcript_text: str | None,
        synthesis: VisionSynthesisResult | None,
        verification_payload: dict[str, Any] | None,
    ) -> dict[str, str]:
        """Render requested artifacts and store them via the storage service."""
        if self._deps.artifact_generator is None:
            return {}

        artifacts: dict[str, str] = {}
        gen = self._deps.artifact_generator

        scene_ctx = [
            {"start": start, "end": end, "description": ""}
            for start, end in _scene_bounds(scene_boundaries, total=metadata.duration_seconds)
        ]

        if request.generate_walkthrough:
            path = job_dir / "walkthrough.md"
            gen.write(
                template_name="walkthrough.md.j2",
                context={
                    "title": job_id,
                    "duration_seconds": metadata.duration_seconds,
                    "width": metadata.width,
                    "height": metadata.height,
                    "synthesis": synthesis.walkthrough if synthesis else None,
                    "scenes": scene_ctx,
                    "transcript": transcript_text,
                },
                output_path=path,
            )
            stored = await self._deps.storage.store_file(job_id, path, "walkthrough.md")
            artifacts["walkthrough.md"] = stored

        if request.generate_ui_inventory:
            path = job_dir / "ui_inventory.md"
            gen.write(
                template_name="ui_inventory.md.j2",
                context={
                    "title": job_id,
                    "ui_elements": [
                        {
                            "label": f"Frame {fa.index}",
                            "first_seen": fa.timestamp,
                            "ocr_text": fa.ocr_text,
                            "description": fa.description,
                        }
                        for fa in frame_analyses
                    ],
                },
                output_path=path,
            )
            stored = await self._deps.storage.store_file(job_id, path, "ui_inventory.md")
            artifacts["ui_inventory.md"] = stored

        if request.generate_frame_index:
            path = job_dir / "frame_index.md"
            gen.write(
                template_name="frame_index.md.j2",
                context={
                    "title": job_id,
                    "frames": [
                        {
                            "index": fa.index,
                            "timestamp": fa.timestamp,
                            "description": fa.description,
                            "ocr_text": fa.ocr_text,
                        }
                        for fa in frame_analyses
                    ],
                },
                output_path=path,
            )
            stored = await self._deps.storage.store_file(job_id, path, "frame_index.md")
            artifacts["frame_index.md"] = stored

        if verification_payload is not None:
            json_path = job_dir / "verification.json"
            import json as _json

            json_path.write_text(_json.dumps(verification_payload, indent=2))
            stored = await self._deps.storage.store_file(job_id, json_path, "verification.json")
            artifacts["verification.json"] = stored

            md_path = job_dir / "verification.md"
            gen.write(
                template_name="verification.md.j2",
                context={"result": verification_payload},
                output_path=md_path,
            )
            stored_md = await self._deps.storage.store_file(job_id, md_path, "verification.md")
            artifacts["verification.md"] = stored_md

        return artifacts


def _select_vision_provider(settings: Settings) -> VisionProvider:
    """Build the configured vision provider (or a fake when no API key is set).

    Resolution order:
      1. ``ANALYZER_PROVIDER`` env (settings.analyzer_provider) — top-level override.
      2. ``VISION_DEFAULT_PROVIDER`` env (settings.vision.default_provider).

    The values ``"stub"`` and ``"fake"`` are aliases and both select
    :class:`FakeVisionProvider` — a deterministic offline provider that returns
    canned-but-valid analysis JSON. This is the supported escape hatch for
    offline / CI environments where no real vision API key is available.
    """
    override = (settings.analyzer_provider or "").strip().lower()
    provider_name = override or settings.vision.default_provider.lower()
    if provider_name in {"stub", "fake"}:
        return FakeVisionProvider()
    # All real vision flows route through worker_management. Per memory
    # `wm-as-sole-llm-gateway.md` w2v services must never call provider
    # SDKs (openai/anthropic/google) directly. The WM worker config
    # selects the underlying vision-capable model; the service-side knob
    # is just ``model_override`` for per-job overrides.
    if provider_name == "wm":
        try:
            from src.services.vision.wm_provider import WMVisionProvider

            return WMVisionProvider(
                model_override=settings.vision.model_override or None,
            )
        except VisionProviderError as exc:
            logger.warning("WM vision provider init failed (%s); using fake", exc)
    else:
        logger.warning(
            "Unknown vision provider %r; using fake. Set VISION_DEFAULT_PROVIDER "
            "to 'wm' (production) or 'stub' (offline tests).",
            provider_name,
        )
    return FakeVisionProvider()


def _frame_analysis_to_dict(fa: FrameAnalysis) -> dict[str, Any]:
    """Serialize a slotted FrameAnalysis to a JSON-safe dict."""
    return {
        "index": fa.index,
        "timestamp": fa.timestamp,
        "path": str(fa.path),
        "description": fa.description,
        "ocr_text": fa.ocr_text,
        "labels": list(fa.labels),
    }


def _build_frame_analyses(
    *,
    frames: list[ExtractedFrame],
    frame_vision: list[FrameVisionResult],
    ocr_by_index: dict[int, str],
) -> list[FrameAnalysis]:
    """Merge vision output and OCR output by frame index."""
    vision_by_index = {r.index: r for r in frame_vision}
    analyses: list[FrameAnalysis] = []
    for frame in frames:
        vision = vision_by_index.get(frame.index)
        analyses.append(
            FrameAnalysis(
                index=frame.index,
                timestamp=frame.timestamp,
                path=str(frame.path),
                description=vision.description if vision else None,
                ocr_text=ocr_by_index.get(frame.index),
                labels=list(vision.labels) if vision else [],
            )
        )
    return analyses


def _scene_bounds(boundaries: list[float], *, total: float) -> list[tuple[float, float]]:
    """Convert a list of boundary timestamps into (start, end) pairs."""
    if not boundaries:
        return []
    sorted_b = sorted(set(boundaries))
    pairs: list[tuple[float, float]] = []
    for idx, start in enumerate(sorted_b):
        end = sorted_b[idx + 1] if idx + 1 < len(sorted_b) else total
        pairs.append((start, end))
    return pairs
