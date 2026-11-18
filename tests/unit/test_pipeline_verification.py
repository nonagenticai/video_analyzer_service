"""Pipeline tests for the verification stage and additional branches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.api.schemas import AnalysisRequest, ReferenceImage, VerificationCriteria
from src.config import Settings, get_settings
from src.domain.analysis import ExtractedFrame, FrameExtraction, QualityMetrics, VideoMetadata
from src.repository.job_repository import JobRepository
from src.services.artifact_generator import ArtifactGenerator
from src.services.image_matcher import ImageMatcher
from src.services.pipeline import (
    PipelineDependencies,
    PipelineRunner,
    _select_vision_provider,
)
from src.services.storage_service import StorageService
from src.services.verifier import Verifier
from src.services.vision.base import (
    FrameInput,
    FrameVisionResult,
    VisionProvider,
    VisionProviderError,
    VisionSynthesisResult,
)
from src.services.vision.fake_provider import FakeVisionProvider
from tests.conftest import TENANT_A


class _FakeStorage(StorageService):
    def __init__(self, root: Path) -> None:
        from w2v_common.storage import StorageService as CommonStorage

        from src.config import StorageSettings

        self._backend_type = "local"
        self._local_path = root
        self._scratch_root = root
        self._storage = CommonStorage(StorageSettings(backend="local", local_path=str(root)))


class _FakeFrameExtractor:
    async def extract(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        metadata: VideoMetadata,
        desired_interval: float,
        max_frames: int,
    ) -> FrameExtraction:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for i in range(2):
            p = output_dir / f"frame_{i:04d}.png"
            Image.new("RGB", (32, 32), color=(50 + i, 100, 150)).save(p, "PNG")
            frames.append(ExtractedFrame(index=i, timestamp=float(i * 3), path=p))
        return FrameExtraction(frames=frames, planned=len(frames))


class _FakeAudioExtractor:
    async def extract(self, *, video_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF")
        return output_path


class _FakeDownloader:
    async def download(self, *, video_url: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "downloaded.mp4"
        path.write_bytes(b"x")
        return path


class _FakeSceneDetector:
    def detect(self, frames: list[ExtractedFrame]) -> list[float]:
        return [f.timestamp for f in frames]


class _FakeOCR:
    def extract_text(self, image_path: Path) -> str:
        return "Login Page"


class _FakeQualityAnalyzer:
    def analyze(
        self, frames: list[ExtractedFrame], *, frames_planned: int | None = None
    ) -> QualityMetrics:
        return QualityMetrics(
            mean_sharpness=500.0,
            mean_brightness=0.5,
            motion_score=0.05,
            dominant_colors=["#ff0000"],
            overall_score=0.85,
        )


class _FakeTranscriptionClient:
    async def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool = True,
        return_srt: bool = False,
    ) -> dict[str, Any]:
        return {
            "backend": "fake",
            "model": "fake",
            "language": language or "en",
            "duration": 6.0,
            "text": "Welcome to the demo of the new feature.",
            "segments": [],
        }


class _RaisingTranscriptionClient(_FakeTranscriptionClient):
    async def transcribe_file(self, **kwargs: Any) -> dict[str, Any]:
        from src.services.transcription_client import TranscriptionClientError

        raise TranscriptionClientError("worker down")


class _SynthesisFailingVision(VisionProvider):
    name = "fail-synth"

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        return await FakeVisionProvider().analyze_frames(frames)

    async def synthesize(
        self,
        *,
        frame_results: list[FrameVisionResult],
        transcript_text: str | None,
    ) -> VisionSynthesisResult:
        raise VisionProviderError("synthesis exploded")


async def _fake_probe(video_path: Path) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=6.0,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=180,
        has_audio=True,
    )


def _build_deps(
    *,
    storage_root: Path,
    transcription_client: Any | None = None,
    vision_provider: VisionProvider | None = None,
    settings: Settings | None = None,
) -> PipelineDependencies:
    if settings is None:
        # Never let a test write into the container default /app/data/uploads:
        # every source is now materialised under upload_dir.
        settings = get_settings().model_copy(deep=True)
        settings.storage.upload_dir = str(storage_root.parent / "uploads")
    return PipelineDependencies(
        storage=_FakeStorage(storage_root),
        settings=settings,
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=transcription_client or _FakeTranscriptionClient(),
        vision_provider=vision_provider or FakeVisionProvider(),
        probe=_fake_probe,
        verifier=Verifier(),
        image_matcher=ImageMatcher(),
        artifact_generator=ArtifactGenerator(),
    )


async def _insert_job(db_session_factory, *, video_path: Path, request: AnalysisRequest) -> str:
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_A,
            video_path=str(video_path),
            video_url=None,
            upload_filename=None,
            request_config=request.model_dump(mode="json"),
            verification_criteria=(
                request.verification.model_dump(mode="json") if request.verification else None
            ),
        )
        return job.job_id


async def test_pipeline_verification_passes(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    # Reference image must match a frame from the FakeFrameExtractor. It lives
    # under THIS TENANT's upload root (_build_deps points upload_dir at
    # tmp_path/"uploads"); the request names it by relative key only.
    ref_root = tmp_path / "uploads" / TENANT_A
    ref_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(ref_root / "reference.png")

    deps = _build_deps(storage_root=tmp_path / "storage")
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    request = AnalysisRequest(
        video_url="https://example.com/v.mp4",
        verification=VerificationCriteria(
            required_phrases=["welcome", "demo"],
            min_duration_seconds=1.0,
            max_duration_seconds=10.0,
            required_screenshots=[ReferenceImage(name="ref", reference_key="reference.png")],
        ),
    )
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"
        assert job.verification_result is not None
        assert job.verification_result["passed"] is True
        # Pin the DENOMINATOR, not just the verdict. `passed is True` alone survives
        # two mutations that gut the gate: emptying `required_phrases`' loop in
        # verifier.check_required_phrases, and — worse — passing
        # `criteria=VerificationCriteria()` instead of the job's own criteria at the
        # pipeline -> verifier seam (pipeline._run_verification_stage). Both leave
        # `passed` True with nothing actually checked. `criteria_total == 4` is the
        # tally of what was configured on the request above: 2 required_phrases +
        # min_duration_seconds + max_duration_seconds + 1 required_screenshot = 5.
        assert job.verification_result["criteria_total"] == 5
        assert job.verification_result["criteria_passed"] == 5
        assert job.verification_result["failures"] == []
        # The quality signal has to actually reach the row. Nothing in this file
        # read `quality_metrics`, so dropping it from the `update_fields` call in
        # `pipeline._run_quality_stage` left every test in the file green.
        assert job.quality_metrics is not None
        assert job.quality_metrics["overall_score"] == 0.85
        assert job.artifacts is not None
        assert "verification.json" in job.artifacts
        assert "verification.md" in job.artifacts


async def test_pipeline_handles_transcription_failure(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    deps = _build_deps(
        storage_root=tmp_path / "storage",
        transcription_client=_RaisingTranscriptionClient(),
    )
    runner = PipelineRunner(dependencies=deps)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        # Job should still complete even though transcription failed.
        assert job.status == "completed"
        # A transient transcription/ASR failure now stores the unavailable
        # sentinel (not None), so verification marks transcript criteria
        # indeterminate instead of falsely failing the content.
        assert job.transcript_data is not None
        assert job.transcript_data.get("_transcript_unavailable") is True


async def test_pipeline_handles_synthesis_failure(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    deps = _build_deps(
        storage_root=tmp_path / "storage",
        vision_provider=_SynthesisFailingVision(),
    )
    runner = PipelineRunner(dependencies=deps)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"


async def test_pipeline_video_url_path(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import clear_settings_cache
    from src.services import pipeline as pipeline_module

    monkeypatch.setenv("STORAGE_UPLOAD_DIR", str(tmp_path / "uploads"))
    clear_settings_cache()

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    deps = _build_deps(storage_root=tmp_path / "storage", settings=get_settings())
    runner = PipelineRunner(dependencies=deps)
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_A,
            video_path=None,
            video_url=request.video_url,
            upload_filename=None,
            request_config=request.model_dump(mode="json"),
            verification_criteria=None,
        )
    await runner.enqueue(job.job_id, TENANT_A)
    await runner.wait_for(job.job_id)

    async with db_session_factory() as session:
        refreshed = await JobRepository(session).get_by_job_id(job.job_id, tenant_id=TENANT_A)
        assert refreshed is not None
        assert refreshed.status == "completed"


async def test_pipeline_no_audio_skips_transcription_call(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    async def _silent_probe(_path: Path) -> VideoMetadata:
        return VideoMetadata(
            duration_seconds=5.0,
            width=640,
            height=480,
            fps=30.0,
            frame_count=150,
            has_audio=False,
        )

    deps = _build_deps(storage_root=tmp_path / "storage")
    deps.probe = _silent_probe
    runner = PipelineRunner(dependencies=deps)
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"x")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)
    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)
    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"
        assert job.transcript_data is None


async def test_pipeline_resolve_source_no_inputs(tmp_path: Path) -> None:
    deps = _build_deps(storage_root=tmp_path / "storage")
    runner = PipelineRunner(dependencies=deps)
    request = AnalysisRequest.for_upload()
    with pytest.raises(ValueError, match="Neither"):
        await runner._resolve_source("job-1", request, TENANT_A, None)


async def test_pipeline_resolve_source_finds_uploaded_file(tmp_path: Path) -> None:
    # /analyze/upload streams the source into the object store under a
    # tenant-scoped key and creates a job with no video_url;
    # _resolve_source must fetch it back from the object store, else every upload
    # fails ("Neither ... provided"). Regression for that dead path.
    settings = get_settings().model_copy(deep=True)
    settings.storage.upload_dir = str(tmp_path / "uploads")
    deps = _build_deps(storage_root=tmp_path / "storage", settings=settings)
    runner = PipelineRunner(dependencies=deps)
    job_id = "job-upload"
    # Simulate the upload route: stream the source into the object store.
    src = tmp_path / "src.mp4"
    src.write_bytes(b"MP4DATA")
    await deps.storage.store_upload(TENANT_A, job_id, src, "clip.mp4")
    resolved = await runner._resolve_source(
        job_id, AnalysisRequest.for_upload(), TENANT_A, "clip.mp4"
    )
    assert resolved.read_bytes() == b"MP4DATA"


async def test_resolve_source_multi_replica_pod_a_to_pod_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core multi-replica guarantee: an upload accepted on 'pod A' is
    resolvable when the analyze work runs on 'pod B' (a DIFFERENT replica with
    an empty local disk), because the source lives in the shared object store
    and is streamed down by tenant-scoped key — never read from pod A's disk and
    never buffered whole into RAM (get_bytes is forbidden here)."""
    from src.config import clear_settings_cache
    from src.services.storage_service import StorageService

    object_store: dict[str, bytes] = {}

    def make_minio_storage() -> StorageService:
        monkeypatch.setenv("STORAGE_BACKEND", "minio")
        monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:9000")
        monkeypatch.setenv("STORAGE_MINIO_ACCESS_KEY", "ak")
        monkeypatch.setenv("STORAGE_MINIO_SECRET_KEY", "sk")
        clear_settings_cache()
        svc = StorageService()

        async def _put_file(key: str, local_path: Any, *, content_type: str = "", **_: Any) -> str:
            # Simulate fput_object: stream the file from disk into the store.
            object_store[key] = Path(local_path).read_bytes()
            return key

        async def _presign(key: str, *, expires: Any = None, bucket: Any = None) -> str:
            return f"https://minio.test/video-analyzer-service/{key}"

        async def _forbidden_get_bytes(*_a: Any, **_k: Any) -> bytes | None:
            raise AssertionError("resolve must stream via the downloader, not buffer via get_bytes")

        monkeypatch.setattr(svc._storage, "put_file", _put_file)
        monkeypatch.setattr(svc._storage, "presign_get", _presign)
        monkeypatch.setattr(svc._storage, "get_bytes", _forbidden_get_bytes)
        return svc

    # --- pod A: accept the upload, stream it into the shared object store ---
    pod_a = make_minio_storage()
    src = tmp_path / "src.mp4"
    src.write_bytes(b"MULTIPOD")
    await pod_a.store_upload("tenant-x", "job-42", src, "clip.mp4")
    assert object_store  # source now durable in the object store

    # --- pod B: a fresh replica (own empty scratch) resolves it from the store ---
    class _StorePresignedDownloader:
        """Mirrors the SSRF-gated VideoDownloader: streams the presigned object
        to local disk by key (never touches pod A's filesystem)."""

        async def download(self, *, video_url: str, output_dir: Path) -> Path:
            key = video_url.split("/video-analyzer-service/", 1)[1]
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / "clip.mp4"
            dest.write_bytes(object_store[key])
            return dest

    settings = get_settings().model_copy(deep=True)
    settings.storage.upload_dir = str(tmp_path / "podb-uploads")
    deps = _build_deps(storage_root=tmp_path / "podb-storage", settings=settings)
    deps.storage = make_minio_storage()  # pod B's own storage, empty local scratch
    deps.video_downloader = _StorePresignedDownloader()
    runner = PipelineRunner(dependencies=deps)

    resolved = await runner._resolve_source(
        "job-42", AnalysisRequest.for_upload(), "tenant-x", "clip.mp4"
    )
    assert resolved.read_bytes() == b"MULTIPOD"
    # Resolved into pod B's own upload dir, not pod A's.
    assert str(resolved).startswith(str(tmp_path / "podb-uploads"))


async def test_pipeline_enqueue_idempotent(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    deps = _build_deps(storage_root=tmp_path / "storage")
    runner = PipelineRunner(dependencies=deps)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    first = runner._tasks[job_id]
    assert runner.active_job_ids() == [job_id]

    await runner.enqueue(job_id, TENANT_A)  # second call must not double-launch
    # THE assertion. `enqueue` has no await before `create_task`, so the first
    # task cannot have started (let alone finished) — `existing.done()` is
    # deterministically False here and the guard in `PipelineRunner.enqueue`
    # must return early, leaving the SAME task object in place. Without this,
    # the test had no assertion at all: deleting the guard double-launches the
    # pipeline, `_run_job` swallows every exception, and `wait_for` awaits only
    # the second (replacement) task — so a duplicate run was completely silent.
    assert runner._tasks[job_id] is first, "second enqueue replaced the in-flight task"
    assert runner.active_job_ids() == [job_id]

    await runner.wait_for(job_id)
    assert runner.active_job_ids() == []


def test_select_vision_provider_stub_returns_fake(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``stub`` is a RECOGNISED alias, not a fall-through to the unknown branch."""
    monkeypatch.setenv("VISION_DEFAULT_PROVIDER", "stub")
    from src.config import clear_settings_cache

    clear_settings_cache()
    settings = get_settings()
    with caplog.at_level(logging.WARNING, logger="src.services.pipeline"):
        provider = _select_vision_provider(settings)
    assert isinstance(provider, FakeVisionProvider)
    # `isinstance(..., FakeVisionProvider)` ALONE cannot fail: `_select_vision_provider`
    # returns FakeVisionProvider() as its terminal statement on every path except a
    # successful `wm` import, so deleting the `{"stub", "fake"}` branch entirely left
    # this test green — the documented offline escape hatch was pinned by nothing.
    # The unknown-provider branch is the only observable difference, and it warns.
    assert not [r for r in caplog.records if "Unknown vision provider" in r.getMessage()], (
        "'stub' fell through to the unknown-provider branch; the alias is not wired"
    )


def test_select_vision_provider_wm_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default ``wm`` provider builds a WMVisionProvider without per-SDK API keys."""
    monkeypatch.setenv("VISION_DEFAULT_PROVIDER", "wm")
    from src.config import clear_settings_cache

    clear_settings_cache()
    settings = get_settings()
    provider = _select_vision_provider(settings)
    # WMVisionProvider.name == "wm" (per src/services/vision/wm_provider.py)
    assert provider.name == "wm"


def test_select_vision_provider_unknown_falls_back_to_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_DEFAULT_PROVIDER", "nonsense-provider")
    from src.config import clear_settings_cache

    clear_settings_cache()
    settings = get_settings()
    provider = _select_vision_provider(settings)
    assert isinstance(provider, FakeVisionProvider)
