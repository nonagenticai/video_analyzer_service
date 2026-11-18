"""Pipeline end-to-end tests with all collaborators faked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from w2v_common.storage import StorageService as CommonStorage

from src.api.schemas import AnalysisRequest, VerificationCriteria
from src.config import Settings, StorageSettings, get_settings
from src.domain.analysis import ExtractedFrame, FrameExtraction, QualityMetrics, VideoMetadata
from src.repository.job_repository import JobRepository
from src.services.artifact_generator import ArtifactGenerator
from src.services.pipeline import (
    PipelineDependencies,
    PipelineRunner,
    _build_frame_analyses,
    _enabled_stages,
    _needs_transcript,
    _scene_bounds,
)
from src.services.storage_service import StorageService
from tests.conftest import TENANT_A


class _FakeStorage(StorageService):
    def __init__(self, root: Path) -> None:
        # Bypass __init__ of StorageService; back it with a local-mode facade.
        self._backend_type = "local"
        self._local_path = root
        self._scratch_root = root
        self._storage = CommonStorage(StorageSettings(backend="local", local_path=str(root)))


def _test_settings(tmp_path: Path) -> Settings:
    """Settings whose ``upload_dir`` is a per-test temp dir.

    The pipeline materialises every source under ``upload_dir`` (the SSRF-gated
    downloader / the object-store fetch). The default is the in-container
    ``/app/data/uploads``, which a test run must never touch.
    """
    settings = get_settings().model_copy(deep=True)
    settings.storage.upload_dir = str(tmp_path / "uploads")
    return settings


def _fake_metadata(has_audio: bool = True) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        has_audio=has_audio,
        format_name="mp4",
    )


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
        for i in range(3):
            p = output_dir / f"frame_{i:04d}.jpg"
            p.write_bytes(b"JFIF")
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
        path.write_bytes(b"FAKE-MP4")
        return path


class _FakeSceneDetector:
    def detect(self, frames: list[ExtractedFrame]) -> list[float]:
        return [f.timestamp for f in frames]


class _FakeOCR:
    def extract_text(self, image_path: Path) -> str:
        return "Hello world"


class _FakeQualityAnalyzer:
    def analyze(
        self, frames: list[ExtractedFrame], *, frames_planned: int | None = None
    ) -> QualityMetrics:
        return QualityMetrics(
            mean_sharpness=500.0,
            mean_brightness=0.5,
            motion_score=0.1,
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
            "duration": 10.0,
            "text": "Welcome to the demo. In this video we will build a feature flag.",
            "segments": [],
        }


async def _fake_probe(video_path: Path) -> VideoMetadata:
    return _fake_metadata()


@pytest.fixture
def pipeline_runner(tmp_path: Path) -> PipelineRunner:
    from src.services.vision.fake_provider import FakeVisionProvider

    settings = _test_settings(tmp_path)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=settings,
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=FakeVisionProvider(),
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    return PipelineRunner(dependencies=deps)


async def _insert_job(db_session_factory, *, video_path: Path, request: AnalysisRequest) -> str:
    async with db_session_factory() as session:
        repo = JobRepository(session)
        job = await repo.create(
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


def test_enabled_stages_full_pipeline() -> None:
    req = AnalysisRequest(video_url="https://example.com/v.mp4")
    stages = _enabled_stages(req)
    assert "probe" in stages
    assert "frame_extract" in stages
    assert "transcribe" in stages
    assert "vision" in stages
    assert "ocr" in stages
    assert "quality" in stages
    assert "synthesis" in stages
    assert "verification" not in stages


def test_enabled_stages_no_vision_no_ocr() -> None:
    req = AnalysisRequest(
        video_url="https://example.com/v.mp4",
        extract_frame_descriptions=False,
        extract_ocr=False,
        compute_quality_metrics=False,
    )
    stages = _enabled_stages(req)
    assert "vision" not in stages
    assert "ocr" not in stages
    assert "quality" not in stages


def test_needs_transcript_flag_logic() -> None:
    url = "https://example.com/a.mp4"
    assert _needs_transcript(AnalysisRequest(video_url=url)) is True
    assert _needs_transcript(AnalysisRequest(video_url=url, generate_walkthrough=False)) is False
    req_with_verif = AnalysisRequest(
        video_url="https://example.com/a.mp4",
        generate_walkthrough=False,
        verification=VerificationCriteria(required_phrases=["hello"]),
    )
    assert _needs_transcript(req_with_verif) is True


def test_build_frame_analyses_merges() -> None:
    from src.services.vision.base import FrameVisionResult

    frames = [ExtractedFrame(index=0, timestamp=0.0, path=Path("/tmp/a.jpg"))]
    vision = [FrameVisionResult(index=0, timestamp=0.0, description="desc", labels=["ui"])]
    ocr = {0: "hello"}
    analyses = _build_frame_analyses(frames=frames, frame_vision=vision, ocr_by_index=ocr)
    assert analyses[0].description == "desc"
    assert analyses[0].ocr_text == "hello"
    assert analyses[0].labels == ["ui"]


def test_scene_bounds_pairs() -> None:
    pairs = _scene_bounds([0.0, 3.0, 6.0], total=10.0)
    assert pairs == [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    assert _scene_bounds([], total=10.0) == []


async def test_pipeline_end_to_end(
    pipeline_runner: PipelineRunner,
    db_session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point get_session_maker at our test session factory
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    video = tmp_path / "input.mp4"
    video.write_bytes(b"FAKE-MP4")

    request = AnalysisRequest(
        video_url="https://example.com/input.mp4",
        extract_frame_descriptions=True,
        extract_ocr=True,
        compute_quality_metrics=True,
        generate_walkthrough=True,
        generate_ui_inventory=True,
        generate_frame_index=True,
    )
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await pipeline_runner.enqueue(job_id, TENANT_A)
    await pipeline_runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.width == 1920
        assert job.height == 1080
        assert job.has_audio == 1
        assert job.transcript_data is not None
        assert isinstance(job.scene_boundaries, list)
        assert isinstance(job.frame_analyses, list)
        assert len(job.frame_analyses) == 3
        assert job.quality_metrics is not None
        assert job.artifacts is not None
        assert "walkthrough.md" in job.artifacts
        assert "ui_inventory.md" in job.artifacts
        assert "frame_index.md" in job.artifacts


async def test_pipeline_marks_failed_when_no_source(
    pipeline_runner: PipelineRunner,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    # An upload-shaped config whose row carries NO upload_filename: the model's
    # source invariant cannot see that (the source lives on the row), so this is
    # still re-validated strictly and _resolve_source has nothing to materialise.
    request = AnalysisRequest.for_upload()
    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_A,
            video_path=None,
            video_url=None,
            upload_filename=None,
            request_config=request.model_dump(mode="json"),
            verification_criteria=None,
        )
    await pipeline_runner.enqueue(job.job_id, TENANT_A)
    await pipeline_runner.wait_for(job.job_id)

    async with db_session_factory() as session:
        refreshed = await JobRepository(session).get_by_job_id(job.job_id, tenant_id=TENANT_A)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None


async def test_pipeline_fails_closed_on_legacy_video_path_config(
    pipeline_runner: PipelineRunner,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job row persisted before the removal must NOT replay the file read.

    ``request_config`` is re-validated in ``_load_job_request``; the rejected
    ``video_path`` surfaces as a terminal failure, never as an /etc/passwd read.
    """
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    async with db_session_factory() as session:
        job = await JobRepository(session).create(
            tenant_id=TENANT_A,
            video_path="/etc/passwd",
            video_url=None,
            upload_filename=None,
            request_config={"video_path": "/etc/passwd"},
            verification_criteria=None,
        )
    await pipeline_runner.enqueue(job.job_id, TENANT_A)
    await pipeline_runner.wait_for(job.job_id)

    async with db_session_factory() as session:
        refreshed = await JobRepository(session).get_by_job_id(job.job_id, tenant_id=TENANT_A)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert refreshed.artifacts in (None, {})


async def test_pipeline_skips_transcription_when_no_audio(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services import pipeline as pipeline_module
    from src.services.vision.fake_provider import FakeVisionProvider

    async def _fake_session_maker():
        return db_session_factory

    async def _fake_probe_no_audio(video_path: Path) -> VideoMetadata:
        return _fake_metadata(has_audio=False)

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=_test_settings(tmp_path),
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=FakeVisionProvider(),
        probe=_fake_probe_no_audio,
        artifact_generator=ArtifactGenerator(),
    )
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "silent.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)
    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"
        # Transcript was attempted but audio_path is None → transcript_data is None
        assert job.transcript_data is None or job.transcript_data == {}


async def test_transcription_stage_returns_unavailable_marker_on_client_error(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """Bug 1: an ASR TranscriptionClientError must yield the infra-unavailable
    sentinel (distinct from None/""), NOT a silent None that later reads as a
    content-'failed' verdict in the verifier."""
    from src.services.transcription_client import TranscriptionClientError
    from src.services.verifier import is_transcript_unavailable

    class _RaisingTranscriptionClient:
        async def transcribe_file(self, **_kwargs: Any) -> dict[str, Any]:
            raise TranscriptionClientError("asr_backend unreachable")

    pipeline_runner._deps.transcription_client = _RaisingTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    result = await pipeline_runner._run_transcription_stage(
        "job-1", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    assert is_transcript_unavailable(result) is True


def test_scene_bounds_handles_duplicates() -> None:
    pairs = _scene_bounds([0.0, 3.0, 3.0, 6.0], total=10.0)
    # duplicates deduped by set
    assert pairs == [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]


async def test_pipeline_cost_cap_exceeded_fails_job(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_vision_cost_usd should fail the job when vision cost exceeds it."""
    from src.services import pipeline as pipeline_module
    from src.services.vision.fake_provider import FakeVisionProvider

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    # FakeVisionProvider with per-frame cost → 3 frames * $1 = $3
    vision = FakeVisionProvider(cost_per_frame=1.0)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=_test_settings(tmp_path),
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=vision,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(video_url="https://example.com/v.mp4", max_vision_cost_usd=0.5)
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "failed"
        assert "cost" in (job.error_message or "").lower()


async def test_cost_cap_is_enforced_before_any_frame_is_billed(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap job must be refused without paying for it.

    The cap used to be checked only AFTER ``analyze_frames`` had billed every
    frame, so a job that could never satisfy its own cap still spent the full
    amount and then failed -- the tenant was charged for a result they never
    received. The assertion that distinguishes pre-flight from post-hoc is that
    ``analyze_frames`` is never called at all.
    """
    from src.services import pipeline as pipeline_module
    from src.services.vision.fake_provider import FakeVisionProvider

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    calls: list[int] = []

    class _CountingProvider(FakeVisionProvider):
        async def analyze_frames(self, frames):  # type: ignore[no-untyped-def]
            calls.append(len(frames))
            return await super().analyze_frames(frames)

    # 3 frames * $1 = $3 estimated, against a $0.50 cap.
    vision = _CountingProvider(cost_per_frame=1.0)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=_test_settings(tmp_path),
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=vision,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(video_url="https://example.com/v.mp4", max_vision_cost_usd=0.5)
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    assert calls == [], (
        f"analyze_frames was called with {calls} frame batch(es) -- the cap was "
        "enforced only AFTER billing. A job that cannot satisfy its cap must "
        "never reach the provider."
    )

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "failed"
        assert "no frames were billed" in (job.error_message or "")


async def test_cost_cap_within_budget_still_runs_the_provider(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-flight must not reject a job that fits its cap."""
    from src.services import pipeline as pipeline_module
    from src.services.vision.fake_provider import FakeVisionProvider

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    calls: list[int] = []

    class _CountingProvider(FakeVisionProvider):
        async def analyze_frames(self, frames):  # type: ignore[no-untyped-def]
            calls.append(len(frames))
            return await super().analyze_frames(frames)

    # 3 frames * $0.01 = $0.03, comfortably under a $5 cap.
    vision = _CountingProvider(cost_per_frame=0.01)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=_test_settings(tmp_path),
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=vision,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(video_url="https://example.com/v.mp4", max_vision_cost_usd=5.0)
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    assert calls, "the provider must still run when the job fits its cap"


async def test_vision_cost_includes_synthesis_cost(
    tmp_path: Path,
    db_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: the synthesis call's cost_usd was dropped from vision_cost — the
    persisted metering total counted only per-frame descriptions. It must now
    include the synthesis cost so spend + the cost cap account for it."""
    from src.services import pipeline as pipeline_module
    from src.services.vision.base import VisionSynthesisResult
    from src.services.vision.fake_provider import FakeVisionProvider

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)

    class _CostedVisionProvider(FakeVisionProvider):
        def __init__(self, *, cost_per_frame: float, synthesis_cost: float) -> None:
            super().__init__(cost_per_frame=cost_per_frame)
            self._synthesis_cost = synthesis_cost

        async def synthesize(
            self,
            *,
            frame_results: list[Any],
            transcript_text: str | None,
        ) -> VisionSynthesisResult:
            result = await super().synthesize(
                frame_results=frame_results, transcript_text=transcript_text
            )
            result.cost_usd = self._synthesis_cost
            return result

    # 3 frames * $0.10 = $0.30 frame cost, plus $0.20 synthesis = $0.50 total.
    vision = _CostedVisionProvider(cost_per_frame=0.10, synthesis_cost=0.20)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=_test_settings(tmp_path),
        frame_extractor=_FakeFrameExtractor(),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=vision,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    runner = PipelineRunner(dependencies=deps)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(video_url="https://example.com/v.mp4")
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)

    await runner.enqueue(job_id, TENANT_A)
    await runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
        assert job is not None
        assert job.status == "completed"
        # frame cost (0.30) + synthesis cost (0.20) — not just the frame cost.
        assert job.vision_cost_usd == pytest.approx(0.50)


class _FailingAudioExtractor:
    """ffmpeg could not extract the audio stream — an INFRA failure, not a silent video."""

    async def extract(self, *, video_path: Path, output_path: Path) -> Path:
        raise RuntimeError("ffmpeg exited with code 1: Invalid data found")


async def test_transcription_stage_marks_unavailable_when_audio_extraction_failed(
    pipeline_runner: PipelineRunner,
) -> None:
    """A failed AUDIO EXTRACTION is a measurement gap, not an empty transcript.

    ``audio_path is None`` conflated "this video has no audio stream" with
    "ffmpeg could not extract the audio". The second must yield the
    infra-unavailable sentinel; the first must still yield ``None``.
    """
    from src.services.verifier import is_transcript_unavailable

    request = AnalysisRequest(video_url="https://example.com/x.mp4")

    no_audio = await pipeline_runner._run_transcription_stage(
        "job-1", request, None, audio_error=None
    )
    assert no_audio is None

    extraction_failed = await pipeline_runner._run_transcription_stage(
        "job-1",
        request,
        None,
        audio_error="audio extraction failed: ffmpeg exited with code 1",
    )
    assert is_transcript_unavailable(extraction_failed) is True


async def test_failed_audio_extraction_does_not_vacuously_pass_forbidden_phrases(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
    db_session_factory,
) -> None:
    """THE MONEY CASE: a video that DOES say the forbidden phrase must not verify
    clean-green just because its audio could not be extracted.

    With the defect, the verifier received ``transcript_data=None``, normalised it
    to an empty haystack, found no forbidden phrase in it, and reported
    ``passed=True`` on a criterion nothing ever measured.
    """
    from src.services.verifier import Verifier

    pipeline_runner._deps.audio_extractor = _FailingAudioExtractor()
    pipeline_runner._deps.verifier = Verifier()

    video = tmp_path / "v.mp4"
    video.write_bytes(b"FAKE-MP4")
    request = AnalysisRequest(
        video_url="https://example.com/v.mp4",
        verification=VerificationCriteria(forbidden_phrases=["internal build"]),
    )
    job_id = await _insert_job(db_session_factory, video_path=video, request=request)
    await pipeline_runner.enqueue(job_id, TENANT_A)
    await pipeline_runner.wait_for(job_id)

    async with db_session_factory() as session:
        job = await JobRepository(session).get_by_job_id(job_id, tenant_id=TENANT_A)
    assert job is not None
    result = job.verification_result
    assert result is not None
    assert result["passed"] is False
    assert "forbidden_phrases" in result["indeterminate"]


class _EmptyDecodeTranscriptionClient:
    """A 200 response whose decode produced NOTHING usable.

    Shape-identical to a clean transcription apart from the advisory
    ``no_speech_reason`` flag: every required field present, HTTP 200.
    """

    async def transcribe_file(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "backend": "faster-whisper",
            "model": "large-v3",
            "language": None,
            "duration": 0.0,
            "text": "",
            "segments": [],
            "srt": None,
            "no_speech": True,
            "no_speech_reason": "empty_decode",
        }


class _SilentAudioTranscriptionClient:
    """A 200 response over MEASURABLE audio that genuinely contained no speech."""

    async def transcribe_file(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "backend": "faster-whisper",
            "model": "large-v3",
            "language": "en",
            "duration": 42.0,
            "text": "",
            "segments": [],
            "srt": None,
            "no_speech": True,
            "no_speech_reason": "silent_audio",
        }


async def test_transcription_stage_marks_empty_decode_unavailable(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """A 200 carrying ``empty_decode`` is a MEASUREMENT GAP, not an empty transcript.

    The producer's own schema says ``empty_decode`` means "the decode produced
    nothing usable ... this is NOT evidence of silence". Consuming it as a plain
    empty transcript makes every ``forbidden_phrases`` criterion pass vacuously.
    """
    from src.services.verifier import is_transcript_unavailable

    pipeline_runner._deps.transcription_client = _EmptyDecodeTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    result = await pipeline_runner._run_transcription_stage(
        "job-empty-decode", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    assert is_transcript_unavailable(result) is True


async def test_transcription_stage_keeps_silent_audio_as_a_real_measurement(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """ANTI-OVER-CORRECTION: ``silent_audio`` IS a real measurement.

    The decode succeeded over a measurable duration and Whisper suppressed every
    segment — a forbidden phrase truly is absent. Marking it indeterminate would
    turn every silent clip into a false alarm.
    """
    from src.services.verifier import is_transcript_unavailable

    pipeline_runner._deps.transcription_client = _SilentAudioTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    result = await pipeline_runner._run_transcription_stage(
        "job-silent", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    assert is_transcript_unavailable(result) is False
    assert result is not None
    assert result["text"] == ""
    assert result["duration"] == 42.0


async def test_empty_decode_does_not_vacuously_pass_forbidden_phrases(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """THE MONEY CASE: an undecodable transcript must not report CLEAN.

    Chains the real transcription stage into the real verifier — the gap has to
    survive the seam, not merely be visible at the boundary.
    """
    from src.services.verifier import Verifier

    pipeline_runner._deps.transcription_client = _EmptyDecodeTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    transcript_data = await pipeline_runner._run_transcription_stage(
        "job-empty-decode", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    result = await Verifier().verify(
        criteria=VerificationCriteria(forbidden_phrases=["internal build"]),
        metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            has_audio=True,
        ),
        transcript_data=transcript_data,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert "forbidden_phrases" in result["indeterminate"]
    assert result["passed"] is False


async def test_silent_audio_still_yields_a_real_forbidden_phrase_verdict(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """ANTI-OVER-CORRECTION, end to end: silence is evidence, so the criterion
    must be EVALUATED (and pass for real), never recorded as indeterminate."""
    from src.services.verifier import Verifier

    pipeline_runner._deps.transcription_client = _SilentAudioTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    transcript_data = await pipeline_runner._run_transcription_stage(
        "job-silent", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    result = await Verifier().verify(
        criteria=VerificationCriteria(forbidden_phrases=["internal build"]),
        metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            has_audio=True,
        ),
        transcript_data=transcript_data,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["indeterminate"] == []
    assert result["passed"] is True
    assert result["criteria_total"] == 1
    assert result["criteria_passed"] == 1


class _MalformedTranscriptionClient:
    """A 200 whose body is JSON but not a transcription response.

    ``transcribe_file`` returns the worker's JSON verbatim, so a partial or
    wrong-shaped 200 used to reach the verifier as a plain dict with no
    ``text`` — an empty haystack, i.e. the same vacuous pass the
    ``empty_decode`` route closes, arriving by a different cause.
    """

    async def transcribe_file(self, **_kwargs: Any) -> dict[str, Any]:
        from src.services.transcription_client import TranscriptionClientError

        raise TranscriptionClientError(
            "transcription_worker returned a structurally invalid 200 body: "
            "response body has no 'text' field"
        )


async def test_empty_decode_is_not_a_pass_alongside_a_measured_criterion(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """THE COMMON SHAPE: a transcript gap next to a criterion that DID measure.

    The headline case above defines a transcript criterion and nothing else, so
    the denominator collapses to zero and the empty-denominator guard catches
    it. Add any non-transcript criterion — here ``expected_resolution``, which
    every real caller sends — and the denominator is 1 again, ``failures`` is
    empty, and ``passed`` used to come back ``True`` while ``indeterminate``
    named ``forbidden_phrases``. A caller reading ``passed`` (the field callers
    actually read) saw GREEN on an undecodable transcript.
    """
    from src.services.verifier import Verifier

    pipeline_runner._deps.transcription_client = _EmptyDecodeTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    transcript_data = await pipeline_runner._run_transcription_stage(
        "job-empty-decode-multi",
        AnalysisRequest(video_url="https://example.com/x.mp4"),
        audio,
    )
    result = await Verifier().verify(
        criteria=VerificationCriteria(
            forbidden_phrases=["internal build"],
            expected_resolution=(1920, 1080),
        ),
        metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            has_audio=True,
        ),
        transcript_data=transcript_data,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    # The resolution criterion genuinely passed, so the denominator is NOT zero.
    assert result["criteria_total"] == 1
    assert result["criteria_passed"] == 1
    assert result["failures"] == []
    assert result["indeterminate"] == ["forbidden_phrases"]
    # ...and yet the run cannot be called passed: one criterion was never measured.
    assert result["passed"] is False


async def test_a_fully_measured_multi_criteria_run_still_passes(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """CONTROL for the guard above: nothing indeterminate ⇒ ``passed`` stays True.

    Same two criteria, same shapes, but the worker answers with a real decode.
    A guard that turned every multi-criteria run red would be worse than the bug.
    """
    from src.services.verifier import Verifier

    pipeline_runner._deps.transcription_client = _SilentAudioTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    transcript_data = await pipeline_runner._run_transcription_stage(
        "job-measured-multi",
        AnalysisRequest(video_url="https://example.com/x.mp4"),
        audio,
    )
    result = await Verifier().verify(
        criteria=VerificationCriteria(
            forbidden_phrases=["internal build"],
            expected_resolution=(1920, 1080),
        ),
        metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            has_audio=True,
        ),
        transcript_data=transcript_data,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["indeterminate"] == []
    assert result["failures"] == []
    assert result["criteria_total"] == 2
    assert result["criteria_passed"] == 2
    assert result["passed"] is True


async def test_malformed_200_is_a_gap_not_a_clean_transcript(
    pipeline_runner: PipelineRunner,
    tmp_path: Path,
) -> None:
    """A structurally invalid 200 must reach the verifier as INDETERMINATE.

    ``is_empty_decode`` answers ``False`` for a body it cannot recognise — the
    right call for an OLDER worker, the wrong one for a body that carries no
    transcript at all. The client now refuses such a body, and the stage's
    existing ``TranscriptionClientError`` handler routes it to the same
    sentinel, so the vacuous pass cannot return by this second cause.
    """
    from src.services.verifier import Verifier, is_transcript_unavailable

    pipeline_runner._deps.transcription_client = _MalformedTranscriptionClient()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    transcript_data = await pipeline_runner._run_transcription_stage(
        "job-malformed", AnalysisRequest(video_url="https://example.com/x.mp4"), audio
    )
    assert is_transcript_unavailable(transcript_data) is True

    result = await Verifier().verify(
        criteria=VerificationCriteria(
            forbidden_phrases=["internal build"],
            expected_resolution=(1920, 1080),
        ),
        metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=300,
            has_audio=True,
        ),
        transcript_data=transcript_data,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["indeterminate"] == ["forbidden_phrases"]
    assert result["passed"] is False
