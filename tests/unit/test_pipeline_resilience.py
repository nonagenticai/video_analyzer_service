"""Pipeline resilience: partial results, wall-clock deadline, throttles, disk reclaim.

Regression suite for the 2026-08-21 live run ``01a021c6-01ba-70dc-a9cb-0aa194de7eb1``,
where a 15-clip ``footage_critique`` polled this service for 3m09s and received
``frame_analyses: null`` — every frame that HAD been scored (and billed) was
discarded because the column is written once, after both vision and OCR finish.

Each test here fails against the pre-fix pipeline:

* :func:`test_vision_results_persist_incrementally_when_a_later_batch_dies`
  — pre-fix ``analyze_frames`` is called ONCE for every frame, so a mid-way
  failure leaves ``frame_analyses`` NULL.
* :func:`test_a_job_has_a_wall_clock_deadline` — pre-fix nothing bounds a job,
  so a hung vision provider runs (and bills) forever.
* :func:`test_vision_batch_size_is_wired` / :func:`test_max_concurrent_jobs_is_wired`
  — both knobs were referenced nowhere in ``src/``.
* :func:`test_job_scratch_is_reclaimed_on_success` / ``..._on_failure`` — pre-fix
  the only ``rmtree`` is on the DELETE route, which nothing calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.api.routes.analysis import _job_to_result
from src.api.schemas import AnalysisRequest
from src.domain.analysis import ExtractedFrame, FrameExtraction, VideoMetadata
from src.repository.job_repository import JobRepository
from src.services.artifact_generator import ArtifactGenerator
from src.services.pipeline import PipelineDependencies, PipelineRunner
from src.services.vision.base import (
    FrameInput,
    FrameVisionResult,
    VisionProvider,
    VisionSynthesisResult,
)
from tests.conftest import TENANT_A
from tests.unit.test_pipeline import (
    _fake_probe,
    _FakeAudioExtractor,
    _FakeDownloader,
    _FakeOCR,
    _FakeQualityAnalyzer,
    _FakeSceneDetector,
    _FakeStorage,
    _FakeTranscriptionClient,
    _insert_job,
    _test_settings,
)

_TENANT = TENANT_A


class _NFrameExtractor:
    """Frame extractor that always writes exactly ``count`` JPEGs."""

    def __init__(self, count: int) -> None:
        self._count = count

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
        for i in range(self._count):
            p = output_dir / f"frame_{i:04d}.jpg"
            p.write_bytes(b"JFIF")
            frames.append(ExtractedFrame(index=i, timestamp=float(i), path=p))
        return FrameExtraction(frames=frames, planned=len(frames))


class _RecordingVisionProvider(VisionProvider):
    """Vision provider that records the size of every ``analyze_frames`` call."""

    name = "recording"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def estimated_cost_for(self, frame_count: int) -> float:
        return 0.0

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        self.batch_sizes.append(len(frames))
        return [
            FrameVisionResult(
                index=f.index, timestamp=f.timestamp, description=f"desc-{f.index}", labels=[]
            )
            for f in frames
        ]

    async def synthesize(
        self, *, frame_results: list[FrameVisionResult], transcript_text: str | None
    ) -> VisionSynthesisResult:
        return VisionSynthesisResult(walkthrough="w", ui_notes="u")


class _DiesAfterNFramesProvider(_RecordingVisionProvider):
    """Scores the first ``limit`` frames, then raises a NON-VisionProviderError.

    A plain ``RuntimeError`` is deliberate: ``VisionProviderError`` is swallowed
    by ``_run_vision_stage``, which would hide whether the frames already scored
    survived. This is the shape of the live failure — an abrupt loss of the
    worker mid-run.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._scored = 0

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        self.batch_sizes.append(len(frames))
        out: list[FrameVisionResult] = []
        for f in frames:
            if self._scored >= self._limit:
                raise RuntimeError("worker vanished mid-run")
            self._scored += 1
            out.append(
                FrameVisionResult(
                    index=f.index,
                    timestamp=f.timestamp,
                    description=f"desc-{f.index}",
                    labels=[],
                    cost_usd=0.01,
                )
            )
        return out


class _HangingVisionProvider(_RecordingVisionProvider):
    """Never returns — stands in for a WM vision call with thinking ON."""

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        self.batch_sizes.append(len(frames))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _ConcurrencyProbeProvider(_RecordingVisionProvider):
    """Records the peak number of jobs inside the vision stage at once."""

    def __init__(self) -> None:
        super().__init__()
        self.inflight = 0
        self.peak_inflight = 0

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            await asyncio.sleep(0.25)
            return await super().analyze_frames(frames)
        finally:
            self.inflight -= 1


def _runner(
    tmp_path: Path,
    *,
    vision_provider: VisionProvider,
    frame_count: int = 4,
    settings_mutator: Any = None,
) -> PipelineRunner:
    settings = _test_settings(tmp_path)
    if settings_mutator is not None:
        settings_mutator(settings)
    deps = PipelineDependencies(
        storage=_FakeStorage(tmp_path / "storage"),
        settings=settings,
        frame_extractor=_NFrameExtractor(frame_count),
        audio_extractor=_FakeAudioExtractor(),
        video_downloader=_FakeDownloader(),
        scene_detector=_FakeSceneDetector(),
        ocr_service=_FakeOCR(),
        quality_analyzer=_FakeQualityAnalyzer(),
        transcription_client=_FakeTranscriptionClient(),
        vision_provider=vision_provider,
        probe=_fake_probe,
        artifact_generator=ArtifactGenerator(),
    )
    return PipelineRunner(dependencies=deps)


def _request() -> AnalysisRequest:
    return AnalysisRequest(
        video_url="https://example.com/input.mp4",
        extract_frame_descriptions=True,
        extract_ocr=True,
        compute_quality_metrics=True,
        generate_walkthrough=True,
    )


@pytest.fixture
def _patched_session_maker(db_session_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services import pipeline as pipeline_module

    async def _fake_session_maker():
        return db_session_factory

    monkeypatch.setattr(pipeline_module, "get_session_maker", _fake_session_maker)


async def _load(db_session_factory, job_id: str):
    async with db_session_factory() as session:
        return await JobRepository(session).get_by_job_id(job_id, tenant_id=_TENANT)


# --------------------------------------------------------------------------
# 1. Partial results must survive a job that is cut off mid-vision.
# --------------------------------------------------------------------------


async def test_vision_results_persist_incrementally_when_a_later_batch_dies(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """Frames already scored (and billed) must be readable after the job dies.

    Pre-fix this asserts ``frame_analyses is not None`` against a NULL column:
    the single write at the end of ``_execute`` is never reached.
    """
    provider = _DiesAfterNFramesProvider(limit=2)

    def _mutate(settings: Any) -> None:
        settings.vision.batch_size = 2

    runner = _runner(tmp_path, vision_provider=provider, frame_count=4, settings_mutator=_mutate)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )

    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=30)

    job = await _load(db_session_factory, job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.frame_analyses is not None, (
        "every frame scored before the failure was discarded — the caller cannot "
        "tell 'nothing to report' from 'I was cut off after N frames'"
    )
    scored = [fa for fa in job.frame_analyses if fa.get("description")]
    assert [fa["index"] for fa in scored] == [0, 1]

    # And the API must SAY it is partial, not just hand back a short list.
    result = _job_to_result(job)
    assert result.frame_analyses_partial is True


async def test_a_completed_job_is_not_reported_partial(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """The partial flag must not fire on a full result — otherwise it is noise."""
    runner = _runner(tmp_path, vision_provider=_RecordingVisionProvider(), frame_count=4)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "completed"
    assert _job_to_result(job).frame_analyses_partial is False


async def test_a_job_has_a_wall_clock_deadline(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """An abandoned job must stop burning spend instead of running to completion.

    Pre-fix there is no ``wait_for``/deadline anywhere in the pipeline and the
    heartbeat renews the lease forever, so this hangs until the test timeout.
    """
    provider = _HangingVisionProvider()

    def _mutate(settings: Any) -> None:
        settings.pipeline.job_deadline_seconds = 1

    runner = _runner(tmp_path, vision_provider=provider, frame_count=2, settings_mutator=_mutate)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )

    await runner.enqueue(job_id, _TENANT)
    try:
        await asyncio.wait_for(runner.wait_for(job_id), timeout=20)
    except TimeoutError:
        pytest.fail(
            "job has no wall-clock deadline: a hung vision provider ran past 20s "
            "with the heartbeat renewing its lease forever"
        )

    job = await _load(db_session_factory, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "deadline" in (job.error_message or "").lower()


# --------------------------------------------------------------------------
# 2. The two dead throttles must actually throttle.
# --------------------------------------------------------------------------


async def test_vision_batch_size_is_wired(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """``VISION_BATCH_SIZE`` must bound how many frames go to one provider call.

    Pre-fix ``settings.vision.batch_size`` appears nowhere in ``src/`` outside its
    own Field definition, so the provider is handed all 5 frames at once.
    """
    provider = _RecordingVisionProvider()

    def _mutate(settings: Any) -> None:
        settings.vision.batch_size = 2

    runner = _runner(tmp_path, vision_provider=provider, frame_count=5, settings_mutator=_mutate)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    assert provider.batch_sizes == [2, 2, 1]


async def test_max_concurrent_jobs_is_wired(
    tmp_path: Path,
    db_session_factory,
    _patched_session_maker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PIPELINE_MAX_CONCURRENT_JOBS`` must bound in-flight pipeline work.

    Pre-fix ``enqueue`` creates unbounded tasks, so both jobs sit in the vision
    stage simultaneously and ``peak_inflight`` is 2.

    Driven through the real env var (not an injected settings object) because the
    bound is per-POD: ``PipelineRunner`` resolves it from ``get_settings()``, the
    same path a deployed pod takes from its ConfigMap.
    """
    from src.config import clear_settings_cache

    # The autouse ``_clear_env`` fixture clears the settings cache at the START of
    # every test, so this value cannot leak into a sibling test.
    monkeypatch.setenv("PIPELINE_MAX_CONCURRENT_JOBS", "1")
    clear_settings_cache()

    provider = _ConcurrencyProbeProvider()
    runner = _runner(tmp_path, vision_provider=provider, frame_count=2)
    job_a = await _insert_job(db_session_factory, video_path=tmp_path / "a.mp4", request=_request())
    job_b = await _insert_job(db_session_factory, video_path=tmp_path / "b.mp4", request=_request())

    await runner.enqueue(job_a, _TENANT)
    await runner.enqueue(job_b, _TENANT)
    await asyncio.wait_for(
        asyncio.gather(runner.wait_for(job_a), runner.wait_for(job_b)), timeout=60
    )

    assert provider.peak_inflight == 1, (
        f"max_concurrent_jobs=1 but {provider.peak_inflight} jobs ran the vision stage at once"
    )
    # Pin the knob itself too, so raising the default cannot silently make the
    # behavioural assertion above vacuous.
    assert runner._job_slots._value == 1


# --------------------------------------------------------------------------
# 3. Pod disk must be reclaimed on BOTH terminal paths.
# --------------------------------------------------------------------------


async def test_job_scratch_is_reclaimed_on_success(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """The source video and extracted frames must not outlive a completed job."""
    runner = _runner(tmp_path, vision_provider=_RecordingVisionProvider(), frame_count=4)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "completed"

    source_dir = Path(runner.dependencies.settings.storage.upload_dir) / job_id
    job_dir = runner.dependencies.storage.get_job_dir(job_id)
    assert not source_dir.exists(), "downloaded source video still on pod disk"
    assert not (job_dir / "frames").exists(), "extracted frames still on pod disk"
    assert not (job_dir / "audio.wav").exists(), "extracted audio still on pod disk"
    # Artifacts are the deliverable and are served from this dir — keep them.
    assert (job_dir / "walkthrough.md").exists()


async def test_job_scratch_is_reclaimed_on_failure(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """A failed job leaks exactly the same bytes — reclaim it too."""
    runner = _runner(tmp_path, vision_provider=_DiesAfterNFramesProvider(limit=0), frame_count=4)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "failed"

    source_dir = Path(runner.dependencies.settings.storage.upload_dir) / job_id
    job_dir = runner.dependencies.storage.get_job_dir(job_id)
    assert not source_dir.exists(), "downloaded source video leaked on the failure path"
    assert not (job_dir / "frames").exists(), "extracted frames leaked on the failure path"


# --------------------------------------------------------------------------
# 4. Edge behaviour of the new guards.
# --------------------------------------------------------------------------


async def test_deadline_zero_disables_the_ceiling(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """``job_deadline_seconds=0`` is the documented escape hatch — no ceiling."""

    def _mutate(settings: Any) -> None:
        settings.pipeline.job_deadline_seconds = 0

    runner = _runner(
        tmp_path,
        vision_provider=_RecordingVisionProvider(),
        frame_count=2,
        settings_mutator=_mutate,
    )
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "completed"


async def test_a_downstream_timeout_is_not_reported_as_the_deadline(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """A ``TimeoutError`` from httpx/ffmpeg must keep its own message.

    ``asyncio.timeout`` re-raises an inner ``TimeoutError`` unchanged, so the
    except site cannot tell it apart by type — only ``timeout_cm.expired()`` can.
    Without that discriminator every downstream read timeout would be mislabelled
    "exceeded the wall-clock deadline", pointing operators at the wrong knob.
    """

    class _DownstreamTimeoutProvider(_RecordingVisionProvider):
        async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
            raise TimeoutError("ReadTimeout talking to worker_management")

    def _mutate(settings: Any) -> None:
        settings.pipeline.job_deadline_seconds = 600

    runner = _runner(
        tmp_path,
        vision_provider=_DownstreamTimeoutProvider(),
        frame_count=2,
        settings_mutator=_mutate,
    )
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "failed"
    assert "ReadTimeout" in (job.error_message or "")
    assert "deadline" not in (job.error_message or "").lower()


async def test_reclaim_survives_a_storage_that_cannot_resolve_the_job_dir(
    tmp_path: Path, db_session_factory, _patched_session_maker: None
) -> None:
    """Reclaim is best-effort: it must never change a job's outcome.

    The source video under ``upload_dir`` is still reclaimed — that path does not
    go through the storage service.
    """
    runner = _runner(tmp_path, vision_provider=_RecordingVisionProvider(), frame_count=2)
    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    def _boom(_job_id: str) -> Path:
        raise OSError("read-only filesystem")

    runner.dependencies.storage.get_job_dir = _boom  # type: ignore[method-assign]
    source_dir = Path(runner.dependencies.settings.storage.upload_dir) / job_id
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "video.mp4").write_bytes(b"x")

    runner._reclaim_job_scratch(job_id)

    assert not source_dir.exists()


async def test_a_failed_checkpoint_does_not_fail_the_run(
    tmp_path: Path, db_session_factory, _patched_session_maker: None, monkeypatch
) -> None:
    """A partial-results checkpoint is an optimisation, never a new failure mode."""
    runner = _runner(tmp_path, vision_provider=_RecordingVisionProvider(), frame_count=2)

    from src.repository.job_repository import JobRepository as _Repo

    real_update = _Repo.update_fields
    calls = {"n": 0}

    async def _flaky(self, job_id: str, *, tenant_id: str, **fields: Any) -> None:
        if "frame_analyses" in fields and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("checkpoint write blew up")
        await real_update(self, job_id, tenant_id=tenant_id, **fields)

    monkeypatch.setattr(_Repo, "update_fields", _flaky)

    job_id = await _insert_job(
        db_session_factory, video_path=tmp_path / "in.mp4", request=_request()
    )
    await runner.enqueue(job_id, _TENANT)
    await asyncio.wait_for(runner.wait_for(job_id), timeout=60)

    job = await _load(db_session_factory, job_id)
    assert job is not None and job.status == "completed"
    assert calls["n"] == 1
    assert job.frame_analyses is not None
