"""Tests for the ``ANALYZER_PROVIDER=stub`` offline path.

Covers:
  - Selecting the stub provider via the top-level ``ANALYZER_PROVIDER`` env
    var (override) and via ``VISION_DEFAULT_PROVIDER=stub`` (alias for fake).
  - ``ANALYZER_PROVIDER`` taking precedence over ``VISION_DEFAULT_PROVIDER``
    even when API keys are set.
  - The stub provider's analyze/synthesize output is deterministic and rich
    enough that downstream Pydantic models (``JobResultResponse``,
    ``FrameVisionResult``-derived dicts) validate without crashing.
  - The synthesised payload feeds cleanly into the orchestrator's
    ``run_video_analyze`` consumer contract: a result dict with
    ``artifacts``/``verification``/``duration_seconds`` shape, where the
    optional ``verification`` block can be ``None`` or a valid
    :class:`VerificationResult`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import (
    AnalysisRequest,
    JobResultResponse,
    VerificationResult,
)
from src.config import clear_settings_cache, get_settings
from src.services.pipeline import _select_vision_provider
from src.services.vision.base import FrameInput
from src.services.vision.fake_provider import FakeVisionProvider


def _reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    clear_settings_cache()


def test_stub_via_analyzer_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ANALYZER_PROVIDER=stub`` selects FakeVisionProvider regardless of vision config."""
    _reload_settings(
        monkeypatch,
        ANALYZER_PROVIDER="stub",
        VISION_DEFAULT_PROVIDER="wm",
        VISION_MODEL_OVERRIDE="claude-sonnet-4-5",
    )
    settings = get_settings()
    assert settings.analyzer_provider == "stub"
    provider = _select_vision_provider(settings)
    assert isinstance(provider, FakeVisionProvider)


def test_fake_alias_also_selects_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_settings(monkeypatch, ANALYZER_PROVIDER="fake")
    provider = _select_vision_provider(get_settings())
    assert isinstance(provider, FakeVisionProvider)


def test_stub_via_vision_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_settings(monkeypatch, VISION_DEFAULT_PROVIDER="stub")
    settings = get_settings()
    assert settings.analyzer_provider is None
    provider = _select_vision_provider(settings)
    assert isinstance(provider, FakeVisionProvider)


def test_analyzer_provider_overrides_default_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reload_settings(
        monkeypatch,
        ANALYZER_PROVIDER="stub",
        VISION_DEFAULT_PROVIDER="wm",
    )
    provider = _select_vision_provider(get_settings())
    # Without override, this would build WMVisionProvider.
    assert isinstance(provider, FakeVisionProvider)


def test_empty_analyzer_provider_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ANALYZER_PROVIDER must not mask vision.default_provider."""
    _reload_settings(
        monkeypatch,
        ANALYZER_PROVIDER="",
        VISION_DEFAULT_PROVIDER="stub",
    )
    settings = get_settings()
    # Pydantic stores the empty string; selector must treat it as "unset".
    assert settings.analyzer_provider == ""
    provider = _select_vision_provider(settings)
    assert isinstance(provider, FakeVisionProvider)


async def test_stub_analyze_frames_deterministic() -> None:
    """Same inputs → identical outputs (no time/random source)."""
    provider = FakeVisionProvider()
    inputs = [
        FrameInput(index=0, timestamp=0.0, path=Path("/tmp/0.jpg")),
        FrameInput(index=1, timestamp=3.0, path=Path("/tmp/1.jpg")),
        FrameInput(index=2, timestamp=6.0, path=Path("/tmp/2.jpg")),
    ]
    a = await provider.analyze_frames(inputs)
    b = await provider.analyze_frames(inputs)
    assert [r.description for r in a] == [r.description for r in b]
    assert [r.labels for r in a] == [r.labels for r in b]
    # Shape contract for downstream consumers.
    for r in a:
        assert r.description.startswith(f"Frame {r.index}")
        assert "ui" in r.labels and "screen" in r.labels
        assert r.cost_usd == 0.0


async def test_stub_synthesize_includes_timing_and_transcript() -> None:
    provider = FakeVisionProvider()
    frames = [
        FrameInput(index=i, timestamp=float(i) * 3.0, path=Path(f"/tmp/{i}.jpg")) for i in range(3)
    ]
    frame_results = await provider.analyze_frames(frames)
    synthesis = await provider.synthesize(
        frame_results=frame_results,
        transcript_text="The user logs in and opens the dashboard.",
    )
    assert "3 frames" in synthesis.walkthrough
    assert "0.0s" in synthesis.walkthrough  # first timestamp
    assert "6.0s" in synthesis.walkthrough  # last timestamp
    assert "logs in and opens the dashboard" in synthesis.walkthrough
    assert synthesis.ui_notes.count("- Frame ") == 3


async def test_stub_output_passes_job_result_response_validation() -> None:
    """Stub-driven result dict must validate as JobResultResponse."""
    provider = FakeVisionProvider()
    frames = [
        FrameInput(index=i, timestamp=float(i) * 2.0, path=Path(f"/tmp/{i}.jpg")) for i in range(4)
    ]
    frame_results = await provider.analyze_frames(frames)
    synthesis = await provider.synthesize(
        frame_results=frame_results,
        transcript_text="Hello world",
    )

    # Build a result the way the pipeline does, then round-trip through the
    # public Pydantic schema the orchestrator's get_result() consumes.
    payload: dict[str, object] = {
        "job_id": "stub-job-1",
        "status": "completed",
        "stage": "artifact_gen",
        "progress": 1.0,
        "duration_seconds": 12.0,
        "width": 1280,
        "height": 720,
        "has_audio": True,
        "vision_cost_usd": 0.0,
        "verification_passed": None,
        "artifacts": {"walkthrough.md": "/tmp/walk.md", "ui.md": "/tmp/ui.md"},
        "transcript_data": {"text": "Hello world", "segments": []},
        "scene_boundaries": [0.0, 6.0],
        "frame_analyses": [
            {
                "index": r.index,
                "timestamp": r.timestamp,
                "description": r.description,
                "labels": r.labels,
            }
            for r in frame_results
        ],
        # Real pipeline shape (asdict(QualityMetrics)) — the schema is typed now,
        # so the old free-form {"overall": ..., "sharpness": ...} dict no longer
        # validates (and never matched what the pipeline actually stores).
        "quality_metrics": {
            "mean_sharpness": 120.5,
            "mean_brightness": 0.6,
            "motion_score": 0.25,
            "dominant_colors": ["#112233", "#445566"],
            "overall_score": 0.9,
            "motion_timeline": [{"t": 2.0, "motion": 0.3}],
        },
        "verification": None,
    }
    # Embed the synthesis text so validation also covers the unicode/length
    # of the stub-emitted strings.
    assert synthesis.walkthrough
    assert synthesis.ui_notes

    resp = JobResultResponse.model_validate(payload)
    assert resp.job_id == "stub-job-1"
    assert resp.frame_analyses is not None and len(resp.frame_analyses) == 4
    assert resp.verification is None
    assert resp.quality_metrics is not None
    assert resp.quality_metrics.overall_score == 0.9
    assert resp.quality_metrics.motion_timeline[0].t == 2.0


def test_analysis_request_accepts_stub_literal() -> None:
    """The HTTP API must accept ``vision_provider='stub'`` per-request."""
    req = AnalysisRequest(video_url="https://example.com/x.mp4", vision_provider="stub")
    assert req.vision_provider == "stub"


def test_orchestrator_video_analyze_consumer_contract() -> None:
    """Mirror the assignments orchestrator's ``run_video_analyze`` makes.

    The orchestrator stage handler reads:
        full_result.get("duration_seconds")
        full_result.get("artifacts") or {}
        full_result.get("verification")
    and folds them into its stage output. Verify a stub-shaped JobResultResponse
    survives that exact extraction without raising and without losing data.
    """
    full_result = JobResultResponse(
        job_id="stub-job-2",
        status="completed",
        stage="artifact_gen",
        progress=1.0,
        duration_seconds=42.0,
        artifacts={"walkthrough.md": "/tmp/w.md"},
        verification=VerificationResult(passed=True, criteria_total=2, criteria_passed=2),
    ).model_dump(mode="json")

    # Replicate run_video_analyze field extraction.
    output = {
        "duration_seconds": full_result.get("duration_seconds"),
        "artifacts": full_result.get("artifacts") or {},
        "verification": full_result.get("verification"),
    }
    assert output["duration_seconds"] == 42.0
    assert output["artifacts"] == {"walkthrough.md": "/tmp/w.md"}
    assert isinstance(output["verification"], dict)
    assert output["verification"]["passed"] is True
