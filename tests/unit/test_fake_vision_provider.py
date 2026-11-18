"""FakeVisionProvider output sanity tests."""

from __future__ import annotations

from pathlib import Path

from src.services.vision.base import FrameInput
from src.services.vision.fake_provider import FakeVisionProvider


async def test_analyze_frames_returns_one_per_input() -> None:
    provider = FakeVisionProvider(cost_per_frame=0.02)
    inputs = [
        FrameInput(index=0, timestamp=0.0, path=Path("/tmp/a.jpg")),
        FrameInput(index=1, timestamp=3.0, path=Path("/tmp/b.jpg")),
    ]
    results = await provider.analyze_frames(inputs)
    assert len(results) == 2
    assert results[0].description.startswith("Frame 0")
    assert results[1].cost_usd == 0.02


async def test_synthesize_includes_transcript_snippet() -> None:
    provider = FakeVisionProvider()
    results = await provider.analyze_frames(
        [FrameInput(index=0, timestamp=0.0, path=Path("/tmp/a.jpg"))]
    )
    synthesis = await provider.synthesize(
        frame_results=results, transcript_text="This is a demo of the feature."
    )
    assert "This is a demo of the feature." in synthesis.walkthrough
    assert "Frame 0" in synthesis.ui_notes


async def test_synthesize_handles_no_transcript() -> None:
    provider = FakeVisionProvider()
    synthesis = await provider.synthesize(frame_results=[], transcript_text=None)
    assert "0 frames" in synthesis.walkthrough
    assert synthesis.ui_notes == ""
