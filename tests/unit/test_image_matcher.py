"""ImageMatcher pHash tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.domain.analysis import ExtractedFrame
from src.services.image_matcher import ImageMatcher, _phash


def _save_pattern(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "PNG")


def test_phash_returns_none_for_missing(tmp_path: Path) -> None:
    assert _phash(tmp_path / "nope.png") is None


def test_match_picks_closest_frame(tmp_path: Path) -> None:
    _save_pattern(tmp_path / "ref.png", seed=42)
    _save_pattern(tmp_path / "frame_a.png", seed=42)  # exact match
    _save_pattern(tmp_path / "frame_b.png", seed=999)  # different

    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=tmp_path / "frame_b.png"),
        ExtractedFrame(index=1, timestamp=2.0, path=tmp_path / "frame_a.png"),
    ]
    matcher = ImageMatcher()
    result = matcher.find_match(name="ref", reference_path=tmp_path / "ref.png", frames=frames)
    assert result.matched_frame == frames[1].path
    assert result.distance == 0


def test_match_respects_time_window(tmp_path: Path) -> None:
    _save_pattern(tmp_path / "ref.png", seed=10)
    _save_pattern(tmp_path / "early.png", seed=10)  # exact match but outside window
    _save_pattern(tmp_path / "in_window.png", seed=10)

    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=tmp_path / "early.png"),
        ExtractedFrame(index=1, timestamp=10.0, path=tmp_path / "in_window.png"),
    ]
    matcher = ImageMatcher()
    result = matcher.find_match(
        name="ref",
        reference_path=tmp_path / "ref.png",
        frames=frames,
        time_window=(8.0, 12.0),
    )
    assert result.matched_frame == frames[1].path


def test_match_returns_default_when_reference_unreadable(tmp_path: Path) -> None:
    matcher = ImageMatcher()
    result = matcher.find_match(name="ref", reference_path=tmp_path / "missing.png", frames=[])
    assert result.matched_frame is None
    assert result.distance == 64


def test_match_skips_unreadable_frames(tmp_path: Path) -> None:
    _save_pattern(tmp_path / "ref.png", seed=5)
    _save_pattern(tmp_path / "good.png", seed=5)
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=tmp_path / "missing.png"),
        ExtractedFrame(index=1, timestamp=2.0, path=tmp_path / "good.png"),
    ]
    matcher = ImageMatcher()
    result = matcher.find_match(name="ref", reference_path=tmp_path / "ref.png", frames=frames)
    assert result.matched_frame == frames[1].path
    assert result.distance == 0
