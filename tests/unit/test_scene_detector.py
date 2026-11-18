"""SceneDetector tests against synthesized frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.domain.analysis import ExtractedFrame
from src.services.scene_detector import SceneDetector


def _write_color_frame(path: Path, color: tuple[int, int, int]) -> None:
    img = np.full((32, 32, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_detects_scene_changes(tmp_path: Path) -> None:
    frames_meta = [
        (0, 0.0, (10, 10, 10)),
        (1, 3.0, (10, 10, 10)),
        (2, 6.0, (250, 20, 20)),  # scene change → red
        (3, 9.0, (250, 20, 20)),
        (4, 12.0, (20, 250, 20)),  # scene change → green
    ]
    extracted: list[ExtractedFrame] = []
    for idx, ts, color in frames_meta:
        p = tmp_path / f"f{idx}.jpg"
        _write_color_frame(p, color)
        extracted.append(ExtractedFrame(index=idx, timestamp=ts, path=p))

    detector = SceneDetector(threshold=0.5)
    boundaries = detector.detect(extracted)
    assert boundaries[0] == 0.0
    assert 6.0 in boundaries
    assert 12.0 in boundaries


def test_single_frame_returns_single_boundary(tmp_path: Path) -> None:
    p = tmp_path / "only.jpg"
    _write_color_frame(p, (100, 100, 100))
    detector = SceneDetector()
    result = detector.detect([ExtractedFrame(index=0, timestamp=2.0, path=p)])
    assert result == [2.0]


def test_skips_unreadable_frame(tmp_path: Path) -> None:
    good = tmp_path / "a.jpg"
    _write_color_frame(good, (50, 50, 50))
    missing = tmp_path / "missing.jpg"  # never written
    extracted = [
        ExtractedFrame(index=0, timestamp=0.0, path=good),
        ExtractedFrame(index=1, timestamp=3.0, path=missing),
    ]
    detector = SceneDetector()
    # missing frame should not crash
    assert detector.detect(extracted)[0] == 0.0
