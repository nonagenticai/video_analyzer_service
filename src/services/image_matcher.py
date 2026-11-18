"""pHash-based reference-image ↔ frame matcher."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from src.domain.analysis import ExtractedFrame

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageMatchResult:
    """Result of matching a reference image against a list of extracted frames."""

    name: str
    matched_frame: Path | None
    timestamp: float | None
    distance: int


def _phash(path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except OSError, ValueError:
        logger.warning("Failed to compute pHash for %s", path)
        return None


class ImageMatcher:
    """Compute pHash distances between reference images and extracted frames."""

    def find_match(
        self,
        *,
        name: str,
        reference_path: Path,
        frames: list[ExtractedFrame],
        time_window: tuple[float, float] | None = None,
    ) -> ImageMatchResult:
        """Return the best (smallest Hamming distance) frame matching the reference."""
        target = _phash(reference_path)
        if target is None:
            return ImageMatchResult(name=name, matched_frame=None, timestamp=None, distance=64)

        best: ImageMatchResult = ImageMatchResult(
            name=name, matched_frame=None, timestamp=None, distance=64
        )
        for frame in frames:
            if time_window is not None:
                lo, hi = time_window
                if frame.timestamp < lo or frame.timestamp > hi:
                    continue
            frame_hash = _phash(frame.path)
            if frame_hash is None:
                continue
            distance = int(frame_hash - target)
            if distance < best.distance:
                best = ImageMatchResult(
                    name=name,
                    matched_frame=frame.path,
                    timestamp=frame.timestamp,
                    distance=distance,
                )
        return best
