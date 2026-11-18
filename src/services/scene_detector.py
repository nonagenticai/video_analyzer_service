"""Histogram-based scene boundary detection (OpenCV)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.domain.analysis import ExtractedFrame

logger = logging.getLogger(__name__)


def _load_gray(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def _histogram(hsv: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1],
        mask=None,
        histSize=[30, 32],
        ranges=[0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


class SceneDetector:
    """Detect scene boundaries by chi-square distance between frame histograms."""

    def __init__(self, *, threshold: float = 0.6) -> None:
        self._threshold = threshold

    def detect(self, frames: list[ExtractedFrame]) -> list[float]:
        """Return a sorted list of timestamps where a scene change starts."""
        if len(frames) < 2:
            return [f.timestamp for f in frames]

        boundaries: list[float] = [frames[0].timestamp]
        prev_hist: np.ndarray | None = None

        for frame in frames:
            hsv = _load_gray(frame.path)
            if hsv is None:
                continue
            hist = _histogram(hsv)
            if prev_hist is not None:
                distance = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR_ALT))
                if distance > self._threshold:
                    boundaries.append(frame.timestamp)
            prev_hist = hist

        return boundaries
