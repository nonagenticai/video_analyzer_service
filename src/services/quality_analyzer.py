"""Visual quality metrics: sharpness, brightness, motion, dominant colors."""

from __future__ import annotations

import logging
from typing import Any, cast

import cv2
import numpy as np

from src.domain.analysis import ExtractedFrame, MotionSample, QualityMetrics, VisualSample

logger = logging.getLogger(__name__)

# Per-frame dominant colour is the only genuinely NEW computation on the frame
# loop, so it is the only one that can regress latency — and the only one that can
# manufacture a false signal. Two decisions, both measured:
#
# 1. DOWNSCALE FIRST. Dominant colour is a *global* statistic; spatial detail
#    contributes nothing. INTER_AREA to a 64px longest edge gives ~4k pixels
#    regardless of source resolution (a 1920x1080 frame: 2.07M → 3.8k, ~540x less
#    work). INTER_AREA is an area average, so it preserves the colour MASS
#    distribution the statistic measures instead of point-sampling it.
#
# 2. MODAL QUANTISED BIN, NOT k-MEANS. `_kmeans_palette` uses KMEANS_PP_CENTERS,
#    whose seeding is RANDOM — measured on a byte-identical frame, repeated calls
#    returned dominant colours up to 22.5 L2 apart. Per frame that is fabricated
#    colour drift, indistinguishable from a real subtle grade shift, which is
#    exactly the failure this field exists to avoid. Quantising to 4 bits/channel
#    and averaging the true pixels of the modal bin is DETERMINISTIC (measured 0.00
#    L2 across repeats), marginally cheaper, and strictly more faithful: on a
#    tri-modal 40/35/25 red/green/blue frame it returns the red, where a per-channel
#    median returns [0,0,0] — a colour present in zero pixels. The aggregate
#    `dominant_colors` palette keeps k-means: it is computed once, and a
#    whole-video palette is not differenced against anything.
_COLOR_MAX_DIM = 64
_COLOR_BITS = 4


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean()) / 255.0


def _kmeans_palette(image: np.ndarray, k: int = 5) -> list[str]:
    data = image.reshape(-1, 3).astype(np.float32)
    if data.shape[0] == 0:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        data,
        k,
        cast("Any", None),
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten().astype(np.int32), minlength=k)
    order = np.argsort(-counts)
    palette: list[str] = []
    for idx in order:
        b, g, r = centers[idx].astype(int)
        palette.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")
    return palette


def _downscale_for_color(image: np.ndarray) -> np.ndarray:
    """Shrink to a ``_COLOR_MAX_DIM`` longest edge, preserving aspect ratio."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= _COLOR_MAX_DIM or longest == 0:
        return image
    scale = _COLOR_MAX_DIM / longest
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _dominant_rgb(image: np.ndarray, bits: int = _COLOR_BITS) -> list[int]:
    """``[r, g, b]`` 0..255 dominant colour of ``image`` — deterministic.

    Downscales, buckets every pixel into a ``bits``-per-channel colour cube,
    finds the fullest bucket, and returns the MEAN of the real pixels in it (so
    the answer has sub-bucket precision and is always a colour the frame
    actually contains). Same input ⇒ same output, always. Returns ``[]`` only
    for a zero-pixel image.
    """
    pixels = _downscale_for_color(image).reshape(-1, 3)
    if pixels.shape[0] == 0:
        return []
    quantised = (pixels >> (8 - bits)).astype(np.int32)
    # Pack the 3 quantised channels into one bucket id so a single bincount ranks them.
    keys = (quantised[:, 0] << (2 * bits)) | (quantised[:, 1] << bits) | quantised[:, 2]
    modal_key = int(np.bincount(keys, minlength=1 << (3 * bits)).argmax())
    b, g, r = pixels[keys == modal_key].mean(axis=0)
    return [int(np.clip(round(float(c)), 0, 255)) for c in (r, g, b)]


def _motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) / 255.0


def _overall_score(sharpness: float, brightness: float, motion: float) -> float:
    sharpness_norm = min(sharpness / 1500.0, 1.0)
    brightness_norm = 1.0 - abs(brightness - 0.5) * 2.0
    brightness_norm = max(min(brightness_norm, 1.0), 0.0)
    motion_norm = max(0.0, min(1.0, 1.0 - abs(motion - 0.1)))
    return round((sharpness_norm * 0.6 + brightness_norm * 0.3 + motion_norm * 0.1), 3)


class QualityAnalyzer:
    """Compute aggregate quality metrics across a set of frames."""

    def analyze(
        self, frames: list[ExtractedFrame], *, frames_planned: int | None = None
    ) -> QualityMetrics:
        """Aggregate the frames into ``QualityMetrics``.

        ``frames_planned`` is ``FrameExtraction.planned`` — how many samples the
        extractor SCHEDULED, which is not knowable from ``frames`` (a dropped
        frame leaves nothing behind). It is recorded verbatim so
        ``verifier.check_quality`` can tell "few frames because the clip is
        short" from "few frames because most of them were lost".

        ``None`` means the caller does not know the plan (a direct/ad-hoc call).
        The delivered count is then used as the plan, which still measures the
        loss this function CAN see — frames handed over but undecodable — and
        never invents a shortfall that was not observed.
        """
        planned = len(frames) if frames_planned is None else frames_planned
        if not frames:
            return QualityMetrics(
                mean_sharpness=0.0,
                mean_brightness=0.0,
                motion_score=0.0,
                dominant_colors=[],
                overall_score=0.0,
                frames_planned=planned,
                frames_analyzed=0,
            )

        sharpness_values: list[float] = []
        brightness_values: list[float] = []
        motion_values: list[float] = []
        # The per-sample motion curve, kept alongside the values we average. A frame
        # that fails to decode is skipped entirely, so a pair may straddle a gap —
        # the timestamp stamped is always the LATER frame's, i.e. when the change
        # has landed on screen.
        motion_timeline: list[MotionSample] = []
        # The per-frame visual curve. Unlike motion this is per-FRAME, not
        # per-pair, so it is appended only for frames that actually decoded and
        # carries that frame's OWN timestamp + index — a skipped frame leaves a
        # visible gap in the curve rather than shifting every later sample.
        visual_timeline: list[VisualSample] = []
        palette: list[str] = []
        prev_gray: np.ndarray | None = None

        for frame in frames:
            image = cv2.imread(str(frame.path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = _laplacian_variance(gray)
            brightness = _mean_brightness(gray)
            sharpness_values.append(sharpness)
            brightness_values.append(brightness)
            visual_timeline.append(
                VisualSample(
                    t=round(float(frame.timestamp), 3),
                    index=frame.index,
                    sharpness=round(sharpness, 6),
                    brightness=round(brightness, 6),
                    dominant_rgb=_dominant_rgb(image),
                )
            )
            if prev_gray is not None:
                motion = _motion_score(prev_gray, gray)
                motion_values.append(motion)
                motion_timeline.append(
                    MotionSample(t=round(float(frame.timestamp), 3), motion=round(motion, 6))
                )
            prev_gray = gray

            if not palette:
                palette = _kmeans_palette(image)

        mean_sharp = float(np.mean(sharpness_values)) if sharpness_values else 0.0
        mean_bright = float(np.mean(brightness_values)) if brightness_values else 0.0
        mean_motion = float(np.mean(motion_values)) if motion_values else 0.0

        return QualityMetrics(
            mean_sharpness=mean_sharp,
            mean_brightness=mean_bright,
            motion_score=mean_motion,
            dominant_colors=palette,
            overall_score=_overall_score(mean_sharp, mean_bright, mean_motion),
            motion_timeline=motion_timeline,
            visual_timeline=visual_timeline,
            frames_planned=planned,
            # NOT ``len(frames)``: the loop above skips a frame cv2 cannot decode,
            # so the delivered count would overstate the evidence the means rest
            # on. This is the sample size, by construction.
            frames_analyzed=len(sharpness_values),
        )
