"""Quality metrics tests on synthesized frames."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from src.domain.analysis import ExtractedFrame
from src.services.quality_analyzer import (
    QualityAnalyzer,
    _dominant_rgb,
    _downscale_for_color,
)


def _sharp_frame(path: Path) -> None:
    img = cast("Any", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(path), img)


def _flat_frame(path: Path, value: int = 128) -> None:
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_empty_frames_returns_zero_metrics() -> None:
    metrics = QualityAnalyzer().analyze([])
    assert metrics.mean_sharpness == 0.0
    assert metrics.dominant_colors == []


def test_sharp_frames_scored_higher(tmp_path: Path) -> None:
    sharp_path = tmp_path / "sharp.jpg"
    flat_path = tmp_path / "flat.jpg"
    _sharp_frame(sharp_path)
    _flat_frame(flat_path, value=128)

    frames_sharp = [ExtractedFrame(index=0, timestamp=0.0, path=sharp_path)]
    frames_flat = [ExtractedFrame(index=0, timestamp=0.0, path=flat_path)]

    sharp = QualityAnalyzer().analyze(frames_sharp)
    flat = QualityAnalyzer().analyze(frames_flat)
    assert sharp.mean_sharpness > flat.mean_sharpness
    assert len(sharp.dominant_colors) >= 1


def test_motion_score_computed(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    _flat_frame(p1, value=0)
    _flat_frame(p2, value=255)
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=p1),
        ExtractedFrame(index=1, timestamp=1.0, path=p2),
    ]
    metrics = QualityAnalyzer().analyze(frames)
    assert metrics.motion_score > 0.5


def test_skips_unreadable_frames(tmp_path: Path) -> None:
    good = tmp_path / "g.jpg"
    _flat_frame(good, 80)
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=tmp_path / "missing.jpg"),
        ExtractedFrame(index=1, timestamp=3.0, path=good),
    ]
    metrics = QualityAnalyzer().analyze(frames)
    assert metrics.mean_brightness > 0
    # Only one frame decoded ⇒ no PAIR ⇒ no motion sample. The timeline must not
    # invent a sample against a frame that never loaded.
    assert metrics.motion_timeline == []


def test_motion_timeline_marks_when_the_action_happens(tmp_path: Path) -> None:
    """The timeline localises motion in TIME — the whole point of keeping the curve.

    Three flat frames: 0→1 is static (no change), 1→2 flips black→white (max change).
    The peak sample must carry the LATER frame's timestamp, so a consumer that
    argmaxes it seeks to the moment the change has landed.
    """
    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    _flat_frame(paths[0], value=0)
    _flat_frame(paths[1], value=0)
    _flat_frame(paths[2], value=255)
    frames = [
        ExtractedFrame(index=i, timestamp=float(i) * 2.0, path=p) for i, p in enumerate(paths)
    ]

    metrics = QualityAnalyzer().analyze(frames)

    # N frames ⇒ N-1 pairs.
    assert [s.t for s in metrics.motion_timeline] == [2.0, 4.0]
    assert metrics.motion_timeline[0].motion < 0.01  # static pair
    assert metrics.motion_timeline[1].motion > 0.9  # the flip
    peak = max(metrics.motion_timeline, key=lambda s: s.motion)
    assert peak.t == 4.0
    # The aggregate stays the MEAN of exactly these samples — the curve is additive,
    # it does not change the existing scalar.
    expected_mean = sum(s.motion for s in metrics.motion_timeline) / 2
    assert metrics.motion_score == pytest.approx(expected_mean, abs=1e-5)


def test_empty_frames_have_empty_timeline() -> None:
    assert QualityAnalyzer().analyze([]).motion_timeline == []


def test_quality_metrics_serialise_to_plain_json(tmp_path: Path) -> None:
    """`pipeline._run_quality_stage` persists `asdict(quality)` into a JSON column,
    so the nested MotionSample must survive as plain dicts."""
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _flat_frame(p1, value=0)
    _flat_frame(p2, value=255)
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=p1),
        ExtractedFrame(index=1, timestamp=1.5, path=p2),
    ]
    payload = asdict(QualityAnalyzer().analyze(frames))
    assert payload["motion_timeline"][0]["t"] == 1.5
    assert isinstance(payload["motion_timeline"][0]["motion"], float)
    json.dumps(payload)  # must not raise


# ── per-frame visual timeline ────────────────────────────────────────────────
# The aggregates above are whole-video means, so every shot of a film reads
# identically and adjacent-shot deltas are 0.0 BY CONSTRUCTION. These tests
# prove the per-frame curve actually DISCRIMINATES, and that a flat film still
# reads flat (so a consumer can tell "no signal" from "signal").


def _shot_frame(path: Path, rgb: tuple[int, int, int], *, blur: bool, seed: int) -> None:
    """A 128x128 frame of solid ``rgb`` + noise, optionally blurred (= soft/low sharpness)."""
    rng = np.random.default_rng(seed)
    base = np.zeros((128, 128, 3), np.uint8)
    base[:, :] = (rgb[2], rgb[1], rgb[0])  # cv2 is BGR
    noise = rng.integers(-40, 40, (128, 128, 3))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (15, 15), 6)
    cv2.imwrite(str(path), img)  # PNG: lossless, so colour assertions are exact


def _rgb_l2(a: list[int], b: list[int]) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


def test_visual_timeline_discriminates_between_shots(tmp_path: Path) -> None:
    """Three genuinely different shots must produce three different feature triples.

    Shot A: dark blue + sharp. Shot B: bright orange + blurred. Shot C: mid green.
    Within a shot the per-frame values must be near-identical; ACROSS a shot
    boundary every component must move. That cross-boundary movement is the
    entire signal a continuity/cohesion gate consumes.
    """
    specs = [
        ((10, 40, 120), False),  # A
        ((10, 40, 120), False),
        ((235, 150, 40), True),  # B
        ((235, 150, 40), True),
        ((90, 140, 90), False),  # C
        ((90, 140, 90), False),
    ]
    frames = []
    for i, (rgb, blur) in enumerate(specs):
        p = tmp_path / f"{i}.png"
        _shot_frame(p, rgb, blur=blur, seed=i)
        frames.append(ExtractedFrame(index=i, timestamp=float(i) * 2.0, path=p))

    tl = QualityAnalyzer().analyze(frames).visual_timeline
    assert len(tl) == 6
    assert [s.t for s in tl] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert [s.index for s in tl] == [0, 1, 2, 3, 4, 5]

    # Dominant colour tracks the shot's actual base colour (the ±40 pixel noise
    # widens the modal bucket, so allow a loose L2 rather than an exact match).
    for s, (rgb, _blur) in zip(tl, specs, strict=True):
        assert _rgb_l2(s.dominant_rgb, list(rgb)) < 28, (s.dominant_rgb, rgb)

    # Brightness ordering: B (bright orange) > C (mid green) > A (dark blue).
    assert tl[2].brightness > tl[4].brightness > tl[0].brightness
    # Sharpness: the blurred shot is an order of magnitude softer.
    assert tl[2].sharpness < tl[0].sharpness / 10

    # WITHIN a shot: deltas are small. ACROSS a boundary: every component moves.
    assert abs(tl[1].brightness - tl[0].brightness) < 0.02
    assert _rgb_l2(tl[1].dominant_rgb, tl[0].dominant_rgb) < 12
    for i in (1, 3):  # boundary pairs 1→2 (A→B) and 3→4 (B→C)
        assert _rgb_l2(tl[i + 1].dominant_rgb, tl[i].dominant_rgb) > 60
        assert abs(tl[i + 1].brightness - tl[i].brightness) > 0.1
        assert abs(tl[i + 1].sharpness - tl[i].sharpness) > 100


def test_visual_timeline_reads_degenerate_when_the_film_is_flat(tmp_path: Path) -> None:
    """Near-identical frames ⇒ near-identical samples, so "no signal" stays legible.

    This is the control for the discrimination test: the curve must NOT
    manufacture variation where the source has none.
    """
    paths = []
    for i in range(5):
        p = tmp_path / f"flat{i}.png"
        _shot_frame(p, (90, 140, 90), blur=False, seed=7)  # identical seed ⇒ identical pixels
        paths.append(p)
    frames = [
        ExtractedFrame(index=i, timestamp=float(i) * 3.0, path=p) for i, p in enumerate(paths)
    ]

    tl = QualityAnalyzer().analyze(frames).visual_timeline
    sharp = [s.sharpness for s in tl]
    bright = [s.brightness for s in tl]
    assert max(sharp) - min(sharp) < 1e-6
    assert max(bright) - min(bright) < 1e-6
    # EXACTLY zero drift. `_dominant_rgb` is deterministic by construction (modal
    # quantised bin, no random seeding), so identical pixels give identical colour.
    # This is the guard that stopped a k-means implementation shipping: measured on
    # this same frame, KMEANS_PP_CENTERS returned colours up to 22.5 L2 apart across
    # repeated calls — fabricated drift a consumer would read as a real grade shift.
    for a, b in pairwise(tl):
        assert _rgb_l2(a.dominant_rgb, b.dominant_rgb) == 0.0


def test_dominant_rgb_is_deterministic_and_a_colour_the_frame_contains(
    tmp_path: Path,
) -> None:
    """Repeat calls must agree exactly, and the answer must be a REAL frame colour.

    The tri-modal case is the one that separates the modal-bin statistic from the
    cheap alternatives: 40% red / 35% green / 25% blue must read as red. A
    per-channel median returns [0, 0, 0] here — a colour present in zero pixels.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    img[:40] = (0, 0, 255)  # BGR red — 40%
    img[40:75] = (0, 255, 0)  # green — 35%
    img[75:] = (255, 0, 0)  # blue — 25%
    p = tmp_path / "trimodal.png"
    cv2.imwrite(str(p), img)
    frames = [ExtractedFrame(index=0, timestamp=0.0, path=p)]

    runs = [QualityAnalyzer().analyze(frames).visual_timeline[0].dominant_rgb for _ in range(5)]
    assert all(r == runs[0] for r in runs), runs
    assert _rgb_l2(runs[0], [255, 0, 0]) < 12, runs[0]


def test_visual_timeline_gaps_instead_of_misaligning_on_undecodable_frame(
    tmp_path: Path,
) -> None:
    """A frame that fails to decode must leave a HOLE, not shift the curve.

    The samples that do land keep their own timestamp + source index, so a
    consumer sees index 1 is missing rather than silently reading frame 2's
    metrics as if they were frame 1's.
    """
    good0, good2 = tmp_path / "0.png", tmp_path / "2.png"
    _shot_frame(good0, (10, 40, 120), blur=False, seed=1)
    _shot_frame(good2, (235, 150, 40), blur=False, seed=2)
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=good0),
        ExtractedFrame(index=1, timestamp=2.0, path=tmp_path / "missing.png"),
        ExtractedFrame(index=2, timestamp=4.0, path=good2),
    ]

    tl = QualityAnalyzer().analyze(frames).visual_timeline
    assert [(s.t, s.index) for s in tl] == [(0.0, 0), (4.0, 2)]
    # The surviving sample carries frame 2's OWN values, not frame 1's slot.
    assert tl[1].brightness > tl[0].brightness


def test_visual_timeline_single_frame(tmp_path: Path) -> None:
    """One frame ⇒ exactly one sample (and no motion pair)."""
    p = tmp_path / "solo.png"
    _shot_frame(p, (200, 30, 30), blur=False, seed=3)
    metrics = QualityAnalyzer().analyze([ExtractedFrame(index=0, timestamp=1.25, path=p)])
    assert len(metrics.visual_timeline) == 1
    assert metrics.motion_timeline == []
    only = metrics.visual_timeline[0]
    assert only.t == 1.25
    assert only.index == 0
    assert len(only.dominant_rgb) == 3
    assert only.dominant_rgb[0] > only.dominant_rgb[1]  # red-dominant


def test_empty_frames_have_empty_visual_timeline() -> None:
    assert QualityAnalyzer().analyze([]).visual_timeline == []


def test_visual_timeline_serialises_to_plain_json(tmp_path: Path) -> None:
    """`pipeline._run_quality_stage` persists `asdict(quality)` into a JSONB column."""
    p = tmp_path / "a.png"
    _shot_frame(p, (10, 200, 10), blur=False, seed=4)
    payload = asdict(QualityAnalyzer().analyze([ExtractedFrame(index=0, timestamp=2.5, path=p)]))
    sample = payload["visual_timeline"][0]
    assert sample["t"] == 2.5
    assert sample["index"] == 0
    assert isinstance(sample["sharpness"], float)
    assert isinstance(sample["brightness"], float)
    assert all(isinstance(c, int) for c in sample["dominant_rgb"])
    json.dumps(payload)  # must not raise


def test_dominant_rgb_edge_shapes() -> None:
    """Degenerate image shapes must not raise: zero pixels, and already-small frames.

    A frame at or below the downscale target must skip the resize entirely
    (nothing to shrink), and a zero-pixel array must return no colour rather
    than tripping the histogram.
    """
    assert _dominant_rgb(np.zeros((0, 0, 3), np.uint8)) == []
    tiny = np.zeros((8, 8, 3), np.uint8)
    tiny[:, :] = (30, 20, 10)  # BGR
    assert _downscale_for_color(tiny).shape == tiny.shape  # no resize below target
    assert _dominant_rgb(tiny) == [10, 20, 30]
    single = np.zeros((1, 1, 3), np.uint8)
    single[0, 0] = (5, 6, 7)
    assert _dominant_rgb(single) == [7, 6, 5]


# --- sample size -------------------------------------------------------------


def test_metrics_carry_the_sample_they_rest_on(tmp_path: Path) -> None:
    """``frames_analyzed`` counts DECODED frames, not delivered ones.

    A JPEG cv2 cannot read is skipped by the loop, so ``len(frames)`` would
    overstate the evidence behind the means by exactly that many.
    """
    good = tmp_path / "good.jpg"
    _sharp_frame(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    frames = [
        ExtractedFrame(index=0, timestamp=0.0, path=good),
        ExtractedFrame(index=1, timestamp=1.0, path=broken),
    ]
    metrics = QualityAnalyzer().analyze(frames, frames_planned=9)
    assert metrics.frames_planned == 9
    assert metrics.frames_analyzed == 1


def test_unknown_plan_falls_back_to_the_delivered_count(tmp_path: Path) -> None:
    """``frames_planned=None`` (ad-hoc caller) must not invent a shortfall, but
    must still record the decode loss it CAN see."""
    good = tmp_path / "good.jpg"
    _sharp_frame(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"nope")
    metrics = QualityAnalyzer().analyze(
        [
            ExtractedFrame(index=0, timestamp=0.0, path=good),
            ExtractedFrame(index=1, timestamp=1.0, path=broken),
        ]
    )
    assert metrics.frames_planned == 2
    assert metrics.frames_analyzed == 1


def test_empty_frames_records_the_plan_it_failed_to_deliver() -> None:
    metrics = QualityAnalyzer().analyze([], frames_planned=12)
    assert (metrics.frames_planned, metrics.frames_analyzed) == (12, 0)


def test_sample_size_is_serialised_for_the_wire(tmp_path: Path) -> None:
    """The orchestrator reads the WIRE, so the counts must survive ``asdict``."""
    good = tmp_path / "good.jpg"
    _sharp_frame(good)
    payload = asdict(
        QualityAnalyzer().analyze(
            [ExtractedFrame(index=0, timestamp=0.0, path=good)], frames_planned=7
        )
    )
    assert payload["frames_planned"] == 7
    assert payload["frames_analyzed"] == 1
