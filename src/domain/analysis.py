"""Domain dataclasses used internally by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VideoMetadata:
    """Result of ffprobe on the source video."""

    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    format_name: str | None = None


@dataclass(slots=True)
class ExtractedFrame:
    """A frame extracted by ffmpeg."""

    index: int
    timestamp: float
    path: Path


@dataclass(slots=True)
class FrameExtraction:
    """What the extractor PLANNED to sample, and what it actually delivered.

    ``extract`` used to return a bare ``list[ExtractedFrame]``, which made a
    dropped frame invisible: a seek that raises, or the silent rc=0/no-file
    case, just ``continue``s, so the survivors are indistinguishable from a
    complete sample. Every consumer of that list — including the whole-video
    quality mean a downstream HARD-FAIL gate votes on — then treated whichever
    frames happened to decode as the entire evidence base.

    ``planned`` is ``len(compute_frame_timestamps(...))``: how many samples the
    planner decided this clip warranted, AFTER the ``MIN_FRAMES_FOR_QUALITY``
    floor and the ``max_frames`` cap have both been applied. It is the only
    honest denominator, and it cannot be recovered from the frames themselves —
    ``max(frame.index)`` understates it exactly when the TAIL is what dropped,
    which is the most common loss (see the EOF-margin measurements in
    ``compute_frame_timestamps``).
    """

    frames: list[ExtractedFrame]
    planned: int


@dataclass(slots=True)
class FrameAnalysis:
    """Per-frame result aggregating vision LLM description + OCR text."""

    index: int
    timestamp: float
    path: str
    description: str | None = None
    ocr_text: str | None = None
    labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MotionSample:
    """One point on the motion curve: how much changed *arriving at* ``t``.

    ``motion`` is the same 0..1 frame-delta the aggregate ``motion_score``
    averages, but kept per-sample instead of collapsed. ``t`` is the timestamp
    of the LATER frame of the pair, i.e. the moment the change has landed.
    """

    t: float
    motion: float


@dataclass(slots=True)
class VisualSample:
    """Per-frame visual metrics at ``t`` — the shot-comparison curve.

    ``mean_sharpness``/``mean_brightness``/``dominant_colors`` on
    ``QualityMetrics`` are whole-video aggregates: identical for every shot of
    a film by construction, so a consumer diffing shots against them measures
    exactly nothing. These are the SAME quantities kept per frame instead of
    collapsed, so continuity / cohesion / colour drift become measurable.

    ``t`` is this frame's own timestamp (unlike ``MotionSample``, which is a
    PAIR and stamps the later frame). ``index`` is the source
    ``ExtractedFrame.index``, carried so a consumer can see where an
    undecodable frame was dropped instead of silently mis-aligning the curve.

    ``dominant_rgb`` is ``[r, g, b]`` 0..255 — the largest k-means cluster
    centre of this frame. Numeric on purpose: it drops straight into a feature
    vector without parsing (hex is ``"#%02x%02x%02x" % tuple(rgb)``). Empty
    only for a degenerate zero-pixel image.
    """

    t: float
    index: int
    sharpness: float
    brightness: float
    dominant_rgb: list[int] = field(default_factory=list)


@dataclass(slots=True)
class QualityMetrics:
    """Aggregate visual quality metrics for a video."""

    mean_sharpness: float
    mean_brightness: float
    motion_score: float
    dominant_colors: list[str]
    overall_score: float
    # The per-sample curve `motion_score` is the mean of. Consumers (e.g. the
    # orchestrator's peak-motion cold-open) need to know WHEN the action happens,
    # not just how much there was on average. Resolution == the frame-extraction
    # interval, so this is a coarse curve (seconds), not per-video-frame.
    # Appended last + defaulted: every existing constructor stays valid.
    motion_timeline: list[MotionSample] = field(default_factory=list)
    # The per-frame visual curve the scalar aggregates above are the mean of.
    # Without it every shot of a film carries the same sharpness/brightness/
    # colour, so adjacent-shot deltas are 0.0 BY CONSTRUCTION and continuity /
    # cohesion are unmeasurable. Appended last + defaulted: every existing
    # constructor and every already-persisted JSONB row stays valid.
    visual_timeline: list[VisualSample] = field(default_factory=list)
    # SAMPLE SIZE — how many frames these aggregates actually rest on, and how
    # many the extractor set out to sample. Without them a 20-frame mean and a
    # 1-frame mean are the same number on the wire, so a downstream hard-fail
    # gate cannot tell a measured clip from a mostly-corrupt one whose single
    # surviving frame happened to be sharp. ``frames_planned`` is
    # ``FrameExtraction.planned``; ``frames_analyzed`` counts only the frames
    # that DECODED here (cv2.imread can still fail on a delivered JPEG), so the
    # pair spans both loss sites — extraction and decode.
    #
    # Both default to 0 for the same reason the timelines default: JSONB rows
    # persisted before this field existed lack the key entirely. 0/0 therefore
    # means "an older analyzer wrote this", NOT "nothing was sampled" — see
    # ``verifier.check_quality``, whose gate is a no-op at ``frames_planned=0``
    # precisely so a legacy row cannot be retro-failed.
    frames_planned: int = 0
    frames_analyzed: int = 0
