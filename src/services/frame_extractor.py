"""Extract evenly spaced frames from a video using ffmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from src.domain.analysis import ExtractedFrame, FrameExtraction, VideoMetadata
from src.services.ffmpeg_utils import FFmpegError, run_checked

logger = logging.getLogger(__name__)

# Minimum number of frames we try to sample from ANY clip, however short.
#
# WHY: the quality metrics (Laplacian-variance sharpness, palette, motion) are
# reported as a MEAN over the extracted frames, and downstream consumers run a
# HARD-FAIL "blank clip" gate on that mean — a gate that can fail a paid render.
# With the production defaults (PIPELINE_FRAME_INTERVAL_SECONDS=3.0) a clip
# shorter than the interval yields exactly ONE frame, so the "mean" is a single
# Laplacian sample and one dark/transitional frame is enough to condemn the
# clip. Observed 2026-08-02: a 1400 ms clip hard-failed as blank on a
# one-sample mean_sharpness. A hard-fail verdict must never rest on a single
# sample, so short clips get their interval NARROWED to reach this floor.
# The ``max_frames`` cap still wins as the upper bound on frame count.
MIN_FRAMES_FOR_QUALITY = 5

# How far before the end of the clip the LAST sample point must stop.
#
# See ``compute_frame_timestamps`` for the measurements behind these two
# numbers and for what they do NOT guarantee.
EOF_SAFETY_MARGIN_SECONDS = 0.10
EOF_SAFETY_MARGIN_FRACTION = 0.02

# How many frame PERIODS the margin must clear when the clip's fps is known.
#
# WHY a third number: the two above are absolute seconds, and 0.10s is SHORTER
# THAN ONE FRAME PERIOD below 10 fps — so on an 8 fps (0.125s) or 5 fps (0.200s)
# clip the "safety" margin did not even reach the frame the seek was going to
# fail on, and the final sample was lost exactly as it was before the margin
# existed. 1.5 periods clears the measured cliff (one period) with the same kind
# of headroom 0.10s gives at 24 fps, and is deliberately small enough that at
# 20 fps and above (1.5/20 = 0.075) it stays UNDER the 0.10s floor and therefore
# changes nothing for normal-rate footage.
EOF_SAFETY_MARGIN_FRAME_PERIODS = 1.5


def compute_effective_interval(
    *, duration_seconds: float, desired_interval: float, max_frames: int
) -> float:
    """Return the sampling interval satisfying ``desired_interval``, the
    ``max_frames`` cap, and the ``MIN_FRAMES_FOR_QUALITY`` floor.

    Precedence: the ``max_frames`` cap is an absolute upper bound on the frame
    count and therefore an absolute lower bound on the interval. Within that,
    a clip too short to yield ``MIN_FRAMES_FOR_QUALITY`` frames at
    ``desired_interval`` has its interval narrowed so the quality mean rests on
    more than one sample. Clips that already yield at least that many frames
    are unaffected.

    ``max_frames <= 0`` means "no cap requested" and, as on the pre-floor
    version of this function, also means "no floor": ``desired_interval`` is
    returned verbatim. Keeping that early return makes this function a strict
    superset of the old behaviour — without it a caller passing
    ``max_frames=0`` would silently start getting ``MIN_FRAMES_FOR_QUALITY``
    frames where it used to get one. No production caller reaches this branch
    (``pipeline.py`` resolves ``request.max_frames or settings.max_frames``, so
    0/None fall back to the configured cap of 200), so there is nothing to gain
    by changing the contract of an input that only arrives as nonsense.
    """
    if duration_seconds <= 0:
        return desired_interval
    if max_frames <= 0:
        return desired_interval

    # Lower bound on the interval imposed by the frame-count cap.
    interval_from_cap = duration_seconds / max_frames

    # Upper bound on the interval imposed by the minimum-frame floor. This only
    # bites when the desired interval is wider, i.e. on short clips.
    interval_from_min_frames = duration_seconds / MIN_FRAMES_FOR_QUALITY

    return max(min(desired_interval, interval_from_min_frames), interval_from_cap)


def compute_frame_timestamps(
    *, duration_seconds: float, interval: float, fps: float = 0.0
) -> list[float]:
    """Return timestamps at which frames should be extracted.

    Timestamps are derived from a sample INDEX (``index * interval``) and stop
    an ``EOF_SAFETY_MARGIN`` short of the clip's end. Both of those properties
    fix a measured production defect; neither is cosmetic.

    WHY the margin — the tail of every clip was invisible. ``ffmpeg -ss`` placed
    BEFORE ``-i`` seeks and then decodes forward; past the last decodable
    position it writes NO output file while still exiting 0 (rc=0, no file, no
    stderr). Measured on a real 4K HEVC clip of duration 18.018s sampled at
    3.0s, whose final sample landed on 18.000: ``-ss 18.000`` -> rc=0 and no
    file; ``-ss 17.980`` -> still no file; ``-ss 17.960`` -> a 434734-byte
    JPEG. Repeated across GOP sizes 24/48/120/240, ``duration - 0.10`` ALWAYS
    produced a file and ``duration - 0.04`` NEVER did, so the cliff sits at
    roughly one frame period rather than at keyframe spacing. 0.10s was chosen
    to clear about two frame periods at 24 fps and three at 30 fps — comfortably
    past the measured cliff, and small enough that the sample it removes was
    never going to decode anyway.

    WHY index-derived — the previous loop accumulated ``t += interval``, so
    float error let a sample land AT the duration. A 10.010s clip has its
    interval narrowed to 2.002 by the ``MIN_FRAMES_FOR_QUALITY`` floor; the
    fifth accumulation is 10.009999999999998, which passes ``t < 10.010``, and
    ``round(t, 3)`` emits a literal 10.01 — a seek to the exact end of the
    clip. Both ``18.00`` and ``10.01`` appear verbatim in production logs as
    ``Frame extraction failed at ...: ffmpeg exited with code 234`` on the pod's
    ffmpeg 7.1.5 (on 4.4 the same case is silent — see ``FrameExtractor.extract``).
    Multiplying by the index instead keeps the error bounded at one rounding of
    a single product, and the margin then absorbs it regardless.

    The fractional clause is load-bearing, not a stylistic flourish: a flat
    0.10s would eat a meaningful share of the short clips the frame floor exists
    to serve (at duration 1.4s the margin is 0.028s, at 0.02s it is 0.0004s), so
    the margin scales down with the clip and short-clip sampling is unchanged.

    WHY ``fps`` — the two numbers above are absolute seconds, and the cliff they
    were measured against is a FRAME PERIOD, which is not. 0.10s is shorter than
    one frame period below 10 fps, so on the 8 fps (0.125s) and 5 fps (0.200s)
    clips this service does receive — screen recordings and timelapses — the
    margin landed INSIDE the undecodable window and the final sample was lost in
    the same silent rc=0/no-file way the margin was introduced to prevent. When
    ``fps`` is known the margin is therefore raised to at least
    ``EOF_SAFETY_MARGIN_FRAME_PERIODS / fps``. The correction to the duration
    resolver in ``video_prober`` makes this more urgent, not less: it SHRINKS the
    duration of audio-padded files, which shrinks the proportional term with it.

    This clause is purely ADDITIVE at normal frame rates, and the regression
    fence in the tests pins that. ``1.5 / fps`` is 0.0625 at 24 fps and 0.050 at
    30 fps — both under the 0.10s floor — so at **20 fps and above the margin
    comes out UNCHANGED at 0.10** and every number measured for the previous
    change at 24 fps still holds verbatim: the 18.018s / 3.0s case still stops at
    15.0, and 10.010s / 2.002s still ends at 8.008. Only below 20 fps does the
    fps term win. ``fps=0.0`` (the default, and what a caller with no metadata
    passes) reproduces the previous behaviour exactly.

    The margin is finally capped at half the duration so a degenerate sub-frame
    clip — where one frame period exceeds the whole clip — still yields a
    non-empty timestamp list instead of terminating with nothing.

    What this does NOT guarantee: we do not probe GOP structure. A long-GOP
    source whose last decodable position is EARLIER than
    ``duration - margin`` can still lose its final sample the same silent way.
    The measurements above say the cliff is a frame period on every encode we
    tested, not that it cannot be wider on an encode we did not. Handling that
    would require probing the last keyframe per clip; it is deliberately out of
    scope, and ``FrameExtractor.extract`` now at least LOGS the loss instead of
    dropping it in silence.
    """
    if duration_seconds <= 0 or interval <= 0:
        return []
    margin = min(EOF_SAFETY_MARGIN_SECONDS, duration_seconds * EOF_SAFETY_MARGIN_FRACTION)
    if fps > 0:
        margin = max(margin, EOF_SAFETY_MARGIN_FRAME_PERIODS / fps)
    # Terminate on degenerate clips: without this a sub-frame duration would give
    # a negative limit and an empty result.
    margin = min(margin, duration_seconds / 2.0)
    limit = duration_seconds - margin
    timestamps: list[float] = []
    index = 0
    while True:
        # float() is deliberate: the annotation promises list[float] and an int
        # ``interval`` (as several callers and tests pass) would otherwise make
        # this product an int.
        t = float(index) * interval
        if t > limit:
            break
        timestamps.append(round(t, 3))
        index += 1
    return timestamps


class FrameExtractor:
    """Extract keyframes at a given interval into an output directory."""

    def __init__(self, *, timeout: float = 600.0) -> None:
        self._timeout = timeout

    async def extract(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        metadata: VideoMetadata,
        desired_interval: float,
        max_frames: int,
    ) -> FrameExtraction:
        """Extract frames at a computed effective interval.

        Returns a :class:`FrameExtraction` carrying the ExtractedFrames (``path``
        pointing to the JPEG written on disk, ordered by timestamp) alongside
        ``planned`` — how many timestamps were scheduled.

        WHY the count is returned rather than just the frames: both drop paths
        below ``continue``, so a partial extraction is shaped exactly like a
        complete one. ``MIN_FRAMES_FOR_QUALITY`` exists to stop a hard-fail
        verdict resting on one sample, but it only ever constrained the PLAN —
        nothing measured the delivery. Returning ``planned`` is what lets
        ``QualityMetrics`` carry a denominator and ``verifier.check_quality``
        refuse a verdict the sample cannot support.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        effective = compute_effective_interval(
            duration_seconds=metadata.duration_seconds,
            desired_interval=desired_interval,
            max_frames=max_frames,
        )
        timestamps = compute_frame_timestamps(
            duration_seconds=metadata.duration_seconds,
            interval=effective,
            fps=metadata.fps,
        )
        if not timestamps:
            return FrameExtraction(frames=[], planned=0)

        frames: list[ExtractedFrame] = []
        for idx, ts in enumerate(timestamps):
            frame_path = output_dir / f"frame_{idx:04d}.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "4",
                str(frame_path),
            ]
            try:
                await run_checked(cmd, timeout=self._timeout)
            except FFmpegError as exc:
                logger.warning("Frame extraction failed at %.2fs: %s", ts, exc)
                continue
            # run_checked only raises on a NON-ZERO exit. ffmpeg can seek past
            # the last decodable position, write nothing, and still exit 0 — so
            # without this branch the frame vanished with no log line at all.
            if not frame_path.exists():
                logger.warning(
                    "Frame extraction produced no output at %.3fs (idx=%d): ffmpeg exited 0 "
                    "but wrote no file",
                    ts,
                    idx,
                )
                continue
            frames.append(ExtractedFrame(index=idx, timestamp=ts, path=frame_path))

        if len(frames) < len(timestamps):
            # The per-frame warnings above say WHERE each loss happened; this one
            # says HOW BIG the hole is, which is the number the quality verdict
            # actually turns on.
            logger.warning(
                "Frame extraction delivered %d of %d planned frames for %s",
                len(frames),
                len(timestamps),
                video_path.name,
            )
        return FrameExtraction(frames=frames, planned=len(timestamps))
