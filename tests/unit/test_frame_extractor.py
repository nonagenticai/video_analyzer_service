"""Frame interval / timestamp calculation tests + FrameExtractor.extract tests."""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import pytest

from src.domain.analysis import ExtractedFrame, VideoMetadata
from src.services.ffmpeg_utils import FFmpegError
from src.services.frame_extractor import (
    EOF_SAFETY_MARGIN_FRACTION,
    EOF_SAFETY_MARGIN_SECONDS,
    MIN_FRAMES_FOR_QUALITY,
    FrameExtractor,
    compute_effective_interval,
    compute_frame_timestamps,
)


def test_effective_interval_short_video_respects_desired() -> None:
    # 30s video at 3s interval → 10 frames, under max=200
    eff = compute_effective_interval(duration_seconds=30, desired_interval=3.0, max_frames=200)
    assert eff == 3.0


def test_effective_interval_long_video_clamps_by_max_frames() -> None:
    # 20 min video (1200s) at 3s interval would be 400 frames
    eff = compute_effective_interval(duration_seconds=1200, desired_interval=3.0, max_frames=200)
    assert eff == 6.0


def test_effective_interval_handles_zero_duration() -> None:
    eff = compute_effective_interval(duration_seconds=0, desired_interval=3.0, max_frames=200)
    assert eff == 3.0


def test_effective_interval_handles_zero_max_frames() -> None:
    eff = compute_effective_interval(duration_seconds=60, desired_interval=2.0, max_frames=0)
    assert eff == 2.0


def test_compute_timestamps_bounds() -> None:
    ts = compute_frame_timestamps(duration_seconds=10, interval=3)
    assert ts == [0.0, 3.0, 6.0, 9.0]
    assert all(t < 10 for t in ts)


def test_compute_timestamps_invalid() -> None:
    assert compute_frame_timestamps(duration_seconds=0, interval=3) == []
    assert compute_frame_timestamps(duration_seconds=10, interval=0) == []


# ── minimum-frame floor (a hard-fail gate must not rest on ONE sample) ───


def test_short_clip_yields_at_least_min_frames() -> None:
    """A 1.4s clip at the production default interval sampled ONE frame."""
    eff = compute_effective_interval(duration_seconds=1.4, desired_interval=3.0, max_frames=40)
    ts = compute_frame_timestamps(duration_seconds=1.4, interval=eff)
    assert len(ts) >= MIN_FRAMES_FOR_QUALITY
    assert eff < 3.0


def test_long_clip_interval_unchanged_by_floor() -> None:
    """Clips already over the floor keep exactly today's cap-derived interval."""
    eff = compute_effective_interval(duration_seconds=300.0, desired_interval=3.0, max_frames=40)
    assert eff == 7.5  # 300 / 40 — the max_frames cap, identical to pre-floor behaviour


def test_mid_clip_keeps_desired_interval() -> None:
    """Desired interval already yields >= MIN_FRAMES_FOR_QUALITY → untouched."""
    eff = compute_effective_interval(duration_seconds=60.0, desired_interval=3.0, max_frames=200)
    assert eff == 3.0


def test_floor_never_exceeds_max_frames_cap() -> None:
    """A tiny max_frames still wins: the floor must not push past the cap."""
    eff = compute_effective_interval(duration_seconds=1.4, desired_interval=3.0, max_frames=2)
    ts = compute_frame_timestamps(duration_seconds=1.4, interval=eff)
    assert eff == pytest.approx(0.7)
    assert len(ts) <= 2


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_floor_handles_degenerate_duration(duration: float) -> None:
    eff = compute_effective_interval(duration_seconds=duration, desired_interval=3.0, max_frames=40)
    assert eff == 3.0
    assert compute_frame_timestamps(duration_seconds=duration, interval=eff) == []


@pytest.mark.parametrize("max_frames", [0, -1])
def test_no_cap_requested_means_no_floor_either(max_frames: int) -> None:
    """``max_frames <= 0`` returns ``desired_interval`` verbatim — no cap, no floor.

    Pins the pre-floor contract so it can never drift silently: at
    ``duration=1.4, desired=3.0`` the floor WOULD narrow the interval to 0.28
    (5 frames) if this branch were removed. This assertion is what makes the
    floor a strict superset of the old behaviour rather than a change to it.
    """
    eff = compute_effective_interval(
        duration_seconds=1.4, desired_interval=3.0, max_frames=max_frames
    )
    assert eff == 3.0
    assert compute_frame_timestamps(duration_seconds=1.4, interval=eff) == [0.0]


def test_degenerate_sub_frame_duration_terminates() -> None:
    """Termination guarantee only: a sub-frame duration must not loop forever.

    This does NOT prove five *distinct* pictures get decoded. At
    ``duration=0.02`` the interval is 0.004s, which is shorter than one frame
    period at any real fps, so all five timestamps land inside the same source
    frame and ffmpeg hands back the same decoded picture five times — the
    "mean" is still effectively a one-sample mean. Below roughly 0.17s
    (5 samples / 30 fps) the floor does NOT rescue the quality metric; at
    ``duration=1e-9`` the timestamps even ``round()`` to literal duplicates.
    That regime is deliberately out of scope: the motivating case is a ~1.4s
    clip (see the test below), and the only guarantee claimed here is that the
    computation terminates and stays inside the cap.
    """
    eff = compute_effective_interval(duration_seconds=0.02, desired_interval=3.0, max_frames=40)
    ts = compute_frame_timestamps(duration_seconds=0.02, interval=eff)
    assert eff > 0.0
    assert MIN_FRAMES_FOR_QUALITY <= len(ts) <= 40


def test_realistic_short_clip_yields_distinct_timestamps() -> None:
    """The case that motivated the floor: 1.4s ⇒ 5 DISTINCT sample points.

    At 0.28s spacing the timestamps are far enough apart to be separate source
    frames at any plausible fps, so ``mean_sharpness`` is a real 5-sample mean.
    """
    eff = compute_effective_interval(duration_seconds=1.4, desired_interval=3.0, max_frames=40)
    ts = compute_frame_timestamps(duration_seconds=1.4, interval=eff)
    assert len(ts) == MIN_FRAMES_FOR_QUALITY
    assert len(set(ts)) == MIN_FRAMES_FOR_QUALITY
    assert min(b - a for a, b in pairwise(ts)) == pytest.approx(0.28)


# ── EOF safety margin (a seek past the last decodable frame writes nothing) ──


@pytest.mark.parametrize(
    ("duration", "interval"),
    [
        (18.018, 3.0),  # the measured 4K HEVC case: the old loop emitted 18.0
        (10.010, 2.002),  # float accumulation emitted a literal 10.01
        (30.0, 3.0),
        (1.4, 0.28),  # short clip: the FRACTIONAL margin must keep all 5 samples
    ],
)
def test_no_timestamp_lands_within_eof_margin(duration: float, interval: float) -> None:
    """No sample may sit inside the last frame-period of the clip.

    ``ffmpeg -ss`` past the final decodable position writes NO file and exits 0,
    so a timestamp in that window is a silently lost frame.
    """
    ts = compute_frame_timestamps(duration_seconds=duration, interval=interval)
    margin = min(EOF_SAFETY_MARGIN_SECONDS, duration * EOF_SAFETY_MARGIN_FRACTION)
    assert ts
    assert max(ts) <= duration - margin + 1e-9


def test_timestamps_are_index_derived_not_accumulated() -> None:
    """``index * interval`` — accumulation drifted a sample onto the duration.

    At 10.010s / 2.002s the fifth ``t += interval`` was 10.009999999999998,
    which passed ``t < duration`` and rounded to a literal 10.01.
    """
    assert compute_frame_timestamps(duration_seconds=10.010, interval=2.002) == [
        0.0,
        2.002,
        4.004,
        6.006,
        8.008,
    ]


def test_timestamps_are_floats_even_for_int_interval() -> None:
    """The annotation promises list[float]; an int interval must not yield ints."""
    ts = compute_frame_timestamps(duration_seconds=10, interval=3)
    assert ts
    assert all(isinstance(t, float) for t in ts)


# ── frame-rate-aware EOF margin (0.10s is under one frame period below 10 fps) ──


@pytest.mark.parametrize("fps", [5.0, 8.0, 10.0])
def test_low_fps_margin_clears_a_full_frame_period(fps: float) -> None:
    """Below 10 fps the flat 0.10s margin does not even reach the failing frame.

    IF REVERTED: the margin stays 0.04 here (the fractional clause) and the last
    timestamp sits 0.05s from the end — inside the undecodable window at all
    three rates, so 5 fps and 8 fps clips lose their final frame the same silent
    rc=0/no-file way the margin exists to prevent.
    """
    duration = 2.0
    ts = compute_frame_timestamps(duration_seconds=duration, interval=0.05, fps=fps)
    assert ts
    assert duration - max(ts) >= 1.0 / fps


def test_low_fps_margin_capped_at_half_duration() -> None:
    """A clip shorter than one frame period must still yield timestamps.

    Without the half-duration cap the margin (1.5/5 = 0.30) exceeds the 0.20s
    clip and the limit goes negative — zero frames from a valid clip.
    """
    ts = compute_frame_timestamps(duration_seconds=0.2, interval=0.02, fps=5.0)
    assert ts
    assert max(ts) <= 0.1 + 1e-9


# ── regression fence: the fps clause must be purely ADDITIVE at >= 20 fps ──


@pytest.mark.parametrize("fps", [20.0, 24.0, 30.0, 60.0])
@pytest.mark.parametrize(
    ("duration", "interval"),
    [(18.018, 3.0), (10.010, 2.002), (30.0, 3.0), (10.0, 2.0)],
)
def test_normal_fps_output_is_byte_identical_to_fps_unknown(
    fps: float, duration: float, interval: float
) -> None:
    """At >= 20 fps ``1.5 / fps`` is <= 0.075, under the 0.10s floor.

    So the margin comes out UNCHANGED at 0.10 and the output must equal what a
    caller passing no fps at all gets. IF REVERTED: nothing — this test exists
    solely to prove the fps clause cannot regress normal-rate footage, which is
    every case the previous change (#104) measured at 24 fps.
    """
    assert compute_frame_timestamps(
        duration_seconds=duration, interval=interval, fps=fps
    ) == compute_frame_timestamps(duration_seconds=duration, interval=interval)


def test_measured_4k_hevc_case_is_pinned() -> None:
    """#104's measured 4K HEVC clip: 18.018s at 3.0s, 24 fps. Numbers unchanged."""
    assert compute_frame_timestamps(duration_seconds=18.018, interval=3.0, fps=24.0) == [
        0.0,
        3.0,
        6.0,
        9.0,
        12.0,
        15.0,
    ]


# ── end-to-end: the padded-audio case the prober fix corrects ────────


def test_padded_audio_clip_samples_inside_the_real_video() -> None:
    """6s video / 12s audio, production defaults (interval 3.0, max_frames 200).

    IF REVERTED (prober returns the container's 12.0): the effective interval is
    12/5 = 2.400 and the timestamps are [0, 2.4, 4.8, 7.2, 9.6] — 7.2 and 9.6 sit
    past the real 6s EOF, so ffmpeg exits 0 writing nothing for both and 2 of 5
    samples (40%) are lost from every mean the hard-fail gate reads.
    """
    true_duration = 6.0
    eff = compute_effective_interval(
        duration_seconds=true_duration, desired_interval=3.0, max_frames=200
    )
    ts = compute_frame_timestamps(duration_seconds=true_duration, interval=eff, fps=30.0)
    assert eff == pytest.approx(1.2)
    assert len(ts) == MIN_FRAMES_FOR_QUALITY
    assert all(t <= true_duration for t in ts)

    container_duration = 12.0
    bad_eff = compute_effective_interval(
        duration_seconds=container_duration, desired_interval=3.0, max_frames=200
    )
    bad_ts = compute_frame_timestamps(
        duration_seconds=container_duration, interval=bad_eff, fps=30.0
    )
    assert bad_eff == pytest.approx(2.4)
    assert sum(1 for t in bad_ts if t > true_duration) == 2


# ── FrameExtractor.extract ───────────────────────────────────────────


def _make_metadata(duration: float = 30.0) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=duration,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=int(duration * 30),
        has_audio=True,
    )


@pytest.fixture
def extractor() -> FrameExtractor:
    return FrameExtractor(timeout=30.0)


async def test_extract_creates_frames(extractor: FrameExtractor, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    out_dir = tmp_path / "frames"
    metadata = _make_metadata(duration=10.0)

    async def fake_run(cmd: list[str], **_: object) -> tuple[str, str]:
        frame_path = Path(cmd[-1])
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.touch()
        return ("", "")

    with patch("src.services.frame_extractor.run_checked", side_effect=fake_run):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=out_dir,
            metadata=metadata,
            desired_interval=5.0,
            max_frames=100,
        )
        frames = extraction.frames

    # 10s at desired=5.0 would be 2 frames; the min-frame floor narrows the
    # interval to 10/5 = 2.0s so the quality mean has >= MIN_FRAMES_FOR_QUALITY
    # samples.
    assert len(frames) == MIN_FRAMES_FOR_QUALITY
    assert all(isinstance(f, ExtractedFrame) for f in frames)
    assert [f.timestamp for f in frames] == [0.0, 2.0, 4.0, 6.0, 8.0]


async def test_extract_skips_failed_frames(extractor: FrameExtractor, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    out_dir = tmp_path / "frames"
    metadata = _make_metadata(duration=6.0)
    call_count = 0

    async def fail_second(cmd: list[str], **_: object) -> tuple[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise FFmpegError("frame extraction failed")
        frame_path = Path(cmd[-1])
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.touch()
        return ("", "")

    with patch("src.services.frame_extractor.run_checked", side_effect=fail_second):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=out_dir,
            metadata=metadata,
            desired_interval=3.0,
            max_frames=100,
        )
        frames = extraction.frames

    # 6s clip → floor narrows the interval to 6/5 = 1.2s → 5 attempts, one of
    # which fails, so 4 frames survive.
    assert len(frames) == MIN_FRAMES_FOR_QUALITY - 1
    assert frames[0].timestamp == 0.0


async def test_extract_raises_on_missing_video(extractor: FrameExtractor, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video not found"):
        await extractor.extract(
            video_path=tmp_path / "missing.mp4",
            output_dir=tmp_path / "frames",
            metadata=_make_metadata(),
            desired_interval=5.0,
            max_frames=100,
        )


async def test_extract_zero_duration_returns_empty(
    extractor: FrameExtractor, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    extraction = await extractor.extract(
        video_path=video,
        output_dir=tmp_path / "frames",
        metadata=_make_metadata(duration=0.0),
        desired_interval=5.0,
        max_frames=100,
    )
    frames = extraction.frames
    assert frames == []


async def test_extract_frame_not_written_is_skipped(
    extractor: FrameExtractor, tmp_path: Path
) -> None:
    """ffmpeg succeeds but doesn't write file → frame skipped."""
    video = tmp_path / "video.mp4"
    video.touch()

    async def no_write(_cmd: list[str], **_kw: object) -> tuple[str, str]:
        return ("", "")

    with patch("src.services.frame_extractor.run_checked", side_effect=no_write):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=tmp_path / "frames",
            metadata=_make_metadata(duration=3.0),
            desired_interval=2.0,
            max_frames=100,
        )
        frames = extraction.frames

    assert frames == []


async def test_ffmpeg_exit_zero_without_output_is_logged(
    extractor: FrameExtractor, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A frame lost to a zero-exit no-output ffmpeg must leave a record.

    ``run_checked`` raises only on a non-zero rc, so this whole failure mode was
    invisible: the ``except FFmpegError`` never fired and the missing file was
    skipped with no log line at all.
    """
    video = tmp_path / "video.mp4"
    video.touch()

    async def no_write(_cmd: list[str], **_kw: object) -> tuple[str, str]:
        return ("", "")

    with (
        caplog.at_level(logging.WARNING),
        patch("src.services.frame_extractor.run_checked", side_effect=no_write),
    ):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=tmp_path / "frames",
            metadata=_make_metadata(duration=10.0),
            desired_interval=2.0,
            max_frames=100,
        )
        frames = extraction.frames

    assert frames == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("produced no output" in r.getMessage() for r in warnings)


async def test_extraction_reports_the_plan_even_when_the_tail_is_lost(
    extractor: FrameExtractor, tmp_path: Path
) -> None:
    """``planned`` cannot be recovered from the surviving frames.

    ``max(frame.index)`` understates the plan by exactly the frames lost at the
    END, which is the most common loss (the rc=0/no-file seek cliff). This drops
    the last two of five, so the frames say "3 samples, indices 0..2" while the
    truth is 3 of 5.
    """
    video = tmp_path / "video.mp4"
    video.touch()
    written = 0

    async def stop_after_three(cmd: list[str], **_: object) -> tuple[str, str]:
        nonlocal written
        written += 1
        if written <= 3:
            frame_path = Path(cmd[-1])
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.touch()
        return ("", "")

    with patch("src.services.frame_extractor.run_checked", side_effect=stop_after_three):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=tmp_path / "frames",
            metadata=_make_metadata(duration=10.0),
            desired_interval=5.0,
            max_frames=100,
        )

    assert extraction.planned == MIN_FRAMES_FOR_QUALITY
    assert len(extraction.frames) == 3
    assert max(f.index for f in extraction.frames) == 2  # the plan is NOT derivable


async def test_full_extraction_reports_plan_equal_to_delivery(
    extractor: FrameExtractor, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    async def fake_run(cmd: list[str], **_: object) -> tuple[str, str]:
        frame_path = Path(cmd[-1])
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.touch()
        return ("", "")

    with patch("src.services.frame_extractor.run_checked", side_effect=fake_run):
        extraction = await extractor.extract(
            video_path=video,
            output_dir=tmp_path / "frames",
            metadata=_make_metadata(duration=10.0),
            desired_interval=5.0,
            max_frames=100,
        )

    assert extraction.planned == len(extraction.frames) == MIN_FRAMES_FOR_QUALITY
