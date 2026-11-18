"""Real ffmpeg/ffprobe integration tests against a synthesized fixture video."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.audio_extractor import AudioExtractor
from src.services.ffmpeg_utils import ffmpeg_available, run_checked
from src.services.frame_extractor import (
    FrameExtractor,
    compute_effective_interval,
    compute_frame_timestamps,
)
from src.services.video_prober import probe_video

needs_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg binary not available on PATH"
)


@pytest.fixture
async def synthetic_video(tmp_path: Path) -> Path:
    """Generate a 4-second 1280x720 silent test video via ffmpeg lavfi."""
    out = tmp_path / "synthetic.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=1280x720:r=30:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=1280x720:r=30:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    await run_checked(cmd)
    assert out.exists()
    return out


@needs_ffmpeg
async def test_probe_video_real(synthetic_video: Path) -> None:
    meta = await probe_video(synthetic_video)
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.duration_seconds > 3.0
    assert meta.duration_seconds < 5.0
    assert meta.has_audio is True


@needs_ffmpeg
async def test_probe_missing_video(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await probe_video(tmp_path / "missing.mp4")


@needs_ffmpeg
async def test_frame_extractor_real(synthetic_video: Path, tmp_path: Path) -> None:
    meta = await probe_video(synthetic_video)
    extractor = FrameExtractor()
    extraction = await extractor.extract(
        video_path=synthetic_video,
        output_dir=tmp_path / "frames",
        metadata=meta,
        desired_interval=1.0,
        max_frames=10,
    )
    frames = extraction.frames
    assert len(frames) >= 3
    for frame in frames:
        assert frame.path.exists()
        assert frame.path.stat().st_size > 0
        assert frame.path.suffix == ".jpg"


@needs_ffmpeg
async def test_frame_extractor_respects_max_frames(synthetic_video: Path, tmp_path: Path) -> None:
    meta = await probe_video(synthetic_video)
    extractor = FrameExtractor()
    extraction = await extractor.extract(
        video_path=synthetic_video,
        output_dir=tmp_path / "frames",
        metadata=meta,
        desired_interval=0.1,  # would generate 40 at this rate
        max_frames=3,
    )
    frames = extraction.frames
    assert len(frames) <= 5  # 3 with some slack from rounding


@pytest.fixture
async def tail_video(tmp_path: Path) -> Path:
    """Generate an 18.018s 24fps clip — the duration that lost its tail frame.

    24 fps is the low end of real footage, so one frame period (~41.7ms) is the
    widest the measured EOF cliff gets on a normal encode.
    """
    out = tmp_path / "tail.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=24:d=18.018",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    await run_checked(cmd)
    assert out.exists()
    return out


@needs_ffmpeg
async def test_tail_frame_survives_near_eof(tail_video: Path, tmp_path: Path) -> None:
    """EVERY requested timestamp must produce a frame — the tail one included.

    Before the EOF safety margin the final timestamp landed past the last
    decodable position, where ffmpeg exits 0 and writes nothing, so the last
    ~3s of every clip was invisible to downstream segment selection.
    """
    meta = await probe_video(tail_video)
    effective = compute_effective_interval(
        duration_seconds=meta.duration_seconds, desired_interval=3.0, max_frames=200
    )
    expected = compute_frame_timestamps(duration_seconds=meta.duration_seconds, interval=effective)

    # Pinned INDEPENDENTLY of compute_frame_timestamps. Comparing the extracted
    # timestamps only against that function's own output is self-consistent: a
    # revert changes both sides together and the assertion still passes. The
    # literal list is what makes this a revert-detector — the pre-fix function
    # appends a sixth sample at 18.0, which is past the last decodable position.
    assert expected == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]

    extractor = FrameExtractor()
    extraction = await extractor.extract(
        video_path=tail_video,
        output_dir=tmp_path / "frames",
        metadata=meta,
        desired_interval=3.0,
        max_frames=200,
    )
    frames = extraction.frames

    assert len(frames) == len(expected)
    assert [f.timestamp for f in frames] == expected
    for frame in frames:
        assert frame.path.exists()
        assert frame.path.stat().st_size > 0


@needs_ffmpeg
async def test_frame_extractor_missing_input(tmp_path: Path) -> None:
    extractor = FrameExtractor()
    from src.domain.analysis import VideoMetadata

    meta = VideoMetadata(
        duration_seconds=10.0,
        width=640,
        height=480,
        fps=30.0,
        frame_count=300,
        has_audio=False,
    )
    with pytest.raises(FileNotFoundError):
        await extractor.extract(
            video_path=tmp_path / "missing.mp4",
            output_dir=tmp_path / "frames",
            metadata=meta,
            desired_interval=1.0,
            max_frames=10,
        )


@needs_ffmpeg
async def test_audio_extractor_real(synthetic_video: Path, tmp_path: Path) -> None:
    extractor = AudioExtractor()
    out = tmp_path / "audio.wav"
    result = await extractor.extract(video_path=synthetic_video, output_path=out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


@needs_ffmpeg
async def test_audio_extractor_missing_input(tmp_path: Path) -> None:
    extractor = AudioExtractor()
    with pytest.raises(FileNotFoundError):
        await extractor.extract(
            video_path=tmp_path / "missing.mp4",
            output_path=tmp_path / "audio.wav",
        )
