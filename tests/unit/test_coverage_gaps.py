"""Targeted tests for remaining coverage gaps."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.routes.analysis import _kickoff_pipeline, _sanitize_filename
from src.services.video_downloader import VideoDownloader
from src.services.video_prober import VideoProbeError, _parse_fps, probe_video
from tests.conftest import TENANT_A

# -- analysis.py route helpers ---------------------------------------------------


def test_sanitize_filename_strips_traversal_and_punctuation() -> None:
    assert _sanitize_filename("../../etc/passwd") == "passwd"
    assert _sanitize_filename("foo bar.mp4") == "foo_bar.mp4"
    assert _sanitize_filename("") == "upload.bin"


async def test_kickoff_pipeline_no_runner() -> None:
    request = MagicMock()
    request.app.state = MagicMock(spec=[])
    # Should silently no-op when pipeline_runner attribute is missing
    await _kickoff_pipeline(request, "job-1", TENANT_A)


async def test_kickoff_pipeline_calls_runner() -> None:
    runner = MagicMock()
    runner.enqueue = AsyncMock()
    request = MagicMock()
    request.app.state.pipeline_runner = runner
    await _kickoff_pipeline(request, "job-2", TENANT_A)
    runner.enqueue.assert_awaited_once_with("job-2", TENANT_A)


# -- video_downloader timeout path ----------------------------------------------


async def test_ytdlp_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.services.video_downloader as vd

    monkeypatch.setattr(vd.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

    class _StallProcess:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(5)
            return (b"", b"")

        def kill(self) -> None:
            return None

        async def wait(self) -> None:
            return None

    async def _fake_create(*args, **kwargs) -> _StallProcess:
        return _StallProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
    dl = VideoDownloader(timeout=0.05)
    with pytest.raises(RuntimeError, match="yt-dlp timed out"):
        await dl.download(video_url="https://www.youtube.com/watch?v=abc", output_dir=tmp_path)


# -- video_prober parser branches ------------------------------------------------


def test_parse_fps_zero_denominator_returns_zero() -> None:
    assert _parse_fps("10/0") == 0.0


def test_parse_fps_handles_value_error_in_division() -> None:
    assert _parse_fps("abc/2") == 0.0


async def test_probe_video_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"junk")

    from src.services import video_prober

    async def _fake_run_checked(*args: Any, **kwargs: Any) -> tuple[str, str]:
        from src.services.ffmpeg_utils import FFmpegError

        raise FFmpegError("ffprobe broke")

    monkeypatch.setattr(video_prober, "run_checked", _fake_run_checked)
    with pytest.raises(VideoProbeError, match="ffprobe failed"):
        await probe_video(target)


async def test_probe_video_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"junk")

    from src.services import video_prober

    async def _fake_run_checked(*args: Any, **kwargs: Any) -> tuple[str, str]:
        return ("not-json", "")

    monkeypatch.setattr(video_prober, "run_checked", _fake_run_checked)
    with pytest.raises(VideoProbeError, match="Invalid ffprobe JSON"):
        await probe_video(target)


# Note: per-SDK vision providers (openai/anthropic/google) were removed
# in the WMClient migration — see src/services/vision/wm_provider.py.
# Output-parsing branches that previously needed direct coverage now
# live in test_wm_vision_provider.py (the WM output-coercion helpers).
