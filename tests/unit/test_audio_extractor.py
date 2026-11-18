"""Tests for audio extractor service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.services.audio_extractor import AudioExtractor


@pytest.fixture
def extractor() -> AudioExtractor:
    return AudioExtractor(timeout=30.0)


async def test_extract_calls_ffmpeg(extractor: AudioExtractor, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "audio" / "out.wav"

    with patch("src.services.audio_extractor.run_checked", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("", "")
        result = await extractor.extract(video_path=video, output_path=output)

    assert result == output
    mock_run.assert_awaited_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert str(video) in cmd
    assert str(output) in cmd
    assert "-ar" in cmd
    assert "16000" in cmd


async def test_extract_creates_parent_dir(extractor: AudioExtractor, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "deep" / "nested" / "audio.wav"

    with patch("src.services.audio_extractor.run_checked", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("", "")
        await extractor.extract(video_path=video, output_path=output)

    assert output.parent.exists()


async def test_extract_raises_on_missing_video(extractor: AudioExtractor, tmp_path: Path) -> None:
    video = tmp_path / "nonexistent.mp4"
    output = tmp_path / "out.wav"

    with pytest.raises(FileNotFoundError, match="Video not found"):
        await extractor.extract(video_path=video, output_path=output)


async def test_extract_uses_timeout(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "out.wav"

    extractor = AudioExtractor(timeout=42.0)
    with patch("src.services.audio_extractor.run_checked", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("", "")
        await extractor.extract(video_path=video, output_path=output)

    assert mock_run.call_args[1]["timeout"] == 42.0
