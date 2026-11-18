"""Extract audio from a video file via ffmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from src.services.ffmpeg_utils import run_checked

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract mono 16 kHz WAV audio suitable for Whisper."""

    def __init__(self, *, timeout: float = 300.0) -> None:
        self._timeout = timeout

    async def extract(self, *, video_path: Path, output_path: Path) -> Path:
        """Extract audio from ``video_path`` into ``output_path``."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        await run_checked(cmd, timeout=self._timeout)
        return output_path
