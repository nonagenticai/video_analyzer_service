"""yt-dlp success path test using mocked subprocess."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


async def test_ytdlp_download_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.services.video_downloader as vd

    monkeypatch.setattr(vd.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

    output_file = tmp_path / "video.mp4"

    class _FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            output_file.write_bytes(b"FAKE-YT-VIDEO")
            return (b"", b"")

    async def _fake_create(*args, **kwargs) -> _FakeProcess:
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    dl = vd.VideoDownloader()
    result = await dl.download(video_url="https://www.youtube.com/watch?v=abc", output_dir=tmp_path)
    assert result == output_file
    assert result.read_bytes() == b"FAKE-YT-VIDEO"


async def test_ytdlp_download_nonzero_rc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.services.video_downloader as vd

    monkeypatch.setattr(vd.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

    class _FailingProcess:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"download error\n")

    async def _fake_create(*args, **kwargs) -> _FailingProcess:
        return _FailingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    dl = vd.VideoDownloader()
    with pytest.raises(RuntimeError, match="yt-dlp failed"):
        await dl.download(video_url="https://youtu.be/abc", output_dir=tmp_path)


async def test_ytdlp_no_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.services.video_downloader as vd

    monkeypatch.setattr(vd.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

    class _SilentProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def _fake_create(*args, **kwargs) -> _SilentProcess:
        return _SilentProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    dl = vd.VideoDownloader()
    with pytest.raises(RuntimeError, match="no output file"):
        await dl.download(video_url="https://vimeo.com/123", output_dir=tmp_path)
