"""VideoDownloader tests: httpx mock + yt-dlp absence.

Downloads are IP-pinned to the address the SSRF guard resolved, so the URL that
reaches the transport carries that address, not the hostname — hence the
``stub_dns`` fixture (deterministic resolution) and ``pinned()`` on every respx
route. Before the pin these tests depended on live DNS for ``example.com``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from src.services.video_downloader import VideoDownloader, _is_supported_share_host
from tests.unit.dns_stub import RebindResolver, pinned


def test_detects_youtube_hosts() -> None:
    assert _is_supported_share_host("https://www.youtube.com/watch?v=x") is True
    assert _is_supported_share_host("https://youtu.be/x") is True
    assert _is_supported_share_host("https://vimeo.com/12345") is True
    assert _is_supported_share_host("https://example.com/video.mp4") is False


@respx.mock
async def test_http_download(tmp_path: Path, stub_dns: RebindResolver) -> None:
    body = b"FAKE-MP4-BYTES" * 4096
    respx.get(pinned("https://example.com/video.mp4")).mock(
        return_value=httpx.Response(200, content=body)
    )
    dl = VideoDownloader(max_bytes=1024 * 1024)
    path = await dl.download(video_url="https://example.com/video.mp4", output_dir=tmp_path)
    assert path.exists()
    assert path.read_bytes() == body


@respx.mock
async def test_http_download_enforces_max_bytes(tmp_path: Path, stub_dns: RebindResolver) -> None:
    body = b"A" * (1024 * 1024 + 1)
    respx.get(pinned("https://example.com/big.mp4")).mock(
        return_value=httpx.Response(200, content=body)
    )
    dl = VideoDownloader(max_bytes=1024 * 1024)
    with pytest.raises(ValueError, match="exceeds max size"):
        await dl.download(video_url="https://example.com/big.mp4", output_dir=tmp_path)
    assert not (tmp_path / "big.mp4").exists()


def test_default_max_bytes_is_the_configured_upload_cap() -> None:
    """The constructor default must track StorageSettings, not drift behind it.

    pipeline.py always passes settings.storage.max_upload_size_bytes, but tests
    and any other constructor get this default - it must not silently stay at
    the old 2 GiB cap.
    """
    from src.config import StorageSettings

    assert VideoDownloader()._max_bytes == 4 * 1024 * 1024 * 1024
    assert VideoDownloader()._max_bytes == StorageSettings().max_upload_size_bytes


@respx.mock
async def test_http_download_raises_on_error_status(
    tmp_path: Path, stub_dns: RebindResolver
) -> None:
    respx.get(pinned("https://example.com/nope.mp4")).mock(return_value=httpx.Response(404))
    dl = VideoDownloader()
    with pytest.raises(httpx.HTTPStatusError):
        await dl.download(video_url="https://example.com/nope.mp4", output_dir=tmp_path)


async def test_ytdlp_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.services.video_downloader as vd

    monkeypatch.setattr(vd.shutil, "which", lambda _: None)
    dl = VideoDownloader()
    with pytest.raises(RuntimeError, match="yt-dlp"):
        await dl.download(video_url="https://www.youtube.com/watch?v=abc", output_dir=tmp_path)


@respx.mock
async def test_pooled_client_is_reused_across_downloads(
    tmp_path: Path, stub_dns: RebindResolver
) -> None:
    body = b"AAA"
    respx.get(pinned("https://example.com/a.mp4")).mock(
        return_value=httpx.Response(200, content=body)
    )
    respx.get(pinned("https://example.com/b.mp4")).mock(
        return_value=httpx.Response(200, content=body)
    )
    dl = VideoDownloader(max_bytes=1024)
    try:
        await dl.download(video_url="https://example.com/a.mp4", output_dir=tmp_path)
        first_pooled = dl._http_client
        assert first_pooled is not None
        await dl.download(video_url="https://example.com/b.mp4", output_dir=tmp_path)
        assert dl._http_client is first_pooled
    finally:
        await dl.aclose()
    assert dl._http_client is None


async def test_aclose_is_noop_when_client_was_injected() -> None:
    injected = httpx.AsyncClient()
    try:
        dl = VideoDownloader(http_client=injected)
        await dl.aclose()
        assert not injected.is_closed
    finally:
        await injected.aclose()
