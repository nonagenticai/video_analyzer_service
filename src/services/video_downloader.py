"""Download a video from an HTTP URL or YouTube/Vimeo share link."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

# ``_pin`` is the shared guard's own URL/Host/SNI rewrite. It is imported rather
# than re-derived so the pinning semantics (IPv6 bracketing, port and userinfo
# preservation, ``sni_hostname`` driving both the ClientHello SNI and httpx's
# certificate hostname verification) have exactly ONE implementation in the
# fleet — the same reason ``url_security`` already imports ``_is_blocked_ip``.
from w2v_common.net.safe_fetch import _pin as _pin_connection

from src.services.url_security import validate_public_url

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 3
_SHARE_HOST_SUFFIXES: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
)


def _is_supported_share_host(url: str) -> bool:
    """Return True for YouTube/Vimeo URLs that require yt-dlp."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == suffix or host.endswith("." + suffix) for suffix in _SHARE_HOST_SUFFIXES)


class VideoDownloader:
    """Async downloader using httpx.stream for HTTP and yt-dlp for share hosts."""

    def __init__(
        self,
        *,
        max_bytes: int = 4 * 1024 * 1024 * 1024,
        timeout: float = 300.0,
        http_client: httpx.AsyncClient | None = None,
        allowed_hosts: frozenset[str] = frozenset(),
    ) -> None:
        self._max_bytes = max_bytes
        self._timeout = timeout
        # Exact hostnames the SSRF guard trusts despite an internal address
        # (e.g. minio.test — the store the record-stage QA fetches clips from).
        self._allowed_hosts = allowed_hosts
        # Hold a single long-lived AsyncClient so the connection pool is
        # shared across downloads; injection (tests, custom wiring) keeps
        # the caller as the lifecycle owner.
        self._http_client: httpx.AsyncClient | None = http_client
        self._owned = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the owned ``AsyncClient`` if we instantiated it."""
        if self._owned and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def download(self, *, video_url: str, output_dir: Path) -> Path:
        """Download ``video_url`` and return the path to the saved file."""
        # The validated address is carried to the transport, not discarded:
        # re-resolving at connect time is the DNS-rebind window.
        pin = validate_public_url(video_url, self._allowed_hosts)
        output_dir.mkdir(parents=True, exist_ok=True)
        if _is_supported_share_host(video_url):
            return await self._download_via_ytdlp(video_url, output_dir)
        return await self._download_via_http(video_url, output_dir, pin)

    async def _download_via_http(self, url: str, output_dir: Path, pin: str | None) -> Path:
        """Stream ``url`` to disk, pinning every hop to its validated address.

        ``w2v_common``'s ``safe_get``/``safe_request`` are the fleet's preferred
        front door and are used everywhere a response fits in memory — but they
        MATERIALISE the body (``chunks: list[bytes]`` -> ``_materialize``)
        before returning. This downloader accepts up to 4 GiB of video and must
        write it to disk chunk-by-chunk, so it cannot use them without turning a
        bounded-memory download into a 4 GiB heap allocation. What it does
        instead is reuse the library's *primitives* — ``validate_public_url``
        for the resolve-and-judge and ``_pin`` for the request rewrite — so the
        only thing hand-rolled here is the streaming write loop.
        """
        parsed = urlparse(url)
        name = Path(parsed.path).name or "video.mp4"
        if "." not in name:
            name = f"{name}.mp4"
        dest = output_dir / name

        client = self._get_client()
        total = 0
        current_url = url
        current_pin = pin
        for redirect_count in range(_MAX_REDIRECTS + 1):
            # Connect to the address that was judged, keeping the original
            # hostname for the Host header and for TLS SNI/cert verification.
            request_url, pin_headers, pin_extensions = _pin_connection(current_url, current_pin)
            request = client.build_request(
                "GET", request_url, headers=pin_headers, extensions=pin_extensions
            )
            resp = await client.send(request, stream=True, follow_redirects=False)
            try:
                if resp.is_redirect:
                    if redirect_count >= _MAX_REDIRECTS:
                        raise ValueError(
                            f"Too many redirects (>{_MAX_REDIRECTS}) downloading {url!r}"
                        )
                    location = resp.headers.get("location")
                    if not location:
                        raise ValueError(
                            f"Redirect response without Location header for {current_url!r}"
                        )
                    current_url = urljoin(current_url, location)
                    # Re-validate AND re-pin: a pinned first hop with an
                    # unpinned redirect leaves the same window wide open.
                    current_pin = validate_public_url(current_url, self._allowed_hosts)
                    continue
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > self._max_bytes:
                            fh.close()
                            dest.unlink(missing_ok=True)
                            raise ValueError(f"Download exceeds max size {self._max_bytes}")
                        fh.write(chunk)
            finally:
                await resp.aclose()
            break
        else:  # pragma: no cover — loop always breaks or raises
            raise ValueError(f"Failed to download {url!r}")
        return dest

    async def _download_via_ytdlp(self, url: str, output_dir: Path) -> Path:
        if not _is_supported_share_host(url):
            raise ValueError(f"yt-dlp is only allowed for share hosts, got {url!r}")
        ytdlp_path = shutil.which("yt-dlp")
        if ytdlp_path is None:
            raise RuntimeError("yt-dlp is not installed or not on PATH")
        template = str(output_dir / "video.%(ext)s")
        cmd = [
            ytdlp_path,
            "--no-playlist",
            "--restrict-filenames",
            "-f",
            "mp4/best",
            "-o",
            template,
            url,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"yt-dlp timed out after {self._timeout}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp failed (rc={proc.returncode}): {stderr.decode('utf-8', errors='replace')[:500]}"
            )
        for candidate in output_dir.iterdir():
            if candidate.name.startswith("video."):
                return candidate
        raise RuntimeError("yt-dlp completed but no output file was found")
