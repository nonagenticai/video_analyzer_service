"""DNS-rebinding TOCTOU regression tests for :mod:`src.services.video_downloader`.

``validate_public_url`` resolves the hostname and judges the answer. If the
caller then hands the *hostname* to the transport, the transport resolves it a
SECOND time — and an attacker who controls the authoritative DNS for that name
can answer public on the validation lookup and internal on the connect lookup.
The guard passes and the socket lands on a ClusterIP / metadata endpoint.

Closing it requires the caller to PIN the connection to the address that was
validated. These tests assert the address the transport actually connects to,
not merely that validation was called.

The DNS seam and its rules live in :mod:`tests.unit.dns_stub`.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from src.services.video_downloader import VideoDownloader
from tests.unit.dns_stub import PUBLIC_IP, RebindResolver, RecordingTransport


async def test_download_pins_the_validated_address(
    tmp_path: Path, rebind_dns: RebindResolver
) -> None:
    """A name that rebinds between validation and connect must NOT be reached.

    Without a pin the transport re-resolves ``rebind.example`` and lands on the
    internal address — the whole point of the attack.
    """
    transport = RecordingTransport(rebind_dns, {"/v.mp4": httpx.Response(200, content=b"MP4" * 8)})
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    dl = VideoDownloader(http_client=client)
    try:
        path = await dl.download(video_url="http://rebind.example/v.mp4", output_dir=tmp_path)
    finally:
        await client.aclose()

    assert transport.connected == [PUBLIC_IP], (
        f"connection landed on {transport.connected} — the validated address was "
        f"{PUBLIC_IP}; the DNS-rebind window is open"
    )
    # The pin must not break vhost routing or TLS: the original name is kept.
    assert transport.host_headers == ["rebind.example"]
    assert transport.url_hosts == [PUBLIC_IP]
    assert path.read_bytes() == b"MP4" * 8


async def test_every_redirect_hop_is_pinned(tmp_path: Path, rebind_dns: RebindResolver) -> None:
    """A pinned first hop with an unpinned redirect is not a fix."""
    transport = RecordingTransport(
        rebind_dns,
        {
            "/a": httpx.Response(302, headers={"location": "http://second.example/b"}),
            "/b": httpx.Response(200, content=b"OK" * 16),
        },
    )
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    dl = VideoDownloader(http_client=client)
    try:
        path = await dl.download(video_url="http://first.example/a", output_dir=tmp_path)
    finally:
        await client.aclose()

    assert transport.connected == [PUBLIC_IP, PUBLIC_IP], (
        f"redirect hops landed on {transport.connected}; every hop must be "
        f"re-validated AND re-pinned"
    )
    assert transport.host_headers == ["first.example", "second.example"]
    assert path.read_bytes() == b"OK" * 16


async def test_https_pin_preserves_sni_hostname(tmp_path: Path, rebind_dns: RebindResolver) -> None:
    """A pinned https fetch must still verify the cert against the HOSTNAME.

    httpx's ``sni_hostname`` extension drives both the ClientHello SNI and the
    certificate hostname check, so pinning the connect address does not
    downgrade TLS to "verify against an IP".
    """
    seen: list[str | None] = []

    class _Capturing(RecordingTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions.get("sni_hostname"))
            return await super().handle_async_request(request)

    transport = _Capturing(rebind_dns, {"/v.mp4": httpx.Response(200, content=b"X")})
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    dl = VideoDownloader(http_client=client)
    try:
        await dl.download(video_url="https://rebind.example/v.mp4", output_dir=tmp_path)
    finally:
        await client.aclose()

    assert seen == ["rebind.example"]
    assert transport.url_hosts == [PUBLIC_IP]


async def test_allowlisted_host_is_not_pinned(tmp_path: Path, rebind_dns: RebindResolver) -> None:
    """Presigned artifact URLs keep their hostname — pinning would break SigV4.

    ``minio.internal`` is on the operator allowlist, so the shared guard returns
    no pin and the request must go out with the original host untouched.
    """
    transport = RecordingTransport(
        rebind_dns, {"/bucket/clip.mp4": httpx.Response(200, content=b"C")}
    )
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    dl = VideoDownloader(http_client=client, allowed_hosts=frozenset({"minio.internal"}))
    try:
        await dl.download(
            video_url="http://minio.internal:8389/bucket/clip.mp4?X-Amz-Signature=x",
            output_dir=tmp_path,
        )
    finally:
        await client.aclose()

    assert transport.url_hosts == ["minio.internal"]
    assert transport.host_headers == ["minio.internal:8389"]
