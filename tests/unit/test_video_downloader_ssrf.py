"""SSRF protection tests for VideoDownloader and url_security."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from w2v_common.net import safe_fetch as _safe_fetch

from src.services.url_security import _is_blocked_address, validate_public_url
from src.services.video_downloader import VideoDownloader
from tests.unit.dns_stub import RebindResolver, pinned
from tests.unit.dns_stub import SocketShim as _SocketShim


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1/video.mp4",
        "http://localhost/video.mp4",
        "http://10.0.0.1/video.mp4",
        "http://192.168.1.1/video.mp4",
        "http://172.16.0.1/video.mp4",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/video.mp4",
        "http://[fe80::1]/video.mp4",
        "http://[::]/video.mp4",
        "http://postgres.internal:5432/video.mp4",
        "http://some-host.local/video.mp4",
        "http://api.cluster.local/video.mp4",
        "http://metadata.google.internal/",
        "http://metadata/",
    ],
)
def test_validate_public_url_blocks_private_targets(bad_url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(bad_url)


@pytest.mark.parametrize(
    "bad_url",
    [
        "ftp://example.com/video.mp4",
        "file:///etc/passwd",
        "gopher://example.com/",
        "http://",
        "",
    ],
)
def test_validate_public_url_blocks_bad_schemes_and_empty(bad_url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(bad_url)


def test_validate_public_url_blocks_dns_resolving_to_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(_safe_fetch, "socket", _SocketShim(fake_getaddrinfo))
    with pytest.raises(ValueError, match="private"):
        validate_public_url("http://attacker-controlled.example/video.mp4")


def test_validate_public_url_allows_public_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(_safe_fetch, "socket", _SocketShim(fake_getaddrinfo))
    validate_public_url("http://example.com/video.mp4")


def test_validate_public_url_allows_allowlisted_internal_host() -> None:
    # minio.internal is blocked by the `.internal` suffix (and would resolve to a
    # private IP); the operator allowlist short-circuits BOTH checks — no DNS is
    # even attempted (early return before _check_resolved_addresses).
    validate_public_url(
        "http://minio.internal:8389/w2v-orchestrator/clip.mp4?X-Amz-Signature=x",
        allowed_hosts=frozenset({"minio.internal"}),
    )


def test_validate_public_url_allowlist_is_exact_and_narrow() -> None:
    # A different internal host stays blocked even when another host is allowlisted.
    with pytest.raises(ValueError, match="internal/cluster host"):
        validate_public_url(
            "http://postgres.internal:5432/video.mp4",
            allowed_hosts=frozenset({"minio.internal"}),
        )


async def test_downloader_rejects_private_url(tmp_path: Path) -> None:
    dl = VideoDownloader()
    with pytest.raises(ValueError):
        await dl.download(video_url="http://127.0.0.1/video.mp4", output_dir=tmp_path)


@respx.mock
async def test_downloader_fetches_allowlisted_internal_host(tmp_path: Path) -> None:
    """An allowlisted internal store URL passes the SSRF gate and downloads — the
    record-stage QA path (minio.internal presigned clip) that this fix unblocks."""
    respx.get("http://minio.internal:8389/w2v-orchestrator/clip.mp4").mock(
        return_value=httpx.Response(200, content=b"MP4" * 8)
    )
    dl = VideoDownloader(allowed_hosts=frozenset({"minio.internal"}))
    path = await dl.download(
        video_url="http://minio.internal:8389/w2v-orchestrator/clip.mp4", output_dir=tmp_path
    )
    assert path.read_bytes() == b"MP4" * 8


async def test_downloader_rejects_metadata_url(tmp_path: Path) -> None:
    dl = VideoDownloader()
    with pytest.raises(ValueError):
        await dl.download(video_url="http://169.254.169.254/latest/meta-data/", output_dir=tmp_path)


@respx.mock
async def test_downloader_blocks_redirect_to_private_target(
    tmp_path: Path, stub_dns: RebindResolver
) -> None:
    """A 302 from a public URL to 169.254.169.254 must be blocked after redirect."""
    respx.get(pinned("http://evil.example.com/redirect")).mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/creds"})
    )
    dl = VideoDownloader()
    with pytest.raises(ValueError, match=r"private|reserved|link-local"):
        await dl.download(video_url="http://evil.example.com/redirect", output_dir=tmp_path)


@respx.mock
async def test_downloader_follows_public_redirect(tmp_path: Path, stub_dns: RebindResolver) -> None:
    respx.get(pinned("http://example.com/a")).mock(
        return_value=httpx.Response(302, headers={"location": "http://example.com/b"})
    )
    respx.get(pinned("http://example.com/b")).mock(
        return_value=httpx.Response(200, content=b"OK" * 16)
    )
    dl = VideoDownloader()
    path = await dl.download(video_url="http://example.com/a", output_dir=tmp_path)
    assert path.read_bytes() == b"OK" * 16


@respx.mock
async def test_downloader_caps_redirect_depth(tmp_path: Path, stub_dns: RebindResolver) -> None:
    for i in range(10):
        respx.get(pinned(f"http://example.com/{i}")).mock(
            return_value=httpx.Response(302, headers={"location": f"http://example.com/{i + 1}"})
        )
    dl = VideoDownloader()
    with pytest.raises(ValueError, match=r"[Tt]oo many redirects"):
        await dl.download(video_url="http://example.com/0", output_dir=tmp_path)


# ---------------------------------------------------------------------------
# RFC 6598 shared address space (100.64.0.0/10) == the Tailscale tailnet
# ---------------------------------------------------------------------------
#
# `ipaddress.ip_address("100.64.0.5").is_private` is False, and so are
# is_loopback / is_link_local / is_multicast / is_reserved / is_unspecified.
# `_is_blocked_address` was an OR over exactly those flags, so every machine on
# the estate (worker4 100.64.0.5, worker3 100.64.0.6, master
# 100.64.0.7, the mini/pgx GPU hosts) was judged "public" and a `video_url`
# aimed at any internal service sailed through the guard.

_TAILNET_ADDRS = (
    "100.64.0.1",  # bottom of the /10
    "100.64.0.5",  # worker4 — bare-metal postgres
    "100.64.0.6",  # worker3 — docker registry
    "100.64.0.7",  # master — k3s control plane
    "100.127.255.254",  # top of the /10
)


@pytest.mark.parametrize("addr", _TAILNET_ADDRS)
def test_cgnat_tailnet_literal_is_blocked(addr: str) -> None:
    assert _is_blocked_address(ipaddress.ip_address(addr)) is True


@pytest.mark.parametrize("addr", _TAILNET_ADDRS)
def test_cgnat_tailnet_url_is_rejected(addr: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(f"http://{addr}/video.mp4")


@pytest.mark.parametrize("addr", _TAILNET_ADDRS)
def test_cgnat_tailnet_via_dns_is_rejected(addr: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Split-DNS resolves internal names onto tailnet IPs that look public."""

    def _fake_gai(*_args: Any, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))]

    monkeypatch.setattr("w2v_common.net.safe_fetch.socket.getaddrinfo", _fake_gai)
    with pytest.raises(ValueError):
        validate_public_url("http://clips.example.com/video.mp4")


@pytest.mark.parametrize(
    "addr",
    [
        "::ffff:100.64.0.1",  # IPv4-mapped tailnet
        "::ffff:100.64.0.5",
        "2002:6471:2030::1",  # 6to4-wrapped 100.64.0.5
        "64:ff9b::6471:2030",  # NAT64 well-known prefix wrapping the tailnet
    ],
)
def test_ipv6_wrapped_tailnet_is_blocked(addr: str) -> None:
    assert _is_blocked_address(ipaddress.ip_address(addr)) is True


@pytest.mark.parametrize("addr", ["8.8.8.8", "93.184.216.34", "1.1.1.1"])
def test_public_ips_are_still_allowed(addr: str) -> None:
    assert _is_blocked_address(ipaddress.ip_address(addr)) is False


@pytest.mark.parametrize("addr", ["8.8.8.8", "93.184.216.34"])
def test_public_ip_literal_url_still_accepted(addr: str) -> None:
    validate_public_url(f"https://{addr}/video.mp4")  # must not raise


def test_public_neighbours_of_the_cgnat_block_still_allowed() -> None:
    # 100.63.255.255 and 100.128.0.0 bracket 100.64.0.0/10 — proof the deny is
    # the /10 and not a sloppy "anything starting with 100.".
    assert _is_blocked_address(ipaddress.ip_address("100.63.255.255")) is False
    assert _is_blocked_address(ipaddress.ip_address("100.128.0.0")) is False


def test_minio_allowlist_still_short_circuits_the_tailnet_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator allowlist is the intended escape hatch and must keep working.

    demo_video_record hands us minio.internal presigned URLs; ALLOWED_URL_HOSTS
    exempts them. Tightening the IP rules must not break that path.
    """
    validate_public_url(
        "http://minio.internal/clip.mp4", allowed_hosts=frozenset({"minio.internal"})
    )


# ---------------------------------------------------------------------------
# Regression cover for the switch to the shared w2v_common.net guard
# ---------------------------------------------------------------------------


def test_allowlist_still_bypasses_after_shared_guard_adoption() -> None:
    """The operator escape hatch survives delegation to w2v_common.net.

    Both the bare hostname and the shared guard's ``host:port`` form must
    short-circuit the name blocklist and the IP rules with no DNS attempted.
    """
    validate_public_url(
        "http://minio.internal:8389/bucket/clip.mp4",
        allowed_hosts=frozenset({"minio.internal"}),
    )
    validate_public_url(
        "http://minio.internal:8389/bucket/clip.mp4",
        allowed_hosts=frozenset({"minio.internal:8389"}),
    )


@pytest.mark.parametrize("addr", ["100.63.255.255", "100.128.0.0"])
def test_cgnat_boundary_neighbours_still_accepted_as_urls(addr: str) -> None:
    """The deny is the /10, not a sloppy ``startswith("100.")``.

    Both addresses bracket 100.64.0.0/10 and must survive the whole
    ``validate_public_url`` path, not just ``_is_blocked_address``.
    """
    validate_public_url(f"http://{addr}/video.mp4")  # must not raise
