"""SSRF protection: validate that a URL targets a public host.

The IP-range judgement lives in :mod:`w2v_common.net.safe_fetch` — the one
fleet-wide guard. This module used to carry its own hand-rolled copy of those
rules; six services carried near-identical copies that drifted apart, and that
drift is what produced a fleet-wide defect class. What remains here is only the
service-local policy the shared guard deliberately does not own:

- the operator escape hatch ``allowed_hosts`` (``ALLOWED_URL_HOSTS``), mapped
  onto the library's ``allow_hosts`` parameter;
- a name-based blocklist for estate/cloud-metadata hostnames — the library's
  own list is narrower and its main defence is IP-based, so this is an extra
  layer, not a duplicate;
- a residual ``is_unspecified`` check (``0.0.0.0`` / ``::``) applied to literal
  hosts and to the address the library pins to.

The library's ``validate_public_url`` returns the address it resolved precisely
so the caller can PIN the connection to it. This wrapper returns that pin too:
swallowing it (as it did until the DNS-rebind fix) re-opens the TOCTOU window
in every caller, because the transport then resolves the hostname a second time
and an attacker-controlled zone can answer differently on the second lookup.

Error messages are kept in the legacy wording so callers (and the SSRF test
suite) keep matching on "private" / "internal/cluster host".
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from w2v_common.net import validate_public_url as _validate_public_url
from w2v_common.net.safe_fetch import SSRFError as _SSRFError
from w2v_common.net.safe_fetch import _is_blocked_ip

_BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
    ".internal",
    ".local",
    ".localhost",
    ".cluster.local",
    ".svc",
)

# Name-based layer kept deliberately: the shared guard is IP-based and its own
# exact-name list is narrower (no bare ``metadata`` / ``instance-data``).
_BLOCKED_EXACT_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata",
        "instance-data",
    }
)


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if the address is not routable on the public internet.

    Delegates the range rules (private, loopback, link-local, multicast,
    reserved, RFC 6598 CGNAT/tailnet, NAT64/6to4/Teredo-wrapped IPv4, ...) to
    the shared :func:`w2v_common.net.safe_fetch._is_blocked_ip`, then applies
    the residual unspecified check the shared list does not express as such.
    """
    return addr.is_unspecified or _is_blocked_ip(str(addr))


def _validate_scheme_and_host(url: str) -> str:
    """Parse *url*, ensure http(s) scheme and a non-empty hostname; return hostname."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme: {url!r}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError(f"URL must have a valid hostname: {url!r}")
    return hostname


def _check_hostname_blocklist(hostname: str, url: str) -> None:
    if hostname in _BLOCKED_EXACT_HOSTS:
        raise ValueError(f"URL targets a blocked host: {url!r}")
    for suffix in _BLOCKED_HOST_SUFFIXES:
        if hostname.endswith(suffix):
            raise ValueError(f"URL targets an internal/cluster host: {url!r}")


def _is_allowlisted(url: str, hostname: str, allowed_hosts: frozenset[str]) -> bool:
    """True if *hostname* (bare or ``host:port``) is on the operator allowlist."""
    if not allowed_hosts:
        return False
    if hostname in allowed_hosts:
        return True
    port = urlparse(url).port
    return port is not None and f"{hostname}:{port}" in allowed_hosts


def _delegate_to_shared_guard(url: str, allowed_hosts: frozenset[str]) -> str | None:
    """Run the shared fleet guard, re-raising with this module's legacy wording.

    Returns the resolved IP the caller must pin the connection to (``None`` for
    an allow-listed host, which is fetched un-pinned so a presigned
    ``Host``/SigV4 signature survives byte-for-byte).
    """
    try:
        pinned = _validate_public_url(url, allow_hosts=allowed_hosts)
    except _SSRFError as exc:
        if "DNS" in str(exc):
            raise ValueError(f"URL hostname could not be resolved: {url!r}") from exc
        raise ValueError(
            f"URL resolves to a private/loopback/reserved address: {url!r} ({exc})"
        ) from exc

    if pinned is None:
        return None
    try:
        resolved = ipaddress.ip_address(pinned)
    except ValueError as exc:
        raise ValueError(f"URL resolved to an unparseable address {pinned!r}") from exc
    # Residual: the shared block-list does not test ``is_unspecified`` as such.
    if resolved.is_unspecified:
        raise ValueError(f"URL resolves to a private/loopback/reserved address ({pinned}): {url!r}")
    return pinned


def validate_public_url(url: str, allowed_hosts: frozenset[str] = frozenset()) -> str | None:
    """Validate ``url`` and return the IP the caller must pin the connection to.

    Raises ``ValueError`` if ``url`` does not target a public HTTP(S) host.

    Blocks private, loopback, link-local, multicast, reserved, and
    unspecified IPs (both literal and DNS-resolved), plus ``.internal`` /
    ``.local`` / cluster-DNS suffixes and the cloud metadata endpoints.

    ``allowed_hosts`` is an operator-configured set of EXACT hostnames trusted
    despite resolving to an internal/private address — e.g. the internal object
    store ``minio.internal`` whose presigned clip URLs the orchestrator's
    demo_video_record QA hands us to fetch. A match short-circuits every check
    (suffix blocklist AND resolved-IP), so keep it to the minimum trusted set.
    The ``host:port`` form is also accepted (the shared guard's convention).

    Returns:
        The resolved IP address to pin the connection to, or ``None`` when
        there is nothing to pin: an allow-listed host (fetched with its
        original hostname so a presigned signature stays valid) or a URL whose
        host is already an IP literal (no name, so no rebind is possible).
        Callers MUST use this value — see :meth:`VideoDownloader._download_via_http`.
    """
    hostname = _validate_scheme_and_host(url)
    if _is_allowlisted(url, hostname, allowed_hosts):
        return None
    _check_hostname_blocklist(hostname, url)

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_address(literal):
            raise ValueError(f"URL targets a private/loopback/reserved address: {url!r}")
        return None

    return _delegate_to_shared_guard(url, allowed_hosts)
