"""Deterministic DNS seam for the SSRF/DNS-pinning tests.

``VideoDownloader`` now connects to the IP that ``validate_public_url``
resolved, so a test that mocks ``https://example.com/x`` no longer sees that
URL on the wire — it sees the pinned address. Tests therefore need to know what
the guard will resolve to, which means stubbing DNS rather than depending on
live resolution of ``example.com`` (which the old tests did, silently).

Two rules are baked in here because breaking either has cost this repo time:

1. **IP literals resolve to themselves.** A stub that maps every lookup to a
   fixed answer would happily re-point a *pinned* request and hide the bug the
   pin is supposed to fix.
2. **``socket.getaddrinfo`` is never patched globally.** Only the ``socket``
   name binding inside :mod:`w2v_common.net.safe_fetch` is swapped — the
   narrowest seam that covers the guard, leaving unrelated machinery
   (asyncio, psycopg, httpx transports) on the real resolver.

``8.8.8.8`` is the public fixture: TEST-NET-3 (``203.0.113.0/24``) is blocked by
this fleet's validator and ``100.64.0.0/10`` is the tailnet, i.e. internal.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

#: Answer returned for the VALIDATION lookup — a routable public address.
PUBLIC_IP = "8.8.8.8"
#: Answer returned for later lookups of a rebinding name — a k8s ClusterIP.
INTERNAL_IP = "10.0.0.7"


def pinned(url: str, ip: str = PUBLIC_IP) -> str:
    """Return ``url`` with its host rewritten to ``ip`` — the URL that hits the wire."""
    parsed = urlparse(url)
    host = f"[{ip}]" if ":" in ip else ip
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


class RebindResolver:
    """A ``getaddrinfo`` stub whose answer flips after the first lookup.

    The first lookup of a name in ``rebinding`` answers :data:`PUBLIC_IP` (so
    validation passes); every later lookup of that name answers
    :data:`INTERNAL_IP`. Names outside ``rebinding`` always answer
    :data:`PUBLIC_IP`, and IP literals always resolve to themselves.
    """

    def __init__(self, rebinding: frozenset[str] = frozenset()) -> None:
        self._rebinding = rebinding
        self.lookups: list[str] = []

    def getaddrinfo(self, host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            # A literal resolves to itself: a pinned connection stays pinned.
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 0))]
        self.lookups.append(host)
        rebound = host in self._rebinding and self.lookups.count(host) > 1
        answer = INTERNAL_IP if rebound else PUBLIC_IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answer, 0))]


class SocketShim:
    """Stand-in for the ``socket`` module with only ``getaddrinfo`` replaced."""

    def __init__(self, getaddrinfo: Callable[..., list[Any]]) -> None:
        self.getaddrinfo = getaddrinfo

    def __getattr__(self, name: str) -> Any:
        return getattr(socket, name)


class RecordingTransport(httpx.AsyncBaseTransport):
    """Transport that resolves the request URL host the way a real connect does.

    It records the address it would have connected to, the ``Host`` header and
    the URL host, so a test can assert the SOCKET TARGET rather than merely
    that validation was called. Responses are routed by URL path.
    """

    def __init__(self, resolver: RebindResolver, routes: dict[str, httpx.Response]) -> None:
        self._resolver = resolver
        self._routes = routes
        self.connected: list[str] = []
        self.host_headers: list[str | None] = []
        self.url_hosts: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.url_hosts.append(host)
        # This is the connect-time resolution the pin is supposed to remove.
        self.connected.append(str(self._resolver.getaddrinfo(host, request.url.port)[0][4][0]))
        self.host_headers.append(request.headers.get("host"))
        response = self._routes[request.url.path]
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )
