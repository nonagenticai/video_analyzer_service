"""Unit-test fixtures shared by the SSRF / DNS-pinning suites."""

from __future__ import annotations

import pytest
from w2v_common.net import safe_fetch as _safe_fetch

from tests.unit.dns_stub import RebindResolver, SocketShim


def _install(monkeypatch: pytest.MonkeyPatch, resolver: RebindResolver) -> RebindResolver:
    # Narrowest seam: swap the ``socket`` NAME BINDING inside the shared guard.
    # Patching ``socket.getaddrinfo`` itself is global and breaks unrelated
    # machinery in this suite.
    monkeypatch.setattr(_safe_fetch, "socket", SocketShim(resolver.getaddrinfo))
    return resolver


@pytest.fixture
def stub_dns(monkeypatch: pytest.MonkeyPatch) -> RebindResolver:
    """Every hostname resolves to ``PUBLIC_IP``; IP literals resolve to themselves."""
    return _install(monkeypatch, RebindResolver())


@pytest.fixture
def rebind_dns(monkeypatch: pytest.MonkeyPatch) -> RebindResolver:
    """``rebind.example`` / ``second.example`` answer public once, then internal."""
    return _install(monkeypatch, RebindResolver(frozenset({"rebind.example", "second.example"})))
