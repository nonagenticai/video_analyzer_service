"""Tenant-id coercion for this service — the one place a ``str`` becomes a ``uuid.UUID``.

w2v-common 0.67.0 made tenant identity a real ``uuid.UUID`` and REMOVED the
nil-UUID substitution that used to happen when a request carried no tenant.
This module is the service-local boundary that enforces the same ruling:

* a tenant that is absent or unparseable raises (``MissingTenantError`` /
  ``InvalidTenantIdError`` from :mod:`w2v_common.tenant`);
* the all-zeros UUID is rejected **by value**, not by identity of its string
  form — ``uuid.UUID`` accepts dashed, undashed, braced and ``urn:uuid:``
  spellings of the same nil value, and w2v-common's older trusted-network path
  wrote exactly that value, so a ``None``-only guard leaves the hole open.

There is no fallback tenant. Substituting one is what mis-attributed cost.
"""

from __future__ import annotations

import uuid

from w2v_common.tenant import InvalidTenantIdError, parse_tenant_id

__all__ = ["require_tenant"]


def require_tenant(value: object, *, source: str = "tenant_id") -> uuid.UUID:
    """Return ``value`` as a real, non-nil ``uuid.UUID`` or raise.

    Raises:
        MissingTenantError: no tenant id at all.
        InvalidTenantIdError: not a UUID, or the nil UUID sentinel.
    """
    parsed = parse_tenant_id(value, source=source)
    if parsed.int == 0:
        raise InvalidTenantIdError(
            f"{source}: nil UUID is not a tenant — every row must carry a real registered tenant"
        )
    return parsed
