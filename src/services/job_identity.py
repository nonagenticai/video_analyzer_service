"""Persist and restore the submitting request's identity for a re-claimed job.

A job submitted on pod A and re-claimed on pod B (A died mid-run) has no request in
scope on B. Its vision and synthesis calls go to ``worker_management``, a METERED
destination, and ``build_outbound_identity`` refuses to delegate a tenant whose
provenance is ``None`` ("unrecorded"). So a re-claim that only rebinds the tenant would
fail at its first metered call — and would also bill the wrong actor/project.

The shape is copied from ``video_compiler_service`` ``services/durable_dispatch.py``
(``build_job_options`` / ``bind_job_attribution`` / ``restore_tenant_source``), which
solved the same problem for crash-orphaned renders.

What is stored is attribution, never a credential: the caller's raw licence token is
deliberately NOT persisted. The re-run delegates through this service's own licence on
the provenance recorded here, exactly as a background task on the submitting pod does.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from w2v_common.auth import (
    ACTOR_ID_HEADER,
    PROJECT_ID_HEADER,
    W2V_JOB_ID_HEADER,
    W2V_PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    get_project_id_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
)
from w2v_common.auth.identity_context import (
    get_correlation_id_from_context,
    get_license_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_source_from_context,
    mark_inbound_request,
    resolve_actor_id,
    set_correlation_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

logger = logging.getLogger(__name__)


def capture_dispatch_identity() -> dict[str, str]:
    """Snapshot the CURRENT request's identity for storage on the job row.

    Called by the submitting route, inside the request. ``""`` means "absent" for every
    key, so the snapshot round-trips through JSONB without ``None``/missing ambiguity.
    """
    source = get_tenant_source_from_context()
    return {
        "license_id": get_license_id_from_context(),
        "subject_id": get_subject_id_from_context(),
        "tenant_source": source.value if source is not None else "",
        "actor_id": resolve_actor_id()[0] or "",
        "project_id": get_project_id_from_context(),
        "w2v_project_id": get_w2v_project_id_from_context(),
        "w2v_job_id": get_w2v_job_id_from_context(),
        "correlation_id": get_correlation_id_from_context(),
        "stage": get_stage_from_context(),
    }


def restore_tenant_source(value: object) -> TenantSource | None:
    """Re-hydrate the stored provenance; ``None`` (never a guess) when absent/unknown.

    ``None`` fails CLOSED: the delegation gate refuses an unvouched tenant, so a row that
    predates migration 007 raises at its first metered call instead of spending a tenant's
    credit on provenance this process cannot vouch for.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return TenantSource(value)
    except ValueError:
        logger.warning("Unrecognised persisted tenant_source %r; treating as unvouched", value)
        return None


def bind_dispatch_identity(tenant_id: str, identity: Mapping[str, object] | None) -> None:
    """Re-bind the submitting request's identity in the CURRENT context.

    Must run inside the re-claimed job's own task (``asyncio.create_task`` copies the
    context, so this never bleeds into the sweeper that spawned it). Provenance comes
    from the PERSISTED snapshot, never the ambient ContextVar — the sweeper claims rows
    across tenants, so the ambient value would vouch one tenant's provenance for another.
    """
    ident = identity or {}

    def _s(key: str) -> str:
        value = ident.get(key)
        return value if isinstance(value, str) else ""

    set_identity_context(
        tenant_id,
        _s("license_id") or None,
        _s("subject_id") or None,
        tenant_source=restore_tenant_source(ident.get("tenant_source")),
    )
    # Caller-originated work: a stored actor bills that actor; none stays actor-less
    # (counted inbound_without_actor) rather than falling back to SERVICE_ACTOR_ID.
    mark_inbound_request()
    headers = {
        ACTOR_ID_HEADER: _s("actor_id"),
        PROJECT_ID_HEADER: _s("project_id"),
        W2V_PROJECT_ID_HEADER: _s("w2v_project_id"),
        W2V_JOB_ID_HEADER: _s("w2v_job_id"),
    }
    bind_attribution_from_headers({k: v for k, v in headers.items() if v}, trusted=True)
    set_correlation_context(_s("correlation_id") or None, _s("stage") or None)
