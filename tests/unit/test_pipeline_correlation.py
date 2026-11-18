"""Regression: the detached pipeline task must inherit the caller's identity +
cost-correlation ContextVars.

``PipelineRunner.enqueue`` dispatches ``_run_job`` via a plain
``asyncio.create_task(coro)``, which snapshots the current contextvars Context
at creation time. That snapshot is what carries the request's tenant /
correlation_id / stage into the background pipeline so its downstream WM/IR cost
events attribute correctly. This test pins that contract so a future refactor
(e.g. moving the work to a thread/process pool or passing a blank ``context=``)
cannot silently drop the context and NULL the correlation.

Mirrors content_service
tests/unit/narrative/test_narrative_routes_endpoints.py
::TestStartStoryline::test_start_storyline_propagates_correlation_and_tenant_to_bg_task.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from src.services.pipeline import PipelineRunner

_TENANT = uuid.UUID("11111111-2222-3333-4444-555555555555")


@pytest.mark.asyncio
async def test_enqueue_propagates_correlation_and_tenant_to_bg_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from w2v_common.auth import (
        clear_identity_context,
        get_correlation_id_from_context,
        get_stage_from_context,
        get_tenant_id_from_context,
        set_correlation_context,
        set_identity_context,
    )

    # MagicMock dependencies short-circuit the heavy default-deps construction;
    # ``_run_job`` is patched out so nothing touches the DB / external services.
    runner = PipelineRunner(dependencies=MagicMock())

    seen: dict[str, object] = {}

    async def _spy(*_args: object, **_kwargs: object) -> None:
        seen["correlation_id"] = get_correlation_id_from_context()
        seen["stage"] = get_stage_from_context()
        seen["tenant_id"] = get_tenant_id_from_context()

    # setattr (not a direct attribute assignment) so the static type checker
    # doesn't reject the spy's signature against the real _run_job method type.
    monkeypatch.setattr(runner, "_run_job", _spy)

    set_identity_context(_TENANT, "lic-1", "subj-1")
    set_correlation_context("run-xyz", "video_analyze")
    try:
        await runner.enqueue("job-1", str(_TENANT))
        # Let the scheduled background task run to completion.
        await asyncio.gather(*list(runner._tasks.values()))
    finally:
        clear_identity_context()
        set_correlation_context(None, None)

    assert seen["correlation_id"] == "run-xyz"
    assert seen["stage"] == "video_analyze"
    # w2v-common 0.67.0: the context var holds a real uuid.UUID, never a slug
    # and never the nil sentinel. Still proves the tenant propagated.
    assert seen["tenant_id"] == _TENANT
