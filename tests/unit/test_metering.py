"""Unit tests for the video-analyzer cost-event emitter.

The emit/field/stream/lifecycle logic now lives in ``w2v_common.metering``;
this service only pins its fixed identity and wraps the GC-safe
``schedule_cost_event``. These tests assert the service-specific identity is
injected correctly and that the wrapper stays fire-and-forget.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from w2v_common.metering import MissingAttributionError

from src.services.metering import (
    APP_ID,
    COST_SOURCE,
    FUNDING,
    PROVIDER,
    SERVED_MODEL,
    SERVICE_KEY,
    analysis_event_key,
    close_metering_redis,
    init_metering_redis,
    resolve_stream,
    schedule_cost_event,
)


async def _drain() -> None:
    """Let the fire-and-forget emit task run to completion."""
    for _ in range(10):
        await asyncio.sleep(0)


# 0.67.0: a cost event's tenant_id must be a real UUID — a slug now raises
# InvalidTenantIdError at emit time instead of being billed under a fake name.
_TENANT = "55555555-5555-5555-5555-555555555555"
_LICENSE = "0199f0a1-0000-7000-8000-000000000001"
_SUBJECT = "0199f0a1-0000-7000-8000-000000000002"
_JOB = "job-0199f0a1"


@pytest.mark.asyncio
async def test_schedule_cost_event_video_analyzer_shape() -> None:
    redis = AsyncMock()
    # 120s of source video -> input_tokens = 120; $0.017/min hint = 0.034.
    schedule_cost_event(
        redis,
        tenant_id=_TENANT,
        job_id=_JOB,
        license_id=_LICENSE,
        subject_id=_SUBJECT,
        input_tokens=120,
        output_tokens=0,
        latency_ms=8200,
        cost_usd=120 / 60 * 0.017,
        usage_extra={"unit": "video_seconds"},
    )
    await _drain()

    redis.xadd.assert_awaited_once()
    args = redis.xadd.await_args
    assert args.args[0] == resolve_stream()
    fields: dict[str, str] = args.args[1]
    # Every value is a string at the Redis layer.
    assert all(isinstance(v, str) for v in fields.values())
    assert fields["tenant_id"] == _TENANT
    assert fields["license_id"] == _LICENSE
    assert fields["subject_id"] == _SUBJECT
    # G5: full attribution stated by the emitter, never left to billing.
    assert fields["app_id"] == APP_ID == "w2v"
    assert fields["funding"] == FUNDING == "local_gpu"
    assert fields["event_key"] == analysis_event_key(_JOB)
    assert fields["provider"] == PROVIDER == "video-analyzer"
    assert fields["model"] == fields["served_model"] == SERVED_MODEL == "local"
    assert fields["input_tokens"] == "120"
    assert fields["output_tokens"] == "0"
    assert fields["cost_usd"] == "0.03400000"
    assert fields["latency_ms"] == "8200"
    assert fields["cost_source"] == COST_SOURCE == "w2v-video-analyzer"
    assert fields["service_key"] == SERVICE_KEY == "w2v-video-analyzer"
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["unit"] == "video_seconds"
    assert args.kwargs["maxlen"] == 100_000
    assert args.kwargs["approximate"] is True


@pytest.mark.asyncio
async def test_schedule_cost_event_swallows_redis_errors() -> None:
    redis = AsyncMock()
    redis.xadd.side_effect = RuntimeError("redis went away")
    # Must not raise; emit is fire-and-forget and the helper swallows.
    schedule_cost_event(
        redis,
        tenant_id=_TENANT,
        job_id=_JOB,
        license_id=_LICENSE,
        subject_id=_SUBJECT,
        input_tokens=1,
        output_tokens=0,
        latency_ms=1,
    )
    await _drain()
    redis.xadd.assert_awaited_once()


@pytest.mark.asyncio
async def test_init_metering_redis_empty_url_returns_none() -> None:
    assert await init_metering_redis("") is None


@pytest.mark.asyncio
async def test_close_metering_redis_none_is_noop() -> None:
    await close_metering_redis(None)  # must not raise


def test_analysis_event_key_is_deterministic_per_job() -> None:
    """A re-emit of the same job dedupes at billing; two jobs never collide."""
    assert analysis_event_key(_JOB) == analysis_event_key(_JOB)
    assert analysis_event_key(_JOB) != analysis_event_key("job-other")
    assert len(analysis_event_key(_JOB)) <= 64


def _emit(**overrides: Any) -> AsyncMock:
    redis = AsyncMock()
    kwargs: dict[str, Any] = dict(
        tenant_id=_TENANT,
        job_id=_JOB,
        license_id=_LICENSE,
        subject_id=_SUBJECT,
        input_tokens=60,
        output_tokens=0,
        latency_ms=10,
    )
    kwargs.update(overrides)
    schedule_cost_event(redis, **kwargs)
    return redis


@pytest.mark.asyncio
async def test_strict_attribution_full_event_is_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """With METERING_REQUIRE_ATTRIBUTION on, a fully attributed event goes out."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    redis = _emit()
    await _drain()
    redis.xadd.assert_awaited_once()
    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["app_id"] == "w2v"
    assert fields["funding"] == "local_gpu"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["license_id", "subject_id"])
async def test_strict_attribution_refuses_a_partial_event(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Fail loud: a missing licence/subject raises instead of shipping a partial row."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError, match=field):
        _emit(**{field: ""})


@pytest.mark.asyncio
async def test_pipeline_emit_cost_carries_job_key_and_bound_identity() -> None:
    """The live call site (PipelineRunner._emit_cost) states every field."""
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        set_identity_context,
    )

    from src.services.pipeline import PipelineRunner

    runner = PipelineRunner.__new__(PipelineRunner)
    redis = AsyncMock()
    runner._metering_redis = redis
    set_identity_context(_TENANT, _LICENSE, _SUBJECT)
    try:
        runner._emit_cost(tenant_id=_TENANT, job_id=_JOB, duration_seconds=90.4, latency_ms=5)
        await _drain()
    finally:
        clear_identity_context()
    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["license_id"] == _LICENSE
    assert fields["subject_id"] == _SUBJECT
    assert fields["app_id"] == "w2v"
    assert fields["funding"] == "local_gpu"
    assert fields["event_key"] == analysis_event_key(_JOB)
    assert fields["input_tokens"] == "90"
