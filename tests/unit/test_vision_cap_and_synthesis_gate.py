"""Two money defects: an INERT operator cap, and a metered stage nobody consumes.

DEFECT 1 — ``VISION_MAX_COST_USD`` was defined in ``VisionSettings`` and read
NOWHERE in ``src/``. Every enforcement site gated on the per-request
``max_vision_cost_usd``, which defaults to ``None``, so an operator reading a
ConfigMap that says ``5.0`` was protected by nothing and every orchestrator
submit (which never sends the field) ran uncapped.

DEFECT 3 — the synthesis LLM stage ran unconditionally while every sibling
optional stage was flag-gated. Its only consumer is the walkthrough artifact, so
a submit with the render flags off paid for one discarded metered
worker_management call per clip, folded invisibly into ``vision_cost_usd``.
"""

from __future__ import annotations

import inspect
from typing import cast

import pytest

from src.api.schemas import AnalysisRequest
from src.config import Settings
from src.services.pipeline import (
    PipelineDependencies,
    PipelineRunner,
    _effective_vision_cap,
    _enabled_stages,
)

_URL = "https://example.com/v.mp4"


def _settings(operator_cap: float) -> Settings:
    settings = Settings()
    settings.vision.max_cost_usd = operator_cap
    return settings


# --------------------------------------------------------------------------
# DEFECT 1 — the cap rule
# --------------------------------------------------------------------------


def test_operator_cap_applies_when_the_request_omits_one() -> None:
    """THE defect: the orchestrator never sends the field, so this was ``None``."""
    request = AnalysisRequest(video_url=_URL)
    assert request.max_vision_cost_usd is None
    assert _effective_vision_cap(request, _settings(5.0)) == 5.0


def test_a_caller_may_lower_the_operator_cap() -> None:
    request = AnalysisRequest(video_url=_URL, max_vision_cost_usd=1.25)
    assert _effective_vision_cap(request, _settings(5.0)) == 1.25


def test_a_caller_may_not_raise_the_operator_cap() -> None:
    """A cap a caller can raise is not a cap. The request body is untrusted."""
    request = AnalysisRequest(video_url=_URL, max_vision_cost_usd=9999.0)
    assert _effective_vision_cap(request, _settings(5.0)) == 5.0


def test_a_non_positive_operator_cap_means_no_ceiling() -> None:
    """The only way to exceed the configured number — and it takes an operator."""
    assert _effective_vision_cap(AnalysisRequest(video_url=_URL), _settings(0.0)) is None
    assert (
        _effective_vision_cap(
            AnalysisRequest(video_url=_URL, max_vision_cost_usd=42.0), _settings(0.0)
        )
        == 42.0
    )


@pytest.mark.parametrize(
    "method",
    ["_run_vision_stage", "_execute"],
)
def test_no_enforcement_site_reads_the_raw_request_field(method: str) -> None:
    """The fix must not be INERT.

    A resolver that is correct but never called is exactly the defect being fixed
    here — ``max_cost_usd`` was itself a correct field with zero read sites. So
    pin the call sites: no cap site may read ``request.max_vision_cost_usd``
    directly, because that expression is the one that bypasses the operator
    ceiling.
    """
    source = inspect.getsource(getattr(PipelineRunner, method))
    assert "request.max_vision_cost_usd" not in source, (
        f"{method} still gates on the raw per-request cap, bypassing the operator ceiling"
    )
    assert "_effective_vision_cap(" in source, f"{method} no longer resolves the enforced cap"


def test_the_configured_default_is_still_five_dollars() -> None:
    """The ConfigMap value the operator is reading. Now load-bearing."""
    assert Settings().vision.max_cost_usd == 5.0


# --------------------------------------------------------------------------
# DEFECT 3 — the unconditional metered stage
# --------------------------------------------------------------------------


def test_synthesis_is_gated_on_its_only_consumers_flag() -> None:
    enabled = _enabled_stages(AnalysisRequest(video_url=_URL, generate_walkthrough=True))
    assert "synthesis" in enabled

    disabled = _enabled_stages(AnalysisRequest(video_url=_URL, generate_walkthrough=False))
    assert "synthesis" not in disabled, (
        "synthesis still runs with no consumer — a metered LLM call per clip, discarded"
    )


def test_all_render_flags_off_still_drops_synthesis() -> None:
    """The literal ``_ANALYZER_OVERRIDES`` trigger shape."""
    request = AnalysisRequest(
        video_url=_URL,
        generate_walkthrough=False,
        generate_ui_inventory=False,
        generate_frame_index=False,
    )
    assert "synthesis" not in _enabled_stages(request)


class _RecordingProvider:
    """Fails the test loudly if the metered call is made."""

    def __init__(self) -> None:
        self.calls = 0

    async def synthesize(self, **_kwargs: object) -> None:
        self.calls += 1


def _runner_with(provider: _RecordingProvider) -> PipelineRunner:
    """A runner whose ONLY live dependency is the vision provider.

    ``_run_synthesis_stage`` touches nothing else, so building the full
    ``PipelineDependencies`` (ffmpeg, storage, a transcription client, a DB) would
    exercise the fakes rather than the gate. Cast rather than silence the checker.
    """
    runner = PipelineRunner.__new__(PipelineRunner)
    runner._deps = cast(PipelineDependencies, _StubDeps(provider))
    return runner


class _StubDeps:
    def __init__(self, provider: _RecordingProvider) -> None:
        self.vision_provider = provider


@pytest.mark.asyncio
async def test_disabled_synthesis_makes_no_provider_call() -> None:
    """The money assertion: not merely 'absent from a list' but 'not billed'."""
    provider = _RecordingProvider()
    runner = _runner_with(provider)

    result = await runner._run_synthesis_stage([], "transcript", enabled=False)

    assert result is None
    assert provider.calls == 0, "the metered synthesis call was made for a disabled stage"


@pytest.mark.asyncio
async def test_enabled_synthesis_still_calls_the_provider() -> None:
    """Both directions: a gate stuck closed silently removes the walkthrough."""
    provider = _RecordingProvider()
    runner = _runner_with(provider)

    await runner._run_synthesis_stage([], "transcript", enabled=True)

    assert provider.calls == 1, "synthesis no longer runs when its consumer IS enabled"


@pytest.mark.asyncio
async def test_a_skipped_stage_reports_no_progress_and_consumes_no_index() -> None:
    """``total_stages`` excludes synthesis when skipped, so the index must not advance."""
    reported: list[tuple[str, int]] = []

    async def _progress(name: str, index: int) -> None:
        reported.append((name, index))

    provider = _RecordingProvider()
    runner = _runner_with(provider)

    synthesis, index = await runner._synthesis_stage_with_progress(
        enabled=False,
        frame_vision=[],
        transcript_text=None,
        stage_index=7,
        progress=_progress,
    )
    assert synthesis is None
    assert index == 7
    assert reported == []

    _, index = await runner._synthesis_stage_with_progress(
        enabled=True,
        frame_vision=[],
        transcript_text=None,
        stage_index=7,
        progress=_progress,
    )
    assert index == 8
    assert reported == [("synthesis", 8)]
