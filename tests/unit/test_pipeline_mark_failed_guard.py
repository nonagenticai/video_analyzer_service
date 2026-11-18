"""Regression: a DB blip while writing terminal state must not mask the real failure.

``PipelineRunner._run_job`` catches every stage failure in one ``except Exception``
and then tries to persist it via ``get_session_maker`` -> ``bind_session_tenant`` ->
``JobRepository.mark_failed``. Before this fix that write was unguarded: if IT
raised (DB blip, tenant-bind failure), the new exception replaced the real stage
exception on whatever awaited the task, and the terminal-state write was skipped
entirely — the job row stayed stuck in its last non-terminal status, reconciled
only later (and with a generic message) by ``stranded_sweeper``.

No database is touched here — ``get_session_maker`` and the real stage work are
both monkeypatched to distinctive exceptions, so this runs without postgres.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest

from src.services import pipeline as pipeline_mod
from src.services.pipeline import PipelineDependencies, PipelineRunner


class _RealStageError(RuntimeError):
    """Stand-in for a genuine stage failure (ffmpeg/OCR/vision/DB read)."""


class _DBWriteError(RuntimeError):
    """Stand-in for the terminal-state write itself failing."""


def _make_runner() -> PipelineRunner:
    settings = pipeline_mod.get_settings()
    deps = PipelineDependencies(
        storage=object(),
        settings=settings,
        frame_extractor=cast(Any, object()),
        audio_extractor=cast(Any, object()),
        video_downloader=cast(Any, object()),
        scene_detector=cast(Any, object()),
        ocr_service=cast(Any, object()),
        quality_analyzer=cast(Any, object()),
        transcription_client=cast(Any, object()),
        vision_provider=object(),
        probe=object(),
    )
    return PipelineRunner(dependencies=deps, instance_id="test")


async def test_a_db_blip_writing_failed_status_does_not_mask_the_real_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The real stage exception must survive a broken mark_failed write.

    Pre-fix, ``_run_job`` raised ``_DBWriteError`` out of the task (superseding
    ``_RealStageError``) and never touched the job row. Post-fix, the task
    completes normally (the terminal-state guarantee), both exceptions are
    logged, and the real exception's diagnosis is what's on record.
    """
    runner = _make_runner()

    async def _claim_noop(job_id: str, tenant_id: str) -> None:
        return None

    async def _boom_real_work(job_id: str, tenant_id: str) -> None:
        raise _RealStageError(f"real stage blew up for {job_id}")

    async def _boom_get_session_maker():
        raise _DBWriteError("DB unreachable while writing terminal state")

    monkeypatch.setattr(runner, "_claim", _claim_noop)
    monkeypatch.setattr(runner, "_run_execute_with_deadline", _boom_real_work)
    monkeypatch.setattr(pipeline_mod, "get_session_maker", _boom_get_session_maker)

    with caplog.at_level(logging.ERROR, logger="src.services.pipeline"):
        # The terminal-state guarantee: this must not raise, even though the
        # write that records the failure itself blew up.
        await asyncio.wait_for(runner._run_job("job-1", "tenant-1"), timeout=5)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Pipeline failed for job job-1" in m for m in messages), (
        "the real stage exception must still be logged"
    )
    assert any("Could not persist failed status for job job-1" in m for m in messages), (
        "the secondary DB failure must be logged too, not silently swallowed"
    )
    # The real exception's traceback (not just the DB one) must actually be in
    # the log record that names the pipeline failure.
    real_failure_record = next(
        r for r in caplog.records if "Pipeline failed for job job-1" in r.getMessage()
    )
    # exc_info is typed `... | Literal[False] | None`; truthiness narrows out both.
    assert real_failure_record.exc_info
    assert real_failure_record.exc_info[0] is _RealStageError
