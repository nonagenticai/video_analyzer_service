"""Tests for src.services.vision.wm_provider.WMVisionProvider."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from w2v_common.clients.wm_client import WMResponse

from src.services.vision.base import (
    FrameInput,
    VisionProviderError,
    VisionProviderUnavailableError,
)
from src.services.vision.wm_provider import (
    WMVisionProvider,
    _parse_frame_output,
    _parse_synthesis_output,
)

# 1x1 transparent PNG, base64-encoded.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture
def tiny_jpeg(tmp_path: Path) -> Path:
    """Write a tiny JPEG-shaped file to disk for FrameInput.path."""
    f = tmp_path / "frame.jpg"
    f.write_bytes(base64.b64decode(_TINY_PNG_B64))
    return f


def _wm_resp(output: object, *, usage: dict | None = None) -> WMResponse:
    return WMResponse(
        output=output,
        output_type="object" if isinstance(output, dict) else "string",
        output_schema_used=None,
        metadata=None,
        usage_metrics=usage,
    )


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


class TestParseFrameOutput:
    def test_dict_output(self):
        d, labels = _parse_frame_output({"description": "a button", "labels": ["btn", "ui"]})
        assert d == "a button"
        assert labels == ["btn", "ui"]

    def test_dict_output_with_only_description(self):
        d, labels = _parse_frame_output({"description": "header"})
        assert d == "header"
        assert labels == []

    def test_string_json_output(self):
        d, labels = _parse_frame_output(json.dumps({"description": "x", "labels": ["a"]}))
        assert d == "x"
        assert labels == ["a"]

    def test_string_non_json_output(self):
        d, labels = _parse_frame_output("not JSON at all")
        assert d == "not JSON at all"
        assert labels == []

    def test_string_json_array_output(self):
        # JSON that parses but isn't a dict — fall back to text.
        d, labels = _parse_frame_output(json.dumps([1, 2, 3]))
        assert "[1, 2, 3]" in d
        assert labels == []

    def test_empty_string_output(self):
        d, labels = _parse_frame_output("")
        assert d == ""
        assert labels == []

    def test_unexpected_type_output(self):
        d, labels = _parse_frame_output(42)
        assert d == ""
        assert labels == []

    def test_labels_non_list_dropped(self):
        d, labels = _parse_frame_output({"description": "x", "labels": "not-a-list"})
        assert d == "x"
        assert labels == []


class TestParseSynthesisOutput:
    def test_dict_output(self):
        wt, notes = _parse_synthesis_output(
            {"walkthrough": "the user clicks", "ui_notes": "blue button"}
        )
        assert wt == "the user clicks"
        assert notes == "blue button"

    def test_string_json_output(self):
        wt, notes = _parse_synthesis_output(json.dumps({"walkthrough": "x", "ui_notes": "y"}))
        assert wt == "x"
        assert notes == "y"

    def test_string_non_json_output(self):
        wt, notes = _parse_synthesis_output("just words")
        assert wt == "just words"
        assert notes == ""

    def test_empty_string_output(self):
        wt, notes = _parse_synthesis_output("")
        assert wt == ""
        assert notes == ""

    def test_unexpected_type_output(self):
        wt, notes = _parse_synthesis_output(None)
        assert wt == ""
        assert notes == ""


# ---------------------------------------------------------------------------
# WMVisionProvider.analyze_frames
# ---------------------------------------------------------------------------


class TestAnalyzeFrames:
    @pytest.mark.asyncio
    async def test_single_frame_dict_output(self, tiny_jpeg):
        wm = AsyncMock()
        wm.message = AsyncMock(return_value=_wm_resp({"description": "a header", "labels": ["h1"]}))
        provider = WMVisionProvider(wm_client=wm, cost_per_frame=0.05)

        results = await provider.analyze_frames(
            [FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)]
        )

        assert len(results) == 1
        r = results[0]
        assert r.index == 0
        assert r.description == "a header"
        assert r.labels == ["h1"]
        assert r.cost_usd == 0.05

        # Verify the WM call carried text + image content blocks.
        kwargs = wm.message.call_args.kwargs
        assert kwargs["worker_name"] == "video-frame-analyzer"
        blocks = kwargs["content_blocks"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["media_type"] == "image/jpeg"
        # The base64 payload roundtrips back to the on-disk bytes.
        assert base64.b64decode(blocks[1]["data"]) == tiny_jpeg.read_bytes()
        # output_schema is enforced.
        assert kwargs["output_schema"]["properties"]["description"]["type"] == "string"
        # WM forbids per-call sampling params (MessageRequest extra=forbid -> 422);
        # temperature must NOT be sent. Determinism is set on the worker config.
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_multiple_frames(self, tiny_jpeg):
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=[
                _wm_resp({"description": "frame-A", "labels": []}),
                _wm_resp({"description": "frame-B", "labels": ["btn"]}),
            ]
        )
        provider = WMVisionProvider(wm_client=wm)

        results = await provider.analyze_frames(
            [
                FrameInput(index=0, timestamp=0.0, path=tiny_jpeg),
                FrameInput(index=1, timestamp=2.5, path=tiny_jpeg),
            ]
        )
        assert [r.description for r in results] == ["frame-A", "frame-B"]
        assert wm.message.call_count == 2

    @pytest.mark.asyncio
    async def test_string_output_falls_back_to_json_parse(self, tiny_jpeg):
        wm = AsyncMock()
        wm.message = AsyncMock(
            return_value=_wm_resp(json.dumps({"description": "stringified", "labels": ["x"]}))
        )
        provider = WMVisionProvider(wm_client=wm)

        results = await provider.analyze_frames(
            [FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)]
        )
        assert results[0].description == "stringified"
        assert results[0].labels == ["x"]

    @pytest.mark.asyncio
    async def test_wm_error_wraps_as_vision_provider_error(self, tiny_jpeg):
        wm = AsyncMock()
        wm.message = AsyncMock(side_effect=RuntimeError("WM unreachable"))
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderError, match="WM vision call failed"):
            await provider.analyze_frames([FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)])

    @pytest.mark.asyncio
    async def test_one_frame_error_does_not_discard_other_frames(self, tiny_jpeg):
        """Bug: a single frame's transient error must NOT abort the whole batch
        and discard the N-1 already-analysed (already-paid) frames. The bad
        frame yields an empty, UNBILLED placeholder; the good frames survive."""
        request = httpx.Request("POST", "http://wm.test/workers/video-frame-analyzer/message")
        response = httpx.Response(500, request=request, json={"detail": "boom"})
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=[
                _wm_resp({"description": "frame-A", "labels": ["a"]}),
                httpx.HTTPStatusError("500", request=request, response=response),
                _wm_resp({"description": "frame-C", "labels": ["c"]}),
            ]
        )
        provider = WMVisionProvider(wm_client=wm, cost_per_frame=0.05)

        results = await provider.analyze_frames(
            [
                FrameInput(index=0, timestamp=0.0, path=tiny_jpeg),
                FrameInput(index=1, timestamp=1.0, path=tiny_jpeg),
                FrameInput(index=2, timestamp=2.0, path=tiny_jpeg),
            ]
        )

        # All three frames are represented, in order.
        assert [r.index for r in results] == [0, 1, 2]
        assert results[0].description == "frame-A"
        assert results[2].description == "frame-C"
        # The failed frame is an empty, unbilled placeholder.
        assert results[1].description == ""
        assert results[1].labels == []
        assert results[1].cost_usd == 0.0
        # The good frames are still billed.
        assert results[0].cost_usd == 0.05
        assert results[2].cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_all_frames_failing_still_hard_fails(self, tiny_jpeg):
        """When EVERY frame fails there is nothing to salvage → re-raise so the
        pipeline marks the job failed (no silent all-empty success)."""
        wm = AsyncMock()
        wm.message = AsyncMock(side_effect=RuntimeError("WM unreachable"))
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderError, match="WM vision call failed"):
            await provider.analyze_frames(
                [
                    FrameInput(index=0, timestamp=0.0, path=tiny_jpeg),
                    FrameInput(index=1, timestamp=1.0, path=tiny_jpeg),
                ]
            )

    @pytest.mark.asyncio
    async def test_model_override_propagated(self, tiny_jpeg):
        wm = AsyncMock()
        wm.message = AsyncMock(return_value=_wm_resp({"description": "x"}))
        provider = WMVisionProvider(wm_client=wm, model_override="claude-sonnet-4-5")

        await provider.analyze_frames([FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)])
        assert wm.message.call_args.kwargs["model_override"] == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_wm_422_surfaces_as_provider_unavailable(self, tiny_jpeg):
        """WM 422 = vision-capable provider missing from chain → typed subclass."""
        request = httpx.Request("POST", "http://wm.test/workers/video-frame-analyzer/message")
        response = httpx.Response(
            422, request=request, json={"detail": "no vision-capable provider"}
        )
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=httpx.HTTPStatusError("422", request=request, response=response)
        )
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderUnavailableError, match="chain"):
            await provider.analyze_frames([FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)])

    @pytest.mark.asyncio
    async def test_wm_5xx_surfaces_as_generic_provider_error_with_status(self, tiny_jpeg):
        """Non-422 HTTPStatusError preserves the status code in the message."""
        request = httpx.Request("POST", "http://wm.test/workers/video-frame-analyzer/message")
        response = httpx.Response(500, request=request, json={"detail": "boom"})
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=httpx.HTTPStatusError("500", request=request, response=response)
        )
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderError, match="HTTP 500") as exc_info:
            await provider.analyze_frames([FrameInput(index=0, timestamp=0.0, path=tiny_jpeg)])
        # And specifically NOT the unavailable subclass.
        assert not isinstance(exc_info.value, VisionProviderUnavailableError)


# ---------------------------------------------------------------------------
# WMVisionProvider.synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_dict_output(self):
        from src.services.vision.base import FrameVisionResult

        wm = AsyncMock()
        wm.message = AsyncMock(
            return_value=_wm_resp({"walkthrough": "the user clicks", "ui_notes": "blue button"})
        )
        provider = WMVisionProvider(wm_client=wm)

        result = await provider.synthesize(
            frame_results=[
                FrameVisionResult(
                    index=0,
                    timestamp=0.0,
                    description="frame description",
                    labels=[],
                    cost_usd=0.0,
                )
            ],
            transcript_text="the speaker says hi",
        )

        assert result.walkthrough == "the user clicks"
        assert result.ui_notes == "blue button"
        kwargs = wm.message.call_args.kwargs
        assert kwargs["worker_name"] == "video-synthesis"
        # synthesize is text-only — it uses message=, not content_blocks=.
        assert "message" in kwargs
        assert kwargs.get("content_blocks") is None or "content_blocks" not in kwargs
        # WM forbids per-call sampling params (MessageRequest extra=forbid -> 422);
        # temperature must NOT be sent. Determinism is set on the worker config.
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_synthesize_wm_error_wraps(self):
        wm = AsyncMock()
        wm.message = AsyncMock(side_effect=RuntimeError("boom"))
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderError, match="WM synthesis failed"):
            await provider.synthesize(frame_results=[], transcript_text=None)

    @pytest.mark.asyncio
    async def test_synthesize_wm_422_surfaces_as_provider_unavailable(self):
        """WM 422 on synthesis path is also surfaced as the typed subclass."""
        request = httpx.Request("POST", "http://wm.test/workers/video-synthesis/message")
        response = httpx.Response(422, request=request, json={"detail": "no provider"})
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=httpx.HTTPStatusError("422", request=request, response=response)
        )
        provider = WMVisionProvider(wm_client=wm)

        with pytest.raises(VisionProviderUnavailableError, match="chain"):
            await provider.synthesize(frame_results=[], transcript_text=None)


# ---------------------------------------------------------------------------
# Constructor: client injection + lazy default
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_construct_a_real_wmclient(self):
        provider = WMVisionProvider()
        # The default WMClient connects lazily; we just assert a client
        # was set, not that it speaks to anything.
        assert provider._wm is not None
        assert provider._cost_per_frame == 0.01

    def test_explicit_client_is_kept(self):
        wm = AsyncMock()
        provider = WMVisionProvider(wm_client=wm)
        assert provider._wm is wm


class TestEmptyDescriptionIsAFailedFrame:
    """HTTP 200 carrying nothing usable is a FAILURE, not a cheap success.

    The rest of this module drives failures via raised exceptions. This class
    covers the untested path: WM answers 200 and the parsed description is
    empty (empty string, a non-dict/None body, or a dict with no
    ``description``). This fleet has a documented "reasoning-on returns after
    950s and returns NOTHING" mode, so the shape is reachable in production.

    Before the fix such a frame was appended with ``cost_usd=cost_per_frame``
    and never counted into ``failures`` — so it accrued spend against
    ``max_vision_cost_usd``, wrote an empty description into the frame index
    and the synthesis prompt, and made the all-frames-failed re-raise
    unreachable for an all-empty batch.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "empty_output",
        [
            pytest.param({"description": ""}, id="dict-empty-description"),
            pytest.param({"labels": ["a"]}, id="dict-missing-description"),
            pytest.param("", id="empty-string"),
            pytest.param("   ", id="whitespace-string"),
            pytest.param(None, id="null-output"),
        ],
    )
    async def test_empty_200_is_unbilled_and_does_not_discard_good_frames(
        self, tiny_jpeg, empty_output
    ):
        wm = AsyncMock()
        wm.message = AsyncMock(
            side_effect=[
                _wm_resp({"description": "frame-A", "labels": ["a"]}),
                _wm_resp(empty_output),
            ]
        )
        provider = WMVisionProvider(wm_client=wm, cost_per_frame=0.05)

        results = await provider.analyze_frames(
            [
                FrameInput(index=0, timestamp=0.0, path=tiny_jpeg),
                FrameInput(index=1, timestamp=1.0, path=tiny_jpeg),
            ]
        )

        assert [r.index for r in results] == [0, 1]
        assert results[0].description == "frame-A"
        assert results[0].cost_usd == 0.05
        # The empty frame is the unbilled placeholder, not a paid "analysis".
        assert results[1].description == ""
        assert results[1].labels == []
        assert results[1].cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_every_frame_empty_on_200_hard_fails(self, tiny_jpeg):
        """The re-raise must fire when EVERY frame is a 200-with-nothing.

        Without the fix this returned two zero-content results and the job
        completed "successfully" with a walkthrough rendered from nothing.
        """
        wm = AsyncMock()
        wm.message = AsyncMock(return_value=_wm_resp({"description": "  "}))
        provider = WMVisionProvider(wm_client=wm, cost_per_frame=0.05)

        with pytest.raises(VisionProviderError, match="no usable description"):
            await provider.analyze_frames(
                [
                    FrameInput(index=0, timestamp=0.0, path=tiny_jpeg),
                    FrameInput(index=1, timestamp=1.0, path=tiny_jpeg),
                ]
            )
