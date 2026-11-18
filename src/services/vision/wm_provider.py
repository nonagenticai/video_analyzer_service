"""Worker-management-backed vision provider.

Per memory `wm-as-sole-llm-gateway.md`, all LLM calls in w2v services
must route through worker_management — including multimodal vision.
WM's `MessageRequest.message` accepts either a plain string or a list
of typed content blocks (text + image), and the pydantic_ai agent layer
hands images to the underlying provider's vision-capable model
(claude-sonnet-4-5, gpt-4o, gemini-2-pro, gemma-4) via Pydantic-AI's
`BinaryContent`.

This provider replaces the per-SDK `OpenAIVisionProvider`,
`AnthropicVisionProvider`, and `GoogleVisionProvider` modules with a
single WM-backed implementation. The model is selected by configuring
the WM worker (or by passing `model_override` per call) — the service
itself no longer owns provider/API-key plumbing.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from w2v_common.clients import WMClient

from src.services.vision.base import (
    FrameInput,
    FrameVisionResult,
    VisionProvider,
    VisionProviderError,
    VisionProviderUnavailableError,
    VisionSynthesisResult,
)
from src.services.vision.prompts import (
    FRAME_SYSTEM_PROMPT,
    FRAME_USER_INSTRUCTION,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
)

logger = logging.getLogger(__name__)

# Stable WM worker names. Both auto-create on first call; pass an
# explicit `model_override` to steer them at a vision-capable model.
_FRAME_WORKER = "video-frame-analyzer"
_SYNTHESIS_WORKER = "video-synthesis"


class WMVisionProvider(VisionProvider):
    """Vision provider backed by worker_management's `/workers/{name}/message`.

    The model is selected at the WM worker level. For per-call overrides
    (e.g. swap claude-sonnet-4-5 → gpt-4o for a specific job) pass
    ``model_override``. ``cost_per_frame`` is the local accounting hint
    used by the cost cap; actual cost is metered by inference-router via
    the X-Tenant-Id / X-Subject-Id identity headers WM forwards.
    """

    name = "wm"

    def __init__(
        self,
        *,
        model_override: str | None = None,
        cost_per_frame: float = 0.01,
        wm_client: WMClient | None = None,
    ) -> None:
        self._model_override = model_override
        self._cost_per_frame = cost_per_frame
        self._wm = wm_client if wm_client is not None else WMClient()

    def estimated_cost_for(self, frame_count: int) -> float:
        """Predicted spend for *frame_count* frames.

        ``_cost_per_frame`` is a fixed constructor value, so the total is known
        before a single WM call is made — which is what lets the pipeline reject
        an over-cap job without paying for it first.
        """
        return max(frame_count, 0) * self._cost_per_frame

    @staticmethod
    def _encode_image(path: str) -> str:
        with Path(path).open("rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        results: list[FrameVisionResult] = []
        # A single frame's transient failure (one WM 5xx, one unreadable frame)
        # must NOT discard the N-1 already-analysed (already-paid) frames by
        # aborting the whole job. Instead we record an empty, UNBILLED
        # placeholder for the bad frame and continue. Only a config/auth-level
        # fatal (WM 422 = no vision-capable model in the chain — identical for
        # every frame) aborts immediately, and if EVERY frame fails we re-raise
        # the last error so the job is marked failed with nothing to salvage.
        last_error: VisionProviderError | None = None
        failures = 0
        for frame in frames:
            try:
                b64 = self._encode_image(str(frame.path))
                response = await self._wm.message(
                    worker_name=_FRAME_WORKER,
                    content_blocks=[
                        {"type": "text", "text": FRAME_USER_INSTRUCTION},
                        {"type": "image", "data": b64, "media_type": "image/jpeg"},
                    ],
                    system_prompt=FRAME_SYSTEM_PROMPT,
                    output_schema={
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["description"],
                    },
                    model_override=self._model_override,
                    # Run provider slot (w2v_common 0.113): a frame carries an image
                    # part, so it is a ``vlm`` call. Explicit, not left to WMClient's
                    # image-part sniff. Only read under W2V_PROVIDER_SLOTS=enforce.
                    slot="vlm",
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 422:
                    # Config/auth-level fatal — will fail identically for every
                    # frame, so aborting the whole job is correct here.
                    raise VisionProviderUnavailableError(
                        "Vision-capable provider not available in WM/inference-router "
                        f"chain (HTTP 422 from WM: {exc.response.text[:200]})"
                    ) from exc
                last_error = VisionProviderError(
                    f"WM vision call failed (HTTP {exc.response.status_code}): {exc}"
                )
                failures += 1
                logger.warning(
                    "Frame %s vision call failed (HTTP %s); recording empty placeholder",
                    frame.index,
                    exc.response.status_code,
                )
                results.append(_empty_frame_result(frame))
                continue
            except Exception as exc:  # noqa: BLE001 — per-frame resilience: one frame's error must not discard the others
                last_error = VisionProviderError(f"WM vision call failed: {exc}")
                failures += 1
                logger.warning(
                    "Frame %s vision call failed (%s); recording empty placeholder",
                    frame.index,
                    exc,
                )
                results.append(_empty_frame_result(frame))
                continue

            description, labels = _parse_frame_output(response.output)
            if not description:
                # HTTP 200 with nothing usable in it (empty string, a non-dict
                # JSON body, or a dict with no `description`). This fleet has a
                # documented "reasoning-on returns after 950s and returns
                # NOTHING" mode, so this is a REACHABLE success-shaped failure,
                # not a theoretical one. Counting it as a successful frame was
                # wrong twice over: the empty description flowed into the frame
                # index and the synthesis prompt (so walkthrough.md /
                # ui_inventory.md render from nothing), and the frame still
                # accrued `cost_per_frame` into `vision_cost` — spend that is
                # persisted as `vision_cost_usd` and charged against the
                # caller's `max_vision_cost_usd` cap. Treating it as a failure
                # prices it at 0.0 AND restores the all-frames-failed re-raise
                # below, which an all-empty batch could otherwise never trip.
                last_error = VisionProviderError(
                    f"WM vision call returned HTTP 200 with no usable description "
                    f"for frame {frame.index}"
                )
                failures += 1
                logger.warning(
                    "Frame %s vision call returned an empty description on HTTP 200; "
                    "recording empty placeholder",
                    frame.index,
                )
                results.append(_empty_frame_result(frame))
                continue

            results.append(
                FrameVisionResult(
                    index=frame.index,
                    timestamp=frame.timestamp,
                    description=description,
                    labels=labels,
                    cost_usd=self._cost_per_frame,
                )
            )

        # Every frame failed → nothing salvageable; surface the last error so
        # the pipeline marks the job failed (preserves prior single-frame
        # hard-fail behaviour when there is exactly one, failing, frame).
        if frames and failures == len(frames) and last_error is not None:
            raise last_error
        return results

    async def synthesize(
        self,
        *,
        frame_results: list[FrameVisionResult],
        transcript_text: str | None,
    ) -> VisionSynthesisResult:
        prompt = build_synthesis_prompt(
            frame_descriptions=[r.description for r in frame_results],
            transcript_text=transcript_text,
        )
        try:
            response = await self._wm.message(
                worker_name=_SYNTHESIS_WORKER,
                message=prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                output_schema={
                    "type": "object",
                    "properties": {
                        "walkthrough": {"type": "string"},
                        "ui_notes": {"type": "string"},
                    },
                    "required": ["walkthrough"],
                },
                model_override=self._model_override,
                # Synthesis is text-only (frame descriptions + transcript): ``llm``.
                slot="llm",
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                raise VisionProviderUnavailableError(
                    "Vision-capable provider not available in WM/inference-router "
                    f"chain (HTTP 422 from WM: {exc.response.text[:200]})"
                ) from exc
            raise VisionProviderError(
                f"WM synthesis failed (HTTP {exc.response.status_code}): {exc}"
            ) from exc
        except Exception as exc:
            raise VisionProviderError(f"WM synthesis failed: {exc}") from exc

        walkthrough, ui_notes = _parse_synthesis_output(response.output)
        return VisionSynthesisResult(
            walkthrough=walkthrough,
            ui_notes=ui_notes,
            cost_usd=self._cost_per_frame,
        )


def _empty_frame_result(frame: FrameInput) -> FrameVisionResult:
    """Unbilled placeholder for a frame whose vision call failed transiently.

    Empty description + labels mark the frame as analysed-but-empty; ``cost_usd``
    is 0.0 so a failed frame is never metered as a successful (billed) analysis
    while its index/timestamp still align with the extracted frame list.
    """
    return FrameVisionResult(
        index=frame.index,
        timestamp=frame.timestamp,
        description="",
        labels=[],
        cost_usd=0.0,
    )


def _parse_frame_output(output: object) -> tuple[str, list[str]]:
    """Coerce a WM frame-analysis output into (description, labels)."""
    from typing import cast

    if isinstance(output, dict):
        data = cast(dict[str, Any], output)
    elif isinstance(output, str):
        text = output.strip()
        if not text:
            return "", []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text, []
        if not isinstance(parsed, dict):
            return text, []
        data = parsed
    else:
        return "", []

    description = str(data.get("description", "")).strip()
    labels_raw = data.get("labels") or []
    labels = [str(label) for label in labels_raw] if isinstance(labels_raw, list) else []
    return description, labels


def _parse_synthesis_output(output: object) -> tuple[str, str]:
    """Coerce a WM synthesis output into (walkthrough, ui_notes)."""
    from typing import cast

    if isinstance(output, dict):
        data = cast(dict[str, Any], output)
    elif isinstance(output, str):
        text = output.strip()
        if not text:
            return "", ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text, ""
        if not isinstance(parsed, dict):
            return text, ""
        data = parsed
    else:
        return "", ""

    return (
        str(data.get("walkthrough", "")).strip(),
        str(data.get("ui_notes", "")).strip(),
    )
