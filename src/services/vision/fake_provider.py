"""Deterministic fake (a.k.a. ``stub``) vision provider.

Used by:
  - Unit tests that need a vision provider without HTTP/API keys.
  - CI / offline pipelines selected via ``ANALYZER_PROVIDER=stub`` or
    ``VISION_DEFAULT_PROVIDER=stub``.

Goals:
  - **Deterministic** — same frame indexes always produce identical output, so
    snapshot tests stay stable across runs.
  - **Realistic enough to pass schema validation** — the description, label
    set and synthesised walkthrough/ui_notes match the shape and order-of-
    magnitude content of the real providers, so downstream consumers
    (orchestrator ``run_video_analyze`` stage handler, ``JobResultResponse``
    Pydantic model) can validate without crashing.
  - **Zero cost / zero IO** — no network calls, no disk reads.

Backward compatibility: the existing ``description.startswith("Frame N")``
prefix is preserved so older tests keep passing while richer text is appended
behind it.
"""

from __future__ import annotations

from src.services.vision.base import (
    FrameInput,
    FrameVisionResult,
    VisionProvider,
    VisionSynthesisResult,
)

# Deterministic label rotation. Picked from labels real vision providers
# typically emit on app/UI walkthrough footage.
_LABEL_POOL: tuple[tuple[str, ...], ...] = (
    ("ui", "screen", "navigation"),
    ("ui", "screen", "form"),
    ("ui", "screen", "list"),
    ("ui", "screen", "modal"),
    ("ui", "screen", "settings"),
    ("ui", "screen", "dashboard"),
)

# Canned scene descriptors keyed by index modulo. Keeps output deterministic
# while looking like a real product walkthrough.
_SCENE_POOL: tuple[str, ...] = (
    "Landing screen with primary call-to-action visible",
    "User opens the navigation menu and selects an item",
    "Form view: the user fills in input fields",
    "List of items rendered with paginated controls",
    "Modal dialog overlays the screen requesting confirmation",
    "Settings view with toggles and grouped preferences",
)


def _labels_for(index: int) -> list[str]:
    return list(_LABEL_POOL[index % len(_LABEL_POOL)])


def _scene_for(index: int) -> str:
    return _SCENE_POOL[index % len(_SCENE_POOL)]


class FakeVisionProvider(VisionProvider):
    """A zero-cost deterministic provider.

    Selected by ``ANALYZER_PROVIDER=stub`` or ``VISION_DEFAULT_PROVIDER=stub``
    (alias: ``fake``).
    """

    name = "fake"

    def __init__(self, *, cost_per_frame: float = 0.0) -> None:
        self._cost_per_frame = cost_per_frame

    def estimated_cost_for(self, frame_count: int) -> float:
        """Predicted spend — defaults to 0.0, so the offline provider stays free.

        Implemented (rather than inherited) so the pipeline's pre-flight cap is
        actually exercised by the test suite: with only the 0.0 base default,
        the pre-flight branch would never fire under ``ANALYZER_PROVIDER=stub``
        and would ship untested.
        """
        return max(frame_count, 0) * self._cost_per_frame

    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        results: list[FrameVisionResult] = []
        for frame in frames:
            scene = _scene_for(frame.index)
            description = f"Frame {frame.index} at {frame.timestamp:.1f}s: {scene}."
            results.append(
                FrameVisionResult(
                    index=frame.index,
                    timestamp=frame.timestamp,
                    description=description,
                    labels=_labels_for(frame.index),
                    cost_usd=self._cost_per_frame,
                )
            )
        return results

    async def synthesize(
        self,
        *,
        frame_results: list[FrameVisionResult],
        transcript_text: str | None,
    ) -> VisionSynthesisResult:
        count = len(frame_results)
        transcript_snippet = (transcript_text or "").strip()[:120]
        if count == 0:
            walkthrough = (
                f"Automated walkthrough covering {count} frames. "
                f"Transcript excerpt: {transcript_snippet}"
            )
            return VisionSynthesisResult(walkthrough=walkthrough, ui_notes="")

        first = frame_results[0]
        last = frame_results[-1]
        walkthrough = (
            f"Automated walkthrough covering {count} frames "
            f"({first.timestamp:.1f}s → {last.timestamp:.1f}s). "
            f"Transcript excerpt: {transcript_snippet}"
        )
        ui_notes = "\n".join(
            f"- Frame {r.index} ({r.timestamp:.1f}s): {r.description}" for r in frame_results
        )
        return VisionSynthesisResult(walkthrough=walkthrough, ui_notes=ui_notes)
