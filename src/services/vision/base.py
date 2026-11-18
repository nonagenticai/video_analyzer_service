"""Abstract vision LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class VisionProviderError(RuntimeError):
    """Raised when a vision provider call fails or is misconfigured."""


class VisionProviderUnavailableError(VisionProviderError):
    """Raised when no vision-capable provider is available in the WM/inference-router chain.

    Distinct subclass so callers (pipeline runners, future API translators)
    can map this to a 4xx config-error rather than treating it as a generic
    upstream failure.
    """


@dataclass(slots=True)
class FrameInput:
    """Input to the vision provider for a single frame."""

    index: int
    timestamp: float
    path: Path


@dataclass(slots=True)
class FrameVisionResult:
    """Output of a single frame vision analysis."""

    index: int
    timestamp: float
    description: str
    labels: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


@dataclass(slots=True)
class VisionSynthesisResult:
    """Output of LLM synthesis (walkthrough text + UI inventory notes)."""

    walkthrough: str
    ui_notes: str
    cost_usd: float = 0.0


class VisionProvider(ABC):
    """Abstract base class for vision LLM providers."""

    name: str = "base"

    def estimated_cost_for(self, frame_count: int) -> float:
        """Return the spend ``analyze_frames`` would incur for *frame_count* frames.

        Exists so the pipeline can enforce ``max_vision_cost_usd`` BEFORE paying
        rather than after. The default is ``0.0``, which makes the pre-flight
        check a no-op for providers that cannot predict their spend (and for
        ``FakeVisionProvider``, mirroring how it already reports 0.0 synthesis
        cost) — a provider that does not know its cost must not be able to
        trigger a spurious cap breach.

        This is an ESTIMATE and deliberately does not replace the post-hoc
        check: actual spend can exceed it (retries, the synthesis fold), so both
        gates are kept.
        """
        del frame_count
        return 0.0

    @abstractmethod
    async def analyze_frames(self, frames: list[FrameInput]) -> list[FrameVisionResult]:
        """Describe a batch of frames."""

    @abstractmethod
    async def synthesize(
        self,
        *,
        frame_results: list[FrameVisionResult],
        transcript_text: str | None,
    ) -> VisionSynthesisResult:
        """Produce a walkthrough + UI inventory from frame analyses and transcript."""
