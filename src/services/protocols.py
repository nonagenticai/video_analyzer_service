"""Structural protocols for ``PipelineDependencies`` collaborators.

Each Protocol declares only the methods the pipeline actually invokes on the
corresponding dep. Production classes satisfy the Protocols structurally
(no subclass relationship needed), and test doubles do too — without any
``# type: ignore[arg-type]`` suppression at the construction site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.domain.analysis import (
    ExtractedFrame,
    FrameExtraction,
    QualityMetrics,
    VideoMetadata,
)


@runtime_checkable
class FrameExtractorProto(Protocol):
    """Pipeline contract for ``FrameExtractor``."""

    async def extract(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        metadata: VideoMetadata,
        desired_interval: float,
        max_frames: int,
    ) -> FrameExtraction: ...


@runtime_checkable
class AudioExtractorProto(Protocol):
    """Pipeline contract for ``AudioExtractor``."""

    async def extract(self, *, video_path: Path, output_path: Path) -> Path: ...


@runtime_checkable
class VideoDownloaderProto(Protocol):
    """Pipeline contract for ``VideoDownloader``."""

    async def download(self, *, video_url: str, output_dir: Path) -> Path: ...


@runtime_checkable
class SceneDetectorProto(Protocol):
    """Pipeline contract for ``SceneDetector``."""

    def detect(self, frames: list[ExtractedFrame]) -> list[float]: ...


@runtime_checkable
class OCRServiceProto(Protocol):
    """Pipeline contract for ``OCRService``."""

    def extract_text(self, image_path: Path) -> str: ...


@runtime_checkable
class QualityAnalyzerProto(Protocol):
    """Pipeline contract for ``QualityAnalyzer``."""

    def analyze(
        self, frames: list[ExtractedFrame], *, frames_planned: int | None = None
    ) -> QualityMetrics: ...


@runtime_checkable
class TranscriptionClientProto(Protocol):
    """Pipeline contract for ``TranscriptionClient``."""

    async def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool = ...,
        return_srt: bool = ...,
    ) -> dict[str, Any]: ...
