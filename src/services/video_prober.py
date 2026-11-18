"""ffprobe-based video metadata extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.domain.analysis import VideoMetadata
from src.services.ffmpeg_utils import FFmpegError, run_checked

logger = logging.getLogger(__name__)

#: One decoded ffprobe JSON object — a ``format`` block, a ``streams[]`` entry,
#: or the top-level payload.
#:
#: This is a genuine JSON boundary: ffprobe's key set varies by container, codec
#: and build, so the values really are unknown until read. Naming that once is
#: strictly better than spelling ``dict[str, Any]`` at six call sites — the shape
#: gets a name the reader can look up, and the module's contribution to the
#: repo-wide ``Any`` ratchet (``tests/test_any_ratchet.py``) drops from 6 to 1.
#: Do NOT expand this back into inline ``dict[str, Any]``; that gate is
#: lower-only and re-inlining would have to be paid for somewhere else.
type FFProbeNode = dict[str, Any]


class VideoProbeError(RuntimeError):
    """Raised when ffprobe output cannot be parsed into VideoMetadata."""


def _pick_video_stream(streams: list[FFProbeNode]) -> FFProbeNode | None:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    return None


def _has_audio_stream(streams: list[FFProbeNode]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in streams)


def _parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, denom = value.split("/", 1)
        try:
            d = float(denom)
            if d == 0:
                return 0.0
            return float(num) / d
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _strict_float(value: object, *, field: str) -> float:
    """Parse an ffprobe numeric field, raising ``VideoProbeError`` on garbage.

    Absent (``None``) means "this container/stream does not carry the field" and
    yields 0.0 so the caller can fall through to the next source. A field that is
    PRESENT but unparseable is a corrupt probe, not a missing one, and must not be
    silently degraded to zero — a zero duration means zero frames extracted, i.e.
    an empty analysis reported as a success.
    """
    if value is None:
        return 0.0
    # ffprobe JSON only ever carries str/int/float here; anything else (a dict, a
    # list) is a shape we do not understand, which is a corrupt probe rather than
    # a missing field — so it takes the raise path with everything else.
    if not isinstance(value, str | int | float):
        raise VideoProbeError(f"ffprobe {field} is not numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise VideoProbeError(f"ffprobe {field} is not numeric") from exc


def _lenient_float(value: object) -> float:
    """Parse an optional numeric field, treating garbage as absent.

    Used for advisory fields (``nb_frames``) that only ever refine a duration we
    can also obtain elsewhere; a bogus value there must not fail the whole probe.
    """
    # Advisory field: any shape we do not understand collapses to "absent" rather
    # than failing the probe, which is the whole difference from ``_strict_float``.
    if not isinstance(value, str | int | float):
        return 0.0
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0


def _parse_duration_tag(value: object) -> float:
    """Parse a Matroska/WebM ``DURATION`` stream tag (``HH:MM:SS.nnnnnnnnn``).

    WHY this exists: Matroska/WebM streams carry NO stream-level ``duration``
    field and usually no ``nb_frames`` either — the per-stream length lives only
    in this tag. Without it a WebM falls all the way through to the container
    duration, which is exactly the max-across-streams number this module is
    trying to stop trusting.
    """
    if not isinstance(value, str):
        return 0.0
    parts = value.strip().split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return 0.0
    total = hours * 3600.0 + minutes * 60.0 + seconds
    return total if total > 0 else 0.0


def _resolve_duration_seconds(video: FFProbeNode, fmt: FFProbeNode) -> float:
    """Return the VIDEO stream's duration, not the container's.

    WHY: ``format.duration`` is the MAXIMUM over every stream in the container by
    construction, so any file whose audio outruns its video reports the AUDIO
    length. Measured: a 6s video muxed with 12s of audio probed as 12.000. That
    number is not merely cosmetic — it is divided by ``MIN_FRAMES_FOR_QUALITY``
    to pick the sampling interval (2.400 instead of 1.200), so 2 of the 5 sample
    points land past the real EOF, ``ffmpeg -ss`` writes nothing for them, and
    40% of the sample set silently disappears from the means that the blank-clip
    HARD-FAIL gate votes on. The EOF safety margin in ``frame_extractor`` cannot
    rescue this: the number it subtracts from is the wrong number. The same value
    is also what ``pipeline._emit_cost`` BILLS on, so a padded file was charged
    for audio it never analysed.

    Fallback chain, most to least authoritative:

    1. ``video["duration"]`` — the stream's own length (mp4/mov/mkv-in-ffmpeg).
    2. ``nb_frames / fps`` — derived; exact for CFR, and the only stream-local
       source when the demuxer omits ``duration``.
    3. the Matroska/WebM ``DURATION`` tag (see ``_parse_duration_tag``).
    4. ``format["duration"]`` — the container. Wrong on padded files, but a
       wrong-and-long duration still extracts frames, whereas 0.0 extracts NONE,
       so it stays the last resort rather than being dropped.

    The result is then clamped with ``min(candidate, format_duration)``: the
    container length is an upper bound over its streams by construction, so a
    stream header claiming to be longer than the file it lives in is lying, and
    the clamp is a free defence against it.
    """
    format_duration = _strict_float(fmt.get("duration"), field="format duration")

    candidate = _strict_float(video.get("duration"), field="video stream duration")

    if candidate <= 0:
        nb_frames = _lenient_float(video.get("nb_frames"))
        fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        if nb_frames > 0 and fps > 0:
            candidate = nb_frames / fps

    if candidate <= 0:
        tags = video.get("tags") or {}
        candidate = _parse_duration_tag(tags.get("DURATION") or tags.get("duration"))

    if candidate <= 0:
        return format_duration
    if format_duration > 0:
        return min(candidate, format_duration)
    return candidate


def parse_ffprobe_json(payload: FFProbeNode) -> VideoMetadata:
    """Convert a parsed ffprobe JSON payload into VideoMetadata."""
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    video = _pick_video_stream(streams)
    if video is None:
        raise VideoProbeError("ffprobe output contains no video stream")

    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise VideoProbeError("ffprobe video stream has invalid dimensions") from exc

    duration = _resolve_duration_seconds(video, fmt)

    fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))

    frame_count_str = video.get("nb_frames")
    try:
        frame_count = int(frame_count_str) if frame_count_str else round(duration * fps)
    except TypeError, ValueError:
        frame_count = round(duration * fps)

    return VideoMetadata(
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=_has_audio_stream(streams),
        format_name=fmt.get("format_name"),
    )


async def probe_video(video_path: Path, *, timeout: float = 30.0) -> VideoMetadata:
    """Run ffprobe against ``video_path`` and return structured metadata."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        stdout, _ = await run_checked(cmd, timeout=timeout)
    except FFmpegError as exc:
        raise VideoProbeError(f"ffprobe failed: {exc}") from exc
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"Invalid ffprobe JSON: {exc}") from exc
    return parse_ffprobe_json(payload)
