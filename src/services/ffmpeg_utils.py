"""Thin async wrapper around ffmpeg / ffprobe subprocess calls."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg / ffprobe exits with non-zero status."""


def ffmpeg_available() -> bool:
    """Return True if ffmpeg and ffprobe are on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


async def run_subprocess_bytes(
    cmd: list[str], *, timeout: float = 300.0
) -> tuple[int, bytes, bytes]:
    """Run a subprocess and return (rc, stdout, stderr) with stdout UNDECODED.

    This is the primitive; :func:`run_subprocess` is the text view over it.
    Two callers need the raw bytes and would be *corrupted* by the text view:
    ``ffmpeg ... -f f32le -`` writes raw little-endian float samples to stdout,
    and ``errors="replace"`` silently rewrites every byte sequence that is not
    valid UTF-8 into U+FFFD — i.e. it would mangle most of the audio and the
    corruption would be invisible (no exception, just wrong numbers).
    """
    logger.debug("Running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise FFmpegError(f"Command timed out after {timeout}s: {cmd[0]}") from exc

    return proc.returncode or 0, stdout_b, stderr_b


async def run_subprocess(cmd: list[str], *, timeout: float = 300.0) -> tuple[int, str, str]:
    """Run a subprocess and return (rc, stdout, stderr) decoded as UTF-8."""
    rc, stdout_b, stderr_b = await run_subprocess_bytes(cmd, timeout=timeout)
    return (
        rc,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


async def run_checked(cmd: list[str], *, timeout: float = 300.0) -> tuple[str, str]:
    """Run a subprocess and raise FFmpegError on non-zero exit."""
    rc, stdout, stderr = await run_subprocess(cmd, timeout=timeout)
    if rc != 0:
        raise FFmpegError(f"{cmd[0]} exited with code {rc}: {stderr.strip()[:500]}")
    return stdout, stderr


async def run_checked_bytes(cmd: list[str], *, timeout: float = 300.0) -> bytes:
    """Run a subprocess, raise FFmpegError on non-zero exit, return raw stdout.

    stderr is still decoded for the error message — only stdout is kept binary.
    """
    rc, stdout_b, stderr_b = await run_subprocess_bytes(cmd, timeout=timeout)
    if rc != 0:
        stderr = stderr_b.decode("utf-8", errors="replace")
        raise FFmpegError(f"{cmd[0]} exited with code {rc}: {stderr.strip()[:500]}")
    return stdout_b


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
