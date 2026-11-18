"""Async subprocess wrapper tests."""

from __future__ import annotations

import pytest

from src.services.ffmpeg_utils import (
    FFmpegError,
    ensure_parent_dir,
    ffmpeg_available,
    run_checked,
    run_checked_bytes,
    run_subprocess,
)


async def test_run_subprocess_success() -> None:
    rc, stdout, stderr = await run_subprocess(["/bin/sh", "-c", "echo hi"])
    assert rc == 0
    assert stdout.strip() == "hi"
    assert stderr == ""


async def test_run_subprocess_failure() -> None:
    rc, _stdout, stderr = await run_subprocess(["/bin/sh", "-c", "echo boom >&2; exit 3"])
    assert rc == 3
    assert "boom" in stderr


async def test_run_checked_raises_on_nonzero() -> None:
    with pytest.raises(FFmpegError, match="exited with code 7"):
        await run_checked(["/bin/sh", "-c", "exit 7"])


async def test_run_subprocess_timeout() -> None:
    with pytest.raises(FFmpegError, match="timed out"):
        await run_checked(["/bin/sh", "-c", "sleep 5"], timeout=0.1)


async def test_run_checked_bytes_preserves_non_utf8_stdout() -> None:
    """The whole point of the bytes variant: raw PCM is not text.

    ``\\xff\\xfe`` is not valid UTF-8, so the decoding path would silently
    rewrite it to U+FFFD and corrupt every audio sample it touched.
    """
    out = await run_checked_bytes(["/bin/sh", "-c", r"printf '\377\376\000\001'"])
    assert out == b"\xff\xfe\x00\x01"


async def test_run_checked_bytes_raises_on_nonzero() -> None:
    with pytest.raises(FFmpegError, match="exited with code 5"):
        await run_checked_bytes(["/bin/sh", "-c", "echo nope >&2; exit 5"])


def test_ffmpeg_available_returns_bool() -> None:
    assert isinstance(ffmpeg_available(), bool)


def test_ensure_parent_dir_creates(tmp_path) -> None:
    target = tmp_path / "nested/dir/file.txt"
    ensure_parent_dir(target)
    assert target.parent.exists()
