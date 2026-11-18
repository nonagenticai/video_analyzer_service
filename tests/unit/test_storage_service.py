"""Local-backend tests — drive StorageService (over the shared facade) on the
``local`` backend. The hand-rolled LocalBackend class was retired; the facade
now provides the local object I/O, and this wrapper keeps the domain concerns
(job dirs, the ``_safe_local_path`` traversal guard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import clear_settings_cache
from src.services.storage_service import StorageService


def _local_service(monkeypatch: pytest.MonkeyPatch, root: Path) -> StorageService:
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(root))
    clear_settings_cache()
    return StorageService()


async def test_local_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _local_service(monkeypatch, tmp_path)
    await svc.initialize()
    stored = await svc.store_bytes("job-1", b"hello", "artifact.txt")
    assert Path(stored).read_text() == "hello"
    retrieved = await svc.retrieve(stored)
    assert retrieved == b"hello"


async def test_local_store_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _local_service(monkeypatch, tmp_path / "store")
    await svc.initialize()
    source = tmp_path / "source.txt"
    source.write_text("abc")
    stored = await svc.store_file("job-2", source, "copy.txt")
    assert Path(stored).read_text() == "abc"


async def test_upload_local_roundtrip_and_tenant_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """store_upload persists a source under a TENANT-SCOPED key; the owning
    tenant resolves it locally, a different tenant cannot (key isolation), and
    presign is a no-op on the local backend."""
    svc = _local_service(monkeypatch, tmp_path / "store")
    await svc.initialize()
    src = tmp_path / "src.mp4"
    src.write_bytes(b"MP4DATA")

    ref = await svc.store_upload("tenant-a", "job-1", src, "clip.mp4")
    assert ref.endswith("uploads/tenant-a/job-1/clip.mp4")

    got = svc.local_upload_path("tenant-a", "job-1", "clip.mp4")
    assert got is not None
    assert got.read_bytes() == b"MP4DATA"

    # A different tenant (or job) maps to a different key and cannot resolve it.
    assert svc.local_upload_path("tenant-b", "job-1", "clip.mp4") is None
    assert svc.local_upload_path("tenant-a", "job-2", "clip.mp4") is None

    # Local backend has no presigned URLs.
    assert await svc.presign_upload("tenant-a", "job-1", "clip.mp4") is None

    # delete_upload removes the tenant-scoped object; resolution then misses.
    await svc.delete_upload("tenant-a", "job-1")
    assert svc.local_upload_path("tenant-a", "job-1", "clip.mp4") is None


def test_upload_key_is_tenant_scoped() -> None:
    key_a = StorageService.upload_key("tenant-a", "job-1", "v.mp4")
    key_b = StorageService.upload_key("tenant-b", "job-1", "v.mp4")
    assert key_a == "uploads/tenant-a/job-1/v.mp4"
    assert key_a != key_b
    assert "tenant-a" in key_a and "tenant-b" in key_b


async def test_local_store_file_same_src_dest_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pipeline writes an artifact into get_job_dir() then stores it under
    the same name — source == destination. That must be a no-op, never a
    SameFileError (the retired LocalBackend guarded this)."""
    svc = _local_service(monkeypatch, tmp_path)
    await svc.initialize()
    job_dir = svc.get_job_dir("job-x")
    artifact = job_dir / "walkthrough.md"
    artifact.write_text("in-place")
    stored = await svc.store_file("job-x", artifact, "walkthrough.md")
    assert Path(stored) == artifact
    assert artifact.read_text() == "in-place"


async def test_local_delete_job_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _local_service(monkeypatch, tmp_path)
    await svc.initialize()
    await svc.store_bytes("job-3", b"payload", "a.bin")
    await svc.delete_job_dir("job-3")
    assert not (tmp_path / "job-3").exists()


async def test_local_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _local_service(monkeypatch, tmp_path / "created-on-init")
    await svc.initialize()
    assert await svc.health_check() is True


async def test_local_retrieve_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    svc = _local_service(monkeypatch, tmp_path)
    await svc.initialize()
    assert await svc.retrieve(str(tmp_path / "nonexistent" / "path")) is None


async def test_storage_service_selects_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    svc = _local_service(monkeypatch, tmp_path)
    await svc.initialize()
    assert svc.backend_type == "local"
    stored = await svc.store_bytes("job-x", b"abc", "test.txt")
    assert Path(stored).read_text() == "abc"
    assert svc.get_local_path(stored) == Path(stored).resolve()


async def test_local_rejects_absolute_outside_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retrieve() with an absolute path outside the storage root must be
    refused even though ``is_absolute()`` would have happily honoured it."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"sensitive")
    svc = _local_service(monkeypatch, root)
    await svc.initialize()
    assert await svc.retrieve(str(outside)) is None
    assert svc.get_local_path(str(outside)) is None
    assert outside.read_bytes() == b"sensitive"


async def test_local_rejects_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A relative path with ``..`` must be confined to the storage root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"sensitive")
    svc = _local_service(monkeypatch, root)
    await svc.initialize()
    assert await svc.retrieve("../secret.bin") is None
    assert svc.get_local_path("../secret.bin") is None
    assert outside.read_bytes() == b"sensitive"
