"""MinIO-backend coverage — asserts the wrapper delegates to the shared
``w2v_common.storage`` facade instead of talking to a MinIO client directly.

The hand-rolled sync-minio backend was retired in favour of the shared facade
(``miniopy-async``). These tests build the service on the ``minio`` backend and
then patch the facade primitives (``svc._storage.<method>``) to assert the
wrapper forwards the right object key / arguments and preserves the
``{job_id}/{filename}`` layout — never touching a real MinIO.

ty gotcha: patch via ``monkeypatch.setattr(svc._storage, ...)`` (dynamic),
never direct attribute assignment ``svc._storage.foo = fn`` (ty flags
invalid-assignment; the mypy method-assign ignore does not silence ty).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import clear_settings_cache
from src.services.storage_service import StorageService


@pytest.fixture
def minio_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> StorageService:
    """A StorageService bound to the ``minio`` backend (facade unprobed)."""
    monkeypatch.setenv("STORAGE_BACKEND", "minio")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("STORAGE_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("STORAGE_MINIO_SECRET_KEY", "sk")
    clear_settings_cache()
    svc = StorageService()
    # Keep the local scratch under tmp so get_job_dir / delete don't touch /tmp.
    svc._scratch_root = tmp_path / "scratch"
    return svc


def test_backend_type_is_minio(minio_service: StorageService) -> None:
    assert minio_service.backend_type == "minio"


async def test_store_bytes_delegates_with_object_key(minio_service: StorageService) -> None:
    seen: dict[str, object] = {}

    async def fake_put_bytes(key: str, data: bytes, *, content_type: str = "") -> str:
        seen["key"] = key
        seen["data"] = data
        seen["content_type"] = content_type
        return key

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(minio_service._storage, "put_bytes", fake_put_bytes)
    ref = await minio_service.store_bytes("job-1", b"abc", "walk.md")
    monkeypatch.undo()

    assert ref == "job-1/walk.md"
    assert seen["key"] == "job-1/walk.md"
    assert seen["data"] == b"abc"
    assert seen["content_type"] == "application/octet-stream"


async def test_store_file_streams_to_facade(minio_service: StorageService, tmp_path: Path) -> None:
    src = tmp_path / "artifact.md"
    src.write_text("body")
    seen: dict[str, object] = {}

    async def fake_put_file(
        key: str, local_path: str | Path, *, content_type: str = "", **_: object
    ) -> str:
        seen["key"] = key
        seen["local_path"] = str(local_path)
        return key

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "put_file", fake_put_file)
    ref = await minio_service.store_file("job-2", src, "uploaded.md")
    mp.undo()

    assert ref == "job-2/uploaded.md"
    assert seen["key"] == "job-2/uploaded.md"
    assert seen["local_path"] == str(src)


async def test_store_upload_streams_via_put_file_never_buffers(
    minio_service: StorageService, tmp_path: Path
) -> None:
    """A user-uploaded source video must stream to MinIO via ``put_file``
    (chunked ``fput_object``), NEVER be read whole into RAM via ``put_bytes`` —
    a multi-GB video would OOM the pod otherwise."""
    src = tmp_path / "big.mp4"
    src.write_bytes(b"VIDEO-BYTES")
    seen: dict[str, object] = {}

    async def fake_put_file(
        key: str, local_path: str | Path, *, content_type: str = "", **_: object
    ) -> str:
        seen["key"] = key
        seen["local_path"] = str(local_path)
        return key

    async def forbidden_put_bytes(*_a: object, **_k: object) -> str:
        raise AssertionError("store_upload must stream via put_file, not buffer via put_bytes")

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "put_file", fake_put_file)
    mp.setattr(minio_service._storage, "put_bytes", forbidden_put_bytes)
    ref = await minio_service.store_upload("tenant-a", "job-1", src, "big.mp4")
    mp.undo()

    assert ref == "uploads/tenant-a/job-1/big.mp4"
    assert seen["key"] == "uploads/tenant-a/job-1/big.mp4"
    assert seen["local_path"] == str(src)


async def test_presign_upload_uses_tenant_scoped_key(minio_service: StorageService) -> None:
    seen: dict[str, object] = {}

    async def fake_presign(
        key: str, *, expires: int | None = None, bucket: str | None = None
    ) -> str:
        seen["key"] = key
        return f"https://minio.test/video-analyzer-service/{key}?sig=x"

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "presign_get", fake_presign)
    url = await minio_service.presign_upload("tenant-a", "job-1", "clip.mp4")
    mp.undo()

    assert seen["key"] == "uploads/tenant-a/job-1/clip.mp4"
    assert url is not None and "uploads/tenant-a/job-1/clip.mp4" in url
    # On minio nothing is materialised on this pod's local disk.
    assert minio_service.local_upload_path("tenant-a", "job-1", "clip.mp4") is None


async def test_delete_upload_delegates_tenant_scoped_prefix(minio_service: StorageService) -> None:
    seen: dict[str, object] = {}

    async def fake_delete_prefix(prefix: str) -> bool:
        seen["prefix"] = prefix
        return True

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "delete_prefix", fake_delete_prefix)
    await minio_service.delete_upload("tenant-a", "job-1")
    mp.undo()

    assert seen["prefix"] == "uploads/tenant-a/job-1/"


async def test_retrieve_delegates_by_key(minio_service: StorageService) -> None:
    async def fake_get_bytes(ref: str) -> bytes | None:
        return b"object-bytes" if ref == "job-1/walk.md" else None

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "get_bytes", fake_get_bytes)
    data = await minio_service.retrieve("job-1/walk.md")
    mp.undo()
    assert data == b"object-bytes"


async def test_delete_job_dir_delegates_prefix_and_clears_scratch(
    minio_service: StorageService,
) -> None:
    seen: dict[str, object] = {}

    async def fake_delete_prefix(prefix: str) -> bool:
        seen["prefix"] = prefix
        return True

    scratch_job = minio_service._scratch_root / "job-9"
    scratch_job.mkdir(parents=True)

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "delete_prefix", fake_delete_prefix)
    await minio_service.delete_job_dir("job-9")
    mp.undo()

    assert seen["prefix"] == "job-9/"
    assert not scratch_job.exists()


async def test_health_check_delegates(minio_service: StorageService) -> None:
    async def fake_check_health() -> bool:
        return True

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "check_health", fake_check_health)
    assert await minio_service.health_check() is True
    mp.undo()


async def test_initialize_delegates_on_minio(minio_service: StorageService) -> None:
    called: dict[str, bool] = {"init": False}

    async def fake_initialize() -> None:
        called["init"] = True

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "initialize", fake_initialize)
    await minio_service.initialize()
    mp.undo()

    assert called["init"] is True
    assert minio_service._scratch_root.exists()


async def test_initialize_tolerates_facade_failure(minio_service: StorageService) -> None:
    """Anti-crashloop: a MinIO blip at boot must NOT propagate out of
    ``initialize`` — the pod still starts and serves non-storage routes."""

    async def boom() -> None:
        raise ConnectionError("minio unreachable")

    mp = pytest.MonkeyPatch()
    mp.setattr(minio_service._storage, "initialize", boom)
    # Must complete without raising.
    await minio_service.initialize()
    mp.undo()
    assert minio_service._scratch_root.exists()


def test_get_job_dir_uses_scratch_on_minio(minio_service: StorageService) -> None:
    job_dir = minio_service.get_job_dir("job-1")
    assert job_dir.exists()
    assert job_dir == minio_service._scratch_root / "job-1"


def test_get_local_path_reads_scratch_on_minio(minio_service: StorageService) -> None:
    (minio_service._scratch_root).mkdir(parents=True, exist_ok=True)
    f = minio_service._scratch_root / "abc.txt"
    f.write_text("ok")
    assert minio_service.get_local_path("abc.txt") == f
    assert minio_service.get_local_path("missing.txt") is None
