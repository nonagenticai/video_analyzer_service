"""Local + MinIO storage for video_analyzer_service artifacts.

Object I/O is delegated to the shared ``w2v_common.storage.StorageService``
facade (the fleet-wide consolidation of the per-service MinIO clients). This
thin wrapper keeps the video-analyzer domain concerns the facade does not know
about: the ``{job_id}/{filename}`` object-key layout, per-job local scratch
directories (``get_job_dir`` / ``get_local_path``), and the traversal guard on
stale/tampered local paths (``_safe_local_path``).

Both backends stay live: on ``minio`` the facade probe at boot is best-effort
so a MinIO blip cannot CrashLoop the pod; on ``local`` writes land under the
configured ``local_path``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from w2v_common.storage import StorageService as CommonStorage

from src.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Video-analyzer artifact storage over the shared object-storage facade."""

    def __init__(self) -> None:
        settings = get_settings()
        self._backend_type = "minio" if settings.storage.backend.lower() == "minio" else "local"
        self._local_path = Path(settings.storage.local_path)
        # Local scratch root where the pipeline stages materialise frames/audio
        # before they are streamed up to MinIO. Mirrors the retired
        # MinIOBackend._tmp_root so get_job_dir / get_local_path are unchanged.
        self._scratch_root = Path(tempfile.gettempdir()) / "video-analyzer-minio"
        self._storage = CommonStorage(settings.storage)

    @property
    def backend_type(self) -> str:
        return self._backend_type

    async def initialize(self) -> None:
        """Initialize the storage backend.

        For ``minio`` the bucket probe is best-effort: if the upstream is
        unreachable at boot we log and continue so the service still starts and
        serves non-storage routes; the bucket is created on first write.
        Crashing here turned a minio blip into fleet-wide w2v CrashLoop on
        2026-05-11.
        """
        if self._backend_type == "minio":
            self._scratch_root.mkdir(parents=True, exist_ok=True)
            try:
                await self._storage.initialize()
            except Exception as exc:  # noqa: BLE001 — startup must tolerate any minio failure
                logger.warning(
                    "MinIO bucket probe failed at startup (%s: %s) — proceeding "
                    "without verification; bucket will be created on first write.",
                    type(exc).__name__,
                    exc,
                )
        else:
            await self._storage.initialize()

    def get_job_dir(self, job_id: str) -> Path:
        """Return a writable local scratch directory for the given job."""
        root = self._local_path if self._backend_type == "local" else self._scratch_root
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    async def store_bytes(self, job_id: str, data: bytes, filename: str) -> str:
        """Persist raw bytes and return the storage reference.

        local → absolute path under ``local_path``. minio → the object key.
        """
        key = self._object_name(job_id, filename)
        return await self._storage.put_bytes(key, data, content_type="application/octet-stream")

    async def store_file(self, job_id: str, file_path: Path, filename: str) -> str:
        """Persist a file from a local Path (STREAMING for MinIO — no full read
        into memory) and return the storage reference.

        For the local backend the pipeline typically writes an artifact directly
        into ``get_job_dir(job_id)`` under the same ``filename`` it stores it as,
        so the source already IS the destination — copying it onto itself would
        raise ``SameFileError``. Preserve the retired LocalBackend's no-op skip
        in that case; otherwise stream through the facade.
        """
        key = self._object_name(job_id, filename)
        if self._backend_type == "local":
            dest = self._local_path / key
            if Path(file_path).resolve() == dest.resolve():
                return str(dest)
        return await self._storage.put_file(key, file_path, content_type="application/octet-stream")

    # --- user-uploaded source videos (multi-replica-safe) --------------------
    #
    # A ``POST /analyze/upload`` may be accepted on one replica while the async
    # analyze work runs on another (replicas>1 / HPA). Storing the source on
    # pod-local disk made it unresolvable on any other pod. These helpers keep
    # the uploaded source in the shared object store under a TENANT-SCOPED key,
    # so ``PipelineRunner._resolve_source`` can materialise it on whichever pod
    # runs the job. The key is namespaced by tenant so one tenant can never
    # resolve (or collide with) another tenant's upload.

    UPLOAD_PREFIX = "uploads"

    @classmethod
    def upload_key(cls, tenant_id: str, job_id: str, filename: str) -> str:
        """Tenant-scoped object key for a user-uploaded source video."""
        return f"{cls.UPLOAD_PREFIX}/{tenant_id}/{job_id}/{filename}"

    async def store_upload(
        self, tenant_id: str, job_id: str, file_path: Path, filename: str
    ) -> str:
        """Stream an uploaded source video into the object store (STREAMING —
        never buffers the whole video in memory; delegates to the facade's
        ``put_file`` / ``fput_object`` chunked multipart, unlike ``put_bytes``
        which would load a multi-GB file into RAM). Returns the storage
        reference (object key on minio, absolute path on local)."""
        key = self.upload_key(tenant_id, job_id, filename)
        return await self._storage.put_file(key, file_path, content_type="application/octet-stream")

    def local_upload_path(self, tenant_id: str, job_id: str, filename: str) -> Path | None:
        """Return the on-disk path of a locally-stored upload if present.

        Only the ``local`` backend materialises the object on this pod's disk;
        for ``minio`` the object lives in the store and callers must fetch it
        (see :meth:`presign_upload`). Returns ``None`` when absent / on minio.
        """
        if self._backend_type != "local":
            return None
        candidate = self._local_path / self.upload_key(tenant_id, job_id, filename)
        return candidate if candidate.exists() else None

    async def presign_upload(self, tenant_id: str, job_id: str, filename: str) -> str | None:
        """Presigned GET URL for the uploaded source (minio only; ``None`` on
        local). The URL targets the internal object store host, which is in the
        SSRF downloader's ``ALLOWED_URL_HOSTS`` allowlist (see #64), so the
        pipeline can stream it down on any replica."""
        key = self.upload_key(tenant_id, job_id, filename)
        return await self._storage.presign_get(key)

    async def delete_upload(self, tenant_id: str, job_id: str) -> None:
        """Delete every uploaded-source object for a job (tenant-scoped prefix)."""
        prefix = f"{self.UPLOAD_PREFIX}/{tenant_id}/{job_id}/"
        await self._storage.delete_prefix(prefix)

    async def retrieve(self, path: str) -> bytes | None:
        """Retrieve file contents by storage reference."""
        if self._backend_type == "minio":
            return await self._storage.get_bytes(path)
        safe = self._safe_local_path(path)
        if safe is None:
            return None
        return await self._storage.get_bytes(str(safe))

    async def delete_job_dir(self, job_id: str) -> None:
        """Delete all stored artefacts for a job (and any local scratch)."""
        if self._backend_type == "minio":
            await self._storage.delete_prefix(f"{job_id}/")
            scratch = self._scratch_root / job_id
            if scratch.exists():
                shutil.rmtree(scratch)
        else:
            await self._storage.delete_prefix(job_id)

    async def health_check(self) -> bool:
        """Return True if the backend is healthy."""
        return await self._storage.check_health()

    def get_local_path(self, path: str) -> Path | None:
        """Return a local Path for ``path`` if it exists on disk, else None."""
        if self._backend_type == "minio":
            candidate = self._scratch_root / path
            return candidate if candidate.exists() else None
        safe = self._safe_local_path(path)
        if safe is None:
            return None
        return safe if safe.exists() else None

    @staticmethod
    def _object_name(job_id: str, filename: str) -> str:
        """Object-key layout: ``{job_id}/{filename}``."""
        return f"{job_id}/{filename}"

    def _safe_local_path(self, raw: str) -> Path | None:
        """Resolve ``raw`` and reject anything outside ``self._local_path``.

        Defense in depth: ``store_bytes`` / ``store_file`` only emit paths under
        the configured root, so a rejection here means a stale row from a
        different configuration or a tampered DB entry. Returning ``None`` keeps
        the existing "treat as not found" contract (and blocks ``../`` traversal
        that a bare ``is_absolute()`` check would miss).
        """
        root = self._local_path.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._local_path / raw
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", raw)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", raw)
            return None
        return resolved
