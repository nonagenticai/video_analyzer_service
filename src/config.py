"""Configuration for video_analyzer_service (port 6317)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import APISettings as _BaseAPISettings
from w2v_common.config import (
    AuthSettings,
    DatabaseSettings,
    LoggingSettings,
    RedisSettings,
)
from w2v_common.config import StorageSettings as _BaseStorageSettings

__all__ = [
    "APISettings",
    "AuthSettings",
    "DatabaseSettings",
    "LoggingSettings",
    "PipelineSettings",
    "RedisSettings",
    "Settings",
    "StorageSettings",
    "TranscriptionClientSettings",
    "VisionSettings",
    "clear_settings_cache",
    "get_settings",
]


class APISettings(_BaseAPISettings):
    """Video analyzer API settings."""

    port: int = Field(default=6317, description="API server port")


class StorageSettings(_BaseStorageSettings):
    """Video-analyzer local + MinIO storage config."""

    minio_bucket: str = Field(default="video-analyzer-service", description="MinIO bucket name")
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs",
    )
    upload_dir: str = Field(
        default="/app/data/uploads",
        description="Local directory for user-uploaded video files",
    )
    max_upload_size_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        description="Maximum accepted upload size (default 4 GiB)",
    )


class PipelineSettings(BaseSettings):
    """Pipeline runtime configuration."""

    frame_interval_seconds: float = Field(
        default=3.0, description="Desired interval between extracted frames"
    )
    max_frames: int = Field(default=200, description="Hard upper bound on extracted frame count")
    max_concurrent_jobs: int = Field(
        default=2,
        ge=1,
        description=(
            "Maximum pipeline jobs this pod executes at once. Enforced by a semaphore "
            "in PipelineRunner._run_job: excess jobs queue with their lease held (so "
            "the sweeper cannot reap a job that is merely waiting its turn) instead of "
            "all landing on ffmpeg/tesseract/vision simultaneously."
        ),
    )
    job_deadline_seconds: int = Field(
        default=1800,
        ge=0,
        description=(
            "Wall-clock ceiling on a single analysis job (0 disables). Past it the job "
            "is cancelled and marked failed, keeping whatever partial frame_analyses "
            "were persisted. 1800s is deliberately ~4x the orchestrator's 420s poll "
            "timeout (PR #890): a job that has run four times longer than any caller "
            "will wait has already lost its caller, so every further vision call is "
            "spend nobody reads. Without it the heartbeat renews the lease forever and "
            "an abandoned job runs to completion and emits a cost event."
        ),
    )
    default_language: str | None = Field(
        default=None, description="Default transcription language (None = auto)"
    )
    enable_vision: bool = Field(default=True, description="Run vision LLM stage by default")
    enable_ocr: bool = Field(default=True, description="Run OCR stage by default")
    enable_quality: bool = Field(default=True, description="Run quality metric stage by default")

    # --- Ownership lease + stranded-job sweeper (migration 005) ---
    # These live on PipelineSettings rather than the top-level Settings on purpose: the
    # PIPELINE_ prefix is already an accepted env prefix in GitOps and in tests'
    # _clear_env fixture, whereas a new top-level field needs an explicit
    # validation_alias and a prefix nothing whitelists.
    job_lease_seconds: int = Field(
        default=300,
        ge=30,
        description=(
            "Claim-lease TTL for a running analysis job. The owning pod renews it every "
            "lease//3, so FIVE missed heartbeats are needed before a live job looks "
            "abandoned. A non-terminal row past its lease lost its worker and is reaped."
        ),
    )
    sweeper_enabled: bool = Field(
        default=True,
        description=(
            "Run the background sweeper that fails jobs whose owning pod died. Without "
            "it, reaping happens ONLY when a client polls that specific job — so a job "
            "nobody polls is never reaped (measured: 23 rows stuck, oldest ~11 days)."
        ),
    )
    sweeper_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds between stranded-job sweeps. Each sweep also RE-CLAIMS lapsed jobs "
            "(migration 007), so this bounds how long a job given back on graceful "
            "shutdown waits before a live pod re-runs it. Both SQL calls are cheap "
            "partial-index scans, so 30s costs nothing."
        ),
    )
    job_max_reclaims: int = Field(
        default=2,
        ge=0,
        description=(
            "How many times a job whose owning pod DIED (lease lapsed without a graceful "
            "release) is re-claimed and re-run by a live pod before it is failed as "
            "stranded. A lease released on SIGTERM does not spend an attempt. 0 disables "
            "re-claim entirely and restores fail-on-lapse."
        ),
    )
    sweeper_max_per_sweep: int = Field(
        default=50,
        ge=1,
        description=(
            "Upper bound on rows failed per sweep, so a large backlog is drained over "
            "several passes instead of in one long-running transaction."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class TranscriptionClientSettings(BaseSettings):
    """Downstream transcription_worker client config."""

    # url is required (TRANSCRIPTION_CLIENT_URL) — set via configmap to
    # asr.test:9201 (mini2 launchd whisper). No localhost fallback.
    url: str = Field(description="transcription_worker base URL")
    timeout_seconds: float = Field(default=600.0, description="Transcription HTTP timeout")

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTION_CLIENT_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


class VisionSettings(BaseSettings):
    """Vision LLM configuration.

    All real vision calls route through worker_management (per memory
    `wm-as-sole-llm-gateway.md`). The service no longer owns LLM provider
    config — WM workers decide the model. The remaining knobs are:

    * ``default_provider``: ``wm`` (default) or ``stub`` for the offline
      deterministic path used in CI / fixtures.
    * ``model_override``: optional per-job model override sent to WM
      (e.g. ``claude-sonnet-4-5``, ``gpt-4o``, ``gemma-4-vl``). Empty
      string means "use whatever the WM worker is configured for".
    * ``batch_size`` / ``max_cost_usd``: pipeline-level controls,
      unchanged.
    """

    default_provider: str = Field(
        default="wm",
        description=(
            "Default vision provider: 'wm' (worker_management — production) "
            "or 'stub' / 'fake' (offline deterministic — CI). "
            "ANALYZER_PROVIDER env var, when set, overrides this field."
        ),
    )
    model_override: str = Field(
        default="",
        description=(
            "Optional per-job model override sent to WM as model_override "
            "(e.g. claude-sonnet-4-5). Empty string = use the WM worker's "
            "configured default."
        ),
    )
    batch_size: int = Field(default=8, description="Frames per vision batch")
    max_cost_usd: float = Field(
        default=5.0, description="Default per-job cost cap before vision is halted"
    )

    model_config = SettingsConfigDict(
        env_prefix="VISION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main video_analyzer_service settings."""

    app_name: str = Field(default="Video Analyzer Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Deployment environment")
    metering_redis_url: str = Field(
        default="",
        description=(
            "Redis URL for the cost-bus cost-event stream (separate from "
            "the service-local Redis). Empty = metering disabled."
        ),
        validation_alias="METERING_REDIS_URL",
    )
    analyzer_provider: str | None = Field(
        default=None,
        description=(
            "Top-level vision provider override (env: ANALYZER_PROVIDER). "
            "When set, takes precedence over vision.default_provider. "
            "Use 'stub' to force the offline deterministic provider."
        ),
        validation_alias="ANALYZER_PROVIDER",
    )
    downstream_license_token: str | None = Field(
        default=None,
        description=(
            "Service-account license token (env DOWNSTREAM_LICENSE_TOKEN, from "
            "video-analyzer-service-secrets). TranscriptionClient sends it as "
            "'X-Tenant-Id: Bearer <token>' so it can authenticate to the "
            "modality-gateway, which enforces license auth. asr.test:9201 "
            "trusts bare in-cluster identity and ignores it, so this is safe "
            "regardless of the transcription target. The real customer tenant "
            "still threads via X-Effective-Tenant-Id (identity_event_hook), which "
            "the gateway honours because this token's subject is a delegator."
        ),
        validation_alias="DOWNSTREAM_LICENSE_TOKEN",
    )
    allowed_url_hosts: list[str] = Field(
        default_factory=list,
        description=(
            "Exact hostnames the SSRF guard permits despite resolving to an "
            "internal/private address (env ALLOWED_URL_HOSTS, JSON list). The "
            "record-stage QA in w2v_orchestrator hands us presigned clip URLs on "
            "the internal object store (minio.test); without this the SSRF "
            "guard rejects them and per-clip visual/OCR QA silently no-ops. Keep "
            "it to the minimum set of trusted internal hosts."
        ),
        validation_alias="ALLOWED_URL_HOSTS",
    )

    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    transcription_client: TranscriptionClientSettings = Field(
        default_factory=TranscriptionClientSettings
    )
    vision: VisionSettings = Field(default_factory=VisionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


def clear_settings_cache() -> None:
    """Drop cached settings (tests)."""
    get_settings.cache_clear()
