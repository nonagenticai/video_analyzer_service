# Video Analyzer Service

Takes a video and runs a flag-gated CPU pipeline that produces a structured walkthrough, UI inventory, frame index, and optional verification report. Port 6317.

## What Video Analyzer Service Does

The service accepts a video two ways: a JSON request carrying a remote `video_url`, or a multipart upload of the file itself. Caller-supplied local filesystem paths are not accepted — the former `video_path` field was removed (it let a request read any file on the pod, and never resolved across replicas) and a request still carrying it is rejected. Each submission creates an `AnalysisJob` row in PostgreSQL, returns a `job_id` with HTTP 202, and schedules the work on a background pipeline runner so the request returns immediately. Callers then poll `GET /api/analyze/{job_id}` for status and stage, and fetch the assembled result or verification report once the job reaches a terminal state. The primary upstream caller is `w2v_orchestrator`'s `video_analyze` stage, but the same endpoints serve standalone user uploads and internal verification of `video_production_service` output against a script and slide deck.

Inside the pipeline the service does all the frame-level work itself on CPU — there is no GPU dependency. It probes metadata with ffprobe, extracts frames and audio with ffmpeg, runs OpenCV histogram chi-square scene detection, reads on-screen text with Tesseract OCR, and computes quality metrics (Laplacian-variance sharpness, brightness, motion, dominant palette) with numpy and imagehash. The two stages it does not do locally are delegated: transcription is handed to the ASR (Whisper) unit, and every vision-LLM call — per-frame descriptions plus the final walkthrough/UI-inventory synthesis — is routed through `worker_management`, which selects the underlying vision-capable model. This service never calls an LLM provider SDK directly.

Transcription targets `transcription_worker_service` at `asr.test:9201`, a macOS launchd Whisper unit (mini2 / m1 / m4 pool), configured through the required `TRANSCRIPTION_CLIENT_URL` setting that has no localhost fallback. Vision work goes to `worker_management` via `POST /workers/{name}/message` using the `video-frame-analyzer` and `video-synthesis` workers, with frames sent as base64 content blocks and an `output_schema` per call. Outbound identity (`X-Tenant-Id` / `X-License-Id` / `X-Subject-Id`) is forwarded automatically on both clients so downstream services stay tenant-scoped.

Results accumulate on the job row as each stage completes — transcript data, scene boundaries, frame analyses, quality metrics, and a verification result — and generated artifacts are written to MinIO (with a local upload directory for multipart inputs). The optional verification stage checks required and forbidden phrases, transcript similarity, and reference-image pHash matches, producing a pass/fail report. Job state and all results are stored in PostgreSQL 18 (psycopg3, async SQLAlchemy) with row-level security enforced, so every read and write is bound to the requesting tenant. The background executor catches broad exceptions to guarantee every job lands in a terminal status rather than wedging, and remote-URL inputs pass through SSRF validation before download.

## Features

- **Two ingestion paths** — accepts a remote URL or a direct multipart upload so the same service handles orchestrator stages, verification, and ad-hoc user submissions; caller-supplied local filesystem paths are rejected
- **Async job model** — returns a `job_id` with 202 and runs the pipeline on a background runner so callers never block on multi-minute video work
- **CPU-only frame analysis** — probing, frame/audio extraction, scene detection, OCR, and quality metrics run locally on OpenCV/Tesseract/ffmpeg/numpy with no GPU dependency
- **Delegated transcription** — offloads speech-to-text to the ASR Whisper unit at `asr.test:9201` rather than bundling a model, keeping the image lean
- **Delegated vision LLM** — routes all frame descriptions and walkthrough synthesis through `worker_management` so the service owns no provider keys and the model is chosen upstream
- **Verification stage** — checks required/forbidden phrases, transcript similarity, and reference-image pHash matches so produced videos can be validated against a script and deck
- **Row-level security** — every session binds the request tenant, and cross-tenant `job_id` lookups 404, so job data is isolated at the database layer
- **Cost bounding** — `max_frames`, `frame_interval_seconds`, and a per-job vision cost cap (`VISION_MAX_COST_USD`) keep pipeline spend and duration in check
- **SSRF-guarded downloads** — remote-URL inputs are validated before fetch to prevent internal-network access via crafted URLs
- **Offline stub provider** — `ANALYZER_PROVIDER=stub` selects a deterministic offline vision provider returning canned-but-valid JSON for CI, sandboxed E2E, and local dev
- **Optional cost metering** — emits fire-and-forget cost events to a separate metering Redis stream when configured, and is a silent no-op when not
- **Health probes** — standard liveness, readiness (DB-checked), and startup endpoints for Kubernetes orchestration

## Authentication

Auth enabled by default (`AUTH_ENABLED=true`). The `w2v_common` `AuthMiddleware` extracts a license token from the `X-Tenant-Id: Bearer <token>` header (the `Authorization` header is reserved for caller-side Keycloak JWTs), introspects it against the licenses service, and caches the result for 5 minutes. This service **enforces row-level security** — the resolved tenant is bound on every database session, so a job created under one tenant is invisible to another.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, cached 5 min (`CACHE_LICENSE_TTL=300`) |
| 2 | Trusted network | Bypass for private CIDRs (`127.0.0.1/32, ::1/128, 172.16.0.0/12, 10.0.0.0/8`) assigning the default tenant |

Requests with no tenant fall back to a nil-UUID `__unattributed__` sentinel that billing drops rather than a named tenant. Health, docs, and metrics paths (`/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/docs`, `/redoc`, `/openapi.json`, `/metrics`) are always exempt.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6317
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Analysis `/api/analyze`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Create an analysis job from JSON (`video_url`; local paths not accepted); returns `job_id` |
| POST | `/analyze/upload` | Create a job from a multipart upload (`file` + optional JSON `config`) |
| GET | `/analyze/{job_id}` | Poll job status, stage, and progress |
| GET | `/analyze/{job_id}/result` | Fetch the full accumulated result |
| GET | `/analyze/{job_id}/verification` | Fetch the verification report |
| GET | `/analyze/{job_id}/artifact/{name}` | Download a single artifact by name |
| DELETE | `/analyze/{job_id}` | Delete a job, its row, and all artifacts |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Combined DB + API health |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks DB) |
| GET | `/healthz` | Liveness probe (alias) |
| GET | `/readyz` | Readiness probe (alias) |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6317` | Listen port |
| `API_DEBUG` | `false` | Enable /docs and /redoc |
| `AUTH_ENABLED` | `true` | License-token auth middleware |
| `AUTH_DEFAULT_TENANT_ID` | `default` | Tenant for trusted-network requests |
| `LOG_LEVEL` | `INFO` | Log level |
| `DATABASE_URL` | -- | Privileged Postgres URL (migrations init container) |
| `RUNTIME_DATABASE_URL` | -- | Restricted `w2v_video_analyzer_app` role URL (runtime, RLS) |
| `DATABASE_POOL_SIZE` | (w2v_common default) | Async SQLAlchemy pool size |
| `TRANSCRIPTION_CLIENT_URL` | -- (**required**) | ASR Whisper base URL — `asr.test:9201`, no localhost fallback |
| `TRANSCRIPTION_CLIENT_TIMEOUT_SECONDS` | `600.0` | Transcription HTTP timeout |
| `VISION_DEFAULT_PROVIDER` | `wm` | Vision provider: `wm` (worker_management) or `stub`/`fake` (offline) |
| `ANALYZER_PROVIDER` | -- | Top-level provider override; `stub` forces the offline deterministic path |
| `VISION_MODEL_OVERRIDE` | `` | Optional per-job model override sent to WM (empty = WM worker default) |
| `VISION_BATCH_SIZE` | `8` | Frames per vision batch |
| `VISION_MAX_COST_USD` | `5.0` | Per-job vision spend cap |
| `PIPELINE_FRAME_INTERVAL_SECONDS` | `3.0` | Target interval between extracted frames |
| `PIPELINE_MAX_FRAMES` | `200` | Hard upper bound on extracted frames |
| `PIPELINE_MAX_CONCURRENT_JOBS` | `2` | Max concurrent pipeline jobs |
| `PIPELINE_ENABLE_VISION` | `true` | Run the vision LLM stage by default |
| `PIPELINE_ENABLE_OCR` | `true` | Run the OCR stage by default |
| `PIPELINE_ENABLE_QUALITY` | `true` | Run the quality-metric stage by default |
| `STORAGE_MINIO_BUCKET` | `video-analyzer-service` | MinIO bucket for artifacts |
| `STORAGE_UPLOAD_DIR` | `/app/data/uploads` | Local dir for multipart uploads |
| `STORAGE_MAX_UPLOAD_SIZE_BYTES` | `2147483648` | Max accepted upload size (2 GiB) |
| `METERING_REDIS_URL` | `` | Cost-event stream Redis (empty = metering disabled) |

## Architecture

```
src/
├── asgi.py                          # FastAPI app, lifespan, CORS + auth + request-id middleware, OpenAPI, root
├── config.py                        # Pydantic settings (port 6317, pipeline/vision/transcription/storage)
├── api/
│   ├── health.py                    # /health, /health/live, /health/ready, /healthz, /readyz
│   ├── schemas.py                   # Request/response DTOs (AnalysisRequest, JobResult, VerificationResult)
│   └── routes/
│       └── analysis.py              # /api/analyze CRUD + upload + artifact download (tenant-bound sessions)
├── domain/
│   └── analysis.py                  # Internal dataclasses (VideoMetadata, ExtractedFrame, FrameAnalysis, QualityMetrics)
├── repository/
│   └── job_repository.py            # Tenant-scoped AnalysisJob queries (create, get, update, mark_failed, delete)
├── database/
│   ├── connection.py                # Async engine lifecycle
│   ├── session.py                   # Session maker
│   ├── models.py                    # AnalysisJob ORM model + RLS DDL registration
│   └── migrations/
│       └── versions/                # 001_initial, 002_rls
├── middleware/
│   └── request_id.py                # Request-ID ContextVar binding (outermost layer)
├── services/
│   ├── pipeline.py                  # PipelineRunner — stage orchestration + background execution
│   ├── protocols.py                 # Structural typing for pipeline stage dependencies
│   ├── video_prober.py              # ffprobe metadata
│   ├── frame_extractor.py           # ffmpeg frame extraction
│   ├── audio_extractor.py           # ffmpeg audio extraction
│   ├── ffmpeg_utils.py              # Shared ffmpeg helpers
│   ├── scene_detector.py            # OpenCV histogram chi-square scene detection
│   ├── ocr_service.py               # Tesseract OCR
│   ├── quality_analyzer.py          # Sharpness / brightness / motion / palette metrics
│   ├── image_matcher.py             # Reference-image pHash matching
│   ├── verifier.py                  # Required/forbidden phrases + transcript similarity
│   ├── transcription_client.py      # asr.test:9201 client
│   ├── video_downloader.py          # Remote-URL / yt-dlp download
│   ├── url_security.py              # SSRF validation for remote URLs
│   ├── artifact_generator.py        # Walkthrough / UI-inventory artifact assembly
│   ├── storage_service.py           # MinIO + local artifact storage
│   ├── metering.py                  # Optional cost-event Redis stream
│   └── vision/
│       ├── base.py                  # Vision provider interface + errors
│       ├── wm_provider.py           # worker_management vision provider (production)
│       ├── fake_provider.py         # Deterministic offline provider (stub)
│       └── prompts.py               # Frame-analysis + synthesis prompt templates
└── templates/                       # Jinja2 artifact templates
```

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests (34 modules)
uv run pytest tests/test_rls_isolation.py  # Tenant row-level-security isolation
uv run pytest tests/test_auth_middleware.py tests/test_openapi.py  # Auth + schema
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Coverage threshold: 95% (`fail_under = 95` in pyproject.toml). Run `./scripts/ci-local.sh` before pushing — it mirrors the CI Lint & Unit Tests job step-for-step.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`.

No image is published from this repository; build it locally with the Dockerfile.

Image: `registry.test:5000/video_analyzer_service:sha-<commit>`.

## Docker

The image bakes ffmpeg, Tesseract, and OpenCV.

```bash
docker build -t video-analyzer-service .
docker run -p 6317:6317 video-analyzer-service
docker run -p 6317:6317 -e LOG_LEVEL=DEBUG -e AUTH_ENABLED=false \
  -e DATABASE_URL=postgresql+psycopg://user:pass@host/db \
  -e TRANSCRIPTION_CLIENT_URL=http://asr.test:9201 video-analyzer-service
```

## Dependencies

- **transcription_worker_service** (`asr.test:9201`) — macOS launchd Whisper unit; transcribes extracted audio (required, no localhost fallback)
- **worker_management** — sole LLM gateway; `POST /workers/{name}/message` for frame descriptions and synthesis (vision model chosen upstream)
- **licenses service** — introspects `X-Tenant-Id` license tokens (via `w2v_common.auth`)
- **storage_service / MinIO** — artifact storage (`video-analyzer-service` bucket)
- **PostgreSQL 18** — job state and results via PgBouncer (`pgbouncer.pgbouncer:6432`), psycopg3, RLS-enforced
- **Redis** — optional cost-event metering stream (separate from any service-local Redis)
- **FastAPI** / **Uvicorn** — web framework and ASGI server
- **Pydantic v2** / **pydantic-settings** — schemas and settings
- **SQLAlchemy** (async) / **psycopg** / **alembic** — ORM, Postgres 18 driver, migrations
- **OpenCV** (headless) / **pytesseract** / **Pillow** / **numpy** / **imagehash** — CPU frame analysis
- **httpx** / **yt-dlp** / **python-multipart** / **Jinja2** / **minio** — HTTP client, downloads, uploads, templating, object storage
- **w2v-common** (>=0.25.0) — shared config, auth middleware, RLS helpers, error envelope

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
