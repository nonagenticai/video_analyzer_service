# Operation ID Plan: video_analyzer_service

Canonical registry of `operation_id` values for every route in
`video_analyzer_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service, video_compiler_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service hosts a single analysis domain plus a health surface.
Verbs are prefixed with `video_` / `analysis_` where the generic verb
(`upload_*`, `download_*`, `delete_*`) would collide across the fleet.

## Files to edit

- `src/asgi.py` (root endpoint)
- `src/api/health.py`
- `src/api/routes/analysis.py`

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function | operation_id   | Rationale                                  |
|--------|----------|----------------|--------------------------------------------|
| GET    | `root`   | `service_root` | Service banner; matches the fleet standard. |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id   | Rationale                                                  |
|--------|----------------|----------------|------------------------------------------------------------|
| GET    | `health_check` | `health_check` | Primary health probe; matches the fleet standard verb.     |
| GET    | `health_live`  | `health_live`  | Liveness sub-probe; one-to-one with path.                  |
| GET    | `health_ready` | `health_ready` | Readiness sub-probe; one-to-one with path.                 |

### `src/api/routes/analysis.py` (tag: `analysis`)

| Method | Function                  | operation_id                  | Rationale                                                                                  |
|--------|---------------------------|-------------------------------|--------------------------------------------------------------------------------------------|
| POST   | `create_analysis_job`     | `start_video_analysis`        | `create_analysis_job` is implementation-leaking; `start_video_analysis` describes the verb. |
| POST   | `analyze_audio_beat_grid` | `analyze_audio_beat_grid`     | Synchronous (not a job): names the artefact it returns, so it reads distinctly from the 202 video verbs. |
| POST   | `upload_and_analyze`      | `upload_video_for_analysis`   | Generic `upload_*` would collide; prefix with the resource verb.                            |
| GET    | `get_job_status`          | `get_analysis_status`         | Domain-prefix to disambiguate from generic job-status verbs elsewhere in the fleet.         |
| GET    | `get_job_result`          | `get_analysis_result`         | Same — domain-prefix.                                                                       |
| GET    | `get_verification_result` | `get_analysis_verification`   | Domain-prefix; verification is part of the analysis domain.                                 |
| GET    | `download_artifact`       | `download_analysis_artifact`  | Generic `download_artifact` collides with other services; prefix with the domain.           |
| DELETE | `delete_job`              | `delete_analysis_job`         | Same — domain-prefix.                                                                       |

## Verification

```bash
cd video_analyzer_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of every right-hand column above.
Drift (rename, missed decorator, fallback) fails the test loudly.

## Disambiguation rationale

Concrete renames where the function name was generic enough to collide
across the fleet or within this service:

- `create_analysis_job` → `start_video_analysis` — source name leaks
  implementation; the op_id describes the operation in the org's standard
  verb_noun shape.
- `upload_and_analyze` → `upload_video_for_analysis` — `upload_*` is too
  generic at fleet scope.
- `get_job_status` / `get_job_result` / `get_verification_result` /
  `download_artifact` / `delete_job` → all domain-prefixed with
  `analysis_` to keep them out of the generic `*_job` / `*_artifact`
  namespaces shared with other services.

## Future routes

When adding a route in this service, **always** set `operation_id="…"`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
