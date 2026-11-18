"""Pydantic schemas for the video_analyzer_service API.

Operational note — caller-supplied filesystem inputs were removed
=================================================================
Two fields that took an unconfined local path are gone: ``AnalysisRequest.
video_path`` and ``ReferenceImage.path``. Both are rejected (not ignored) by
``mode="before"`` validators, and because ``PipelineRunner._load_job_request``
re-validates a job row's persisted ``request_config`` at *execution* time, a row
written before this deploy fails when it runs.

That failure is **terminal, not a wedge**: the pipeline's broad handler
(``pipeline._run_job``) catches it and calls ``mark_failed``, so the job lands in
``failed`` with an ``error_message`` that names ``video_path`` verbatim — an
operator reading the row sees the cause without a code dive.

Blast radius, measured read-only against prod on **2026-08-08**:

.. code-block:: sql

    -- rows that would hit the rejection at all:
    SELECT status, count(*) FROM analysis_jobs
     WHERE video_path IS NOT NULL GROUP BY 1;
    -- rows that would hit it AT EXECUTION TIME (the only ones that matter):
    SELECT count(*) FROM analysis_jobs
     WHERE video_path IS NOT NULL
       AND status IN ('pending', 'queued', 'running');

Result: **7 historical rows total (6 ``failed``, 1 ``completed``), 0 in flight.**
No row can hit the rejection at execution time, so this deploy needs no backfill
and no drain. The hazard is not zero going forward: if the deploy lags its
clients, a not-yet-updated caller can still create a fresh ``video_path`` row in
the gap — which is exactly why the rejection message names the field explicitly.
Re-run the second query before deploying if more than a few days have passed.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class ReferenceImage(BaseModel):
    """A reference image to check against extracted frames."""

    name: str = Field(..., description="Human-readable label")
    # NOTE: ``path`` (a caller-supplied local filesystem path) was REMOVED for
    # the same reason as ``AnalysisRequest.video_path``. It was handed straight
    # to ``Path(...)`` and only existence-checked, so any request could aim
    # verification at ANY file readable by the pod and read back a pHash-distance
    # + existence oracle for it. It was equally dead architecturally: the
    # analyzer runs multi-replica, so a path minted by a remote caller names a
    # filesystem the verifying pod does not have.
    #
    # The replacement is ``reference_key``: a key RELATIVE to this tenant's
    # upload root, resolved (and containment-checked, symlinks followed) under
    # ``<storage.upload_dir>/<tenant_id>/`` by
    # :func:`src.services.verifier._resolve_reference_path`. Requests still
    # carrying ``path`` are rejected by :meth:`_reject_path`, not ignored.
    reference_key: str | None = Field(
        default=None,
        description=(
            "Reference image key RELATIVE to this tenant's upload root "
            "(<upload_dir>/<tenant_id>/). Absolute paths, '..' segments, "
            "backslashes and percent-escapes are rejected."
        ),
    )
    url: str | None = Field(default=None, description="Remote reference image URL")
    required_at_timestamp: float | None = Field(
        default=None, description="Timestamp in seconds where this image must appear"
    )
    timestamp_tolerance_seconds: float = Field(
        default=2.0, description="Allowed ±window around required_at_timestamp"
    )

    PATH_REJECTED: ClassVar[str] = (
        "'path' is no longer accepted on ReferenceImage: an unconfined "
        "filesystem path let verification probe any file readable by the pod, "
        "and never resolved across replicas. Use 'reference_key' (relative to "
        "the tenant upload root) or 'url'."
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_path(cls, data: object) -> object:
        """Fail closed on the removed ``path`` input.

        Also runs when the pipeline re-validates a persisted
        ``verification_criteria`` blob, so a job row written before the removal
        fails terminally instead of replaying the arbitrary-file read.
        """
        # `cast` only to give the untyped `dict` a key type; `isinstance` above
        # is what actually narrows.
        if isinstance(data, dict) and cast("dict[str, object]", data).get("path") is not None:
            raise ValueError(cls.PATH_REJECTED)
        return data

    @field_validator("reference_key")
    @classmethod
    def _key_must_be_confined(cls, value: str | None) -> str | None:
        """Syntactic half of confinement: the key must be a plain relative key.

        The semantic half (resolve + containment under the tenant root, with
        symlinks followed) lives in the verifier, which is where the root and
        the tenant are known. Both halves are required: this one alone cannot
        see a symlink, and that one alone would accept keys whose meaning
        depends on decoding.
        """
        if value is None:
            return None
        key = value.strip()
        if not key:
            raise ValueError("reference_key must not be empty")
        # A percent-escape in an object key here can only be an attempt to
        # smuggle a separator ('%2f', '%5c') or a dot segment ('%2e%2e') past
        # this check — including double-encoded ('%252f', which survives one
        # round of decoding anywhere downstream). Reject the whole class.
        if "%" in key:
            raise ValueError("reference_key must not contain percent-escapes")
        if "\\" in key or "\x00" in key:
            raise ValueError("reference_key must not contain backslashes or NUL bytes")
        if key.startswith("~") or PurePosixPath(key).is_absolute():
            raise ValueError("reference_key must be relative to the tenant upload root")
        # Segment-wise, never substring: '..foo' is a legal name, '..' is not.
        if any(part == ".." for part in PurePosixPath(key).parts):
            raise ValueError("reference_key must not contain '..' segments")
        return key

    @model_validator(mode="after")
    def _one_source(self) -> ReferenceImage:
        if not self.reference_key and not self.url:
            raise ValueError("ReferenceImage requires either 'reference_key' or 'url'")
        return self


class VerificationCriteria(BaseModel):
    """Assertions the analyzer checks against extracted data."""

    required_phrases: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    expected_transcript: str | None = None
    min_transcript_similarity: float = Field(default=0.85, ge=0.0, le=1.0)

    required_ocr_text: list[str] = Field(default_factory=list)

    required_screenshots: list[ReferenceImage] = Field(default_factory=list)
    screenshot_match_threshold: int = Field(
        default=10,
        ge=0,
        description="Max pHash Hamming distance (lower = stricter; 10 ≈ same scene)",
    )

    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    min_scene_count: int | None = None
    max_scene_count: int | None = None
    expected_resolution: tuple[int, int] | None = None

    min_sharpness_score: float | None = None
    min_overall_quality: float | None = None

    fail_on_any: bool = Field(
        default=True,
        description="If True, a single failed criterion fails the whole verification",
    )


class VerificationFailure(BaseModel):
    """A single failed verification criterion."""

    criterion: str
    expected: Any | None = None
    actual: Any | None = None
    evidence: str | None = None


class ScreenshotMatch(BaseModel):
    """pHash match from a reference image to a frame."""

    name: str
    matched_frame: str | None = None
    timestamp: float | None = None
    distance: int


class VerificationResult(BaseModel):
    """Outcome of running a VerificationCriteria against a job."""

    passed: bool
    criteria_total: int
    criteria_passed: int
    failures: list[VerificationFailure] = Field(default_factory=list)
    # Criteria that could NOT be evaluated because an upstream dependency was
    # unavailable (e.g. the transcription backend was down). These are excluded
    # from ``criteria_total`` and never counted as failures — an infra outage
    # must not produce a content-"failed" verdict.
    indeterminate: list[str] = Field(default_factory=list)
    transcript_similarity: float | None = None
    screenshot_matches: list[ScreenshotMatch] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Incoming request for video analysis."""

    # NOTE: ``video_path`` (a caller-supplied local filesystem path) was REMOVED.
    # It was resolved with an existence check only — no storage root, no tenant
    # scope — so any request could make the pipeline read (and surface frames /
    # OCR / transcript of) ANY file readable by the pod. It was also
    # architecturally dead: the analyzer runs multi-replica, so a path from a
    # remote caller points at a filesystem the analyzing pod does not have. Use
    # ``video_url`` (SSRF-gated downloader) or ``POST /api/analyze/upload``
    # (tenant-scoped object-store key). Requests still carrying the field are
    # rejected by :meth:`_reject_video_path` rather than silently ignored.
    video_url: str | None = Field(
        default=None, description="HTTP(S) URL or YouTube/Vimeo share link"
    )
    language: str | None = Field(default=None, description="Transcription language hint")
    vision_provider: Literal["openai", "anthropic", "google", "fake", "stub"] | None = None

    generate_walkthrough: bool = True
    generate_ui_inventory: bool = True
    generate_frame_index: bool = True
    extract_frame_descriptions: bool = True
    extract_ocr: bool = True
    compute_quality_metrics: bool = True

    frame_interval_seconds: float | None = Field(
        default=None, description="Override default frame interval"
    )
    max_frames: int | None = Field(default=None, description="Override default max frames")
    max_vision_cost_usd: float | None = Field(
        default=None, description="Hard cap on vision stage spend"
    )

    verification: VerificationCriteria | None = None

    # Declared by the only construction path that is legitimately sourceless:
    # the multipart upload route (and the pipeline's re-validation of an upload
    # job's persisted config). Passed as pydantic validation context.
    UPLOAD_SOURCE_CONTEXT: ClassVar[dict[str, str]] = {"source": "upload"}

    SOURCE_REQUIRED: ClassVar[str] = (
        "AnalysisRequest requires 'video_url' (or use POST /api/analyze/upload)"
    )

    VIDEO_PATH_REJECTED: ClassVar[str] = (
        "'video_path' is no longer accepted: a caller-supplied filesystem path "
        "let the pipeline read any file on the pod and never worked across "
        "replicas. Use 'video_url' or POST /api/analyze/upload."
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_video_path(cls, data: object) -> object:
        """Fail closed on the removed ``video_path`` input.

        Rejecting beats silently ignoring: an ignored field would make the
        service answer 202 and then analyze *nothing the caller asked for*.
        This also runs when the pipeline re-validates a persisted
        ``request_config``, so a job row written before the removal fails
        terminally instead of replaying the arbitrary-file read.
        """
        if isinstance(data, dict) and cast("dict[str, object]", data).get("video_path") is not None:
            raise ValueError(cls.VIDEO_PATH_REJECTED)
        return data

    @model_validator(mode="after")
    def _require_a_source(self, info: ValidationInfo) -> AnalysisRequest:
        """At-least-one-source invariant, on the MODEL rather than one route.

        Previously only ``POST /api/analyze`` checked this, so any other
        construction of the model produced a sourceless request that only failed
        much later, inside the pipeline, as ``_resolve_source``'s "Neither ...
        provided".

        The multipart route is legitimately sourceless — its bytes arrive
        out-of-band and are addressed by the job row's ``upload_filename``, not
        by any field on this model — so it declares itself via validation
        context (:meth:`for_upload` / ``UPLOAD_SOURCE_CONTEXT``). The pipeline
        must pass the same context when re-validating an upload job's persisted
        ``request_config``, or every upload job would fail at execution time.
        """
        context = info.context if isinstance(info.context, dict) else {}
        if context.get("source") == "upload":
            return self
        if not self.video_url:
            raise ValueError(self.SOURCE_REQUIRED)
        return self

    @classmethod
    def for_upload(cls, payload: object = None) -> AnalysisRequest:
        """Validate a multipart-upload config, whose source is the uploaded file.

        The only sanctioned way to build a sourceless ``AnalysisRequest``.
        """
        return cls.model_validate(
            payload if payload is not None else {}, context=cls.UPLOAD_SOURCE_CONTEXT
        )


class AnalysisResponse(BaseModel):
    """Initial response from POST /api/analyze — returns 202."""

    job_id: str
    status: str
    stage: str
    progress: float


class JobStatusResponse(BaseModel):
    """Poll response for GET /api/analyze/{job_id}."""

    job_id: str
    status: str
    stage: str
    progress: float
    error_message: str | None = None
    artifacts: dict[str, str] | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    vision_cost_usd: float = 0.0
    verification_passed: bool | None = None


class MotionSampleResponse(BaseModel):
    """One point on the motion curve: how much changed *arriving at* ``t``.

    Mirrors the pipeline's ``MotionSample`` dataclass: ``motion`` is the same
    0..1 frame-delta the aggregate ``motion_score`` averages, kept per-sample;
    ``t`` is the timestamp of the LATER frame of the pair.
    """

    t: float = Field(..., description="Timestamp (seconds) the change lands at")
    motion: float = Field(..., description="0..1 frame-delta at this sample")


class VisualSampleResponse(BaseModel):
    """Per-frame visual metrics — mirrors the pipeline's ``VisualSample``.

    The aggregates on ``QualityMetricsResponse`` are whole-video means, so they
    are identical for every shot; these are the same quantities kept per frame,
    which is what makes shot-to-shot continuity and colour drift measurable.
    """

    t: float = Field(..., description="This frame's timestamp (seconds)")
    index: int = Field(..., description="Source ExtractedFrame.index (gaps = undecodable frames)")
    sharpness: float = Field(..., description="Laplacian variance at this frame")
    brightness: float = Field(..., description="0..1 mean luma at this frame")
    dominant_rgb: list[int] = Field(
        default_factory=list, description="[r, g, b] 0..255 dominant colour of this frame"
    )


class QualityMetricsResponse(BaseModel):
    """Aggregate visual quality metrics — mirrors the pipeline's ``QualityMetrics``.

    Stored verbatim (``asdict``) in the job row's JSONB column and read back
    here, so field names/shape must track the dataclass exactly.
    """

    mean_sharpness: float
    mean_brightness: float
    motion_score: float
    dominant_colors: list[str]
    overall_score: float
    # Added in PR #69 — JSONB rows persisted before it LACK the key entirely,
    # so this MUST default (a required field would 500 old jobs' /result).
    motion_timeline: list[MotionSampleResponse] = Field(default_factory=list)
    # Same back-compat contract as motion_timeline: rows persisted before this
    # field existed lack the key entirely, and an in-flight job's result was
    # written by the previous image — so it MUST default, never be required.
    visual_timeline: list[VisualSampleResponse] = Field(default_factory=list)
    # SAMPLE SIZE. ``mean_sharpness`` alone cannot distinguish a 20-frame mean
    # from a 1-frame one, and the orchestrator's blank hard-fail gate reads this
    # body — so without these a mostly-corrupt clip whose one decodable frame was
    # sharp is indistinguishable on the wire from a clean clip.
    #
    # ``None``, not 0, for a row an older analyzer persisted: absent is UNMEASURED,
    # and a consumer must be able to tell that apart from a measured zero. (Rows
    # written by this version always carry integers; the dataclass defaults them
    # to 0 because a dataclass cannot express "key missing".)
    frames_planned: int | None = Field(
        default=None, description="Frames the extractor scheduled (None on a pre-#122 job)"
    )
    frames_analyzed: int | None = Field(
        default=None, description="Frames the aggregates rest on (None on a pre-#122 job)"
    )


class AudioBeatGridRequest(BaseModel):
    """Request body for POST /api/analyze/audio.

    Only an ``audio_url`` is accepted, on purpose. This service just REMOVED
    every caller-supplied filesystem path from its request surface
    (``AnalysisRequest.video_path``, ``ReferenceImage.path`` — see the module
    docstring), and a new endpoint must not reintroduce the same class of
    unconfined read through a different field name. ``s3://`` is refused for the
    same reason it is refused elsewhere: the SSRF guard only reasons about
    HTTP(S), so a non-HTTP scheme would reach the fetcher un-judged.

    The *scheme* check lives here so an obviously-wrong body fails validation
    (422) without work. The real judgement — DNS resolution, private/reserved
    address ranges, internal-suffix blocklist, and the pin that closes the
    rebind window — is ``url_security.validate_public_url``, invoked by
    ``VideoDownloader.download`` exactly as it is for ``video_url``. It is NOT
    called here: it does blocking DNS, and pydantic validation runs on the event
    loop.
    """

    audio_url: str = Field(..., description="HTTP(S) URL of the audio (or video) file to analyse")
    meter: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Beats per bar used for downbeat phase inference (4 = 4/4)",
    )

    @field_validator("audio_url")
    @classmethod
    def _must_be_http_url(cls, value: str) -> str:
        url = value.strip()
        if not url:
            raise ValueError("audio_url must not be empty")
        scheme = urlparse(url).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                "audio_url must be an http(s) URL. Object-store references "
                "(s3://), file:// URLs and filesystem paths are not accepted: "
                "they bypass the SSRF guard and never resolve across replicas."
            )
        return url


class AudioBeatGrid(BaseModel):
    """Beat grid returned by POST /api/analyze/audio. Times are seconds from 0."""

    duration_s: float = Field(..., description="Decoded audio duration in seconds")
    bpm: float = Field(..., description="Estimated tempo in beats per minute")
    bpm_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How metronomically the beats fit the reported tempo: mean over "
            "inter-beat intervals of max(0, 1 - |interval - 60/bpm|/(60/bpm)/0.25). "
            "1.0 = a click track; 0.7-0.9 = typical live/recorded music; 0.0 = no "
            "single tempo fits (or fewer than 3 beats). It does NOT claim the "
            "tempo is correct — a half/double-tempo lock scores just as high."
        ),
    )
    beats: list[float] = Field(default_factory=list, description="Beat times in seconds")
    downbeats: list[float] = Field(
        default_factory=list,
        description=(
            "Inferred bar starts: every ``meter``-th beat, at the phase whose "
            "beats carry the most onset energy. Inferred, not DBN-tracked."
        ),
    )
    onsets: list[float] = Field(default_factory=list, description="Detected onset times in seconds")
    meter: int = Field(..., description="Beats per bar used for the downbeat inference")
    grid_source: str = Field(..., description="Algorithm that produced this grid")


class JobResultResponse(JobStatusResponse):
    """Full result returned by GET /api/analyze/{job_id}/result."""

    transcript_data: dict[str, Any] | None = None
    scene_boundaries: list[float] | None = None
    frame_analyses: list[dict[str, Any]] | None = None
    frame_analyses_partial: bool = Field(
        default=False,
        description=(
            "True when ``frame_analyses`` holds only the frames scored before the job "
            "stopped — i.e. the job is still running, or reached a terminal state other "
            "than 'completed' (failure, cost cap, wall-clock deadline). The pipeline "
            "checkpoints frame results after every vision batch, so a caller that times "
            "out mid-run gets the frames it already paid for rather than null. Use this "
            "to tell 'nothing to report' (frame_analyses null/empty, partial False) from "
            "'I was cut off after N frames' (N entries, partial True)."
        ),
    )
    quality_metrics: QualityMetricsResponse | None = None
    verification: VerificationResult | None = None
