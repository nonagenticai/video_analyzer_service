"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AnalysisRequest,
    JobResultResponse,
    MotionSampleResponse,
    QualityMetricsResponse,
    ReferenceImage,
    VerificationCriteria,
    VerificationFailure,
    VerificationResult,
    VisualSampleResponse,
)


def test_reference_image_requires_source() -> None:
    with pytest.raises(ValidationError):
        ReferenceImage(name="foo")


def test_reference_image_accepts_relative_key() -> None:
    ref = ReferenceImage(name="intro", reference_key="refs/intro.png", required_at_timestamp=3.0)
    assert ref.reference_key == "refs/intro.png"
    assert ref.timestamp_tolerance_seconds == 2.0


def test_reference_image_has_no_path_field() -> None:
    """The unconfined pod-file input is gone from the model, not just guarded."""
    assert "path" not in ReferenceImage.model_fields


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "/app/data/uploads/../../etc/shadow",
        "../../etc/shadow",
    ],
)
def test_reference_image_rejects_path(hostile: str) -> None:
    """``path`` is rejected, not silently ignored (pydantic's extra='ignore').

    Reachable from the request body via ``verification.required_screenshots[]``,
    persisted into ``verification_criteria``, and re-validated at run time.
    """
    with pytest.raises(ValidationError, match="no longer accepted"):
        ReferenceImage.model_validate({"name": "x", "path": hostile})


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "../../etc/shadow",
        "refs/../../../etc/shadow",
        # URL-encoded and double-encoded separators / dot segments.
        "%2Fetc%2Fpasswd",
        "..%2f..%2fetc%2fshadow",
        "%252e%252e%252fetc%252fshadow",
        # Backslash separator, home expansion, and empty/whitespace keys.
        "..\\..\\etc\\shadow",
        "~/.ssh/id_rsa",
        "   ",
    ],
)
def test_reference_image_rejects_escaping_key(hostile: str) -> None:
    with pytest.raises(ValidationError):
        ReferenceImage(name="x", reference_key=hostile)


def test_reference_image_allows_dotted_name_that_is_not_a_dot_segment() -> None:
    """Containment is per SEGMENT: '..png' and 'a..b' are ordinary filenames."""
    assert ReferenceImage(name="x", reference_key="shots/a..b/..png").reference_key


def test_reference_image_accepts_url() -> None:
    ref = ReferenceImage(name="intro", url="https://example.com/intro.png")
    assert ref.url == "https://example.com/intro.png"


def test_verification_criteria_defaults() -> None:
    vc = VerificationCriteria()
    assert vc.required_phrases == []
    assert vc.fail_on_any is True
    assert vc.min_transcript_similarity == 0.85
    assert vc.screenshot_match_threshold == 10


def test_analysis_request_accepts_verification() -> None:
    req = AnalysisRequest(
        video_url="https://example.com/video.mp4",
        verification=VerificationCriteria(required_phrases=["login", "dashboard"]),
    )
    assert req.verification is not None
    assert req.verification.required_phrases == ["login", "dashboard"]


def test_analysis_request_has_no_video_path_field() -> None:
    """The arbitrary-pod-file input is gone from the model, not just guarded."""
    assert "video_path" not in AnalysisRequest.model_fields


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "/app/data/uploads/../../etc/shadow",
        # Sibling directory whose name merely PREFIXES a tenant dir — the classic
        # string-prefix containment bypass. Rejected here because the field is gone.
        "/app/data/uploads/tenant-aevil/x.mp4",
        "%2Fetc%2Fpasswd",
        "%252e%252e%252fetc%252fpasswd",
    ],
)
def test_analysis_request_rejects_video_path(hostile: str) -> None:
    with pytest.raises(ValidationError, match="no longer accepted"):
        AnalysisRequest.model_validate({"video_path": hostile})


def test_analysis_request_rejects_persisted_video_path() -> None:
    """A job row written BEFORE the removal must fail closed on replay.

    ``PipelineRunner._load_job_request`` re-validates ``request_config``; if that
    silently ignored ``video_path`` the queued arbitrary-file read would still
    execute after deploy.
    """
    with pytest.raises(ValidationError, match="no longer accepted"):
        AnalysisRequest.model_validate({"video_path": "/etc/passwd", "generate_walkthrough": True})


def test_analysis_request_accepts_url_and_upload_shapes() -> None:
    """The two supported sources validate — and ONLY via their sanctioned path.

    A URL request carries ``video_url``. An upload request carries no source
    field at all (the bytes are addressed by the job row's ``upload_filename``),
    so it must declare itself through ``for_upload``. A bare ``AnalysisRequest()``
    is now a validation error rather than a sourceless request that only blows
    up later inside the pipeline.
    """
    assert AnalysisRequest(video_url="https://example.com/v.mp4").video_url is not None
    assert AnalysisRequest.for_upload().video_url is None
    assert AnalysisRequest.for_upload({"extract_ocr": False}).extract_ocr is False


def test_analysis_request_requires_a_source() -> None:
    """The at-least-one-source invariant lives on the MODEL, not one route."""
    with pytest.raises(ValidationError, match="requires 'video_url'"):
        AnalysisRequest()
    with pytest.raises(ValidationError, match="requires 'video_url'"):
        AnalysisRequest.model_validate({"generate_walkthrough": True})


def test_verification_result_roundtrip() -> None:
    result = VerificationResult(
        passed=False,
        criteria_total=3,
        criteria_passed=2,
        failures=[VerificationFailure(criterion="required_phrases", expected=["login"], actual=[])],
    )
    dumped = result.model_dump(mode="json")
    restored = VerificationResult.model_validate(dumped)
    assert restored.passed is False
    assert restored.failures[0].criterion == "required_phrases"


def test_job_result_response_allows_nones() -> None:
    resp = JobResultResponse(
        job_id="abc",
        status="pending",
        stage="queued",
        progress=0.0,
    )
    assert resp.duration_seconds is None
    assert resp.verification is None
    assert resp.quality_metrics is None


def test_quality_metrics_validates_modern_jsonb_row() -> None:
    """A post-#69 JSONB row (all 6 keys incl. motion_timeline) validates typed."""
    resp = JobResultResponse(
        job_id="abc",
        status="completed",
        stage="artifact_gen",
        progress=1.0,
        quality_metrics=QualityMetricsResponse.model_validate(
            {
                "mean_sharpness": 150.2,
                "mean_brightness": 0.55,
                "motion_score": 0.4,
                "dominant_colors": ["#000000", "#ffffff"],
                "overall_score": 0.82,
                "motion_timeline": [{"t": 3.0, "motion": 0.2}, {"t": 6.0, "motion": 0.6}],
            }
        ),
    )
    qm = resp.quality_metrics
    assert qm is not None
    assert isinstance(qm, QualityMetricsResponse)
    assert qm.overall_score == 0.82
    assert [s.t for s in qm.motion_timeline] == [3.0, 6.0]
    assert qm.motion_timeline[1].motion == 0.6


def test_quality_metrics_validates_pre_pr69_jsonb_row() -> None:
    """Rows persisted before PR #69 lack the motion_timeline key entirely.

    The typed field must default to [] rather than 500 old jobs' /result.
    """
    resp = JobResultResponse(
        job_id="old-job",
        status="completed",
        stage="artifact_gen",
        progress=1.0,
        quality_metrics=QualityMetricsResponse.model_validate(
            {
                "mean_sharpness": 90.0,
                "mean_brightness": 0.4,
                "motion_score": 0.1,
                "dominant_colors": ["#abcdef"],
                "overall_score": 0.5,
            }
        ),
    )
    assert resp.quality_metrics is not None
    assert resp.quality_metrics.motion_timeline == []


def test_quality_metrics_rejects_missing_core_key() -> None:
    """The 5 original keys have been present since the initial schema — required."""
    with pytest.raises(ValidationError):
        QualityMetricsResponse.model_validate(
            {
                "mean_sharpness": 90.0,
                "mean_brightness": 0.4,
                "motion_score": 0.1,
                "dominant_colors": ["#abcdef"],
                # overall_score missing
            }
        )


def test_quality_metrics_json_roundtrip() -> None:
    """model_dump(mode='json') → model_validate is lossless (wire round-trip)."""
    original = QualityMetricsResponse(
        mean_sharpness=1.0,
        mean_brightness=2.0,
        motion_score=0.3,
        dominant_colors=["#123456"],
        overall_score=0.7,
        motion_timeline=[MotionSampleResponse(t=1.5, motion=0.9)],
    )
    restored = QualityMetricsResponse.model_validate(original.model_dump(mode="json"))
    assert restored == original


def test_quality_metrics_validates_visual_timeline() -> None:
    """The per-frame visual curve survives raw-JSONB → typed-schema validation."""
    qm = QualityMetricsResponse.model_validate(
        {
            "mean_sharpness": 300.0,
            "mean_brightness": 0.4,
            "motion_score": 0.2,
            "dominant_colors": ["#123456"],
            "overall_score": 0.6,
            "motion_timeline": [],
            "visual_timeline": [
                {
                    "t": 0.0,
                    "index": 0,
                    "sharpness": 900.0,
                    "brightness": 0.16,
                    "dominant_rgb": [17, 41, 118],
                },
                {
                    "t": 3.0,
                    "index": 2,
                    "sharpness": 12.5,
                    "brightness": 0.63,
                    "dominant_rgb": [231, 150, 39],
                },
            ],
        }
    )
    assert [s.index for s in qm.visual_timeline] == [0, 2]
    assert qm.visual_timeline[1].dominant_rgb == [231, 150, 39]
    assert qm.visual_timeline[0].sharpness == 900.0


def test_quality_metrics_row_without_visual_timeline_defaults_empty() -> None:
    """Rows persisted before this field existed (and in-flight jobs written by the
    previous image) lack the key entirely — a required field would 500 their /result."""
    qm = QualityMetricsResponse.model_validate(
        {
            "mean_sharpness": 1.0,
            "mean_brightness": 0.5,
            "motion_score": 0.1,
            "dominant_colors": [],
            "overall_score": 0.5,
            "motion_timeline": [{"t": 1.0, "motion": 0.2}],
        }
    )
    assert qm.visual_timeline == []


def test_visual_sample_json_roundtrip() -> None:
    original = QualityMetricsResponse(
        mean_sharpness=1.0,
        mean_brightness=2.0,
        motion_score=0.3,
        dominant_colors=["#123456"],
        overall_score=0.7,
        visual_timeline=[
            VisualSampleResponse(
                t=1.5, index=3, sharpness=42.0, brightness=0.25, dominant_rgb=[1, 2, 3]
            )
        ],
    )
    assert QualityMetricsResponse.model_validate(original.model_dump(mode="json")) == original
