"""Verifier unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.api.schemas import (
    ReferenceImage,
    VerificationCriteria,
)
from src.domain.analysis import (
    ExtractedFrame,
    FrameAnalysis,
    QualityMetrics,
    VideoMetadata,
)
from src.services.frame_extractor import MIN_FRAMES_FOR_QUALITY
from src.services.image_matcher import ImageMatcher
from src.services.verifier import (
    Verifier,
    _count_criteria,
    check_phrases,
    check_quality,
    check_required_ocr,
    check_screenshots,
    check_structural,
    check_transcript_similarity,
    extract_transcript_text,
    is_transcript_unavailable,
    quality_indeterminate_criteria,
    transcript_unavailable_marker,
)


def _meta(**overrides: Any) -> VideoMetadata:
    base: dict[str, Any] = {
        "duration_seconds": 30.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "frame_count": 900,
        "has_audio": True,
    }
    base.update(overrides)
    return VideoMetadata(**base)


def test_check_phrases_required_missing() -> None:
    failures = check_phrases(
        transcript_text="welcome to the demo",
        criteria=VerificationCriteria(required_phrases=["welcome", "feature flag"]),
    )
    assert len(failures) == 1
    assert failures[0].expected == "feature flag"


def test_check_phrases_forbidden_present() -> None:
    failures = check_phrases(
        transcript_text="this contains the forbidden word secret",
        criteria=VerificationCriteria(forbidden_phrases=["secret"]),
    )
    assert len(failures) == 1
    assert failures[0].criterion == "forbidden_phrases"


def test_transcript_similarity_above_threshold() -> None:
    similarity, failure = check_transcript_similarity(
        transcript_text="hello world",
        criteria=VerificationCriteria(
            expected_transcript="hello world", min_transcript_similarity=0.9
        ),
    )
    assert similarity == 1.0
    assert failure is None


def test_transcript_similarity_below_threshold() -> None:
    similarity, failure = check_transcript_similarity(
        transcript_text="goodbye world",
        criteria=VerificationCriteria(
            expected_transcript="hello world",
            min_transcript_similarity=0.95,
        ),
    )
    assert failure is not None
    assert failure.criterion == "min_transcript_similarity"
    # `similarity < 0.95` was implied by `failure is not None` — the failure is
    # returned ONLY on that branch, so the old assertion restated the branch
    # condition instead of checking the value. Multiplying difflib's ratio by any
    # constant < 1 in check_transcript_similarity kept it green. Pin the real
    # ratio for "hello world" vs "goodbye world" instead.
    assert similarity == pytest.approx(0.5833333333333334)
    assert failure.actual == pytest.approx(0.5833333333333334)
    assert failure.expected == 0.95


def test_check_required_ocr_present() -> None:
    failures = check_required_ocr(
        frame_analyses=[
            FrameAnalysis(index=0, timestamp=0.0, path="/x", ocr_text="Login Page"),
            FrameAnalysis(index=1, timestamp=3.0, path="/y", ocr_text="Dashboard"),
        ],
        criteria=VerificationCriteria(required_ocr_text=["login", "dashboard"]),
    )
    assert failures == []


def test_check_required_ocr_missing() -> None:
    failures = check_required_ocr(
        frame_analyses=[FrameAnalysis(index=0, timestamp=0.0, path="/x", ocr_text="Hello")],
        criteria=VerificationCriteria(required_ocr_text=["Settings"]),
    )
    assert len(failures) == 1


def test_check_structural_duration_violations() -> None:
    failures = check_structural(
        metadata=_meta(duration_seconds=10.0),
        criteria=VerificationCriteria(min_duration_seconds=20.0, max_duration_seconds=30.0),
        scene_count=2,
    )
    assert any(f.criterion == "min_duration_seconds" for f in failures)


def test_check_structural_resolution_mismatch() -> None:
    failures = check_structural(
        metadata=_meta(width=640, height=480),
        criteria=VerificationCriteria(expected_resolution=(1920, 1080)),
        scene_count=1,
    )
    assert any(f.criterion == "expected_resolution" for f in failures)


def test_check_structural_scene_count_bounds() -> None:
    failures = check_structural(
        metadata=_meta(),
        criteria=VerificationCriteria(min_scene_count=5, max_scene_count=2),
        scene_count=1,
    )
    assert any(f.criterion == "min_scene_count" for f in failures)


def test_check_structural_max_scene_count() -> None:
    failures = check_structural(
        metadata=_meta(),
        criteria=VerificationCriteria(max_scene_count=2),
        scene_count=10,
    )
    assert any(f.criterion == "max_scene_count" for f in failures)


def test_check_quality_thresholds() -> None:
    quality = QualityMetrics(
        mean_sharpness=100.0,
        mean_brightness=0.5,
        motion_score=0.1,
        dominant_colors=[],
        overall_score=0.4,
    )
    failures = check_quality(
        quality=quality,
        criteria=VerificationCriteria(min_sharpness_score=200.0, min_overall_quality=0.7),
    )
    assert {f.criterion for f in failures} == {"min_sharpness_score", "min_overall_quality"}


def test_every_quality_criterion_counted_is_also_reportable_as_unmeasured() -> None:
    """REPLACES ``test_check_quality_no_metrics``, which PINNED the defect.

    That test asserted ``check_quality(quality=None, ...) == []`` — i.e. that an
    unmeasured quality criterion produces no failure and no other signal — while
    ``_count_criteria`` counted it anyway, so it was scored as PASSED. The bug
    was the DRIFT between the two lists, so the invariant to pin is that they
    cannot drift: every quality criterion ``_count_criteria`` counts must also
    be one ``quality_indeterminate_criteria`` can report as unevaluated.
    """
    baseline = _count_criteria(VerificationCriteria())
    cases = [
        VerificationCriteria(min_sharpness_score=10.0),
        VerificationCriteria(min_overall_quality=0.5),
        VerificationCriteria(min_sharpness_score=10.0, min_overall_quality=0.5),
    ]
    for criteria in cases:
        counted = _count_criteria(criteria) - baseline
        assert counted == len(quality_indeterminate_criteria(criteria)), criteria
    assert quality_indeterminate_criteria(VerificationCriteria()) == []


def test_check_quality_zero_sharpness_fails_closed() -> None:
    """Regression guard: a COMPUTED QualityMetrics with mean_sharpness=0.0

    (failed frame extraction / every frame failed to decode) must still FAIL
    the gate. This is a real measurement, not an unmeasured one — do not
    reroute it through the indeterminate path.
    """
    quality = QualityMetrics(
        mean_sharpness=0.0,
        mean_brightness=0.0,
        motion_score=0.0,
        dominant_colors=[],
        overall_score=0.0,
    )
    failures = check_quality(
        quality=quality,
        criteria=VerificationCriteria(min_sharpness_score=10.0),
    )
    assert {f.criterion for f in failures} == {"min_sharpness_score"}


def test_quality_indeterminate_criteria_lists_requested_names() -> None:
    assert quality_indeterminate_criteria(
        VerificationCriteria(min_sharpness_score=10.0, min_overall_quality=0.5)
    ) == ["min_sharpness_score", "min_overall_quality"]
    assert quality_indeterminate_criteria(VerificationCriteria()) == []


def _tenant_root(tmp_path: Path, tenant: str = "tenant") -> Path:
    root = tmp_path / "uploads" / tenant
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_check_screenshots_match(tmp_path: Path) -> None:
    root = _tenant_root(tmp_path)
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(root / "ref.png")
    frame_path = tmp_path / "frame_0.png"
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(frame_path)
    frames = [ExtractedFrame(index=0, timestamp=0.0, path=frame_path)]
    matches, failures = check_screenshots(
        criteria=VerificationCriteria(
            required_screenshots=[ReferenceImage(name="ref", reference_key="ref.png")]
        ),
        frames=frames,
        matcher=ImageMatcher(),
        reference_root=root,
    )
    assert matches[0].matched_frame == str(frame_path)
    assert failures == []


def test_check_screenshots_missing_reference(tmp_path: Path) -> None:
    matches, failures = check_screenshots(
        criteria=VerificationCriteria(
            required_screenshots=[ReferenceImage(name="missing", reference_key="nope.png")]
        ),
        frames=[],
        matcher=ImageMatcher(),
        reference_root=_tenant_root(tmp_path),
    )
    assert matches == []
    assert any(f.criterion == "required_screenshots" for f in failures)


def test_check_screenshots_without_root_reads_nothing(tmp_path: Path) -> None:
    """No ``reference_root`` must mean "resolve nothing", never "use the pod FS"."""
    stray = tmp_path / "secret.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(stray)
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(frame)
    matches, failures = check_screenshots(
        criteria=VerificationCriteria(
            required_screenshots=[ReferenceImage(name="ref", reference_key="secret.png")]
        ),
        frames=[ExtractedFrame(index=0, timestamp=0.0, path=frame)],
        matcher=ImageMatcher(),
    )
    assert matches == []
    assert any(f.criterion == "required_screenshots" for f in failures)


def test_check_screenshots_rejects_symlink_escape_to_prefix_sibling(tmp_path: Path) -> None:
    """A symlink inside the root must not smuggle a read out of the root.

    The target is the sibling ``…/uploads/tenantevil`` — a directory whose name
    merely PREFIXES the tenant's, the classic string-containment bypass. It
    holds a byte-identical image, so if confinement leaked the pHash would match
    and the check would pass: a pass here is the oracle firing.
    """
    root = _tenant_root(tmp_path)
    evil = tmp_path / "uploads" / "tenantevil"
    evil.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(evil / "secret.png")
    (root / "link").symlink_to(evil, target_is_directory=True)

    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(frame)
    matches, failures = check_screenshots(
        criteria=VerificationCriteria(
            required_screenshots=[ReferenceImage(name="ref", reference_key="link/secret.png")]
        ),
        frames=[ExtractedFrame(index=0, timestamp=0.0, path=frame)],
        matcher=ImageMatcher(),
        reference_root=root,
    )
    assert matches == []
    assert any(f.criterion == "required_screenshots" for f in failures)


def test_count_criteria_zero_for_empty() -> None:
    assert _count_criteria(VerificationCriteria()) == 0


def test_count_criteria_sum() -> None:
    criteria = VerificationCriteria(
        required_phrases=["a", "b"],
        forbidden_phrases=["c"],
        expected_transcript="x",
        required_ocr_text=["y"],
        # `required_screenshots` was absent here, and `test_verifier_passes_when_all_satisfied`
        # only asserted `criteria_passed == criteria_total` (a tautology whenever there
        # are no failures). So `_count_criteria`'s `count += len(criteria.required_screenshots)`
        # was pinned by NO assertion in this file: a screenshot criterion could silently
        # stop counting toward the gate's denominator and every test stayed green.
        required_screenshots=[ReferenceImage(name="ref", reference_key="ref.png")],
        min_duration_seconds=1.0,
        max_duration_seconds=5.0,
        min_scene_count=1,
        max_scene_count=3,
        expected_resolution=(1920, 1080),
        min_sharpness_score=10.0,
        min_overall_quality=0.5,
    )
    assert _count_criteria(criteria) == 13


async def test_verifier_passes_when_all_satisfied(tmp_path: Path) -> None:
    root = _tenant_root(tmp_path)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(root / "ref.png")
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(frame_path)
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            required_phrases=["hello"],
            expected_transcript="hello world",
            min_transcript_similarity=0.5,
            required_screenshots=[ReferenceImage(name="ref", reference_key="ref.png")],
            min_duration_seconds=1.0,
        ),
        metadata=_meta(),
        transcript_data={"text": "hello world"},
        frame_analyses=[],
        frames=[ExtractedFrame(index=0, timestamp=0.0, path=frame_path)],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=ImageMatcher(),
        reference_root=root,
    )
    assert result["passed"] is True
    # `criteria_passed == criteria_total` alone is a TAUTOLOGY: with no failures,
    # `criteria_passed = max(criteria_total - 0, 0)` reduces to `criteria_total`
    # by construction, so the two sides can never differ. Pin the denominator to
    # the literal number of criteria configured above (required_phrases 1 +
    # expected_transcript 1 + required_screenshots 1 + min_duration_seconds 1).
    assert result["criteria_total"] == 4
    assert result["criteria_passed"] == 4


async def test_verifier_fails_with_failures(tmp_path: Path) -> None:
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            required_phrases=["never appears"],
            expected_transcript="hello world",
            min_transcript_similarity=0.99,
            min_duration_seconds=100.0,
        ),
        metadata=_meta(duration_seconds=10.0),
        transcript_data={"text": "goodbye"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["passed"] is False
    assert len(result["failures"]) >= 3


def test_is_transcript_unavailable_distinguishes_empty_from_infra() -> None:
    # Infra sentinel → unavailable; empty/present transcript → available.
    assert is_transcript_unavailable(transcript_unavailable_marker("asr down")) is True
    assert is_transcript_unavailable({"text": ""}) is False  # silent video, legit
    assert is_transcript_unavailable({"text": "hi"}) is False
    assert is_transcript_unavailable(None) is False


def test_extract_transcript_text_missing_key_is_none_not_literal_none() -> None:
    # Bug 2: a dict without "text" must yield None, never the string "None".
    assert extract_transcript_text({"segments": []}) is None
    assert extract_transcript_text(transcript_unavailable_marker("x")) is None
    assert extract_transcript_text({"text": "hello"}) == "hello"
    assert extract_transcript_text(None) is None


async def test_verifier_transcript_unavailable_is_indeterminate_not_failed() -> None:
    """Bug 1: an ASR infra outage must NOT read as a content-'failed' verdict.

    Transcript-dependent criteria are surfaced as indeterminate and excluded
    from the total, so a good video is not falsely FAILED when ASR is down.
    """
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            required_phrases=["welcome", "feature flag"],
            forbidden_phrases=["error"],
            expected_transcript="welcome to the feature flag demo",
            min_transcript_similarity=0.9,
            min_duration_seconds=1.0,  # non-transcript criterion that still passes
        ),
        metadata=_meta(),
        transcript_data=transcript_unavailable_marker("asr_backend unreachable"),
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    transcript_criteria = {"required_phrases", "forbidden_phrases", "min_transcript_similarity"}
    # No transcript-derived content failures were emitted.
    assert all(f["criterion"] not in transcript_criteria for f in result["failures"])
    # They are recorded as indeterminate instead.
    assert "required_phrases" in result["indeterminate"]
    assert "min_transcript_similarity" in result["indeterminate"]
    # 4 transcript criteria excluded → only the duration criterion counts, and it passes.
    assert result["criteria_total"] == 1
    assert result["criteria_passed"] == 1
    # ...but the RUN is not a pass. This assertion read ``is True`` from #117
    # until the empty-decode work, and that is the whole defect in miniature:
    # the outage is correctly kept out of ``failures`` (this test's actual
    # subject, asserted above and unchanged), yet ``passed`` still told every
    # caller the video was verified — while nothing had checked whether it says
    # "error". Not failed and passed are not the same verdict.
    assert result["passed"] is False
    assert result["indeterminate"]  # the reason is still readable


async def test_verifier_unmeasured_quality_is_indeterminate_not_passed() -> None:
    """compute_quality_metrics=False must NOT read as a passed quality gate.

    Requested-but-unmeasured quality criteria land in ``indeterminate`` and
    are excluded from ``criteria_total`` — mirroring the transcript-outage
    handling — so a blank/garbled render cannot verify as good merely
    because the quality stage never ran.
    """
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            min_sharpness_score=200.0,
            min_overall_quality=0.7,
            min_duration_seconds=1.0,  # non-quality criterion that still passes
            fail_on_any=True,
        ),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert all(
        f["criterion"] not in {"min_sharpness_score", "min_overall_quality"}
        for f in result["failures"]
    )
    assert "min_sharpness_score" in result["indeterminate"]
    assert "min_overall_quality" in result["indeterminate"]
    # 2 quality criteria excluded → only the duration criterion counts, and it passes.
    assert result["criteria_total"] == 1
    assert result["criteria_passed"] == 1
    # ...and the run is NOT a pass — which is what this test's own name has
    # claimed since #118 while its assertion said the opposite. Asking for
    # ``min_sharpness_score`` and switching the measurement off is a
    # self-contradictory request; answering it with ``passed=True`` is the
    # "blank render verifies as good" case the docstring above describes.
    assert result["passed"] is False


async def test_verifier_quality_present_and_failing_still_fails() -> None:
    verifier = Verifier()
    quality = QualityMetrics(
        mean_sharpness=1.0,
        mean_brightness=0.5,
        motion_score=0.1,
        dominant_colors=[],
        overall_score=0.1,
    )
    result = await verifier.verify(
        criteria=VerificationCriteria(min_sharpness_score=200.0, fail_on_any=True),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=quality,
        image_matcher=None,
    )
    assert result["passed"] is False
    assert any(f["criterion"] == "min_sharpness_score" for f in result["failures"])
    assert result["indeterminate"] == []


async def test_verifier_quality_present_and_passing_still_passes() -> None:
    verifier = Verifier()
    quality = QualityMetrics(
        mean_sharpness=500.0,
        mean_brightness=0.5,
        motion_score=0.1,
        dominant_colors=[],
        overall_score=0.9,
    )
    result = await verifier.verify(
        criteria=VerificationCriteria(min_sharpness_score=200.0, fail_on_any=True),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=quality,
        image_matcher=None,
    )
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["indeterminate"] == []


async def test_verifier_zero_sharpness_from_failed_extraction_fails_closed() -> None:
    """Regression guard for the path the fix must NOT touch.

    QualityAnalyzer.analyze returning mean_sharpness=0.0 (extraction/decode
    failure) is a real measurement and must still FAIL the gate closed.
    """
    verifier = Verifier()
    quality = QualityMetrics(
        mean_sharpness=0.0,
        mean_brightness=0.0,
        motion_score=0.0,
        dominant_colors=[],
        overall_score=0.0,
    )
    result = await verifier.verify(
        criteria=VerificationCriteria(min_sharpness_score=10.0, fail_on_any=True),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=quality,
        image_matcher=None,
    )
    assert result["passed"] is False
    assert any(f["criterion"] == "min_sharpness_score" for f in result["failures"])
    assert result["indeterminate"] == []


async def test_verifier_missing_text_key_does_not_phrase_match_literal_none() -> None:
    """Bug 2: a 'None' haystack must never satisfy/deny a phrase check.

    Pre-fix, str(None)=='None' meant required_phrases=['none'] spuriously matched.
    """
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(required_phrases=["none"]),
        metadata=_meta(),
        transcript_data={"segments": []},  # dict, but no "text" key
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    # "none" is absent from an empty (not literal "None") transcript → genuine miss.
    assert any(f["criterion"] == "required_phrases" for f in result["failures"])


async def test_verifier_lenient_mode_allows_some_failures() -> None:
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            required_phrases=["a", "b", "c", "d"],
            fail_on_any=False,
        ),
        metadata=_meta(),
        transcript_data={"text": "a b"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[],
        quality_metrics=None,
        image_matcher=None,
    )
    # 2 failures out of 4 criteria → still passes in lenient mode
    assert result["criteria_total"] == 4
    assert result["criteria_passed"] == 2
    assert result["passed"] is True


async def test_verifier_all_quality_criteria_unmeasured_is_never_a_pass() -> None:
    """EMPTY DENOMINATOR: every defined criterion was unevaluated.

    ``compute_quality_metrics=False`` with a quality-only criteria set leaves
    ``criteria_total=0``. Zero criteria evaluated is a measurement gap, not
    evidence of correctness, so it must never report ``passed=True``.
    """
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            min_sharpness_score=10.0, min_overall_quality=0.5, fail_on_any=True
        ),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["criteria_total"] == 0
    assert result["failures"] == []  # still not a content failure
    assert result["indeterminate"] == ["min_sharpness_score", "min_overall_quality"]
    assert result["passed"] is False


async def test_verifier_empty_denominator_from_transcript_outage_is_never_a_pass() -> None:
    """The verifier-side consequence of a transcription transport blip.

    With the transcript unavailable and only transcript criteria defined,
    ``criteria_total`` collapses to 0 — which used to report ``passed=True``.
    """
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(
            required_phrases=["welcome"],
            forbidden_phrases=["error"],
            fail_on_any=True,
        ),
        metadata=_meta(),
        transcript_data=transcript_unavailable_marker("connect error"),
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["criteria_total"] == 0
    assert result["passed"] is False


async def test_verifier_no_criteria_defined_is_not_a_pass() -> None:
    """Nothing to verify is not evidence of correctness; the reason is recorded."""
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["criteria_total"] == 0
    assert result["passed"] is False
    assert result["indeterminate"] == ["no_criteria_defined"]


async def test_verifier_lenient_mode_also_refuses_an_empty_denominator() -> None:
    """``fail_on_any=False`` used to pass an empty denominator via ``0 <= 0 // 2``."""
    verifier = Verifier()
    result = await verifier.verify(
        criteria=VerificationCriteria(min_overall_quality=0.5, fail_on_any=False),
        metadata=_meta(),
        transcript_data={"text": "hello"},
        frame_analyses=[],
        frames=[],
        scene_boundaries=[0.0],
        quality_metrics=None,
        image_matcher=None,
    )
    assert result["criteria_total"] == 0
    assert result["passed"] is False


# --- sample-size guard -------------------------------------------------------
#
# ``MIN_FRAMES_FOR_QUALITY`` constrained only the PLAN; nothing ever compared it
# against the frames that were actually delivered and decoded, so the whole-video
# means — which a downstream HARD-FAIL gate votes on — could rest on a single
# survivor of a botched extraction and be indistinguishable from a clean sample.


def _quality(*, sharpness: float, planned: int, analyzed: int) -> QualityMetrics:
    return QualityMetrics(
        mean_sharpness=sharpness,
        mean_brightness=0.5,
        motion_score=0.1,
        dominant_colors=[],
        overall_score=0.85,
        frames_planned=planned,
        frames_analyzed=analyzed,
    )


def test_planned_many_delivered_few_does_not_pass_on_a_sharp_survivor() -> None:
    """THE core case: 20 frames planned, 2 decoded, and those two happen to be sharp.

    Every scalar here clears its threshold — that is the point. Before the guard
    this returned no failures at all, so a mostly-corrupt clip verified as good
    and shipped.
    """
    failures = check_quality(
        quality=_quality(sharpness=900.0, planned=20, analyzed=2),
        criteria=VerificationCriteria(min_sharpness_score=200.0),
    )
    assert [f.criterion for f in failures] == ["quality_sample_size"]
    assert failures[0].expected == MIN_FRAMES_FOR_QUALITY
    assert failures[0].actual == 2
    assert "2 of 20 planned frames" in (failures[0].evidence or "")


def test_short_clip_that_delivered_its_whole_plan_still_passes() -> None:
    """The regression guard that matters most.

    A genuinely short (or ``max_frames``-capped) clip PLANS fewer than
    ``MIN_FRAMES_FOR_QUALITY`` samples by design — the floor narrows the
    interval, it cannot manufacture frames the clip does not contain. Gating on
    the bare constant would fail every short clip in the fleet; the requirement
    is ``min(floor, planned)``, so delivering the whole plan always passes.
    """
    for planned in range(1, MIN_FRAMES_FOR_QUALITY):
        failures = check_quality(
            quality=_quality(sharpness=900.0, planned=planned, analyzed=planned),
            criteria=VerificationCriteria(min_sharpness_score=200.0),
        )
        assert failures == [], f"planned={planned} delivered in full must not fail"


def test_short_clip_that_lost_frames_is_still_caught() -> None:
    """The other side of the same ``min``: a 3-frame plan delivering 1 is loss."""
    failures = check_quality(
        quality=_quality(sharpness=900.0, planned=3, analyzed=1),
        criteria=VerificationCriteria(min_sharpness_score=200.0),
    )
    assert [f.criterion for f in failures] == ["quality_sample_size"]
    assert failures[0].expected == 3


def test_loss_above_the_floor_is_not_a_verdict_problem() -> None:
    """20 planned, 6 delivered: six samples support a mean. Noise, not a gap."""
    assert (
        check_quality(
            quality=_quality(sharpness=900.0, planned=20, analyzed=6),
            criteria=VerificationCriteria(min_sharpness_score=200.0),
        )
        == []
    )


def test_zero_frames_still_fails_closed_on_sharpness_alone() -> None:
    """UNCHANGED. The all-or-nothing case was already fail-closed via
    ``mean_sharpness=0.0`` (``quality_indeterminate_criteria`` documents that it
    must FAIL rather than sit as indeterminate). The guard deliberately does not
    re-report it under a second criterion.
    """
    failures = check_quality(
        quality=_quality(sharpness=0.0, planned=20, analyzed=0),
        criteria=VerificationCriteria(min_sharpness_score=200.0),
    )
    assert [f.criterion for f in failures] == ["min_sharpness_score"]


def test_full_healthy_sample_is_unchanged() -> None:
    assert (
        check_quality(
            quality=_quality(sharpness=900.0, planned=20, analyzed=20),
            criteria=VerificationCriteria(min_sharpness_score=200.0),
        )
        == []
    )


def test_legacy_row_without_the_counts_is_not_retro_failed() -> None:
    """A JSONB row persisted before these fields deserialises as 0/0.

    ``frames_planned=0`` makes the requirement 0, so an old job's ``/result``
    keeps the verdict it always had instead of acquiring a new failure.
    """
    legacy = QualityMetrics(
        mean_sharpness=900.0,
        mean_brightness=0.5,
        motion_score=0.1,
        dominant_colors=[],
        overall_score=0.85,
    )
    assert legacy.frames_planned == 0
    assert (
        check_quality(quality=legacy, criteria=VerificationCriteria(min_sharpness_score=200.0))
        == []
    )


def test_disabled_quality_threshold_does_not_raise_a_sample_size_failure() -> None:
    """A ``0.0`` threshold is the operator turning the quality verdict OFF.

    ``check_quality``'s scalar comparisons already can never fail against it, and
    the orchestrator disables its own blank gate the same way (``fail_thr > 0``,
    "an explicit operator choice must not nag"). A guard protecting a verdict
    that does not exist would just nag.
    """
    assert (
        check_quality(
            quality=_quality(sharpness=900.0, planned=20, analyzed=1),
            criteria=VerificationCriteria(min_sharpness_score=0.0),
        )
        == []
    )
    assert (
        check_quality(
            quality=_quality(sharpness=900.0, planned=20, analyzed=1),
            criteria=VerificationCriteria(),
        )
        == []
    )


async def test_verify_reports_the_sample_size_failure_to_the_caller() -> None:
    """End of the chain: the guard must reach the wire, since the orchestrator's
    ``demo_video_record`` decides strict-eligibility from ``failures[].criterion``.
    """
    result = await Verifier().verify(
        criteria=VerificationCriteria(min_sharpness_score=200.0, fail_on_any=True),
        metadata=_meta(),
        transcript_data=None,
        frame_analyses=[],
        frames=[],
        scene_boundaries=[],
        quality_metrics=_quality(sharpness=900.0, planned=20, analyzed=2),
        image_matcher=None,
    )
    assert result["passed"] is False
    assert [f["criterion"] for f in result["failures"]] == ["quality_sample_size"]
