"""Verification stage: check criteria against pipeline output."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from src.api.schemas import (
    ReferenceImage,
    ScreenshotMatch,
    VerificationCriteria,
    VerificationFailure,
    VerificationResult,
)
from src.domain.analysis import (
    ExtractedFrame,
    FrameAnalysis,
    QualityMetrics,
    VideoMetadata,
)
from src.services.frame_extractor import MIN_FRAMES_FOR_QUALITY
from src.services.image_matcher import ImageMatcher

logger = logging.getLogger(__name__)

# The raw transcription payload (whisper JSON) as it crosses the service seam.
type TranscriptData = dict[str, Any]

# Marker key used inside the ``transcript_data`` dict to signal that the
# transcription stage could not run (ASR/infra failure) — as opposed to a
# legitimately empty transcript (``""``, e.g. a silent video). This lets the
# verifier mark transcript-dependent criteria as *indeterminate* instead of
# emitting false content-"failed" verdicts when the transcription backend is
# merely down. Distinct from both ``None`` and ``""``.
TRANSCRIPT_UNAVAILABLE_KEY = "_transcript_unavailable"


def transcript_unavailable_marker(reason: str) -> TranscriptData:
    """Build the sentinel transcript payload for an infra/ASR failure."""
    return {TRANSCRIPT_UNAVAILABLE_KEY: True, "reason": reason}


def is_transcript_unavailable(transcript_data: TranscriptData | None) -> bool:
    """True when transcription was UNAVAILABLE due to infra, not empty content."""
    return (
        isinstance(transcript_data, dict)
        and transcript_data.get(TRANSCRIPT_UNAVAILABLE_KEY) is True
    )


def extract_transcript_text(transcript_data: TranscriptData | None) -> str | None:
    """Pull the transcript text, or ``None`` when absent/unavailable.

    Never returns the literal string ``"None"``: a dict missing the ``text``
    key (or the infra-unavailable sentinel) yields ``None``, not ``str(None)``.
    """
    if not isinstance(transcript_data, dict):
        return None
    text = transcript_data.get("text")
    return str(text) if text else None


def _normalize(text: str | None) -> str:
    return (text or "").lower().strip()


def check_phrases(
    *, transcript_text: str | None, criteria: VerificationCriteria
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    haystack = _normalize(transcript_text)
    for needle in criteria.required_phrases:
        if _normalize(needle) and _normalize(needle) not in haystack:
            failures.append(
                VerificationFailure(
                    criterion="required_phrases",
                    expected=needle,
                    actual=None,
                    evidence="phrase not found in transcript",
                )
            )
    for needle in criteria.forbidden_phrases:
        if _normalize(needle) and _normalize(needle) in haystack:
            failures.append(
                VerificationFailure(
                    criterion="forbidden_phrases",
                    expected=f"absent: {needle}",
                    actual=needle,
                    evidence="forbidden phrase present in transcript",
                )
            )
    return failures


def check_transcript_similarity(
    *, transcript_text: str | None, criteria: VerificationCriteria
) -> tuple[float | None, VerificationFailure | None]:
    if criteria.expected_transcript is None:
        return None, None
    actual = _normalize(transcript_text)
    expected = _normalize(criteria.expected_transcript)
    similarity = difflib.SequenceMatcher(None, expected, actual).ratio()
    if similarity < criteria.min_transcript_similarity:
        return similarity, VerificationFailure(
            criterion="min_transcript_similarity",
            expected=criteria.min_transcript_similarity,
            actual=similarity,
            evidence="transcript similarity below threshold",
        )
    return similarity, None


def check_required_ocr(
    *, frame_analyses: list[FrameAnalysis], criteria: VerificationCriteria
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    haystacks = [_normalize(fa.ocr_text) for fa in frame_analyses if fa.ocr_text]
    for needle in criteria.required_ocr_text:
        normalised = _normalize(needle)
        if normalised and not any(normalised in hay for hay in haystacks):
            failures.append(
                VerificationFailure(
                    criterion="required_ocr_text",
                    expected=needle,
                    actual=None,
                    evidence="OCR text not found in any frame",
                )
            )
    return failures


def check_structural(
    *, metadata: VideoMetadata, criteria: VerificationCriteria, scene_count: int
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    if (
        criteria.min_duration_seconds is not None
        and metadata.duration_seconds < criteria.min_duration_seconds
    ):
        failures.append(
            VerificationFailure(
                criterion="min_duration_seconds",
                expected=criteria.min_duration_seconds,
                actual=metadata.duration_seconds,
            )
        )
    if (
        criteria.max_duration_seconds is not None
        and metadata.duration_seconds > criteria.max_duration_seconds
    ):
        failures.append(
            VerificationFailure(
                criterion="max_duration_seconds",
                expected=criteria.max_duration_seconds,
                actual=metadata.duration_seconds,
            )
        )
    if criteria.min_scene_count is not None and scene_count < criteria.min_scene_count:
        failures.append(
            VerificationFailure(
                criterion="min_scene_count",
                expected=criteria.min_scene_count,
                actual=scene_count,
            )
        )
    if criteria.max_scene_count is not None and scene_count > criteria.max_scene_count:
        failures.append(
            VerificationFailure(
                criterion="max_scene_count",
                expected=criteria.max_scene_count,
                actual=scene_count,
            )
        )
    if criteria.expected_resolution is not None:
        expected_w, expected_h = criteria.expected_resolution
        if metadata.width != expected_w or metadata.height != expected_h:
            failures.append(
                VerificationFailure(
                    criterion="expected_resolution",
                    expected=f"{expected_w}x{expected_h}",
                    actual=f"{metadata.width}x{metadata.height}",
                )
            )
    return failures


def quality_indeterminate_criteria(criteria: VerificationCriteria) -> list[str]:
    """Requested quality criterion names that are UNMEASURABLE right now.

    This covers only the ``compute_quality_metrics=False`` path, where no
    ``QualityMetrics`` was ever produced. It must NOT be reached when a
    ``QualityMetrics`` with ``mean_sharpness=0.0`` exists (failed frame
    extraction) — that is a legitimate measurement and must FAIL the gate via
    :func:`check_quality`, not sit here as indeterminate.
    """
    names: list[str] = []
    if criteria.min_sharpness_score is not None:
        names.append("min_sharpness_score")
    if criteria.min_overall_quality is not None:
        names.append("min_overall_quality")
    return names


#: Criterion name for the sample-size guard below. Not a criterion the CALLER
#: can define — it is a property of the evidence, reported through the same
#: channel so a consumer that already reads ``failures[].criterion`` (the
#: orchestrator's ``demo_video_record`` does) sees it with no client change.
QUALITY_SAMPLE_SIZE_CRITERION = "quality_sample_size"


def required_quality_sample(quality: QualityMetrics) -> int:
    """How many frames a quality verdict on ``quality`` must rest on.

    ``min(MIN_FRAMES_FOR_QUALITY, frames_planned)`` — and the ``min`` is the
    whole point of this function.

    ``MIN_FRAMES_FOR_QUALITY`` alone is the WRONG floor: a legitimately short or
    ``max_frames``-capped clip can PLAN fewer than five samples by design (see
    ``compute_effective_interval`` — the floor narrows the interval, it does not
    manufacture frames beyond the clip), and failing those would fail every short
    clip in the fleet for doing exactly what it was told. ``frames_planned``
    alone is equally wrong in the other direction: it would fail a 20-frame plan
    that delivered 19, where the mean is in no danger at all.

    Taking the smaller asks only "did we get the evidence we set out to get, up
    to the point where more stops mattering". A clip that planned 3 and
    delivered 3 passes; a clip that planned 20 and delivered 2 does not; a clip
    that planned 20 and delivered 6 does, because six samples already support a
    mean and the loss above the floor is noise, not a verdict problem.

    ``frames_planned == 0`` yields 0, i.e. the guard is a no-op. That covers the
    two cases where there is nothing to compare against: a JSONB row written by
    an analyzer that predates these fields, and a clip for which no timestamp
    was ever schedulable.
    """
    return min(MIN_FRAMES_FOR_QUALITY, quality.frames_planned)


def _quality_threshold_can_fail(criteria: VerificationCriteria) -> bool:
    """Whether a quality criterion is set to a value that can actually fail.

    A threshold of ``0.0`` is not "unset" — it is a threshold nothing can fall
    below, i.e. the operator turning the quality verdict OFF by configuration.
    ``check_quality`` already treats it that way implicitly (``x < 0.0`` is never
    true), and the orchestrator says so explicitly: ``fail_thr > 0 and ...``,
    with the comment "an explicit operator choice must not nag on every run".
    The sample-size guard protects a verdict; where there is no verdict to
    protect it stays quiet rather than inventing failures on a disabled gate.
    """
    return (criteria.min_sharpness_score or 0.0) > 0.0 or (
        criteria.min_overall_quality or 0.0
    ) > 0.0


def check_quality(
    *,
    quality: QualityMetrics | None,
    criteria: VerificationCriteria,
) -> list[VerificationFailure]:
    """Score the quality criteria, and refuse a verdict the sample cannot carry.

    The scalar checks below compare a MEAN whose sample size used to be invisible
    on both the dataclass and the wire, so a mostly-corrupt clip whose one
    surviving frame happened to be sharp scored identically to a clean twenty-frame
    clip — and passed. That is the failure this service exists to catch.

    The sample-size guard fires only in the PARTIAL case. Zero analyzed frames is
    deliberately left alone: it already fails closed here on ``mean_sharpness=0.0``
    (and ``quality_indeterminate_criteria`` documents that it must fail rather
    than sit as indeterminate), so re-reporting it under a second criterion would
    change an established verdict shape to say nothing new.
    """
    failures: list[VerificationFailure] = []
    if quality is None:
        return failures
    required = required_quality_sample(quality)
    if (
        _quality_threshold_can_fail(criteria)
        and 0 < quality.frames_analyzed < required  # 0 == the all-or-nothing case, handled below
    ):
        failures.append(
            VerificationFailure(
                criterion=QUALITY_SAMPLE_SIZE_CRITERION,
                expected=required,
                actual=quality.frames_analyzed,
                evidence=(
                    f"quality metrics rest on {quality.frames_analyzed} of "
                    f"{quality.frames_planned} planned frames; a verdict on this clip "
                    f"needs at least {required}"
                ),
            )
        )
    if (
        criteria.min_sharpness_score is not None
        and quality.mean_sharpness < criteria.min_sharpness_score
    ):
        failures.append(
            VerificationFailure(
                criterion="min_sharpness_score",
                expected=criteria.min_sharpness_score,
                actual=quality.mean_sharpness,
            )
        )
    if (
        criteria.min_overall_quality is not None
        and quality.overall_score < criteria.min_overall_quality
    ):
        failures.append(
            VerificationFailure(
                criterion="min_overall_quality",
                expected=criteria.min_overall_quality,
                actual=quality.overall_score,
            )
        )
    return failures


def check_screenshots(
    *,
    criteria: VerificationCriteria,
    frames: list[ExtractedFrame],
    matcher: ImageMatcher,
    reference_root: Path | None = None,
) -> tuple[list[ScreenshotMatch], list[VerificationFailure]]:
    """Match each reference image against the extracted frames.

    ``reference_root`` is this tenant's upload root
    (``<storage.upload_dir>/<tenant_id>``); every ``reference_key`` resolves
    under it or is not read at all. It defaults to ``None`` — i.e. *nothing*
    resolves — so a caller that forgets to pass it fails closed rather than
    falling back to the pod filesystem.
    """
    matches: list[ScreenshotMatch] = []
    failures: list[VerificationFailure] = []
    for ref in criteria.required_screenshots:
        ref_path = _resolve_reference_path(ref, reference_root=reference_root)
        if ref_path is None or not ref_path.exists():
            failures.append(
                VerificationFailure(
                    criterion="required_screenshots",
                    expected=ref.name,
                    actual=None,
                    evidence="reference image is missing or outside the tenant reference root",
                )
            )
            continue

        time_window: tuple[float, float] | None = None
        if ref.required_at_timestamp is not None:
            tol = ref.timestamp_tolerance_seconds
            time_window = (
                max(0.0, ref.required_at_timestamp - tol),
                ref.required_at_timestamp + tol,
            )
        result = matcher.find_match(
            name=ref.name,
            reference_path=ref_path,
            frames=frames,
            time_window=time_window,
        )
        matches.append(
            ScreenshotMatch(
                name=result.name,
                matched_frame=str(result.matched_frame) if result.matched_frame else None,
                timestamp=result.timestamp,
                distance=result.distance,
            )
        )
        if result.distance > criteria.screenshot_match_threshold:
            failures.append(
                VerificationFailure(
                    criterion="required_screenshots",
                    expected=ref.name,
                    actual=f"distance={result.distance}",
                    evidence="no frame within Hamming distance threshold",
                )
            )
    return matches, failures


def _resolve_reference_path(ref: ReferenceImage, *, reference_root: Path | None) -> Path | None:
    """Resolve ``ref.reference_key`` inside ``reference_root``, or return ``None``.

    Semantic half of the confinement (the schema does the syntactic half:
    no absolute keys, no ``..`` segments, no backslashes, no percent-escapes).

    Both sides are ``resolve()``d first, so a symlink planted inside the root is
    followed before the check rather than after it, and containment is then
    decided by ``relative_to`` — i.e. **by path segment**. A substring/prefix
    test would accept the sibling ``…/uploads/tenantevil`` for the tenant
    ``…/uploads/tenant``; ``relative_to`` rejects it.

    ``url``-only references resolve to ``None`` here: remote fetching is not
    implemented, so they are reported missing rather than read off local disk.
    """
    if not ref.reference_key or reference_root is None:
        return None
    try:
        root = reference_root.resolve(strict=False)
        candidate = (root / ref.reference_key).resolve(strict=False)
        # Raises ValueError when `candidate` is outside `root`. The extra
        # equality test rejects the root directory itself, which is contained
        # but is not a readable reference image.
        candidate.relative_to(root)
        if candidate == root:
            return None
    except OSError, ValueError:
        logger.warning("Reference key %r does not resolve inside the tenant root", ref.name)
        return None
    return candidate


class Verifier:
    """Top-level entry point for the verification stage."""

    async def verify(
        self,
        *,
        criteria: VerificationCriteria,
        metadata: VideoMetadata,
        transcript_data: TranscriptData | None,
        frame_analyses: list[FrameAnalysis],
        frames: list[ExtractedFrame],
        scene_boundaries: list[float],
        quality_metrics: QualityMetrics | None,
        image_matcher: ImageMatcher | None,
        reference_root: Path | None = None,
    ) -> dict[str, Any]:
        """Run all checks and return the serialised VerificationResult dict.

        ``reference_root`` confines ``required_screenshots`` reads to the
        requesting tenant's upload root; omitting it resolves no reference image
        at all (fail closed).
        """
        transcript_text = extract_transcript_text(transcript_data)
        transcript_missing = is_transcript_unavailable(transcript_data)

        failures: list[VerificationFailure] = []
        indeterminate: list[str] = []
        similarity: float | None = None
        if transcript_missing:
            # ASR/infra was unavailable — the transcript-dependent criteria
            # CANNOT be evaluated. Record them as indeterminate instead of
            # emitting false content failures (a transient outage must not read
            # as a genuine content mismatch).
            indeterminate.extend("required_phrases" for _ in criteria.required_phrases)
            indeterminate.extend("forbidden_phrases" for _ in criteria.forbidden_phrases)
            if criteria.expected_transcript is not None:
                indeterminate.append("min_transcript_similarity")
        else:
            failures.extend(check_phrases(transcript_text=transcript_text, criteria=criteria))
            similarity, sim_failure = check_transcript_similarity(
                transcript_text=transcript_text, criteria=criteria
            )
            if sim_failure is not None:
                failures.append(sim_failure)
        failures.extend(check_required_ocr(frame_analyses=frame_analyses, criteria=criteria))
        failures.extend(
            check_structural(
                metadata=metadata,
                criteria=criteria,
                scene_count=len(scene_boundaries),
            )
        )
        if quality_metrics is None:
            # compute_quality_metrics=False (or the stage never ran): the
            # requested quality criteria were never MEASURED — record them
            # as indeterminate rather than silently letting them count as
            # passed (a blank/garbled render must not verify as good). This
            # mirrors the transcript-unavailable handling above. A quality
            # analysis that DID run and legitimately scored mean_sharpness=0.0
            # (failed extraction) still reaches check_quality below and FAILS
            # closed — only "never computed" lands here.
            indeterminate.extend(quality_indeterminate_criteria(criteria))
        else:
            failures.extend(check_quality(quality=quality_metrics, criteria=criteria))

        matcher = image_matcher or ImageMatcher()
        screenshot_matches, screenshot_failures = check_screenshots(
            criteria=criteria, frames=frames, matcher=matcher, reference_root=reference_root
        )
        failures.extend(screenshot_failures)

        # Indeterminate criteria (transcript checks skipped because ASR was
        # down; quality checks skipped because compute_quality_metrics=False)
        # are excluded from the total so an unavailable input cannot swing the
        # pass/fail ratio — they are neither passed nor failed, just unevaluated.
        # That exclusion is what makes "not measured" readable as distinct from
        # "measured and passed".
        defined_total = _count_criteria(criteria)
        criteria_total = max(defined_total - len(indeterminate), 0)
        criteria_passed = max(criteria_total - len(failures), 0)
        if criteria_total == 0:
            # EMPTY DENOMINATOR. Nothing was actually evaluated: either every
            # defined criterion is indeterminate (e.g. a transcript-only
            # criteria set while ASR was unreachable), or no criterion was
            # defined at all. Both are measurement gaps, and a measurement gap
            # is not evidence of correctness — reporting passed=True here is
            # the vacuous-gate bug this branch exists to close. Note the
            # lenient branch below used to pass it too, via ``0 <= 0 // 2``.
            # ``indeterminate`` carries the reason, so a caller can still tell
            # a gap apart from a real failure.
            passed = False
            if defined_total == 0:
                indeterminate.append("no_criteria_defined")
        else:
            # ``fail_on_any`` is about FAILURES: an indeterminate criterion is
            # not one. compute_quality_metrics=False and a transient ASR outage
            # are both legitimate, and failing those runs would be worse than
            # the bug. Such a criterion is instead removed from the denominator
            # and listed explicitly, so it can never be read as a pass.
            passed = (
                len(failures) == 0 if criteria.fail_on_any else len(failures) <= criteria_total // 2
            )

        # A PASS IS A CLAIM ABOUT EVERY DEFINED CRITERION, and an indeterminate
        # criterion is one nothing measured. Removing it from the denominator
        # (above) stops it swinging the RATIO; it does not license claiming the
        # run passed. The empty-denominator branch already refuses that when the
        # gap is total, which meant ``passed`` flipped only for a criteria set
        # that was ENTIRELY transcript- or quality-dependent — the minority
        # shape. Add any measurable criterion (``expected_resolution``, which
        # every real caller sends) and ``criteria_total`` was 1 again, so an
        # undecodable transcript reported ``passed=True`` with the gap visible
        # only in a field most callers never read. Assembled here, once, rather
        # than special-cased per input, so a future indeterminate source cannot
        # reintroduce it: whatever the reason, an unmeasured criterion is not a
        # pass. ``indeterminate`` and ``failures`` stay untouched, so a consumer
        # can still tell a GAP ("we do not know") from a FAILURE ("we measured,
        # and it is wrong") — the distinction #117/#118 exist to preserve.
        if indeterminate:
            passed = False

        result = VerificationResult(
            passed=passed,
            criteria_total=criteria_total,
            criteria_passed=criteria_passed,
            failures=failures,
            indeterminate=indeterminate,
            transcript_similarity=similarity,
            screenshot_matches=screenshot_matches,
        )
        return result.model_dump(mode="json")


def _count_criteria(criteria: VerificationCriteria) -> int:
    """Tally the number of independent checks the criteria define."""
    count = 0
    count += len(criteria.required_phrases)
    count += len(criteria.forbidden_phrases)
    if criteria.expected_transcript is not None:
        count += 1
    count += len(criteria.required_ocr_text)
    count += len(criteria.required_screenshots)
    if criteria.min_duration_seconds is not None:
        count += 1
    if criteria.max_duration_seconds is not None:
        count += 1
    if criteria.min_scene_count is not None:
        count += 1
    if criteria.max_scene_count is not None:
        count += 1
    if criteria.expected_resolution is not None:
        count += 1
    # Shared with ``quality_indeterminate_criteria`` rather than re-listing the
    # fields here. The original defect WAS that drift: this function counted
    # quality criteria that nothing ever reported as unevaluated, so they were
    # scored as passed. One list makes the two sides impossible to desync.
    count += len(quality_indeterminate_criteria(criteria))
    return count
