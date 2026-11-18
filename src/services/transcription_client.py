"""HTTP client that calls transcription_worker_service /api/transcribe."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import httpx
from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.providers import (
    ASRRequest,
    ProviderSlotError,
    asr_transcribe,
    provider_slots_enforced,
)
from w2v_common.tenant import UnvouchedTenantError

logger = logging.getLogger(__name__)

#: The worker's transcript JSON, returned verbatim (named once; see the Any ratchet).
type TranscriptPayload = dict[str, Any]

# Transport errors that prove the request never reached the application: the
# connection could not be established, or it was closed/reset before any
# response byte was produced. Mirrors ``w2v_common.clients.base``'s
# connection-reset set — the dominant production cause is a downstream pod
# restart (every w2v service is ``replicas: 1`` with ``maxSurge: 0``, so a roll
# is a hard gap, not a drain) or a stale keepalive connection from this
# client's own long-lived pool.
#
# Deliberately EXCLUDED: ``ReadTimeout``/``WriteTimeout`` — the worker may be
# mid-transcription, and replaying a multi-minute job on a timeout would
# multiply load rather than recover. Also excluded: every HTTP status. A 4xx or
# 5xx is an ANSWER from a healthy service and is a real failure; retrying it
# into a long budget would hide it.
_CONNECTION_RESET_ERRORS: tuple[type[httpx.TransportError], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

#: Wall-clock budget for reconnecting across a downstream restart. A rolling
#: ``replicas: 1`` pod is unreachable for tens of seconds; the sibling fix in
#: w2v_orchestrator #944 settled on the same 300s for the same shape. Bounded
#: so a downstream that is genuinely gone fails the stage instead of parking
#: the job past its lease.
DEFAULT_RECONNECT_BUDGET_SECONDS = 300.0
_RECONNECT_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


class NoSpeechReason(StrEnum):
    """MIRROR of ``transcription_worker_service``'s ``NoSpeechReason``.

    The worker returns an empty transcript for two causes that are
    indistinguishable in the body alone, and this consumer must never conflate
    them — an empty transcript here becomes the factual claim "this video does
    not say the forbidden phrase".

    * ``SILENT_AUDIO`` — the decode SUCCEEDED over a measurable media duration
      and every segment was suppressed. This IS a real measurement: the phrase
      genuinely is absent, so the criterion must be EVALUATED, not skipped.
    * ``EMPTY_DECODE`` — the decode produced NOTHING usable. There is no
      evidence about the audio at all, so no claim may be made from it. It
      arrives as HTTP 200 with every required field present, shape-identical to
      a clean transcription apart from this flag.

    Keep the spellings byte-identical to the producer's enum; a drift silently
    reclassifies every undecodable transcript back to "clean".
    """

    SILENT_AUDIO = "silent_audio"
    EMPTY_DECODE = "empty_decode"


#: Response key carrying the ``NoSpeechReason``. ``transcribe_file`` returns the
#: worker's JSON verbatim (no pydantic response model), so nothing strips it —
#: the field was reaching this repo intact and was simply never read.
NO_SPEECH_REASON_KEY = "no_speech_reason"


def is_empty_decode(payload: object) -> bool:
    """True only when the worker explicitly reported ``EMPTY_DECODE``.

    Deliberately conservative in both directions:

    * ``SILENT_AUDIO`` is NOT a gap — treating it as one would turn every
      genuinely silent clip into an indeterminate verdict (a false alarm).
    * An ABSENT or UNRECOGNISED reason is NOT a gap either. An older worker
      that predates the flag, or a value this mirror has never heard of, is
      unknown; promoting an unknown to a gap would make every legacy
      transcript indeterminate. Only the value we can positively name counts.
    """
    if not isinstance(payload, dict):
        return False
    return payload.get(NO_SPEECH_REASON_KEY) == NoSpeechReason.EMPTY_DECODE.value


#: The ONE field a transcription response must carry for a verdict to rest on
#: it. Everything else the worker sends (``backend``, ``duration``, ``segments``,
#: ``no_speech_reason``…) is advisory here: ``extract_transcript_text`` reads
#: ``text`` and nothing else, so ``text`` is the whole measurement.
TRANSCRIPT_TEXT_KEY = "text"


def structural_defect(payload: object) -> str | None:
    """Why ``payload`` cannot carry a transcript verdict, or ``None`` if it can.

    ``transcribe_file`` returns the worker's JSON verbatim — there is no
    response model — so a malformed or partial 200 used to flow straight
    through. ``is_empty_decode`` answers ``False`` for such a body, i.e. "not a
    gap", and ``extract_transcript_text`` then yields ``None``: an empty
    haystack against which every ``forbidden_phrases`` criterion passes. That is
    the SAME vacuous pass the ``empty_decode`` route closes, arriving by a
    different cause — a fail-open on a measurement gap inside the code that
    exists to close one.

    Scoped deliberately to what is load-bearing:

    * A MISSING ``no_speech_reason`` is NOT a defect. An older
      ``transcription_worker_service`` predating the flag returns a perfectly
      good transcript without it, and rejecting that would turn every legacy
      response into a gap — the exact over-correction ``is_empty_decode``
      already refuses to make.
    * A missing/non-string ``text`` IS a defect: there is no transcript to
      measure, so any verdict drawn from it is invented.
    """
    if not isinstance(payload, dict):
        return f"response body is {type(payload).__name__}, not a JSON object"
    if TRANSCRIPT_TEXT_KEY not in payload:
        return f"response body has no {TRANSCRIPT_TEXT_KEY!r} field"
    if not isinstance(payload[TRANSCRIPT_TEXT_KEY], str):
        return (
            f"{TRANSCRIPT_TEXT_KEY!r} is "
            f"{type(payload[TRANSCRIPT_TEXT_KEY]).__name__}, not a string"
        )
    return None


class TranscriptionClientError(RuntimeError):
    """Raised when the transcription_worker call fails."""


class TranscriptionClient:
    """Thin async client around /api/transcribe.

    Holds a single long-lived ``httpx.AsyncClient`` per instance so the
    connection pool is shared across calls — every transcription request
    in a pipeline used to pay a fresh TCP/TLS handshake otherwise. Tests
    can still inject their own client via ``http_client=``; that client's
    lifecycle is owned by the caller. ``aclose()`` closes the owned client
    at shutdown.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 600.0,
        license_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        reconnect_budget_seconds: float = DEFAULT_RECONNECT_BUDGET_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.reconnect_budget_seconds = max(reconnect_budget_seconds, 0.0)
        # Service-account license token. The transcription backend behind the
        # modality-gateway enforces license auth (X-Tenant-Id: Bearer <token>);
        # the mac-host asr.test:9201 unit trusts bare in-cluster identity and
        # ignores it. Baking it as the client's default X-Tenant-Id header (Bearer
        # form) makes the gateway authenticate us as the w2v service account, while
        # identity_event_hook still stamps X-Effective-Tenant-Id = the real
        # customer tenant (preserving the pre-set Bearer), which the gateway
        # honours because this token's subject is a configured delegator.
        self._license_token = license_token or None
        # When a client is injected, we don't own its lifecycle.
        self._http_client: httpx.AsyncClient | None = http_client
        self._owned = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            headers = (
                {"X-Tenant-Id": f"Bearer {self._license_token}"} if self._license_token else None
            )
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                event_hooks={"request": [identity_event_hook]},
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the owned ``AsyncClient`` if we instantiated it."""
        if self._owned and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool = True,
        return_srt: bool = False,
    ) -> TranscriptPayload:
        """POST ``audio_path`` to /api/transcribe and return the JSON payload.

        Connection-class transport failures are retried within
        ``reconnect_budget_seconds``. Without that, a single ``ConnectError``
        — routine, because every w2v service is ``replicas: 1`` with
        ``maxSurge: 0``, so any downstream roll is a hard connectivity gap —
        made the whole transcript indeterminate, which drops every
        transcript criterion out of the verifier's denominator.
        """
        if not audio_path.exists():
            raise TranscriptionClientError(f"Audio file not found: {audio_path}")

        # Run provider slot ``asr`` (w2v_common 0.113). ``W2V_PROVIDER_SLOTS=off``
        # (default) awaits the platform call and nothing else — byte-identical to
        # before. ``enforce``: a TENANT slot transcribes on the tenant's own provider
        # (its Bearer only, no identity headers); a DEPLOYMENT slot is the platform
        # call unchanged. A re-claimed job resolves the SAME pinned slot because
        # ``bind_dispatch_identity`` restores the job's tenant + X-W2V-Job-Id.
        try:
            # ``request`` is read only on a TENANT slot; with the switch off the
            # (possibly large) audio file is never loaded into memory.
            request = ASRRequest(
                audio=audio_path.read_bytes() if provider_slots_enforced() else b"",
                filename=audio_path.name,
                content_type="audio/wav",
                language=language or None,
                word_timestamps=word_timestamps,
                return_srt=return_srt,
            )
            return await asr_transcribe(
                request,
                platform=lambda: self._transcribe_platform(
                    audio_path=audio_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    return_srt=return_srt,
                ),
                from_tenant=_validated_payload,
                timeout_s=self.timeout,
            )
        except (ProviderSlotError, UnvouchedTenantError) as exc:
            # The run's ASR provider could not be used (byo_required with no tenant
            # provider, tenant provider down/malformed, unvouched provenance). That is
            # a measurement gap: TranscriptionClientError routes it to the
            # INDETERMINATE sentinel, never to a vacuous "phrase absent" pass. The
            # tenant path NEVER falls back to the platform.
            raise TranscriptionClientError(f"run ASR provider unavailable: {exc}") from exc

    async def _transcribe_platform(
        self,
        *,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
        return_srt: bool,
    ) -> TranscriptPayload:
        """Today's platform call: POST to transcription_worker ``/api/transcribe``."""
        client = self._get_client()
        with audio_path.open("rb") as fh:
            data: dict[str, str] = {
                "word_timestamps": "true" if word_timestamps else "false",
                "return_srt": "true" if return_srt else "false",
            }
            if language:
                data["language"] = language
            deadline = time.monotonic() + self.reconnect_budget_seconds
            attempt = 0
            while True:
                # The handle is re-read on every attempt. httpx's multipart
                # FileField seeks the handle back to 0 before rendering, so a
                # retry re-uploads the whole file rather than the zero bytes
                # left after the previous attempt exhausted it — asserted by
                # ``test_connect_error_inside_budget_succeeds_on_retry``.
                files = {"file": (audio_path.name, fh, "audio/wav")}
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/transcribe",
                        files=files,
                        data=data,
                    )
                    break
                except _CONNECTION_RESET_ERRORS as exc:
                    remaining = deadline - time.monotonic()
                    delay = _RECONNECT_BACKOFF_SECONDS[
                        min(attempt, len(_RECONNECT_BACKOFF_SECONDS) - 1)
                    ]
                    if delay >= remaining:
                        raise TranscriptionClientError(
                            "transcription_worker unreachable after "
                            f"{attempt + 1} attempt(s) within a "
                            f"{self.reconnect_budget_seconds:.0f}s reconnect budget: {exc}"
                        ) from exc
                    logger.warning(
                        "transcription_worker connection failure (attempt %d, "
                        "retrying in %.1fs, %.1fs of budget left): %s",
                        attempt + 1,
                        delay,
                        remaining,
                        exc,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                except httpx.HTTPError as exc:
                    # Not connection-class (read/write timeout, protocol error
                    # on our side, …): surface immediately rather than replay a
                    # request the worker may already be processing.
                    raise TranscriptionClientError(
                        f"transcription_worker call failed: {exc}"
                    ) from exc

        if resp.status_code >= 400:
            raise TranscriptionClientError(
                f"transcription_worker returned {resp.status_code}: {resp.text[:500]}"
            )
        # A 200 is not by itself a measurement. Both refusals below raise
        # TranscriptionClientError DELIBERATELY rather than a new exception
        # type: ``_run_transcription_stage`` already catches exactly that and
        # routes it to ``transcript_unavailable_marker``, so an unusable body
        # lands as INDETERMINATE — the same destination as ``empty_decode``,
        # and not a hard job failure.
        try:
            payload: object = resp.json()
        except ValueError as exc:
            # Not JSON at all: an HTML error page injected by a proxy, or a
            # truncated body. This escaped as a bare ValueError, which the
            # stage's handler does not catch — neither a verdict nor a gap.
            raise TranscriptionClientError(
                f"transcription_worker returned an unparseable 200 body "
                f"({resp.headers.get('content-type', 'no content-type')}): {exc}"
            ) from exc
        return _validated_payload(payload, source="transcription_worker")


def _validated_payload(payload: object, source: str = "tenant ASR provider") -> TranscriptPayload:
    """``payload`` as a transcript dict, or ``TranscriptionClientError`` (→ INDETERMINATE)."""
    defect = structural_defect(payload)
    if defect is not None:
        raise TranscriptionClientError(
            f"{source} returned a structurally invalid 200 body: {defect}"
        )
    return cast("TranscriptPayload", payload)
