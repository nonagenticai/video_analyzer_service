"""TranscriptionClient integration with transcription_worker."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.services import transcription_client
from src.services.transcription_client import TranscriptionClient, TranscriptionClientError

# w2v-common 0.67.0: the outbound identity hook REFUSES to send an S2S request
# with no tenant bound to the execution context (MissingTenantError) — there is
# no default tenant any more. In production AuthMiddleware binds it per request;
# these tests must bind it the same way or they assert nothing but the refusal.
_TENANT = "66666666-6666-6666-6666-666666666666"


@pytest.fixture(autouse=True)
def _bound_identity() -> Iterator[None]:
    set_identity_context(_TENANT, "lic-1", "subj-1")
    try:
        yield
    finally:
        clear_identity_context()


def _wav_payload() -> bytes:
    return b"RIFF\x00\x00\x00\x00WAVEfmt "


@respx.mock
async def test_transcribe_file_happy_path(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "faster",
                "model": "tiny",
                "language": "en",
                "duration": 2.5,
                "text": "hello world",
                "segments": [],
            },
        )
    )
    client = TranscriptionClient(base_url="http://worker")
    result = await client.transcribe_file(audio_path=audio, language="en")
    assert result["text"] == "hello world"
    assert result["backend"] == "faster"


async def test_transcribe_missing_file(tmp_path: Path) -> None:
    client = TranscriptionClient(base_url="http://worker")
    with pytest.raises(TranscriptionClientError, match="not found"):
        await client.transcribe_file(audio_path=tmp_path / "missing.wav", language="en")


@respx.mock
async def test_transcribe_error_status(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(500, json={"detail": "broken"})
    )
    client = TranscriptionClient(base_url="http://worker")
    with pytest.raises(TranscriptionClientError, match="returned 500"):
        await client.transcribe_file(audio_path=audio, language=None)


@respx.mock
async def test_transcribe_transport_error_exhausts_the_budget(tmp_path: Path) -> None:
    """A connect error that outlasts the reconnect budget is still a failure."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    # Zero budget = the pre-fix behaviour, and keeps the test instant.
    client = TranscriptionClient(base_url="http://worker", reconnect_budget_seconds=0.0)
    with pytest.raises(TranscriptionClientError, match="unreachable after 1 attempt"):
        await client.transcribe_file(audio_path=audio, language=None)
    assert route.call_count == 1


@respx.mock
async def test_connect_error_inside_budget_succeeds_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEFECT 2: one ConnectError used to lose the whole transcript.

    A downstream roll (replicas: 1, maxSurge: 0) is a routine connectivity gap,
    not a verdict. Within the reconnect budget the call must SUCCEED.
    """
    slept: list[float] = []

    async def _no_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(transcription_client.asyncio, "sleep", _no_sleep)
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            httpx.Response(
                200,
                json={
                    "backend": "faster",
                    "model": "tiny",
                    "language": "en",
                    "duration": 1.0,
                    "text": "recovered",
                    "segments": [],
                },
            ),
        ]
    )
    client = TranscriptionClient(base_url="http://worker", reconnect_budget_seconds=300.0)
    try:
        result = await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert result["text"] == "recovered"
    assert route.call_count == 3
    assert slept == [1.0, 2.0]
    # The retried attempt must carry the FULL audio body, not an exhausted
    # (already-read) handle — otherwise the recovery uploads zero bytes.
    assert _wav_payload() in route.calls[-1].request.content


@respx.mock
async def test_server_error_is_not_retried(tmp_path: Path) -> None:
    """A 500 is an ANSWER from a reachable service: a real failure, sent once."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(500, json={"detail": "broken"})
    )
    client = TranscriptionClient(base_url="http://worker", reconnect_budget_seconds=300.0)
    with pytest.raises(TranscriptionClientError, match="returned 500"):
        await client.transcribe_file(audio_path=audio, language=None)
    assert route.call_count == 1


@respx.mock
async def test_client_error_is_not_retried(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(422, json={"detail": "bad audio"})
    )
    client = TranscriptionClient(base_url="http://worker", reconnect_budget_seconds=300.0)
    with pytest.raises(TranscriptionClientError, match="returned 422"):
        await client.transcribe_file(audio_path=audio, language=None)
    assert route.call_count == 1


@respx.mock
async def test_read_timeout_is_not_retried(tmp_path: Path) -> None:
    """The worker may be mid-transcription — replaying it would multiply load."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    client = TranscriptionClient(base_url="http://worker", reconnect_budget_seconds=300.0)
    with pytest.raises(TranscriptionClientError, match="call failed"):
        await client.transcribe_file(audio_path=audio, language=None)
    assert route.call_count == 1


@respx.mock
async def test_transcribe_reuses_pooled_client_across_calls(tmp_path: Path) -> None:
    """Two calls on one TranscriptionClient must reuse the same AsyncClient."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "faster",
                "model": "tiny",
                "language": "en",
                "duration": 0.0,
                "text": "",
                "segments": [],
            },
        )
    )
    client = TranscriptionClient(base_url="http://worker")
    try:
        await client.transcribe_file(audio_path=audio, language=None)
        first_pooled = client._http_client
        assert first_pooled is not None
        await client.transcribe_file(audio_path=audio, language=None)
        assert client._http_client is first_pooled
    finally:
        await client.aclose()
    assert client._http_client is None


@respx.mock
async def test_license_token_sent_as_bearer_tenant_header(tmp_path: Path) -> None:
    """When a license_token is set, the client authenticates to the gateway with
    X-Tenant-Id: Bearer <token> (the modality-gateway enforces license auth)."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://gw/api/transcribe").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "b",
                "model": "m",
                "language": "en",
                "duration": 0.0,
                "text": "",
                "segments": [],
            },
        )
    )
    client = TranscriptionClient(base_url="http://gw", license_token="sk-svc-token")
    try:
        await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert route.calls.last.request.headers["X-Tenant-Id"] == "Bearer sk-svc-token"


@respx.mock
async def test_no_bearer_header_when_license_token_absent(tmp_path: Path) -> None:
    """Without a token (e.g. direct asr.test path) no Bearer X-Tenant-Id is set."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    route = respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "b",
                "model": "m",
                "language": "en",
                "duration": 0.0,
                "text": "",
                "segments": [],
            },
        )
    )
    client = TranscriptionClient(base_url="http://worker")
    try:
        await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert not route.calls.last.request.headers.get("X-Tenant-Id", "").lower().startswith("bearer ")


async def test_aclose_is_noop_when_client_was_injected() -> None:
    """Injected clients are not owned by TranscriptionClient — aclose must
    not close them out from under the caller."""
    injected = httpx.AsyncClient()
    try:
        client = TranscriptionClient(base_url="http://worker", http_client=injected)
        await client.aclose()
        assert not injected.is_closed
    finally:
        await injected.aclose()


def test_no_speech_reason_mirror_matches_the_producer_enum() -> None:
    """The mirror must carry BOTH producer values, spelled exactly.

    A drifted spelling here silently reclassifies every undecodable transcript
    back to "clean", which is the defect this module guards.
    """
    from src.services.transcription_client import NoSpeechReason

    assert NoSpeechReason.SILENT_AUDIO.value == "silent_audio"
    assert NoSpeechReason.EMPTY_DECODE.value == "empty_decode"
    assert {m.value for m in NoSpeechReason} == {"silent_audio", "empty_decode"}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"text": "", "no_speech": True, "no_speech_reason": "empty_decode"}, True),
        ({"text": "", "no_speech": True, "no_speech_reason": "silent_audio"}, False),
        ({"text": "hello", "no_speech": False, "no_speech_reason": None}, False),
        # An older worker that predates the flag: absent is UNKNOWN, and an
        # unknown must not be promoted to a measurement gap.
        ({"text": ""}, False),
        # A value this consumer has never heard of must not be guessed at.
        ({"text": "", "no_speech": True, "no_speech_reason": "future_reason"}, False),
        (None, False),
    ],
)
def test_is_empty_decode_classifies_the_payload(payload: object, expected: bool) -> None:
    from src.services.transcription_client import is_empty_decode

    assert is_empty_decode(payload) is expected


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        # Valid JSON, wrong shape: a list/scalar/null carries no transcript, but
        # ``is_empty_decode`` answers False for it — "not a gap" — so it used to
        # reach the verifier as an empty haystack and pass every
        # ``forbidden_phrases`` criterion vacuously.
        ("[]", "not a JSON object"),
        ('"ok"', "not a JSON object"),
        ("null", "not a JSON object"),
        # A partial 200: the worker answered, but with no transcript in it.
        ('{"backend": "faster", "duration": 2.5}', "no 'text' field"),
        # ``text`` present but not a string — ``extract_transcript_text`` would
        # stringify it into a haystack that never existed.
        ('{"text": 42}', "not a string"),
        ('{"text": null}', "not a string"),
    ],
)
@respx.mock
async def test_structurally_invalid_200_is_refused_not_consumed(
    tmp_path: Path, body: str, fragment: str
) -> None:
    """A malformed 200 is a MEASUREMENT GAP, not a clean transcript.

    Raising ``TranscriptionClientError`` is deliberately NOT a hard failure:
    ``_run_transcription_stage`` already routes that exception to the
    ``transcript_unavailable_marker`` sentinel, so the criteria go
    indeterminate — the same destination as ``empty_decode``.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"})
    )
    client = TranscriptionClient(base_url="http://worker")
    try:
        with pytest.raises(TranscriptionClientError) as excinfo:
            await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert fragment in str(excinfo.value)


@respx.mock
async def test_unparseable_200_body_is_refused_not_consumed(tmp_path: Path) -> None:
    """A 200 that is not JSON at all (an HTML error page from a proxy, a
    truncated body) used to escape as a bare ``ValueError`` from ``resp.json()``
    — an exception the stage's handler does not catch, so it was neither a
    verdict nor a recorded gap."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(200, content=b"<html>502 upstream</html>")
    )
    client = TranscriptionClient(base_url="http://worker")
    try:
        with pytest.raises(TranscriptionClientError) as excinfo:
            await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert "unparseable" in str(excinfo.value)


@respx.mock
async def test_a_legacy_worker_payload_still_yields_a_real_verdict(tmp_path: Path) -> None:
    """ANTI-OVER-CORRECTION: a MISSING ``no_speech_reason`` is not a gap.

    An older ``transcription_worker_service`` predating the flag returns a
    perfectly good transcript without it. The structural check must accept that
    body unchanged — only ``text`` is load-bearing.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(_wav_payload())
    respx.post("http://worker/api/transcribe").mock(
        return_value=httpx.Response(
            200,
            json={"backend": "faster", "model": "tiny", "duration": 2.5, "text": "hello world"},
        )
    )
    client = TranscriptionClient(base_url="http://worker")
    try:
        result = await client.transcribe_file(audio_path=audio, language=None)
    finally:
        await client.aclose()
    assert result["text"] == "hello world"
    assert "no_speech_reason" not in result
