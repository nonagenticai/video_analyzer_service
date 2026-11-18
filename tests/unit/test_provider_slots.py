"""Run provider slots (w2v_common 0.113): ASR + VLM route through the run's pinned slot.

``W2V_PROVIDER_SLOTS=off`` (default) must be byte-identical to before — no resolve call,
today's transcription_worker / inference-router wire. ``enforce`` with a TENANT slot
must hit the tenant's own provider with its Bearer secret ONLY (no licence, no identity
headers). A job re-claimed by another pod (#131) must resolve the SAME run's slot: the
re-run binds the job's saved tenant + X-W2V-Job-Id, so the resolve carries the same
``run_id`` the submitting pod sent.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER, reset_cell_state
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_correlation_context,
    set_identity_context,
    set_w2v_job_context,
)
from w2v_common.clients import vllm_direct
from w2v_common.clients.wm_client import WMClient
from w2v_common.discovery import service_url
from w2v_common.providers import _http as provider_http
from w2v_common.providers import clear_provider_slot_cache
from w2v_common.providers import slots as slots_mod
from w2v_common.providers.slots import RESOLVE_PATH, set_provider_slot_client
from w2v_common.tenant import TenantSource

from src.services.job_identity import capture_dispatch_identity
from src.services.transcription_client import TranscriptionClient, TranscriptionClientError
from src.services.vision.base import FrameInput
from src.services.vision.wm_provider import WMVisionProvider
from tests.unit.test_durable_reclaim import _pod
from tests.unit.test_pipeline_resilience import _RecordingVisionProvider

TENANT = uuid.UUID("0199a1b2-0000-7000-8000-00000000000a")
RUN = "0199a1b2-1111-7000-8000-000000000001"
CONN = "0199a1b2-2222-7000-8000-000000000001"
SA_TOKEN = "sa-analyzer-licence"  # pragma: allowlist secret
IR_TOKEN = "ir-licence-abc"  # pragma: allowlist secret
TENANT_SECRET = "sk-tenant-xyz789"  # pragma: allowlist secret
PUBLIC_IP = "93.184.216.34"
WORKER = "http://worker"

RESOLVE_URL = f"{service_url('orchestrator')}{RESOLVE_PATH}"
IR_URL = f"{service_url('inference_router')}/v1/chat/completions"

#: Headers that must never reach a tenant provider.
IDENTITY_HEADERS = (
    "x-tenant-id",
    "x-effective-tenant-id",
    "x-license-id",
    "x-subject-id",
    "x-actor-id",
    "x-project-id",
    "x-w2v-project-id",
    "x-w2v-job-id",
    "x-correlation-id",
    "x-job-id",
    CELL_TOKEN_HEADER.lower(),
)

TRANSCRIPT = {"text": "hello world", "segments": [], "backend": "whisper", "model": "m"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    jwk = json.loads(OKPAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": "va-1", "iss": "video-analyzer"})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [jwk]}))
    for name in ("LLM_BACKEND", "LLM_IR_BASE_URL", "LLM_IR_TOKEN_FILE", "W2V_LOCAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(slots_mod.SWITCH_ENV, raising=False)
    monkeypatch.setenv("LLM_IR_TOKEN", IR_TOKEN)
    monkeypatch.setenv("LLM_IR_USAGE_REPORT_ENABLED", "false")
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SA_TOKEN)
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "va-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", "video-analyzer")
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "strato-w2v")
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))

    async def _no_sleep(_seconds: float) -> None:
        return None

    async def _public(_host: str, _port: int) -> list[str]:
        return [PUBLIC_IP]

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)
    monkeypatch.setattr(slots_mod, "_sleep", _no_sleep)
    monkeypatch.setattr(provider_http, "_sleep", _no_sleep)
    monkeypatch.setattr(provider_http, "_resolve", _public)
    reset_cell_state()
    clear_identity_context()
    clear_provider_slot_cache()
    set_provider_slot_client(None)
    yield
    reset_cell_state()
    clear_identity_context()
    clear_provider_slot_cache()
    set_provider_slot_client(None)


@pytest.fixture
def enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(slots_mod.SWITCH_ENV, "enforce")


def _bind(run: str = RUN) -> None:
    set_identity_context(TENANT, "lic-1", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN)
    set_w2v_job_context(run)
    set_correlation_context(run, "analyze")


def _tenant_slot(slot: str, *, kind: str, base_url: str) -> dict[str, Any]:
    return {
        "source": "tenant",
        "slot": slot,
        "kind": kind,
        "base_url": base_url,
        "model": "tenant-model",
        "options": {},
        "secret": TENANT_SECRET,
        "ref": {"connection_id": CONN, "version": 3},
    }


def _deployment_slot(slot: str) -> dict[str, Any]:
    return {
        "source": "deployment",
        "slot": slot,
        "kind": "platform",
        "base_url": None,
        "model": None,
        "options": {},
        "secret": None,
        "ref": None,
    }


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        },
    )


def _audio(tmp_path: Path) -> Path:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return audio


def _frame(tmp_path: Path) -> FrameInput:
    path = tmp_path / "f.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0jpeg")
    return FrameInput(index=0, timestamp=0.0, path=path)


def _assert_tenant_only(request: httpx.Request) -> None:
    assert request.headers["authorization"] == f"Bearer {TENANT_SECRET}"
    for name in IDENTITY_HEADERS:
        assert name not in request.headers, name
    assert SA_TOKEN not in str(request.headers)
    assert IR_TOKEN not in str(request.headers)


# ── off ⇒ identical ─────────────────────────────────────────────────────────


@respx.mock(assert_all_called=False)
async def test_off_asr_is_the_platform_call_and_never_resolves(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(500))
    worker = respx_mock.post(f"{WORKER}/api/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    _bind()
    client = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
    try:
        out = await client.transcribe_file(audio_path=_audio(tmp_path), language="en")
    finally:
        await client.aclose()
    assert out == TRANSCRIPT
    assert resolve.call_count == 0
    req = worker.calls[0].request
    assert req.headers["x-tenant-id"] == f"Bearer {SA_TOKEN}"
    assert b'name="language"' in req.content and b'name="file"' in req.content


@respx.mock(assert_all_called=False)
async def test_off_vision_takes_inference_router_and_never_resolves(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(500))
    ir = respx_mock.post(IR_URL).mock(return_value=_completion('{"description": "a button"}'))
    _bind()
    wm = WMClient()
    try:
        out = await WMVisionProvider(wm_client=wm).analyze_frames([_frame(tmp_path)])
    finally:
        await wm.aclose()
    assert out[0].description == "a button"
    assert resolve.call_count == 0
    assert ir.call_count == 1


# ── enforce + tenant ────────────────────────────────────────────────────────


@respx.mock(assert_all_called=False)
async def test_enforce_tenant_asr_goes_to_the_tenant_url_with_bearer_only(
    respx_mock: respx.MockRouter, tmp_path: Path, enforce: None
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_tenant_slot(
                "asr", kind="nonagentic_endpoint", base_url="https://asr.tenant.example"
            ),
        )
    )
    worker = respx_mock.post(f"{WORKER}/api/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    tenant = respx_mock.post(f"https://{PUBLIC_IP}/api/asr/transcribe").mock(
        return_value=httpx.Response(200, json={**TRANSCRIPT, "text": "tenant words"})
    )
    _bind()
    client = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
    try:
        out = await client.transcribe_file(audio_path=_audio(tmp_path), language="en")
    finally:
        await client.aclose()
    assert out["text"] == "tenant words"
    assert worker.call_count == 0
    assert json.loads(resolve.calls[0].request.content) == {"slot": "asr", "run_id": RUN}
    req = tenant.calls[0].request
    _assert_tenant_only(req)
    assert req.headers["host"] == "asr.tenant.example"
    assert b"WAVEfmt" in req.content, "the audio itself must reach the tenant provider"


@respx.mock(assert_all_called=False)
async def test_enforce_deployment_asr_is_the_platform_call(
    respx_mock: respx.MockRouter, tmp_path: Path, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_deployment_slot("asr"))
    )
    worker = respx_mock.post(f"{WORKER}/api/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    _bind()
    client = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
    try:
        out = await client.transcribe_file(audio_path=_audio(tmp_path), language=None)
    finally:
        await client.aclose()
    assert out == TRANSCRIPT
    assert worker.calls[0].request.headers["x-tenant-id"] == f"Bearer {SA_TOKEN}"


@respx.mock(assert_all_called=False)
async def test_enforce_byo_required_is_a_transcription_gap_not_a_platform_fallback(
    respx_mock: respx.MockRouter, tmp_path: Path, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(
            403, json={"detail": {"code": "byo_provider_required", "message": "bring one"}}
        )
    )
    worker = respx_mock.post(f"{WORKER}/api/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    _bind()
    client = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
    try:
        with pytest.raises(TranscriptionClientError, match="run ASR provider unavailable"):
            await client.transcribe_file(audio_path=_audio(tmp_path), language="en")
    finally:
        await client.aclose()
    assert worker.call_count == 0


@respx.mock(assert_all_called=False)
async def test_enforce_tenant_vlm_frames_and_llm_synthesis(
    respx_mock: respx.MockRouter, tmp_path: Path, enforce: None
) -> None:
    def by_slot(request: httpx.Request) -> httpx.Response:
        slot = json.loads(request.content)["slot"]
        return httpx.Response(
            200,
            json=_tenant_slot(
                slot, kind="openai-compatible", base_url="https://vlm.tenant.example/v1"
            ),
        )

    resolve = respx_mock.post(RESOLVE_URL).mock(side_effect=by_slot)
    ir = respx_mock.post(IR_URL).mock(return_value=_completion("{}"))
    tenant = respx_mock.post(f"https://{PUBLIC_IP}/v1/chat/completions").mock(
        side_effect=[
            _completion('{"description": "tenant saw a button"}'),
            _completion('{"walkthrough": "tenant walkthrough"}'),
        ]
    )
    _bind()
    wm = WMClient()
    provider = WMVisionProvider(wm_client=wm)
    try:
        frames = await provider.analyze_frames([_frame(tmp_path)])
        synthesis = await provider.synthesize(frame_results=frames, transcript_text=None)
    finally:
        await wm.aclose()
    assert frames[0].description == "tenant saw a button"
    assert synthesis.walkthrough == "tenant walkthrough"
    assert ir.call_count == 0
    sent = [json.loads(c.request.content) for c in resolve.calls]
    assert sent == [{"slot": "vlm", "run_id": RUN}, {"slot": "llm", "run_id": RUN}]
    frame_req = tenant.calls[0].request
    _assert_tenant_only(frame_req)
    parts = json.loads(frame_req.content)["messages"][-1]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


# ── #131 re-claim: the re-run resolves the SAME run's slot ──────────────────


@respx.mock(assert_all_called=False)
async def test_reclaimed_job_resolves_the_same_run_slot_as_the_submitter(
    respx_mock: respx.MockRouter, tmp_path: Path, enforce: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_tenant_slot(
                "asr", kind="nonagentic_endpoint", base_url="https://asr.tenant.example"
            ),
        )
    )
    respx_mock.post(f"https://{PUBLIC_IP}/api/asr/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    audio = _audio(tmp_path)

    # Pod A, inside the submitting request: snapshot stored on the job row.
    _bind()
    snapshot = capture_dispatch_identity()
    client_a = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
    try:
        await client_a.transcribe_file(audio_path=audio, language="en")
    finally:
        await client_a.aclose()

    # Pod B: a different process (empty cache, no request in scope) re-claims it.
    clear_identity_context()
    clear_provider_slot_cache()
    pod = _pod(tmp_path, "pod-b", _RecordingVisionProvider())

    async def _run_job(job_id: str, tenant_id: str) -> None:
        client_b = TranscriptionClient(base_url=WORKER, license_token=SA_TOKEN)
        try:
            await client_b.transcribe_file(audio_path=audio, language="en")
        finally:
            await client_b.aclose()

    monkeypatch.setattr(pod, "_run_job", _run_job)
    await pod.resume("job-x", str(TENANT), json.loads(json.dumps(snapshot)))
    await pod.wait_for("job-x")

    bodies = [json.loads(c.request.content) for c in resolve.calls]
    assert bodies == [{"slot": "asr", "run_id": RUN}, {"slot": "asr", "run_id": RUN}]
    # Both resolves are on the cell lane for the job's own tenant.
    for call in resolve.calls:
        assert CELL_TOKEN_HEADER in call.request.headers
        assert "x-tenant-id" not in call.request.headers
