"""Beat-grid detection tests: synthesised audio only, never the network.

The click-track test is the load-bearing one — it is the only place the real
librosa pipeline runs end to end, so it also pays the numba JIT (~10s once per
process). Everything else drives the pure helpers directly.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import librosa
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.analysis import router as analysis_router
from src.api.schemas import AudioBeatGridRequest
from src.config import clear_settings_cache, get_settings
from src.services import beat_detector
from src.services.beat_detector import (
    HOP_LENGTH,
    SAMPLE_RATE,
    AudioTooLongError,
    BeatGridError,
    _beat_confidence,
    _drop_phantom_final_beat,
    _pick_downbeat_phase,
    _scalar_tempo,
    analyze_samples,
    warm_up,
)
from src.services.ffmpeg_utils import ffmpeg_available, run_checked
from tests.conftest import TENANT_A, attach_test_tenant


def _click_track(
    *, bpm: float = 120.0, seconds: float = 8.0, noise_floor: float = 0.0
) -> np.ndarray:
    """A metronome: a 20 ms decaying 2 kHz blip every ``60/bpm`` seconds.

    ``noise_floor`` adds deterministic gaussian dither at that amplitude. It
    defaults to 0 — mathematically exact digital silence between the clicks —
    which is fine for the tempo/phase/downbeat tests but is NOT a safe fixture
    for anything that depends on the onset envelope's *head*; see
    :func:`test_beats_are_not_trimmed_off_the_head_when_music_starts_at_zero`.
    """
    total = int(SAMPLE_RATE * seconds)
    track = np.zeros(total, dtype=np.float32)
    click_len = int(SAMPLE_RATE * 0.02)
    t = np.arange(click_len, dtype=np.float32) / SAMPLE_RATE
    click = (np.sin(2.0 * np.pi * 2000.0 * t) * np.exp(-t * 180.0)).astype(np.float32)
    period = round(SAMPLE_RATE * 60.0 / bpm)
    for start in range(0, total - click_len, period):
        track[start : start + click_len] += click
    if noise_floor:
        rng = np.random.default_rng(0)
        track = (track + rng.normal(0.0, noise_floor, total)).astype(np.float32)
    return track


def _fold_to_octave(bpm: float) -> float:
    """Fold a tempo into [90, 180) bpm.

    Tempo octave errors (locking onto half or double the true tempo) are a known
    and accepted property of every beat tracker, librosa's included. Folding
    means this test asserts "the pulse is 120-ish", not "librosa never makes an
    octave error" — the latter would be a test of librosa, and a flaky one.
    """
    if bpm <= 0:
        return 0.0
    while bpm < 90.0:
        bpm *= 2.0
    while bpm >= 180.0:
        bpm /= 2.0
    return bpm


# ---------------------------------------------------------------------------
# Real pipeline against synthesised audio
# ---------------------------------------------------------------------------


def test_click_track_at_120_bpm_is_detected() -> None:
    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0), meter=4)

    assert grid.duration_s == pytest.approx(8.0, abs=0.01)
    assert _fold_to_octave(grid.bpm) == pytest.approx(120.0, abs=2.0)
    # A metronome is metronomic: this is the top of the confidence scale.
    assert grid.bpm_confidence > 0.9
    assert len(grid.beats) >= 12  # ~16 clicks in 8s at 120 bpm
    assert grid.grid_source == beat_detector.GRID_SOURCE
    assert grid.meter == 4


def test_click_track_beats_land_on_the_clicks() -> None:
    """Beats must be evenly spaced at the detected period AND phase-locked to a click."""
    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0), meter=4)
    beats = np.asarray(grid.beats)
    period = 60.0 / grid.bpm

    # Regularity: every beat sits on the same period grid as the first one.
    offset = (beats - beats[0] + period / 2.0) % period - period / 2.0
    assert float(np.max(np.abs(offset))) < 0.06

    # Phase: the grid is anchored on an actual click (clicks are at multiples of
    # 0.5s), not floating half a beat away from one.
    first = grid.beats[0]
    assert abs(first - round(first / 0.5) * 0.5) < 0.06


def test_downbeats_are_every_meter_th_beat() -> None:
    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0), meter=4)
    assert grid.downbeats  # non-empty
    # Each downbeat is one of the beats, and they are 4 beats apart.
    assert set(grid.downbeats).issubset(set(grid.beats))
    if len(grid.downbeats) >= 2:
        gaps = np.diff(np.asarray(grid.downbeats))
        assert float(np.max(gaps)) == pytest.approx(4.0 * 60.0 / grid.bpm, abs=0.15)


def test_meter_one_makes_every_beat_a_downbeat() -> None:
    grid = analyze_samples(_click_track(bpm=120.0, seconds=4.0), meter=1)
    assert grid.downbeats == grid.beats


def test_analyze_samples_rejects_empty_audio() -> None:
    with pytest.raises(BeatGridError, match="empty"):
        analyze_samples(np.zeros(0, dtype=np.float32))


def test_analyze_samples_rejects_bad_meter() -> None:
    with pytest.raises(BeatGridError, match="meter"):
        analyze_samples(_click_track(seconds=1.0), meter=0)


def test_analyze_samples_accepts_a_readonly_frombuffer_view() -> None:
    """Regression: ``np.frombuffer`` over ffmpeg's stdout is READ-ONLY.

    ``decode_pcm`` returns exactly such a view, and librosa/numba write into
    their inputs, so ``analyze_samples`` must copy before analysing.
    """
    raw = _click_track(bpm=120.0, seconds=4.0).tobytes()
    view = np.frombuffer(raw, dtype="<f4")
    assert not view.flags.writeable
    grid = analyze_samples(view, meter=4)
    assert grid.bpm > 0
    # ...and the caller's buffer is untouched.
    assert not view.flags.writeable


def test_beats_are_not_trimmed_off_the_head_when_music_starts_at_zero() -> None:
    """Regression for the ``trim=True`` head-trim.

    librosa's default trimmed leading beats that onset strength *did* support:
    on curated beds it dropped 32 real beats (corporate_uplift's first beat moved
    from 0.093 s to 14.408 s), so every action_reel silently fell back to even
    spacing. Asserted here on a synthesised signal — the clicks start at t=0, so
    the first beat must too, and the count must cover the whole track.

    ⚠ ``noise_floor`` is LOAD-BEARING, not decoration. Until the librosa
    0.11.0 -> 1.0.0 bump this test used the digitally-silent click track and
    passed for the wrong reason. ``__trim_beats`` suppresses leading beats while
    ``localscore[n] <= threshold``, and with ``trim=False`` that threshold is
    exactly 0.0. On digital silence librosa 0.11.0's float64 ``localscore``
    held 3.07e-222 at frame 1 — a denormal 200+ orders of magnitude below any
    signal, but strictly ``> 0.0``, so the loop stopped and the head beat
    survived. librosa 1.0.0 computes ``localscore`` in float32, where that
    denormal flushes to exactly 0.0, the loop runs on to frame 44 and the head
    beat goes. Neither number is evidence about real audio: measured, an ffmpeg
    decode of real music never produces exact-zero head frames, and with any
    realistic room-tone floor (1e-3, ~-60 dBFS) the two librosa versions agree
    bit for bit — ``trim=False`` -> 17 beats from 0.0232 s, ``trim=True`` -> 15
    from 0.5224 s, on BOTH. So the floor here is what makes the fixture
    reproduce the defect the test is named for instead of a denormal accident.
    The assertion threshold is unchanged.
    """
    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0, noise_floor=1e-3), meter=4)
    assert grid.beats[0] < 0.25, f"leading beats were trimmed: first beat {grid.beats[0]}"
    # 16 clicks at 0.0, 0.5, ... 7.5 — the grid must reach the head, not start
    # a beat or more in.
    assert len(grid.beats) >= 15
    # ...and the head beat is a real click, not a stray sub-beat.
    assert abs(grid.beats[0] - round(grid.beats[0] / 0.5) * 0.5) < 0.06


def test_synthetic_metronome_phantom_final_beat_is_dropped() -> None:
    """The one genuine thing ``trim=True`` bought must survive its removal.

    With ``trim=False`` the DP tracker terminates at the last envelope frame and
    emits a beat at ~7.999 s that is ~489 ms from the last real click at 7.5 s.
    A hard cut there lands on silence at the end of the bed.
    """
    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0), meter=4)
    last = grid.beats[-1]
    # Clicks live at multiples of 0.5 s; the final beat must sit on one.
    assert abs(last - round(last / 0.5) * 0.5) < 0.06, f"phantom final beat at {last}"
    assert last < 7.9


def test_phantom_guard_drops_a_gapped_trailing_beat() -> None:
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.2])
    onsets = beats.copy()  # even a supported onset cannot save a 2.4x gap
    kept = _drop_phantom_final_beat(beats, onsets)
    assert kept.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_phantom_guard_drops_an_unsupported_trailing_beat() -> None:
    """In-period gap, but no onset near it — the metronome-boundary signature."""
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.476])
    onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert _drop_phantom_final_beat(beats, onsets).tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_phantom_guard_keeps_a_real_beat_in_a_sparse_onset_passage() -> None:
    """No nearby onset, but NOT past the last onset — that is not a phantom.

    A sustained pad or a held chord produces long onset-free stretches: 14.6% /
    13.8% / 3.8% of beats on the three curated beds have no onset within half a
    median interval, and on a truncated sub-clip the bare onset rule deletes a
    genuine final beat 25.3% / 17.5% / 4.1% of the time. The beat at 2.0 here has
    its nearest onset 1.0 s away (4x the 0.25 s window) yet audio events continue
    to 3.0 s, so it must be kept. The gap rule is not in play: 0.5 s is exactly
    the median.
    """
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    onsets = np.array([0.0, 0.5, 3.0])
    assert _drop_phantom_final_beat(beats, onsets).tolist() == beats.tolist()


def test_phantom_guard_never_fires_on_a_normal_grid() -> None:
    beats = np.arange(20, dtype=float) * 0.5
    onsets = beats + 0.01
    assert _drop_phantom_final_beat(beats, onsets).tolist() == beats.tolist()


def test_phantom_guard_drops_at_most_one_beat() -> None:
    """Two trailing gaps: only the final beat may go, never a cascade."""
    beats = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 5.0])
    kept = _drop_phantom_final_beat(beats, np.array([]))
    assert kept.tolist() == [0.0, 0.5, 1.0, 1.5, 3.0]


def test_phantom_guard_is_inert_without_enough_evidence() -> None:
    """Fewer than four beats, or no onsets at all, leaves the grid alone."""
    short = np.array([0.0, 0.5, 4.0])
    assert _drop_phantom_final_beat(short, np.array([0.0, 0.5])).tolist() == short.tolist()
    normal = np.array([0.0, 0.5, 1.0, 1.5])
    assert _drop_phantom_final_beat(normal, np.array([])).tolist() == normal.tolist()


def test_phantom_guard_median_ignores_the_phantom_interval() -> None:
    """A huge final gap must not inflate the yardstick and hide itself."""
    beats = np.array([0.0, 0.5, 1.0, 9.0])
    assert _drop_phantom_final_beat(beats, beats.copy()).tolist() == [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Downbeat phase selection
# ---------------------------------------------------------------------------


def _envelope_for(beats: np.ndarray, accented_phase: int, meter: int) -> np.ndarray:
    frames = librosa.time_to_frames(beats, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    envelope = np.zeros(int(np.max(frames)) + 4, dtype=float)
    envelope[frames] = 1.0
    envelope[frames[accented_phase::meter]] = 6.0
    return envelope


@pytest.mark.parametrize("accented_phase", [0, 1, 2, 3])
def test_downbeat_phase_follows_the_accent(accented_phase: int) -> None:
    beats = np.arange(16, dtype=float) * 0.5
    envelope = _envelope_for(beats, accented_phase, meter=4)
    assert _pick_downbeat_phase(beats, envelope, meter=4) == accented_phase


def test_downbeat_phase_defaults_to_zero_without_an_accent() -> None:
    """Flat material has no metrical evidence, so the first beat is the downbeat."""
    beats = np.arange(16, dtype=float) * 0.5
    frames = librosa.time_to_frames(beats, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    envelope = np.zeros(int(np.max(frames)) + 4, dtype=float)
    envelope[frames] = 1.0
    assert _pick_downbeat_phase(beats, envelope, meter=4) == 0


def test_downbeat_phase_handles_empty_beats_and_unit_meter() -> None:
    envelope = np.ones(10, dtype=float)
    assert _pick_downbeat_phase(np.zeros(0), envelope, meter=4) == 0
    assert _pick_downbeat_phase(np.arange(4, dtype=float), envelope, meter=1) == 0


def test_downbeat_phase_clips_beats_past_the_envelope() -> None:
    """A beat reported at/past the end of the envelope must not raise IndexError."""
    beats = np.array([0.0, 100.0])
    envelope = np.ones(5, dtype=float)
    assert _pick_downbeat_phase(beats, envelope, meter=2) in (0, 1)


# ---------------------------------------------------------------------------
# tempo normalisation — librosa's return type is not a float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        123.5,
        np.float64(123.5),
        np.array(123.5),  # 0-d
        np.array([123.5]),  # 1-element
        np.array([[123.5]]),  # nested, as multi-channel input produces
    ],
)
def test_scalar_tempo_normalises_every_shape(raw: object) -> None:
    assert _scalar_tempo(raw) == pytest.approx(123.5)


def test_scalar_tempo_of_empty_array_is_zero() -> None:
    """No tempo estimate at all yields 0.0, so the caller reports zero confidence
    rather than raising a 500 out of a successful decode."""
    assert _scalar_tempo(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_is_one_for_perfect_beats() -> None:
    beats = np.arange(20, dtype=float) * 0.5
    assert _beat_confidence(beats, 120.0) == pytest.approx(1.0)


def test_confidence_drops_with_jitter() -> None:
    rng = np.random.default_rng(7)
    beats = np.arange(40, dtype=float) * 0.5
    jittered = beats + rng.normal(0.0, 0.03, size=beats.size)
    tight = _beat_confidence(beats, 120.0)
    loose = _beat_confidence(np.sort(jittered), 120.0)
    assert loose < tight
    # Real music jitter (~22-34 ms measured on curated tracks) must still read as
    # a confident grid — it is musical drift, not detection failure.
    assert loose > 0.5


def test_confidence_is_zero_without_enough_beats() -> None:
    assert _beat_confidence(np.array([0.0, 0.5]), 120.0) == 0.0
    assert _beat_confidence(np.array([]), 120.0) == 0.0


def test_confidence_is_zero_for_a_nonsense_tempo() -> None:
    assert _beat_confidence(np.arange(10, dtype=float) * 0.5, 0.0) == 0.0


def test_confidence_floors_at_zero_for_a_wrong_period() -> None:
    """Intervals nowhere near 60/bpm score 0, never a negative number."""
    beats = np.arange(10, dtype=float) * 2.0  # 30 bpm intervals...
    assert _beat_confidence(beats, 120.0) == 0.0  # ...against a 120 bpm claim


# ---------------------------------------------------------------------------
# Duration guard
# ---------------------------------------------------------------------------


async def test_analyze_file_rejects_long_audio_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard runs off the ffprobe header — nothing is decoded."""
    audio = tmp_path / "long.mp3"
    audio.write_bytes(b"not-really-audio")

    async def _fake_probe(path: Path, *, timeout: float = 30.0) -> float:
        return 901.0

    async def _explode(path: Path, *, timeout: float = 120.0) -> np.ndarray:
        raise AssertionError("decode_pcm must not run for over-long audio")

    monkeypatch.setattr(beat_detector, "probe_duration", _fake_probe)
    monkeypatch.setattr(beat_detector, "decode_pcm", _explode)

    with pytest.raises(AudioTooLongError, match="exceeds"):
        await beat_detector.analyze_file(audio, max_duration_s=600.0)


async def test_analyze_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(BeatGridError, match="not found"):
        await beat_detector.analyze_file(tmp_path / "nope.mp3")


async def test_analyze_file_analyses_within_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "ok.mp3"
    audio.write_bytes(b"x")

    async def _fake_probe(path: Path, *, timeout: float = 30.0) -> float:
        return 8.0

    async def _fake_decode(path: Path, *, timeout: float = 120.0) -> np.ndarray:
        return _click_track(bpm=120.0, seconds=8.0)

    monkeypatch.setattr(beat_detector, "probe_duration", _fake_probe)
    monkeypatch.setattr(beat_detector, "decode_pcm", _fake_decode)

    grid = await beat_detector.analyze_file(audio, meter=4)
    assert _fold_to_octave(grid.bpm) == pytest.approx(120.0, abs=2.0)


# ---------------------------------------------------------------------------
# Real ffmpeg: probe + decode
# ---------------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg binary not available on PATH"
)


@pytest.fixture
async def synthetic_wav(tmp_path: Path) -> Path:
    """3 seconds of a 440 Hz sine, written by ffmpeg's own lavfi source."""
    out = tmp_path / "tone.wav"
    await run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out),
        ]
    )
    return out


@needs_ffmpeg
async def test_probe_duration_reads_the_container_header(synthetic_wav: Path) -> None:
    assert await beat_detector.probe_duration(synthetic_wav) == pytest.approx(3.0, abs=0.05)


@needs_ffmpeg
async def test_decode_pcm_downmixes_and_resamples(synthetic_wav: Path) -> None:
    """Stereo 44.1 kHz in, mono float32 at SAMPLE_RATE out — and never via soundfile."""
    samples = await beat_detector.decode_pcm(synthetic_wav)
    assert samples.dtype == np.dtype("<f4")
    assert samples.size == pytest.approx(3.0 * SAMPLE_RATE, rel=0.02)
    assert float(np.max(np.abs(samples))) > 0.1


@needs_ffmpeg
async def test_analyze_file_end_to_end(synthetic_wav: Path) -> None:
    grid = await beat_detector.analyze_file(synthetic_wav, meter=4)
    assert grid.duration_s == pytest.approx(3.0, abs=0.1)
    assert grid.meter == 4
    assert 0.0 <= grid.bpm_confidence <= 1.0


@needs_ffmpeg
async def test_probe_duration_rejects_a_non_media_file(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not audio")
    with pytest.raises(BeatGridError, match="probe"):
        await beat_detector.probe_duration(junk)


@needs_ffmpeg
async def test_decode_pcm_rejects_a_non_media_file(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not audio")
    with pytest.raises(BeatGridError, match="decode"):
        await beat_detector.decode_pcm(junk)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


async def test_warm_up_is_idempotent() -> None:
    assert await warm_up() is True
    assert await warm_up() is True
    assert beat_detector._WARMED is True


async def test_warm_up_proves_the_real_pipeline_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """A True from ``warm_up`` must mean the librosa/numba pipeline actually ran.

    Idempotency alone is satisfied by a totally broken pipeline — which is exactly
    what shipped: under a read-only rootfs with no NUMBA_CACHE_DIR every librosa
    call raised ``cannot cache function ...: no locator available``, warm_up
    swallowed it, logged "Beat detector warm" and returned True, and every real
    request then 500'd. So this test runs warm_up un-latched and then measures a
    120 BPM click track through the SAME code path. No mock can satisfy it: the
    tempo has to come out of the real tracker.
    """
    monkeypatch.setattr(beat_detector, "_WARMED", False)
    assert await warm_up() is True
    assert beat_detector._WARMED is True

    grid = analyze_samples(_click_track(bpm=120.0, seconds=8.0), meter=4)
    assert _fold_to_octave(grid.bpm) == pytest.approx(120.0, abs=1.5)
    # 8s at 120 bpm is 16 clicks. With trim=False the head is kept and the
    # boundary phantom is dropped by _drop_phantom_final_beat, so this lands on
    # 16 — the bounds stay as they were, they are just no longer slack for a
    # head-trim that must not happen.
    assert 12 <= len(grid.beats) <= 17
    assert grid.bpm_confidence > 0.9


async def test_warm_up_returns_false_when_the_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact prod failure: numba raises a bare RuntimeError on every call.

    It must NOT latch (a later attempt has to be able to succeed) and must NOT
    claim success — the caller stores this value and gates the endpoint on it.
    """

    def _numba_cache_failure(samples: np.ndarray, *, meter: int = 4) -> beat_detector.BeatGrid:
        raise RuntimeError("cannot cache function '__o_fold': no locator available for file")

    monkeypatch.setattr(beat_detector, "_WARMED", False)
    monkeypatch.setattr(beat_detector, "analyze_samples", _numba_cache_failure)

    assert await warm_up() is False
    assert beat_detector._WARMED is False


# ---------------------------------------------------------------------------
# POST /api/analyze/audio — no DB, no network
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FastAPI:
    """The analysis router on a bare app. This endpoint touches no database."""
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "jobs"))
    monkeypatch.setenv("STORAGE_UPLOAD_DIR", str(tmp_path / "uploads"))
    clear_settings_cache()
    app = FastAPI()
    app.state.settings = get_settings()
    # ``beat_detector_ready`` now defaults to NOT-ready when unset, so a test app
    # that wants the endpoint open must record the warm-up the way ``_startup``
    # does. (test_audio_endpoint_503_when_warmup_failed flips it back to False.)
    app.state.beat_detector_ready = True
    app.include_router(analysis_router)
    return attach_test_tenant(app)


@pytest.mark.parametrize(
    "hostile",
    [
        "s3://some-bucket/track.mp3",
        "file:///etc/passwd",
        "/etc/passwd",
        "../../etc/shadow",
        "/app/data/uploads/track.mp3",
        "",
    ],
)
def test_audio_endpoint_rejects_non_http_sources(audio_app: FastAPI, hostile: str) -> None:
    """No filesystem path and no object-store scheme may re-enter the request surface."""
    resp = TestClient(audio_app).post("/api/analyze/audio", json={"audio_url": hostile})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "hostile",
    [
        "http://169.254.169.254/latest/meta-data/track.mp3",  # cloud metadata
        "http://127.0.0.1:9000/track.mp3",  # loopback
        "http://10.0.0.5/track.mp3",  # RFC1918
        "http://minio.test/bucket/track.mp3",  # cluster suffix
        "http://metadata.google.test/track.mp3",  # named metadata host
        "http://localhost/track.mp3",
    ],
)
def test_audio_endpoint_rejects_ssrf_targets(audio_app: FastAPI, hostile: str) -> None:
    """Same SSRF posture as ``video_url``: url_security.validate_public_url, via
    VideoDownloader.download. Every case here is judged on the literal address or
    the hostname blocklist, so no DNS lookup and no socket are involved."""
    resp = TestClient(audio_app).post("/api/analyze/audio", json={"audio_url": hostile})
    assert resp.status_code == 400, resp.text


def test_audio_endpoint_rejects_out_of_range_meter(audio_app: FastAPI) -> None:
    resp = TestClient(audio_app).post(
        "/api/analyze/audio",
        json={"audio_url": "https://example.com/track.mp3", "meter": 0},
    )
    assert resp.status_code == 422


def test_audio_endpoint_returns_the_grid(
    audio_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path with the fetch stubbed out — the wire shape is the assertion."""
    from src.api.routes import analysis as analysis_module

    downloaded: list[str] = []

    class _StubDownloader:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def download(self, *, video_url: str, output_dir: Path) -> Path:
            downloaded.append(video_url)
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "track.mp3"
            target.write_bytes(b"stub")
            return target

        async def aclose(self) -> None:
            return None

    async def _fake_analyze(
        path: Path, *, meter: int = 4, max_duration_s: float = 600.0
    ) -> beat_detector.BeatGrid:
        return beat_detector.BeatGrid(
            duration_s=8.0,
            bpm=120.0,
            bpm_confidence=0.97,
            beats=[0.0, 0.5, 1.0, 1.5],
            downbeats=[0.0],
            onsets=[0.0, 0.5, 1.0, 1.5],
            meter=meter,
        )

    monkeypatch.setattr(analysis_module, "VideoDownloader", _StubDownloader)
    monkeypatch.setattr(beat_detector, "analyze_file", _fake_analyze)

    resp = TestClient(audio_app).post(
        "/api/analyze/audio",
        json={"audio_url": "https://example.com/track.mp3", "meter": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert downloaded == ["https://example.com/track.mp3"]
    assert body == {
        "duration_s": 8.0,
        "bpm": 120.0,
        "bpm_confidence": 0.97,
        "beats": [0.0, 0.5, 1.0, 1.5],
        "downbeats": [0.0],
        "onsets": [0.0, 0.5, 1.0, 1.5],
        "meter": 4,
        "grid_source": beat_detector.GRID_SOURCE,
    }


def test_audio_endpoint_is_503_when_the_detector_never_warmed(audio_app: FastAPI) -> None:
    """503, not a numba 500 — and refused before a single byte is fetched."""
    from src.api.routes import analysis as analysis_module

    class _NeverDownloader:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("nothing may be fetched when beat detection is unavailable")

    audio_app.state.beat_detector_ready = False
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(analysis_module, "VideoDownloader", _NeverDownloader)
        resp = TestClient(audio_app).post(
            "/api/analyze/audio", json={"audio_url": "https://example.com/track.mp3"}
        )
    assert resp.status_code == 503, resp.text
    assert "beat detection unavailable" in resp.text


def test_audio_endpoint_maps_too_long_to_413(
    audio_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.api.routes import analysis as analysis_module

    class _StubDownloader:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def download(self, *, video_url: str, output_dir: Path) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "track.mp3"
            target.write_bytes(b"stub")
            return target

        async def aclose(self) -> None:
            return None

    async def _too_long(
        path: Path, *, meter: int = 4, max_duration_s: float = 600.0
    ) -> beat_detector.BeatGrid:
        raise AudioTooLongError("audio is 901.0s, which exceeds the 600s limit")

    monkeypatch.setattr(analysis_module, "VideoDownloader", _StubDownloader)
    monkeypatch.setattr(beat_detector, "analyze_file", _too_long)

    resp = TestClient(audio_app).post(
        "/api/analyze/audio", json={"audio_url": "https://example.com/long.mp3"}
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.text


def test_audio_endpoint_maps_undecodable_to_422(
    audio_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.api.routes import analysis as analysis_module

    class _StubDownloader:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def download(self, *, video_url: str, output_dir: Path) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "track.bin"
            target.write_bytes(b"stub")
            return target

        async def aclose(self) -> None:
            return None

    async def _undecodable(
        path: Path, *, meter: int = 4, max_duration_s: float = 600.0
    ) -> beat_detector.BeatGrid:
        raise BeatGridError("could not decode audio: ffmpeg exited with code 1")

    monkeypatch.setattr(analysis_module, "VideoDownloader", _StubDownloader)
    monkeypatch.setattr(beat_detector, "analyze_file", _undecodable)

    resp = TestClient(audio_app).post(
        "/api/analyze/audio", json={"audio_url": "https://example.com/x.bin"}
    )
    assert resp.status_code == 422


async def test_audio_endpoint_reclaims_workdir_even_if_the_cleanup_await_dies(
    audio_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``finally`` must reclaim the downloaded bytes even when its own await dies.

    Under a real client disconnect this ``finally`` runs inside an already-cancelled
    anyio scope, where every ``await`` re-raises ``CancelledError`` immediately. With
    ``await downloader.aclose()`` placed BEFORE the ``rmtree``, the rmtree was never
    reached and up to ``_MAX_AUDIO_BYTES`` (512 MiB) of audio stayed on the pod's
    ephemeral disk. ``rmtree`` is synchronous and uninterruptible, so it goes first —
    the same rule ``_abort_upload_job`` already documents and follows.
    """
    from src.api.routes import analysis as analysis_routes

    captured: dict[str, Path] = {}

    class _FakeDownloader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def download(self, *, video_url: str, output_dir: Path) -> Path:
            captured["work_dir"] = output_dir
            path = output_dir / "track.mp3"
            path.write_bytes(b"ID3" + b"\x00" * 1024)
            return path

        async def aclose(self) -> None:
            # Models the cancelled scope: this await never completes.
            raise asyncio.CancelledError

    async def _fake_analyze_file(path: Path, *, meter: int) -> beat_detector.BeatGrid:
        return beat_detector.BeatGrid(
            bpm=120.0,
            bpm_confidence=0.9,
            beats=[0.0, 0.5],
            downbeats=[0.0],
            meter=meter,
            duration_s=1.0,
            onsets=[0.0],
        )

    monkeypatch.setattr(analysis_routes, "VideoDownloader", _FakeDownloader)
    monkeypatch.setattr(beat_detector, "analyze_file", _fake_analyze_file)

    request = SimpleNamespace(
        app=audio_app,
        headers={"x-tenant-id": str(TENANT_A)},
        state=SimpleNamespace(tenant_id=TENANT_A),
    )
    with contextlib.suppress(asyncio.CancelledError):
        await analysis_routes.analyze_audio_beat_grid(
            cast("Any", request), AudioBeatGridRequest(audio_url="https://example.com/track.mp3")
        )

    work_dir = captured["work_dir"]
    assert not work_dir.exists(), f"{work_dir} leaked: the cleanup await ran before the rmtree"
