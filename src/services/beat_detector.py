"""Beat / onset grid detection for music-cut video products.

Why this exists
===============
The music-cut product type cuts user footage *to the beat* and time-warps each
clip so its out-point lands on one. Nothing in the fleet could produce a beat
grid, so every "cut to the music" attempt was hand-timed. This module is the
missing measurement: audio in, a grid of beat/downbeat/onset timestamps out.

Three implementation decisions carry the real traps:

1. **ffmpeg decodes, librosa never touches the file.** ``librosa.load`` goes
   through ``soundfile`` -> ``libsndfile``, a C library that has to be present
   in the image and that does not read mp3/m4a/aac in every build. We already
   ship ffmpeg (this service is built around it), so we decode to raw
   little-endian f32 mono PCM on stdout and hand librosa a numpy array. That
   removes ``soundfile``/``libsndfile`` from the dependency surface entirely.
   Measured: 0.15-0.33s to decode a 58s mp3.

2. **The first call costs ~9.7s, every later call costs ~0.04s.** librosa's
   beat tracker runs through numba, and numba JIT-compiles on first use. That
   cost is paid by whoever calls first — which, without :func:`warm_up`, is a
   real customer request that appears to hang for ten seconds. Warming up at
   startup is therefore mandatory, not an optimisation.

3. **Tempo comes back as an array, not a float.** ``librosa.beat.beat_track``
   returns ``tempo`` as a 0-d or 1-element ``ndarray`` depending on version and
   on whether the input was multi-channel. ``float(tempo)`` on a 0-d array works
   but on a 1-element array is deprecated, and ``tempo == 120`` is an array
   comparison, not a bool. :func:`_scalar_tempo` normalises the whole family.

Downbeats are INFERRED, not tracked
-----------------------------------
librosa has no DBN downbeat tracker (that is madmom's territory, which drags in
a heavyweight and unmaintained-on-3.14 dependency tree). What we do instead is
pick the phase: for each candidate phase ``p`` in ``[0, meter)``, sum the onset
strength over the beats ``p, p+meter, p+2*meter, ...`` and take the phase whose
beats are collectively the loudest. On 4/4 material with any kick-on-1 emphasis
this lands on the true downbeat; on material with no metrical accent it degrades
to "every meter-th beat starting at the first", which is still a usable cut grid
and is honestly reported through ``bpm_confidence`` — never as a false claim of
downbeat tracking.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.services.ffmpeg_utils import FFmpegError, run_checked, run_checked_bytes

logger = logging.getLogger(__name__)

# 22050 Hz mono is librosa's own default analysis rate. Beat tracking works on
# an onset-strength envelope derived from a mel spectrogram; nothing above ~11
# kHz contributes, so resampling higher costs CPU and buys nothing.
SAMPLE_RATE = 22050
# 256 samples @ 22050 Hz = 11.6 ms per frame. It MUST be passed identically to
# onset_strength, beat_track, onset_detect and time_to_frames — a hop mismatch
# between any two of them silently shifts every timestamp by a constant factor.
#
# ⚠ NOT librosa's default of 512. Beat times are quantised to whole frames, and
# at 23.2 ms/frame a common tempo lands on a NON-INTEGER number of frames per
# beat, so the grid drifts. Measured on a synthetic 120 BPM metronome (exactly
# 0.5000 s apart): hop 512 reports 117.45 BPM with a 0.5108 s median gap, and
# the accumulated error reaches ~490 ms — a whole beat out by the end of a 30 s
# track, which is precisely the "cuts slide off the beat" failure this product
# cannot tolerate. hop 256 reports 120.185 BPM at 0.4992 s and holds every beat
# within 19.5 ms. Real music benefits too (a curated bed's inter-beat jitter
# fell from 21.9 ms to 8.5 ms). 128/64 buy no further accuracy and cost 4-15x
# the CPU, so 256 is the knee of the curve, not a guess.
HOP_LENGTH = 256

# Server-side ceiling. Deliberately NOT caller-settable: this endpoint is
# synchronous, so a caller who could raise the cap could hold a worker (and a
# CPU) for as long as they liked. 10 minutes covers any music bed a cut-to-beat
# product uses; longer input is a different product and should be a job.
MAX_DURATION_S = 600.0

# Identifies which algorithm produced the grid, so a consumer that later gets a
# DBN-tracked or human-authored grid can tell them apart without guessing.
GRID_SOURCE = "librosa.beat_track+onset_phase"

# Tolerance band for :func:`_beat_confidence`. An inter-beat interval that is
# off the nominal period by this fraction (25%) or more contributes nothing.
_CONFIDENCE_TOLERANCE = 0.25

# Tail-phantom guard (see :func:`_drop_phantom_final_beat`). A final beat is a
# phantom if it is more than 1.5x the median inter-beat interval after its
# predecessor, OR if no detected onset sits within half that interval of it.
_TAIL_GAP_FACTOR = 1.5
_TAIL_ONSET_WINDOW = 0.5


class BeatGridError(RuntimeError):
    """Raised when audio cannot be decoded into a usable beat grid."""


class AudioTooLongError(BeatGridError):
    """Raised when the input exceeds :data:`MAX_DURATION_S`."""


@dataclass(slots=True)
class BeatGrid:
    """A detected beat grid. All timestamps are seconds from the start of audio."""

    duration_s: float
    bpm: float
    bpm_confidence: float
    beats: list[float]
    downbeats: list[float]
    onsets: list[float]
    meter: int
    grid_source: str = GRID_SOURCE


def _scalar_tempo(tempo: object) -> float:
    """Normalise librosa's ``tempo`` return into a plain float.

    Handles all four shapes seen across librosa versions: a python float, a
    numpy scalar, a 0-d ndarray and a 1-element ndarray. An empty array (no
    tempo could be estimated at all) yields 0.0 rather than raising, so the
    caller reports a zero-confidence grid instead of a 500.

    Typed ``object`` rather than ``float | np.ndarray`` deliberately: librosa's
    own annotation widens to a union including ``np.bool`` and the integer
    scalar types, so a narrower parameter type is a type error at the one call
    site that matters. ``np.asarray`` accepts all of them, so widening here
    costs nothing and keeps the checker honest instead of silenced.
    """
    flat = np.atleast_1d(np.asarray(tempo, dtype=float)).ravel()
    if flat.size == 0:
        return 0.0
    return float(flat[0])


def _beat_confidence(beats: np.ndarray, bpm: float) -> float:
    """How metronomically the detected beats fit the detected tempo — 0.0 to 1.0.

    Definition, precisely: let ``period = 60 / bpm`` be the interval the
    reported tempo implies, and for each consecutive pair of beats let
    ``dev = |interval - period| / period`` be its *relative* deviation from it.
    Each interval scores ``max(0, 1 - dev / 0.25)`` and the confidence is the
    mean of those scores. So:

    * 1.0 = every beat lands exactly ``60/bpm`` apart (a click track / a
      quantised electronic bed);
    * ~0.7-0.9 = real recorded music. Measured on three curated 58s tracks:
      22-34 ms of jitter around a ~500 ms period, i.e. dev ~0.05-0.07. That
      jitter is real musical drift and is *desirable* — a track that scores 1.0
      is a machine, not a performance;
    * near 0.0 = the interval sequence does not fit a single tempo at all
      (tempo changes, rubato, or the tracker locked onto nothing).

    What it is NOT: a probability that the tempo is *correct*. A tracker that
    locks onto exactly half or double the true tempo produces an equally regular
    interval sequence and scores just as high. It measures grid regularity — the
    property a cut-to-beat editor actually depends on.

    Fewer than two intervals (i.e. fewer than three beats) is not enough
    evidence for any claim, so it returns 0.0.
    """
    if bpm <= 0.0 or beats.size < 3:
        return 0.0
    period = 60.0 / bpm
    intervals = np.diff(beats)
    if intervals.size == 0:
        return 0.0
    deviation = np.abs(intervals - period) / period
    scores = np.clip(1.0 - deviation / _CONFIDENCE_TOLERANCE, 0.0, 1.0)
    return float(np.mean(scores))


def _drop_phantom_final_beat(beats: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Drop a trailing beat that nothing in the audio supports. At most ONE.

    Replaces librosa's ``trim=True``, which trimmed the HEAD as well and cost
    real leading beats (see :func:`analyze_samples`). Two independent phantom
    signatures, because the measured artefact does not match the obvious one:

    * a gap of more than ``1.5x`` the median inter-beat interval after the
      predecessor — the textbook "extra beat hanging off the end";
    * no detected onset within half a median interval of it AND the beat sits
      past the LAST detected onset. This is the one the synthetic metronome
      actually needs: the DP tracker terminates at the last envelope frame, so a
      30 s / 120 BPM click track gets a 61st beat at 29.989 whose gap from its
      predecessor is 0.476 s — *shorter* than the 0.4992 s median, so the gap
      rule alone would never fire. It is 489 ms from the last real click, i.e. a
      beat predicted past the end of the audio content.

    The ``past the last onset`` half is corroboration, not decoration: the onset
    test on its own is a false-positive machine. Measured on the three curated
    beds, the share of beats with NO onset inside that window is 14.6% / 13.8% /
    3.8% (ambient_tech / corporate_uplift / jazz_lounge), and a truncation
    simulation — analysing a SUB-CLIP of a bed — deletes a genuine final beat in
    25.3% / 17.5% / 4.1% of cases. Today ``music_select._beat_grid`` always sends
    the whole track, so nothing reaches that path, but a trimmed or user-uploaded
    bed would. Being past the last onset is the actual phantom signature (a beat
    predicted beyond any detected audio event); a sparse-onset passage in the
    middle of real audio is not. With NO onsets at all there is nothing to
    corroborate against, so this rule cannot fire — only the gap rule can.

    The median is taken over the intervals EXCLUDING the last one, so a phantom
    cannot bias the yardstick it is measured against. Fewer than four beats is
    too little evidence for either rule, and both rules slice ``[:-1]`` exactly
    once, so a normal grid is returned untouched.
    """
    if beats.size < 4:
        return beats
    intervals = np.diff(beats)
    median = float(np.median(intervals[:-1]))
    if median <= 0.0:
        return beats
    if float(intervals[-1]) > _TAIL_GAP_FACTOR * median:
        return beats[:-1]
    if (
        onsets.size
        and float(beats[-1]) > float(onsets[-1])
        and float(np.min(np.abs(onsets - beats[-1]))) > _TAIL_ONSET_WINDOW * median
    ):
        return beats[:-1]
    return beats


def _pick_downbeat_phase(beats: np.ndarray, onset_envelope: np.ndarray, *, meter: int) -> int:
    """Return the phase in ``[0, meter)`` whose beats carry the most onset energy.

    Ties resolve to the LOWEST phase (``argmax`` semantics), which means silent
    or perfectly flat material degrades to "the first beat is the downbeat" —
    the same answer a human would give with no other information.
    """
    if beats.size == 0 or meter <= 1 or onset_envelope.size == 0:
        return 0
    frames = librosa.time_to_frames(beats, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # A beat can be reported at (or fractionally past) the end of the envelope;
    # clipping keeps the fancy-index in bounds instead of raising.
    frames = np.clip(np.asarray(frames, dtype=int), 0, max(onset_envelope.size - 1, 0))
    strength = onset_envelope[frames]
    scores = [float(strength[phase::meter].sum()) for phase in range(meter)]
    return int(np.argmax(scores))


def analyze_samples(samples: np.ndarray, *, meter: int = 4) -> BeatGrid:
    """Compute a beat grid from mono float32 samples at :data:`SAMPLE_RATE`.

    Pure, synchronous and CPU-bound — this is the function that must run off the
    event loop, and the one tests drive directly with synthesised audio.
    """
    if meter < 1:
        raise BeatGridError(f"meter must be >= 1, got {meter}")
    # ``np.frombuffer`` hands back a READ-ONLY view over the ffmpeg stdout
    # buffer, and librosa/numba write into their inputs in places. Force a real
    # copy: ``ascontiguousarray`` would NOT do it — for an already-contiguous
    # array of the right dtype it returns the same read-only object, so the trap
    # would survive the "defensive copy" that looks like it is fixing it.
    y = np.array(samples, dtype=np.float32, copy=True)
    if y.size == 0:
        raise BeatGridError("decoded audio is empty (no audio stream?)")

    duration_s = float(y.size) / float(SAMPLE_RATE)

    # ``aggregate=np.median`` across mel bands instead of the default mean: it
    # is markedly more robust to a single loud band (a bass drone, a hiss floor)
    # dominating the envelope and smearing the onset peaks.
    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, aggregate=np.median
    )
    #: ``trim=False``, NOT librosa's default. ``trim=True`` trims the HEAD as
    #: well as the tail and silently discarded real leading beats on every
    #: curated bed. Measured at this SR/hop: corporate_uplift 98 beats first at
    #: 14.408 s vs 130 beats first at 0.093 s; ambient_tech 4.725 s vs 0.000 s;
    #: jazz_lounge 0.778 s vs 0.023 s — same bpm, same LAST beat, so trim dropped
    #: 32 leading beats and zero trailing ones (the head is not quiet: -18.3 dBFS
    #: at t=0). Downstream, every action_reel fell back to even spacing.
    #: The one real thing trim bought — a phantom final beat at the signal
    #: boundary — is now handled explicitly by :func:`_drop_phantom_final_beat`.
    tempo, beat_times = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        trim=False,
        units="time",
    )
    bpm = _scalar_tempo(tempo)
    beats = np.atleast_1d(np.asarray(beat_times, dtype=float)).ravel()

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_envelope, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time"
    )
    onsets = np.atleast_1d(np.asarray(onset_times, dtype=float)).ravel()

    beats = _drop_phantom_final_beat(beats, onsets)

    phase = _pick_downbeat_phase(beats, onset_envelope, meter=meter)
    downbeats = beats[phase::meter]

    return BeatGrid(
        duration_s=round(duration_s, 6),
        bpm=round(bpm, 4),
        bpm_confidence=round(_beat_confidence(beats, bpm), 4),
        beats=[round(float(t), 6) for t in beats],
        downbeats=[round(float(t), 6) for t in downbeats],
        onsets=[round(float(t), 6) for t in onsets],
        meter=meter,
    )


async def probe_duration(path: Path, *, timeout: float = 30.0) -> float:
    """Return the container duration of ``path`` in seconds via ffprobe.

    Used as a CHEAP pre-flight gate: it reads the container header only, so a
    three-hour file is rejected without ever being decoded. ``probe_video`` is
    not reusable here — it requires a video stream and raises on a bare audio
    file, which is exactly what this endpoint receives.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        stdout, _ = await run_checked(cmd, timeout=timeout)
    except FFmpegError as exc:
        raise BeatGridError(f"could not probe audio: {exc}") from exc
    try:
        return float(stdout.strip())
    except ValueError as exc:
        # Streams with no container-level duration (some raw/streamed formats)
        # print "N/A" or nothing at all.
        raise BeatGridError(f"audio has no readable duration: {stdout.strip()!r}") from exc


async def decode_pcm(path: Path, *, timeout: float = 120.0) -> np.ndarray:
    """Decode ``path`` to mono float32 samples at :data:`SAMPLE_RATE` via ffmpeg.

    Output goes to stdout, not to a temp file: at 22050 Hz mono f32 the
    :data:`MAX_DURATION_S` ceiling caps this at ~53 MB, which is bounded enough
    to hold in memory and saves a write+read round-trip per request.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        # -vn: an mp3 with embedded cover art has a video stream; without this
        # ffmpeg tries to map it and fails on an audio-only output format.
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    try:
        raw = await run_checked_bytes(cmd, timeout=timeout)
    except FFmpegError as exc:
        raise BeatGridError(f"could not decode audio: {exc}") from exc
    if not raw:
        raise BeatGridError("ffmpeg produced no audio samples (no audio stream?)")
    # Truncate to a whole number of samples: a killed/short write could leave a
    # partial 4-byte frame, which frombuffer rejects outright.
    usable = len(raw) - (len(raw) % 4)
    return np.frombuffer(raw[:usable], dtype="<f4")


async def analyze_file(
    path: Path, *, meter: int = 4, max_duration_s: float = MAX_DURATION_S
) -> BeatGrid:
    """Probe, guard, decode and analyse an audio/video file on disk.

    Raises :class:`AudioTooLongError` BEFORE decoding anything if the file is
    longer than ``max_duration_s``.
    """
    if not path.exists():
        raise BeatGridError(f"audio file not found: {path}")

    duration = await probe_duration(path)
    if duration > max_duration_s:
        raise AudioTooLongError(
            f"audio is {duration:.1f}s, which exceeds the {max_duration_s:.0f}s limit"
        )

    samples = await decode_pcm(path)
    # librosa/numba is CPU-bound C/LLVM code with no awaits; running it inline
    # would stall every other request on this worker for the whole analysis.
    # ``asyncio.to_thread`` matches what the rest of this service uses
    # (see analysis.py's rmtree offload) — anyio is not used directly here.
    return await asyncio.to_thread(analyze_samples, samples, meter=meter)


# Module-level latch so the ~9.7s JIT is paid exactly once per process, whether
# it is triggered by startup, by a test, or by the first request racing startup.
_WARMED = False


def _synthesise_warmup_audio() -> np.ndarray:
    """One second of digital silence at :data:`SAMPLE_RATE`.

    Content is irrelevant — the point is to walk every numba-compiled code path
    (mel spectrogram -> onset strength -> beat tracker -> onset detect) so the
    LLVM compilation happens here rather than inside a customer request.
    """
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


async def warm_up() -> bool:
    """Pay the numba JIT cost up front. Returns True IFF the pipeline really ran.

    Idempotent and non-fatal: it is logged and returned, never raised into
    startup, because this service also does video analysis (pipeline, OCR,
    vision) that stays perfectly usable while beat detection is broken.

    False is a real answer, not a formality. The failure this signature exists
    for is numba's ``cannot cache function ...: no locator available`` under a
    read-only root filesystem with no ``NUMBA_CACHE_DIR``: it fires on EVERY
    call, not just this one, so a swallowed warm-up failure means every
    ``/api/analyze/audio`` request 500s from inside numba while startup logs
    "warm". The latch is therefore set ONLY on the success path — a failed
    warm-up must stay un-latched so a later attempt can still succeed.
    """
    global _WARMED
    if _WARMED:
        return True
    try:
        await asyncio.to_thread(analyze_samples, _synthesise_warmup_audio(), meter=4)
    except Exception:  # numba's cache/JIT failures surface as bare RuntimeError/OSError from its C layer; none of them may kill startup, and none of them may be reported as success
        logger.warning("Beat-detector warm-up did not complete cleanly", exc_info=True)
        return False
    _WARMED = True
    logger.info("Beat detector warm (numba JIT compiled)")
    return True
