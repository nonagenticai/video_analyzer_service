"""ffprobe JSON parser tests."""

from __future__ import annotations

import pytest

from src.services.video_prober import VideoProbeError, _parse_fps, parse_ffprobe_json


def _ffprobe_payload(**overrides):
    video_stream = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30000/1001",
        "nb_frames": "900",
    }
    video_stream.update(overrides.get("video", {}))
    payload = {
        "streams": [video_stream, {"codec_type": "audio"}],
        "format": {"duration": "30.0", "format_name": "mov,mp4"},
    }
    if "format" in overrides:
        payload["format"].update(overrides["format"])
    if "streams" in overrides:
        payload["streams"] = overrides["streams"]
    return payload


def test_parse_ffprobe_json_happy_path() -> None:
    meta = parse_ffprobe_json(_ffprobe_payload())
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.duration_seconds == 30.0
    assert meta.has_audio is True
    assert meta.fps == pytest.approx(29.97, rel=1e-3)
    assert meta.frame_count == 900
    assert meta.format_name == "mov,mp4"


def test_parse_ffprobe_json_no_audio_stream() -> None:
    meta = parse_ffprobe_json(
        {
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480, "avg_frame_rate": "30/1"}
            ],
            "format": {"duration": "10"},
        }
    )
    assert meta.has_audio is False


def test_parse_ffprobe_json_rejects_no_video() -> None:
    with pytest.raises(VideoProbeError):
        parse_ffprobe_json({"streams": [{"codec_type": "audio"}], "format": {}})


def test_parse_ffprobe_json_rejects_bad_dimensions() -> None:
    with pytest.raises(VideoProbeError):
        parse_ffprobe_json(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": "abc",
                        "height": 1080,
                        "avg_frame_rate": "30/1",
                    }
                ],
                "format": {"duration": "5"},
            }
        )


def test_parse_ffprobe_json_rejects_nonnumeric_duration() -> None:
    with pytest.raises(VideoProbeError):
        parse_ffprobe_json(
            {
                "streams": [
                    {"codec_type": "video", "width": 1, "height": 1, "avg_frame_rate": "30/1"}
                ],
                "format": {"duration": "bananas"},
            }
        )


def test_parse_fps_variants() -> None:
    assert _parse_fps("30/1") == 30.0
    assert _parse_fps("30000/1001") == pytest.approx(29.97, rel=1e-3)
    assert _parse_fps("25") == 25.0
    assert _parse_fps("0/0") == 0.0
    assert _parse_fps(None) == 0.0
    assert _parse_fps("bad/value") == 0.0
    assert _parse_fps("nonsense") == 0.0


# ── duration is the VIDEO stream's, not the container's ──────────────


def test_video_stream_duration_wins_over_padded_container() -> None:
    """A 6s video muxed with 12s of audio must probe as 6.0, not 12.0.

    IF REVERTED: ``format.duration`` (the max across streams) wins and this
    returns 12.0. The sampling interval then becomes 12/5 = 2.400 instead of
    1.200, so 2 of the 5 sample points land past the real EOF, ``ffmpeg -ss``
    writes nothing for them, and 40% of the sample set vanishes from the means
    the blank-clip hard-fail gate votes on.
    """
    meta = parse_ffprobe_json(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1",
                    "duration": "6.000000",
                    "nb_frames": "180",
                },
                {"codec_type": "audio", "duration": "12.000000"},
            ],
            "format": {"duration": "12.000000", "format_name": "mov,mp4"},
        }
    )
    assert meta.duration_seconds == 6.0
    assert meta.frame_count == 180


def test_webm_without_stream_duration_falls_back_to_format() -> None:
    """Matroska/WebM carries no stream ``duration`` and no ``nb_frames``.

    IF REVERTED (i.e. if the format fallback were dropped from the chain rather
    than merely demoted): the resolver returns 0.0, ``compute_frame_timestamps``
    returns [], and the job completes "successfully" having extracted ZERO
    frames.
    """
    meta = parse_ffprobe_json(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "25/1",
                    "codec_name": "vp9",
                }
            ],
            "format": {"duration": "8.500000", "format_name": "matroska,webm"},
        }
    )
    assert meta.duration_seconds == 8.5


def test_webm_duration_tag_beats_padded_container() -> None:
    """The Matroska ``DURATION`` tag is the only stream-local length a WebM has.

    IF REVERTED: the padded container duration (11s of audio) is used for a 6.5s
    video and the sampling interval is inflated by ~70%.
    """
    meta = parse_ffprobe_json(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "25/1",
                    "tags": {"DURATION": "00:00:06.500000000"},
                },
                {"codec_type": "audio", "tags": {"DURATION": "00:00:11.000000000"}},
            ],
            "format": {"duration": "11.000000", "format_name": "matroska,webm"},
        }
    )
    assert meta.duration_seconds == pytest.approx(6.5)


def test_duration_derived_from_nb_frames_when_stream_duration_absent() -> None:
    """nb_frames / fps is exact for CFR and beats the padded container.

    IF REVERTED: 20.0 (the audio length) instead of 5.0.
    """
    meta = parse_ffprobe_json(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30/1",
                    "nb_frames": "150",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "20.0"},
        }
    )
    assert meta.duration_seconds == pytest.approx(5.0)


def test_lying_stream_duration_is_clamped_to_container() -> None:
    """The container length is an upper bound over its streams by construction."""
    meta = parse_ffprobe_json(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30/1",
                    "duration": "999.0",
                }
            ],
            "format": {"duration": "12.0"},
        }
    )
    assert meta.duration_seconds == 12.0


def test_nonnumeric_stream_duration_still_raises() -> None:
    """A PRESENT-but-unparseable duration is a corrupt probe, not a missing one."""
    with pytest.raises(VideoProbeError):
        parse_ffprobe_json(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1,
                        "height": 1,
                        "avg_frame_rate": "30/1",
                        "duration": "bananas",
                    }
                ],
                "format": {"duration": "5"},
            }
        )


def test_parse_ffprobe_json_infers_frame_count_from_duration() -> None:
    meta = parse_ffprobe_json(
        {
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480, "avg_frame_rate": "30/1"}
            ],
            "format": {"duration": "2"},
        }
    )
    assert meta.frame_count == 60
