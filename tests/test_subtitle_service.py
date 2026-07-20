"""Tests for timestamp formatting and SRT/WebVTT generation."""

from models.transcript import Segment
from services.subtitle_service import (
    generate_srt,
    generate_vtt,
    seconds_to_srt_timestamp,
    seconds_to_vtt_timestamp,
)


def test_srt_timestamp_basic():
    assert seconds_to_srt_timestamp(0) == "00:00:00,000"
    assert seconds_to_srt_timestamp(3.2) == "00:00:03,200"
    assert seconds_to_srt_timestamp(65.5) == "00:01:05,500"
    assert seconds_to_srt_timestamp(3661.123) == "01:01:01,123"


def test_vtt_timestamp_basic():
    assert seconds_to_vtt_timestamp(0) == "00:00:00.000"
    assert seconds_to_vtt_timestamp(3.2) == "00:00:03.200"
    assert seconds_to_vtt_timestamp(3661.123) == "01:01:01.123"


def test_timestamp_rounding_and_negative_clamp():
    assert seconds_to_srt_timestamp(1.2346) == "00:00:01,235"  # rounds to ms
    assert seconds_to_srt_timestamp(-5) == "00:00:00,000"  # clamps negatives


def test_generate_srt_structure():
    segments = [
        Segment(id=1, start_seconds=0.0, end_seconds=3.2, text="Welcome to this lesson."),
        Segment(id=2, start_seconds=3.2, end_seconds=6.0, text="Let's begin."),
    ]
    srt = generate_srt(segments)
    assert "1\r\n00:00:00,000 --> 00:00:03,200\r\nWelcome to this lesson." in srt
    assert "2\r\n00:00:03,200 --> 00:00:06,000\r\nLet's begin." in srt


def test_generate_vtt_structure():
    segments = [
        Segment(id=1, start_seconds=0.0, end_seconds=3.2, text="Welcome to this lesson."),
    ]
    vtt = generate_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:03.200" in vtt
    assert "Welcome to this lesson." in vtt


def test_srt_and_vtt_reindex_sequentially():
    segments = [
        Segment(id=7, start_seconds=0.0, end_seconds=1.0, text="A"),
        Segment(id=9, start_seconds=1.0, end_seconds=2.0, text="B"),
    ]
    srt = generate_srt(segments)
    # Output indices are 1,2 regardless of the source ids.
    assert srt.splitlines()[0] == "1"
    assert "\r\n2\r\n" in "\r\n" + srt
