"""Tests for transcript validation, ordering, overlap detection, and repair."""

import pytest
from pydantic import ValidationError

from models.transcript import (
    Segment,
    Transcript,
    build_transcript,
    repair_segments,
)


def test_valid_transcript_parses():
    payload = {
        "language": "en",
        "duration_seconds": 6.0,
        "segments": [
            {"id": 1, "start_seconds": 0.0, "end_seconds": 3.2, "text": "Hello."},
            {"id": 2, "start_seconds": 3.2, "end_seconds": 6.0, "text": "World."},
        ],
        "full_transcript": "Hello. World.",
    }
    t = build_transcript(payload)
    assert t.language == "en"
    assert len(t.segments) == 2


def test_segment_end_must_be_after_start():
    with pytest.raises(ValidationError):
        Segment(id=1, start_seconds=5.0, end_seconds=5.0, text="x")
    with pytest.raises(ValidationError):
        Segment(id=1, start_seconds=5.0, end_seconds=4.0, text="x")


def test_blank_text_rejected():
    with pytest.raises(ValidationError):
        Segment(id=1, start_seconds=0.0, end_seconds=1.0, text="   ")


def test_overlapping_segments_rejected():
    with pytest.raises(ValidationError):
        Transcript(
            segments=[
                Segment(id=1, start_seconds=0.0, end_seconds=4.0, text="A"),
                Segment(id=2, start_seconds=3.0, end_seconds=6.0, text="B"),
            ]
        )


def test_out_of_order_segments_rejected():
    with pytest.raises(ValidationError):
        Transcript(
            segments=[
                Segment(id=1, start_seconds=5.0, end_seconds=6.0, text="A"),
                Segment(id=2, start_seconds=0.0, end_seconds=1.0, text="B"),
            ]
        )


def test_malformed_payload_raises():
    with pytest.raises(ValueError):
        build_transcript({"language": "en"})  # no segments
    with pytest.raises(ValueError):
        build_transcript("not a dict")  # type: ignore[arg-type]


def test_repair_sorts_and_removes_overlap():
    raw = [
        {"id": 2, "start_seconds": 3.0, "end_seconds": 6.0, "text": "second"},
        {"id": 1, "start_seconds": 0.0, "end_seconds": 4.0, "text": "first"},
    ]
    repaired = repair_segments(raw)
    # Sorted by start; ids re-assigned; overlap removed.
    assert [s["text"] for s in repaired] == ["first", "second"]
    assert repaired[0]["id"] == 1 and repaired[1]["id"] == 2
    assert repaired[0]["end_seconds"] <= repaired[1]["start_seconds"]


def test_repair_drops_blank_and_fixes_zero_duration():
    raw = [
        {"start_seconds": 0.0, "end_seconds": 0.0, "text": "kept"},
        {"start_seconds": 1.0, "end_seconds": 2.0, "text": "   "},
    ]
    repaired = repair_segments(raw)
    assert len(repaired) == 1
    assert repaired[0]["end_seconds"] > repaired[0]["start_seconds"]


def test_build_transcript_fills_full_transcript_when_missing():
    payload = {
        "segments": [
            {"id": 1, "start_seconds": 0.0, "end_seconds": 1.0, "text": "Hi"},
            {"id": 2, "start_seconds": 1.0, "end_seconds": 2.0, "text": "there"},
        ],
    }
    t = build_transcript(payload)
    assert t.full_transcript == "Hi there"


def test_repair_handles_bad_numeric_values():
    raw = [
        {"start_seconds": "oops", "end_seconds": 2.0, "text": "bad"},
        {"start_seconds": -1.0, "end_seconds": 1.0, "text": "clamped"},
    ]
    repaired = repair_segments(raw)
    assert len(repaired) == 1
    assert repaired[0]["start_seconds"] == 0.0  # negative clamped
