"""Strict data models for the Gemini transcript response.

The Gemini API is asked to return JSON shaped like::

    {
      "language": "en",
      "duration_seconds": 120.5,
      "segments": [
        {"id": 1, "start_seconds": 0.0, "end_seconds": 3.2, "text": "Hi."}
      ],
      "full_transcript": "Hi."
    }

`Segment` and `Transcript` enforce the invariants required by the spec:

* each ``end_seconds`` is strictly after its ``start_seconds``
* segments are ordered chronologically
* segments do not overlap

`repair_segments` performs best-effort cleanup on the *raw* model output
(sorting, clamping tiny overlaps, dropping empty text, re-indexing) so that a
slightly-imperfect model response can still be turned into a valid transcript
instead of being rejected outright.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Minimum on-screen duration we allow when repairing degenerate segments.
_MIN_SEGMENT_SECONDS = 0.05


class Segment(BaseModel):
    """A single timestamped subtitle segment."""

    id: int = Field(..., ge=1, description="1-based segment number")
    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float = Field(..., ge=0.0)
    text: str

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("segment text must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _end_after_start(self) -> "Segment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"segment {self.id}: end_seconds ({self.end_seconds}) must be "
                f"after start_seconds ({self.start_seconds})"
            )
        return self


class Transcript(BaseModel):
    """A full, validated transcript.

    Instantiating this model enforces every ordering/overlap invariant, so any
    ``Transcript`` object in the app is guaranteed to be safe to render to SRT
    or WebVTT.
    """

    language: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    segments: List[Segment]
    full_transcript: str = ""

    @field_validator("segments")
    @classmethod
    def _segments_not_empty(cls, value: List[Segment]) -> List[Segment]:
        if not value:
            raise ValueError("transcript must contain at least one segment")
        return value

    @model_validator(mode="after")
    def _ordered_and_non_overlapping(self) -> "Transcript":
        previous: Optional[Segment] = None
        for seg in self.segments:
            if previous is not None:
                if seg.start_seconds < previous.start_seconds:
                    raise ValueError(
                        "segments are not ordered chronologically "
                        f"(segment {seg.id} starts before segment {previous.id})"
                    )
                if seg.start_seconds < previous.end_seconds:
                    raise ValueError(
                        f"segments {previous.id} and {seg.id} overlap "
                        f"({previous.end_seconds} > {seg.start_seconds})"
                    )
            previous = seg
        return self

    def rebuild_full_transcript(self) -> "Transcript":
        """Return a copy whose ``full_transcript`` is the joined segment text."""
        joined = " ".join(seg.text for seg in self.segments).strip()
        return self.model_copy(update={"full_transcript": joined})


def repair_segments(raw_segments: List[dict]) -> List[dict]:
    """Best-effort cleanup of raw segment dicts from the model.

    This does *not* raise on the kinds of small imperfections a language model
    commonly produces. It:

    * drops segments with missing/blank text,
    * coerces start/end to floats and clamps negatives to 0,
    * sorts by start time (then end time),
    * clamps an overlapping ``end`` down to the next segment's ``start``,
    * guarantees a minimum visible duration,
    * re-indexes ``id`` sequentially from 1.

    The result is still validated by :class:`Transcript`, which will reject
    anything that could not be repaired.
    """
    cleaned: List[dict] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        try:
            start = max(0.0, float(raw.get("start_seconds", 0.0)))
            end = float(raw.get("end_seconds", 0.0))
        except (TypeError, ValueError):
            continue
        cleaned.append({"start_seconds": start, "end_seconds": end, "text": text})

    cleaned.sort(key=lambda s: (s["start_seconds"], s["end_seconds"]))

    repaired: List[dict] = []
    for index, seg in enumerate(cleaned):
        start = seg["start_seconds"]
        end = seg["end_seconds"]

        # Ensure a minimum visible duration.
        if end <= start:
            end = start + _MIN_SEGMENT_SECONDS

        # Clamp against the following segment's start to remove overlap.
        if index + 1 < len(cleaned):
            next_start = cleaned[index + 1]["start_seconds"]
            if end > next_start:
                end = max(start + _MIN_SEGMENT_SECONDS, next_start)
                # If clamping pushed us past the neighbour, nudge the neighbour.
                if end > next_start:
                    cleaned[index + 1]["start_seconds"] = end

        repaired.append(
            {
                "id": index + 1,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "text": seg["text"],
            }
        )
    return repaired


def build_transcript(payload: dict) -> Transcript:
    """Build a validated :class:`Transcript` from a raw decoded JSON payload.

    Raises ``ValueError`` (or ``pydantic.ValidationError``) if the payload is
    structurally unusable even after repair.
    """
    if not isinstance(payload, dict):
        raise ValueError("transcript payload must be a JSON object")
    if "segments" not in payload or not isinstance(payload["segments"], list):
        raise ValueError("transcript payload is missing a 'segments' array")

    repaired = repair_segments(payload["segments"])
    if not repaired:
        raise ValueError("no usable segments were produced from the response")

    transcript = Transcript(
        language=payload.get("language"),
        duration_seconds=payload.get("duration_seconds"),
        segments=repaired,  # type: ignore[arg-type]
        full_transcript=str(payload.get("full_transcript", "")),
    )
    if not transcript.full_transcript.strip():
        transcript = transcript.rebuild_full_transcript()
    return transcript
