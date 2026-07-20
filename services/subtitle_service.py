"""Convert a validated transcript into SRT and WebVTT subtitle files."""

from __future__ import annotations

from typing import Iterable

from models.transcript import Segment, Transcript


def _split_ms(seconds: float) -> tuple[int, int, int, int]:
    """Split a non-negative second value into (hh, mm, ss, milliseconds)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return hours, minutes, secs, millis


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    hours, minutes, secs, millis = _split_ms(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp ``HH:MM:SS.mmm``."""
    hours, minutes, secs, millis = _split_ms(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _iter_segments(source: Transcript | Iterable[Segment]) -> list[Segment]:
    if isinstance(source, Transcript):
        return list(source.segments)
    return list(source)


def generate_srt(source: Transcript | Iterable[Segment]) -> str:
    """Render segments as an SRT document (CRLF line endings per spec)."""
    lines: list[str] = []
    for index, seg in enumerate(_iter_segments(source), start=1):
        start = seconds_to_srt_timestamp(seg.start_seconds)
        end = seconds_to_srt_timestamp(seg.end_seconds)
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\r\n".join(lines).strip("\r\n") + "\r\n"


def generate_vtt(source: Transcript | Iterable[Segment]) -> str:
    """Render segments as a WebVTT document."""
    lines: list[str] = ["WEBVTT", ""]
    for index, seg in enumerate(_iter_segments(source), start=1):
        start = seconds_to_vtt_timestamp(seg.start_seconds)
        end = seconds_to_vtt_timestamp(seg.end_seconds)
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines).strip("\n") + "\n"
