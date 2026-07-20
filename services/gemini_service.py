"""Gemini transcription using the official ``google-genai`` SDK.

Flow:
  1. Upload the video via the Files API.
  2. Wait for the file to become ACTIVE.
  3. Ask the configured model for a strict JSON transcript (timestamped
     segments) using ``response_mime_type='application/json'`` plus a response
     schema.
  4. Parse, repair, and validate the JSON into a :class:`Transcript`.

The design is single-video but structured so batch processing can be layered
on later (see ``transcribe`` / ``TranscriptionError`` and the README).

The API key is only ever passed to the SDK client; it is never logged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types

from config import SUPPORTED_VIDEO_TYPES
from models.transcript import Transcript, build_transcript
from utils.files import cleanup_dir
from utils.logging_config import get_logger

logger = get_logger()

ProgressFn = Callable[[str], None]


class TranscriptionError(RuntimeError):
    """Raised for any user-facing transcription failure."""


class QuotaError(TranscriptionError):
    """Raised specifically when the API quota / rate limit is hit (HTTP 429)."""


# The prompt that instructs the model how to transcribe. Kept explicit about
# the required behaviour from the spec (no summarising, no hallucination, etc.).
_SYSTEM_INSTRUCTION = (
    "You are a precise speech-to-text transcription engine for video. "
    "Transcribe exactly what is spoken. Do not summarise, translate, or "
    "rewrite the speaker's words. Preserve punctuation and capitalisation. "
    "Detect the spoken language and report it as an ISO 639-1 code. "
    "Transcribe the ENTIRE media continuously from the first second to the "
    "last. Do NOT skip, omit, or leave out any spoken section — cover the "
    "beginning, the middle, and the end completely, with no unexplained gaps "
    "between spoken lines. "
    "Do not invent words during silence or music. If a section has no speech, "
    "do not emit a segment for it. Break the transcript into short, "
    "readable subtitle segments of at most ~10 words or ~7 seconds each, "
    "suitable for on-screen captions of no more than two lines. Ensure every "
    "segment's end time is after its start time, segments are in chronological "
    "order, and segments never overlap."
)

def _single_prompt(duration_seconds: Optional[float] = None) -> str:
    """Prompt for transcribing a whole media file in one request.

    Passing the known duration ("duration grounding") helps the model anchor
    its timeline to the real clock and reduces cumulative timestamp drift.
    """
    prompt = (
        "Transcribe the speech in this media into timestamped caption segments. "
        "Return ONLY a JSON object matching the provided schema. Timestamps are "
        "in seconds from the start of the media. Ground every timestamp to the "
        "actual audio you hear: the start time must match when each phrase is "
        "spoken. Do not let timestamps drift ahead of or behind the speech as "
        "the media progresses."
    )
    if duration_seconds and duration_seconds > 0:
        prompt += (
            f" The media is approximately {duration_seconds:.1f} seconds long; "
            "the last segment must end at or before that time."
        )
    return prompt


# Prompt for a single short clip in chunked mode. Timestamps are LOCAL to the
# clip (0.0 == start of clip); the caller adds the clip's absolute offset back.
_CHUNK_PROMPT = (
    "This is a short audio clip taken from a longer recording. Transcribe the "
    "speech into timestamped caption segments. Timestamps are in seconds "
    "measured from the START OF THIS CLIP, where 0.0 is the very beginning of "
    "the clip. Return ONLY a JSON object matching the provided schema."
)


def _offset_segments(segments: list, offset: float) -> list[dict]:
    """Shift a chunk's local segment timestamps by its absolute ``offset``."""
    shifted: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg["start_seconds"]) + offset
            end = float(seg["end_seconds"]) + offset
        except (KeyError, TypeError, ValueError):
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        shifted.append({"start_seconds": start, "end_seconds": end, "text": text})
    return shifted

# Response schema (OpenAPI subset understood by the SDK). Using a plain schema
# keeps parsing under our control so we can repair minor issues before strict
# validation.
_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["segments", "full_transcript"],
    properties={
        "language": types.Schema(type=types.Type.STRING, nullable=True),
        "duration_seconds": types.Schema(type=types.Type.NUMBER, nullable=True),
        "segments": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["id", "start_seconds", "end_seconds", "text"],
                properties={
                    "id": types.Schema(type=types.Type.INTEGER),
                    "start_seconds": types.Schema(type=types.Type.NUMBER),
                    "end_seconds": types.Schema(type=types.Type.NUMBER),
                    "text": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "full_transcript": types.Schema(type=types.Type.STRING),
    },
)


def _noop(_: str) -> None:
    return None


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429 or "429" in text or "resource_exhausted" in text or "quota" in text


def _mime_for(path: Path) -> Optional[str]:
    return SUPPORTED_VIDEO_TYPES.get(path.suffix.lower().lstrip("."))


class GeminiTranscriber:
    """Thin wrapper around the google-genai client for transcription."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise TranscriptionError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    # -- introspection ---------------------------------------------------

    def list_models(self) -> list[str]:
        """Return model names your key can use (best-effort)."""
        names: list[str] = []
        try:
            for model in self._client.models.list():
                name = getattr(model, "name", "") or ""
                actions = getattr(model, "supported_actions", None) or []
                if not actions or "generateContent" in actions:
                    names.append(name.replace("models/", ""))
        except Exception as exc:  # pragma: no cover - network dependent
            raise TranscriptionError(f"Could not list models: {exc}") from exc
        return sorted(set(names))

    # -- upload ----------------------------------------------------------

    def _upload_and_wait(
        self,
        video_path: Path,
        progress: ProgressFn,
        poll_timeout: int = 300,
        mime: Optional[str] = None,
    ):
        mime = mime or _mime_for(video_path)
        config = types.UploadFileConfig(mime_type=mime) if mime else None

        progress("Uploading video to Gemini Files API…")
        try:
            uploaded = self._client.files.upload(file=str(video_path), config=config)
        except Exception as exc:
            if _is_quota_error(exc):
                raise QuotaError(
                    "Upload rejected — the API quota may have been reached (HTTP 429)."
                ) from exc
            raise TranscriptionError(f"Video upload failed: {exc}") from exc

        logger.info("Gemini upload completed (file=%s)", uploaded.name)

        # Wait for processing to reach ACTIVE.
        deadline = time.monotonic() + poll_timeout
        while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
            if time.monotonic() > deadline:
                raise TranscriptionError("Timed out waiting for Gemini to process the upload.")
            progress("Gemini is processing the uploaded video…")
            time.sleep(3)
            uploaded = self._client.files.get(name=uploaded.name)

        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state != "ACTIVE":
            raise TranscriptionError(f"Uploaded file is not usable (state={state}).")
        return uploaded

    # -- generate --------------------------------------------------------

    def _generate_json(
        self, uploaded, prompt: str, progress: ProgressFn, max_retries: int = 2
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
        )

        attempt = 0
        while True:
            attempt += 1
            progress("Sending request to Gemini and generating transcript…")
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[uploaded, prompt],
                    config=config,
                )
            except Exception as exc:
                if _is_quota_error(exc):
                    # Do NOT retry a quota error endlessly / duplicate paid calls.
                    raise QuotaError(
                        "The API quota may have been reached (HTTP 429). "
                        "Check your per-minute and per-day limits and try again later."
                    ) from exc
                # Small capped retry for transient errors only.
                if attempt <= max_retries:
                    backoff = 2 * attempt
                    logger.warning(
                        "Transient Gemini error (attempt %s/%s); retrying in %ss",
                        attempt,
                        max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise TranscriptionError(f"Gemini request failed: {exc}") from exc

            text = getattr(response, "text", None)
            if not text:
                raise TranscriptionError("Gemini returned an empty response.")
            return text

    def _transcribe_media(
        self, media_path: Path, prompt: str, progress: ProgressFn, *,
        mime: Optional[str], delete_remote: bool,
    ) -> str:
        """Upload one media file, generate the JSON transcript, clean up."""
        uploaded = self._upload_and_wait(media_path, progress, mime=mime)
        try:
            return self._generate_json(uploaded, prompt, progress)
        finally:
            if delete_remote:
                try:
                    self._client.files.delete(name=uploaded.name)
                except Exception:  # pragma: no cover - cleanup best-effort
                    logger.debug("Could not delete remote file %s", uploaded.name)

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            snippet = raw_text[:200].replace("\n", " ")
            raise TranscriptionError(
                "Gemini returned malformed JSON that could not be parsed. "
                f"First characters: {snippet!r}"
            ) from exc

    # -- public API ------------------------------------------------------

    def transcribe(
        self,
        video_path: str | Path,
        progress_callback: Optional[ProgressFn] = None,
        *,
        delete_remote: bool = True,
        duration_seconds: Optional[float] = None,
        window_seconds: Optional[int] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> Transcript:
        """Transcribe one video and return a validated :class:`Transcript`.

        By default this is a single API request over the whole media, with the
        known ``duration_seconds`` used to reduce timestamp drift.

        When ``window_seconds`` is set (and the media is longer than one
        window), high-accuracy chunked mode is used: the audio is split into
        windows, each transcribed independently, and timestamps are offset back
        to absolute time. This bounds drift to a single window's length at the
        cost of ~one API request per window.
        """
        progress = progress_callback or _noop
        path = Path(video_path)
        if not path.exists():
            raise TranscriptionError(f"Video not found: {path}")

        logger.info("Processing started (model=%s)", self._model)

        if window_seconds and duration_seconds and duration_seconds > window_seconds:
            transcript = self._transcribe_chunked(
                path, progress, window_seconds, duration_seconds,
                ffmpeg_path, delete_remote,
            )
        else:
            raw_text = self._transcribe_media(
                path,
                _single_prompt(duration_seconds),
                progress,
                mime=_mime_for(path),
                delete_remote=delete_remote,
            )
            progress("Validating timestamps…")
            payload = self._parse_json(raw_text)
            if duration_seconds and "duration_seconds" not in payload:
                payload["duration_seconds"] = duration_seconds
            try:
                transcript = build_transcript(payload)
            except Exception as exc:
                raise TranscriptionError(
                    f"The transcript failed validation: {exc}"
                ) from exc

        logger.info(
            "Transcription completed (%d segments, language=%s)",
            len(transcript.segments),
            transcript.language,
        )
        if logger.isEnabledFor(10):  # logging.DEBUG
            logger.debug("Transcript text: %s", transcript.full_transcript)
        return transcript

    def _transcribe_chunked(
        self,
        path: Path,
        progress: ProgressFn,
        window_seconds: int,
        duration_seconds: float,
        ffmpeg_path: Optional[str],
        delete_remote: bool,
    ) -> Transcript:
        """High-accuracy path: window the audio and stitch offset timestamps."""
        import math
        import tempfile

        # Import here to avoid a hard dependency for the single-request path.
        from services.video_service import (
            FFmpegError,
            check_ffmpeg,
            extract_audio_segment,
        )

        ff = ffmpeg_path or check_ffmpeg().ffmpeg_path
        if not ff:
            raise TranscriptionError(
                "High-accuracy mode needs FFmpeg to slice the audio, but FFmpeg "
                "was not found."
            )

        n_chunks = math.ceil(duration_seconds / window_seconds)
        all_segments: list[dict] = []
        language: Optional[str] = None
        work_dir = Path(tempfile.mkdtemp(prefix="chunks_"))
        try:
            for i in range(n_chunks):
                start = i * window_seconds
                length = min(float(window_seconds), duration_seconds - start)
                progress(
                    f"High-accuracy mode: transcribing chunk {i + 1}/{n_chunks} "
                    f"({start:.0f}–{start + length:.0f}s)…"
                )
                chunk_path = work_dir / f"chunk_{i:03d}.m4a"
                try:
                    extract_audio_segment(ff, path, chunk_path, start, length)
                except FFmpegError as exc:
                    raise TranscriptionError(
                        f"Could not extract audio chunk {i + 1}: {exc}"
                    ) from exc

                raw_text = self._transcribe_media(
                    chunk_path,
                    _CHUNK_PROMPT,
                    _noop,
                    mime="audio/mp4",
                    delete_remote=delete_remote,
                )
                payload = self._parse_json(raw_text)
                if language is None:
                    language = payload.get("language")
                all_segments.extend(
                    _offset_segments(payload.get("segments", []), float(start))
                )
        finally:
            cleanup_dir(work_dir)

        if not all_segments:
            raise TranscriptionError("No speech segments were produced from any chunk.")

        try:
            return build_transcript(
                {
                    "language": language,
                    "duration_seconds": duration_seconds,
                    "segments": all_segments,
                    "full_transcript": "",
                }
            )
        except Exception as exc:
            raise TranscriptionError(f"The transcript failed validation: {exc}") from exc
