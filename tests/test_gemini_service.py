"""Tests for the Gemini service using a fully mocked SDK client.

These tests never make a real API request. We monkeypatch the client on an
already-constructed ``GeminiTranscriber`` so no network or key is needed.
"""

import json
import types as pytypes

import pytest

from services.gemini_service import (
    GeminiTranscriber,
    QuotaError,
    TranscriptionError,
    _is_quota_error,
    _offset_segments,
    _single_prompt,
)


def test_single_prompt_includes_duration_grounding():
    assert "seconds long" in _single_prompt(120.0)
    assert "120.0" in _single_prompt(120.0)
    # Without a duration, no length claim is made.
    assert "seconds long" not in _single_prompt(None)
    # Always instructs against drift.
    assert "drift" in _single_prompt(None)


def test_offset_segments_shifts_and_filters():
    segs = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "text": "a"},
        {"start_seconds": 2.0, "end_seconds": 4.0, "text": "  "},  # blank -> dropped
        {"start_seconds": 4.0, "end_seconds": 5.0, "text": "b"},
        {"start_seconds": "x", "end_seconds": 1.0, "text": "bad"},  # unparsable
    ]
    out = _offset_segments(segs, 45.0)
    assert [s["text"] for s in out] == ["a", "b"]
    assert out[0]["start_seconds"] == 45.0
    assert out[1]["end_seconds"] == 50.0


def _make_transcriber() -> GeminiTranscriber:
    t = GeminiTranscriber.__new__(GeminiTranscriber)  # bypass real client creation
    t._model = "test-model"  # type: ignore[attr-defined]
    return t


class _FakeState:
    def __init__(self, name):
        self.name = name


class _FakeFile:
    def __init__(self, name="files/abc", state="ACTIVE"):
        self.name = name
        self.state = _FakeState(state)
        self.uri = "https://example/abc"


class _FakeFiles:
    def __init__(self, file):
        self._file = file
        self.deleted = False

    def upload(self, file, config=None):
        return self._file

    def get(self, name):
        return self._file

    def delete(self, name):
        self.deleted = True


class _FakeModels:
    def __init__(self, response):
        self._response = response

    def generate_content(self, model, contents, config):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _install_client(transcriber, *, file, response):
    client = pytypes.SimpleNamespace()
    client.files = _FakeFiles(file)
    client.models = _FakeModels(response)
    transcriber._client = client  # type: ignore[attr-defined]
    return client


def test_is_quota_error_detects_429():
    assert _is_quota_error(Exception("Error 429 RESOURCE_EXHAUSTED"))
    assert _is_quota_error(Exception("quota exceeded"))
    assert not _is_quota_error(Exception("some other error"))


def test_transcribe_happy_path(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    payload = {
        "language": "en",
        "duration_seconds": 6.0,
        "segments": [
            {"id": 1, "start_seconds": 0.0, "end_seconds": 3.0, "text": "Hello."},
            {"id": 2, "start_seconds": 3.0, "end_seconds": 6.0, "text": "World."},
        ],
        "full_transcript": "Hello. World.",
    }
    response = pytypes.SimpleNamespace(text=json.dumps(payload))

    t = _make_transcriber()
    client = _install_client(t, file=_FakeFile(), response=response)

    transcript = t.transcribe(video)
    assert transcript.language == "en"
    assert len(transcript.segments) == 2
    assert client.files.deleted is True  # remote cleanup happened


def test_transcribe_malformed_json_raises(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    response = pytypes.SimpleNamespace(text="{not valid json")

    t = _make_transcriber()
    _install_client(t, file=_FakeFile(), response=response)

    with pytest.raises(TranscriptionError) as exc:
        t.transcribe(video)
    assert "malformed json" in str(exc.value).lower()


def test_transcribe_quota_error_raises_quota(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    t = _make_transcriber()
    _install_client(
        t, file=_FakeFile(), response=Exception("429 RESOURCE_EXHAUSTED")
    )

    with pytest.raises(QuotaError):
        t.transcribe(video)


def test_transcribe_missing_file_raises(tmp_path):
    t = _make_transcriber()
    _install_client(t, file=_FakeFile(), response=pytypes.SimpleNamespace(text="{}"))
    with pytest.raises(TranscriptionError):
        t.transcribe(tmp_path / "does_not_exist.mp4")
