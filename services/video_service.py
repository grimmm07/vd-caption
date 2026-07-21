"""FFmpeg/FFprobe integration: detection, probing, and caption rendering.

All FFmpeg invocations use an explicit argument list passed to ``subprocess``
(never a shell string), so untrusted filenames can never be interpreted as
shell syntax.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from utils.logging_config import get_logger

logger = get_logger()


def _candidate_dirs() -> list[Path]:
    """Common install locations to search when a tool is not on PATH.

    This lets the app find FFmpeg right after a `winget`/`choco`/`brew` install
    without the user having to restart their shell so PATH refreshes.
    """
    dirs: list[Path] = []
    # Explicit override wins.
    override = os.getenv("FFMPEG_DIR")
    if override:
        dirs.append(Path(override))
    local = os.getenv("LOCALAPPDATA")
    if local:
        # winget "Links" shims + the actual Gyan.FFmpeg package bin.
        dirs.append(Path(local) / "Microsoft" / "WinGet" / "Links")
        pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if pkgs.is_dir():
            dirs.extend(p for p in pkgs.glob("Gyan.FFmpeg*/**/bin") if p.is_dir())
    dirs += [
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ]
    return dirs


def _locate(tool: str) -> Optional[str]:
    """Find an executable on PATH, or in known install locations as a fallback."""
    found = shutil.which(tool)
    if found:
        return found
    exe = f"{tool}.exe" if os.name == "nt" else tool
    for directory in _candidate_dirs():
        candidate = directory / exe
        if candidate.is_file():
            return str(candidate)
    return None


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg/FFprobe invocation fails."""


@dataclass
class FFmpegStatus:
    ffmpeg_path: Optional[str]
    ffprobe_path: Optional[str]

    @property
    def ok(self) -> bool:
        return bool(self.ffmpeg_path and self.ffprobe_path)

    @property
    def message(self) -> str:
        if self.ok:
            return "FFmpeg and FFprobe were found."
        missing = []
        if not self.ffmpeg_path:
            missing.append("ffmpeg")
        if not self.ffprobe_path:
            missing.append("ffprobe")
        return (
            "Missing required tool(s): "
            + ", ".join(missing)
            + ". Install FFmpeg (see the README). If you just installed it, "
            "restart this app, or set FFMPEG_DIR in .env to its bin folder."
        )


@dataclass
class VideoMetadata:
    duration_seconds: Optional[float]
    width: Optional[int]
    height: Optional[int]
    video_codec: Optional[str]
    audio_codec: Optional[str]
    has_audio: bool

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"


def check_ffmpeg() -> FFmpegStatus:
    """Locate ffmpeg and ffprobe on PATH, or in common install locations."""
    return FFmpegStatus(
        ffmpeg_path=_locate("ffmpeg"),
        ffprobe_path=_locate("ffprobe"),
    )


def extract_audio_segment(
    ffmpeg_path: str,
    input_path: str | Path,
    output_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Extract a mono 16 kHz AAC audio slice ``[start, start+duration)``.

    Used by high-accuracy chunked transcription. ``-ss``/``-t`` are placed
    *after* ``-i`` for sample-accurate seeking, so the offset we add back to the
    returned timestamps matches the real media clock.
    """
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-t",
        f"{max(0.0, duration_seconds):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        str(output_path),
    ]
    run_ffmpeg(cmd)
    return Path(output_path)


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Return duration/resolution/codec info using ffprobe."""
    status = check_ffmpeg()
    if not status.ffprobe_path:
        raise FFmpegError("ffprobe is not installed or not on PATH")

    cmd = [
        status.ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=120
        )
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffprobe failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("ffprobe timed out") from exc

    data = json.loads(completed.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    duration: Optional[float] = None
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None

    return VideoMetadata(
        duration_seconds=duration,
        width=int(video_stream["width"]) if video_stream and video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream and video_stream.get("height") else None,
        video_codec=video_stream.get("codec_name") if video_stream else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        has_audio=audio_stream is not None,
    )


def _escape_subtitles_path(srt_path: Path) -> str:
    """Escape a path for use inside the ffmpeg ``subtitles=`` filter.

    The subtitles filter parses its argument, so on Windows the drive-letter
    colon and backslashes must be escaped. We normalise to forward slashes and
    escape the colon and single quotes.
    """
    text = str(srt_path).replace("\\", "/")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    return text


def font_size_for_height(height: Optional[int]) -> int:
    """Choose a readable subtitle font size for the given video height.

    Sized conservatively (~2.4% of frame height) so captions sit in the lower
    portion of the frame instead of dominating it. Callers can override.
    """
    if not height:
        return 22
    return max(14, min(30, round(height * 0.024)))


def build_burn_command(
    ffmpeg_path: str,
    input_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    *,
    font_size: int = 24,
    has_audio: bool = True,
) -> list[str]:
    """Build the argv for burning subtitles into the video.

    Styling: white text, black outline, semi-transparent box, bottom-centre,
    with a safe bottom margin and max two lines (enforced by short segments).
    Video is re-encoded to H.264; audio is copied when present, else dropped.
    """
    style = (
        f"FontName=Arial,FontSize={int(font_size)},"
        "PrimaryColour=&H00FFFFFF,"  # white text (AABBGGRR)
        "OutlineColour=&H00000000,"  # black outline
        "BackColour=&H80000000,"  # semi-transparent black box
        "BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=30"  # bottom-centre, 30px margin
    )
    subtitles_arg = f"subtitles='{_escape_subtitles_path(Path(srt_path))}':force_style='{style}'"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        subtitles_arg,
        "-c:v",
        "libx264",
        "-preset",
        # "veryfast" renders several times faster than "medium" with negligible
        # visible-quality loss for captioned clips — important so a 1.5–3 min
        # video finishes in seconds rather than feeling stuck.
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(output_path)]
    return cmd


def build_soft_subtitle_command(
    ffmpeg_path: str,
    input_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    *,
    language: str = "eng",
) -> list[str]:
    """Build the argv for muxing a *selectable* subtitle track (mov_text).

    The track is given a language, a human-readable title, and marked as the
    default subtitle stream. Those hints help players enumerate and expose the
    track (Windows Media Player in particular only lists tracks that carry this
    metadata). Viewers can still toggle it off.
    """
    return [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(srt_path),
        "-map",
        "0",
        "-map",
        "1",
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        f"language={language}",
        "-metadata:s:s:0",
        "title=Captions",
        "-disposition:s:0",
        "default",
        str(output_path),
    ]


def _parse_progress_seconds(line: str) -> Optional[float]:
    """Parse ``out_time_ms=`` (microseconds) lines from ``-progress pipe:1``."""
    if line.startswith("out_time_ms="):
        try:
            return int(line.split("=", 1)[1]) / 1_000_000.0
        except (ValueError, IndexError):
            return None
    return None


def run_ffmpeg(
    cmd: list[str],
    *,
    total_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout: int = 1800,
) -> None:
    """Run an ffmpeg command, optionally reporting 0..1 progress.

    Progress is read from ``-progress pipe:1`` on **stdout**, while ffmpeg's
    verbose logging (the subtitles/libass filter is chatty) is redirected to a
    temporary **file**. Sending stderr to a file instead of a pipe is essential:
    a stderr pipe can fill its OS buffer and deadlock ffmpeg while we are only
    draining stdout. A file has no such limit.

    Raises :class:`FFmpegError` with the captured log tail on failure.
    """
    # Insert progress reporting right after the executable.
    full_cmd = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    logger.info("FFmpeg started")
    logger.debug("FFmpeg argv: %s", full_cmd)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errfile:
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=errfile,  # -> file, never blocks
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                seconds = _parse_progress_seconds(line.strip())
                if seconds is not None and progress_callback and total_seconds:
                    progress_callback(max(0.0, min(1.0, seconds / total_seconds)))
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise FFmpegError("FFmpeg timed out") from exc
        finally:
            if process.stdout is not None:
                process.stdout.close()

        if process.returncode != 0:
            errfile.seek(0)
            tail = "\n".join(errfile.read().strip().splitlines()[-20:])
            logger.error("Processing failed: FFmpeg exited with %s", process.returncode)
            raise FFmpegError(f"FFmpeg failed (exit {process.returncode}):\n{tail}")

    if progress_callback:
        progress_callback(1.0)
    logger.info("FFmpeg completed")


def burn_captions(
    input_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    *,
    metadata: Optional[VideoMetadata] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    font_size: Optional[int] = None,
) -> Path:
    """Burn subtitles into a new MP4. The input file is never modified.

    ``font_size`` overrides the automatic size derived from the video height.
    """
    status = check_ffmpeg()
    if not status.ffmpeg_path:
        raise FFmpegError("ffmpeg is not installed or not on PATH")

    has_audio = metadata.has_audio if metadata else True
    height = metadata.height if metadata else None
    effective_size = font_size if font_size else font_size_for_height(height)
    cmd = build_burn_command(
        status.ffmpeg_path,
        input_path,
        srt_path,
        output_path,
        font_size=effective_size,
        has_audio=has_audio,
    )
    run_ffmpeg(
        cmd,
        total_seconds=metadata.duration_seconds if metadata else None,
        progress_callback=progress_callback,
    )
    return Path(output_path)


def mux_soft_subtitles(
    input_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    *,
    language: str = "eng",
) -> Path:
    """Create an MP4 with a selectable (soft) subtitle track."""
    status = check_ffmpeg()
    if not status.ffmpeg_path:
        raise FFmpegError("ffmpeg is not installed or not on PATH")
    cmd = build_soft_subtitle_command(
        status.ffmpeg_path, input_path, srt_path, output_path, language=language
    )
    run_ffmpeg(cmd)
    return Path(output_path)


# ISO 639-1 (2-letter) -> ISO 639-2/T (3-letter) for common languages, since
# MP4 subtitle tracks expect the 3-letter code.
_LANG_2_TO_3 = {
    "en": "eng", "fr": "fre", "es": "spa", "de": "ger", "it": "ita",
    "pt": "por", "nl": "dut", "ar": "ara", "ru": "rus", "zh": "chi",
    "ja": "jpn", "ko": "kor", "hi": "hin", "tr": "tur", "pl": "pol",
}


def iso3_language(code: str | None) -> str:
    """Best-effort map a detected language code to a 3-letter MP4 code."""
    if not code:
        return "und"
    code = code.strip().lower()
    if len(code) == 3:
        return code
    return _LANG_2_TO_3.get(code[:2], "und")
