"""Streamlit UI for the video caption prototype.

Flow: upload → inspect → generate transcript (Gemini) → edit → generate
captioned video (FFmpeg) → preview → download. Plus a batch cost estimator.
"""

from __future__ import annotations

import base64
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# On Streamlit Community Cloud / HF Spaces, secrets are exposed via st.secrets
# rather than environment variables. Bridge them into the environment so the
# existing os.getenv-based config picks them up. Must run before get_settings().
try:
    for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "MAX_UPLOAD_MB", "DEBUG"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # no secrets file present (e.g. local run) — that's fine
    pass

from config import SUPPORTED_VIDEO_TYPES, get_settings
from models.transcript import Segment, Transcript
from services.gemini_service import GeminiTranscriber, QuotaError, TranscriptionError
from services.subtitle_service import generate_srt, generate_vtt
from services.video_service import (
    FFmpegError,
    burn_captions,
    check_ffmpeg,
    font_size_for_height,
    iso3_language,
    mux_soft_subtitles,
    probe_video,
)
from utils.files import cleanup_dir, new_work_dir, sanitize_filename
from utils.logging_config import get_logger
from utils.synced_player import (
    INLINE_PLAYABLE,
    MAX_INLINE_BYTES,
    build_synced_player_html,
    preview_layout_heights,
    segments_to_cues,
)

logger = get_logger()
settings = get_settings()

st.set_page_config(page_title="Video Caption Prototype", page_icon="🎬", layout="centered")

# Light visual polish (kept minimal and version-tolerant).
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 900px; }
      h1 { font-weight: 800; letter-spacing: -0.4px; }
      h2 { margin-top: 1.4rem; padding-bottom: .35rem;
           border-bottom: 1px solid rgba(128,128,128,.22); }
      h3 { margin-top: .6rem; }
      [data-testid="stMetricValue"] { font-size: 1.05rem; }
      [data-testid="stMetricLabel"] { opacity: .7; }
      .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
      [data-testid="stFileUploaderDropzone"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "work_dir": None,
        "video_path": None,
        "video_name": None,
        "video_size": None,
        "metadata": None,
        "segments_df": None,  # editable pandas DataFrame
        "language": None,
        "outputs": {},  # label -> path
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_processing() -> None:
    if st.session_state.get("work_dir"):
        cleanup_dir(st.session_state["work_dir"])
    for key in (
        "work_dir",
        "video_path",
        "video_name",
        "video_size",
        "metadata",
        "segments_df",
        "language",
    ):
        st.session_state[key] = None
    st.session_state["outputs"] = {}


def _segments_from_df(df: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        segments.append(
            Segment(
                id=i,
                start_seconds=float(row.start_seconds),
                end_seconds=float(row.end_seconds),
                text=str(row.text),
            )
        )
    return segments


def _validated_transcript() -> Transcript:
    """Rebuild a validated Transcript from the (possibly edited) table."""
    df = st.session_state["segments_df"]
    segments = _segments_from_df(df)
    return Transcript(
        language=st.session_state.get("language"),
        duration_seconds=(
            st.session_state["metadata"].duration_seconds
            if st.session_state.get("metadata")
            else None
        ),
        segments=segments,
    ).rebuild_full_transcript()


def _render_synced_preview(video_path, edited_df) -> None:
    """Show the clean video with captions underneath, synced to playback.

    Falls back to a plain player + static caption list when the video can't be
    embedded inline (unsupported container or too large for a data URI).
    """
    if not video_path:
        return
    path = Path(video_path)
    ext = path.suffix.lower().lstrip(".")
    cues = segments_to_cues(edited_df.to_dict("records"))

    # Diagnostic: if the preview shows fewer lines than the table, say so and
    # why, so a mismatch is never silent.
    table_rows = len(edited_df)
    if len(cues) < table_rows:
        st.warning(
            f"Preview is showing {len(cues)} of {table_rows} table rows — "
            f"{table_rows - len(cues)} were skipped because their caption text "
            "is blank or their start/end time is not a number. Fix those rows "
            "in the table above."
        )
    else:
        st.caption(f"Preview: {len(cues)} caption lines (matches the table).")

    if not cues:
        st.info("Add at least one caption to preview.")
        return

    size = path.stat().st_size if path.exists() else 0
    if ext not in INLINE_PLAYABLE or size > MAX_INLINE_BYTES:
        # Fallback: Streamlit's own player + a read-only caption list.
        reason = (
            f"{ext.upper()} isn't supported for inline sync"
            if ext not in INLINE_PLAYABLE
            else f"video is {size / 1e6:.0f} MB (over the inline limit)"
        )
        st.info(
            f"Showing a plain preview because {reason}. The captions below are "
            "static (not click-to-seek). Burn them in, or use an MP4/WebM for "
            "the synced view."
        )
        st.video(str(path))
        st.dataframe(
            [{"time": f"{c['start']:.1f}s", "caption": c["text"]} for c in cues],
            use_container_width=True,
            hide_index=True,
        )
        return

    # Use the real video aspect ratio so the picture fills the column (no black
    # bars) and lines up with the caption panels below it.
    meta = st.session_state.get("metadata")
    if meta and meta.width and meta.height:
        aspect = meta.width / meta.height
        aspect_css = f"{meta.width} / {meta.height}"
    else:
        aspect, aspect_css = 16 / 9, "16 / 9"

    list_h, total_h = preview_layout_heights(len(cues), aspect)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    html = build_synced_player_html(
        b64, INLINE_PLAYABLE[ext], cues, list_height_px=list_h, aspect_ratio=aspect_css
    )
    components.html(html, height=total_h, scrolling=True)


_init_state()

# --------------------------------------------------------------------------
# Header + environment checks
# --------------------------------------------------------------------------
st.title("🎬 Video Caption Prototype")
st.caption(
    "Transcribe a video with Google Gemini, edit the captions, and burn them "
    "into the video with FFmpeg."
)

ffmpeg_status = check_ffmpeg()

with st.sidebar:
    st.header("Environment")
    st.write(f"**Model:** `{settings.gemini_model}`")
    st.write(
        f"**Gemini API key:** {'✅ set' if settings.has_api_key else '❌ not set'}"
    )
    st.write(f"**FFmpeg:** {'✅' if ffmpeg_status.ffmpeg_path else '❌'}")
    st.write(f"**FFprobe:** {'✅' if ffmpeg_status.ffprobe_path else '❌'}")
    st.write(f"**Max upload:** {settings.max_upload_mb} MB")
    if not ffmpeg_status.ok:
        st.error(ffmpeg_status.message)

    if settings.has_api_key and st.button("List models my key can use"):
        try:
            names = GeminiTranscriber(
                settings.gemini_api_key, settings.gemini_model
            ).list_models()
            st.success("Models supporting generateContent:")
            st.code("\n".join(names) or "(none returned)")
        except TranscriptionError as exc:
            st.error(str(exc))

    st.divider()
    st.caption(
        "⚠️ Uploaded video is sent to Google for processing when you generate a "
        "transcript. See the README's privacy section."
    )

if not settings.has_api_key:
    st.warning(
        "No `GEMINI_API_KEY` found. Copy `.env.example` to `.env` and add your "
        "key to enable transcription. You can still explore the UI and the "
        "cost estimator below."
    )


# --------------------------------------------------------------------------
# Step 1 — Upload
# --------------------------------------------------------------------------
st.header("1 · Upload a video")
uploaded = st.file_uploader(
    "Choose a video file",
    type=list(SUPPORTED_VIDEO_TYPES.keys()),
    accept_multiple_files=False,
)

if uploaded is not None:
    size = getattr(uploaded, "size", 0)
    if size > settings.max_upload_bytes:
        st.error(
            f"File is {size / 1e6:.1f} MB, larger than the "
            f"{settings.max_upload_mb} MB limit."
        )
    elif st.session_state["video_name"] != uploaded.name:
        # New file: reset any previous processing and persist to a work dir.
        _reset_processing()
        work_dir = new_work_dir(settings.temp_dir)
        safe_name = sanitize_filename(uploaded.name)
        dest = work_dir / safe_name
        dest.write_bytes(uploaded.getbuffer())

        st.session_state.update(
            work_dir=str(work_dir),
            video_path=str(dest),
            video_name=uploaded.name,
            video_size=size,
        )
        try:
            st.session_state["metadata"] = probe_video(dest)
        except FFmpegError as exc:
            st.session_state["metadata"] = None
            logger.warning("Could not probe video: %s", exc)

if st.session_state.get("video_path"):
    meta = st.session_state.get("metadata")
    cols = st.columns(4)
    cols[0].metric("File name", st.session_state["video_name"])
    cols[1].metric("Size", f"{st.session_state['video_size'] / 1e6:.1f} MB")
    if meta and meta.duration_seconds:
        cols[2].metric("Duration", f"{meta.duration_seconds:.1f} s")
    else:
        cols[2].metric("Duration", "unknown")
    cols[3].metric("Resolution", meta.resolution if meta else "unknown")

    if meta:
        st.caption(
            f"Video codec: `{meta.video_codec}` · Audio codec: "
            f"`{meta.audio_codec or 'none'}`"
        )
    st.video(st.session_state["video_path"])


# --------------------------------------------------------------------------
# Step 2 — Generate transcript
# --------------------------------------------------------------------------
st.header("2 · Generate transcript")
st.caption("Transcription is only triggered by this button — never automatically.")

can_transcribe = bool(st.session_state.get("video_path")) and settings.has_api_key

_meta = st.session_state.get("metadata")
_duration = _meta.duration_seconds if _meta else None

high_accuracy = st.checkbox(
    "🎯 High-accuracy sync (recommended for drift)",
    help=(
        "Splits the audio into short windows and transcribes each separately so "
        "caption timing can't drift over the video. Uses about one extra API "
        "request per window (see the estimate below) instead of a single "
        "request. Needs FFmpeg."
    ),
    disabled=not ffmpeg_status.ok,
)
_window = 45  # seconds per window in high-accuracy mode
if high_accuracy and _duration:
    _n = math.ceil(_duration / _window)
    st.caption(
        f"High-accuracy mode will make ~{_n} API request(s) "
        f"({_window}s windows) instead of 1. Mind your daily quota."
    )

if st.button("🎙️ Generate transcript", disabled=not can_transcribe, type="primary"):
    status_box = st.status("Starting…", expanded=True)

    def _progress(message: str) -> None:
        status_box.update(label=message)
        status_box.write(message)

    try:
        transcriber = GeminiTranscriber(settings.gemini_api_key, settings.gemini_model)
        transcript = transcriber.transcribe(
            st.session_state["video_path"],
            progress_callback=_progress,
            duration_seconds=_duration,
            window_seconds=_window if high_accuracy else None,
            ffmpeg_path=ffmpeg_status.ffmpeg_path,
        )
        st.session_state["language"] = transcript.language
        st.session_state["segments_df"] = pd.DataFrame(
            [
                {
                    "start_seconds": s.start_seconds,
                    "end_seconds": s.end_seconds,
                    "text": s.text,
                }
                for s in transcript.segments
            ]
        )
        # New transcript invalidates any previously rendered outputs.
        st.session_state["outputs"] = {}
        status_box.update(label="Transcript ready.", state="complete")
    except QuotaError as exc:
        status_box.update(label="Quota error", state="error")
        st.error(f"🚦 {exc}")
    except TranscriptionError as exc:
        status_box.update(label="Transcription failed", state="error")
        st.error(f"❌ {exc}")
    except Exception as exc:  # unexpected — surface, don't swallow
        status_box.update(label="Unexpected error", state="error")
        logger.exception("Unexpected transcription failure")
        st.error(f"❌ Unexpected error: {exc}")


# --------------------------------------------------------------------------
# Step 3 — Edit transcript
# --------------------------------------------------------------------------
if st.session_state.get("segments_df") is not None:
    st.header("3 · Review & edit captions")
    if st.session_state.get("language"):
        st.caption(f"Detected language: **{st.session_state['language']}**")
    st.caption("Edit the caption text (and timestamps if needed) before rendering.")

    edited = st.data_editor(
        st.session_state["segments_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "start_seconds": st.column_config.NumberColumn(
                "Start (s)", min_value=0.0, step=0.1, format="%.3f"
            ),
            "end_seconds": st.column_config.NumberColumn(
                "End (s)", min_value=0.0, step=0.1, format="%.3f"
            ),
            "text": st.column_config.TextColumn("Caption text", width="large"),
        },
        key="editor",
    )
    st.session_state["segments_df"] = edited

    # Live validation preview so the user sees problems before rendering.
    try:
        transcript = _validated_transcript()
        st.success(f"{len(transcript.segments)} valid segments.")
        valid = True
    except Exception as exc:
        st.warning(f"Fix before generating files: {exc}")
        valid = False

    # ----------------------------------------------------------------------
    # Synced preview — captions shown BELOW the (clean) video, following play.
    # ----------------------------------------------------------------------
    st.subheader("▶️ Preview with synced captions")
    st.caption(
        "The original video plays clean; the caption appears below it and "
        "follows playback. Reflects your edits above. Click a line to jump."
    )
    _render_synced_preview(st.session_state.get("video_path"), edited)

    # ----------------------------------------------------------------------
    # Step 4 — Subtitle files + captioned video
    # ----------------------------------------------------------------------
    st.header("4 · Generate subtitle files & captioned video")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("📝 Generate SRT & WebVTT", disabled=not valid):
            transcript = _validated_transcript()
            work = Path(st.session_state["work_dir"])
            srt_path = work / "captions.srt"
            vtt_path = work / "captions.vtt"
            srt_path.write_text(generate_srt(transcript), encoding="utf-8")
            vtt_path.write_text(generate_vtt(transcript), encoding="utf-8")
            (work / "transcript.json").write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )
            st.session_state["outputs"].update(
                {
                    "srt": str(srt_path),
                    "vtt": str(vtt_path),
                    "json": str(work / "transcript.json"),
                }
            )
            logger.info("Subtitle file generated (SRT + WebVTT + JSON)")
            st.success("Subtitle files generated.")

    with col_b:
        render_disabled = not (valid and ffmpeg_status.ok)
        _auto_size = font_size_for_height(_meta.height if _meta else None)

        _BURNED = "Burned-in (permanent)"
        _SOFT = "Selectable — viewer can turn on/off"
        _BOTH = "Both"
        caption_type = st.radio(
            "Captioned video type",
            [_BURNED, _SOFT, _BOTH],
            help=(
                "Burned-in: captions are permanently drawn into the picture. "
                "Selectable: captions are a separate subtitle track the viewer "
                "can turn on or off in players that support it (VLC, QuickTime, "
                "most TVs/phones). Note: some in-browser players don't show the "
                "toggle — download the file to add/remove captions there."
            ),
        )
        want_burned = caption_type in (_BURNED, _BOTH)
        want_soft = caption_type in (_SOFT, _BOTH)

        if want_burned:
            caption_size = st.slider(
                "Caption font size",
                min_value=10,
                max_value=48,
                value=int(_auto_size),
                help="Auto-sized for this video's height; lower it if captions "
                "cover too much of the frame.",
            )
        else:
            caption_size = int(_auto_size)

        if st.button("🎞️ Generate captioned video", disabled=render_disabled):
            transcript = _validated_transcript()
            work = Path(st.session_state["work_dir"])
            srt_path = work / "captions.srt"
            srt_path.write_text(generate_srt(transcript), encoding="utf-8")
            vtt_path = work / "captions.vtt"
            vtt_path.write_text(generate_vtt(transcript), encoding="utf-8")
            json_path = work / "transcript.json"
            json_path.write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )
            st.session_state["outputs"].update(
                {"srt": str(srt_path), "vtt": str(vtt_path), "json": str(json_path)}
            )
            # Drop any stale rendered videos so Step 5 reflects this run only.
            st.session_state["outputs"].pop("burned", None)
            st.session_state["outputs"].pop("soft", None)

            try:
                if want_burned:
                    out_burned = work / "captioned.mp4"
                    bar = st.progress(0.0, text="Rendering burned-in captions…")
                    burn_captions(
                        st.session_state["video_path"],
                        srt_path,
                        out_burned,
                        metadata=st.session_state.get("metadata"),
                        font_size=caption_size,
                        progress_callback=lambda p: bar.progress(
                            p, text=f"Rendering… {p * 100:.0f}%"
                        ),
                    )
                    st.session_state["outputs"]["burned"] = str(out_burned)

                if want_soft:
                    out_soft = work / "captioned_soft.mp4"
                    with st.spinner("Muxing selectable subtitle track…"):
                        mux_soft_subtitles(
                            st.session_state["video_path"],
                            srt_path,
                            out_soft,
                            language=iso3_language(st.session_state.get("language")),
                        )
                    st.session_state["outputs"]["soft"] = str(out_soft)

                st.success("Captioned video ready.")
            except FFmpegError as exc:
                st.error(f"❌ FFmpeg failed: {exc}")


# --------------------------------------------------------------------------
# Step 5 — Preview + downloads
# --------------------------------------------------------------------------
outputs = st.session_state.get("outputs", {})
if outputs:
    st.header("5 · Preview & download")

    # Preview the burned-in video if present (captions are visible in-browser);
    # otherwise preview the selectable-track video.
    if outputs.get("burned"):
        st.video(outputs["burned"])
    elif outputs.get("soft"):
        st.video(outputs["soft"])
        st.info(
            "This file has a **selectable** subtitle track (captions default to "
            "ON, and the viewer can turn them off). Player support varies:\n\n"
            "- **VLC / QuickTime / phones / smart TVs** — show the track under "
            "their Subtitles menu.\n"
            "- **Windows Media Player** has weak embedded-subtitle support and "
            "may not list it. If so, use **Subtitles → Choose subtitle file** "
            "and pick the **`captions.srt`** below, or open the video in VLC.\n"
            "- **In-browser players (including this preview)** usually don't "
            "show a caption toggle — download the file to use it."
        )

    dl_cols = st.columns(4)
    if outputs.get("burned"):
        with dl_cols[0]:
            st.download_button(
                "⬇️ Video (burned-in)",
                data=Path(outputs["burned"]).read_bytes(),
                file_name="captioned.mp4",
                mime="video/mp4",
            )
    if outputs.get("srt"):
        with dl_cols[1]:
            st.download_button(
                "⬇️ SRT",
                data=Path(outputs["srt"]).read_text(encoding="utf-8"),
                file_name="captions.srt",
                mime="application/x-subrip",
            )
    if outputs.get("vtt"):
        with dl_cols[2]:
            st.download_button(
                "⬇️ WebVTT",
                data=Path(outputs["vtt"]).read_text(encoding="utf-8"),
                file_name="captions.vtt",
                mime="text/vtt",
            )
    if outputs.get("json"):
        with dl_cols[3]:
            st.download_button(
                "⬇️ Transcript JSON",
                data=Path(outputs["json"]).read_text(encoding="utf-8"),
                file_name="transcript.json",
                mime="application/json",
            )
    if outputs.get("soft"):
        st.download_button(
            "⬇️ Video with selectable captions (viewer can add/remove)",
            data=Path(outputs["soft"]).read_bytes(),
            file_name="captioned_selectable.mp4",
            mime="video/mp4",
        )


# --------------------------------------------------------------------------
# Batch cost & quota estimator
# --------------------------------------------------------------------------
st.divider()
st.header("📊 Batch cost & quota estimator")
st.caption(
    "Plan the full project. Prices are **not** hard-coded — enter the current "
    "Gemini price yourself from official pricing docs."
)

e1, e2, e3 = st.columns(3)
num_videos = e1.number_input("Number of videos", min_value=1, value=450, step=1)
min_minutes = e2.number_input(
    "Min avg duration (minutes)", min_value=0.1, value=1.5, step=0.1
)
max_minutes = e3.number_input(
    "Max avg duration (minutes)", min_value=0.1, value=3.0, step=0.1
)

f1, f2 = st.columns(2)
price_per_min = f1.number_input(
    "Price per minute (your currency)", min_value=0.0, value=0.0, step=0.01,
    help="Enter the verified current Gemini price per minute of video.",
)
requests_per_day = f2.number_input(
    "Requests allowed per day", min_value=1, value=20, step=1
)

min_total_min = num_videos * min_minutes
max_total_min = num_videos * max_minutes
min_days = math.ceil(num_videos / requests_per_day)

g = st.columns(3)
g[0].metric("Min total duration", f"{min_total_min:,.0f} min")
g[0].metric("Max total duration", f"{max_total_min:,.0f} min")
g[1].metric("Min estimated cost", f"{min_total_min * price_per_min:,.2f}")
g[1].metric("Max estimated cost", f"{max_total_min * price_per_min:,.2f}")
g[2].metric("Estimated API requests", f"{num_videos:,} (1 per video)")
g[2].metric(f"Min days @ {requests_per_day}/day", f"{min_days:,}")

st.caption(
    f"With {num_videos} videos and {requests_per_day} requests/day (one request "
    f"per video), the mathematical minimum is **{min_days} days**."
)