"""Build a self-contained HTML player that shows captions BELOW the video and
keeps them in sync with playback.

This renders inside a Streamlit ``components.html`` iframe. The video is
embedded as a base64 ``data:`` URI so the iframe can play it without any server
route. A ``timeupdate`` listener highlights the active caption, shows it in a
large panel under the video, and auto-scrolls the transcript — i.e. the
captions are "aligned and going with the video", in the UI rather than burned
into the file.
"""

from __future__ import annotations

import json
from typing import Iterable

# Formats HTML5 <video> can reliably play inline. Others fall back to st.video.
INLINE_PLAYABLE = {"mp4": "video/mp4", "webm": "video/webm", "ogg": "video/ogg"}

# Cap the base64 payload so the iframe stays responsive.
MAX_INLINE_BYTES = 40 * 1024 * 1024

# The player is a centred column of this max width; the video and the caption
# panels all share it, so their edges line up.
PLAYER_MAX_WIDTH = 620

_TEMPLATE = """
<style>
  .scp-wrap { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
              color: #e8e8e8; max-width: __MAXW__px; margin: 0 auto; }
  /* aspect-ratio makes the element box exactly match the video, so there are
     no black bars and the video fills the same width as the panels below. */
  .scp-video { width: 100%; aspect-ratio: __ASPECT__; height: auto;
               max-height: 70vh; background: #000; border-radius: 8px;
               display: block; }
  .scp-now { margin: 10px 0; padding: 14px 16px; min-height: 2.4em;
             background: #14161c; border: 1px solid #2b2f3a; border-radius: 8px;
             font-size: 20px; line-height: 1.35; text-align: center;
             color: #ffffff; }
  .scp-now:empty::before { content: "\\2026"; color: #666; }
  .scp-list { max-height: __LISTH__px; overflow-y: auto; border: 1px solid #2b2f3a;
              border-radius: 8px; }
  .scp-row { padding: 7px 12px; cursor: pointer; border-bottom: 1px solid #21242d;
             font-size: 14px; line-height: 1.4; }
  .scp-row:hover { background: #1b1e26; }
  .scp-row.active { background: #2d4a63; color: #fff; }
  .scp-t { color: #8aa0b6; font-variant-numeric: tabular-nums;
           margin-right: 8px; font-size: 12px; }
  @media (prefers-color-scheme: light) {
    .scp-wrap { color: #1a1a1a; }
    .scp-now { background: #f4f6f9; border-color: #d5dae2; color: #111; }
    .scp-list { border-color: #d5dae2; }
    .scp-row { border-bottom-color: #eceff3; }
    .scp-row:hover { background: #eef2f7; }
    .scp-row.active { background: #cfe3f5; color: #0b2a44; }
    .scp-t { color: #5a7088; }
  }
</style>
<div class="scp-wrap">
  <video id="scpVideo" class="scp-video" controls playsinline>
    <source src="data:__MIME__;base64,__B64__" type="__MIME__">
    Your browser cannot play this video inline.
  </video>
  <div id="scpNow" class="scp-now"></div>
  <div id="scpList" class="scp-list"></div>
</div>
<script>
  const SEG = __SEGS__;
  const v = document.getElementById('scpVideo');
  const now = document.getElementById('scpNow');
  const list = document.getElementById('scpList');

  function fmt(x) {
    const m = Math.floor(x / 60), s = Math.floor(x % 60);
    return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }
  function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

  SEG.forEach((seg, i) => {
    const row = document.createElement('div');
    row.className = 'scp-row';
    row.innerHTML = '<span class="scp-t">' + fmt(seg.start) + '</span>' + esc(seg.text);
    row.addEventListener('click', () => { v.currentTime = seg.start + 0.001; v.play(); });
    list.appendChild(row);
  });

  let cur = -1;
  v.addEventListener('timeupdate', () => {
    const t = v.currentTime;
    let idx = -1;
    for (let i = 0; i < SEG.length; i++) {
      if (t >= SEG[i].start && t < SEG[i].end) { idx = i; break; }
    }
    if (idx === cur) return;
    cur = idx;
    now.textContent = idx >= 0 ? SEG[idx].text : '';
    for (let i = 0; i < list.children.length; i++) list.children[i].classList.remove('active');
    if (idx >= 0) {
      const el = list.children[idx];
      el.classList.add('active');
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  });
</script>
"""


def segments_to_cues(segments: Iterable) -> list[dict]:
    """Normalise segments (objects or dicts) to ``{start, end, text}`` cues."""
    cues: list[dict] = []
    for seg in segments:
        if isinstance(seg, dict):
            # Accept both raw ("start_seconds") and already-normalised ("start")
            # keys so this function is safe to apply more than once.
            start = seg.get("start_seconds", seg.get("start"))
            end = seg.get("end_seconds", seg.get("end"))
            text = seg.get("text")
        else:
            start, end, text = seg.start_seconds, seg.end_seconds, seg.text
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            continue
        text_s = str(text or "").strip()
        if not text_s:
            continue
        cues.append({"start": round(start_f, 3), "end": round(end_f, 3), "text": text_s})
    return cues


def build_synced_player_html(
    video_b64: str,
    mime: str,
    segments: Iterable,
    list_height_px: int = 260,
    aspect_ratio: str = "16 / 9",
    max_width: int = PLAYER_MAX_WIDTH,
) -> str:
    """Return the full HTML for the synced below-video caption player.

    ``list_height_px`` controls how tall the transcript panel is before it
    starts scrolling. ``aspect_ratio`` is a CSS aspect-ratio (e.g. ``"1920 /
    1080"``) so the video fills the column with no black bars and lines up with
    the caption panels. ``max_width`` is the shared column width.
    """
    cues = segments_to_cues(segments)
    return (
        _TEMPLATE.replace("__MIME__", mime)
        .replace("__LISTH__", str(int(list_height_px)))
        .replace("__ASPECT__", aspect_ratio)
        .replace("__MAXW__", str(int(max_width)))
        .replace("__SEGS__", json.dumps(cues))
        .replace("__B64__", video_b64)  # last: the largest substitution
    )


# Rough per-row height (px) used to size the transcript panel to its content.
ROW_HEIGHT_PX = 34

# Keep the transcript panel compact: show a handful of rows, then scroll. The
# active line auto-scrolls into view during playback.
MAX_VISIBLE_ROWS = 6


def preview_layout_heights(
    num_cues: int, aspect: float = 16 / 9, max_width: int = PLAYER_MAX_WIDTH
) -> tuple[int, int]:
    """Return (transcript_list_height, total_component_height).

    The transcript panel is kept compact (``MAX_VISIBLE_ROWS`` tall) and scrolls
    for longer transcripts, auto-following playback. The video area is derived
    from the column width and aspect ratio so the iframe height matches what
    actually renders.
    """
    rows = min(max(num_cues, 2), MAX_VISIBLE_ROWS)
    list_h = rows * ROW_HEIGHT_PX + 12
    aspect = aspect if aspect and aspect > 0 else 16 / 9
    video_area = max(180, min(560, round(max_width / aspect)))
    now_area = 110
    total = video_area + now_area + list_h + 24
    return list_h, total
