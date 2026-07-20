"""Tests for the below-video synced caption player HTML builder."""

from models.transcript import Segment
from utils.synced_player import (
    INLINE_PLAYABLE,
    build_synced_player_html,
    preview_layout_heights,
    segments_to_cues,
)


def test_segments_to_cues_from_objects():
    segs = [
        Segment(id=1, start_seconds=0.0, end_seconds=2.0, text="Hi"),
        Segment(id=2, start_seconds=2.0, end_seconds=4.0, text="there"),
    ]
    cues = segments_to_cues(segs)
    assert cues == [
        {"start": 0.0, "end": 2.0, "text": "Hi"},
        {"start": 2.0, "end": 4.0, "text": "there"},
    ]


def test_segments_to_cues_from_dicts_and_filters_bad_rows():
    rows = [
        {"start_seconds": 0.0, "end_seconds": 1.0, "text": "keep"},
        {"start_seconds": 1.0, "end_seconds": 2.0, "text": "   "},  # blank
        {"start_seconds": "x", "end_seconds": 3.0, "text": "bad"},  # unparsable
    ]
    cues = segments_to_cues(rows)
    assert [c["text"] for c in cues] == ["keep"]


def test_build_html_embeds_video_and_cues():
    cues = [{"start": 0.0, "end": 1.0, "text": "Hello <b>world</b>"}]
    html = build_synced_player_html("QUJD", "video/mp4", cues)
    # Video embedded as a data URI, and the cue text is present as JSON data.
    assert "data:video/mp4;base64,QUJD" in html
    assert "Hello <b>world</b>" in html  # escaped safely at render time in JS
    assert "timeupdate" in html  # sync logic present
    # every placeholder filled in
    for placeholder in ("__B64__", "__SEGS__", "__MIME__", "__LISTH__", "__ASPECT__", "__MAXW__"):
        assert placeholder not in html


def test_build_html_uses_given_aspect_ratio():
    html = build_synced_player_html("QUJD", "video/mp4", [], aspect_ratio="1920 / 1080")
    assert "aspect-ratio: 1920 / 1080" in html


def test_inline_playable_contains_mp4():
    assert INLINE_PLAYABLE["mp4"] == "video/mp4"
    assert "webm" in INLINE_PLAYABLE


def test_preview_layout_video_area_follows_aspect():
    # A tall (portrait) video reserves more vertical space than a wide one.
    _, wide_total = preview_layout_heights(5, aspect=16 / 9)
    _, tall_total = preview_layout_heights(5, aspect=9 / 16)
    assert tall_total > wide_total


def test_preview_layout_grows_with_cues_then_caps():
    small_list, small_total = preview_layout_heights(2)
    big_list, big_total = preview_layout_heights(30)
    huge_list, _ = preview_layout_heights(500)
    # More cues -> taller list (until the cap).
    assert big_list > small_list
    assert big_total > small_total
    # Capped so a very long transcript doesn't produce an enormous iframe.
    assert huge_list == big_list or huge_list >= big_list
    assert huge_list <= 40 * 34 + 12
