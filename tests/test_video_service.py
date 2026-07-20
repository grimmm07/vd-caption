"""Tests for FFmpeg command construction (no FFmpeg execution required)."""

from services.video_service import (
    build_burn_command,
    build_soft_subtitle_command,
    font_size_for_height,
)


def test_burn_command_is_argument_list():
    cmd = build_burn_command(
        "ffmpeg", "in.mp4", "captions.srt", "out.mp4", font_size=24, has_audio=True
    )
    assert isinstance(cmd, list)
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] == "ffmpeg"


def test_burn_command_preserves_audio_with_aac():
    cmd = build_burn_command("ffmpeg", "in.mp4", "c.srt", "out.mp4", has_audio=True)
    assert "-c:a" in cmd and "aac" in cmd
    assert "-an" not in cmd
    # Input and output present, in order.
    assert cmd.index("-i") < cmd.index("out.mp4")


def test_burn_command_drops_audio_when_absent():
    cmd = build_burn_command("ffmpeg", "in.mp4", "c.srt", "out.mp4", has_audio=False)
    assert "-an" in cmd
    assert "aac" not in cmd


def test_burn_command_uses_h264_and_subtitles_filter():
    cmd = build_burn_command("ffmpeg", "in.mp4", "c.srt", "out.mp4")
    assert "libx264" in cmd
    vf_index = cmd.index("-vf")
    assert "subtitles=" in cmd[vf_index + 1]
    assert "force_style" in cmd[vf_index + 1]


def test_burn_command_does_not_use_shell_string():
    # The whole command must be a token list — never a single shell string.
    cmd = build_burn_command("ffmpeg", "my video.mp4", "c.srt", "out file.mp4")
    assert "my video.mp4" in cmd  # spaces stay in one token, not split
    assert "out file.mp4" in cmd


def test_soft_subtitle_command_uses_mov_text():
    cmd = build_soft_subtitle_command("ffmpeg", "in.mp4", "c.srt", "out.mp4")
    assert "mov_text" in cmd
    assert cmd.count("-i") == 2  # video + subtitle inputs


def test_font_size_scales_with_height():
    assert font_size_for_height(None) == 22
    assert font_size_for_height(480) < font_size_for_height(1080)
    assert 14 <= font_size_for_height(2160) <= 30  # clamped
