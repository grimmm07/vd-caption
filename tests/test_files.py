"""Tests for filename sanitization and path-traversal protection."""

from utils.files import safe_join, sanitize_filename


def test_sanitize_strips_directory_components():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\a.mp4") == "a.mp4"
    assert sanitize_filename("/absolute/path/movie.MOV") == "movie.mov"


def test_sanitize_removes_illegal_characters():
    # Illegal chars become underscores; trailing underscores are trimmed.
    assert sanitize_filename('my:vi|deo?.mp4') == "my_vi_deo.mp4"
    assert sanitize_filename('a<b>c*d.mp4') == "a_b_c_d.mp4"


def test_sanitize_lowercases_extension_only():
    out = sanitize_filename("MyClip.MP4")
    assert out.endswith(".mp4")
    assert out.startswith("MyClip")


def test_sanitize_falls_back_for_empty():
    assert sanitize_filename("") == "video"
    assert sanitize_filename("...") == "video"
    assert sanitize_filename("///") == "video"


def test_sanitize_collapses_repeats():
    assert sanitize_filename("a....b.mp4") == "a_b.mp4"


def test_safe_join_blocks_traversal(tmp_path):
    # Even a traversal attempt resolves to a child of the base dir.
    joined = safe_join(tmp_path, "../../secret.mp4")
    assert str(joined).startswith(str(tmp_path.resolve()))
    assert joined.name == "secret.mp4"


def test_safe_join_normal_file(tmp_path):
    joined = safe_join(tmp_path, "clip.mp4")
    assert joined == (tmp_path.resolve() / "clip.mp4")
