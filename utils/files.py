"""Safe file handling helpers: filename sanitization and working directories."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

# Characters that are unsafe on Windows and/or POSIX filesystems.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_DOT = re.compile(r"\.{2,}")
_MULTI_UNDERSCORE = re.compile(r"_{2,}")

_MAX_STEM_LENGTH = 80


def sanitize_filename(filename: str, default_stem: str = "video") -> str:
    """Return a safe basename with no path components or traversal.

    * strips any directory components (defeats path traversal like ``../..``),
    * removes control characters and characters illegal on Windows,
    * collapses runs of dots/underscores,
    * preserves a single lowercase extension,
    * falls back to ``default_stem`` when nothing usable remains.
    """
    if not filename:
        return f"{default_stem}"

    # Drop any directory part from either separator style.
    base = os.path.basename(filename.replace("\\", "/"))

    stem, ext = os.path.splitext(base)
    ext = _UNSAFE_CHARS.sub("", ext).lower().lstrip(".")

    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = _MULTI_DOT.sub("_", stem)
    stem = stem.strip(" ._")
    stem = _MULTI_UNDERSCORE.sub("_", stem)
    if not stem:
        stem = default_stem
    stem = stem[:_MAX_STEM_LENGTH]

    return f"{stem}.{ext}" if ext else stem


def new_work_dir(base_dir: str | os.PathLike[str]) -> Path:
    """Create and return a unique working directory under ``base_dir``."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    work = base / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    return work


def safe_join(base_dir: str | os.PathLike[str], filename: str) -> Path:
    """Join ``filename`` onto ``base_dir`` guaranteeing it stays inside it."""
    base = Path(base_dir).resolve()
    candidate = (base / sanitize_filename(filename)).resolve()
    if base not in candidate.parents and candidate != base:
        raise ValueError("resolved path escapes the base directory")
    return candidate


def cleanup_dir(path: str | os.PathLike[str]) -> None:
    """Recursively remove a directory, ignoring errors."""
    shutil.rmtree(path, ignore_errors=True)
