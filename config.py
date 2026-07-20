"""Central configuration, loaded from environment variables / .env.

No secret is ever hard-coded here. The API key is read from the environment
and never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Optional: load a local .env if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

BASE_DIR = Path(__file__).resolve().parent

# A stable, widely-available default. The *authoritative* list of models your
# key can use comes from ``client.models.list()`` (see gemini_service and the
# "List models" button in the app). Override with the GEMINI_MODEL env var.
DEFAULT_MODEL = "gemini-2.5-flash"

# Supported upload types (lowercase, no dot) mapped to a MIME type hint.
SUPPORTED_VIDEO_TYPES: dict[str, str] = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
    "m4v": "video/mp4",
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL
    )
    max_upload_mb: int = field(default_factory=lambda: _int_env("MAX_UPLOAD_MB", 500))
    temp_dir: Path = field(default_factory=lambda: BASE_DIR / "temp")
    outputs_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "").strip().lower()
        in {"1", "true", "yes"}
    )

    def __post_init__(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
