"""Structured logging setup.

Logs are intentionally coarse-grained and never include the API key or raw
media. Transcript text is only logged when ``debug`` is enabled.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(debug: bool | None = None) -> logging.Logger:
    """Configure and return the application logger (idempotent)."""
    global _CONFIGURED
    logger = logging.getLogger("video_caption")
    if _CONFIGURED:
        return logger

    if debug is None:
        debug = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes"}

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    return configure_logging()
