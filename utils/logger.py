"""
utils/logger.py — Structured logging setup
"""
import io
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure root logger with rich formatting."""
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    # Force UTF-8 on stdout so Unicode chars (✓, —, etc.) don't crash on Windows cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "selenium", "WDM", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
