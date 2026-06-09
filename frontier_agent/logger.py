"""Central logging setup for Frontier Agent.

All pipeline activity — tool calls, node transitions, confidence scores, routing
decisions, escalations — is written to ~/.frontier/mcp.log as structured text.

Usage in any module:
    from .logger import get_logger
    log = get_logger(__name__)
    log.info("message")

The log file rotates at 5 MB and keeps 3 backups so it never grows unbounded.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path.home() / ".frontier"
_LOG_FILE = _LOG_DIR / "mcp.log"

_FMT = "%(asctime)s  %(levelname)-7s  %(name)-35s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _setup() -> None:
    root = logging.getLogger("frontier")
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)


_setup()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger namespaced under 'frontier'."""
    # strip package prefix so log names stay readable
    short = name.replace("frontier_agent.", "").replace("frontier_agent", "core")
    return logging.getLogger(f"frontier.{short}")


def log_path() -> Path:
    """Return the path to the active log file."""
    return _LOG_FILE
