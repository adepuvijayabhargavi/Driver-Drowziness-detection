"""Logging bootstrap for the whole application."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", stream=sys.stdout) -> logging.Logger:
    logger = logging.getLogger("src")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
    return logger


__all__ = ["setup_logging"]
