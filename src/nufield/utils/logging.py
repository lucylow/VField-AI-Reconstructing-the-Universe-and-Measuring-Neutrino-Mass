"""Standardized logging configuration for CLI and scripts."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown logging level: {level}")
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("nufield")
