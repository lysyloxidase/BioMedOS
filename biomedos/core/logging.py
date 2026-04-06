"""Structured logging helpers."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a Rich-backed root logger.

    Args:
        level: Logging verbosity.

    Returns:
        The configured root logger.
    """

    global _CONFIGURED

    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
        )
        _CONFIGURED = True

    logger = logging.getLogger("biomedos")
    logger.setLevel(level)
    return logger
