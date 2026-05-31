from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog


class _DynamicStdoutLoggerFactory:
    """Factory that re-resolves ``sys.stdout`` on every logger construction.

    structlog's built-in ``PrintLoggerFactory`` captures the stdout handle once
    (at construction or at ``PrintLogger.__init__`` via a module-level
    ``stdout`` import). When pytest's ``capsys``/``capfd`` swaps ``sys.stdout``
    mid-suite and that captured stream is later closed, log calls raise
    ``ValueError: I/O operation on closed file``. Looking up ``sys.stdout``
    per call dodges that and costs ~microseconds.
    """

    def __call__(self, *args: Any) -> structlog.PrintLogger:
        return structlog.PrintLogger(sys.stdout)


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=_DynamicStdoutLoggerFactory(),
        # Pair with the dynamic factory above: caching would defeat the
        # per-call stdout re-resolution and re-introduce the closed-file bug.
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
