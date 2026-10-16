"""Structured logging.

- `console` format for humans during development, `json` for log platforms in production.
- Every event carries a timestamp, level, and logger name; callers add context as key-values.
- A redaction processor masks anything that looks like a credential, as a last line of defence.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from typing import Any, Literal

import structlog

LogFormat = Literal["console", "json"]

# "token$" catches access_token / auth_token but not counters such as input_tokens.
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|secret|password|authorization|credential|token$)", re.IGNORECASE
)
# Libraries that log every HTTP request or protocol message at INFO.
_NOISY_LIBRARIES = ("httpx", "httpcore", "mcp", "anyio")
REDACTED = "***REDACTED***"


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if _SENSITIVE_KEY.search(key):
            event_dict[key] = REDACTED
    return event_dict


class _StderrHandler(logging.Handler):
    """Writes to whatever `sys.stderr` is at emit time, so redirected streams keep working."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:  # pragma: no cover - logging must never crash the caller
            self.handleError(record)


def configure_logging(level: str | None = None, fmt: LogFormat | None = None) -> None:
    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    chosen: str = fmt or os.environ.get("LOG_FORMAT", "console")

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if chosen == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    shared: list[Any] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets,
    ]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            *shared,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        # Resolve sys.stderr at call time (not import time) so redirected streams keep working.
        logger_factory=lambda *_args: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=False,
    )

    # Standard-library logs (SDKs, MCP) go through the same renderer and redaction, so the
    # output stays one format. Owning the root handler also makes libraries that call
    # logging.basicConfig() (the MCP server does) a no-op.
    handler = _StderrHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[structlog.stdlib.add_logger_name, *shared],
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric_level)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
