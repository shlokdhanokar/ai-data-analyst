"""Structured (JSON-lines) logging for observability.

Every LLM call, tool call and user question is logged with timings so a run can be
reconstructed from logs/app.log.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("analyst")
    root.setLevel(level.upper())
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError:  # read-only filesystem (e.g. some hosts): console logging only
        root.warning("file logging disabled", extra={"data": {"log_dir": log_dir}})
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"analyst.{name}")


def log_event(logger: logging.Logger, msg: str, **data: Any) -> None:
    logger.info(msg, extra={"data": data})


@contextmanager
def timed(logger: logging.Logger, msg: str, **data: Any) -> Iterator[dict[str, Any]]:
    """Log `msg` with duration_ms; callers can add fields to the yielded dict."""
    start = time.perf_counter()
    fields: dict[str, Any] = dict(data)
    try:
        yield fields
        fields.setdefault("ok", True)
    except Exception as exc:
        fields["ok"] = False
        fields["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        fields["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info(msg, extra={"data": fields})
