import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RunContext:
    """Identifies a workload run in every log line."""

    bundle: str
    target: str
    workload: str
    run_id: str


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "context", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ContextAdapter(logging.LoggerAdapter):
    """Merges the run context and per-call fields into the `context` of each record."""

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping]:
        fields = dict(self.extra or {})
        fields.update(kwargs.pop("extra", None) or {})
        kwargs["extra"] = {"context": fields}
        return msg, kwargs


def get_logger(name: str, context: RunContext, **fields: Any) -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), {**asdict(context), **fields})


def configure_logging(level: int = logging.INFO, stream=None) -> None:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_metrics(logger: ContextAdapter, message: str, metrics: Mapping[str, Any]) -> None:
    logger.info(message, extra=dict(metrics))
