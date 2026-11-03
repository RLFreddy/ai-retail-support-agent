"""Logging and optional OpenTelemetry tracing.

- ``LOG_FORMAT=json`` writes one JSON object per line, with the ``extra``
  fields of each record (action, latency...), ready for a log pipeline.
- Tracing turns on only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set and the
  ``otel`` extra is installed; Pydantic AI then emits a span per agent run,
  model request and tool call.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from pydantic_ai import Agent

log = logging.getLogger(__name__)

#: Attributes every LogRecord has; anything else came from ``extra=``.
_STANDARD = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry |= {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "WARNING", fmt: str = "text") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=level.upper(), handlers=[handler])


def configure_tracing(service_name: str = "tiendahogar") -> bool:
    """Export traces over OTLP when an endpoint is configured. Returns whether it is on."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but the 'otel' extra is not installed")
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    Agent.instrument_all()
    return True
