"""Tests of the JSON log format and the opt-in tracing switch."""

import json
import logging

from tiendahogar.telemetry import JsonFormatter, configure_tracing


def test_json_formatter_keeps_extra_fields():
    record = logging.makeLogRecord(
        {"name": "tiendahogar.api", "levelname": "INFO", "msg": "chat", "action": "answer"}
    )
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "chat"
    assert entry["action"] == "answer"
    assert entry["logger"] == "tiendahogar.api"
    assert "args" not in entry


def test_tracing_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert configure_tracing() is False
