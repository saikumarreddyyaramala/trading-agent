from __future__ import annotations

import json

import pytest

from trading_agent.observability.logging import REDACTED, configure_logging, get_logger


def test_json_logs_redact_secret_looking_keys(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    get_logger("test").info("connected", api_key="sk-123", secret_key="abc", user="sai")

    line = capsys.readouterr().err.strip().splitlines()[-1]
    event = json.loads(line)
    assert event["event"] == "connected"
    assert event["api_key"] == REDACTED
    assert event["secret_key"] == REDACTED
    assert event["user"] == "sai"
    assert event["level"] == "info"
    assert "timestamp" in event


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", fmt="json")
    get_logger("test").info("hidden")
    get_logger("test").warning("shown")
    output = capsys.readouterr().err
    assert "hidden" not in output
    assert "shown" in output
