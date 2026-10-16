from __future__ import annotations

import json
import logging

import pytest
from mcp.server import MCPServer

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


def test_token_counts_are_not_mistaken_for_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    get_logger("test").info("llm", input_tokens=10, output_tokens=2, access_token="t0k")
    event = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert event["input_tokens"] == 10
    assert event["output_tokens"] == 2
    assert event["access_token"] == REDACTED


def test_stdlib_logs_share_the_format_and_redaction(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("some.sdk").warning("retrying", extra={"note": "x"})
    event = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert event["event"] == "retrying"
    assert event["level"] == "warning"
    assert event["logger"] == "some.sdk"


def test_noisy_libraries_are_quiet_at_info(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("httpx").info("HTTP Request: POST https://api.example.com")
    logging.getLogger("httpx").warning("connection reset")
    output = capsys.readouterr().err
    assert "HTTP Request" not in output
    assert "connection reset" in output


def test_mcp_server_does_not_replace_our_handlers(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    handlers = list(logging.getLogger().handlers)
    MCPServer(name="probe")
    assert logging.getLogger().handlers == handlers
    logging.getLogger("other").warning("still json")
    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])["event"] == "still json"
