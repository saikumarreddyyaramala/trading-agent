from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import FakeBroker
from trading_agent import __version__, cli
from trading_agent.bootstrap import Container

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_show_config_never_prints_secrets(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-appear")
    result = runner.invoke(cli.app, ["show-config", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    assert json.loads(result.output)["environment"] == "paper"
    assert "sk-should-not-appear" not in result.output


def test_bad_config_exits_with_code_2(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["show-config", "--config-dir", str(tmp_path)])
    assert result.exit_code == 2


def test_healthcheck_without_keys_fails(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["healthcheck", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "missing: ALPACA_API_KEY" in result.output


def test_healthcheck_json_success(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(Container, "build_broker", lambda self: FakeBroker())

    result = runner.invoke(
        cli.app, ["healthcheck", "--config-dir", str(config_dir), "--skip-llm", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} >= {"Broker account", "Market clock"}
