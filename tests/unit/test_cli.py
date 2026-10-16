from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    FakeNews,
    healthy_market_data,
    make_article,
    make_quote,
)
from trading_agent import __version__, cli
from trading_agent.bootstrap import Container
from trading_agent.data import MarketDataCollector

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
    # The probe quote is checked against the real clock, so make it current.
    probe = healthy_market_data("SPY")
    probe.quotes["SPY"] = make_quote("SPY", now=datetime.now(UTC))
    monkeypatch.setattr(Container, "build_market_data", lambda self: probe)
    monkeypatch.setattr(Container, "build_news", lambda self: FakeNews([make_article()]))

    result = runner.invoke(
        cli.app, ["healthcheck", "--config-dir", str(config_dir), "--skip-llm", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} >= {
        "Broker account",
        "Market clock",
        "Market data",
        "News",
    }


# ---------- market-data ----------


def _use_fake_collector(
    monkeypatch: pytest.MonkeyPatch, market_data: FakeMarketData, news: FakeNews | None = None
) -> None:
    def build(self: Container) -> MarketDataCollector:
        return MarketDataCollector(
            market_data,
            news or FakeNews([make_article("9", ("AAPL",))]),
            self.config.market_data,
            max_data_age_seconds=self.config.risk.max_data_age_seconds,
            clock=lambda: NOW,
        )

    monkeypatch.setattr(Container, "build_collector", build)


def test_market_data_human_output(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_collector(monkeypatch, healthy_market_data("AAPL"))
    result = runner.invoke(cli.app, ["market-data", "aapl", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "AAPL Inc. (NASDAQ, tradable)" in result.output
    assert "bid $100.00 / ask $100.10 (spread 0.10%), 2s old" in result.output
    assert "last close $101.00, 3d change +1.00%, avg volume 2,000,000" in result.output
    assert "Headline 9" in result.output


def test_market_data_json_output(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_collector(monkeypatch, healthy_market_data("AAPL"))
    result = runner.invoke(
        cli.app, ["market-data", "AAPL", "--json", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    symbol = payload["symbols"][0]
    assert symbol["symbol"] == "AAPL"
    assert symbol["quote"]["age_seconds"] == 2
    assert symbol["change_pct"] == "1.00"
    assert symbol["news"][0]["headline"] == "Headline 9"
    assert payload["rejected"] == {}


def test_market_data_without_usable_quote(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    market_data = healthy_market_data("AAPL")
    market_data.quotes.clear()
    _use_fake_collector(monkeypatch, market_data)
    human = runner.invoke(cli.app, ["market-data", "AAPL", "--config-dir", str(config_dir)])
    as_json = runner.invoke(
        cli.app, ["market-data", "AAPL", "--json", "--config-dir", str(config_dir)]
    )
    assert "Quote   none usable" in human.output
    assert json.loads(as_json.stdout)["symbols"][0]["quote"] is None


def test_market_data_rejection_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    market_data = healthy_market_data("AAPL")
    market_data.quotes["AAPL"] = make_quote("AAPL", age_seconds=3_600)
    _use_fake_collector(monkeypatch, market_data)
    result = runner.invoke(
        cli.app, ["market-data", "AAPL", "--fresh", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "REJECTED  AAPL: stale quote" in result.output


def test_market_data_invalid_symbol_exits_2(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_collector(monkeypatch, healthy_market_data())
    result = runner.invoke(
        cli.app, ["market-data", "NOT A TICKER", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2
    assert "invalid symbol" in result.output


def test_market_data_without_keys_exits_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["market-data", "AAPL", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "missing market data credentials" in result.output


def test_market_data_outage_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_collector(monkeypatch, FakeMarketData(fail=True))
    result = runner.invoke(cli.app, ["market-data", "AAPL", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "Market data unavailable" in result.output


# ---------- mcp-server ----------


def test_mcp_server_serves_stdio(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[object, str]] = []
    sentinel = object()
    monkeypatch.setattr(Container, "build_research_tools", lambda self: sentinel)

    class _Server:
        def __init__(self, tools: object) -> None:
            self.tools = tools

        def run(self, transport: str) -> None:
            served.append((self.tools, transport))

    monkeypatch.setattr(cli, "build_server", _Server)
    result = runner.invoke(cli.app, ["mcp-server", "--config-dir", str(config_dir)])

    assert result.exit_code == 0, result.output
    assert served == [(sentinel, "stdio")]
    assert result.stdout == ""  # stdout belongs to the MCP protocol


def test_mcp_server_without_keys_exits_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["mcp-server", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "missing market data credentials" in result.output
