from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
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
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.bootstrap import Container
from trading_agent.data import MarketDataCollector
from trading_agent.domain.models import Evidence, LLMUsage, TradeProposal

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


# ---------- analyze ----------


def _analysis(status: str, proposal: TradeProposal | None) -> AnalystResult:
    return AnalystResult(
        run_id="abc123",
        status=status,  # type: ignore[arg-type]
        reason="evidence verified" if status == "proposed" else "something",
        proposal=proposal,
        prompt_version="p1",
        turns=3,
        tool_calls=(ToolCallRecord(call_id="c1", tool="get_quote", arguments={}, is_error=False),),
        usage=LLMUsage(input_tokens=1000, output_tokens=234),
        started_at=NOW,
        duration_seconds=4.2,
    )


BUY = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="Strong tape.",
    reference_price=Decimal("100.05"),
    confidence="medium",
    evidence=(
        Evidence(claim="Tight spread", source_call_id="c1"),
        Evidence(claim="Uptrend", source_call_id="c2"),
    ),
    risks=("Reversal",),
)


def _use_fake_analyst(monkeypatch: pytest.MonkeyPatch, result: AnalystResult) -> None:
    class _Agent:
        async def run(self) -> AnalystResult:
            return result

    monkeypatch.setattr(Container, "build_analyst", lambda self: _Agent())


def test_analyze_prints_verified_proposal(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "PROPOSED  evidence verified  [run abc123, 3 turns, 1 tool calls, 1,234 tokens" in (
        result.output
    )
    assert "BUY AAPL near $100.05 (confidence medium)" in result.output
    assert "Evidence  - Tight spread [c1]" in result.output
    assert "Risk      - Reversal" in result.output
    assert "No order was placed" in result.output


def test_analyze_no_trade(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proposal = TradeProposal(decision="no_trade", thesis="Quiet day.", confidence="high")
    _use_fake_analyst(monkeypatch, _analysis("no_trade", proposal))
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    assert "NO TRADE (confidence high)" in result.output


def test_analyze_failure_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_analyst(monkeypatch, _analysis("failed", None))
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "Decision" not in result.output


def test_analyze_json(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    result = runner.invoke(cli.app, ["analyze", "--json", "--config-dir", str(config_dir)])
    payload = json.loads(result.stdout)
    assert payload["status"] == "proposed"
    assert payload["proposal"]["symbol"] == "AAPL"
    assert payload["proposal"]["reference_price"] == "100.05"


def test_analyze_without_keys_exits_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "missing" in result.output
