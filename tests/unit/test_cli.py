from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from typer.testing import CliRunner

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    FakeNews,
    FakeOrders,
    FakeTime,
    healthy_market_data,
    make_article,
    make_position,
    make_quote,
    make_trend_bars,
)
from trading_agent import __version__, cli
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.agents.critic import CriticResult, EvidenceCheck
from trading_agent.bootstrap import Container
from trading_agent.data import MarketDataCollector
from trading_agent.domain.models import Evidence, LLMUsage, RiskDecision, TradeProposal
from trading_agent.execution.executor import Executor
from trading_agent.orchestration.cycle import DayResult
from trading_agent.reporting.delivery import DeliveryError
from trading_agent.risk import RiskEngine
from trading_agent.settings.models import AppConfig
from trading_agent.storage import schema
from trading_agent.storage.database import head_revision, make_engine, upgrade
from trading_agent.storage.store import NullAuditStore, SqlAuditStore

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


def _critique(status: str = "approved") -> CriticResult:
    return CriticResult(
        run_id="abc123",
        status=status,  # type: ignore[arg-type]
        model_verdict="approve" if status == "approved" else "veto",
        summary="Claims match the data." if status == "approved" else "Momentum is fading.",
        reasons=() if status == "approved" else ("the critic vetoed the trade",),
        concerns=("Broad market weakness",),
        evidence_checks=(
            EvidenceCheck(source_call_id="c1", supported=True, note="spread 0.02% as stated"),
        ),
        model="gpt-6.1-sol",
        prompt_version="critic-p",
        usage=LLMUsage(input_tokens=3000, output_tokens=300),
        created_at=NOW,
    )


def _use_fake_analyst(
    monkeypatch: pytest.MonkeyPatch,
    result: AnalystResult,
    critique: CriticResult | None = None,
) -> list[tuple[str, int]]:
    """Fake analyst and critic. Returns the critic's calls: (run id, tool calls seen)."""
    reviews: list[tuple[str, int]] = []

    class _Agent:
        async def run(self) -> AnalystResult:
            return result

    class _Critic:
        def review(self, run_id: str, proposal: TradeProposal, calls: list[object]) -> CriticResult:
            reviews.append((run_id, len(calls)))
            return critique or _critique()

    monkeypatch.setattr(Container, "build_analyst", lambda self: _Agent())
    monkeypatch.setattr(Container, "build_critic", lambda self: _Critic())
    return reviews


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
    assert payload["analysis"]["status"] == "proposed"
    assert payload["analysis"]["proposal"]["symbol"] == "AAPL"
    assert payload["analysis"]["proposal"]["reference_price"] == "100.05"
    assert payload["critique"]["status"] == "approved"


def test_analyze_without_keys_exits_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "missing" in result.output


# ---------- risk-check ----------


def _use_fake_risk(monkeypatch: pytest.MonkeyPatch, broker: FakeBroker) -> list[TradeProposal]:
    seen: list[TradeProposal] = []

    class _Engine:
        def evaluate(self, proposal: TradeProposal) -> RiskDecision:
            seen.append(proposal)
            market_data = healthy_market_data("AAPL")
            market_data.bars["AAPL"] = make_trend_bars()  # enough history to score
            return RiskEngine(
                broker,
                market_data,
                AppConfig(),
                live_trading_confirmed=False,
                clock=lambda: NOW,
            ).evaluate(proposal)

    monkeypatch.setattr(Container, "build_risk_engine", lambda self: _Engine())
    return seen


def test_risk_check_approves(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app, ["risk-check", "aapl", "--price", "100.05", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "PASS  Liquidity" in result.output
    assert "APPROVED" in result.output
    assert "Buy 0.4899 AAPL (~$49.03), limit $102.05, stop $95.09, target $110.11" in result.output
    assert "The risk check itself places no order." in result.output
    assert seen[0].symbol == "AAPL"


def test_risk_check_rejects(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_risk(monkeypatch, FakeBroker(market_open=False))
    result = runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", "100.05", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "FAIL  Market open        the market is closed" in result.output
    assert "REJECTED" in result.output


def test_risk_check_json(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app,
        ["risk-check", "AAPL", "--price", "100.05", "--json", "--config-dir", str(config_dir)],
    )
    payload = json.loads(result.stdout)
    assert payload["approved"] is True
    assert payload["plan"]["limit_price"] == "102.05"


@pytest.mark.parametrize("price", ["abc", "-5", "0"])
def test_risk_check_rejects_bad_prices(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, price: str
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", price, "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2
    assert "invalid input" in result.output


def test_risk_check_without_keys_exits_2(config_dir: Path) -> None:
    result = runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", "100", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2
    assert "missing broker credentials" in result.output


# ---------- buy / manage / flatten ----------


def _use_fake_executor(
    monkeypatch: pytest.MonkeyPatch, orders: FakeOrders, broker: FakeBroker | None = None
) -> FakeOrders:
    time = FakeTime()
    executor = Executor(
        orders,
        broker or FakeBroker(),
        healthy_market_data("AAPL"),
        AppConfig(),
        clock=time.clock,
        sleep=time.sleep,
    )
    monkeypatch.setattr(Container, "build_executor", lambda self: executor)
    return orders


def test_buy_requires_confirm(config_dir: Path) -> None:
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2
    assert "Refusing without --confirm" in result.output


def test_buy_enters_with_a_stop(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    orders = _use_fake_executor(monkeypatch, FakeOrders("instant"))
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "APPROVED" in result.output
    assert "ENTERED  filled and protected by a stop-loss" in result.output
    assert "Stop   new: sell 0.4899 AAPL at $95.09" in result.output
    assert [action for action, _ in orders.log] == ["buy", "stop"]
    assert orders.log[0][1].split("[")[1].startswith("ta-manual-")


def test_buy_rejected_by_risk_places_nothing(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker(market_open=False))
    orders = _use_fake_executor(monkeypatch, FakeOrders())
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "REJECTED" in result.output
    assert orders.log == []


def test_buy_that_does_not_fill_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    _use_fake_executor(monkeypatch, FakeOrders("none"))
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "NOT_FILLED" in result.output


def test_buy_bad_price_exits_2(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_executor(monkeypatch, FakeOrders())
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "abc", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2


def test_live_buy_is_refused_when_gates_fail(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = AppConfig.model_validate(
        {"environment": "live", "broker": {"paper": False}, "storage": {"required": True}}
    )
    orders = FakeOrders("instant")
    time = FakeTime()
    blocked = Executor(
        orders,
        FakeBroker(is_paper=False),
        healthy_market_data("AAPL"),
        live,
        clock=time.clock,
        sleep=time.sleep,
        entries_blocked="live gates failed: Paper evaluation: says NO-GO",
    )
    monkeypatch.setattr(Container, "build_executor", lambda self: blocked)
    _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "REFUSED  live gates failed: Paper evaluation: says NO-GO" in result.output
    assert orders.log == []


def test_manage_pass(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = FakeBroker(positions=(make_position("AAPL", "0.4899", "100"),))
    orders = _use_fake_executor(monkeypatch, FakeOrders(), broker)
    result = runner.invoke(cli.app, ["manage", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "STOP_RESTORED  AAPL   placed missing stop at $95.00" in result.output
    assert orders.log[0][0] == "stop"


def test_manage_error_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_executor(monkeypatch, FakeOrders(), FakeBroker(fail_positions=True))
    result = runner.invoke(cli.app, ["manage", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_flatten_requires_confirm(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["flatten", "--config-dir", str(config_dir)])
    assert result.exit_code == 2


def test_flatten(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = FakeBroker(positions=(make_position("AAPL"),))
    orders = _use_fake_executor(monkeypatch, FakeOrders(), broker)
    result = runner.invoke(cli.app, ["flatten", "--confirm", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "FLATTENED" in result.output
    assert orders.log == [("close", "AAPL")]


def test_flatten_failure_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = FakeBroker(positions=(make_position("AAPL"),))
    _use_fake_executor(monkeypatch, FakeOrders(fail_close=True), broker)
    result = runner.invoke(cli.app, ["flatten", "--confirm", "--config-dir", str(config_dir)])
    assert result.exit_code == 1


def test_execution_commands_without_keys_exit_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["manage", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "Execution unavailable" in result.output


# ---------- Audit trail ----------


@pytest.fixture
def audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sa.Engine:
    url = f"sqlite:///{tmp_path / 'audit.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    engine = make_engine(url)
    upgrade(engine)
    return engine


def _rows(engine: sa.Engine, table: sa.Table) -> list[dict[str, object]]:
    with engine.connect() as connection:
        return [dict(r) for r in connection.execute(sa.select(table)).mappings()]


def test_db_upgrade_and_status(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}")
    before = runner.invoke(cli.app, ["db", "status", "--config-dir", str(config_dir)])
    assert before.exit_code == 1
    assert "needs `trading-agent db upgrade`" in before.output

    result = runner.invoke(cli.app, ["db", "upgrade", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert f"Audit database (sqlite) is at schema {head_revision()}." in result.output

    after = runner.invoke(cli.app, ["db", "status", "--config-dir", str(config_dir)])
    assert after.exit_code == 0
    assert "up to date" in after.output


def test_db_commands_need_a_url(config_dir: Path) -> None:
    for args in (["db", "upgrade"], ["db", "status"], ["runs"], ["show-run", "x"], ["sync-orders"]):
        result = runner.invoke(cli.app, [*args, "--config-dir", str(config_dir)])
        assert result.exit_code == 2, args
        assert "DATABASE_URL is not set" in result.output


def test_analyze_is_recorded(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    (run,) = _rows(audit_db, schema.runs)
    assert (run["run_id"], run["kind"], run["status"]) == ("abc123", "analyze", "proposed")
    (analysis,) = _rows(audit_db, schema.analyses)
    assert analysis["symbol"] == "AAPL"
    assert len(_rows(audit_db, schema.tool_calls)) == 1


def test_risk_check_is_recorded(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", "100.05", "--config-dir", str(config_dir)]
    )
    (run,) = _rows(audit_db, schema.runs)
    assert run["kind"] == "risk_check"
    assert run["status"] == "approved"
    assert run["run_id"].startswith("check-")
    assert _rows(audit_db, schema.risk_decisions)[0]["approved"] is True


def test_buy_records_decision_orders_and_result(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    _use_fake_executor(monkeypatch, FakeOrders("instant"))
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    (run,) = _rows(audit_db, schema.runs)
    assert (run["kind"], run["status"]) == ("manual_buy", "entered")
    orders = {o["role"]: o for o in _rows(audit_db, schema.orders)}
    assert set(orders) == {"entry", "stop"}
    assert orders["entry"]["run_id"] == run["run_id"]
    assert orders["entry"]["status"] == "filled"


def test_rejected_buy_is_recorded(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker(market_open=False))
    _use_fake_executor(monkeypatch, FakeOrders())
    runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    (run,) = _rows(audit_db, schema.runs)
    assert run["status"] == "rejected"
    assert _rows(audit_db, schema.orders) == []


def test_no_audit_record_means_no_trade(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nstorage:\n  required: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'never-migrated.db'}")
    _use_fake_risk(monkeypatch, FakeBroker())
    orders = _use_fake_executor(monkeypatch, FakeOrders("instant"))
    result = runner.invoke(
        cli.app, ["buy", "AAPL", "--price", "100.05", "--confirm", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "Audit write failed (risk decision)" in result.output
    assert orders.log == []  # nothing was traded


def test_required_storage_without_url_exits_2(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nstorage:\n  required: true\n",
        encoding="utf-8",
    )
    _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", "100", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2
    assert "Audit store unavailable" in result.output


def test_manage_records_only_changes(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    broker = FakeBroker(positions=(make_position("AAPL", "0.4899", "100"),))
    _use_fake_executor(monkeypatch, FakeOrders(), broker)
    runner.invoke(cli.app, ["manage", "--config-dir", str(config_dir)])  # restores the stop
    runner.invoke(cli.app, ["manage", "--config-dir", str(config_dir)])  # now holds: not recorded
    (event,) = _rows(audit_db, schema.position_events)
    assert event["action"] == "stop_restored"
    assert _rows(audit_db, schema.orders)[0]["role"] == "restore_stop"


def test_flatten_is_recorded(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_executor(monkeypatch, FakeOrders(), FakeBroker(positions=(make_position("AAPL"),)))
    runner.invoke(cli.app, ["flatten", "--confirm", "--config-dir", str(config_dir)])
    (event,) = _rows(audit_db, schema.position_events)
    assert event["action"] == "flattened"


def test_runs_and_show_run(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])

    listing = runner.invoke(cli.app, ["runs", "--config-dir", str(config_dir)])
    assert listing.exit_code == 0
    assert "abc123" in listing.output
    assert "analyze" in listing.output

    shown = runner.invoke(cli.app, ["show-run", "abc123", "--config-dir", str(config_dir)])
    assert shown.exit_code == 0
    report = json.loads(shown.stdout)
    assert report["analysis"]["symbol"] == "AAPL"
    assert report["run"]["kind"] == "analyze"

    missing = runner.invoke(cli.app, ["show-run", "nope", "--config-dir", str(config_dir)])
    assert missing.exit_code == 1


def test_sync_orders(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    orders = FakeOrders("instant")
    orders.submit_limit_buy("AAPL", Decimal("1"), Decimal("100"), "ta-old-entry")
    monkeypatch.setattr(Container, "build_order_port", lambda self: orders)
    result = runner.invoke(cli.app, ["sync-orders", "--days", "2", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "Synced 1 order(s) from the last 2 day(s)." in result.output
    assert _rows(audit_db, schema.orders)[0]["client_order_id"] == "ta-old-entry"


def test_sync_orders_broker_failure(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    class _Broken:
        def get_orders_since(self, since: object) -> None:
            raise RuntimeError("broker down")

    monkeypatch.setattr(Container, "build_order_port", lambda self: _Broken())
    result = runner.invoke(cli.app, ["sync-orders", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "Could not fetch orders: broker down" in result.output


# ---------- run / cycle-status ----------


def _use_fake_cycle(monkeypatch: pytest.MonkeyPatch, result: DayResult) -> list[str]:
    calls: list[str] = []

    class _Cycle:
        def __init__(self, services: object, config: object) -> None:
            pass

        def run_day(self, day: object) -> DayResult:
            calls.append("run_day")
            return result

        def run_forever(self) -> None:
            calls.append("run_forever")
            raise KeyboardInterrupt

    monkeypatch.setattr(Container, "build_cycle_services", lambda self: object())
    monkeypatch.setattr(cli, "DailyCycle", _Cycle)
    return calls


TRADED = DayResult(
    run_id="daily-20261015",
    day=date(2026, 10, 15),
    status="traded",
    summary="entered 0.4899 AAPL at $102",
    steps={"preflight": "ok", "entry": "entered"},
)


def test_run_once(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _use_fake_cycle(monkeypatch, TRADED)
    result = runner.invoke(cli.app, ["run", "--once", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert calls == ["run_day"]
    assert "TRADED  daily-20261015: entered 0.4899 AAPL at $102" in result.output
    assert "entry      entered" in result.output


def test_run_once_error_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_cycle(monkeypatch, TRADED.model_copy(update={"status": "error"}))
    result = runner.invoke(cli.app, ["run", "--once", "--config-dir", str(config_dir)])
    assert result.exit_code == 1


def test_run_forever_stops_cleanly_on_ctrl_c(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _use_fake_cycle(monkeypatch, TRADED)
    result = runner.invoke(cli.app, ["run", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    assert calls == ["run_forever"]
    assert "Open positions keep their broker-side stop-loss" in result.output


def test_run_refuses_without_keys(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["run", "--once", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "Cannot start the cycle" in result.output


def test_cycle_status(config_dir: Path, audit_db: sa.Engine) -> None:
    store = SqlAuditStore(audit_db, AppConfig())
    store.start_run("daily-20261015", "daily")
    store.record_step("daily-20261015", "preflight", "ok", "all checks passed", NOW)
    store.finish_run("daily-20261015", "no_trade", "skipped")

    result = runner.invoke(
        cli.app, ["cycle-status", "--date", "2026-10-15", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "daily-20261015: no_trade" in result.output
    assert "preflight  ok         all checks passed" in result.output

    missing = runner.invoke(
        cli.app, ["cycle-status", "--date", "2026-10-14", "--config-dir", str(config_dir)]
    )
    assert missing.exit_code == 1
    bad = runner.invoke(
        cli.app, ["cycle-status", "--date", "Oct 15", "--config-dir", str(config_dir)]
    )
    assert bad.exit_code == 2


# ---------- report ----------


def _use_fake_report(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent.reporting.report import ReportBuilder

    def build(self: Container, store: object = None) -> ReportBuilder:
        return ReportBuilder(
            store or NullAuditStore(),  # type: ignore[arg-type]
            healthy_market_data("SPY"),
            FakeBroker(equity="100000"),
            self.config,
            clock=lambda: NOW,
        )

    monkeypatch.setattr(Container, "build_report_builder", build)


def test_report_prints_and_saves(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_fake_report(monkeypatch)
    result = runner.invoke(
        cli.app, ["report", "--date", "2026-10-15", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "# Daily report: Thursday 2026-10-15" in result.stdout
    assert (tmp_path / "reports" / "2026-10-15.md").exists()
    assert (tmp_path / "reports" / "2026-10-15.json").exists()


def test_report_json_without_saving(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_fake_report(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["report", "--date", "2026-10-15", "--json", "--no-save", "--config-dir", str(config_dir)],
    )
    assert json.loads(result.stdout)["run_id"] == "daily-20261015"
    assert not (tmp_path / "reports").exists()


def test_report_with_the_audit_trail(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_report(monkeypatch)
    store = SqlAuditStore(audit_db, AppConfig())
    store.start_run("daily-20261015", "daily")
    store.finish_run("daily-20261015", "no_trade", "no verified proposal")
    result = runner.invoke(
        cli.app, ["report", "--date", "2026-10-15", "--no-save", "--config-dir", str(config_dir)]
    )
    assert "**NO_TRADE**: no verified proposal" in result.stdout


def test_report_send(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_report(monkeypatch)
    sent: list[str] = []
    monkeypatch.setattr(cli, "post_webhook", lambda url, text: sent.append(url))
    monkeypatch.setenv("REPORT_WEBHOOK_URL", "https://hooks.example/x")
    result = runner.invoke(
        cli.app,
        ["report", "--date", "2026-10-15", "--send", "--no-save", "--config-dir", str(config_dir)],
    )
    assert result.exit_code == 0, result.output
    assert sent == ["https://hooks.example/x"]
    assert "Sent to the webhook." in result.output


def test_report_send_without_url_exits_2(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["report", "--send", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "REPORT_WEBHOOK_URL is not set" in result.output


def test_report_delivery_failure_exits_1(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_report(monkeypatch)

    def fail(url: str, text: str) -> None:
        raise DeliveryError("webhook rejected the report: HTTP 403")

    monkeypatch.setattr(cli, "post_webhook", fail)
    monkeypatch.setenv("REPORT_WEBHOOK_URL", "https://hooks.example/x")
    result = runner.invoke(
        cli.app, ["report", "--send", "--no-save", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "HTTP 403" in result.output


def test_report_bad_date_and_required_store(config_dir: Path) -> None:
    bad = runner.invoke(cli.app, ["report", "--date", "yesterday", "--config-dir", str(config_dir)])
    assert bad.exit_code == 2
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nstorage:\n  required: true\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["report", "--config-dir", str(config_dir)])
    assert result.exit_code == 2


# ---------- Critic in the CLI ----------


def test_analyze_runs_the_critic_on_a_buy(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert reviews == [("abc123", 1)]  # the analyst's tool calls reach the critic
    assert "CRITIC APPROVED  [gpt-6.1-sol, 3,300 tokens]" in result.output
    assert "Check     supported   c1: spread 0.02% as stated" in result.output
    assert "Concern   - Broad market weakness" in result.output
    assert result.output.rstrip().endswith("Approved proposals go to the risk engine.")


def test_analyze_shows_a_veto(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY), _critique("vetoed"))
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert "CRITIC VETOED" in result.output
    assert "Reason    - the critic vetoed the trade" in result.output


def test_no_critic_for_no_trade_or_when_disabled(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_trade = TradeProposal(decision="no_trade", thesis="Quiet.", confidence="high")
    reviews = _use_fake_analyst(monkeypatch, _analysis("no_trade", no_trade))
    runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert reviews == []

    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    monkeypatch.setattr(Container, "build_critic", lambda self: None)
    result = runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    assert "CRITIC" not in result.output


def test_analyze_records_the_critique(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY), _critique("vetoed"))
    runner.invoke(cli.app, ["analyze", "--config-dir", str(config_dir)])
    (row,) = _rows(audit_db, schema.critiques)
    assert (row["run_id"], row["status"], row["model_verdict"]) == ("abc123", "vetoed", "veto")


def _recorded_buy(audit_db: sa.Engine) -> None:
    store = SqlAuditStore(audit_db, AppConfig())
    store.start_run("abc123", "analyze")
    store.record_analysis(_analysis("proposed", BUY))


def test_critique_command_reviews_a_recorded_run(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _recorded_buy(audit_db)
    reviews = _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    result = runner.invoke(cli.app, ["critique", "abc123", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert reviews == [("abc123", 1)]  # tool calls (with outputs) loaded from the store
    assert "CRITIC APPROVED" in result.output
    assert _rows(audit_db, schema.critiques) == []  # a re-review records nothing

    as_json = runner.invoke(
        cli.app, ["critique", "abc123", "--json", "--config-dir", str(config_dir)]
    )
    assert json.loads(as_json.stdout)["status"] == "approved"


def test_critique_command_errors(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY))
    missing = runner.invoke(cli.app, ["critique", "nope", "--config-dir", str(config_dir)])
    assert missing.exit_code == 1
    assert "No verified buy proposal" in missing.output

    _recorded_buy(audit_db)
    monkeypatch.setattr(Container, "build_critic", lambda self: None)
    disabled = runner.invoke(cli.app, ["critique", "abc123", "--config-dir", str(config_dir)])
    assert disabled.exit_code == 2
    assert "disabled" in disabled.output

    _use_fake_analyst(monkeypatch, _analysis("proposed", BUY), _critique("failed"))
    failed = runner.invoke(cli.app, ["critique", "abc123", "--config-dir", str(config_dir)])
    assert failed.exit_code == 1


# ---------- evaluate / evaluate-critic ----------

GO_CONFIG = (
    "environment: paper\nbroker:\n  paper: true\n"
    "evaluation:\n  min_cycle_days: 5\n  min_closed_trades: 2\n"
    "costs:\n  llm_prices:\n    gpt-6.1-sol: {input: 0.10, output: 1.00}\n"
)


@pytest.fixture
def history(audit_db: sa.Engine, monkeypatch: pytest.MonkeyPatch) -> SqlAuditStore:
    from tests.unit.test_evaluation import _bars, _history

    store = SqlAuditStore(audit_db, AppConfig())
    _history(store)
    monkeypatch.setattr(Container, "build_market_data", lambda self: _bars())
    return store


def test_evaluate_go_exits_0_and_saves(
    config_dir: Path, history: SqlAuditStore, tmp_path: Path
) -> None:
    (config_dir / "paper.yaml").write_text(GO_CONFIG, encoding="utf-8")
    result = runner.invoke(
        cli.app,
        ["evaluate", "--from", "2026-10-12", "--to", "2026-10-16", "--config-dir", str(config_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "## Decision: **GO**" in result.stdout
    assert (tmp_path / "reports" / "evaluation-2026-10-16.md").exists()
    assert (tmp_path / "reports" / "evaluation-2026-10-16.json").exists()


def test_evaluate_with_the_real_bar_is_no_go(config_dir: Path, history: SqlAuditStore) -> None:
    # The shipped thresholds (40 days, 20 trades) cannot be met by five days of history.
    result = runner.invoke(
        cli.app,
        ["evaluate", "--to", "2026-10-16", "--no-save", "--json", "--config-dir", str(config_dir)],
    )
    assert result.exit_code == 1
    card = json.loads(result.stdout)
    assert card["decision"] == "NO-GO"
    assert card["start"] == "2026-10-12"  # defaults to the first recorded daily cycle


def test_evaluate_needs_a_database_and_a_valid_window(
    config_dir: Path, audit_db: sa.Engine
) -> None:
    bad = runner.invoke(
        cli.app,
        ["evaluate", "--from", "2026-10-16", "--to", "2026-10-12", "--config-dir", str(config_dir)],
    )
    assert bad.exit_code == 2
    junk = runner.invoke(cli.app, ["evaluate", "--from", "soon", "--config-dir", str(config_dir)])
    assert junk.exit_code == 2


def test_evaluate_without_database(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["evaluate", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "DATABASE_URL is not set" in result.output


def test_evaluate_save_failure(config_dir: Path, history: SqlAuditStore, tmp_path: Path) -> None:
    (tmp_path / "reports").write_text("not a folder", encoding="utf-8")
    result = runner.invoke(
        cli.app, ["evaluate", "--to", "2026-10-16", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "Could not save the scorecard" in result.output


def test_evaluate_critic_replay(
    config_dir: Path, history: SqlAuditStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_evaluation import _Critic

    monkeypatch.setattr(Container, "build_critic", lambda self: _Critic("vetoed"))
    result = runner.invoke(
        cli.app,
        [
            "evaluate-critic",
            "--from",
            "2026-10-12",
            "--to",
            "2026-10-16",
            "--config-dir",
            str(config_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "Replayed 4 proposal(s) from 2026-10-12 to 2026-10-16: 1 agree, 3 now vetoed"
        in result.output
    )
    assert "daily-20261012 AAPL: approved -> vetoed. now vetoed" in result.output

    as_json = runner.invoke(
        cli.app,
        ["evaluate-critic", "--to", "2026-10-16", "--json", "--config-dir", str(config_dir)],
    )
    assert json.loads(as_json.stdout)["reviewed"] == 4


def test_evaluate_critic_lists_skipped_runs(
    config_dir: Path, audit_db: sa.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date as day_type

    from tests.unit.test_evaluation import _Critic, _record_day

    store = SqlAuditStore(audit_db, AppConfig())
    _record_day(store, day_type(2026, 10, 12), "traded", with_outputs=False)
    monkeypatch.setattr(Container, "build_critic", lambda self: _Critic("approved"))
    monkeypatch.setattr(Container, "build_market_data", lambda self: FakeMarketData())
    result = runner.invoke(
        cli.app,
        [
            "evaluate-critic",
            "--from",
            "2026-10-12",
            "--to",
            "2026-10-12",
            "--config-dir",
            str(config_dir),
        ],
    )
    assert "skipped daily-20261012: no tool outputs recorded" in result.output


def test_evaluate_critic_disabled(
    config_dir: Path, history: SqlAuditStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Container, "build_critic", lambda self: None)
    result = runner.invoke(cli.app, ["evaluate-critic", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "disabled" in result.output


# ---------- live-check ----------


def _fake_live_checks(monkeypatch: pytest.MonkeyPatch, *, passing: bool) -> None:
    from trading_agent.health import CheckResult, CheckStatus

    status = CheckStatus.PASS if passing else CheckStatus.FAIL
    results = [
        CheckResult("Live switches", CheckStatus.PASS, "live config, confirmed"),
        CheckResult(
            "Paper evaluation", status, "GO" if passing else "evaluation-x.json says NO-GO"
        ),
    ]
    monkeypatch.setattr(Container, "live_checks", lambda self: results)


def test_live_check_needs_the_live_environment(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["live-check", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "--env live" in result.output


def test_live_check_ready(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_live_checks(monkeypatch, passing=True)
    result = runner.invoke(
        cli.app, ["live-check", "--env", "live", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "PASS  Paper evaluation" in result.output
    assert "Ready: live entries are allowed." in result.output


def test_live_check_blocked(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_live_checks(monkeypatch, passing=False)
    result = runner.invoke(
        cli.app, ["live-check", "--env", "live", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "FAIL  Paper evaluation     evaluation-x.json says NO-GO" in result.output
    assert "Live entries are BLOCKED. Exits (manage, flatten) still work." in result.output


def test_env_file_selects_a_separate_secrets_file(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_env = tmp_path / ".env.live"
    live_env.write_text("LIVE_TRADING_CONFIRMED=true\nALPACA_API_KEY=live-key\n", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(live_env))
    container = cli._container(None, config_dir)
    assert container.secrets.live_trading_confirmed is True
    assert container.secrets.require("alpaca_api_key") == "live-key"

    monkeypatch.delenv("ENV_FILE")
    assert cli._container(None, config_dir).secrets.live_trading_confirmed is False


# ---------- Lessons (self-learning) ----------


def _add_lessons(engine: sa.Engine) -> list[int]:
    from datetime import timedelta

    from trading_agent.storage.store import NewLessonRow

    store = SqlAuditStore(engine, AppConfig())
    now = datetime.now(UTC)
    ids = store.record_lessons(
        None,
        [
            NewLessonRow(text="Check volume first.", category="research", evidence={}),
            NewLessonRow(text="Gap-ups fade by noon.", category="entry", evidence={}),
        ],
        expires_at=now + timedelta(days=30),
        max_active=15,
    )
    store.record_lessons(
        None,
        [NewLessonRow(text="An expired lesson.", category="market", evidence={})],
        expires_at=now - timedelta(days=1),
        max_active=15,
    )
    return ids


def test_lessons_lists_active_then_all(config_dir: Path, audit_db: sa.Engine) -> None:
    empty = runner.invoke(cli.app, ["lessons", "--config-dir", str(config_dir)])
    assert empty.exit_code == 0
    assert "No lessons yet" in empty.output

    first, second = _add_lessons(audit_db)
    retired = runner.invoke(
        cli.app,
        ["retire-lesson", str(second), "--reason", "too vague", "--config-dir", str(config_dir)],
    )
    assert retired.exit_code == 0, retired.output
    assert f"Retired lesson #{second}." in retired.output

    active = runner.invoke(cli.app, ["lessons", "--config-dir", str(config_dir)])
    assert "[research] Check volume first." in active.output
    assert "Gap-ups" not in active.output
    assert "expired" in active.output  # still active in the table, but past its date

    everything = runner.invoke(cli.app, ["lessons", "--all", "--config-dir", str(config_dir)])
    assert "retired  [entry] Gap-ups fade by noon." in everything.output
    assert "retired: by operator: too vague" in everything.output

    as_json = runner.invoke(cli.app, ["lessons", "--json", "--config-dir", str(config_dir)])
    assert {row["id"] for row in json.loads(as_json.stdout)} >= {first}


def test_retire_unknown_lesson_exits_1(config_dir: Path, audit_db: sa.Engine) -> None:
    result = runner.invoke(
        cli.app, ["retire-lesson", "999", "--reason", "x", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "No active lesson #999." in result.output


class _Learner:
    def __init__(self, outcome: str | Exception) -> None:
        self.outcome = outcome
        self.days: list[date] = []

    def learn(self, day: date) -> str:
        self.days.append(day)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _use_learner(monkeypatch: pytest.MonkeyPatch, learner: _Learner) -> None:
    monkeypatch.setattr(Container, "build_learning_service", lambda self, store=None: learner)


def test_learn_reviews_a_day(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    learner = _Learner("1 new, 0 retired: ok")
    _use_learner(monkeypatch, learner)
    result = runner.invoke(
        cli.app, ["learn", "--date", "2026-10-15", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "daily-20261015: 1 new, 0 retired: ok" in result.output
    assert learner.days == [date(2026, 10, 15)]


def test_learn_failure_and_bad_input(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, audit_db: sa.Engine
) -> None:
    from trading_agent.learning import LearningError

    no_key = runner.invoke(cli.app, ["learn", "--config-dir", str(config_dir)])
    assert no_key.exit_code == 2
    assert "Configuration error" in no_key.output

    _use_learner(monkeypatch, _Learner(LearningError("reflector unavailable: boom")))
    failed = runner.invoke(cli.app, ["learn", "--config-dir", str(config_dir)])
    assert failed.exit_code == 1
    assert "Review failed: reflector unavailable: boom" in failed.output

    bad = runner.invoke(cli.app, ["learn", "--date", "soon", "--config-dir", str(config_dir)])
    assert bad.exit_code == 2


def test_lesson_commands_need_a_database(config_dir: Path) -> None:
    for args in (["lessons"], ["retire-lesson", "1", "--reason", "x"], ["learn"]):
        result = runner.invoke(cli.app, [*args, "--config-dir", str(config_dir)])
        assert result.exit_code == 2, args
        assert "DATABASE_URL is not set" in result.output


# ---------- costs ----------


def test_costs_command(config_dir: Path, audit_db: sa.Engine) -> None:
    from trading_agent.domain.models import LLMCallRecord

    store = SqlAuditStore(audit_db, AppConfig())
    now = datetime.now(UTC)
    for agent, cost in (("analyst", Decimal("0.40")), ("critic", Decimal("0.10"))):
        store.record_llm_call(
            LLMCallRecord(
                created_at=now,
                provider="openai",
                model="gpt-6.1-sol",
                agent=agent,
                run_id=None,
                status="ok",
                usage=LLMUsage(input_tokens=1000, output_tokens=100),
                latency_ms=1.0,
            ),
            cost,
        )
    result = runner.invoke(cli.app, ["costs", "--days", "2", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "analyst $0.40" in result.output
    assert "critic $0.10" in result.output
    assert "Total: $0.50" in result.output
    assert "Daily LLM budget: $2.00" in result.output

    as_json = runner.invoke(
        cli.app, ["costs", "--days", "1", "--json", "--config-dir", str(config_dir)]
    )
    assert Decimal(json.loads(as_json.stdout)["total"]["llm_usd"]) == Decimal("0.50")


def test_costs_needs_a_database(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["costs", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "DATABASE_URL is not set" in result.output


def test_risk_check_shows_the_score_breakdown(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_risk(monkeypatch, FakeBroker())
    result = runner.invoke(
        cli.app, ["risk-check", "AAPL", "--price", "100.05", "--config-dir", str(config_dir)]
    )
    # A manual check marks the analyst's confidence as low.
    assert "Trade score 83.0/100 (needs 60)" in result.output
    assert "  Technicals           0.75  x20  RSI 100 overbought" in result.output
    assert (
        "  Reward/risk          0.88  x20  1.88 after spread (10% target, 5% stop, 0.100% spread)"
        in result.output
    )
    assert (
        "  History              0.50  x10  0 of 10 closed trades needed: neutral" in result.output
    )


def test_repo_config_trades_three_times_a_day_on_paper_only(config_dir: Path) -> None:
    from datetime import time

    from trading_agent.settings import load_config

    paper = load_config("paper", config_dir)
    assert (paper.risk.max_trades_per_day, paper.risk.max_open_positions) == (3, 3)
    assert paper.schedule.extra_analyses == (time(10, 30), time(12))
    assert paper.scoring.enabled
    assert paper.scoring.min_score == 60
    live = load_config("live", config_dir)
    assert (live.risk.max_trades_per_day, live.risk.max_open_positions) == (1, 1)


# ---------- watch and plan (ADR 0016) ----------


def test_watch_command_scans_now(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import time as clock_time

    from tests.unit.test_market_watch import snap

    zone = ZoneInfo("America/New_York")
    now = datetime.now(zone)
    traded = max(now, datetime.combine(now.date(), clock_time(4, 5), tzinfo=zone))
    market = FakeMarketData(
        snapshots={
            "AAPL": snap("AAPL", "104", traded, "100", now - timedelta(days=1)),
            "MSFT": snap("MSFT", "100.5", traded, "100", now - timedelta(days=1)),
        }
    )
    monkeypatch.setattr(Container, "build_market_data", lambda self: market)
    monkeypatch.setattr(Container, "build_news", lambda self: FakeNews())
    result = runner.invoke(
        cli.app, ["watch", "AAPL", "MSFT", "--phase", "pre", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "Pre-market watch" in result.output
    assert "AAPL" in result.output
    assert "ALERT" in result.output

    as_json = runner.invoke(
        cli.app, ["watch", "--phase", "pre", "--json", "--config-dir", str(config_dir)]
    )
    assert json.loads(as_json.stdout)["phase"] == "pre"


def test_watch_command_needs_market_data_keys(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["watch", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "Configuration error" in result.output


def test_plan_command(config_dir: Path, audit_db: sa.Engine) -> None:
    from trading_agent.domain.models import DayPlan, PlanItem

    store = SqlAuditStore(audit_db, AppConfig())
    empty = runner.invoke(
        cli.app, ["plan", "--date", "2026-10-16", "--config-dir", str(config_dir)]
    )
    assert empty.exit_code == 1
    assert "No plan for 2026-10-16" in empty.output

    store.record_plan(
        None,
        DayPlan(
            source_day=date(2026, 10, 15),
            for_day=date(2026, 10, 16),
            created_at=NOW,
            items=(PlanItem(kind="ops", symbol=None, text="Fix the thing.", priority=1),),
            summary="1 item(s) for 2026-10-16: 1 ops",
        ),
    )
    result = runner.invoke(
        cli.app, ["plan", "--date", "2026-10-16", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "1 item(s) for 2026-10-16: 1 ops" in result.output
    assert "1  ops     Fix the thing." in result.output

    as_json = runner.invoke(
        cli.app, ["plan", "--date", "2026-10-16", "--json", "--config-dir", str(config_dir)]
    )
    assert json.loads(as_json.stdout)["items"][0]["kind"] == "ops"
    bad = runner.invoke(cli.app, ["plan", "--date", "soon", "--config-dir", str(config_dir)])
    assert bad.exit_code == 2


def test_watch_rejects_an_unknown_phase(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["watch", "--phase", "lunch", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "--phase must be pre or post" in result.output


def test_watch_shows_data_problems_as_warnings(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    market = FakeMarketData(fail_snapshots=True)
    monkeypatch.setattr(Container, "build_market_data", lambda self: market)
    monkeypatch.setattr(Container, "build_news", lambda self: FakeNews())
    result = runner.invoke(cli.app, ["watch", "AAPL", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    assert "Warning: snapshots unavailable" in result.output


# ---------- guidebook and playbooks (ADR 0017) ----------


def test_guidebook_command(config_dir: Path, audit_db: sa.Engine, tmp_path: Path) -> None:
    store = SqlAuditStore(audit_db, AppConfig())
    none = runner.invoke(cli.app, ["guidebook", "--config-dir", str(config_dir)])
    assert none.exit_code == 1
    assert "No guidebook yet" in none.output

    store.record_guidebook("# Guidebook\nfirst", ["first edition"])
    store.record_guidebook("# Guidebook\nsecond", ["promoted lesson #1"])
    latest = runner.invoke(cli.app, ["guidebook", "--config-dir", str(config_dir)])
    assert latest.exit_code == 0
    assert "second" in latest.output
    assert "Version 2: promoted lesson #1" in latest.output
    first = runner.invoke(
        cli.app, ["guidebook", "--version", "1", "--save", "--config-dir", str(config_dir)]
    )
    assert "first" in first.output
    assert (tmp_path / "reports" / "guidebook.md").read_text(encoding="utf-8").endswith("first")
    missing = runner.invoke(
        cli.app, ["guidebook", "--version", "9", "--config-dir", str(config_dir)]
    )
    assert missing.exit_code == 1


def test_playbooks_commands(config_dir: Path, audit_db: sa.Engine) -> None:
    from tests.unit.test_guidebook_store import playbook

    store = SqlAuditStore(audit_db, AppConfig())
    empty = runner.invoke(cli.app, ["playbooks", "--config-dir", str(config_dir)])
    assert "No playbooks yet" in empty.output
    store.record_playbook(playbook("gap-check"))
    listed = runner.invoke(cli.app, ["playbooks", "--config-dir", str(config_dir)])
    assert "gap-check  candidate  Earnings gap check" in listed.output
    assert "1. Check the gap size with get_quote." in listed.output

    retired = runner.invoke(
        cli.app,
        ["retire-playbook", "gap-check", "--reason", "too vague", "--config-dir", str(config_dir)],
    )
    assert retired.exit_code == 0
    assert "Retired playbook gap-check." in retired.output
    again = runner.invoke(
        cli.app, ["retire-playbook", "gap-check", "--reason", "x", "--config-dir", str(config_dir)]
    )
    assert again.exit_code == 1
    everything = runner.invoke(cli.app, ["playbooks", "--all", "--config-dir", str(config_dir)])
    assert "retired: by operator: too vague" in everything.output


# ---------- dashboard (ADR 0018) ----------


def test_dashboard_serves_locally(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    monkeypatch.setattr(cli, "_port_busy", lambda host, port: False)

    served: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.append(kw))
    result = runner.invoke(cli.app, ["dashboard", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert served == [{"host": "127.0.0.1", "port": 8765, "log_level": "warning"}]
    assert "http://127.0.0.1:8765" in result.output
    assert "read-only" in result.output


def test_dashboard_refuses_the_network_unless_allowed(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    monkeypatch.setattr(cli, "_port_busy", lambda host, port: False)

    served: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.append(kw))
    refused = runner.invoke(
        cli.app, ["dashboard", "--host", "0.0.0.0", "--config-dir", str(config_dir)]
    )
    assert refused.exit_code == 2
    assert "no login" in refused.output
    assert served == []
    allowed = runner.invoke(
        cli.app,
        ["dashboard", "--host", "0.0.0.0", "--allow-network", "--config-dir", str(config_dir)],
    )
    assert allowed.exit_code == 0
    assert served[0]["host"] == "0.0.0.0"


def test_dashboard_explains_a_busy_port(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    import uvicorn

    served: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.append(kw))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        result = runner.invoke(
            cli.app, ["dashboard", "--port", str(port), "--config-dir", str(config_dir)]
        )
    assert result.exit_code == 2
    assert f"Port {port} is already in use" in result.output
    assert "--port" in result.output
    assert served == []


def test_an_unknown_host_is_left_for_the_server_to_report() -> None:
    assert cli._port_busy("no-such-host.invalid", 8765) is False


def test_technicals_command(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_research_tools import _technical_data, _tools

    tools = _tools(_technical_data())
    monkeypatch.setattr(Container, "build_research_tools", lambda self: tools)
    result = runner.invoke(cli.app, ["technicals", "aapl", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "AAPL technicals" in result.output
    assert "Bias:" in result.output
    assert "RSI" in result.output
    assert "200-day average" in result.output
    assert "Support" in result.output or "supports" in result.output
    assert "describe the past" in result.output

    as_json = runner.invoke(
        cli.app, ["technicals", "AAPL", "--json", "--config-dir", str(config_dir)]
    )
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output)["symbol"] == "AAPL"


def test_technicals_command_reports_bad_input(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_research_tools import _technical_data, _tools

    monkeypatch.setattr(Container, "build_research_tools", lambda self: _tools(_technical_data()))
    result = runner.invoke(cli.app, ["technicals", "TSLA", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "no daily bars available for TSLA" in result.output


def test_technicals_command_needs_market_data_keys(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["technicals", "AAPL", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "Configuration error" in result.output


# ---------- Phone alerts and crash restart (ADR 0021) ----------


class _Phone:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    def send(self, text: str) -> None:
        from trading_agent.alerts import AlertError

        if self.fail:
            raise AlertError("telegram rejected the alert: HTTP 401")
        self.sent.append(text)


def _phone(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> _Phone:
    from trading_agent.alerts import Alerter
    from trading_agent.settings.models import AlertsConfig

    phone = _Phone(fail)
    alerter = Alerter(phone, AlertsConfig(), label="paper")
    monkeypatch.setattr(Container, "alerter", property(lambda self: alerter))
    return phone


def test_run_announces_the_start(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phone = _phone(monkeypatch)
    _use_fake_cycle(monkeypatch, TRADED)
    result = runner.invoke(cli.app, ["run", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert phone.sent == ["[paper] Trading cycle started."]


def test_run_restarts_after_a_crash_and_gives_up_on_a_loop(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import functools

    from trading_agent.orchestration.supervisor import supervise

    phone = _phone(monkeypatch)
    runs: list[int] = []

    class _Crashing:
        def __init__(self, services: object, config: object) -> None:
            pass

        def run_forever(self) -> None:
            runs.append(1)
            raise RuntimeError("database is locked")

    monkeypatch.setattr(Container, "build_cycle_services", lambda self: object())
    monkeypatch.setattr(cli, "DailyCycle", _Crashing)
    monkeypatch.setattr(cli, "supervise", functools.partial(supervise, sleep=lambda s: None))
    result = runner.invoke(cli.app, ["run", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert len(runs) == 5  # orchestration.max_crashes_per_hour
    assert "stopped after 5 crashes" in phone.sent[-1]
    assert "stopped after 5 crashes" in result.output


def test_alerts_setup_explains_the_first_step(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["alerts-setup", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "TELEGRAM_BOT_TOKEN is not set" in result.output
    assert "@BotFather" in result.output


def test_alerts_setup_lists_the_chats(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent.alerts import TelegramClient

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:abc")
    monkeypatch.setattr(TelegramClient, "find_chats", lambda self: [("987654", "Sai")])
    result = runner.invoke(cli.app, ["alerts-setup", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "TELEGRAM_CHAT_ID=987654" in result.output
    assert "1:abc" not in result.output


def test_alerts_setup_with_no_messages_yet(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_agent.alerts import TelegramClient

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:abc")
    monkeypatch.setattr(TelegramClient, "find_chats", lambda self: [])
    result = runner.invoke(cli.app, ["alerts-setup", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "send your bot a message" in result.output


def test_alerts_setup_reports_a_bad_token(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_agent.alerts import AlertError, TelegramClient

    def rejected(self: object) -> list[tuple[str, str]]:
        raise AlertError("telegram rejected the alert: HTTP 401")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:abc")
    monkeypatch.setattr(TelegramClient, "find_chats", rejected)
    result = runner.invoke(cli.app, ["alerts-setup", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "HTTP 401" in result.output


def test_alerts_test_sends_one_message(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phone = _phone(monkeypatch)
    result = runner.invoke(cli.app, ["alerts-test", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert len(phone.sent) == 1
    assert "test alert" in phone.sent[0]
    assert "Sent" in result.output


def test_alerts_test_reports_a_failure(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _phone(monkeypatch, fail=True)
    result = runner.invoke(cli.app, ["alerts-test", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "not sent" in result.output


def test_alerts_test_without_telegram(config_dir: Path) -> None:
    result = runner.invoke(cli.app, ["alerts-test", "--config-dir", str(config_dir)])
    assert result.exit_code == 2
    assert "alerts-setup" in result.output


# ---------- Chat (ADR 0022) ----------


def test_run_starts_the_chat_listener(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    started: list[threading.Event] = []

    class _Listener:
        def start(self) -> threading.Event:
            stop = threading.Event()
            started.append(stop)
            return stop

    monkeypatch.setattr(Container, "build_chat_listener", lambda self: _Listener())
    _use_fake_cycle(monkeypatch, TRADED)
    result = runner.invoke(cli.app, ["run", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "Answering your Telegram messages" in result.output
    (stop,) = started
    assert stop.is_set()  # stopped when the cycle stopped


def test_run_without_chat_says_nothing_about_it(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_cycle(monkeypatch, TRADED)
    result = runner.invoke(cli.app, ["run", "--config-dir", str(config_dir)])
    assert "Telegram messages" not in result.output


def test_pause_and_resume_from_the_computer(config_dir: Path, tmp_path: Path) -> None:
    import yaml

    base = config_dir / "base.yaml"
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    data["chat"]["pause_file"] = str(tmp_path / "pause.json")
    base.write_text(yaml.safe_dump(data), encoding="utf-8")

    paused = runner.invoke(cli.app, ["pause", "--config-dir", str(config_dir)])
    assert paused.exit_code == 0, paused.output
    assert "No new trades for the rest of today" in paused.output
    assert (tmp_path / "pause.json").exists()
    resumed = runner.invoke(cli.app, ["resume", "--config-dir", str(config_dir)])
    assert "Trading resumed" in resumed.output
    assert not (tmp_path / "pause.json").exists()
    again = runner.invoke(cli.app, ["resume", "--config-dir", str(config_dir)])
    assert "was not paused" in again.output


def test_chat_command_answers_like_the_bot(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Bot:
        def reply(self, text: str) -> str:
            return f"you said {text}"

    monkeypatch.setattr(Container, "build_chat_bot", lambda self: _Bot())
    result = runner.invoke(cli.app, ["chat", "hi", "there", "--config-dir", str(config_dir)])
    assert result.exit_code == 0, result.output
    assert "you said hi there" in result.output
