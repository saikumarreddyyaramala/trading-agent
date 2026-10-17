from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

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
            return RiskEngine(
                broker,
                healthy_market_data("AAPL"),
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
    "reporting:\n  llm_usd_per_million_input_tokens: 0.10\n"
    "  llm_usd_per_million_output_tokens: 1.00\n"
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
