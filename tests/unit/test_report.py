from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import sqlalchemy as sa

from tests.conftest import FakeBroker, FakeMarketData
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.domain.models import (
    Bar,
    BrokerOrder,
    Evidence,
    LLMUsage,
    OrderPlan,
    RiskCheck,
    RiskDecision,
    TradeProposal,
)
from trading_agent.execution.executor import ManageResult
from trading_agent.reporting.delivery import DeliveryError, post_webhook, save_report
from trading_agent.reporting.report import DailyReport, ReportBuilder, render_markdown
from trading_agent.settings.models import AppConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import AuditError, NullAuditStore, SqlAuditStore

NY = ZoneInfo("America/New_York")
DAY = date(2026, 10, 15)
RUN = "daily-20261015"
GENERATED = datetime(2026, 10, 15, 20, 15, tzinfo=UTC)


def _ny(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 15, hour, minute, tzinfo=NY)


def _order(client_id: str, side: str, role_price: str, at: datetime, **extra: Any) -> BrokerOrder:
    return BrokerOrder(
        id=f"id-{client_id}",
        client_order_id=client_id,
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        order_type="limit" if side == "buy" else "market",
        status="filled",
        quantity=Decimal("0.5"),
        filled_quantity=Decimal("0.5"),
        filled_avg_price=Decimal(role_price),
        submitted_at=at,
        **extra,
    )


PROPOSAL = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="Strong earnings and a tight spread.",
    reference_price=Decimal("100"),
    confidence="medium",
    evidence=(
        Evidence(claim="Beat estimates", source_call_id="c1"),
        Evidence(claim="Spread 0.02%", source_call_id="c2"),
    ),
    risks=("Market reversal",),
)


def _spy(open_: str = "600", close: str = "603") -> FakeMarketData:
    bar = Bar(
        symbol="SPY",
        timestamp=datetime(2026, 10, 15, 4, tzinfo=UTC),
        open=Decimal(open_),
        high=Decimal("610"),
        low=Decimal("590"),
        close=Decimal(close),
        volume=1,
    )
    return FakeMarketData(bars={"SPY": (bar,)})


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: GENERATED)


def _record_trading_day(store: SqlAuditStore, *, exit_price: str = "102") -> None:
    store.start_run(RUN, "daily")
    store.record_analysis(
        AnalystResult(
            run_id=RUN,
            status="proposed",
            reason="evidence verified",
            proposal=PROPOSAL,
            prompt_version="p",
            turns=4,
            tool_calls=(
                ToolCallRecord(call_id="c1", tool="get_news", arguments={}, is_error=False),
                ToolCallRecord(call_id="c2", tool="get_quote", arguments={}, is_error=False),
            ),
            usage=LLMUsage(input_tokens=18_000, output_tokens=2_000),
            started_at=_ny(8, 45),
            duration_seconds=30,
        )
    )
    plan = OrderPlan(
        symbol="AAPL",
        quantity=Decimal("0.5"),
        fractional=True,
        notional_usd=Decimal("50"),
        entry_reference_price=Decimal("100"),
        limit_price=Decimal("102"),
        stop_loss_price=Decimal("95"),
        take_profit_price=Decimal("110"),
    )
    store.record_risk_decision(
        RUN,
        RiskDecision(
            approved=True,
            symbol="AAPL",
            checks=(RiskCheck(name="All", passed=True, detail=""),),
            plan=plan,
            evaluated_at=_ny(9, 45),
        ),
    )
    store.record_orders(
        RUN,
        [
            _order(f"ta-{RUN}-entry", "buy", "100", _ny(9, 45)),
            _order("close-1", "sell", exit_price, _ny(15, 50)),
        ],
    )
    store.record_position_event(
        RUN, ManageResult(action="force_close", detail="force close at 15:50", symbol="AAPL")
    )
    for step, status in (("preflight", "ok"), ("entry", "entered"), ("close", "ok")):
        store.record_step(RUN, step, status, "", _ny(8))
    store.finish_run(RUN, "traded", "entered 0.5 AAPL at $100")


def _builder(
    store: Any, *, market_data: Any = None, broker: Any = None, config: AppConfig | None = None
) -> ReportBuilder:
    return ReportBuilder(
        store,
        market_data or _spy(),
        broker or FakeBroker(equity="100001.00"),
        config or AppConfig(),
        clock=lambda: GENERATED,
    )


# ---------- Building ----------


def test_trading_day_report(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    report = _builder(store).build(DAY)

    assert report.run_id == RUN
    assert report.cycle_status == "traded"
    assert report.steps == {"preflight": "ok", "entry": "entered", "close": "ok"}
    assert report.analysis is not None
    assert report.analysis.symbol == "AAPL"
    assert report.analysis.tool_calls == 2
    assert report.analysis.cost_usd is None  # no prices configured: tokens only
    assert report.risk is not None
    assert report.risk.approved
    assert report.trade is not None
    assert report.trade.pnl_usd == Decimal("1.00")
    assert report.trade.return_pct == Decimal("2.00")
    assert report.benchmark_pct == Decimal("0.50")
    assert report.excess_pct == Decimal("1.50")
    assert report.exits == ["force_close: force close at 15:50"]
    assert report.account_equity == Decimal("100001.00")
    assert report.performance.trades == 1
    assert report.performance.win_rate_pct == Decimal("100.00")
    assert report.llm.input_tokens == 18_000
    assert report.warnings == []


def test_llm_cost_appears_only_with_configured_prices(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    priced = AppConfig.model_validate(
        {
            "reporting": {
                "llm_usd_per_million_input_tokens": "2.50",
                "llm_usd_per_million_output_tokens": "10",
            }
        }
    )
    report = _builder(store, config=priced).build(DAY)
    assert report.analysis is not None
    assert report.analysis.cost_usd == Decimal("0.0650")  # 18k x $2.50/M + 2k x $10/M
    assert report.llm.cost_usd == Decimal("0.0650")


def test_day_without_a_cycle(store: SqlAuditStore) -> None:
    report = _builder(store).build(DAY)
    assert report.cycle_status is None
    assert report.analysis is None
    assert report.risk is None
    assert report.trade is None
    assert report.performance.trades == 0


def test_rejected_day(store: SqlAuditStore) -> None:
    store.start_run(RUN, "daily")
    store.record_risk_decision(
        RUN,
        RiskDecision(
            approved=False,
            symbol="AAPL",
            checks=(RiskCheck(name="Spread", passed=False, detail="wide"),),
            plan=None,
            evaluated_at=_ny(9, 45),
        ),
    )
    report = _builder(store).build(DAY)
    assert report.risk is not None
    assert report.risk.failed_checks == ["Spread"]
    assert "Rejected. Failed: Spread." in render_markdown(report)


def test_missing_data_becomes_warnings_not_failures(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    report = _builder(
        store, market_data=FakeMarketData(fail=True), broker=FakeBroker(fail_account=True)
    ).build(DAY)
    assert report.benchmark_pct is None
    assert report.account_equity is None
    # FakeMarketData(fail=True) only fails quotes; bars are simply missing here.
    assert any("account unavailable" in w for w in report.warnings)
    assert report.trade is not None


def test_benchmark_outage_is_a_warning(store: SqlAuditStore) -> None:
    class _Down(FakeMarketData):
        def get_daily_bars(self, symbols: Any, *, days: int) -> Any:
            raise RuntimeError("data provider down")

    report = _builder(store, market_data=_Down()).build(DAY)
    assert report.warnings == ["benchmark data unavailable: data provider down"]


def test_unreadable_audit_store_is_a_warning() -> None:
    class _Down(NullAuditStore):
        def get_run(self, run_id: str) -> None:
            raise AuditError("db down")

        def orders_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
            raise AuditError("db down")

        def analysis_totals(self, run_id_prefix: str) -> dict[str, int]:
            raise AuditError("db down")

    report = _builder(_Down()).build(DAY)
    assert report.warnings == ["audit store unavailable: db down", "orders unavailable: db down"]
    assert report.llm.analyses == 0


# ---------- Rendering ----------


def test_markdown_tells_the_whole_story(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    markdown = render_markdown(_builder(store).build(DAY))
    for expected in [
        "# Daily report: Thursday 2026-10-15",
        "**TRADED**: entered 0.5 AAPL at $100",
        "| entry | entered |",
        "**BUY AAPL** (confidence medium). 2 tool calls, 20,000 tokens.",
        "> Strong earnings and a tight spread.",
        "- Beat estimates (`c1`)",
        "- Market reversal",
        "Approved: every check passed.",
        "| AAPL | 0.5 | $100.0000 | $102.0000 | $1.00 | +2.00% | exit order |",
        "- force_close: force close at 15:50",
        "Benchmark (SPY open to close): +0.50%. Excess return: +1.50 pts.",
        "| 1 | 1 | 100.00% | $1.00 | +2.00% | +0.50% | +1.50 pts |",
        "Account equity: $100,001.00 (paper).",
        "not financial advice",
    ]:
        assert expected in markdown, expected


def test_markdown_for_a_losing_trade_and_an_empty_day(store: SqlAuditStore) -> None:
    _record_trading_day(store, exit_price="95")
    losing = render_markdown(_builder(store).build(DAY))
    assert "| -$2.50 | -5.00% |" in losing

    empty = render_markdown(_builder(NullAuditStore(), market_data=FakeMarketData()).build(DAY))
    assert "**NO CYCLE RECORDED**" in empty
    assert "No analysis recorded." in empty
    assert "No risk check ran." in empty
    assert "No trade today." in empty
    assert "Benchmark (SPY open to close): n/a." in empty


@pytest.mark.parametrize(
    ("decision", "status", "headline"),
    [
        ("no_trade", "no_trade", "**NO TRADE** (confidence high)"),
        (None, "failed", "**FAILED**: boom"),
    ],
)
def test_markdown_analysis_headlines(
    store: SqlAuditStore, decision: str | None, status: str, headline: str
) -> None:
    store.start_run(RUN, "daily")
    proposal = (
        TradeProposal(decision="no_trade", thesis="Quiet tape.", confidence="high")
        if decision
        else None
    )
    store.record_analysis(
        AnalystResult(
            run_id=RUN,
            status=status,  # type: ignore[arg-type]
            reason="boom" if status == "failed" else "chose not to trade",
            proposal=proposal,
            prompt_version="p",
            turns=1,
            tool_calls=(),
            usage=LLMUsage(),
            started_at=_ny(8, 45),
            duration_seconds=1,
        )
    )
    assert headline in render_markdown(_builder(store).build(DAY))


def test_warnings_are_rendered(store: SqlAuditStore) -> None:
    report = _builder(store, broker=FakeBroker(fail_account=True)).build(DAY)
    markdown = render_markdown(report)
    assert "Warnings:" in markdown
    assert "Account equity: n/a" in markdown


# ---------- Delivery ----------


def test_save_report_writes_markdown_and_json(store: SqlAuditStore, tmp_path: Path) -> None:
    _record_trading_day(store)
    report = _builder(store).build(DAY)
    md_path, json_path = save_report(report, render_markdown(report), tmp_path / "reports")
    assert md_path.name == "2026-10-15.md"
    assert md_path.read_text(encoding="utf-8").startswith("# Daily report")
    assert DailyReport.model_validate(json.loads(json_path.read_text(encoding="utf-8"))) == report


def test_save_report_failure(store: SqlAuditStore, tmp_path: Path) -> None:
    blocker = tmp_path / "reports"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    report = _builder(store).build(DAY)
    with pytest.raises(DeliveryError, match="could not save the report"):
        save_report(report, "x", blocker)


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_webhook_posts_slack_style_text() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text="ok")

    post_webhook("https://hooks.example/secret-token", "# Report", client=_client(handler))
    assert seen == [{"text": "# Report"}]


def test_long_reports_are_truncated_for_the_webhook() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["text"])
        return httpx.Response(200)

    post_webhook("https://hooks.example/x", "x" * 50_000, client=_client(handler))
    assert len(seen[0]) < 40_000
    assert seen[0].endswith("…")


def test_webhook_rejection_and_outage_never_leak_the_url() -> None:
    secret = "https://hooks.example/T000/B000/very-secret"
    with pytest.raises(DeliveryError, match="HTTP 403") as rejected:
        post_webhook(secret, "x", client=_client(lambda r: httpx.Response(403)))
    assert "very-secret" not in str(rejected.value)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(DeliveryError, match="webhook unreachable: ConnectError") as outage:
        post_webhook(secret, "x", client=_client(down))
    assert "very-secret" not in str(outage.value)


def test_webhook_without_injected_client_closes_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class _Client:
        def __init__(self, timeout: float) -> None:
            pass

        def post(self, url: str, json: Any) -> httpx.Response:
            return httpx.Response(200, request=httpx.Request("POST", url))

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(httpx, "Client", _Client)
    post_webhook("https://hooks.example/x", "x")
    assert closed == [True]


def test_engine_used_for_reports_is_plain_sqlalchemy(store: SqlAuditStore) -> None:
    # Guards the fixture: reports read through the same store the cycle writes to.
    assert isinstance(store._engine, sa.Engine)


class _RecordingBars(FakeMarketData):
    def __init__(self, bars: tuple[Bar, ...]) -> None:
        super().__init__(bars={"SPY": bars})
        self.requested: list[int] = []

    def get_daily_bars(self, symbols: Any, *, days: int) -> Any:
        self.requested.append(days)
        return super().get_daily_bars(symbols, days=days)


def test_benchmark_window_reaches_back_to_the_report_day(store: SqlAuditStore) -> None:
    # Reporting on 2026-10-15 from a month later: the window must cover a month of bars.
    market_data = _RecordingBars(_spy().bars["SPY"])
    later = datetime(2026, 11, 16, 21, tzinfo=UTC)
    report = ReportBuilder(
        store, market_data, FakeBroker(), AppConfig(), clock=lambda: later
    ).build(DAY)
    assert market_data.requested[0] >= 22  # ~32 calendar days = ~22 trading days
    assert report.benchmark_pct == Decimal("0.50")


def test_missing_benchmark_bar_for_a_past_trading_day_is_flagged(store: SqlAuditStore) -> None:
    other_day = Bar(
        symbol="SPY",
        timestamp=datetime(2026, 10, 14, 4, tzinfo=UTC),
        open=Decimal("600"),
        high=Decimal("601"),
        low=Decimal("599"),
        close=Decimal("600"),
        volume=1,
    )
    later = datetime(2026, 10, 20, 21, tzinfo=UTC)
    report = ReportBuilder(
        store,
        FakeMarketData(bars={"SPY": (other_day,)}),
        FakeBroker(),
        AppConfig(),
        clock=lambda: later,
    ).build(DAY)
    assert report.benchmark_pct is None
    assert report.warnings == [
        "no SPY bar for 2026-10-15 (a market holiday, or older than the data window)"
    ]
