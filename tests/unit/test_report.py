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
    (trade,) = report.trades
    assert trade.pnl_usd == Decimal("1.00")
    assert trade.return_pct == Decimal("2.00")
    assert report.benchmark_pct == Decimal("0.50")
    assert report.excess_pct == Decimal("1.50")
    assert report.exits == ["force_close: force close at 15:50"]
    assert report.account_equity == Decimal("100001.00")
    assert report.performance.trades == 1
    assert report.performance.win_rate_pct == Decimal("100.00")
    assert report.llm.input_tokens == 18_000
    assert report.warnings == []


def _ledger(store: SqlAuditStore, agent: str, cost: str | None, run_id: str | None = RUN) -> None:
    from trading_agent.domain.models import LLMCallRecord

    store.record_llm_call(
        LLMCallRecord(
            created_at=_ny(9),
            provider="openai",
            model="gpt-6.1-sol" if cost is not None else "mystery",
            agent=agent,
            run_id=run_id,
            status="ok",
            usage=LLMUsage(input_tokens=9_000, output_tokens=1_000),
            latency_ms=1.0,
        ),
        None if cost is None else Decimal(cost),
    )


def test_costs_today_come_from_the_ledger(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    _ledger(store, "analyst", "0.0400")
    _ledger(store, "analyst", "0.0250")
    _ledger(store, "critic", "0.0100")
    config = AppConfig.model_validate(
        {"costs": {"services": [{"name": "Data plan", "monthly_usd": "99"}]}}
    )
    report = _builder(store, config=config).build(DAY)
    assert report.analysis is not None
    assert report.analysis.cost_usd == Decimal("0.0650")  # this run's analyst calls
    assert report.costs is not None
    assert report.costs.llm_usd == Decimal("0.0750")
    assert report.costs.services_usd == Decimal("3.2548")
    assert report.costs.total_usd == Decimal("3.3298")
    markdown = render_markdown(report)
    assert "## Costs today" in markdown
    assert "LLM $0.08 (analyst $0.07, critic $0.01; 3 calls, 30,000 tokens)" in markdown
    assert "Services $3.25. Total $3.33." in markdown


def test_unpriced_calls_are_called_out(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    _ledger(store, "analyst", "0.0400")
    _ledger(store, "reflector", None)
    report = _builder(store).build(DAY)
    assert report.costs is not None
    assert report.costs.llm_usd is None
    assert report.unpriced_models == ["mystery"]
    markdown = render_markdown(report)
    assert "LLM $0.04 + unpriced" in markdown
    assert "No price set for mystery" in markdown


def test_unreadable_ledger_is_a_warning_in_the_report(store: SqlAuditStore) -> None:
    _record_trading_day(store)

    def down(start: Any, end: Any) -> Any:
        raise AuditError("ledger down")

    store.llm_calls_between = down  # type: ignore[method-assign]
    report = _builder(store).build(DAY)
    assert report.costs is None
    assert "cost ledger unavailable: ledger down" in report.warnings


def test_day_without_a_cycle(store: SqlAuditStore) -> None:
    report = _builder(store).build(DAY)
    assert report.cycle_status is None
    assert report.analysis is None
    assert report.risk is None
    assert report.trades == []
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
    assert report.trades


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


def _vetoed_critique() -> Any:
    from trading_agent.agents.critic import CriticResult, EvidenceCheck

    return CriticResult(
        run_id=RUN,
        status="vetoed",
        model_verdict="approve",
        summary="Evidence c2 overstates the spread.",
        reasons=("evidence 'c2' is unsupported: spread is 0.2%, not 0.02%",),
        concerns=("Market is falling",),
        evidence_checks=(
            EvidenceCheck(source_call_id="c1", supported=True, note="ok"),
            EvidenceCheck(source_call_id="c2", supported=False, note="spread is 0.2%, not 0.02%"),
        ),
        model="judge",
        prompt_version="critic-p",
        usage=LLMUsage(input_tokens=4000, output_tokens=400),
        created_at=_ny(8, 46),
    )


def test_report_shows_the_critics_veto(store: SqlAuditStore) -> None:
    store.start_run(RUN, "daily")
    store.record_critique(_vetoed_critique())
    report = _builder(store).build(DAY)
    assert report.critic is not None
    assert report.critic.status == "vetoed"
    assert report.critic.unsupported == ["c2: spread is 0.2%, not 0.02%"]
    assert report.llm.critiques == 1
    markdown = render_markdown(report)
    for expected in [
        "## Critic",
        "**VETOED** by judge, 4,400 tokens. Evidence c2 overstates the spread.",
        "Unsupported evidence:\n- c2: spread is 0.2%, not 0.02%",
        "Why it did not pass:\n- evidence 'c2' is unsupported",
        "Concerns:\n- Market is falling",
        "1 critiques",
    ]:
        assert expected in markdown, expected


def test_report_without_a_critique_has_no_critic_section(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    report = _builder(store).build(DAY)
    assert report.critic is None
    assert "## Critic" not in render_markdown(report)


# ---------- Learning ----------


def test_report_shows_what_was_learned_and_retired(store: SqlAuditStore) -> None:
    from datetime import timedelta

    from trading_agent.storage.store import NewLessonRow

    _record_trading_day(store)
    later = GENERATED + timedelta(days=30)
    [old] = store.record_lessons(
        None,
        [NewLessonRow(text="Gap-ups fade by noon.", category="entry", evidence={})],
        expires_at=later,
        max_active=15,
    )
    store.record_lessons(
        RUN,
        [
            NewLessonRow(
                text="Earnings beats held into the close.", category="research", evidence={}
            )
        ],
        expires_at=later,
        max_active=15,
    )
    store.retire_lesson(old, f"contradicted by the review of {RUN}")

    report = _builder(store).build(DAY)
    assert report.learning is not None
    assert report.learning.lessons_used == 0
    assert report.learning.active == 1
    assert report.learning.added == ["[research] Earnings beats held into the close."]
    assert report.learning.retired == [
        f"[entry] Gap-ups fade by noon. (contradicted by the review of {RUN})"
    ]
    markdown = render_markdown(report)
    assert "## Learning" in markdown
    assert "The analyst used 0 lesson(s) this morning; 1 active now." in markdown
    assert "Learned today:\n- [research] Earnings beats held into the close." in markdown
    assert "Retired today:\n- [entry] Gap-ups fade by noon." in markdown


def test_report_counts_the_lessons_the_analyst_used(store: SqlAuditStore) -> None:
    store.start_run(RUN, "daily")
    store.record_analysis(
        AnalystResult(
            run_id=RUN,
            status="no_trade",
            reason="nothing qualified",
            proposal=None,
            prompt_version="p",
            turns=1,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=_ny(8, 45),
            duration_seconds=1,
            lesson_ids=(1, 2, 3),
        )
    )
    learning = _builder(store).build(DAY).learning
    assert learning is not None
    assert (learning.lessons_used, learning.added, learning.retired) == (3, [], [])


def test_no_learning_section_when_disabled_or_unreadable(store: SqlAuditStore) -> None:
    config = AppConfig.model_validate({"learning": {"enabled": False}})
    report = _builder(store, config=config).build(DAY)
    assert report.learning is None
    assert "## Learning" not in render_markdown(report)

    class _Down(NullAuditStore):
        def list_lessons(self, *, include_retired: bool) -> list[dict[str, Any]]:
            raise AuditError("db down")

    assert _builder(_Down()).build(DAY).learning is None


# ---------- Several trade slots (ADR 0015) ----------


def _score_json(total: str = "72.5") -> dict[str, Any]:
    return {
        "total": total,
        "min_score": "60",
        "passed": True,
        "components": [
            {"name": "Trend", "value": "1.00", "weight": "25", "detail": "price above average"},
            {"name": "History", "value": "0.50", "weight": "10", "detail": "3 of 10: neutral"},
        ],
        "hard_fails": [],
    }


def _record_slot(store: SqlAuditStore, number: int, symbol: str, exit_price: str) -> str:
    from trading_agent.domain.models import ScoreComponent, TradeScore

    run_id = f"{RUN}-{number}"
    store.start_run(run_id, "daily")
    proposal = PROPOSAL.model_copy(update={"symbol": symbol})
    store.record_analysis(
        AnalystResult(
            run_id=run_id,
            status="proposed",
            reason="evidence verified",
            proposal=proposal,
            prompt_version="p",
            turns=2,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=_ny(10, 30),
            duration_seconds=1,
        )
    )
    plan = OrderPlan(
        symbol=symbol,
        quantity=Decimal("0.5"),
        fractional=True,
        notional_usd=Decimal("50"),
        entry_reference_price=Decimal("100"),
        limit_price=Decimal("102"),
        stop_loss_price=Decimal("95"),
        take_profit_price=Decimal("110"),
    )
    score = TradeScore(
        total=Decimal("72.5"),
        min_score=Decimal(60),
        passed=True,
        components=(
            ScoreComponent(name="Trend", value=Decimal(1), weight=Decimal(25), detail="up"),
        ),
        hard_fails=(),
        reward_risk=Decimal("1.88"),
        atr_pct=Decimal("2.00"),
        stop_atr=Decimal("2.50"),
        extension_atr=Decimal("0.53"),
        expected_value_pct=None,
        history_trades=1,
    )
    store.record_risk_decision(
        run_id,
        RiskDecision(
            approved=True,
            symbol=symbol,
            checks=(RiskCheck(name="All", passed=True, detail=""),),
            plan=plan,
            evaluated_at=_ny(10, 31),
            score=score,
        ),
    )
    entry = _order(f"ta-{run_id}-entry", "buy", "100", _ny(10, 31))
    exit_order = _order(f"close-{number}", "sell", exit_price, _ny(15, 50))
    store.record_orders(
        run_id,
        [
            entry.model_copy(update={"symbol": symbol}),
            exit_order.model_copy(update={"symbol": symbol}),
        ],
    )
    store.finish_run(run_id, "traded", f"entered 0.5 {symbol} at $100")
    return run_id


def test_a_day_with_several_slots_reports_every_trade(store: SqlAuditStore) -> None:
    _record_trading_day(store)  # slot 1: AAPL +2%
    _record_slot(store, 2, "MSFT", "99")  # slot 2: MSFT -1%
    report = _builder(store).build(DAY)

    assert [t.symbol for t in report.trades] == ["AAPL", "MSFT"]
    assert report.excess_pct == Decimal("0.00")  # mean of +2% and -1%, minus SPY's +0.5%
    (slot,) = report.slots
    assert (slot.run_id, slot.status, slot.symbol) == (f"{RUN}-2", "traded", "MSFT")
    assert slot.score == "72.5/100 (needs 60)"
    markdown = render_markdown(report)
    assert "## Trade slots" in markdown
    assert f"| 2 | `{RUN}-2` | TRADED | MSFT | 72.5/100 (needs 60) |" in markdown
    assert "| AAPL |" in markdown
    assert "| MSFT |" in markdown


def test_the_risk_section_shows_the_trade_score(store: SqlAuditStore) -> None:
    _record_slot(store, 2, "MSFT", "101")
    store.start_run(RUN, "daily")
    report = _builder(store).build(DAY)
    (slot,) = report.slots
    assert slot.score_detail == "trend 1.00"


def test_unreadable_slots_are_a_warning(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    real = store.get_run

    def flaky(run_id: str) -> Any:
        if run_id.endswith("-2"):
            raise AuditError("db down")
        return real(run_id)

    store.get_run = flaky  # type: ignore[method-assign]
    report = _builder(store).build(DAY)
    assert report.slots == []
    assert "trade slots unavailable: db down" in report.warnings
