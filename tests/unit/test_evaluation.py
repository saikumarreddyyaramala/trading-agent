from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FakeMarketData
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.agents.critic import CriticResult
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
from trading_agent.evaluation.replay import replay_critic
from trading_agent.evaluation.scorecard import ScorecardBuilder, render_scorecard
from trading_agent.evaluation.simulate import simulate_intraday
from trading_agent.evaluation.stats import (
    bootstrap_mean_interval,
    max_drawdown,
    mean,
    profit_factor,
)
from trading_agent.settings.models import AppConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import AuditError, NullAuditStore, SqlAuditStore

NY = ZoneInfo("America/New_York")
D = [date(2026, 10, d) for d in (12, 13, 14, 15, 16)]  # Monday to Friday
NOW = datetime(2026, 10, 16, 21, tzinfo=UTC)


def D_(value: str) -> Decimal:
    return Decimal(value)


# ---------- Statistics ----------


def test_mean_and_profit_factor() -> None:
    assert mean([]) is None
    assert mean([D_("1"), D_("2")]) == D_("1.50")
    assert profit_factor([D_("3"), D_("-1"), D_("-2")]) == D_("1.00")
    assert profit_factor([D_("3")]) is None


def test_max_drawdown() -> None:
    assert max_drawdown([]) == D_("0.00")
    assert max_drawdown([D_("5"), D_("-3"), D_("-4"), D_("10"), D_("-2")]) == D_("7.00")


def test_bootstrap_interval_is_reproducible_and_sensible() -> None:
    values = [D_(v) for v in ("1", "2", "3", "-1", "0.5", "1.5")]
    first = bootstrap_mean_interval(values, samples=2000)
    assert first == bootstrap_mean_interval(values, samples=2000)  # seeded
    assert first is not None
    low, high = first
    assert low < D_("1.17") < high  # contains the sample mean
    assert bootstrap_mean_interval([D_("1")]) is None
    assert bootstrap_mean_interval([D_("2"), D_("2")]) == (D_("2.00"), D_("2.00"))


# ---------- Counterfactual simulation ----------


def _bar(day: date, o: str, h: str, low: str, c: str, symbol: str = "AAPL") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime.combine(day, time(0), tzinfo=NY),
        open=D_(o),
        high=D_(h),
        low=D_(low),
        close=D_(c),
        volume=1,
    )


@pytest.mark.parametrize(
    ("bar", "outcome", "ret"),
    [
        (_bar(D[0], "100", "112", "94", "105"), "stop", "-5.00"),  # both hit: stop first
        (_bar(D[0], "100", "111", "97", "105"), "target", "10.00"),
        (_bar(D[0], "100", "103", "97", "102"), "close", "2.00"),
    ],
)
def test_simulation_is_conservative(bar: Bar, outcome: str, ret: str) -> None:
    result = simulate_intraday(bar, D_("100"), stop_loss_pct=D_("5"), take_profit_pct=D_("10"))
    assert (result.outcome, result.return_pct) == (outcome, D_(ret))


def test_simulation_needs_a_positive_entry() -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_intraday(
            _bar(D[0], "1", "1", "1", "1"), D_("0"), stop_loss_pct=D_("5"), take_profit_pct=D_("10")
        )


# ---------- A recorded history ----------

PROPOSAL = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="t",
    reference_price=D_("100"),
    confidence="medium",
    evidence=(Evidence(claim="a", source_call_id="c1"), Evidence(claim="b", source_call_id="c1")),
    risks=("r",),
)
PLAN = OrderPlan(
    symbol="AAPL",
    quantity=D_("0.5"),
    fractional=True,
    notional_usd=D_("50"),
    entry_reference_price=D_("100"),
    limit_price=D_("102"),
    stop_loss_price=D_("95"),
    take_profit_price=D_("110"),
)


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=NY)


def _record_day(
    store: SqlAuditStore,
    day: date,
    kind: str,
    *,
    exit_price: str = "102",
    status: str | None = None,
    with_outputs: bool = True,
) -> str:
    run_id = f"daily-{day:%Y%m%d}"
    store.start_run(run_id, "daily")
    if kind == "error":
        store.finish_run(run_id, "error", "positions may still be open")
        return run_id
    proposal = (
        PROPOSAL
        if kind != "no_trade"
        else TradeProposal(decision="no_trade", thesis="q", confidence="low")
    )
    store.record_analysis(
        AnalystResult(
            run_id=run_id,
            status="proposed" if kind != "no_trade" else "no_trade",
            reason="r",
            proposal=proposal,
            prompt_version="p",
            turns=2,
            tool_calls=(
                ToolCallRecord(
                    call_id="c1",
                    tool="get_quote",
                    arguments={},
                    is_error=False,
                    result='{"ask": "100"}' if with_outputs else None,
                ),
            ),
            usage=LLMUsage(input_tokens=1_000_000, output_tokens=100_000),
            started_at=_at(day, 8, 45),
            duration_seconds=1,
        )
    )
    if kind in ("traded", "rejected", "vetoed"):
        critic_status = "vetoed" if kind == "vetoed" else "approved"
        store.record_critique(
            CriticResult(
                run_id=run_id,
                status=critic_status,  # type: ignore[arg-type]
                model_verdict="approve" if critic_status == "approved" else "veto",
                summary="s",
                reasons=(),
                concerns=(),
                evidence_checks=(),
                model="judge",
                prompt_version="cp",
                usage=LLMUsage(input_tokens=0, output_tokens=0),
                created_at=_at(day, 8, 50),
            )
        )
    if kind in ("traded", "rejected"):
        approved = kind == "traded"
        store.record_risk_decision(
            run_id,
            RiskDecision(
                approved=approved,
                symbol="AAPL",
                checks=(RiskCheck(name="Spread", passed=approved, detail=""),),
                plan=PLAN if approved else None,
                evaluated_at=_at(day, 9, 45),
            ),
        )
    if kind == "traded":
        store.record_orders(
            run_id,
            [
                BrokerOrder(
                    id=f"e-{run_id}",
                    client_order_id=f"ta-{run_id}-entry",
                    symbol="AAPL",
                    side="buy",
                    order_type="limit",
                    status="filled",
                    quantity=D_("0.5"),
                    filled_quantity=D_("0.5"),
                    filled_avg_price=D_("100"),
                    submitted_at=_at(day, 9, 45),
                ),
                BrokerOrder(
                    id=f"x-{run_id}",
                    client_order_id=f"close-{run_id}",
                    symbol="AAPL",
                    side="sell",
                    order_type="market",
                    status="filled",
                    quantity=D_("0.5"),
                    filled_quantity=D_("0.5"),
                    filled_avg_price=D_(exit_price),
                    submitted_at=_at(day, 15, 50),
                ),
            ],
        )
        store.record_step(run_id, "entry", "entered", "", _at(day, 9, 45))
    store.finish_run(run_id, status or {"traded": "traded"}.get(kind, "no_trade"), "")
    return run_id


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: NOW)


def _bars() -> FakeMarketData:
    spy = tuple(_bar(d, "600", "606", "597", "603", "SPY") for d in D)  # +0.50% every day
    aapl = (
        _bar(D[2], "100", "103", "94", "101"),  # vetoed day: would have been stopped (-5%)
        _bar(D[3], "100", "111", "99", "104"),  # rejected day: would have hit the target
    )
    return FakeMarketData(bars={"SPY": spy, "AAPL": aapl})


def _history(store: SqlAuditStore) -> None:
    _record_day(store, D[0], "traded", exit_price="103")  # +3.00%
    _record_day(store, D[1], "traded", exit_price="101")  # +1.00%
    _record_day(store, D[2], "vetoed")
    _record_day(store, D[3], "rejected")
    _record_day(store, D[4], "no_trade")


def _config(**evaluation: Any) -> AppConfig:
    return AppConfig.model_validate(
        {
            "evaluation": {"min_cycle_days": 5, "min_closed_trades": 2, **evaluation},
            "reporting": {
                "llm_usd_per_million_input_tokens": "0.10",
                "llm_usd_per_million_output_tokens": "1.00",
            },
        }
    )


def _builder(
    store: Any, config: AppConfig | None = None, market_data: Any = None
) -> ScorecardBuilder:
    return ScorecardBuilder(store, market_data or _bars(), config or _config(), clock=lambda: NOW)


# ---------- The scorecard ----------


def test_funnel_counts_every_stage(store: SqlAuditStore) -> None:
    _history(store)
    funnel = _builder(store).build(D[0], D[4]).funnel
    assert (funnel.calendar_days, funnel.cycle_days) == (5, 5)
    assert funnel.statuses == {"traded": 2, "no_trade": 3}
    assert (funnel.analyses, funnel.buy_proposals, funnel.no_trade_decisions) == (5, 4, 1)
    assert funnel.critic == {"approved": 3, "vetoed": 1}
    assert (funnel.risk_approved, funnel.risk_rejected) == (2, 1)
    assert funnel.risk_failed_checks == {"Spread": 1}
    assert funnel.entries == {"entered": 2}
    assert funnel.closed_trades == 2


def test_performance_against_spy(store: SqlAuditStore) -> None:
    _history(store)
    perf = _builder(store).build(D[0], D[4]).performance
    assert (perf.closed_trades, perf.wins, perf.win_rate_pct) == (2, 2, D_("100.00"))
    assert perf.total_pnl_usd == D_("2.00")
    assert perf.mean_return_pct == D_("2.00")
    assert perf.mean_benchmark_pct == D_("0.50")
    assert perf.mean_excess_pct == D_("1.50")
    assert perf.excess_interval_pct == (D_("0.50"), D_("2.50"))
    assert perf.max_drawdown_usd == D_("0.00")
    assert (perf.best_trade_pct, perf.worst_trade_pct) == (D_("3.00"), D_("1.00"))


def test_counterfactuals_show_what_the_gates_stopped(store: SqlAuditStore) -> None:
    _history(store)
    critic, risk = _builder(store).build(D[0], D[4]).counterfactuals
    assert (critic.gate, critic.stopped, critic.simulated, critic.mean_return_pct) == (
        "critic",
        1,
        1,
        D_("-5.00"),
    )
    assert (critic.would_have_won, critic.would_have_lost) == (0, 1)  # the veto saved money
    assert (risk.gate, risk.mean_return_pct, risk.would_have_won) == ("risk engine", D_("10.00"), 1)


def test_cost_from_recorded_tokens(store: SqlAuditStore) -> None:
    _history(store)
    cost = _builder(store).build(D[0], D[4]).cost
    assert (cost.analyses, cost.critiques) == (5, 4)
    assert cost.input_tokens == 5_000_000
    assert cost.total_usd == D_("1.00")  # 5M x $0.10/M + 0.5M x $1/M
    assert cost.per_cycle_day_usd == D_("0.20")


def test_a_record_meeting_every_bar_is_a_go(store: SqlAuditStore) -> None:
    _history(store)
    card = _builder(store).build(D[0], D[4])
    assert all(c.passed for c in card.criteria), card.criteria
    assert card.decision == "GO"


def test_any_failed_criterion_is_a_no_go(store: SqlAuditStore) -> None:
    _history(store)
    card = _builder(
        store, _config(min_win_rate_pct=100, max_drawdown_usd=1, min_closed_trades=3)
    ).build(D[0], D[4])
    failed = {c.name for c in card.criteria if not c.passed}
    assert failed == {"Closed trades"}
    assert card.decision == "NO-GO"


def test_drawdown_and_losses_fail(store: SqlAuditStore) -> None:
    _record_day(store, D[0], "traded", exit_price="90")  # -$5
    _record_day(store, D[1], "traded", exit_price="70")  # -$15
    card = _builder(store, _config(min_cycle_days=2)).build(D[0], D[1])
    failed = {c.name for c in card.criteria if c.passed is False}
    assert {"Win rate", "Mean excess return vs SPY", "Total P&L", "Max drawdown"} <= failed
    assert card.performance.max_drawdown_usd == D_("20.00")  # over the $15 limit
    assert card.performance.profit_factor == D_("0.00")  # no gains against real losses


def test_unknown_values_are_failures_not_passes(store: SqlAuditStore) -> None:
    card = _builder(NullAuditStore(), AppConfig()).build(D[0], D[4])
    assert card.decision == "NO-GO"
    unknown = {c.name for c in card.criteria if c.passed is None}
    assert "Win rate" in unknown
    assert "LLM cost per cycle day" in unknown
    cost = next(c for c in card.criteria if c.name == "LLM cost per cycle day")
    assert cost.actual.startswith("unknown: set reporting.llm_usd_per_million_*")


def test_cost_criterion_can_be_removed(store: SqlAuditStore) -> None:
    card = _builder(store, _config(max_llm_cost_per_day_usd=None)).build(D[0], D[4])
    assert "LLM cost per cycle day" not in {c.name for c in card.criteria}


def test_error_days_count_against_reliability(store: SqlAuditStore) -> None:
    _history(store)
    _record_day(store, date(2026, 10, 19), "error")
    card = _builder(store).build(D[0], date(2026, 10, 19))
    errors = next(c for c in card.criteria if c.name == "Cycle days with errors")
    assert errors.actual == "16.67%"
    assert errors.passed is False


def test_first_cycle_day_and_window_checks(store: SqlAuditStore) -> None:
    builder = _builder(store)
    assert builder.first_cycle_day() is None
    _history(store)
    store.start_run("manual-x", "manual_buy")
    assert builder.first_cycle_day() == D[0]
    with pytest.raises(ValueError, match="before the start"):
        builder.build(D[4], D[0])


def test_unavailable_data_becomes_warnings() -> None:
    class _Down(NullAuditStore):
        def get_run(self, run_id: str) -> None:
            raise AuditError("db down")

        def orders_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
            raise AuditError("db down")

        def list_runs(self, limit: int) -> Any:
            raise AuditError("db down")

    class _NoBars(FakeMarketData):
        def get_daily_bars(self, symbols: Any, *, days: int) -> Any:
            raise RuntimeError("provider down")

    builder = _builder(_Down(), market_data=_NoBars())
    card = builder.build(D[0], D[1])
    assert card.warnings == [
        "audit store unavailable: db down",
        "orders unavailable: db down",
        "market data unavailable: provider down",
    ]
    assert builder.first_cycle_day() is None


def test_scorecard_markdown(store: SqlAuditStore) -> None:
    _history(store)
    markdown = render_scorecard(_builder(store).build(D[0], D[4]))
    for expected in [
        "# Evaluation: 2026-10-12 to 2026-10-16",
        "## Decision: **GO**",
        "| Win rate | >= 45% | 100.00% | pass |",
        "- Critic: approved 3, vetoed 1.",
        "(most failed: Spread 1)",
        "| 2 | 100.00% | $2.00 | 2.00% | 0.50% | 1.50% | 0.50% to 2.50% |",
        "| critic | 1 | 1 | -5.00% | 0 | 1 |",
        "($1.00, $0.20 per cycle day).",
        "Past paper results do not guarantee live results.",
    ]:
        assert expected in markdown, expected


def test_empty_scorecard_markdown() -> None:
    markdown = render_scorecard(_builder(NullAuditStore(), AppConfig()).build(D[0], D[0]))
    assert "## Decision: **NO-GO**" in markdown
    assert "FAIL (no data)" in markdown
    assert "Set LLM prices in config to see dollars." in markdown
    assert "Critic: no reviews." in markdown


# ---------- Critic replay ----------


class _Critic:
    def __init__(self, status: str) -> None:
        self.status = status
        self.reviewed: list[str] = []

    def review(
        self, run_id: str, proposal: TradeProposal, calls: list[ToolCallRecord]
    ) -> CriticResult:
        self.reviewed.append(run_id)
        return CriticResult(
            run_id=run_id,
            status=self.status,  # type: ignore[arg-type]
            model_verdict="approve" if self.status == "approved" else "veto",
            summary=f"now {self.status}",
            reasons=(),
            concerns=(),
            evidence_checks=(),
            model="judge",
            prompt_version="cp2",
            usage=LLMUsage(),
            created_at=NOW,
        )


def test_replay_reports_agreement_and_flips(store: SqlAuditStore) -> None:
    _history(store)  # 3 approved by the critic, 1 vetoed, 1 no-trade
    run_ids = [f"daily-{d:%Y%m%d}" for d in D]
    result = replay_critic(store, _Critic("vetoed"), run_ids)  # type: ignore[arg-type]
    assert (result.reviewed, result.agreed, result.now_vetoed, result.now_approved) == (4, 1, 3, 0)
    assert {i.run_id for i in result.items if i.changed} == {
        "daily-20261012",
        "daily-20261013",
        "daily-20261015",
    }

    result = replay_critic(store, _Critic("approved"), run_ids)  # type: ignore[arg-type]
    assert (result.agreed, result.now_approved) == (3, 1)


def test_replay_skips_runs_without_recorded_outputs(store: SqlAuditStore) -> None:
    run_id = _record_day(store, D[0], "traded", with_outputs=False)
    critic = _Critic("approved")
    result = replay_critic(store, critic, [run_id, "daily-19990101"])  # type: ignore[arg-type]
    assert result.reviewed == 0
    assert result.skipped == {run_id: "no tool outputs recorded (run predates migration 0003)"}
    assert critic.reviewed == []


def test_replay_of_an_unreviewed_proposal(store: SqlAuditStore) -> None:
    store.start_run("daily-20261012", "daily")
    store.record_analysis(
        AnalystResult(
            run_id="daily-20261012",
            status="proposed",
            reason="r",
            proposal=PROPOSAL,
            prompt_version="p",
            turns=1,
            tool_calls=(
                ToolCallRecord(
                    call_id="c1", tool="get_quote", arguments={}, is_error=False, result="x"
                ),
            ),
            usage=LLMUsage(),
            started_at=_at(D[0], 8, 45),
            duration_seconds=1,
        )
    )
    result = replay_critic(store, _Critic("vetoed"), ["daily-20261012"])  # type: ignore[arg-type]
    (item,) = result.items
    assert (item.recorded, item.replayed, item.changed) == ("none", "vetoed", True)
