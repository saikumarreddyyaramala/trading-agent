"""The daily cycle, run through whole simulated trading days on a fake clock."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa

from tests.conftest import FakeTime
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.agents.critic import CriticResult
from trading_agent.domain.models import (
    BrokerOrder,
    Evidence,
    LLMUsage,
    MarketClock,
    OrderPlan,
    RiskCheck,
    RiskDecision,
    TradeProposal,
)
from trading_agent.execution.executor import EntryResult, ManageResult
from trading_agent.health import CheckResult, CheckStatus
from trading_agent.orchestration.cycle import CycleServices, DailyCycle
from trading_agent.settings.models import AppConfig
from trading_agent.storage import schema
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import (
    AuditError,
    AuditRecorder,
    AuditStore,
    NullAuditStore,
    SqlAuditStore,
)

NY = ZoneInfo("America/New_York")
THURSDAY = date(2026, 10, 15)
RUN_ID = "daily-20261015"


def at(hour: int, minute: int = 0, day: date = THURSDAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=NY)


class FakeCalendar:
    """Market clock for the fake time: weekdays 09:30-16:00, with holidays and early closes."""

    def __init__(
        self,
        clock: FakeTime,
        *,
        holidays: tuple[date, ...] = (),
        early: dict[date, time] | None = None,
        fail: bool = False,
    ) -> None:
        self._clock = clock
        self._holidays = set(holidays)
        self._early = early or {}
        self._fail = fail

    def _trading(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self._holidays

    def _close(self, day: date) -> datetime:
        return at(self._early.get(day, time(16)).hour, self._early.get(day, time(16)).minute, day)

    def __call__(self) -> MarketClock:
        if self._fail:
            raise RuntimeError("clock unavailable")
        now = self._clock.now.astimezone(NY)
        day = now.date()
        is_open = self._trading(day) and at(9, 30, day) <= now < self._close(day)
        candidate = day
        while not self._trading(candidate) or now >= self._close(candidate):
            candidate += timedelta(days=1)
        return MarketClock(
            timestamp=now,
            is_open=is_open,
            next_open=at(9, 30, candidate)
            if not is_open
            else at(9, 30, candidate + timedelta(days=1)),
            next_close=self._close(candidate),
        )


PROPOSAL = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="Tight spread.",
    reference_price=Decimal("100.05"),
    confidence="medium",
    evidence=(
        Evidence(claim="a", source_call_id="c1"),
        Evidence(claim="b", source_call_id="c2"),
    ),
    risks=("reversal",),
)
PLAN = OrderPlan(
    symbol="AAPL",
    quantity=Decimal("0.4899"),
    fractional=True,
    notional_usd=Decimal("49.03"),
    entry_reference_price=Decimal("100.10"),
    limit_price=Decimal("102.05"),
    stop_loss_price=Decimal("95.09"),
    take_profit_price=Decimal("110.11"),
)


def _order(role: str, side: str = "buy") -> BrokerOrder:
    return BrokerOrder(
        id=f"id-{role}",
        client_order_id=f"ta-{RUN_ID}-{role}",
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        order_type="limit" if role == "entry" else "stop",
        status="filled" if role == "entry" else "new",
        quantity=Decimal("0.4899"),
        filled_quantity=Decimal("0.4899") if role == "entry" else Decimal("0"),
        filled_avg_price=Decimal("102") if role == "entry" else None,
    )


class FakeExecutor:
    def __init__(
        self, clock: FakeTime, *, stuck: bool = False, flatten_fails: bool = False
    ) -> None:
        self._clock = clock
        self.holding = False
        self.stuck = stuck
        self.flatten_fails = flatten_fails
        self.calls: list[tuple[str, Any]] = []

    def enter(self, decision: RiskDecision, run_id: str) -> EntryResult:
        self.calls.append(("enter", run_id))
        self.holding = True
        return EntryResult(
            status="entered",
            reason="filled and protected by a stop-loss",
            symbol="AAPL",
            entry_order=_order("entry"),
            stop_order=_order("stop", "sell"),
            filled_quantity=Decimal("0.4899"),
            avg_fill_price=Decimal("102"),
        )

    def manage(self, force_close_at: time | None = None) -> tuple[ManageResult, ...]:
        now = self._clock.now.astimezone(NY)
        self.calls.append(("manage", (now.time(), force_close_at)))
        if not self.holding:
            return (ManageResult(action="flat", detail="no open positions"),)
        if force_close_at is not None and now.time() >= force_close_at and not self.stuck:
            self.holding = False
            return (ManageResult(action="force_close", detail="force close", symbol="AAPL"),)
        return (ManageResult(action="hold", detail="bid $101", symbol="AAPL"),)

    def flatten(self) -> ManageResult:
        self.calls.append(("flatten", None))
        if self.flatten_fails:
            return ManageResult(action="error", detail="close failed")
        self.holding = False
        return ManageResult(action="flattened", detail="closed 1 position(s)")

    def count(self, name: str) -> int:
        return sum(1 for call, _ in self.calls if call == name)


def _analysis(status: str = "proposed", run_id: str = RUN_ID) -> AnalystResult:
    return AnalystResult(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        reason={"proposed": "evidence verified", "no_trade": "chose not to trade"}.get(
            status, "boom"
        ),
        proposal=PROPOSAL if status == "proposed" else None,
        prompt_version="p",
        turns=3,
        tool_calls=(),
        usage=LLMUsage(input_tokens=10, output_tokens=2),
        started_at=at(8, 45),
        duration_seconds=1.0,
    )


def _decision(approved: bool, when: datetime) -> RiskDecision:
    checks = (RiskCheck(name="Spread", passed=approved, detail="ok" if approved else "too wide"),)
    return RiskDecision(
        approved=approved,
        symbol="AAPL",
        checks=checks,
        plan=PLAN if approved else None,
        evaluated_at=when,
    )


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        start: datetime | None = None,
        health_ok: bool = True,
        analysis: str | Exception = "proposed",
        approve: bool = True,
        executor: dict[str, Any] | None = None,
        calendar: dict[str, Any] | None = None,
        store: AuditStore | None = None,
        required: bool = False,
        config: AppConfig | None = None,
        capacity: Any = None,
    ) -> None:
        self.time = FakeTime(start or at(8))
        self.contexts: list[tuple[str, str | None]] = []
        self.engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
        upgrade(self.engine)
        self.store = store or SqlAuditStore(self.engine, AppConfig(), clock=self.time.clock)
        self.executor = FakeExecutor(self.time, **(executor or {}))
        self.calls: list[tuple[str, datetime]] = []
        self._health_ok = health_ok
        self._analysis = analysis
        self._approve = approve
        self.services = CycleServices(
            health_check=self._health,
            analyze=self._analyze,
            evaluate=self._evaluate,
            executor=self.executor,
            market_clock=FakeCalendar(self.time, **(calendar or {})),
            orders_since=self._orders_since,
            recorder=AuditRecorder(self.store, required=required),
            capacity=capacity,
        )
        self.config = config or AppConfig()
        self.cycle = DailyCycle(
            self.services, self.config, clock=self.time.clock, sleep=self.time.sleep
        )

    def _now(self) -> datetime:
        return self.time.now.astimezone(NY)

    def _health(self) -> list[CheckResult]:
        self.calls.append(("health", self._now()))
        status = CheckStatus.PASS if self._health_ok else CheckStatus.FAIL
        return [CheckResult("Broker account", status, "ok" if self._health_ok else "blocked")]

    def _analyze(self, run_id: str, context: str | None = None) -> AnalystResult:
        self.calls.append(("analyze", self._now()))
        self.contexts.append((run_id, context))
        if isinstance(self._analysis, Exception):
            raise self._analysis
        return _analysis(self._analysis, run_id)

    def _evaluate(self, proposal: TradeProposal) -> RiskDecision:
        self.calls.append(("evaluate", self._now()))
        return _decision(self._approve, self.time.now)

    def _orders_since(self, since: datetime) -> tuple[BrokerOrder, ...]:
        self.calls.append(("reconcile", self._now()))
        return (_order("entry"),)

    def when(self, name: str) -> datetime:
        return next(t for call, t in self.calls if call == name)

    def called(self, name: str) -> bool:
        return any(call == name for call, _ in self.calls)

    def rows(self, table: sa.Table) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [dict(r) for r in connection.execute(sa.select(table)).mappings()]


# ---------- A full trading day ----------


def test_trading_day_runs_every_step_on_schedule(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = h.cycle.run_day(THURSDAY)

    assert result.status == "traded"
    assert result.summary == "entered 0.4899 AAPL at $102"
    assert result.steps == {
        "preflight": "ok",
        "analysis": "proposed",
        "critique": "disabled",
        "entry": "entered",
        "close": "ok",
        "reconcile": "ok",
    }
    assert h.when("health") == at(8, 30)
    assert h.when("analyze") == at(8, 45)
    assert h.when("evaluate") == at(9, 45)
    assert h.when("reconcile") == at(16, 15)
    assert h.executor.calls[0] == ("enter", RUN_ID)
    # Managed about once a minute from the entry to the 15:50 force-close.
    assert 360 <= h.executor.count("manage") <= 380
    assert h.executor.holding is False


def test_trading_day_is_fully_recorded(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.cycle.run_day(THURSDAY)

    (run,) = h.rows(schema.runs)
    assert (run["run_id"], run["kind"], run["status"]) == (RUN_ID, "daily", "traded")
    assert [s["step"] for s in h.rows(schema.run_steps)] == [
        "preflight",
        "analysis",
        "critique",
        "entry",
        "close",
        "reconcile",
    ]
    assert h.rows(schema.analyses)[0]["run_id"] == RUN_ID
    assert h.rows(schema.risk_decisions)[0]["approved"] is True
    assert {o["role"] for o in h.rows(schema.orders)} == {"entry", "stop"}
    assert [e["action"] for e in h.rows(schema.position_events)] == ["force_close"]


def test_quiet_day_stops_managing_once_flat(tmp_path: Path) -> None:
    h = Harness(tmp_path, analysis="no_trade")
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "no_trade"
    assert result.steps["entry"] == "skipped"
    assert not h.called("evaluate")
    assert h.executor.count("manage") <= 3  # one flat pass, then only the close checks
    assert h.called("reconcile")


def test_risk_rejection_means_no_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path, approve=False)
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "no_trade"
    assert result.steps["entry"] == "rejected"
    assert h.executor.count("enter") == 0
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "entry"]
    assert step["detail"] == "risk checks failed: Spread"


# ---------- Failures stop new trades, never the exits ----------


def test_failed_preflight_halts_trading_but_still_closes_and_reconciles(tmp_path: Path) -> None:
    h = Harness(tmp_path, health_ok=False)
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "halted"
    assert result.steps["preflight"] == "failed"
    assert result.steps["analysis"] == "skipped"
    assert not h.called("analyze")  # no tokens spent on a halted day
    assert result.steps["close"] == "ok"
    assert h.called("reconcile")


def test_analysis_crash_is_contained(tmp_path: Path) -> None:
    h = Harness(tmp_path, analysis=RuntimeError("model down"))
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["analysis"] == "error"
    assert result.steps["entry"] == "skipped"
    assert result.status == "no_trade"
    assert result.steps["reconcile"] == "ok"


def test_failed_analysis_status_means_no_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path, analysis="failed")
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["analysis"] == "failed"
    assert result.steps["entry"] == "skipped"


def test_stuck_position_is_flattened(tmp_path: Path) -> None:
    h = Harness(tmp_path, executor={"stuck": True})
    result = h.cycle.run_day(THURSDAY)
    assert h.executor.count("flatten") == 1
    assert result.steps["close"] == "ok"
    assert result.status == "traded"


def test_unflattenable_position_is_an_error(tmp_path: Path) -> None:
    h = Harness(tmp_path, executor={"stuck": True, "flatten_fails": True})
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "error"
    assert result.summary == "positions may still be open after the force-close"


def test_required_audit_store_failure_halts_trading(tmp_path: Path) -> None:
    class _Broken(NullAuditStore):
        def start_run(self, run_id: str, kind: str) -> None:
            raise AuditError("database down")

    h = Harness(tmp_path, store=_Broken(), required=True)
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "halted"
    assert not h.called("analyze")
    assert h.executor.count("enter") == 0
    assert h.called("reconcile")  # exits and reconciliation still run


def test_unrecordable_risk_decision_blocks_the_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path, required=True)

    real = h.store

    class _NoDecisions(SqlAuditStore):
        def record_risk_decision(self, run_id: str, decision: RiskDecision) -> None:
            raise AuditError("disk full")

    h.services.recorder.store = _NoDecisions(h.engine, AppConfig(), clock=h.time.clock)
    assert real is not h.services.recorder.store
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["entry"] == "skipped"
    assert h.executor.count("enter") == 0


# ---------- Calendar ----------


@pytest.mark.parametrize("day", [date(2026, 10, 17), date(2026, 10, 18)])
def test_weekends_are_skipped(tmp_path: Path, day: date) -> None:
    h = Harness(tmp_path, start=at(8, 0, day))
    result = h.cycle.run_day(day)
    assert result.status == "skipped"
    assert result.summary == "market closed"
    assert h.calls == []
    assert h.rows(schema.runs) == []


def test_holidays_are_skipped(tmp_path: Path) -> None:
    h = Harness(tmp_path, calendar={"holidays": (THURSDAY,)})
    assert h.cycle.run_day(THURSDAY).status == "skipped"


def test_early_close_moves_the_force_close(tmp_path: Path) -> None:
    h = Harness(tmp_path, calendar={"early": {THURSDAY: time(13)}})
    session = h.cycle.session(THURSDAY)
    assert session is not None
    assert session.early_close
    assert session.force_close == at(12, 50)
    assert session.trade_end == at(12, 30)
    assert session.report == at(16, 15)

    result = h.cycle.run_day(THURSDAY)
    assert result.status == "traded"
    assert h.executor.calls[-1][1][1] == time(12, 50)  # manage told to close at 12:50
    assert h.executor.holding is False


def test_calendar_failure_is_an_error(tmp_path: Path) -> None:
    h = Harness(tmp_path, calendar={"fail": True})
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "error"
    assert "clock unavailable" in result.summary


# ---------- Restarts ----------


def test_restart_resumes_without_repeating_finished_steps(tmp_path: Path) -> None:
    first = Harness(tmp_path, start=at(8))
    store = first.store
    store.start_run(RUN_ID, "daily")
    store.record_step(RUN_ID, "preflight", "ok", "all checks passed", at(8, 30))
    store.record_analysis(_analysis())
    store.record_step(RUN_ID, "analysis", "proposed", "evidence verified", at(8, 45))

    restarted = Harness(tmp_path, start=at(10, 30), store=store)
    result = restarted.cycle.run_day(THURSDAY)

    assert not restarted.called("health")
    assert not restarted.called("analyze")  # no second LLM bill
    assert restarted.when("evaluate") == at(10, 30)  # proposal recovered from the store
    assert result.steps["entry"] == "entered"
    assert result.status == "traded"


def test_finished_day_is_not_rerun(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.cycle.run_day(THURSDAY)
    again = Harness(tmp_path, start=at(16, 20), store=h.store)
    # After the close the calendar no longer offers today, so nothing at all happens.
    assert again.cycle.run_day(THURSDAY).status == "skipped"
    assert again.calls == []


def test_late_start_skips_analysis_but_still_closes(tmp_path: Path) -> None:
    h = Harness(tmp_path, start=at(15, 45))
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["analysis"] == "skipped"
    assert result.steps["entry"] == "skipped"
    assert result.steps["close"] == "ok"
    assert not h.called("analyze")


def test_without_a_database_a_restart_cannot_double_buy(tmp_path: Path) -> None:
    """No store means no resume, but the entry reuses the day's run id, which the executor
    turns into the same client order id; the broker refuses a duplicate."""
    h = Harness(tmp_path, store=NullAuditStore())
    h.cycle.run_day(THURSDAY)
    assert h.executor.calls[0] == ("enter", RUN_ID)


# ---------- Day after day ----------


def test_run_forever_moves_from_day_to_day(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    days: list[str] = []

    def stop() -> bool:
        runs = {r["run_id"] for r in h.rows(schema.runs)}
        days[:] = sorted(runs)
        return len(runs) >= 2

    h.cycle.run_forever(should_stop=stop)
    assert days == ["daily-20261015", "daily-20261016"]  # Thursday, then Friday
    assert h.time.now.astimezone(NY).date() == date(2026, 10, 16)


# ---------- Every failure is contained ----------


def _boom(*_: Any) -> Any:
    raise RuntimeError("boom")


def _rebuild(h: Harness, **services: Any) -> DailyCycle:
    h.services = replace(h.services, **services)
    h.cycle = DailyCycle(h.services, h.config, clock=h.time.clock, sleep=h.time.sleep)
    return h.cycle


def test_health_check_crash_halts_trading(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _rebuild(h, health_check=_boom).run_day(THURSDAY)
    assert result.steps["preflight"] == "error"
    assert result.status == "halted"
    assert result.steps["reconcile"] == "ok"


def test_risk_check_crash_means_no_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _rebuild(h, evaluate=_boom).run_day(THURSDAY)
    assert result.steps["entry"] == "error"
    assert h.executor.count("enter") == 0


def test_unfilled_entry_is_no_trade(tmp_path: Path) -> None:
    h = Harness(tmp_path)

    def not_filled(decision: RiskDecision, run_id: str) -> EntryResult:
        return EntryResult(status="not_filled", reason="entry order canceled without a fill")

    h.executor.enter = not_filled  # type: ignore[method-assign]
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["entry"] == "not_filled"
    assert result.status == "no_trade"


def test_manage_crash_does_not_stop_the_loop(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    real = h.executor.manage
    crashes = iter([True, True])

    def flaky(force_close_at: time | None = None) -> tuple[ManageResult, ...]:
        if next(crashes, False):
            raise RuntimeError("broker timeout")
        return real(force_close_at)

    h.executor.manage = flaky  # type: ignore[method-assign]
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "traded"
    assert result.steps["close"] == "ok"


def test_close_crash_is_an_error(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    real = h.executor.manage

    def fails_at_close(force_close_at: time | None = None) -> tuple[ManageResult, ...]:
        if h.time.now.astimezone(NY).time() >= time(15, 50):
            raise RuntimeError("broker down")
        return real(force_close_at)

    h.executor.manage = fails_at_close  # type: ignore[method-assign]
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["close"] == "error"
    assert result.status == "error"


def test_reconcile_crash_is_recorded(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _rebuild(h, orders_since=_boom).run_day(THURSDAY)
    assert result.steps["reconcile"] == "error"
    assert result.status == "traded"  # the trade itself is unaffected


def test_unreadable_audit_store_falls_back_to_a_fresh_day(tmp_path: Path) -> None:
    class _Unreadable(NullAuditStore):
        def completed_steps(self, run_id: str) -> dict[str, str]:
            raise AuditError("read failed")

    h = Harness(tmp_path, store=_Unreadable())
    result = h.cycle.run_day(THURSDAY)
    assert result.status == "traded"
    assert h.called("health")


def test_unreadable_proposal_on_resume_means_no_entry(tmp_path: Path) -> None:
    class _Resumed(NullAuditStore):
        def completed_steps(self, run_id: str) -> dict[str, str]:
            return {"preflight": "ok", "analysis": "proposed"}

        def get_proposal(self, run_id: str) -> TradeProposal | None:
            raise AuditError("read failed")

    h = Harness(tmp_path, start=at(10), store=_Resumed())
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["entry"] == "skipped"
    assert not h.called("analyze")


def _resumed_store(h: Harness, steps: dict[str, str]) -> SqlAuditStore:
    store = SqlAuditStore(h.engine, AppConfig(), clock=h.time.clock)
    store.start_run(RUN_ID, "daily")
    if "analysis" in steps:
        store.record_analysis(_analysis())
    for step, status in steps.items():
        store.record_step(RUN_ID, step, status, "", at(8))
    return store


def test_resume_after_the_trade_window_skips_the_entry(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(setup, {"preflight": "ok", "analysis": "proposed"})
    h = Harness(tmp_path, start=at(15, 35), store=store)
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["entry"] == "skipped"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "entry"]
    assert step["detail"] == "missed the trade window"


def test_resume_after_the_close_only_reconciles(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(
        setup, {"preflight": "ok", "analysis": "proposed", "entry": "entered", "close": "ok"}
    )
    h = Harness(tmp_path, start=at(15, 55), store=store)
    result = h.cycle.run_day(THURSDAY)
    assert h.executor.count("manage") == 0
    assert h.executor.count("enter") == 0
    assert h.when("reconcile") == at(16, 15)
    assert result.status == "traded"
    assert result.summary == "entered AAPL"


def test_resume_with_a_failed_close_reports_an_error(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(setup, {"preflight": "ok", "close": "error", "analysis": "no_trade"})
    h = Harness(tmp_path, start=at(15, 55), store=store)
    assert h.cycle.run_day(THURSDAY).status == "error"


def test_run_forever_can_stop_before_starting(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.cycle.run_forever(should_stop=lambda: True)
    assert h.calls == []


# ---------- The report step ----------


def test_report_runs_last_with_the_days_result(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    seen: list[tuple[date, str | None]] = []

    def report(day: date) -> str:
        (run,) = h.rows(schema.runs)
        seen.append((day, run["status"]))  # the result is already recorded
        return "saved reports/2026-10-15.md"

    result = _rebuild(h, report=report).run_day(THURSDAY)
    assert seen == [(THURSDAY, "traded")]
    assert result.steps["report"] == "ok"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "report"]
    assert step["detail"] == "saved reports/2026-10-15.md"


def test_report_failure_is_contained(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _rebuild(h, report=_boom).run_day(THURSDAY)
    assert result.steps["report"] == "error"
    assert result.status == "traded"


def test_no_report_service_means_no_report_step(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    assert "report" not in h.cycle.run_day(THURSDAY).steps


def test_finished_report_is_not_repeated_on_resume(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(
        setup,
        {
            "preflight": "ok",
            "analysis": "no_trade",
            "entry": "skipped",
            "close": "ok",
            "reconcile": "ok",
            "report": "ok",
        },
    )
    h = Harness(tmp_path, start=at(15, 55), store=store)
    calls: list[date] = []
    _rebuild(h, report=lambda day: calls.append(day) or "x").run_day(THURSDAY)
    assert calls == []


# ---------- The reflect step (self-learning) ----------


def test_reflect_runs_after_the_result_and_before_the_report(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    order: list[str] = []

    def reflect(day: date) -> str:
        (run,) = h.rows(schema.runs)
        order.append(f"reflect {day} {run['status']}")
        return "1 new, 0 retired: a clean day"

    def report(day: date) -> str:
        order.append("report")
        return "saved"

    result = _rebuild(h, reflect=reflect, report=report).run_day(THURSDAY)
    assert order == [f"reflect {THURSDAY} traded", "report"]
    assert result.steps["reflect"] == "ok"
    assert result.status == "traded"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "reflect"]
    assert step["detail"] == "1 new, 0 retired: a clean day"


def test_reflect_failure_is_contained(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _rebuild(h, reflect=_boom, report=lambda day: "saved").run_day(THURSDAY)
    assert result.steps["reflect"] == "error"
    assert result.steps["report"] == "ok"  # a failed review never blocks the report
    assert result.status == "traded"


def test_no_reflect_service_means_no_reflect_step(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    assert "reflect" not in h.cycle.run_day(THURSDAY).steps


def test_finished_reflect_is_not_repeated_on_resume(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(
        setup,
        {
            "preflight": "ok",
            "analysis": "no_trade",
            "entry": "skipped",
            "close": "ok",
            "reconcile": "ok",
            "reflect": "ok",
        },
    )
    h = Harness(tmp_path, start=at(15, 55), store=store)
    calls: list[date] = []
    _rebuild(h, reflect=lambda day: calls.append(day) or "x").run_day(THURSDAY)
    assert calls == []


# ---------- The critique step ----------


def _critic_result(status: str) -> CriticResult:
    return CriticResult(
        run_id=RUN_ID,
        status=status,  # type: ignore[arg-type]
        model_verdict="approve" if status == "approved" else "veto",
        summary="ok" if status == "approved" else "momentum is fading",
        reasons=() if status == "approved" else ("the critic vetoed the trade",),
        concerns=(),
        evidence_checks=(),
        model="judge",
        prompt_version="p",
        usage=LLMUsage(input_tokens=5, output_tokens=1),
        created_at=at(8, 46),
    )


def _with_critic(h: Harness, result: CriticResult | Exception | None) -> list[int]:
    seen: list[int] = []

    def critique(run_id: str, proposal: TradeProposal, calls: list[Any]) -> CriticResult | None:
        seen.append(len(calls))
        if isinstance(result, Exception):
            raise result
        return result

    _rebuild(h, critique=critique)
    return seen


def test_approved_proposal_goes_on_to_the_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    _with_critic(h, _critic_result("approved"))
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["critique"] == "approved"
    assert result.status == "traded"
    assert h.rows(schema.critiques)[0]["status"] == "approved"


@pytest.mark.parametrize("status", ["vetoed", "failed"])
def test_veto_or_failure_means_no_entry(tmp_path: Path, status: str) -> None:
    h = Harness(tmp_path)
    _with_critic(h, _critic_result(status))
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["critique"] == status
    assert result.steps["entry"] == "skipped"
    assert not h.called("evaluate")  # the risk engine never sees a vetoed proposal
    assert result.status == "no_trade"
    assert result.summary == "skipped: the critic did not approve the proposal"


def test_critic_crash_means_no_entry(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    _with_critic(h, RuntimeError("model down"))
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["critique"] == "error"
    assert h.executor.count("enter") == 0


def test_disabled_critic_lets_the_proposal_through(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    _with_critic(h, None)
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["critique"] == "disabled"
    assert result.status == "traded"


def test_no_critique_without_a_proposal(tmp_path: Path) -> None:
    h = Harness(tmp_path, analysis="no_trade")
    seen = _with_critic(h, _critic_result("approved"))
    result = h.cycle.run_day(THURSDAY)
    assert seen == []
    assert "critique" not in result.steps


def test_critic_sees_the_analysts_tool_calls(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    calls = [
        ToolCallRecord(call_id="c1", tool="get_quote", arguments={}, is_error=False, result="x")
    ]
    _rebuild(
        h,
        analyze=lambda run_id, context=None: _analysis(run_id=run_id).model_copy(
            update={"tool_calls": tuple(calls)}
        ),
    )
    seen = _with_critic(h, _critic_result("approved"))
    h.cycle.run_day(THURSDAY)
    assert seen == [1]


@pytest.mark.parametrize(("recorded", "entered"), [("approved", True), ("vetoed", False)])
def test_resume_respects_the_recorded_critique(
    tmp_path: Path, recorded: str, entered: bool
) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(setup, {"preflight": "ok", "analysis": "proposed", "critique": recorded})
    h = Harness(tmp_path, start=at(10), store=store)
    seen = _with_critic(h, _critic_result("approved"))
    result = h.cycle.run_day(THURSDAY)
    assert seen == []  # no second review
    assert (h.executor.count("enter") == 1) is entered
    assert result.steps["critique"] == recorded


# ---------- Several trade slots a day (paper, ADR 0015) ----------

THREE = AppConfig.model_validate(
    {
        "risk": {"max_trades_per_day": 3, "max_open_positions": 3},
        "schedule": {"extra_analyses": ["10:30", "12:00"]},
    }
)


def _times(h: Harness, name: str) -> list[datetime]:
    return [t for call, t in h.calls if call == name]


def test_three_slots_run_on_schedule_each_as_its_own_run(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)
    result = h.cycle.run_day(THURSDAY)

    assert _times(h, "analyze") == [at(8, 45), at(10, 30), at(12)]
    assert _times(h, "evaluate") == [at(9, 45), at(10, 30), at(12)]
    assert [c for c in h.executor.calls if c[0] == "enter"] == [
        ("enter", RUN_ID),
        ("enter", f"{RUN_ID}-2"),
        ("enter", f"{RUN_ID}-3"),
    ]
    assert result.status == "traded"
    assert result.steps["slot2"] == "entered"
    assert result.steps["slot3"] == "entered"
    assert result.steps["entry"] == "entered"  # slot 1 keeps the original step names
    assert "slot 2: entered 0.4899 AAPL at $102" in result.summary

    runs = {r["run_id"]: r for r in h.rows(schema.runs)}
    assert set(runs) == {RUN_ID, f"{RUN_ID}-2", f"{RUN_ID}-3"}
    assert runs[f"{RUN_ID}-2"]["status"] == "traded"
    slot_steps = [s["step"] for s in h.rows(schema.run_steps) if s["run_id"] == f"{RUN_ID}-2"]
    assert slot_steps == ["analysis", "critique", "entry"]
    assert {a["run_id"] for a in h.rows(schema.analyses)} == set(runs)


def test_later_slots_are_told_what_is_already_held(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)
    h.cycle.run_day(THURSDAY)
    (_, first), (_, second), (_, third) = h.contexts
    assert first is None
    assert second is not None
    assert "trade slot 2 of 3" in second
    assert "Already bought today: AAPL" in second
    assert third is not None
    assert "trade slot 3 of 3" in third


def test_a_full_book_skips_the_slot_without_spending_tokens(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE, capacity=lambda: "3 positions open (max 3)")
    result = h.cycle.run_day(THURSDAY)
    assert _times(h, "analyze") == [at(8, 45)]
    assert result.steps["slot2"] == "skipped"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "slot2"]
    assert step["detail"] == "3 positions open (max 3)"


def test_a_broken_capacity_check_skips_the_slot(tmp_path: Path) -> None:
    def broken() -> str | None:
        raise RuntimeError("broker down")

    h = Harness(tmp_path, config=THREE, capacity=broken)
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["slot2"] == "skipped"
    assert _times(h, "analyze") == [at(8, 45)]


def test_a_halted_day_runs_no_slots(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE, health_ok=False)
    result = h.cycle.run_day(THURSDAY)
    assert not h.called("analyze")
    assert result.steps["slot2"] == "skipped"
    assert result.steps["slot3"] == "skipped"
    assert result.status == "halted"


def test_a_failing_slot_does_not_stop_the_next(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)
    calls = {"n": 0}

    def flaky(run_id: str, context: str | None = None) -> AnalystResult:
        calls["n"] += 1
        if run_id.endswith("-2"):
            raise RuntimeError("model down")
        return _analysis(run_id=run_id)

    result = _rebuild(h, analyze=flaky).run_day(THURSDAY)
    assert result.steps["slot2"] == "no_trade"
    assert result.steps["slot3"] == "entered"
    assert calls["n"] == 3
    slot2 = [s for s in h.rows(schema.run_steps) if s["run_id"] == f"{RUN_ID}-2"]
    assert {s["step"]: s["status"] for s in slot2}["analysis"] == "error"


def test_only_as_many_slots_as_trades_allowed(tmp_path: Path) -> None:
    two = THREE.model_copy(update={"risk": THREE.risk.model_copy(update={"max_trades_per_day": 2})})
    h = Harness(tmp_path, config=two)
    result = h.cycle.run_day(THURSDAY)
    assert "slot2" in result.steps
    assert "slot3" not in result.steps
    assert len(_times(h, "analyze")) == 2


def test_slots_after_an_early_trade_end_are_dropped(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE, calendar={"early": {THURSDAY: time(12)}})
    session = h.cycle.session(THURSDAY)
    assert session is not None
    assert session.trade_end == at(11, 30)
    assert session.slots == (at(10, 30),)


def test_finished_slots_are_not_repeated_on_resume(tmp_path: Path) -> None:
    setup = Harness(tmp_path, config=THREE)
    store = _resumed_store(
        setup,
        {
            "preflight": "ok",
            "analysis": "proposed",
            "critique": "disabled",
            "entry": "entered",
            "slot2": "entered",
        },
    )
    h = Harness(tmp_path, start=at(11), store=store, config=THREE)
    result = h.cycle.run_day(THURSDAY)
    assert _times(h, "analyze") == [at(12)]  # only slot 3 is left
    assert result.steps["slot2"] == "entered"
    assert result.steps["slot3"] == "entered"


def test_a_slot_interrupted_mid_way_resumes_its_own_steps(tmp_path: Path) -> None:
    setup = Harness(tmp_path, config=THREE)
    store = _resumed_store(
        setup,
        {"preflight": "ok", "analysis": "no_trade", "critique": "disabled", "entry": "skipped"},
    )
    slot = f"{RUN_ID}-2"
    store.start_run(slot, "daily")
    store.record_analysis(_analysis(run_id=slot))
    store.record_step(slot, "analysis", "proposed", "", at(10, 30))
    h = Harness(tmp_path, start=at(10, 35), store=store, config=THREE)
    result = h.cycle.run_day(THURSDAY)
    assert _times(h, "analyze") == [at(12)]  # slot 2's analysis is not repeated
    assert ("enter", slot) in h.executor.calls
    assert result.steps["slot2"] == "entered"


def test_extra_analysis_times_are_validated() -> None:
    with pytest.raises(ValueError, match="inside the trade window"):
        AppConfig.model_validate({"schedule": {"extra_analyses": ["09:00"]}})
    with pytest.raises(ValueError, match="inside the trade window"):
        AppConfig.model_validate({"schedule": {"extra_analyses": ["15:30"]}})
    with pytest.raises(ValueError, match="in order"):
        AppConfig.model_validate({"schedule": {"extra_analyses": ["12:00", "10:30"]}})


def test_a_slot_reached_after_the_trade_window_is_skipped(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)
    session = h.cycle.session(THURSDAY)
    assert session is not None
    late = replace(session, trade_end=at(10, 0))  # e.g. the process was paused
    h.cycle.session = lambda day: late  # type: ignore[method-assign]
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["slot2"] == "skipped"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "slot2"]
    assert step["detail"] == "too late: trade window over"


def test_a_slot_without_an_audit_record_does_not_trade(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE, required=True)

    class _NoSlotRuns(SqlAuditStore):
        def start_run(self, run_id: str, kind: str) -> None:
            if run_id.endswith(("-2", "-3")):
                raise AuditError("disk full")
            super().start_run(run_id, kind)

    h.services.recorder.store = _NoSlotRuns(h.engine, AppConfig(), clock=h.time.clock)
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["slot2"] == "skipped"
    assert [c for c in h.executor.calls if c[0] == "enter"] == [("enter", RUN_ID)]


def test_unreadable_slot_steps_start_the_slot_afresh(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)

    class _Forgetful(SqlAuditStore):
        def completed_steps(self, run_id: str) -> dict[str, str]:
            if run_id.endswith("-2"):
                raise AuditError("db down")
            return super().completed_steps(run_id)

    h.services.recorder.store = _Forgetful(h.engine, AppConfig(), clock=h.time.clock)
    result = h.cycle.run_day(THURSDAY)
    assert result.steps["slot2"] == "entered"


# ---------- Market watch, plan, and morning brief (ADR 0016) ----------


def _watched(h: Harness, **overrides: Any) -> DailyCycle:
    def tracked(name: str, result: str | None = "ok") -> Any:
        def call(day: date) -> str | None:
            h.calls.append((name, h._now()))
            return result

        return call

    services: dict[str, Any] = {
        "premarket": tracked("premarket", "scanned 12 symbols, 2 alerts"),
        "postmarket": tracked("postmarket", "scanned 8 symbols, 1 alert"),
        "plan": tracked("plan", "4 item(s) for 2026-10-16"),
        "brief": tracked("brief", "Morning brief (advisory: verify everything with your tools)."),
        "reflect": tracked("reflect", "1 new, 0 retired"),
        "report": tracked("report", "saved"),
    }
    services.update(overrides)
    return _rebuild(h, **services)


def test_the_day_starts_with_a_scan_and_ends_with_a_plan(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _watched(h).run_day(THURSDAY)
    order = [name for name, _ in h.calls if name != "evaluate"]
    assert order[:4] == ["premarket", "health", "brief", "analyze"]
    assert order[-5:] == ["reconcile", "postmarket", "reflect", "plan", "report"]
    assert h.when("premarket") == at(8)
    assert h.when("postmarket") == at(16, 45)
    assert result.steps["premarket"] == "ok"
    assert result.steps["postmarket"] == "ok"
    assert result.steps["plan"] == "ok"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "plan"]
    assert step["detail"] == "4 item(s) for 2026-10-16"


def test_the_analyst_gets_the_morning_brief(tmp_path: Path) -> None:
    h = Harness(tmp_path, config=THREE)
    _watched(h).run_day(THURSDAY)
    (_, first), (_, second), _ = h.contexts
    assert first == "Morning brief (advisory: verify everything with your tools)."
    assert second is not None
    assert second.startswith("Morning brief")
    assert "trade slot 2 of 3" in second


def test_a_failing_brief_never_blocks_the_analysis(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _watched(h, brief=_boom).run_day(THURSDAY)
    assert h.contexts == [(RUN_ID, None)]
    assert result.steps["analysis"] == "proposed"


@pytest.mark.parametrize("step", ["premarket", "postmarket", "plan"])
def test_watch_and_plan_failures_are_contained(tmp_path: Path, step: str) -> None:
    h = Harness(tmp_path)
    result = _watched(h, **{step: _boom}).run_day(THURSDAY)
    assert result.steps[step] == "error"
    assert result.status == "traded"
    assert result.steps["report"] == "ok"


def test_a_halted_day_still_scans_and_plans(tmp_path: Path) -> None:
    h = Harness(tmp_path, health_ok=False)
    result = _watched(h).run_day(THURSDAY)
    assert result.status == "halted"
    assert result.steps["postmarket"] == "ok"
    assert result.steps["plan"] == "ok"
    assert not h.called("brief")  # no analysis, so no brief


def test_finished_watch_steps_are_not_repeated(tmp_path: Path) -> None:
    setup = Harness(tmp_path)
    store = _resumed_store(
        setup,
        {
            "premarket": "ok",
            "preflight": "ok",
            "analysis": "no_trade",
            "entry": "skipped",
            "close": "ok",
            "reconcile": "ok",
            "postmarket": "ok",
            "reflect": "ok",
            "plan": "ok",
        },
    )
    h = Harness(tmp_path, start=at(15, 55), store=store)
    _watched(h).run_day(THURSDAY)
    assert {"premarket", "postmarket", "plan", "reflect"}.isdisjoint(n for n, _ in h.calls)
    assert h.called("report")


def test_post_market_waits_for_an_early_close(tmp_path: Path) -> None:
    h = Harness(tmp_path, calendar={"early": {THURSDAY: time(13)}})
    session = h.cycle.session(THURSDAY)
    assert session is not None
    assert session.premarket == at(8)
    assert session.post_market == at(16, 45)  # later of 16:45 and 30 minutes after 13:00


def test_the_next_day_starts_in_time_for_the_premarket_scan(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    _watched(h)
    days = {"n": 0}

    def stop() -> bool:
        days["n"] = len([c for c in h.calls if c[0] == "premarket"])
        return days["n"] >= 2

    h.cycle.run_forever(should_stop=stop)
    first, second = [t for name, t in h.calls if name == "premarket"]
    assert first == at(8)
    assert second == at(8, 0, date(2026, 10, 16))  # Friday's scan at 08:00, not 08:30


# ---------- Curation (ADR 0017) ----------


def test_curation_runs_between_the_review_and_the_plan(tmp_path: Path) -> None:
    h = Harness(tmp_path)

    def curate(day: date) -> str:
        h.calls.append(("curate", h._now()))
        return "promoted lesson #3; guidebook v2"

    result = _watched(h, curate=curate).run_day(THURSDAY)
    order = [name for name, _ in h.calls]
    assert order[-4:] == ["reflect", "curate", "plan", "report"]
    assert result.steps["curate"] == "ok"
    (step,) = [s for s in h.rows(schema.run_steps) if s["step"] == "curate"]
    assert step["detail"] == "promoted lesson #3; guidebook v2"


def test_a_failing_curation_is_contained(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    result = _watched(h, curate=_boom).run_day(THURSDAY)
    assert result.steps["curate"] == "error"
    assert result.steps["plan"] == "ok"
