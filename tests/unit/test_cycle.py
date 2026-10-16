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
from trading_agent.agents.analyst import AnalystResult
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
    ) -> None:
        self.time = FakeTime(start or at(8))
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
        )
        self.cycle = DailyCycle(
            self.services, AppConfig(), clock=self.time.clock, sleep=self.time.sleep
        )

    def _now(self) -> datetime:
        return self.time.now.astimezone(NY)

    def _health(self) -> list[CheckResult]:
        self.calls.append(("health", self._now()))
        status = CheckStatus.PASS if self._health_ok else CheckStatus.FAIL
        return [CheckResult("Broker account", status, "ok" if self._health_ok else "blocked")]

    def _analyze(self, run_id: str) -> AnalystResult:
        self.calls.append(("analyze", self._now()))
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
    h.cycle = DailyCycle(h.services, AppConfig(), clock=h.time.clock, sleep=h.time.sleep)
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
