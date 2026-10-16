"""The daily trading cycle: one run per trading day, driven by the configured schedule.

    preflight (data_collection) -> analysis (analysis) -> entry (trade_window_start)
    -> manage every minute -> close (force_close) -> reconcile (report)

Rules:
- One run id per day (`daily-YYYYMMDD`). Every step is recorded once; a restarted process
  skips finished steps. Even without a database a restart cannot double-buy: entry order
  ids derive from the run id, and the risk engine counts today's orders.
- Each step is guarded. A failure stops new trades, never the exits: manage, close, and
  reconcile always run.
- The broker's market clock decides trading days and the session's close, so weekends,
  holidays, and early closes are handled.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.agents.critic import CriticResult
from trading_agent.domain.models import (
    BrokerOrder,
    MarketClock,
    RiskDecision,
    TradeProposal,
)
from trading_agent.execution.executor import EntryResult, ManageResult
from trading_agent.health import CheckResult, overall_ok
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditError, AuditRecorder, AuditStore

MAX_SLEEP_SECONDS = 60  # wake at least this often while waiting, to log and stay responsive
STEPS = ("preflight", "analysis", "critique", "entry", "close", "reconcile", "report")


class ExecutorLike(Protocol):
    def enter(self, decision: RiskDecision, run_id: str) -> EntryResult: ...

    def manage(self, force_close_at: time | None = None) -> tuple[ManageResult, ...]: ...

    def flatten(self) -> ManageResult: ...


@dataclass(frozen=True)
class CycleServices:
    """What the cycle needs, as plain callables, so each piece can be faked in tests."""

    health_check: Callable[[], list[CheckResult]]
    analyze: Callable[[str], AnalystResult]
    evaluate: Callable[[TradeProposal], RiskDecision]
    executor: ExecutorLike
    market_clock: Callable[[], MarketClock]
    orders_since: Callable[[datetime], tuple[BrokerOrder, ...]]
    recorder: AuditRecorder
    # Builds, saves, and sends the day's report; returns a short description. Optional.
    report: Callable[[date], str] | None = None
    # Reviews a buy proposal against the analyst's tool outputs; None result = disabled.
    critique: Callable[[str, TradeProposal, list[ToolCallRecord]], CriticResult | None] | None = (
        None
    )


@dataclass(frozen=True)
class Session:
    """One trading day's step times, adjusted for early closes."""

    day: date
    preflight: datetime
    analysis: datetime
    trade_start: datetime
    trade_end: datetime
    force_close: datetime
    report: datetime
    market_close: datetime
    early_close: bool


class DayResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    day: date
    status: str
    summary: str
    steps: dict[str, str]


class DailyCycle:
    def __init__(
        self,
        services: CycleServices,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        logger: Any = None,
    ) -> None:
        self._services = services
        self._config = config
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or _time.sleep
        self._log = logger or get_logger(__name__)
        # Outcome of each step of the day in progress, whether or not a database is present.
        self._steps: dict[str, str] = {}
        self._details: dict[str, str] = {}

    # ---------- Time ----------

    def _now(self) -> datetime:
        return self._clock().astimezone(self._zone)

    def _at(self, day: date, at: time) -> datetime:
        return datetime.combine(day, at, tzinfo=self._zone)

    def _wait_until(self, target: datetime, step: str) -> None:
        if self._now() < target:
            self._log.info("cycle.waiting", step=step, until=target.isoformat())
        while (remaining := (target - self._now()).total_seconds()) > 0:
            self._sleep(min(remaining, MAX_SLEEP_SECONDS))

    @staticmethod
    def run_id(day: date) -> str:
        return f"daily-{day:%Y%m%d}"

    def session(self, day: date) -> Session | None:
        """Step times for `day`, or None when the market does not open that day."""
        clock = self._services.market_clock()
        opens = clock.next_open.astimezone(self._zone)
        closes = clock.next_close.astimezone(self._zone)
        trading = (clock.is_open and clock.timestamp.astimezone(self._zone).date() == day) or (
            opens.date() == day
        )
        if not trading or closes.date() != day:
            return None

        schedule, tuning = self._config.schedule, self._config.orchestration
        force_close = min(
            self._at(day, schedule.force_close),
            closes - timedelta(minutes=tuning.early_close_buffer_minutes),
        )
        trade_end = min(
            self._at(day, schedule.trade_window_end), force_close - timedelta(minutes=20)
        )
        return Session(
            day=day,
            preflight=self._at(day, schedule.data_collection),
            analysis=self._at(day, schedule.analysis),
            trade_start=self._at(day, schedule.trade_window_start),
            trade_end=trade_end,
            force_close=force_close,
            report=max(self._at(day, schedule.report), closes + timedelta(minutes=5)),
            market_close=closes,
            early_close=force_close < self._at(day, schedule.force_close),
        )

    # ---------- Recording ----------

    def _record(self, what: str, fn: Callable[[AuditStore], None]) -> bool:
        """Write to the audit store. Returns False (never raises) when the write failed."""
        try:
            return self._services.recorder.write(what, fn)
        except AuditError:  # required store: already logged; callers decide what to block
            return False

    def _step(self, run_id: str, step: str, status: str, detail: str, started: datetime) -> None:
        self._steps[step] = status
        self._details[step] = detail
        self._log.info("cycle.step", run_id=run_id, step=step, status=status, detail=detail)
        self._record(f"step {step}", lambda s: s.record_step(run_id, step, status, detail, started))

    def _record_manage(self, run_id: str, results: tuple[ManageResult, ...]) -> None:
        changes = [r for r in results if r.action not in ("flat", "hold")]

        def record(store: AuditStore) -> None:
            for result in changes:
                store.record_position_event(run_id, result)

        if changes:
            self._record("position events", record)

    # ---------- The day ----------

    def run_day(self, day: date) -> DayResult:
        run_id = self.run_id(day)
        try:
            session = self.session(day)
        except Exception as exc:
            self._log.error("cycle.calendar_failed", run_id=run_id, error=str(exc))
            return DayResult(
                run_id=run_id, day=day, status="error", summary=f"calendar: {exc}", steps={}
            )
        if session is None:
            return DayResult(
                run_id=run_id, day=day, status="skipped", summary="market closed", steps={}
            )

        store = self._services.recorder.store
        try:
            done = dict(store.completed_steps(run_id))
        except AuditError:
            done = {}
        self._steps = done  # steps finished before a restart, then this run's as they finish
        self._details = {}
        audit_ok = self._record("run", lambda s: s.start_run(run_id, "daily"))
        if not audit_ok and self._services.recorder.required:
            self._log.error("cycle.audit_unavailable", run_id=run_id)
        if session.early_close:
            self._log.warning("cycle.early_close", force_close=session.force_close.isoformat())

        halted = self._preflight(run_id, session, done)
        if not audit_ok and self._services.recorder.required:
            halted = True  # no record, no trade
        proposal, calls = self._analysis(run_id, session, done, halted)
        proposal = self._critique(run_id, done, proposal, calls)
        traded = self._entry(run_id, session, done, halted, proposal)
        self._manage_until(run_id, session)
        closed_ok = self._close(run_id, session, done)
        self._reconcile(run_id, session, done)

        steps = dict(self._steps)
        if not closed_ok:
            status, summary = "error", "positions may still be open after the force-close"
        elif traded is not None:
            status, summary = "traded", traded
        elif halted:
            status, summary = "halted", "pre-flight failed or audit unavailable: no new trades"
        else:
            entry = steps.get("entry", "no entry")
            detail = self._details.get("entry")
            status, summary = "no_trade", f"{entry}: {detail}" if detail else entry
        self._record("run result", lambda s: s.finish_run(run_id, status, summary))
        self._report(run_id, session, done)  # after the result, so the report includes it
        self._log.info("cycle.day_finished", run_id=run_id, status=status, summary=summary)
        return DayResult(
            run_id=run_id, day=day, status=status, summary=summary, steps=dict(self._steps)
        )

    def _preflight(self, run_id: str, session: Session, done: dict[str, str]) -> bool:
        """Returns True when trading must halt for the day."""
        if "preflight" in done:
            return done["preflight"] != "ok"
        self._wait_until(session.preflight, "preflight")
        started = self._now()
        try:
            results = self._services.health_check()
        except Exception as exc:
            self._step(run_id, "preflight", "error", str(exc), started)
            done["preflight"] = "error"
            return True
        ok = overall_ok(results)
        failed = [f"{r.name}: {r.detail}" for r in results if r.status.value == "FAIL"]
        detail = "all checks passed" if ok else "; ".join(failed)
        status = "ok" if ok else "failed"
        self._step(run_id, "preflight", status, detail, started)
        done["preflight"] = status
        return not ok

    def _analysis(
        self, run_id: str, session: Session, done: dict[str, str], halted: bool
    ) -> tuple[TradeProposal | None, list[ToolCallRecord]]:
        """The verified proposal, if any, and the tool calls that produced its evidence."""
        if "analysis" in done:
            store = self._services.recorder.store
            try:
                return store.get_proposal(run_id), store.tool_calls(run_id)
            except AuditError:
                return None, []
        started = self._now()
        if halted:
            self._step(run_id, "analysis", "skipped", "trading halted", started)
            return None, []
        if self._now() >= session.trade_end:
            self._step(run_id, "analysis", "skipped", "too late: trade window over", started)
            return None, []
        self._wait_until(session.analysis, "analysis")
        started = self._now()
        try:
            result = self._services.analyze(run_id)
        except Exception as exc:
            self._step(run_id, "analysis", "error", str(exc), started)
            return None, []
        self._record("analysis", lambda s: s.record_analysis(result))
        self._step(run_id, "analysis", result.status, result.reason, started)
        done["analysis"] = result.status
        if result.status == "proposed" and result.proposal is not None:
            return result.proposal, list(result.tool_calls)
        return None, []

    def _critique(
        self,
        run_id: str,
        done: dict[str, str],
        proposal: TradeProposal | None,
        calls: list[ToolCallRecord],
    ) -> TradeProposal | None:
        """The proposal if the critic approves it (or is disabled); None after a veto."""
        if proposal is None:
            return None
        if "critique" in done:
            return proposal if done["critique"] in ("approved", "disabled") else None
        started = self._now()
        if self._services.critique is None:
            self._step(run_id, "critique", "disabled", "no critic configured", started)
            return proposal
        try:
            result = self._services.critique(run_id, proposal, calls)
        except Exception as exc:
            self._step(run_id, "critique", "error", str(exc), started)
            return None  # no review, no trade
        if result is None:
            self._step(run_id, "critique", "disabled", "critic disabled in config", started)
            return proposal
        self._record("critique", lambda s: s.record_critique(result))
        detail = "; ".join(result.reasons) or result.summary
        self._step(run_id, "critique", result.status, detail, started)
        return proposal if result.status == "approved" else None

    def _entry(
        self,
        run_id: str,
        session: Session,
        done: dict[str, str],
        halted: bool,
        proposal: TradeProposal | None,
    ) -> str | None:
        """Returns a summary when a position was entered."""
        if "entry" in done:
            return f"entered {proposal.symbol}" if done["entry"] == "entered" and proposal else None
        started = self._now()
        if halted:
            self._step(run_id, "entry", "skipped", "trading halted", started)
            return None
        if proposal is None:
            reason = (
                "the critic did not approve the proposal"
                if self._steps.get("critique") in ("vetoed", "failed", "error")
                else "no verified proposal"
            )
            self._step(run_id, "entry", "skipped", reason, started)
            return None
        if self._now() > session.trade_end:
            self._step(run_id, "entry", "skipped", "missed the trade window", started)
            return None
        self._wait_until(session.trade_start, "entry")
        started = self._now()
        try:
            decision = self._services.evaluate(proposal)
            # When the audit store is required, no record of the decision means no trade.
            if (
                not self._record(
                    "risk decision", lambda s: s.record_risk_decision(run_id, decision)
                )
                and self._services.recorder.required
            ):
                self._step(run_id, "entry", "skipped", "audit store unavailable", started)
                return None
            if not decision.approved:
                failed = ", ".join(c.name for c in decision.failed_checks)
                self._step(run_id, "entry", "rejected", f"risk checks failed: {failed}", started)
                return None
            entry = self._services.executor.enter(decision, run_id)
        except Exception as exc:
            self._step(run_id, "entry", "error", str(exc), started)
            return None
        orders = [o for o in (entry.entry_order, entry.stop_order) if o is not None]
        self._record("entry orders", lambda s: s.record_orders(run_id, orders))
        self._step(run_id, "entry", entry.status, entry.reason, started)
        if entry.status == "entered":
            return f"entered {entry.filled_quantity} {entry.symbol} at ${entry.avg_fill_price}"
        return None

    def _manage_until(self, run_id: str, session: Session) -> None:
        """Manage open positions until the force-close. Stops early once flat for good."""
        interval = self._config.orchestration.manage_interval_seconds
        self._wait_until(min(session.trade_start, session.force_close), "manage")
        while self._now() < session.force_close:
            try:
                results = self._services.executor.manage(session.force_close.time())
            except Exception as exc:
                self._log.error("cycle.manage_failed", run_id=run_id, error=str(exc))
                results = ()
            self._record_manage(run_id, results)
            if results and all(r.action == "flat" for r in results):
                break  # entries happen only in the entry step, which is over: nothing to manage
            remaining = (session.force_close - self._now()).total_seconds()
            self._sleep(max(min(interval, remaining), 0))

    def _close(self, run_id: str, session: Session, done: dict[str, str]) -> bool:
        """Force-close and verify the account is flat. Returns False if it may not be."""
        if "close" in done:
            return done["close"] == "ok"
        self._wait_until(session.force_close, "close")
        started = self._now()
        close_at = session.force_close.time()
        try:
            self._record_manage(run_id, self._services.executor.manage(close_at))
            self._sleep(self._config.orchestration.close_settle_seconds)
            check = self._services.executor.manage(close_at)
            if all(r.action == "flat" for r in check):
                self._step(run_id, "close", "ok", "flat", started)
                return True
            self._record_manage(run_id, check)
            flattened = self._services.executor.flatten()  # last resort
            self._record_manage(run_id, (flattened,))
            status = "ok" if flattened.action == "flattened" else "error"
            self._step(run_id, "close", status, f"flatten: {flattened.detail}", started)
            return status == "ok"
        except Exception as exc:
            self._step(run_id, "close", "error", str(exc), started)
            return False

    def _reconcile(self, run_id: str, session: Session, done: dict[str, str]) -> None:
        if "reconcile" in done:
            return
        self._wait_until(session.report, "reconcile")
        started = self._now()
        try:
            orders = self._services.orders_since(self._at(session.day, time(0)))
        except Exception as exc:
            self._step(run_id, "reconcile", "error", str(exc), started)
            return
        self._record("orders", lambda s: s.record_orders(None, orders))
        self._step(run_id, "reconcile", "ok", f"synced {len(orders)} order(s)", started)

    def _report(self, run_id: str, session: Session, done: dict[str, str]) -> None:
        if "report" in done or self._services.report is None:
            return
        started = self._now()
        try:
            detail = self._services.report(session.day)
        except Exception as exc:
            self._step(run_id, "report", "error", str(exc), started)
            return
        self._step(run_id, "report", "ok", detail, started)

    # ---------- Forever ----------

    def run_forever(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Run each trading day in turn. Stop with Ctrl+C; broker-side stops stay in place."""
        while not should_stop():
            today = self._now().date()
            result = self.run_day(today)
            self._log.info("cycle.next_day", finished=result.run_id, status=result.status)
            if should_stop():
                break
            tomorrow = today + timedelta(days=1)
            self._wait_until(self._at(tomorrow, self._config.schedule.data_collection), "next day")
