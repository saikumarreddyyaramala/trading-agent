"""The learning loop: review a finished day, keep what it teaches, forget what it disproves."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from trading_agent.agents.reflection import Reflector
from trading_agent.domain.ports import MarketDataPort
from trading_agent.evaluation.simulate import simulate_intraday
from trading_agent.observability.logging import get_logger
from trading_agent.reporting.performance import (
    SYSTEM_RUN_PREFIX,
    intraday_return_pct,
    trades_from_orders,
)
from trading_agent.reporting.slots import slot_runs, summarize_slot
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditStore, NewLessonRow


class LearningError(RuntimeError):
    """The day could not be reviewed (the reflector failed). Nothing was stored."""


class LearningService:
    def __init__(
        self,
        store: AuditStore,
        market_data: MarketDataPort,
        reflector: Reflector,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._market_data = market_data
        self._reflector = reflector
        self._config = config
        self._settings = config.learning
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def learn(self, day: date) -> str:
        """Review `day`, store new lessons, retire disproved ones. Returns a short summary."""
        run_id = f"{SYSTEM_RUN_PREFIX}{day:%Y%m%d}"
        run = self._store.get_run(run_id)
        if not run or not run["analysis"]:
            return "nothing to learn: no analysis that day"
        record = self.day_record(day, run)
        now = self._clock()
        active = self._store.active_lessons(now, self._settings.max_active)
        result = self._reflector.reflect(run_id, record, active)
        if result.status != "ok":
            raise LearningError(result.summary)

        evidence = {
            "run_id": run_id,
            "symbol": record["analysis"]["symbol"],
            "decision": record["analysis"]["decision"],
            "critic": (record["critic"] or {}).get("status"),
            "trade_returns_pct": [t["return_pct"] for t in record["trades"]],
            "spy_pct": record["benchmark_spy_open_to_close_pct"],
        }
        added = self._store.record_lessons(
            run_id,
            [
                NewLessonRow(text=lesson.text, category=lesson.category, evidence=evidence)
                for lesson in result.lessons
            ],
            expires_at=now + timedelta(days=self._settings.lesson_ttl_days),
            max_active=self._settings.max_active,
        )
        retired = [
            lesson_id
            for lesson_id in result.retire
            if self._store.retire_lesson(lesson_id, f"contradicted by the review of {run_id}")
        ]
        self._log.info(
            "learning.day_reviewed",
            run_id=run_id,
            added=added,
            retired=retired,
            rejected=list(result.rejected),
        )
        detail = f"{len(added)} new, {len(retired)} retired"
        if result.rejected:
            detail += f", {len(result.rejected)} rejected by guardrails"
        return f"{detail}: {result.summary}"

    def day_record(self, day: date, run: dict[str, Any]) -> dict[str, Any]:
        """The structured facts of the day. Deliberately excludes raw news and tool output."""
        analysis = run["analysis"]
        proposal = analysis.get("proposal") or {}
        critique = run.get("critique")
        decision = run["risk_decisions"][-1] if run["risk_decisions"] else None

        start = datetime.combine(day, time(0), tzinfo=self._zone)
        trades = trades_from_orders(
            self._store.orders_between(start, start + timedelta(days=1)), self._zone
        )
        today = [t for t in trades if t.day == day]

        symbol = proposal.get("symbol")
        benchmark = self._config.reporting.benchmark_symbol
        bars = self._bars([benchmark] + ([symbol] if symbol else []), day)
        spy = intraday_return_pct(bars.get(benchmark, ()), day, self._zone)

        stopped = None
        gate = (
            "critic"
            if critique and critique["status"] != "approved"
            else "risk engine"
            if decision and not decision["approved"]
            else None
        )
        if gate and symbol and proposal.get("reference_price"):
            bar = next(
                (
                    b
                    for b in bars.get(symbol, ())
                    if b.timestamp.astimezone(self._zone).date() == day
                ),
                None,
            )
            if bar is not None:
                sim = simulate_intraday(
                    bar,
                    Decimal(str(proposal["reference_price"])),
                    stop_loss_pct=self._config.risk.stop_loss_pct,
                    take_profit_pct=self._config.risk.take_profit_pct,
                )
                stopped = {
                    "gate": gate,
                    "simulated_return_pct": str(sim.return_pct),
                    "outcome": sim.outcome,
                }

        return {
            "day": str(day),
            "cycle_status": run["run"]["status"],
            "steps": {s["step"]: s["status"] for s in run["steps"]},
            "analysis": {
                "status": analysis["status"],
                "reason": analysis["reason"],
                "decision": analysis["decision"],
                "symbol": symbol,
                "confidence": analysis["confidence"],
                "thesis": proposal.get("thesis"),
                "evidence_claims": [e["claim"] for e in proposal.get("evidence", [])],
                "risks": proposal.get("risks", []),
            },
            "critic": None
            if not critique
            else {
                "status": critique["status"],
                "summary": critique["summary"],
                "reasons": critique["reasons"],
                "concerns": critique["concerns"],
            },
            "risk": None
            if not decision
            else {"approved": decision["approved"], "failed_checks": decision["failed_checks"]},
            "trades": [
                {
                    "symbol": trade.symbol,
                    "run_id": trade.run_id,
                    "entry_price": str(trade.entry_price),
                    "exit_price": None if trade.exit_price is None else str(trade.exit_price),
                    "return_pct": None if trade.return_pct is None else str(trade.return_pct),
                    "pnl_usd": None if trade.pnl_usd is None else str(trade.pnl_usd),
                    "exit_via": trade.exit_via,
                }
                for trade in today
            ],
            # Extra trade slots (ADR 0015): what each later analysis did and why.
            "slots": [
                summarize_slot(slot).model_dump(mode="json")
                for slot in slot_runs(self._store, run["run"]["run_id"])
            ],
            "benchmark_spy_open_to_close_pct": None if spy is None else str(spy),
            "stopped_idea_simulation": stopped,
            "position_events": [f"{e['action']}: {e['detail']}" for e in run["position_events"]],
            # The plan made for this day (ADR 0016): was it worth following?
            "plan_for_today": [
                f"[{i['kind']}] {i['text']}"
                for i in (self._store.plan_for(day) or {}).get("items", [])
            ],
            "after_hours_alerts": list(post.alerts)
            if (post := self._store.get_watch(day, "post"))
            else [],
        }

    def _bars(self, symbols: list[str], day: date) -> dict[str, Any]:
        today = self._clock().astimezone(self._zone).date()
        span = min(250, max((today - day).days, 0) * 5 // 7 + 5)
        try:
            return self._market_data.get_daily_bars(symbols, days=span)
        except Exception as exc:
            self._log.warning("learning.bars_unavailable", error=str(exc))
            return {}
