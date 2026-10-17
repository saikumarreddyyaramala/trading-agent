"""The next-day plan: a to-do list for the next session, from the day's facts (ADR 0016).

Built by fixed rules, not a model, so every item can be traced to a recorded fact:

    review  trades that hit their stop, lost, or won; ideas the risk engine rejected
    watch   ideas the critic vetoed; after-hours alerts from the post-market scan
    ops     positions still open after the close, failed steps, models without a price
    apply   lessons the daily review learned today

The plan is shown in the report, and the next morning's analyst gets it (without the ops
items) in its brief. The next evening's review sees it too, so the loop can tell whether
the plan was worth following.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from trading_agent.agents.reflection import sanitize
from trading_agent.domain.models import DayPlan, PlanItem
from trading_agent.observability.logging import get_logger
from trading_agent.reporting.performance import SYSTEM_RUN_PREFIX, trades_from_orders
from trading_agent.reporting.slots import slot_runs
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditStore

Kind = Literal["watch", "review", "avoid", "ops", "apply"]


def _short(text: str, limit: int = 100) -> str:
    clean = sanitize(text)
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _next_weekday(day: date) -> date:
    following = day + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return following


class Planner:
    def __init__(
        self,
        store: AuditStore,
        config: AppConfig,
        *,
        next_session: Callable[[], date],
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        """`next_session` gives the next trading day (from the broker's calendar)."""
        self._store = store
        self._config = config
        self._zone = ZoneInfo(config.schedule.timezone)
        self._next_session = next_session
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def _for_day(self, day: date) -> date:
        try:
            following = self._next_session()
        except Exception as exc:
            self._log.warning("plan.calendar_unavailable", error=str(exc))
            return _next_weekday(day)
        return following if following > day else _next_weekday(day)

    def plan(self, day: date) -> DayPlan:
        """Build the plan for the session after `day`, store it, and return it."""
        run_id = f"{SYSTEM_RUN_PREFIX}{day:%Y%m%d}"
        items: list[PlanItem] = []

        def add(kind: Kind, priority: int, text: str, symbol: str | None = None) -> None:
            items.append(PlanItem(kind=kind, symbol=symbol, text=text, priority=priority))

        start = datetime.combine(day, time(0), tzinfo=self._zone)
        end = start + timedelta(days=1)

        # ---- Trades ----
        for trade in trades_from_orders(self._store.orders_between(start, end), self._zone):
            if trade.day != day:
                continue
            symbol = trade.symbol
            if not trade.closed:
                add(
                    "ops",
                    1,
                    f"{symbol} is still open after the close: check the broker and run "
                    "`trading-agent manage`.",
                    symbol,
                )
            elif trade.exit_via == "stop-loss":
                add(
                    "review",
                    2,
                    f"{symbol} hit its stop-loss ({trade.return_pct:+.2f}%): check what the "
                    "thesis missed before trading it again.",
                    symbol,
                )
            elif trade.pnl_usd is not None and trade.pnl_usd > 0:
                add(
                    "review",
                    3,
                    f"{symbol} won {trade.return_pct:+.2f}% ({trade.exit_via}): note what worked.",
                    symbol,
                )
            else:
                add(
                    "review",
                    2,
                    f"{symbol} lost {trade.return_pct:+.2f}% ({trade.exit_via}): check what "
                    "went wrong.",
                    symbol,
                )

        # ---- Ideas the gatekeepers stopped, and failed steps ----
        main = self._store.get_run(run_id)
        runs = [main, *slot_runs(self._store, run_id)] if main else []
        for run in runs:
            proposal = (run["analysis"] or {}).get("proposal") or {}
            idea: str | None = proposal.get("symbol") if proposal.get("decision") == "buy" else None
            critique = run.get("critique")
            decision = run["risk_decisions"][-1] if run["risk_decisions"] else None
            if idea and critique and critique["status"] != "approved":
                why = _short((critique["reasons"] or [critique["summary"]])[0])
                verdict = "vetoed by" if critique["status"] == "vetoed" else "not cleared by"
                add(
                    "watch",
                    2,
                    f"{idea} was {verdict} the critic ({why}): see how it trades before "
                    "trusting a similar setup.",
                    idea,
                )
            elif idea and decision and not decision["approved"]:
                failed = ", ".join(decision["failed_checks"]) or "a failed check"
                add(
                    "review",
                    3,
                    f"{idea} was rejected by the risk engine ({failed}): check whether the "
                    "rule saved money.",
                    idea,
                )
            for step in run["steps"]:
                if step["status"] == "error":
                    add(
                        "ops",
                        1,
                        f"Step {step['step']} failed in {run['run']['run_id']}: "
                        f"{_short(step['detail'] or 'no detail', 160).rstrip('.')}.",
                    )

        # ---- After-hours alerts ----
        post = self._store.get_watch(day, "post")
        for moved in post.items if post else ():
            if post and moved.alert:
                sentence = next(
                    (a for a in post.alerts if a.startswith(moved.symbol)), moved.symbol
                )
                add(
                    "watch",
                    1,
                    f"{sentence}: check the pre-market before considering it.",
                    moved.symbol,
                )

        # ---- Costs ----
        unpriced = sorted(
            {
                r["model"]
                for r in self._store.llm_calls_between(start, end)
                if r["cost_usd"] is None and (r["input_tokens"] or r["output_tokens"])
            }
        )
        for model in unpriced:
            add(
                "ops",
                2,
                f"Set a price for {model} under costs.llm_prices: until then costs and the "
                "daily budget cannot count it.",
            )

        # ---- Lessons learned today ----
        for lesson in self._store.list_lessons(include_retired=False):
            if lesson["source_run_id"] == run_id:
                add("apply", 2, f"Lesson learned: {_short(lesson['text'], 300)}")

        items.sort(key=lambda i: i.priority)  # stable: rule order within a priority
        items = items[: self._config.watch.max_plan_items]
        for_day = self._for_day(day)
        counts: dict[str, int] = {}
        for item in items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        summary = (
            f"{len(items)} item(s) for {for_day}: "
            + ", ".join(f"{n} {kind}" for kind, n in counts.items())
            if items
            else f"Nothing to follow up from {day}."
        )
        plan = DayPlan(
            source_day=day,
            for_day=for_day,
            created_at=self._clock(),
            items=tuple(items),
            summary=summary,
        )
        self._store.record_plan(run_id, plan)
        self._log.info("plan.made", day=str(day), for_day=str(for_day), items=len(items))
        return plan
