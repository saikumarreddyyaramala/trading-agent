"""Outcome scoring (ADR 0017): which lessons and playbooks went with better trades.

Every closed system trade is joined with what its analysis knew: the lessons in the
analyst's prompt and the playbook it said it followed.

- Playbooks are named by the analyst, so their trades are directly attributable.
- Lessons are not: every lesson active at the time is in every prompt. A lesson is judged
  by comparing the trades that had it in context with those that did not, which mixes in
  whatever else changed over time. The curator therefore needs many trades on both sides
  before it acts, and the evidence is reported as what it is.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.reporting.performance import trades_from_orders
from trading_agent.storage.store import AuditStore

_HUNDREDTH = Decimal("0.01")


class TradeOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    day: date
    symbol: str
    return_pct: Decimal
    lesson_ids: tuple[int, ...]
    playbook: str | None


class Stats(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    mean_pct: Decimal
    wins: int
    win_rate_pct: Decimal

    def describe(self) -> str:
        return f"{self.mean_pct:+.2f}% over {self.n} trade{'s' if self.n != 1 else ''}"


class LessonOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    lesson_id: int
    with_: Stats | None
    without: Stats | None
    lift_pct: Decimal | None  # mean with minus mean without


def stats(returns: Sequence[Decimal]) -> Stats | None:
    if not returns:
        return None
    wins = sum(1 for r in returns if r > 0)
    return Stats(
        n=len(returns),
        mean_pct=(sum(returns, Decimal(0)) / len(returns)).quantize(_HUNDREDTH, ROUND_HALF_UP),
        wins=wins,
        win_rate_pct=(Decimal(wins) / len(returns) * 100).quantize(_HUNDREDTH, ROUND_HALF_UP),
    )


def trade_outcomes(
    store: AuditStore, zone: ZoneInfo, start: datetime, end: datetime
) -> list[TradeOutcome]:
    """Closed system trades in [start, end), each with what its analysis knew."""
    contexts = {c["run_id"]: c for c in store.analysis_contexts(start, end)}
    outcomes = []
    for trade in trades_from_orders(store.orders_between(start, end), zone):
        if not trade.closed or trade.return_pct is None:
            continue
        context = contexts.get(trade.run_id, {})
        outcomes.append(
            TradeOutcome(
                run_id=trade.run_id,
                day=trade.day,
                symbol=trade.symbol,
                return_pct=trade.return_pct,
                lesson_ids=tuple(context.get("lesson_ids", ())),
                playbook=context.get("playbook"),
            )
        )
    return outcomes


def lesson_outcome(lesson_id: int, trades: Sequence[TradeOutcome]) -> LessonOutcome:
    with_ = stats([t.return_pct for t in trades if lesson_id in t.lesson_ids])
    without = stats([t.return_pct for t in trades if lesson_id not in t.lesson_ids])
    lift = None if with_ is None or without is None else with_.mean_pct - without.mean_pct
    return LessonOutcome(lesson_id=lesson_id, with_=with_, without=without, lift_pct=lift)


def playbook_outcomes(trades: Sequence[TradeOutcome]) -> dict[str, Stats]:
    grouped: dict[str, list[Decimal]] = defaultdict(list)
    for trade in trades:
        if trade.playbook:
            grouped[trade.playbook].append(trade.return_pct)
    return {name: s for name, returns in grouped.items() if (s := stats(returns)) is not None}
