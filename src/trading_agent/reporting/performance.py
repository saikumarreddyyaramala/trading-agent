"""Trade results and benchmark comparison, computed from recorded orders. No I/O here.

A trade is one filled entry order placed by the daily cycle (`run_id` starting "daily-"),
plus the same-day filled sells of that symbol that closed it. Manual orders (the CLI `buy`
command, test round trips) are excluded from the performance record.

The benchmark is SPY's open-to-close move on the trade's day: positions are intraday, so
that is the return the same money would have earned by holding the index for the session.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.domain.models import Bar

SYSTEM_RUN_PREFIX = "daily-"
_ZERO = Decimal("0")
_TOLERANCE = Decimal("0.000001")  # fractional fills round at the broker


def _pct(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _usd(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


class TradeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    run_id: str
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    exit_quantity: Decimal
    exit_price: Decimal | None
    closed: bool
    pnl_usd: Decimal | None
    return_pct: Decimal | None
    exit_via: str  # "stop-loss", "exit order", or "still open"


class Performance(BaseModel):
    model_config = ConfigDict(frozen=True)

    trades: int
    closed_trades: int
    wins: int
    win_rate_pct: Decimal | None
    total_pnl_usd: Decimal
    avg_return_pct: Decimal | None
    avg_benchmark_pct: Decimal | None
    avg_excess_pct: Decimal | None


def _filled(order: Mapping[str, Any]) -> bool:
    return order["filled_quantity"] is not None and order["filled_quantity"] > 0


def trades_from_orders(
    orders: Sequence[Mapping[str, Any]], zone: ZoneInfo, *, system_only: bool = True
) -> list[TradeResult]:
    """Rebuild trades from order rows (as returned by the audit store), oldest first."""

    def local_day(order: Mapping[str, Any]) -> date | None:
        submitted: datetime | None = order["submitted_at"]
        return None if submitted is None else submitted.astimezone(zone).date()

    trades: list[TradeResult] = []
    for entry in orders:
        if entry["role"] != "entry" or entry["side"] != "buy" or not _filled(entry):
            continue
        run_id = entry["run_id"] or ""
        if system_only and not run_id.startswith(SYSTEM_RUN_PREFIX):
            continue
        day = local_day(entry)
        if day is None or entry["filled_avg_price"] is None:
            continue
        sells = [
            o
            for o in orders
            if o["side"] == "sell"
            and o["symbol"] == entry["symbol"]
            and _filled(o)
            and o["filled_avg_price"] is not None
            and local_day(o) == day
            and o["submitted_at"] >= entry["submitted_at"]
        ]
        quantity: Decimal = entry["filled_quantity"]
        entry_price: Decimal = entry["filled_avg_price"]
        sold = sum((o["filled_quantity"] for o in sells), _ZERO)
        proceeds = sum((o["filled_quantity"] * o["filled_avg_price"] for o in sells), _ZERO)
        closed = sold >= quantity - _TOLERANCE
        exit_price = proceeds / sold if sold > 0 else None
        pnl = _usd(proceeds - quantity * entry_price) if closed else None
        cost = quantity * entry_price
        if any(o["role"] in ("stop", "restore_stop") for o in sells):
            exit_via = "stop-loss"
        elif sells:
            exit_via = "exit order"
        else:
            exit_via = "still open"
        trades.append(
            TradeResult(
                day=day,
                run_id=run_id,
                symbol=entry["symbol"],
                quantity=quantity,
                entry_price=entry_price,
                exit_quantity=sold,
                exit_price=None if exit_price is None else exit_price.quantize(Decimal("0.0001")),
                closed=closed,
                pnl_usd=pnl,
                return_pct=None if pnl is None or cost <= 0 else _pct(pnl / cost * 100),
                exit_via=exit_via if closed or not sells else "partly closed",
            )
        )
    return trades


def intraday_return_pct(bars: Iterable[Bar], day: date, zone: ZoneInfo) -> Decimal | None:
    """Open-to-close move of the daily bar for `day`, in percent."""
    for bar in bars:
        if bar.timestamp.astimezone(zone).date() == day and bar.open > 0:
            return _pct((bar.close - bar.open) / bar.open * 100)
    return None


def summarize(
    trades: Sequence[TradeResult], benchmark: Mapping[date, Decimal | None]
) -> Performance:
    closed = [t for t in trades if t.closed and t.return_pct is not None]
    wins = sum(1 for t in closed if (t.pnl_usd or _ZERO) > 0)
    returns = [t.return_pct for t in closed if t.return_pct is not None]
    paired = [
        (t.return_pct, benchmark[t.day])
        for t in closed
        if t.return_pct is not None and benchmark.get(t.day) is not None
    ]

    def mean(values: Sequence[Decimal]) -> Decimal | None:
        return _pct(sum(values, _ZERO) / len(values)) if values else None

    return Performance(
        trades=len(trades),
        closed_trades=len(closed),
        wins=wins,
        win_rate_pct=_pct(Decimal(wins) / len(closed) * 100) if closed else None,
        total_pnl_usd=_usd(sum((t.pnl_usd or _ZERO for t in closed), _ZERO)),
        avg_return_pct=mean(returns),
        avg_benchmark_pct=mean([b for _, b in paired if b is not None]),
        avg_excess_pct=mean([r - b for r, b in paired if r is not None and b is not None]),
    )
