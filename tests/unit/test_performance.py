from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from tests.conftest import make_bars
from trading_agent.domain.models import Bar
from trading_agent.reporting.performance import (
    TradeResult,
    intraday_return_pct,
    summarize,
    trades_from_orders,
)

NY = ZoneInfo("America/New_York")
DAY = date(2026, 10, 15)


def _at(hour: int, minute: int = 0, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(UTC)


def _row(
    role: str,
    side: str,
    qty: str,
    price: str | None,
    at: datetime,
    *,
    symbol: str = "AAPL",
    run_id: str | None = "daily-20261015",
) -> dict[str, Any]:
    return {
        "role": role,
        "side": side,
        "symbol": symbol,
        "run_id": run_id,
        "filled_quantity": Decimal(qty),
        "filled_avg_price": None if price is None else Decimal(price),
        "submitted_at": at,
    }


ENTRY = _row("entry", "buy", "0.5", "100", _at(9, 45))


def test_profitable_exit() -> None:
    (trade,) = trades_from_orders([ENTRY, _row("exit", "sell", "0.5", "110", _at(15, 50))], NY)
    assert trade.closed
    assert trade.pnl_usd == Decimal("5.00")
    assert trade.return_pct == Decimal("10.00")
    assert trade.exit_price == Decimal("110.0000")
    assert trade.exit_via == "exit order"
    assert trade.day == DAY


def test_stop_loss_exit() -> None:
    (trade,) = trades_from_orders([ENTRY, _row("stop", "sell", "0.5", "95", _at(11))], NY)
    assert trade.pnl_usd == Decimal("-2.50")
    assert trade.return_pct == Decimal("-5.00")
    assert trade.exit_via == "stop-loss"


def test_exit_in_several_fills_uses_the_average_price() -> None:
    orders = [
        ENTRY,
        _row("exit", "sell", "0.2", "104", _at(15, 50)),
        _row("exit", "sell", "0.3", "106", _at(15, 51)),
    ]
    (trade,) = trades_from_orders(orders, NY)
    assert trade.exit_price == Decimal("105.2000")
    assert trade.pnl_usd == Decimal("2.60")


def test_open_and_partly_closed_positions_have_no_pnl() -> None:
    (open_trade,) = trades_from_orders([ENTRY], NY)
    assert (open_trade.closed, open_trade.pnl_usd, open_trade.exit_via) == (
        False,
        None,
        "still open",
    )
    (partial,) = trades_from_orders([ENTRY, _row("exit", "sell", "0.2", "101", _at(12))], NY)
    assert (partial.closed, partial.pnl_usd, partial.exit_via) == (False, None, "partly closed")


def test_fractional_rounding_still_counts_as_closed() -> None:
    (trade,) = trades_from_orders(
        [ENTRY, _row("exit", "sell", "0.4999999", "100", _at(15, 50))], NY
    )
    assert trade.closed


def test_only_matching_same_day_sells_after_the_entry_count() -> None:
    orders = [
        _row("exit", "sell", "0.5", "50", _at(9)),  # before the entry
        ENTRY,
        _row("exit", "sell", "0.5", "50", _at(10), symbol="MSFT"),  # other symbol
        _row("exit", "sell", "0.5", "50", _at(10, day=DAY + timedelta(days=1))),  # next day
        _row("exit", "sell", "0", None, _at(11)),  # unfilled
    ]
    (trade,) = trades_from_orders(orders, NY)
    assert not trade.closed


def test_manual_trades_are_excluded_unless_asked() -> None:
    manual = _row("entry", "buy", "1", "12.5", _at(14), symbol="F", run_id="itest-abc")
    orphan = _row("entry", "buy", "1", "12.5", _at(14), symbol="F", run_id=None)
    assert trades_from_orders([manual, orphan], NY) == []
    assert len(trades_from_orders([manual, orphan], NY, system_only=False)) == 2


def test_unusable_entries_are_skipped() -> None:
    orders = [
        _row("entry", "buy", "0", None, _at(9, 45)),
        _row("entry", "buy", "1", None, _at(9, 45)),
        {**ENTRY, "submitted_at": None},
        _row("stop", "sell", "1", "90", _at(10)),
    ]
    assert trades_from_orders(orders, NY) == []


def test_intraday_benchmark_return() -> None:
    bar = Bar(
        symbol="SPY",
        timestamp=datetime(
            2026, 10, 15, 4, 0, tzinfo=UTC
        ),  # Alpaca stamps daily bars at midnight ET
        open=Decimal("600"),
        high=Decimal("610"),
        low=Decimal("598"),
        close=Decimal("603"),
        volume=1,
    )
    assert intraday_return_pct([bar], DAY, NY) == Decimal("0.50")
    assert intraday_return_pct([bar], DAY + timedelta(days=1), NY) is None
    assert intraday_return_pct(make_bars("SPY", closes=("0",)), DAY, NY) is None


def _trade(day: date, pnl: str | None, ret: str | None) -> TradeResult:
    return TradeResult(
        day=day,
        run_id="daily-x",
        symbol="AAPL",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        exit_quantity=Decimal("1") if pnl else Decimal("0"),
        exit_price=None,
        closed=pnl is not None,
        pnl_usd=None if pnl is None else Decimal(pnl),
        return_pct=None if ret is None else Decimal(ret),
        exit_via="exit order",
    )


def test_summary_against_the_benchmark() -> None:
    d1, d2, d3 = DAY, DAY + timedelta(days=1), DAY + timedelta(days=4)
    trades = [_trade(d1, "2.00", "4.00"), _trade(d2, "-1.00", "-2.00"), _trade(d3, None, None)]
    perf = summarize(trades, {d1: Decimal("1.00"), d2: Decimal("-1.00")})
    assert perf.trades == 3
    assert perf.closed_trades == 2
    assert perf.wins == 1
    assert perf.win_rate_pct == Decimal("50.00")
    assert perf.total_pnl_usd == Decimal("1.00")
    assert perf.avg_return_pct == Decimal("1.00")
    assert perf.avg_benchmark_pct == Decimal("0.00")
    assert perf.avg_excess_pct == Decimal("1.00")


def test_summary_without_trades_or_benchmark() -> None:
    empty = summarize([], {})
    assert (empty.trades, empty.win_rate_pct, empty.avg_return_pct) == (0, None, None)
    no_bench = summarize([_trade(DAY, "1.00", "2.00")], {DAY: None})
    assert no_bench.avg_return_pct == Decimal("2.00")
    assert no_bench.avg_benchmark_pct is None
    assert no_bench.avg_excess_pct is None
