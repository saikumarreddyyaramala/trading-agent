from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

from trading_agent.execution.alpaca_broker import AlpacaBroker, BrokerError, mask_account_number


class _Status(Enum):
    ACTIVE = "ACTIVE"


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get_account(self) -> Any:
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(
            account_number="PA3XYZ9876",
            status=_Status.ACTIVE,
            currency="USD",
            cash="100000.5",
            equity="100250",
            buying_power="200000",
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
            multiplier="1",
            non_marginable_buying_power="99.5",
            daytrade_count=2,
        )

    def get_clock(self) -> Any:
        if self.fail:
            raise ConnectionError("network down")
        now = datetime(2026, 10, 15, 13, 30, tzinfo=UTC)
        return SimpleNamespace(timestamp=now, is_open=False, next_open=now, next_close=now)


def test_account_is_translated_and_masked() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_Client())
    account = broker.get_account()
    assert account.account_number_masked == "****9876"
    assert account.status == "ACTIVE"
    assert account.cash == Decimal("100000.5")
    assert account.equity == Decimal("100250")
    assert account.is_paper is True
    assert account.multiplier == Decimal("1")
    assert account.non_marginable_buying_power == Decimal("99.5")
    assert account.daytrade_count == 2


def test_clock_is_translated() -> None:
    clock = AlpacaBroker("k", "s", paper=True, client=_Client()).get_clock()
    assert clock.is_open is False


def test_errors_are_wrapped() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_Client(fail=True))
    with pytest.raises(BrokerError, match="could not fetch account"):
        broker.get_account()
    with pytest.raises(BrokerError, match="could not fetch market clock"):
        broker.get_clock()


@pytest.mark.parametrize(
    ("raw", "masked"),
    [("PA3XYZ9876", "****9876"), (None, "unknown"), ("", "unknown"), ("12", "****12")],
)
def test_mask_account_number(raw: str | None, masked: str) -> None:
    assert mask_account_number(raw) == masked


# ---------- Positions and orders ----------


class _TradingClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.order_filters: list[Any] = []

    def get_all_positions(self) -> Any:
        if self.fail:
            raise ConnectionError("network down")
        return [
            SimpleNamespace(
                symbol="AAPL", qty="0.1430", market_value="49.01", avg_entry_price="342.7"
            )
        ]

    def get_orders(self, filter: Any) -> Any:
        self.order_filters.append(filter)
        if self.fail:
            raise ConnectionError("network down")
        return [SimpleNamespace(id="o1"), SimpleNamespace(id="o2")]


def test_positions_are_translated() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_TradingClient())
    (position,) = broker.get_positions()
    assert position.symbol == "AAPL"
    assert position.quantity == Decimal("0.1430")
    assert position.market_value == Decimal("49.01")


def test_buy_orders_since_queries_all_statuses() -> None:
    client = _TradingClient()
    since = datetime(2026, 10, 15, 4, 0, tzinfo=UTC)
    assert AlpacaBroker("k", "s", paper=True, client=client).count_buy_orders_since(since) == 2
    request = client.order_filters[0]
    assert request.status is QueryOrderStatus.ALL
    assert request.side is OrderSide.BUY
    assert request.limit == 500
    assert request.after is not None


def test_position_and_order_errors_are_wrapped() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_TradingClient(fail=True))
    with pytest.raises(BrokerError, match="could not fetch positions"):
        broker.get_positions()
    with pytest.raises(BrokerError, match="could not fetch orders"):
        broker.count_buy_orders_since(datetime(2026, 10, 15, tzinfo=UTC))


# ---------- Dashboard support (ADR 0018) ----------


class _HistoryClient(_Client):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(fail=fail)
        self.requests: list[Any] = []

    def get_account(self) -> Any:
        account = super().get_account()
        account.last_equity = "100000"
        return account

    def get_portfolio_history(self, request: Any) -> Any:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(
            timestamp=[1760533200, 1760533500, 1760533800],
            equity=[100000.0, None, 100012.5],
            base_value=100000.0,
        )


def test_last_equity_gives_the_day_change() -> None:
    account = AlpacaBroker("k", "s", paper=True, client=_HistoryClient()).get_account()
    assert account.last_equity == Decimal("100000")


@pytest.mark.parametrize(
    ("span", "period", "timeframe"),
    [("1D", "1D", "5Min"), ("1W", "1W", "1H"), ("1M", "1M", "1D"), ("1Y", "1A", "1D")],
)
def test_portfolio_history_by_span(span: str, period: str, timeframe: str) -> None:
    client = _HistoryClient()
    history = AlpacaBroker("k", "s", paper=True, client=client).get_portfolio_history(span)
    assert (client.requests[0].period, client.requests[0].timeframe) == (period, timeframe)
    assert [p.equity for p in history.points] == [Decimal("100000.0"), Decimal("100012.5")]
    assert history.points[0].at == datetime(2025, 10, 15, 13, 0, tzinfo=UTC)
    assert history.base_value == Decimal("100000.0")
    assert history.span == span


def test_portfolio_history_errors_are_wrapped() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_HistoryClient(fail=True))
    with pytest.raises(BrokerError, match="could not fetch portfolio history"):
        broker.get_portfolio_history("1D")
    with pytest.raises(ValueError, match="span"):
        broker.get_portfolio_history("5Y")


# ---------- Market calendar (ADR 0023) ----------


class _CalendarClient:
    """Alpaca returns market-local naive times; holidays are simply missing."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[Any] = []

    def get_calendar(self, request: Any) -> list[Any]:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        days = {
            "2026-11-25": ("09:30", "16:00"),
            "2026-11-27": ("09:30", "13:00"),  # the day after Thanksgiving closes early
        }
        out = []
        for day, (open_, close) in days.items():
            if request.start.isoformat() <= day <= request.end.isoformat():
                d = datetime.fromisoformat(day)
                out.append(
                    SimpleNamespace(
                        date=d.date(),
                        open=datetime.fromisoformat(f"{day}T{open_}"),
                        close=datetime.fromisoformat(f"{day}T{close}"),
                    )
                )
        return out


def test_a_trading_day_has_its_hours_in_market_time() -> None:
    from datetime import date
    from zoneinfo import ZoneInfo

    broker = AlpacaBroker("k", "s", paper=True, client=_CalendarClient())
    day = broker.get_market_day(date(2026, 11, 27))
    assert day is not None
    ny = ZoneInfo("America/New_York")
    assert day.open == datetime(2026, 11, 27, 9, 30, tzinfo=ny)
    assert day.close == datetime(2026, 11, 27, 13, 0, tzinfo=ny)  # early close
    assert day.early_close


def test_a_holiday_has_no_market_day() -> None:
    from datetime import date

    client = _CalendarClient()
    broker = AlpacaBroker("k", "s", paper=True, client=client)
    assert broker.get_market_day(date(2026, 11, 26)) is None  # Thanksgiving
    assert client.requests[0].start == client.requests[0].end == date(2026, 11, 26)


def test_calendar_errors_are_wrapped() -> None:
    from datetime import date

    broker = AlpacaBroker("k", "s", paper=True, client=_CalendarClient(fail=True))
    with pytest.raises(BrokerError, match="could not fetch the market calendar"):
        broker.get_market_day(date(2026, 11, 25))
