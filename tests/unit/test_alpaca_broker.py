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
