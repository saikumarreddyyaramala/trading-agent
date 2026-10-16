from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest

from trading_agent.execution.alpaca_broker import BrokerError
from trading_agent.execution.alpaca_orders import AlpacaOrderExecutor, to_broker_order


class _E(Enum):
    BUY = "buy"
    SELL = "sell"
    LIMIT = "limit"
    STOP = "stop"
    NEW = "new"
    FILLED = "filled"


def _sdk_order(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "id": "uuid-1",
        "client_order_id": "ta-run1-entry",
        "symbol": "AAPL",
        "side": _E.BUY,
        "order_type": _E.LIMIT,
        "type": _E.LIMIT,
        "status": _E.NEW,
        "qty": "0.4899",
        "filled_qty": "0",
        "filled_avg_price": None,
        "limit_price": "102.05",
        "stop_price": None,
        "submitted_at": datetime(2026, 10, 15, 14, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _api_error(status: int) -> APIError:
    return APIError("boom", SimpleNamespace(response=SimpleNamespace(status_code=status)))


class _Client:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []

    def _maybe_fail(self) -> None:
        if self.fail is not None:
            raise self.fail

    def submit_order(self, request: Any) -> Any:
        self.calls.append(("submit", request))
        self._maybe_fail()
        side = _E.SELL if request.side is OrderSide.SELL else _E.BUY
        return _sdk_order(client_order_id=request.client_order_id, side=side)

    def get_order_by_client_id(self, client_id: str) -> Any:
        self.calls.append(("get", client_id))
        self._maybe_fail()
        return _sdk_order(
            client_order_id=client_id, status=_E.FILLED, filled_qty="0.4899", filled_avg_price="102"
        )

    def get_orders(self, filter: Any) -> Any:
        self.calls.append(("list", filter))
        self._maybe_fail()
        return [_sdk_order()]

    def cancel_order_by_id(self, order_id: str) -> None:
        self.calls.append(("cancel", order_id))
        self._maybe_fail()

    def close_position(self, symbol: str) -> Any:
        self.calls.append(("close", symbol))
        self._maybe_fail()
        return _sdk_order(side=_E.SELL, order_type=None, type=_E.LIMIT, client_order_id="close-1")


def _executor(client: _Client) -> AlpacaOrderExecutor:
    return AlpacaOrderExecutor("k", "s", paper=True, client=client)


def test_limit_buy_request() -> None:
    client = _Client()
    order = _executor(client).submit_limit_buy(
        "AAPL", Decimal("0.4899"), Decimal("102.05"), "ta-run1-entry"
    )
    request = client.calls[0][1]
    assert isinstance(request, LimitOrderRequest)
    assert request.side is OrderSide.BUY
    assert request.time_in_force is TimeInForce.DAY
    assert request.qty == 0.4899
    assert request.limit_price == 102.05
    assert request.client_order_id == "ta-run1-entry"
    assert not request.extended_hours
    assert order.side == "buy"
    assert order.quantity == Decimal("0.4899")
    assert order.limit_price == Decimal("102.05")


def test_stop_sell_request() -> None:
    client = _Client()
    order = _executor(client).submit_stop_sell(
        "AAPL", Decimal("0.4899"), Decimal("95.09"), "ta-run1-stop"
    )
    request = client.calls[0][1]
    assert isinstance(request, StopOrderRequest)
    assert request.side is OrderSide.SELL
    assert request.stop_price == 95.09
    assert order.side == "sell"


@pytest.mark.parametrize("client_id", ["", "x" * 129])
def test_client_order_id_is_validated(client_id: str) -> None:
    client = _Client()
    with pytest.raises(ValueError, match="client_order_id"):
        _executor(client).submit_limit_buy("AAPL", Decimal("1"), Decimal("1"), client_id)
    assert client.calls == []


def test_get_order_translates_fills() -> None:
    order = _executor(_Client()).get_order("ta-run1-entry")
    assert order is not None
    assert order.is_filled
    assert order.is_terminal
    assert order.filled_quantity == Decimal("0.4899")
    assert order.filled_avg_price == Decimal("102")


def test_unknown_client_order_id_is_none() -> None:
    assert _executor(_Client(fail=_api_error(404))).get_order("nope") is None


def test_other_lookup_errors_are_raised() -> None:
    with pytest.raises(BrokerError, match="could not fetch order"):
        _executor(_Client(fail=_api_error(500))).get_order("x")
    with pytest.raises(BrokerError, match="could not fetch order"):
        _executor(_Client(fail=ConnectionError("down"))).get_order("x")


def test_open_orders_query() -> None:
    client = _Client()
    (order,) = _executor(client).get_open_orders("AAPL")
    request = client.calls[0][1]
    assert request.status is QueryOrderStatus.OPEN
    assert request.symbols == ["AAPL"]
    assert order.order_type == "limit"
    assert _executor(client).get_open_orders() is not None
    assert client.calls[1][1].symbols is None


def test_cancel_and_close() -> None:
    client = _Client()
    executor = _executor(client)
    executor.cancel_order("uuid-1")
    order = executor.close_position("AAPL")
    assert client.calls == [("cancel", "uuid-1"), ("close", "AAPL")]
    assert order.side == "sell"
    assert order.order_type == "limit"  # falls back to `type` when `order_type` is missing


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda e: e.submit_limit_buy("AAPL", Decimal("1"), Decimal("1"), "id"), "limit buy"),
        (lambda e: e.submit_stop_sell("AAPL", Decimal("1"), Decimal("1"), "id"), "stop sell"),
        (lambda e: e.get_open_orders(), "could not fetch open orders"),
        (lambda e: e.cancel_order("x"), "could not cancel order x"),
        (lambda e: e.close_position("AAPL"), "could not close AAPL"),
    ],
)
def test_errors_are_wrapped(call: Any, message: str) -> None:
    with pytest.raises(BrokerError, match=message):
        call(_executor(_Client(fail=ConnectionError("down"))))


def test_to_broker_order_handles_missing_numbers() -> None:
    order = to_broker_order(_sdk_order(qty=None, filled_qty=""))
    assert order.quantity == Decimal("0")
    assert order.filled_quantity == Decimal("0")


def test_orders_since_queries_all_statuses() -> None:
    client = _Client()
    since = datetime(2026, 10, 14, tzinfo=UTC)
    (order,) = _executor(client).get_orders_since(since)
    request = client.calls[0][1]
    assert request.status is QueryOrderStatus.ALL
    assert request.after is not None
    assert order.client_order_id == "ta-run1-entry"
    with pytest.raises(BrokerError, match="could not fetch orders"):
        _executor(_Client(fail=ConnectionError("down"))).get_orders_since(since)
