"""Alpaca adapter implementing `OrderExecutionPort`: the code that can change the account.

Kept apart from the read-only `AlpacaBroker` so that nothing built for research or agents
can reach it by accident. Orders are plain DAY orders with no extended hours. Alpaca does
not support bracket or OCO orders for fractional shares, so exits are separate orders
placed and managed by the executor.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest

from trading_agent.domain.models import BrokerOrder
from trading_agent.execution.alpaca_broker import BrokerError

MAX_CLIENT_ORDER_ID = 128  # Alpaca's limit


def _text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _decimal(value: Any) -> Decimal | None:
    return None if value in (None, "") else Decimal(str(value))


def to_broker_order(order: Any) -> BrokerOrder:
    side = _text(order.side).lower()
    return BrokerOrder(
        id=str(order.id),
        client_order_id=str(order.client_order_id),
        symbol=str(order.symbol),
        side="sell" if side == "sell" else "buy",
        order_type=_text(order.order_type or order.type),
        status=_text(order.status).lower(),
        quantity=_decimal(order.qty) or Decimal("0"),
        filled_quantity=_decimal(order.filled_qty) or Decimal("0"),
        filled_avg_price=_decimal(order.filled_avg_price),
        limit_price=_decimal(order.limit_price),
        stop_price=_decimal(order.stop_price),
        submitted_at=order.submitted_at,
    )


def _check_client_id(client_order_id: str) -> None:
    if not client_order_id or len(client_order_id) > MAX_CLIENT_ORDER_ID:
        raise ValueError(f"client_order_id must be 1-{MAX_CLIENT_ORDER_ID} characters")


class AlpacaOrderExecutor:
    def __init__(
        self, api_key: str, secret_key: str, *, paper: bool, client: Any | None = None
    ) -> None:
        self._client: Any = client or TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )

    def _submit(self, request: Any, what: str) -> BrokerOrder:
        try:
            return to_broker_order(self._client.submit_order(request))
        except Exception as exc:
            raise BrokerError(f"{what} rejected: {type(exc).__name__}: {exc}") from exc

    def submit_limit_buy(
        self, symbol: str, quantity: Decimal, limit_price: Decimal, client_order_id: str
    ) -> BrokerOrder:
        _check_client_id(client_order_id)
        request = LimitOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=float(limit_price),
            client_order_id=client_order_id,
        )
        return self._submit(request, f"limit buy {quantity} {symbol}")

    def submit_stop_sell(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> BrokerOrder:
        _check_client_id(client_order_id)
        request = StopOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=float(stop_price),
            client_order_id=client_order_id,
        )
        return self._submit(request, f"stop sell {quantity} {symbol}")

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        try:
            order = self._client.get_order_by_client_id(client_order_id)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError(f"could not fetch order {client_order_id}: {exc}") from exc
        except Exception as exc:
            raise BrokerError(f"could not fetch order {client_order_id}: {exc}") from exc
        return to_broker_order(order)

    def get_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]:
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol] if symbol else None, limit=500
        )
        try:
            orders = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"could not fetch open orders: {type(exc).__name__}: {exc}") from exc
        return tuple(to_broker_order(order) for order in orders)

    def get_orders_since(self, since: datetime) -> tuple[BrokerOrder, ...]:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=500)
        try:
            orders = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"could not fetch orders: {type(exc).__name__}: {exc}") from exc
        return tuple(to_broker_order(order) for order in orders)

    def cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as exc:
            raise BrokerError(f"could not cancel order {order_id}: {exc}") from exc

    def close_position(self, symbol: str) -> BrokerOrder:
        try:
            return to_broker_order(self._client.close_position(symbol))
        except Exception as exc:
            raise BrokerError(f"could not close {symbol}: {type(exc).__name__}: {exc}") from exc
