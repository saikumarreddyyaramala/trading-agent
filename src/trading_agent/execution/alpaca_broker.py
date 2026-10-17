"""Alpaca adapter implementing `BrokerPort`.

Read-only so far: account, market clock, positions, and today's order count. Order placement
arrives in Phase 5 and will only ever be called after the risk engine approves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

from trading_agent.domain.models import (
    AccountSnapshot,
    MarketClock,
    PortfolioHistory,
    PortfolioPoint,
    Position,
)

# Alpaca returns at most 500 orders per request; far above one trade a day.
MAX_ORDERS_PER_QUERY = 500


class BrokerError(RuntimeError):
    """The broker could not be reached or returned an unusable response."""


def mask_account_number(value: str | None) -> str:
    """Show only the last 4 characters, enough to tell accounts apart in logs."""
    if not value:
        return "unknown"
    return f"****{value[-4:]}"


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value not in (None, "") else Decimal("0")


def _to_text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


class AlpacaBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool,
        client: Any | None = None,
    ) -> None:
        self._paper = paper
        # Typed as Any: the SDK's return types are unions with raw dicts (raw_data mode),
        # and we always use model mode, so we translate attributes explicitly below.
        self._client: Any = client or TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )

    @property
    def is_paper(self) -> bool:
        return self._paper

    def get_account(self) -> AccountSnapshot:
        try:
            account = self._client.get_account()
        except Exception as exc:  # SDK raises APIError and network errors
            raise BrokerError(f"could not fetch account: {type(exc).__name__}: {exc}") from exc

        return AccountSnapshot(
            account_number_masked=mask_account_number(account.account_number),
            status=_to_text(account.status),
            currency=account.currency or "USD",
            cash=_to_decimal(account.cash),
            equity=_to_decimal(account.equity),
            buying_power=_to_decimal(account.buying_power),
            pattern_day_trader=bool(account.pattern_day_trader),
            trading_blocked=bool(account.trading_blocked),
            account_blocked=bool(account.account_blocked),
            is_paper=self._paper,
            multiplier=_to_decimal(account.multiplier or "1"),
            non_marginable_buying_power=None
            if account.non_marginable_buying_power in (None, "")
            else _to_decimal(account.non_marginable_buying_power),
            daytrade_count=int(account.daytrade_count or 0),
            last_equity=None
            if getattr(account, "last_equity", None) in (None, "")
            else _to_decimal(account.last_equity),
        )

    def get_portfolio_history(self, span: str) -> PortfolioHistory:
        return _history(self._client, span)

    def get_clock(self) -> MarketClock:
        try:
            clock = self._client.get_clock()
        except Exception as exc:
            raise BrokerError(f"could not fetch market clock: {type(exc).__name__}: {exc}") from exc

        return MarketClock(
            timestamp=clock.timestamp,
            is_open=bool(clock.is_open),
            next_open=clock.next_open,
            next_close=clock.next_close,
        )

    def get_positions(self) -> tuple[Position, ...]:
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            raise BrokerError(f"could not fetch positions: {type(exc).__name__}: {exc}") from exc
        return tuple(
            Position(
                symbol=str(position.symbol),
                quantity=_to_decimal(position.qty),
                market_value=_to_decimal(position.market_value),
                avg_entry_price=_to_decimal(position.avg_entry_price),
            )
            for position in positions
        )

    def count_buy_orders_since(self, since: datetime) -> int:
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            side=OrderSide.BUY,
            after=since,
            limit=MAX_ORDERS_PER_QUERY,
        )
        try:
            orders = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"could not fetch orders: {type(exc).__name__}: {exc}") from exc
        return len(orders)


# Dashboard chart spans (ADR 0018): our name -> Alpaca's (period, timeframe).
HISTORY_SPANS: dict[str, tuple[str, str]] = {
    "1D": ("1D", "5Min"),
    "1W": ("1W", "1H"),
    "1M": ("1M", "1D"),
    "3M": ("3M", "1D"),
    "1Y": ("1A", "1D"),
}


def _history(client: Any, span: str) -> PortfolioHistory:
    if span not in HISTORY_SPANS:
        raise ValueError(f"unknown span {span!r}: use one of {', '.join(HISTORY_SPANS)}")
    period, timeframe = HISTORY_SPANS[span]
    try:
        raw = client.get_portfolio_history(
            GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
        )
    except Exception as exc:
        raise BrokerError(
            f"could not fetch portfolio history: {type(exc).__name__}: {exc}"
        ) from exc
    points = tuple(
        PortfolioPoint(at=datetime.fromtimestamp(int(ts), UTC), equity=_to_decimal(equity))
        for ts, equity in zip(raw.timestamp or [], raw.equity or [], strict=False)
        if equity is not None
    )
    base = None if raw.base_value is None else _to_decimal(raw.base_value)
    return PortfolioHistory(span=span, base_value=base, points=points)
