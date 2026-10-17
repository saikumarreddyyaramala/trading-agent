"""Ports: the interfaces the core depends on (hexagonal / ports-and-adapters architecture).

Business logic talks only to these protocols. Concrete adapters live in `execution/`,
`llm/`, `data/` and can be swapped (another broker, another LLM vendor, a fake in tests)
without touching the core.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from trading_agent.domain.models import (
    AccountSnapshot,
    ActiveStock,
    Asset,
    Bar,
    BrokerOrder,
    LLMRequest,
    LLMResponse,
    MarketClock,
    Mover,
    NewsArticle,
    Position,
    Quote,
    Snapshot,
)


@runtime_checkable
class BrokerPort(Protocol):
    """Read-only broker access for now.

    Order placement is added in Phase 5, behind the risk engine.
    """

    @property
    def is_paper(self) -> bool: ...

    def get_account(self) -> AccountSnapshot: ...

    def get_clock(self) -> MarketClock: ...

    def get_positions(self) -> tuple[Position, ...]: ...

    def count_buy_orders_since(self, since: datetime) -> int:
        """Buy orders submitted at or after `since`, in any status (filled, open, canceled)."""
        ...


@runtime_checkable
class LLMProviderPort(Protocol):
    """One LLM vendor. The gateway composes providers and adds fallback, limits, and logging."""

    @property
    def name(self) -> str: ...

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse: ...


@runtime_checkable
class MarketDataPort(Protocol):
    """Read-only prices and reference data. Symbols missing from a response had no data."""

    def get_asset(self, symbol: str) -> Asset: ...

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]: ...

    def get_daily_bars(self, symbols: Sequence[str], *, days: int) -> dict[str, tuple[Bar, ...]]:
        """The most recent `days` daily bars per symbol, oldest first."""
        ...

    def get_snapshots(self, symbols: Sequence[str]) -> dict[str, Snapshot]:
        """Latest trade (extended hours included) and the last two daily sessions."""
        ...

    def get_movers(self, limit: int) -> tuple[Mover, ...]:
        """Up to `limit` top gainers and `limit` top losers by percent change."""
        ...

    def get_most_active(self, limit: int) -> tuple[ActiveStock, ...]:
        """Today's most traded stocks by share volume, most active first."""
        ...


@runtime_checkable
class NewsPort(Protocol):
    def get_news(
        self, symbols: Sequence[str], *, since: datetime, limit: int
    ) -> tuple[NewsArticle, ...]:
        """Articles newer than `since`, newest first. An empty `symbols` means all news."""
        ...


@runtime_checkable
class OrderExecutionPort(Protocol):
    """The only way to change broker state. Deliberately separate from `BrokerPort`:
    research tools and agents are built from read-only ports and never receive this one.

    Every order carries a caller-chosen `client_order_id`, which the broker enforces as
    unique, so retrying a submission can never create a second order.
    """

    def submit_limit_buy(
        self, symbol: str, quantity: Decimal, limit_price: Decimal, client_order_id: str
    ) -> BrokerOrder: ...

    def submit_stop_sell(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> BrokerOrder: ...

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        """The order with this client id, or None if it was never submitted."""
        ...

    def get_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]: ...

    def get_orders_since(self, since: datetime) -> tuple[BrokerOrder, ...]:
        """Every order submitted at or after `since`, in any status (for reconciliation)."""
        ...

    def cancel_order(self, order_id: str) -> None: ...

    def close_position(self, symbol: str) -> BrokerOrder:
        """Sell the whole position at market."""
        ...
