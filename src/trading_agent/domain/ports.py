"""Ports: the interfaces the core depends on (hexagonal / ports-and-adapters architecture).

Business logic talks only to these protocols. Concrete adapters live in `execution/`,
`llm/`, `data/` and can be swapped (another broker, another LLM vendor, a fake in tests)
without touching the core.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from trading_agent.domain.models import (
    AccountSnapshot,
    Asset,
    Bar,
    LLMRequest,
    LLMResponse,
    MarketClock,
    NewsArticle,
    Quote,
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


@runtime_checkable
class NewsPort(Protocol):
    def get_news(
        self, symbols: Sequence[str], *, since: datetime, limit: int
    ) -> tuple[NewsArticle, ...]:
        """Articles newer than `since`, newest first. An empty `symbols` means all news."""
        ...
