"""Core business types shared across the system.

These are plain, immutable Pydantic models with no dependency on any vendor SDK,
so every adapter (Alpaca, Anthropic, ...) translates into and out of them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------- Broker ----------


class AccountSnapshot(_Frozen):
    account_number_masked: str
    status: str
    currency: str
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    is_paper: bool


class MarketClock(_Frozen):
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


# ---------- Market data ----------


class Asset(_Frozen):
    symbol: str
    name: str | None
    exchange: str
    status: str
    tradable: bool
    fractionable: bool


class Quote(_Frozen):
    """Latest top-of-book (best bid and ask) for one symbol."""

    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: Decimal
    ask_price: Decimal
    ask_size: Decimal

    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_pct(self) -> Decimal:
        """Bid/ask spread as a percentage of the mid price (0 when the quote is empty)."""
        mid = self.mid_price
        return (self.ask_price - self.bid_price) / mid * 100 if mid > 0 else Decimal("0")

    def age_seconds(self, now: datetime) -> float:
        return (now - self.timestamp).total_seconds()


class Bar(_Frozen):
    """One OHLCV candle. Daily bars are split- and dividend-adjusted."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None


class ActiveStock(_Frozen):
    """One entry of the most-active screener (today's cumulative activity)."""

    symbol: str
    volume: int
    trade_count: int


class NewsArticle(_Frozen):
    """A news item. Headline and summary are untrusted third-party text."""

    id: str
    headline: str
    summary: str
    source: str
    url: str | None
    symbols: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


class SymbolData(_Frozen):
    """Everything collected for one symbol in one run.

    `quote` is None only when fresh quotes were not required (for example before the open)
    and no usable quote was available.
    """

    asset: Asset
    quote: Quote | None
    daily_bars: tuple[Bar, ...]
    news: tuple[NewsArticle, ...]

    @property
    def symbol(self) -> str:
        return self.asset.symbol

    @property
    def last_close(self) -> Decimal:
        return self.daily_bars[-1].close

    @property
    def avg_daily_volume(self) -> int:
        return sum(bar.volume for bar in self.daily_bars) // len(self.daily_bars)

    @property
    def change_pct(self) -> Decimal:
        """Close-to-close change across the collected bars, in percent."""
        first = self.daily_bars[0].close
        return (self.last_close - first) / first * 100 if first > 0 else Decimal("0")


class MarketDataSnapshot(_Frozen):
    """Result of one collection run: usable symbols plus the ones rejected and why."""

    collected_at: datetime
    symbols: dict[str, SymbolData]
    rejected: dict[str, str]


# ---------- LLM ----------


class ChatMessage(_Frozen):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(_Frozen):
    system: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int | None = None
    # Free-form labels (run_id, agent name, ...) carried into logs and traces.
    tags: dict[str, str] = Field(default_factory=dict)


class LLMUsage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(_Frozen):
    text: str
    provider: str
    model: str
    stop_reason: str | None
    usage: LLMUsage
    latency_ms: float
