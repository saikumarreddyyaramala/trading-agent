"""Typed, validated application configuration.

Every value the system depends on is declared here with a type and a sane default.
Unknown keys are rejected, so a typo in YAML fails loudly at startup instead of
silently using a default.
"""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Environment(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScheduleConfig(_Strict):
    timezone: str = "America/New_York"
    data_collection: time = time(8, 30)
    analysis: time = time(8, 45)
    trade_window_start: time = time(9, 45)
    trade_window_end: time = time(15, 30)
    force_close: time = time(15, 50)
    report: time = time(16, 15)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def _ordered(self) -> ScheduleConfig:
        steps = [
            self.data_collection,
            self.analysis,
            self.trade_window_start,
            self.trade_window_end,
            self.force_close,
            self.report,
        ]
        if steps != sorted(steps):
            raise ValueError("schedule times must be in chronological order")
        return self


class RiskConfig(_Strict):
    max_trades_per_day: int = Field(default=1, ge=0, le=10)
    max_position_usd: Decimal = Field(default=Decimal("50"), gt=0)
    stop_loss_pct: Decimal = Field(default=Decimal("5"), gt=0, lt=100)
    take_profit_pct: Decimal = Field(default=Decimal("10"), gt=0)
    account_floor_usd: Decimal = Field(default=Decimal("70"), ge=0)
    min_price_usd: Decimal = Field(default=Decimal("5"), ge=0)
    min_avg_daily_volume: int = Field(default=1_000_000, ge=0)
    allowed_exchanges: tuple[str, ...] = ("NYSE", "NASDAQ")
    max_price_deviation_pct: Decimal = Field(default=Decimal("2"), gt=0)
    max_data_age_seconds: int = Field(default=300, gt=0)


class LLMConfig(_Strict):
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-6.1-sol"
    fallback_model: str | None = None
    # OpenAI reasoning models only; ignored by other providers.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = (
        None
    )
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)


class BrokerConfig(_Strict):
    provider: Literal["alpaca"] = "alpaca"
    paper: bool = True


class MarketDataConfig(_Strict):
    provider: Literal["alpaca"] = "alpaca"
    # Real-time quotes: "iex" is free; "sip" needs a paid subscription.
    quote_feed: Literal["iex", "sip", "delayed_sip"] = "iex"
    # Daily bars: "sip" (all US exchanges) is free when older than 15 minutes; "iex" volume
    # covers only one exchange and would understate liquidity.
    bars_feed: Literal["iex", "sip"] = "sip"
    daily_bars: int = Field(default=30, ge=2, le=250)
    news_lookback_hours: int = Field(default=24, gt=0, le=168)
    max_news_articles: int = Field(default=50, gt=0, le=200)
    news_per_symbol: int = Field(default=5, gt=0, le=50)


class AppConfig(_Strict):
    environment: Environment = Environment.PAPER
    kill_switch: bool = False
    schedule: ScheduleConfig = ScheduleConfig()
    risk: RiskConfig = RiskConfig()
    llm: LLMConfig = LLMConfig()
    broker: BrokerConfig = BrokerConfig()
    market_data: MarketDataConfig = MarketDataConfig()

    @model_validator(mode="after")
    def _environment_matches_broker(self) -> AppConfig:
        # Guard against the most dangerous misconfiguration: "paper" config pointed at real money,
        # or "live" config that silently trades on the paper account.
        if self.environment is Environment.PAPER and not self.broker.paper:
            raise ValueError("environment 'paper' requires broker.paper = true")
        if self.environment is Environment.LIVE and self.broker.paper:
            raise ValueError("environment 'live' requires broker.paper = false")
        return self

    @property
    def is_live(self) -> bool:
        return self.environment is Environment.LIVE
