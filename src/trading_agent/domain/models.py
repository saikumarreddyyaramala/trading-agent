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
