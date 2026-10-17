"""Core business types shared across the system.

These are plain, immutable Pydantic models with no dependency on any vendor SDK,
so every adapter (Alpaca, Anthropic, ...) translates into and out of them.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    # 1 = cash account; 2-4 = margin. Live trading requires a cash account.
    multiplier: Decimal = Decimal("1")
    # Buying power that cannot come from margin; None if the broker did not say.
    non_marginable_buying_power: Decimal | None = None
    daytrade_count: int = 0


class MarketClock(_Frozen):
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class Position(_Frozen):
    """An open position at the broker."""

    symbol: str
    quantity: Decimal
    market_value: Decimal
    avg_entry_price: Decimal


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


class Snapshot(_Frozen):
    """The latest trade and the last two daily sessions of a symbol (ADR 0016).

    Outside regular hours the latest trade is a pre-market or after-hours trade, which is
    what the market watch compares against the last regular close."""

    symbol: str
    last_price: Decimal | None
    last_trade_at: datetime | None
    session_close: Decimal | None  # the latest daily bar (may be today's, even pre-market)
    session_at: datetime | None
    previous_close: Decimal | None  # the daily bar before it
    session_volume: int | None = None


class Mover(_Frozen):
    """One of the day's biggest percentage gainers or losers."""

    symbol: str
    price: Decimal
    change_pct: Decimal
    direction: Literal["gainer", "loser"]


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


class ToolSpec(_Frozen):
    """A tool the model may call, described by a JSON schema for its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(_Frozen):
    """A model's request to run a tool. Arguments stay raw JSON: the caller validates them."""

    id: str
    name: str
    arguments_json: str


class ToolResult(_Frozen):
    call_id: str
    content: str
    is_error: bool = False


class ChatMessage(_Frozen):
    """One conversation turn.

    An assistant turn may request tools (`tool_calls`); the next user turn answers them
    (`tool_results`). Plain turns carry only `content`.
    """

    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


class LLMRequest(_Frozen):
    system: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int | None = None
    tools: tuple[ToolSpec, ...] = ()
    # Free-form labels (run_id, agent name, ...) carried into logs and traces.
    tags: dict[str, str] = Field(default_factory=dict)


class LLMUsage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMCallRecord(_Frozen):
    """One attempt at a model call, for the cost ledger (ADR 0014). Failed attempts count."""

    created_at: datetime
    provider: str
    model: str
    agent: str  # analyst | critic | reflector | ... ; "other" when untagged
    run_id: str | None
    status: Literal["ok", "error"]
    usage: LLMUsage
    latency_ms: float | None
    error: str | None = None


class LLMResponse(_Frozen):
    text: str
    provider: str
    model: str
    stop_reason: str | None
    usage: LLMUsage
    latency_ms: float
    tool_calls: tuple[ToolCall, ...] = ()


# ---------- Trade proposals ----------


class Evidence(_Frozen):
    """One claim backing a proposal, tied to the tool call that produced the data."""

    claim: str = Field(min_length=1, max_length=400)
    source_call_id: str = Field(min_length=1, max_length=100)


class TradeProposal(_Frozen):
    """The analyst's answer: one long intraday trade, or no trade.

    It is only a proposal. Sizing, limits, and the go/no-go decision belong to the risk engine.
    """

    decision: Literal["buy", "no_trade"]
    symbol: str | None = None
    thesis: str = Field(min_length=1, max_length=2000)
    reference_price: Decimal | None = Field(default=None, gt=0)
    confidence: Literal["low", "medium", "high"]
    evidence: tuple[Evidence, ...] = Field(default=(), max_length=10)
    risks: tuple[str, ...] = Field(default=(), max_length=10)
    # The playbook the analyst followed, if any (ADR 0017); unknown names are dropped.
    playbook: str | None = Field(default=None, max_length=40)

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, value: str | None) -> str | None:
        return value.strip().upper() or None if value is not None else None

    @model_validator(mode="after")
    def _buy_is_complete(self) -> TradeProposal:
        if self.decision == "buy":
            missing = [
                name
                for name, ok in (
                    ("symbol", self.symbol),
                    ("reference_price", self.reference_price),
                    ("at least 2 evidence items", len(self.evidence) >= 2),
                    ("at least 1 risk", self.risks),
                )
                if not ok
            ]
            if missing:
                raise ValueError(f"a buy proposal needs {', '.join(missing)}")
        return self


# ---------- Risk decisions ----------


class ScoreComponent(_Frozen):
    name: str
    value: Decimal = Field(ge=0, le=1)  # 0 = worst, 1 = best
    weight: Decimal = Field(ge=0)
    detail: str


class TradeScore(_Frozen):
    """The parametric pre-trade evaluation (ADR 0015). Deterministic: same inputs, same score."""

    total: Decimal = Field(ge=0, le=100)
    min_score: Decimal
    passed: bool
    components: tuple[ScoreComponent, ...]
    hard_fails: tuple[str, ...]
    reward_risk: Decimal  # (target - 2 x spread) / (stop + 2 x spread)
    atr_pct: Decimal | None  # average daily range, % of price
    stop_atr: Decimal | None  # stop distance in ATRs
    extension_atr: Decimal | None  # price above its 20-day average, in ATRs
    expected_value_pct: Decimal | None  # mean return of past closed trades, once enough
    history_trades: int

    @property
    def summary(self) -> str:
        return f"{self.total}/100 (needs {self.min_score.normalize():f})"


class RiskCheck(_Frozen):
    name: str
    passed: bool
    detail: str


class OrderPlan(_Frozen):
    """What the risk engine approved. Execution (Phase 5) must not exceed any of these values."""

    symbol: str
    side: Literal["buy"] = "buy"
    quantity: Decimal = Field(gt=0)
    fractional: bool
    notional_usd: Decimal = Field(gt=0)
    entry_reference_price: Decimal = Field(gt=0)
    limit_price: Decimal = Field(gt=0)
    stop_loss_price: Decimal = Field(gt=0)
    take_profit_price: Decimal = Field(gt=0)


class RiskDecision(_Frozen):
    """The risk engine's verdict on one proposal, with every check for the audit trail."""

    approved: bool
    symbol: str | None
    checks: tuple[RiskCheck, ...]
    plan: OrderPlan | None
    evaluated_at: datetime
    score: TradeScore | None = None  # the parametric evaluation (ADR 0015), when it ran

    @property
    def failed_checks(self) -> tuple[RiskCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @model_validator(mode="after")
    def _approval_is_consistent(self) -> RiskDecision:
        # An approval without a plan, or with any failed check, must be impossible to build.
        if self.approved and (self.plan is None or self.failed_checks or not self.checks):
            raise ValueError("an approved decision needs a plan and only passed checks")
        if not self.approved and self.plan is not None:
            raise ValueError("a rejected decision cannot carry an order plan")
        return self


# ---------- Orders ----------

# Statuses after which an order can no longer fill.
TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "replaced",
        "done_for_day",
        "stopped",
        "suspended",
    }
)


class BrokerOrder(_Frozen):
    """An order as the broker reports it, in vendor-neutral form."""

    id: str
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: str
    status: str
    quantity: Decimal
    filled_quantity: Decimal
    filled_avg_price: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    submitted_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"


# ---------- Learning ----------


class Lesson(_Frozen):
    """Something the system learned from its own results. Advisory text for the analyst:
    it can never change a rule, a limit, or a check."""

    id: int
    text: str
    category: str
    source_run_id: str | None
    created_at: datetime
    expires_at: datetime
    promoted: bool = False  # a principle: proven by results (ADR 0017)


# ---------- The next-day plan (ADR 0016) ----------


class PlanItem(_Frozen):
    """One thing to do next session, derived by fixed rules from the day's facts."""

    kind: Literal["watch", "review", "avoid", "ops", "apply"]
    symbol: str | None = None
    text: str = Field(min_length=1, max_length=500)
    priority: int = Field(ge=1, le=3)  # 1 = first thing


class DayPlan(_Frozen):
    source_day: date
    for_day: date
    created_at: datetime
    items: tuple[PlanItem, ...]
    summary: str


# ---------- Playbooks (ADR 0017) ----------


class Playbook(_Frozen):
    """A step-by-step research procedure the system wrote from its own lessons.

    Text for the analyst, never code. It is tried on paper as a candidate, becomes active
    when its trades do well, and is retired when they do not."""

    id: int | None = None
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,39}$")
    title: str = Field(min_length=3, max_length=80)
    when_to_use: str = Field(min_length=3, max_length=300)
    steps: tuple[str, ...] = Field(min_length=1, max_length=6)
    category: str = Field(max_length=16)
    status: Literal["candidate", "active", "retired"]
    source_lesson_ids: tuple[int, ...] = ()
    created_at: datetime
    retired_reason: str | None = None

    @field_validator("steps")
    @classmethod
    def _short_steps(cls, steps: tuple[str, ...]) -> tuple[str, ...]:
        if any(not step.strip() or len(step) > 200 for step in steps):
            raise ValueError("steps must be 1-200 characters each")
        return steps
