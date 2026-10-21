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
    # Extra analysis-and-entry slots during the session (ADR 0015). How many run is capped
    # by risk.max_trades_per_day, so live (one trade a day) never uses them.
    extra_analyses: tuple[time, ...] = ()
    # Market watch (ADR 0016): a pre-market scan before the pre-flight, and an after-hours
    # scan followed by the next-day plan once the session is over.
    premarket_watch: time = time(8, 0)
    post_market: time = time(20, 30)  # after the post-market ends at 20:00 (ADR 0023)

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
        if not self.premarket_watch < self.data_collection:
            raise ValueError("premarket_watch must come before data_collection")
        if self.post_market < self.report:
            raise ValueError("post_market must not come before report")
        for extra in self.extra_analyses:
            if not self.trade_window_start <= extra < self.trade_window_end:
                raise ValueError(
                    f"extra analysis at {extra:%H:%M} must be inside the trade window "
                    f"({self.trade_window_start:%H:%M}-{self.trade_window_end:%H:%M})"
                )
        if list(self.extra_analyses) != sorted(set(self.extra_analyses)):
            raise ValueError("extra analysis times must be unique and in order")
        return self


class RiskConfig(_Strict):
    max_trades_per_day: int = Field(default=1, ge=0, le=10)
    # Positions held at the same time. Live is limited to 1 (ADR 0015).
    max_open_positions: int = Field(default=1, ge=1, le=5)
    max_position_usd: Decimal = Field(default=Decimal("50"), gt=0)
    stop_loss_pct: Decimal = Field(default=Decimal("5"), gt=0, lt=100)
    take_profit_pct: Decimal = Field(default=Decimal("10"), gt=0)
    account_floor_usd: Decimal = Field(default=Decimal("70"), ge=0)
    min_price_usd: Decimal = Field(default=Decimal("5"), ge=0)
    min_avg_daily_volume: int = Field(default=1_000_000, ge=0)
    allowed_exchanges: tuple[str, ...] = ("NYSE", "NASDAQ")
    max_price_deviation_pct: Decimal = Field(default=Decimal("2"), gt=0)
    max_data_age_seconds: int = Field(default=300, gt=0)
    # Bid/ask spread as % of mid; wide spreads make intraday round trips expensive.
    max_spread_pct: Decimal = Field(default=Decimal("0.5"), gt=0)
    # Exclude funds, ETFs/ETNs, and leveraged or inverse products.
    exclude_funds: bool = True
    # Trading days averaged for the liquidity rule.
    volume_lookback_days: int = Field(default=20, ge=5, le=120)


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


class ToolsConfig(_Strict):
    """Limits for the read-only research tools agents use (MCP server and in-process)."""

    # Calls allowed per agent run or MCP session; protects the data provider from loops.
    max_calls_per_session: int = Field(default=100, gt=0, le=1000)


class AnalystConfig(_Strict):
    """Limits for one analyst run. Hitting any limit ends the run with no trade."""

    max_turns: int = Field(default=15, ge=2, le=50)
    max_tool_calls_per_turn: int = Field(default=8, ge=1, le=20)
    # Input + output tokens across the whole run (a cost ceiling).
    max_total_tokens: int = Field(default=300_000, gt=0, le=5_000_000)
    # Chances to fix an answer that fails validation or evidence checks.
    repair_attempts: int = Field(default=1, ge=0, le=3)
    # strict: "no_trade" whenever the evidence is weak or mixed (always, for live money).
    # gather_data: paper only; take the best available setup unless there is a clear reason
    # not to, so the system collects data. Risk checks, the score, and the critic still decide.
    stance: Literal["strict", "gather_data"] = "strict"


class ExecutionConfig(_Strict):
    """Order handling. Exits are managed in code because Alpaca has no OCO for fractional."""

    # How long an entry limit order may rest before the unfilled part is cancelled.
    entry_timeout_seconds: int = Field(default=120, ge=5, le=900)
    poll_interval_seconds: float = Field(default=2, gt=0, le=30)
    # An approval older than this is stale: prices may have moved since the checks ran.
    max_decision_age_seconds: int = Field(default=60, ge=5, le=600)
    # How long to wait for cancellations to settle before closing a position.
    cancel_timeout_seconds: int = Field(default=30, ge=5, le=300)


class CriticConfig(_Strict):
    """A second, independent LLM review of every buy proposal. It can only veto."""

    enabled: bool = True
    # A different model gives a more independent review. Unset: the analyst's model.
    model: str | None = None
    # Chances to fix an answer that is not valid JSON in the expected shape.
    repair_attempts: int = Field(default=1, ge=0, le=2)
    # Upper bound on the evidence text shown to the critic.
    max_evidence_chars: int = Field(default=60_000, ge=5_000, le=200_000)


class EvaluationConfig(_Strict):
    """The go/no-go bar for live trading, fixed in advance (ADR 0011). Every criterion must
    pass; one that cannot be evaluated (missing data) counts as a fail."""

    min_cycle_days: int = Field(default=40, ge=1)
    min_closed_trades: int = Field(default=20, ge=1)
    min_win_rate_pct: Decimal = Field(default=Decimal("45"), ge=0, le=100)
    # Mean per-trade return minus SPY's same-day open-to-close move must be above this.
    min_mean_excess_pct: Decimal = Decimal("0")
    # ...and the lower end of its 90% bootstrap interval must be above this.
    min_excess_ci_lower_pct: Decimal = Decimal("-0.5")
    max_drawdown_usd: Decimal = Field(default=Decimal("15"), gt=0)
    max_error_day_pct: Decimal = Field(default=Decimal("5"), ge=0, le=100)
    # Needs costs.llm_prices for every model used. None removes this criterion.
    max_llm_cost_per_day_usd: Decimal | None = Field(default=Decimal("1.00"), gt=0)
    bootstrap_samples: int = Field(default=5000, ge=500, le=100_000)


class ScoreWeights(_Strict):
    """Relative weight of each part of the trade score. Only the ratios matter."""

    reward_risk: Decimal = Field(default=Decimal(20), ge=0)
    volatility: Decimal = Field(default=Decimal(20), ge=0)
    trend: Decimal = Field(default=Decimal(25), ge=0)
    extension: Decimal = Field(default=Decimal(15), ge=0)
    confidence: Decimal = Field(default=Decimal(10), ge=0)
    history: Decimal = Field(default=Decimal(10), ge=0)
    technicals: Decimal = Field(default=Decimal(20), ge=0)

    @model_validator(mode="after")
    def _not_all_zero(self) -> ScoreWeights:
        if not any(v > 0 for v in self.model_dump().values()):
            raise ValueError("the trade score needs at least one weight above zero")
        return self


class ScoringConfig(_Strict):
    """The parametric trade score (ADR 0015): every buy must earn a minimum score from fixed
    rules, on top of the risk checks. Hard fails reject a trade whatever its score."""

    enabled: bool = True
    min_score: Decimal = Field(default=Decimal(60), ge=0, le=100)
    # Target vs stop after paying the spread on the way in and out.
    min_reward_risk: Decimal = Field(default=Decimal("1.5"), gt=0)
    # The stop must sit at least this many average daily ranges (ATR) below the entry.
    min_stop_atr: Decimal = Field(default=Decimal("0.75"), gt=0)
    # Reject buying a price this many ATRs above its 20-day average.
    max_extension_atr: Decimal = Field(default=Decimal(3), gt=1)
    # Past closed trades needed before history counts; until then it is neutral.
    min_history_trades: int = Field(default=10, ge=1, le=500)
    history_trades: int = Field(default=50, ge=1, le=1000)  # most recent trades used
    weights: ScoreWeights = ScoreWeights()


class WatchConfig(_Strict):
    """Pre-market and after-hours watch, and the next-day plan (ADR 0016)."""

    enabled: bool = True
    # A move this big (either way) since the last regular close raises an alert.
    gap_alert_pct: Decimal = Field(default=Decimal(3), gt=0, le=50)
    movers: int = Field(default=10, ge=0, le=50)  # top gainers and losers to include
    max_symbols: int = Field(default=25, ge=1, le=100)
    news_lookback_hours: int = Field(default=16, ge=1, le=72)
    max_plan_items: int = Field(default=12, ge=1, le=50)


class GuidebookConfig(_Strict):
    """The guidebook (ADR 0017): principles proven by results, playbooks, and versions."""

    enabled: bool = True
    # Closed trades needed on each side (with the lesson in context, and without) before
    # a lesson is promoted to a principle or retired for its results.
    min_trades: int = Field(default=10, ge=1, le=500)
    promote_lift_pct: Decimal = Field(default=Decimal("0.5"), ge=0)
    retire_lift_pct: Decimal = Field(default=Decimal("0.5"), ge=0)
    # Trades a playbook needs before it is judged; below this mean it is retired.
    playbook_min_trades: int = Field(default=5, ge=1, le=500)
    playbook_retire_below_pct: Decimal = Decimal("-1.0")
    # Active lessons in one category that prompt the writer to draft a playbook.
    min_lessons_for_playbook: int = Field(default=3, ge=2, le=20)
    max_playbooks: int = Field(default=8, ge=1, le=30)
    write_playbooks: bool = True
    history_days: int = Field(default=365, ge=30, le=3650)


class LearningConfig(_Strict):
    """Self-learning (ADR 0013): a daily review turns results into advisory lessons."""

    enabled: bool = True
    max_new_per_day: int = Field(default=2, ge=0, le=3)
    # The analyst sees at most this many; older ones are retired when new ones arrive.
    max_active: int = Field(default=15, ge=1, le=50)
    # Markets change: a lesson expires unless re-learned.
    lesson_ttl_days: int = Field(default=60, ge=5, le=365)
    repair_attempts: int = Field(default=1, ge=0, le=2)


class ModelPrice(_Strict):
    """USD per million tokens. Use your provider's current price list."""

    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class PaidService(_Strict):
    """A subscription the system depends on (a data plan, hosting, ...)."""

    name: str = Field(min_length=1, max_length=80)
    monthly_usd: Decimal = Field(ge=0)
    note: str = ""


class CostsConfig(_Strict):
    """What running the system costs (ADR 0014). Nothing is ever guessed: a model with no
    price here shows tokens only, and totals that include it are shown as unknown."""

    # Model id -> price. Dated ids ("gpt-6.1-sol-2026-09-01") match their base id.
    llm_prices: dict[str, ModelPrice] = Field(default_factory=dict)
    services: tuple[PaidService, ...] = (
        PaidService(
            name="Alpaca paper trading and market data (free plan)", monthly_usd=Decimal(0)
        ),
    )
    # New LLM calls are refused once today's priced spend reaches this. None: no limit.
    daily_llm_budget_usd: Decimal | None = Field(default=Decimal("2.00"), gt=0)


class LiveConfig(_Strict):
    """Extra gates for real money (ADR 0012). Checked by `trading-agent live-check`."""

    # The paper evaluation that said GO must be at most this old.
    max_evaluation_age_days: int = Field(default=7, ge=1, le=30)
    # The live account must be small: this system is not built to manage more.
    max_account_equity_usd: Decimal = Field(default=Decimal("150"), gt=0, le=10_000)


class OrchestrationConfig(_Strict):
    """The daily cycle. Step times come from `schedule`; these tune the loop."""

    # How often open positions are managed during the session.
    manage_interval_seconds: int = Field(default=60, ge=10, le=600)
    # On early-close days, force-close this long before the market closes.
    early_close_buffer_minutes: int = Field(default=10, ge=5, le=60)
    # After the force-close, wait this long before checking that the account is flat.
    close_settle_seconds: int = Field(default=30, ge=5, le=300)
    # Ask Windows not to sleep while the cycle runs (ADR 0021). Closing a laptop lid can
    # still sleep it: that is a power setting this program does not change.
    keep_awake: bool = True
    # Restart the cycle after an unexpected crash; give up after this many in an hour.
    max_crashes_per_hour: int = Field(default=5, ge=1, le=20)
    # Only one `trading-agent run` at a time: it holds an OS lock on this file.
    lock_file: str = "data/cycle.lock"


AlertKind = Literal["start", "trade", "exit", "failure", "halt", "report", "crash"]
ALERT_KINDS: tuple[AlertKind, ...] = (
    "start",
    "trade",
    "exit",
    "failure",
    "halt",
    "report",
    "crash",
)


class JournalSourcesConfig(_Strict):
    """Where the journal looks for the "why" (ADR 0023). All free; a paid one can be added."""

    company_news: bool = True  # Alpaca / Benzinga
    federal_reserve: bool = True  # press releases (RSS)
    bls: bool = True  # Bureau of Labor Statistics latest numbers (RSS)
    federal_register: bool = True  # presidential documents and significant rules


class JournalConfig(_Strict):
    """The all-day market journal and the day notebook (ADR 0023). It only watches and
    records, from the start of pre-market to the end of post-market; it never trades."""

    enabled: bool = True
    directory: str = "journal"
    pre_market_start: time = time(4, 0)  # market time
    post_market_hours: int = Field(default=4, ge=1, le=6)  # after the close: 20:00 normally
    snapshot_minutes: int = Field(default=30, ge=5, le=120)
    market_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    ai_notes: bool = True  # a short AI note after each session, and the AI notebook
    notebook_delay_minutes: int = Field(default=20, ge=0, le=120)
    max_sources_in_prompt: int = Field(default=60, ge=5, le=200)
    sources: JournalSourcesConfig = JournalSourcesConfig()


class ChatConfig(_Strict):
    """Chatting with the bot over Telegram (ADR 0022). Only the chat in TELEGRAM_CHAT_ID is
    answered. Common questions are answered free from the records; others use the AI."""

    enabled: bool = True
    ai_answers: bool = True
    # AI answers allowed per day (market time); free answers are unlimited.
    max_ai_answers_per_day: int = Field(default=30, ge=0, le=500)
    # Messages older than this (sent while the bot was off) are skipped, not answered.
    max_message_age_hours: int = Field(default=12, ge=1, le=72)
    # Where "pause" is remembered; it ends by itself at midnight (market time).
    pause_file: str = "data/pause.json"


class AlertsConfig(_Strict):
    """Phone alerts over Telegram (ADR 0021). Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    (secrets); without them alerts are silently off."""

    enabled: bool = True
    events: tuple[AlertKind, ...] = ALERT_KINDS
    # The same message is not repeated within this many minutes.
    dedupe_minutes: int = Field(default=60, ge=0, le=1440)
    # A flood is cut off after this many alerts in an hour.
    max_per_hour: int = Field(default=20, ge=1, le=200)


class ReportingConfig(_Strict):
    """The daily report. Delivery to a webhook needs REPORT_WEBHOOK_URL (a secret)."""

    benchmark_symbol: str = "SPY"
    output_dir: str = "reports"
    # Calendar days of history for the performance-to-date section.
    history_days: int = Field(default=365, ge=7, le=3650)


class StorageConfig(_Strict):
    """The audit trail. The database URL is a secret (DATABASE_URL), not config."""

    # True: no audit record means no trade (and a missing database is an error).
    # Must be true for live trading; optional for paper so local use needs no database.
    required: bool = False


class AppConfig(_Strict):
    environment: Environment = Environment.PAPER
    kill_switch: bool = False
    schedule: ScheduleConfig = ScheduleConfig()
    risk: RiskConfig = RiskConfig()
    llm: LLMConfig = LLMConfig()
    broker: BrokerConfig = BrokerConfig()
    market_data: MarketDataConfig = MarketDataConfig()
    tools: ToolsConfig = ToolsConfig()
    analyst: AnalystConfig = AnalystConfig()
    critic: CriticConfig = CriticConfig()
    execution: ExecutionConfig = ExecutionConfig()
    storage: StorageConfig = StorageConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    reporting: ReportingConfig = ReportingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    live: LiveConfig = LiveConfig()
    learning: LearningConfig = LearningConfig()
    scoring: ScoringConfig = ScoringConfig()
    watch: WatchConfig = WatchConfig()
    guidebook: GuidebookConfig = GuidebookConfig()
    costs: CostsConfig = CostsConfig()
    alerts: AlertsConfig = AlertsConfig()
    chat: ChatConfig = ChatConfig()
    journal: JournalConfig = JournalConfig()

    @model_validator(mode="after")
    def _environment_matches_broker(self) -> AppConfig:
        # Guard against the most dangerous misconfiguration: "paper" config pointed at real money,
        # or "live" config that silently trades on the paper account.
        if self.environment is Environment.PAPER and not self.broker.paper:
            raise ValueError("environment 'paper' requires broker.paper = true")
        if self.environment is Environment.LIVE and self.broker.paper:
            raise ValueError("environment 'live' requires broker.paper = false")
        if self.environment is Environment.LIVE and not self.storage.required:
            raise ValueError("environment 'live' requires storage.required = true")
        if self.environment is Environment.LIVE and (
            self.risk.max_trades_per_day > 1 or self.risk.max_open_positions > 1
        ):
            raise ValueError(
                "live trading allows one trade and one position per day (ADR 0015); "
                "more are for paper data gathering only"
            )
        if self.environment is Environment.LIVE and not self.critic.enabled:
            raise ValueError("live trading requires the critic (critic.enabled: true)")
        if self.environment is Environment.LIVE and self.analyst.stance != "strict":
            raise ValueError(
                "live trading requires the strict analyst (analyst.stance: strict); "
                "gather_data is for paper only"
            )
        return self

    @property
    def is_live(self) -> bool:
        return self.environment is Environment.LIVE
