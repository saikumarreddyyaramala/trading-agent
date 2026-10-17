"""The trade score: a deterministic, parametric evaluation of a buy before it is placed.

Six parts, each scored from 0 (worst) to 1 (best) and combined with configured weights:

    Reward/risk         target vs stop after paying the spread both ways
    Stop vs volatility  how many average daily ranges (ATR) the stop sits below the entry
    Trend               price above its 20-day average, and up over the last 5 days
    Extension           how far price has run above its 20-day average, in ATRs
    Confidence          the analyst's own confidence (a small weight: it is self-reported)
    History             the mean return of the system's own closed trades, once enough

Some conditions reject a trade whatever the total (hard fails): too little reward for the
risk after costs, a stop inside normal daily noise, a badly overextended price, and too
little price history to judge. Missing history is neutral, never a pass by default.

Nothing here predicts the future. The score keeps out trades whose structure is poor, and
learns from results through the history part (ADR 0015).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from trading_agent.domain.models import Bar, ScoreComponent, TradeScore
from trading_agent.settings.models import RiskConfig, ScoringConfig

ATR_PERIOD = 14
SMA_PERIOD = 20
MOMENTUM_DAYS = 5
MIN_BARS = SMA_PERIOD + 1
_CONFIDENCE = {"low": Decimal("0.2"), "medium": Decimal("0.6"), "high": Decimal(1)}
_HUNDREDTH = Decimal("0.01")
_ZERO, _ONE, _HALF = Decimal(0), Decimal(1), Decimal("0.5")


def _q(value: Decimal, step: Decimal = _HUNDREDTH) -> Decimal:
    return value.quantize(step, rounding=ROUND_HALF_UP)


def _clamp(value: Decimal) -> Decimal:
    return _q(min(_ONE, max(_ZERO, value)))


def atr(bars: Sequence[Bar], period: int = ATR_PERIOD) -> Decimal | None:
    """Average true range of the last `period` bars, each using the previous close."""
    if len(bars) < period + 1:
        return None
    ranges = [
        max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))
        for prev, bar in zip(bars[-period - 1 : -1], bars[-period:], strict=True)
    ]
    return sum(ranges, _ZERO) / period


def atr_pct(bars: Sequence[Bar], price: Decimal, period: int = ATR_PERIOD) -> Decimal | None:
    value = atr(bars, period)
    return None if value is None else _q(value / price * 100)


def sma(bars: Sequence[Bar], period: int = SMA_PERIOD) -> Decimal | None:
    if len(bars) < period:
        return None
    return sum((b.close for b in bars[-period:]), _ZERO) / period


def score_trade(
    *,
    price: Decimal,
    spread_pct: Decimal,
    bars: Sequence[Bar],
    confidence: Literal["low", "medium", "high"],
    history: Sequence[Decimal] | None,
    risk: RiskConfig,
    scoring: ScoringConfig,
) -> TradeScore:
    """Score a buy at `price`. `bars` are daily bars, oldest first; `history` holds the
    returns (%) of past closed trades, newest last, or None when it could not be read."""
    if price <= 0:
        raise ValueError("price must be positive")
    weights = scoring.weights
    stop, target = risk.stop_loss_pct, risk.take_profit_pct
    components: list[ScoreComponent] = []
    hard_fails: list[str] = []

    def add(name: str, value: Decimal, weight: Decimal, detail: str) -> None:
        components.append(
            ScoreComponent(name=name, value=_clamp(value), weight=weight, detail=detail)
        )

    # ---- Reward/risk after costs ----
    cost = 2 * spread_pct  # the spread is paid on the way in and on the way out
    reward_risk = _q(max(target - cost, _ZERO) / (stop + cost))
    add(
        "Reward/risk",
        reward_risk - 1,
        weights.reward_risk,
        f"{reward_risk} after spread ({target}% target, {stop}% stop, "
        f"{_q(spread_pct, Decimal('0.001'))}% spread)",
    )
    if reward_risk < scoring.min_reward_risk:
        hard_fails.append(
            f"reward/risk {reward_risk} after spread costs is below {scoring.min_reward_risk}"
        )

    # ---- Volatility, trend, and extension need enough daily bars ----
    daily_range = atr(bars)
    average = sma(bars)
    vol_pct = stop_atr = extension = None
    if len(bars) < MIN_BARS or daily_range is None or average is None:
        hard_fails.append(f"{len(bars)} daily bars; at least {MIN_BARS} are needed to score")
        for name, weight in (
            ("Stop vs volatility", weights.volatility),
            ("Trend", weights.trend),
            ("Extension", weights.extension),
        ):
            add(name, _ZERO, weight, "not enough price history")
    elif daily_range == 0:
        hard_fails.append(f"no price range in the last {ATR_PERIOD} days (ATR is zero)")
        for name, weight in (
            ("Stop vs volatility", weights.volatility),
            ("Trend", weights.trend),
            ("Extension", weights.extension),
        ):
            add(name, _ZERO, weight, "no price range to measure")
    else:
        vol_pct = _q(daily_range / price * 100)
        stop_atr = _q(stop / vol_pct) if vol_pct > 0 else None
        if stop_atr is None:  # range rounds to 0.00% of a high price: treat as no range
            hard_fails.append(f"no price range in the last {ATR_PERIOD} days (ATR is zero)")
            add("Stop vs volatility", _ZERO, weights.volatility, "no price range to measure")
        else:
            floor = scoring.min_stop_atr
            add(
                "Stop vs volatility",
                (stop_atr - floor) / floor,
                weights.volatility,
                f"stop is {stop_atr} ATRs away (ATR {vol_pct}%; target is "
                f"{_q(target / vol_pct)} ATRs away)",
            )
            if stop_atr < floor:
                hard_fails.append(
                    f"the {stop}% stop is only {stop_atr} daily ranges (ATR) away: normal "
                    f"noise would hit it (needs {floor})"
                )

        momentum = bars[-1].close - bars[-1 - MOMENTUM_DAYS].close
        above = price > average
        rising = momentum > 0
        add(
            "Trend",
            (_HALF if above else _ZERO) + (_HALF if rising else _ZERO),
            weights.trend,
            f"price {'above' if above else 'at or below'} its {SMA_PERIOD}-day average "
            f"${_q(average)}; {MOMENTUM_DAYS}-day move {'up' if rising else 'flat or down'}",
        )

        extension = _q((price - average) / daily_range)
        limit = scoring.max_extension_atr
        add(
            "Extension",
            _ONE if extension <= 1 else (limit - extension) / (limit - 1),
            weights.extension,
            f"{extension} ATRs from its {SMA_PERIOD}-day average",
        )
        if extension > limit:
            hard_fails.append(
                f"price is {extension} daily ranges above its {SMA_PERIOD}-day average "
                f"(limit {limit}): buying an overextended move"
            )

    # ---- Confidence ----
    add("Confidence", _CONFIDENCE[confidence], weights.confidence, f"analyst said {confidence}")

    # ---- History ----
    expected = None
    trades = 0 if history is None else len(history[-scoring.history_trades :])
    if history is None:
        add("History", _HALF, weights.history, "history unavailable: neutral")
    elif trades < scoring.min_history_trades:
        add(
            "History",
            _HALF,
            weights.history,
            f"{trades} of {scoring.min_history_trades} closed trades needed: neutral",
        )
    else:
        recent = history[-scoring.history_trades :]
        expected = _q(sum(recent, _ZERO) / len(recent))
        wins = sum(1 for r in recent if r > 0)
        add(
            "History",
            _HALF + expected / (2 * stop),
            weights.history,
            f"last {len(recent)} closed trades: mean {expected}%, {wins} won",
        )

    total_weight = sum((c.weight for c in components), _ZERO)
    total = _q(
        sum((c.value * c.weight for c in components), _ZERO) / total_weight * 100,
        Decimal("0.1"),
    )
    return TradeScore(
        total=total,
        min_score=scoring.min_score,
        passed=not hard_fails and total >= scoring.min_score,
        components=tuple(components),
        hard_fails=tuple(hard_fails),
        reward_risk=reward_risk,
        atr_pct=vol_pct,
        stop_atr=stop_atr,
        extension_atr=extension,
        expected_value_pct=expected,
        history_trades=trades,
    )
