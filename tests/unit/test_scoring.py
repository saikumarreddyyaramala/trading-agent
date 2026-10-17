"""The trade score (ADR 0015): a deterministic, parametric pre-trade evaluation.

Written first, as failing tests: exact component maths, every hard-fail rule, and
properties over random markets (bounded, monotonic, deterministic).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import NOW
from trading_agent.domain.models import Bar
from trading_agent.risk.scoring import atr_pct, score_trade, sma
from trading_agent.settings.models import AppConfig, RiskConfig, ScoringConfig

D = Decimal
RISK = RiskConfig()  # stop 5%, target 10%
SCORING = ScoringConfig()


ONE = D(1)
START = D("97.1")
STEP = D("0.1")


def _bars(closes: list[Decimal], spread: Decimal = ONE) -> list[Bar]:
    """Daily bars, oldest first: high/low are `spread` above and below each close."""
    start = NOW - timedelta(days=len(closes))
    return [
        Bar(
            symbol="AAPL",
            timestamp=start + timedelta(days=i),
            open=close,
            high=close + spread,
            low=close - spread,
            close=close,
            volume=2_000_000,
        )
        for i, close in enumerate(closes)
    ]


def _uptrend(n: int = 30, start: Decimal = START, step: Decimal = STEP) -> list[Bar]:
    return _bars([start + step * i for i in range(n)])


def _score(
    *,
    price: str = "100.10",
    spread: str = "0.10",
    bars: list[Bar] | None = None,
    confidence: str = "medium",
    history: list[Decimal] | None = None,
    risk: RiskConfig = RISK,
    scoring: ScoringConfig = SCORING,
):  # type: ignore[no-untyped-def]
    return score_trade(
        price=D(price),
        spread_pct=D(spread),
        bars=_uptrend() if bars is None else bars,
        confidence=confidence,  # type: ignore[arg-type]
        history=history if history is not None else [],
        risk=risk,
        scoring=scoring,
    )


def _component(score, name: str):  # type: ignore[no-untyped-def]
    return next(c for c in score.components if c.name == name)


# ---------- Indicators ----------


def test_atr_uses_the_true_range_with_the_previous_close() -> None:
    bars = _bars([D(100), D(110)], spread=D(1))  # gap up: TR = 111 - 100 = 11
    assert atr_pct(bars, D(110), period=1) == D("10.00")
    assert atr_pct(bars[:1], D(110), period=1) is None  # needs a previous close


def test_sma_needs_enough_bars() -> None:
    bars = _bars([D(1), D(2), D(3)])
    assert sma(bars, 3) == D(2)
    assert sma(bars, 4) is None


# ---------- A clean setup ----------


def test_clean_uptrend_scores_high_and_passes() -> None:
    score = _score()
    assert score.passed
    assert score.hard_fails == ()
    assert [c.name for c in score.components] == [
        "Reward/risk",
        "Stop vs volatility",
        "Trend",
        "Extension",
        "Confidence",
        "History",
        "Technicals",
    ]
    rr = _component(score, "Reward/risk")
    assert score.reward_risk == D("1.88")  # (10 - 2x0.1) / (5 + 2x0.1)
    assert rr.value == D("0.88")
    assert score.atr_pct == D("2.00")  # TR 2 on a ~100 price
    assert score.stop_atr == D("2.50")
    assert _component(score, "Stop vs volatility").value == D("1.00")
    assert _component(score, "Trend").value == D("1.00")
    assert _component(score, "Extension").value == D("1.00")
    assert _component(score, "Confidence").value == D("0.60")
    assert _component(score, "History").value == D("0.50")
    # A straight line up: RSI 100 is overbought, the other three checks pass (30 bars are
    # too few for MACD).
    assert _component(score, "Technicals").value == D("0.75")
    # (20x0.88 + 20 + 25 + 15 + 10x0.6 + 10x0.5 + 20x0.75) / 120 = 86.3
    assert score.total == D("86.3")
    assert score.expected_value_pct is None
    assert score.summary == "86.3/100 (needs 60)"


# ---------- Hard fails ----------


def test_thin_reward_after_costs_is_a_hard_fail() -> None:
    risk = RISK.model_copy(update={"take_profit_pct": D(6)})
    score = _score(risk=risk, spread="0.40")  # (6 - 0.8) / (5 + 0.8) = 0.90
    assert not score.passed
    assert any("reward/risk 0.90 after spread costs is below 1.5" in f for f in score.hard_fails)


def test_a_stop_inside_normal_noise_is_a_hard_fail() -> None:
    wild = _bars([D(100)] * 30, spread=D(4))  # ATR 8%: a 5% stop is 0.625 ATR away
    score = _score(bars=wild, price="100")
    assert not score.passed
    assert any("only 0.63 daily ranges (ATR) away" in f for f in score.hard_fails)


def test_an_overextended_price_is_a_hard_fail() -> None:
    flat = _bars([D(90)] * 30, spread=D(1))  # ATR 2, SMA 90: price 100 is 5 ATR above it
    score = _score(bars=flat, price="100")
    assert not score.passed
    assert any("5.00 daily ranges above its 20-day average" in f for f in score.hard_fails)


def test_too_little_price_history_is_a_hard_fail() -> None:
    score = _score(bars=_uptrend(n=12))
    assert not score.passed
    assert any("12 daily bars; at least 21 are needed" in f for f in score.hard_fails)


def test_a_high_total_cannot_override_a_hard_fail() -> None:
    wild = _bars([D(100) + i for i in range(30)], spread=D(5))  # strong trend, huge ATR
    score = _score(bars=wild, price="129", confidence="high")
    assert score.hard_fails
    assert not score.passed


def test_price_must_be_positive() -> None:
    with pytest.raises(ValueError, match="price must be positive"):
        _score(price="0")


# ---------- Soft components ----------


def test_downtrend_lowers_the_score() -> None:
    down = _bars([D("103") - D("0.1") * i for i in range(30)])
    score = _score(bars=down, price="100.10")
    assert _component(score, "Trend").value == D("0.00")
    assert score.total < _score().total


def test_low_confidence_and_mixed_signals_can_fail_on_score_alone() -> None:
    down = _bars([D("103") - D("0.1") * i for i in range(30)])
    scoring = SCORING.model_copy(update={"min_score": D(75)})
    score = _score(bars=down, confidence="low", scoring=scoring)
    assert score.hard_fails == ()
    # (17.6 + 20 + 0 + 15 + 2 + 5 + 20x0.13) / 120 = 51.8 < 75
    assert not score.passed
    assert score.total == D("51.8")


def test_history_counts_only_once_there_is_enough() -> None:
    few = _score(history=[D("-5")] * 9)
    assert _component(few, "History").value == D("0.50")
    assert "9 of 10 closed trades" in _component(few, "History").detail
    assert few.expected_value_pct is None

    losing = _score(history=[D("-5")] * 6 + [D("2")] * 4)  # mean -2.2%
    assert losing.expected_value_pct == D("-2.20")
    assert _component(losing, "History").value == D("0.28")  # 0.5 - 2.2 / 10
    assert losing.history_trades == 10

    winning = _score(history=[D("3")] * 10)
    assert _component(winning, "History").value == D("0.80")
    assert winning.total > losing.total


def test_unavailable_history_is_neutral_and_says_so() -> None:
    score = score_trade(
        price=D("100.10"),
        spread_pct=D("0.10"),
        bars=_uptrend(),
        confidence="medium",
        history=None,
        risk=RISK,
        scoring=SCORING,
    )
    assert _component(score, "History").value == D("0.50")
    assert _component(score, "History").detail == "history unavailable: neutral"


def test_weights_come_from_config() -> None:
    only_trend = ScoringConfig.model_validate(
        {
            "weights": {
                "reward_risk": 0,
                "volatility": 0,
                "trend": 1,
                "extension": 0,
                "confidence": 0,
                "history": 0,
                "technicals": 0,
            }
        }
    )
    assert _score(scoring=only_trend).total == D("100.0")


def test_scoring_config_is_validated() -> None:
    with pytest.raises(ValueError, match="scoring"):
        AppConfig.model_validate({"scoring": {"min_score": 101}})
    with pytest.raises(ValueError, match="at least one weight"):
        AppConfig.model_validate(
            {
                "scoring": {
                    "weights": {
                        "reward_risk": 0,
                        "volatility": 0,
                        "trend": 0,
                        "extension": 0,
                        "confidence": 0,
                        "history": 0,
                        "technicals": 0,
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="scoring"):
        AppConfig.model_validate({"scoring": {"min_stop_atr": 0}})


# ---------- Technicals (ADR 0020) ----------


def _healthy(n: int = 60) -> list[Bar]:
    """Two steps up, one step back: a steady trend that is not overbought."""
    closes, price = [], D(80)
    for i in range(n):
        price += D("-0.2") if i % 3 == 2 else D("0.2")
        closes.append(price)
    return _bars(closes, spread=D("0.8"))


def test_a_healthy_trend_earns_full_technical_marks() -> None:
    bars = _healthy()
    score = _score(bars=bars, price=str(bars[-1].close))
    technicals = _component(score, "Technicals")
    assert technicals.value == D("1.00")
    assert "RSI" in technicals.detail
    assert "5 of 5" in technicals.detail


def test_a_straight_line_rally_is_marked_down_as_overbought() -> None:
    rally = _uptrend(n=60)  # every day up: RSI 100
    score = _score(bars=rally, price=str(rally[-1].close))
    technicals = _component(score, "Technicals")
    assert "overbought" in technicals.detail
    healthy = _healthy()
    assert (
        technicals.value
        < _component(_score(bars=healthy, price=str(healthy[-1].close)), "Technicals").value
    )


def test_a_downtrend_scores_poorly_on_technicals() -> None:
    down = _bars([D(110) - D("0.2") * i + (D("0.3") if i % 3 == 2 else 0) for i in range(60)])
    score = _score(bars=down, price=str(down[-1].close))
    assert _component(score, "Technicals").value <= D("0.25")


def test_technicals_without_enough_history_are_neutral() -> None:
    score = _score(bars=_uptrend(n=12))
    technicals = _component(score, "Technicals")
    assert technicals.value == D("0.50")
    assert technicals.detail == "not enough price history: neutral"


def test_technicals_use_what_the_history_allows() -> None:
    # 30 bars: too few for MACD (needs 35), enough for the other four checks.
    detail = _component(_score(), "Technicals").detail
    assert "of 4" in detail
    assert "MACD" not in detail


# ---------- Properties over random markets ----------

closes = st.lists(
    st.decimals(min_value=5, max_value=500, places=2, allow_nan=False, allow_infinity=False),
    min_size=21,
    max_size=40,
)
spreads = st.decimals(min_value=0, max_value=2, places=3, allow_nan=False, allow_infinity=False)
returns = st.lists(
    st.decimals(min_value=-10, max_value=10, places=2, allow_nan=False, allow_infinity=False),
    max_size=30,
)


@settings(max_examples=150)
@given(
    closes=closes,
    spread=spreads,
    history=returns,
    confidence=st.sampled_from(["low", "medium", "high"]),
)
def test_score_is_bounded_and_deterministic(
    closes: list[Decimal], spread: Decimal, history: list[Decimal], confidence: str
) -> None:
    bars = _bars(closes, spread=closes[-1] / 50)
    price = closes[-1]
    first = _score(
        price=str(price), spread=str(spread), bars=bars, history=history, confidence=confidence
    )
    again = _score(
        price=str(price), spread=str(spread), bars=bars, history=history, confidence=confidence
    )
    assert first == again
    assert D(0) <= first.total <= D(100)
    assert all(D(0) <= c.value <= D(1) for c in first.components)
    assert first.passed == (not first.hard_fails and first.total >= SCORING.min_score)


@settings(max_examples=100)
@given(closes=closes, low=spreads, extra=spreads)
def test_a_wider_spread_never_helps(closes: list[Decimal], low: Decimal, extra: Decimal) -> None:
    bars = _bars(closes, spread=closes[-1] / 50)
    tight = _score(price=str(closes[-1]), spread=str(low), bars=bars)
    wide = _score(price=str(closes[-1]), spread=str(low + extra), bars=bars)
    assert wide.total <= tight.total
    assert wide.reward_risk <= tight.reward_risk


@settings(max_examples=100)
@given(closes=closes)
def test_more_confidence_never_hurts(closes: list[Decimal]) -> None:
    bars = _bars(closes, spread=closes[-1] / 50)
    totals = [
        _score(price=str(closes[-1]), bars=bars, confidence=level).total
        for level in ("low", "medium", "high")
    ]
    assert totals == sorted(totals)


@settings(max_examples=100)
@given(
    history=st.lists(st.decimals(min_value=-10, max_value=10, places=2), min_size=10, max_size=30),
    lift=st.decimals(min_value=0, max_value=5, places=2),
)
def test_better_history_never_hurts(history: list[Decimal], lift: Decimal) -> None:
    worse = _score(history=history)
    better = _score(history=[r + lift for r in history])
    assert better.total >= worse.total


def test_a_market_with_no_range_cannot_be_scored() -> None:
    flat = _bars([D(100)] * 30, spread=D(0))
    score = _score(bars=flat, price="100")
    assert not score.passed
    assert any("no price range in the last 14 days" in f for f in score.hard_fails)


def test_details_are_human_sized() -> None:
    detail = _component(_score(spread="0.0999500249875062468765617191"), "Reward/risk").detail
    assert detail == "1.88 after spread (10% target, 5% stop, 0.100% spread)"


def test_a_range_too_small_to_measure_is_a_hard_fail() -> None:
    tiny = _bars([D(1000)] * 30, spread=D("0.00001"))  # ATR rounds to 0.00% of the price
    score = _score(bars=tiny, price="1000")
    assert any("ATR is zero" in f for f in score.hard_fails)
    assert score.stop_atr is None
