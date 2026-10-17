"""The technical indicator library (ADR 0020): hand-checked values and invariants.

Every indicator is checked against values worked out by hand from its textbook definition,
plus properties that must hold for any input (bounded oscillators, ordered bands, aligned
lengths). Written before the code.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_agent.analysis import indicators as ta


def approx(value: float | None, expected: float, tol: float = 1e-6) -> bool:
    return value is not None and abs(value - expected) <= tol


# ---------- Moving averages ----------


def test_sma() -> None:
    assert ta.sma([1, 2, 3, 4, 5], 3) == [None, None, 2, 3, 4]
    assert ta.sma([1, 2], 3) == [None, None]


def test_ema_is_seeded_with_the_sma() -> None:
    # alpha = 2 / (2 + 1); seed = mean(2, 4) = 3; then 6*2/3 + 3/3 = 5; 8*2/3 + 5/3 = 7
    result = ta.ema([2, 4, 6, 8], 2)
    assert result[0] is None
    assert all(approx(a, b) for a, b in zip(result[1:], [3, 5, 7], strict=True))


def test_ema_of_a_constant_is_the_constant() -> None:
    assert all(approx(v, 7) for v in ta.ema([7.0] * 30, 10)[9:])


# ---------- Momentum ----------


def test_rsi_uses_wilder_smoothing() -> None:
    # changes +1 -1 +1 +1; first averages 0.5/0.5 -> 50; then (0.5+1)/2=0.75 vs 0.25 -> 75;
    # then (0.75+1)/2=0.875 vs 0.125 -> 87.5
    result = ta.rsi([1, 2, 1, 2, 3], 2)
    assert result[:2] == [None, None]
    assert all(approx(a, b) for a, b in zip(result[2:], [50, 75, 87.5], strict=True))


def test_rsi_extremes() -> None:
    assert approx(ta.rsi([float(i) for i in range(30)], 14)[-1], 100)
    assert approx(ta.rsi([float(30 - i) for i in range(30)], 14)[-1], 0)
    assert approx(ta.rsi([5.0] * 30, 14)[-1], 50)  # no movement at all: neutral


def test_macd_is_fast_minus_slow_with_a_signal() -> None:
    flat = ta.macd([10.0] * 60)
    assert all(approx(v, 0) for v in flat.line[-5:])
    rising = ta.macd([float(i) for i in range(1, 80)])
    assert rising.line[-1] is not None
    assert rising.line[-1] > 0
    assert rising.signal[-1] is not None
    assert approx(rising.histogram[-1], rising.line[-1] - rising.signal[-1])


def test_stochastic_and_williams_r() -> None:
    highs = [10.0, 12, 14, 16]
    lows = [8.0, 9, 10, 11]
    at_top = ta.stochastic(highs, lows, [9.0, 11, 13, 16], k_period=3, d_period=2)
    assert approx(at_top.k[-1], 100)
    at_bottom = ta.stochastic(highs, lows, [9.0, 11, 13, 9], k_period=3, d_period=2)
    assert approx(at_bottom.k[-1], 0)
    assert approx(ta.williams_r(highs, lows, [9.0, 11, 13, 16], 3)[-1], 0)
    assert approx(ta.williams_r(highs, lows, [9.0, 11, 13, 9], 3)[-1], -100)


def test_cci_roc_and_mfi() -> None:
    same = [10.0] * 25
    assert approx(ta.cci(same, same, same, 20)[-1], 0)
    assert approx(ta.roc([100.0, 110.0], 1)[-1], 10)
    rising = [float(i) for i in range(1, 30)]
    assert approx(ta.mfi(rising, rising, rising, [1000.0] * 29, 14)[-1], 100)


# ---------- Volatility ----------


def test_true_range_and_atr() -> None:
    closes = [100.0] * 20
    highs = [101.0] * 20
    lows = [99.0] * 20
    assert all(approx(v, 2) for v in ta.atr(highs, lows, closes, 14)[13:])
    # A gap up: the true range reaches back to the previous close.
    assert ta.true_range([101.0, 111.0], [99.0, 109.0], [100.0, 110.0])[-1] == 11


def test_bollinger_bands() -> None:
    bands = ta.bollinger([1.0, 2.0, 3.0], 3, 2)
    std = math.sqrt(2 / 3)
    assert approx(bands.middle[-1], 2)
    assert approx(bands.upper[-1], 2 + 2 * std)
    assert approx(bands.lower[-1], 2 - 2 * std)
    flat = ta.bollinger([5.0] * 20, 20, 2)
    assert approx(flat.upper[-1], 5)
    assert approx(flat.lower[-1], 5)


def test_keltner_and_donchian() -> None:
    k = ta.keltner([101.0] * 30, [99.0] * 30, [100.0] * 30, 20, 2)
    assert approx(k.middle[-1], 100)
    assert approx(k.upper[-1], 104)
    assert approx(k.lower[-1], 96)
    d = ta.donchian([1.0, 5, 3, 4], [0.0, 2, 1, 2], 3)
    assert (d.upper[-1], d.lower[-1]) == (5, 1)


# ---------- Volume ----------


def test_obv() -> None:
    assert ta.obv([10.0, 11, 10, 10, 12], [100.0, 200, 300, 400, 500]) == [0, 200, -100, -100, 400]


def test_vwap_is_volume_weighted() -> None:
    prices = [10.0, 20.0]
    result = ta.vwap(prices, prices, prices, [1.0, 3.0])
    assert approx(result[0], 10)
    assert approx(result[1], 17.5)  # (10*1 + 20*3) / 4


def test_chaikin_money_flow() -> None:
    highs, lows = [11.0] * 25, [9.0] * 25
    assert approx(ta.cmf(highs, lows, [11.0] * 25, [100.0] * 25, 20)[-1], 1)
    assert approx(ta.cmf(highs, lows, [9.0] * 25, [100.0] * 25, 20)[-1], -1)


# ---------- Trend strength ----------


def test_adx_sees_a_strong_trend_and_a_choppy_market() -> None:
    up = [100.0 + i for i in range(60)]
    trend = ta.adx([c + 1 for c in up], [c - 1 for c in up], up, 14)
    assert trend.plus_di[-1] is not None
    assert trend.minus_di[-1] is not None
    assert trend.plus_di[-1] > trend.minus_di[-1]
    assert trend.adx[-1] is not None
    assert trend.adx[-1] > 25
    chop = [100.0 + (1 if i % 2 else -1) for i in range(60)]
    flat = ta.adx([c + 1 for c in chop], [c - 1 for c in chop], chop, 14)
    assert flat.adx[-1] is not None
    assert flat.adx[-1] < 20


# ---------- Levels ----------


def test_classic_pivot_points() -> None:
    p = ta.pivots(110, 90, 100)
    assert (p["P"], p["R1"], p["S1"], p["R2"], p["S2"], p["R3"], p["S3"]) == (
        100,
        110,
        90,
        120,
        80,
        130,
        70,
    )


def test_swing_levels_find_peaks_and_troughs() -> None:
    highs = [10.0, 12, 15, 12, 10, 11, 14, 11, 10]
    lows = [8.0, 9, 11, 7, 6, 8, 10, 5, 7]
    supports, resistances = ta.swing_levels(highs, lows, 2)
    assert 15 in resistances
    assert 14 in resistances
    assert 6 in supports


# ---------- Candlesticks ----------


@pytest.mark.parametrize(
    ("bars", "pattern"),
    [
        ([(10, 10.5, 9.5, 10.02)], "doji"),
        ([(10, 10.1, 8, 10.05)], "hammer"),
        ([(10, 12, 9.95, 10.05)], "shooting_star"),
        ([(10, 10.2, 8.8, 9), (8.9, 10.6, 8.8, 10.5)], "bullish_engulfing"),
        ([(9, 10.2, 8.8, 10), (10.1, 10.2, 8.4, 8.5)], "bearish_engulfing"),
        ([(10, 10.1, 7.9, 8), (7.9, 8, 7.6, 7.8), (7.9, 9.6, 7.8, 9.5)], "morning_star"),
        ([(8, 10.1, 7.9, 10), (10.1, 10.4, 10, 10.2), (10.1, 10.2, 8.4, 8.5)], "evening_star"),
        ([(10, 12, 10, 12)], "bullish_marubozu"),
        ([(12, 12, 10, 10)], "bearish_marubozu"),
        ([(10, 14, 8, 13), (11, 13, 9, 12)], "inside_bar"),
    ],
)
def test_candlestick_patterns(bars: list[tuple[float, float, float, float]], pattern: str) -> None:
    opens, highs, lows, closes = (list(col) for col in zip(*bars, strict=True))
    assert pattern in ta.candlestick_patterns(opens, highs, lows, closes)


def test_an_ordinary_candle_has_no_pattern() -> None:
    assert ta.candlestick_patterns([10.0], [11.0], [9.6], [10.6]) == []


# ---------- Properties over random markets ----------


def bars_strategy() -> st.SearchStrategy[list[tuple[float, float, float, float, float]]]:
    price = st.floats(min_value=5, max_value=500, allow_nan=False, allow_infinity=False)
    spread = st.floats(min_value=0, max_value=0.1)
    volume = st.floats(min_value=0, max_value=1e7)

    def bar(p: float, up: float, down: float, frac: float, v: float) -> tuple[float, ...]:
        high, low = p * (1 + up), p * (1 - down)
        return (p, high, low, low + (high - low) * frac, v)

    return st.lists(
        st.builds(bar, price, spread, spread, st.floats(min_value=0, max_value=1), volume),
        min_size=1,
        max_size=80,
    )  # type: ignore[return-value]


@settings(max_examples=120)
@given(bars=bars_strategy())
def test_indicators_stay_in_range_and_aligned(bars: list[tuple[float, ...]]) -> None:
    opens, highs, lows, closes, volumes = (list(col) for col in zip(*bars, strict=True))
    n = len(closes)
    series = {
        "rsi": ta.rsi(closes, 14),
        "stoch": ta.stochastic(highs, lows, closes).k,
        "williams": ta.williams_r(highs, lows, closes, 14),
        "mfi": ta.mfi(highs, lows, closes, volumes, 14),
        "cmf": ta.cmf(highs, lows, closes, volumes, 20),
        "atr": ta.atr(highs, lows, closes, 14),
        "adx": ta.adx(highs, lows, closes, 14).adx,
    }
    assert all(len(values) == n for values in series.values())
    bounds = {
        "rsi": (0, 100),
        "stoch": (0, 100),
        "williams": (-100, 0),
        "mfi": (0, 100),
        "cmf": (-1, 1),
        "atr": (0, math.inf),
        "adx": (0, 100),
    }
    for name, values in series.items():
        low, high = bounds[name]
        assert all(low - 1e-9 <= v <= high + 1e-9 for v in values if v is not None), name
    bands = ta.bollinger(closes, 20, 2)
    for lo, mid, up in zip(bands.lower, bands.middle, bands.upper, strict=True):
        if mid is not None:
            assert lo is not None
            assert up is not None
            assert lo <= mid + 1e-9 <= up + 2e-9
    assert len(ta.candlestick_patterns(opens, highs, lows, closes)) <= 10
