"""Technical indicators (ADR 0020): textbook definitions, pure functions, no dependencies.

Every series function takes plain float sequences (oldest first) and returns a list of the
same length, with None where the indicator has not warmed up yet. Smoothing follows the
original definitions: Wilder's for RSI, ATR, and ADX; EMAs seeded with an SMA.

These are statistics for research, so they use floats. Money and order sizes elsewhere stay
exact Decimals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

Series = list[float | None]


# ---------- Moving averages ----------


def sma(values: Sequence[float], period: int) -> Series:
    out: Series = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1 : i + 1]) / period
    return out


def ema(values: Sequence[float], period: int) -> Series:
    """Exponential moving average, seeded with the SMA of the first `period` values."""
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2 / (period + 1)
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = values[i] * alpha + current * (1 - alpha)
        out[i] = current
    return out


def _ema_of(series: Series, period: int) -> Series:
    """EMA of a series that starts with a warm-up of Nones."""
    start = next((i for i, v in enumerate(series) if v is not None), len(series))
    tail = ema([v for v in series[start:] if v is not None], period)
    return [None] * start + tail


def _wilder(values: Sequence[float], period: int, start: int = 0) -> Series:
    """Wilder smoothing: first value is the mean of `period` values from `start`."""
    out: Series = [None] * len(values)
    first = start + period - 1
    if len(values) <= first:
        return out
    current = sum(values[start : first + 1]) / period
    out[first] = current
    for i in range(first + 1, len(values)):
        current = (current * (period - 1) + values[i]) / period
        out[i] = current
    return out


# ---------- Momentum ----------


def rsi(closes: Sequence[float], period: int = 14) -> Series:
    """Relative Strength Index (Wilder). 50 when there is no movement at all."""
    out: Series = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = [0.0] + [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [0.0] + [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = _wilder(gains, period, start=1)
    avg_loss = _wilder(losses, period, start=1)
    for i in range(period, len(closes)):
        gain, loss = avg_gain[i], avg_loss[i]
        if gain is None or loss is None:
            continue
        if loss == 0:
            out[i] = 50.0 if gain == 0 else 100.0
        else:
            out[i] = 100 - 100 / (1 + gain / loss)
    return out


class MACD(NamedTuple):
    line: Series
    signal: Series
    histogram: Series


def macd(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> MACD:
    fast_ema, slow_ema = ema(closes, fast), ema(closes, slow)
    line: Series = [
        None if f is None or s is None else f - s for f, s in zip(fast_ema, slow_ema, strict=True)
    ]
    signal_line = _ema_of(line, signal)
    histogram: Series = [
        None if m is None or s is None else m - s for m, s in zip(line, signal_line, strict=True)
    ]
    return MACD(line, signal_line, histogram)


class Stochastic(NamedTuple):
    k: Series
    d: Series


def _window_extremes(
    highs: Sequence[float], lows: Sequence[float], i: int, period: int
) -> tuple[float, float]:
    return max(highs[i - period + 1 : i + 1]), min(lows[i - period + 1 : i + 1])


def stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int = 14,
    d_period: int = 3,
) -> Stochastic:
    k: Series = [None] * len(closes)
    for i in range(k_period - 1, len(closes)):
        hh, ll = _window_extremes(highs, lows, i, k_period)
        k[i] = 50.0 if hh == ll else 100 * (closes[i] - ll) / (hh - ll)
    start = next((i for i, v in enumerate(k) if v is not None), len(k))
    d = [None] * start + sma([v for v in k[start:] if v is not None], d_period)
    return Stochastic(k, d)


def williams_r(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> Series:
    out: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        hh, ll = _window_extremes(highs, lows, i, period)
        out[i] = -50.0 if hh == ll else -100 * (hh - closes[i]) / (hh - ll)
    return out


def _typical(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    return [(h + lo + c) / 3 for h, lo, c in zip(highs, lows, closes, strict=True)]


def cci(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 20
) -> Series:
    """Commodity Channel Index. 0 when prices do not vary."""
    tp = _typical(highs, lows, closes)
    out: Series = [None] * len(tp)
    for i in range(period - 1, len(tp)):
        window = tp[i - period + 1 : i + 1]
        mean = sum(window) / period
        deviation = sum(abs(v - mean) for v in window) / period
        out[i] = 0.0 if deviation == 0 else (tp[i] - mean) / (0.015 * deviation)
    return out


def roc(closes: Sequence[float], period: int = 10) -> Series:
    out: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        base = closes[i - period]
        out[i] = None if base == 0 else (closes[i] - base) / base * 100
    return out


def mfi(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 14,
) -> Series:
    """Money Flow Index: a volume-weighted RSI. 50 when no money moved either way."""
    tp = _typical(highs, lows, closes)
    out: Series = [None] * len(tp)
    for i in range(period, len(tp)):
        positive = negative = 0.0
        for j in range(i - period + 1, i + 1):
            flow = tp[j] * volumes[j]
            if tp[j] > tp[j - 1]:
                positive += flow
            elif tp[j] < tp[j - 1]:
                negative += flow
        if negative == 0:
            out[i] = 50.0 if positive == 0 else 100.0
        else:
            out[i] = 100 - 100 / (1 + positive / negative)
    return out


# ---------- Volatility ----------


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    ranges = []
    for i, (high, low) in enumerate(zip(highs, lows, strict=True)):
        if i == 0:
            ranges.append(high - low)
        else:
            previous = closes[i - 1]
            ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
    return ranges


def atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> Series:
    """Average True Range (Wilder)."""
    return _wilder(true_range(highs, lows, closes), period)


class Bands(NamedTuple):
    upper: Series
    middle: Series
    lower: Series


def bollinger(closes: Sequence[float], period: int = 20, k: float = 2.0) -> Bands:
    middle = sma(closes, period)
    upper: Series = [None] * len(closes)
    lower: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        mean = middle[i]
        assert mean is not None  # noqa: S101 - set by sma for this index
        window = closes[i - period + 1 : i + 1]
        std = math.sqrt(sum((v - mean) ** 2 for v in window) / period)
        upper[i], lower[i] = mean + k * std, mean - k * std
    return Bands(upper, middle, lower)


def keltner(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 20,
    multiplier: float = 2.0,
) -> Bands:
    middle = ema(closes, period)
    ranges = atr(highs, lows, closes, period)
    upper: Series = [
        None if m is None or r is None else m + multiplier * r
        for m, r in zip(middle, ranges, strict=True)
    ]
    lower: Series = [
        None if m is None or r is None else m - multiplier * r
        for m, r in zip(middle, ranges, strict=True)
    ]
    return Bands(upper, middle, lower)


def donchian(highs: Sequence[float], lows: Sequence[float], period: int = 20) -> Bands:
    upper: Series = [None] * len(highs)
    lower: Series = [None] * len(highs)
    middle: Series = [None] * len(highs)
    for i in range(period - 1, len(highs)):
        hh, ll = _window_extremes(highs, lows, i, period)
        upper[i], lower[i], middle[i] = hh, ll, (hh + ll) / 2
    return Bands(upper, middle, lower)


# ---------- Volume ----------


def obv(closes: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """On-Balance Volume, starting at 0."""
    out = [0.0] * len(closes)
    for i in range(1, len(closes)):
        step = (
            volumes[i]
            if closes[i] > closes[i - 1]
            else -volumes[i]
            if closes[i] < closes[i - 1]
            else 0.0
        )
        out[i] = out[i - 1] + step
    return out


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Cumulative volume-weighted average price (pass one session of intraday bars)."""
    out: list[float] = []
    value = volume = 0.0
    for tp, v in zip(_typical(highs, lows, closes), volumes, strict=True):
        value += tp * v
        volume += v
        out.append(value / volume if volume else tp)
    return out


def cmf(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 20,
) -> Series:
    """Chaikin Money Flow: +1 when every close is at the high, -1 at the low."""
    flows = []
    for h, lo, c, v in zip(highs, lows, closes, volumes, strict=True):
        multiplier = 0.0 if h == lo else ((c - lo) - (h - c)) / (h - lo)
        flows.append(multiplier * v)
    out: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        total = sum(volumes[i - period + 1 : i + 1])
        out[i] = 0.0 if total == 0 else sum(flows[i - period + 1 : i + 1]) / total
    return out


# ---------- Trend strength ----------


class ADX(NamedTuple):
    adx: Series
    plus_di: Series
    minus_di: Series


def adx(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> ADX:
    """Average Directional Index with the +DI and -DI lines (Wilder)."""
    n = len(closes)
    plus_dm, minus_dm = [0.0] * n, [0.0] * n
    for i in range(1, n):
        up, down = highs[i] - highs[i - 1], lows[i - 1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0
    tr = true_range(highs, lows, closes)
    s_tr, s_plus, s_minus = (_wilder(x, period, start=1) for x in (tr, plus_dm, minus_dm))
    plus_di: Series = [None] * n
    minus_di: Series = [None] * n
    dx: list[float] = []
    dx_index: list[int] = []
    for i in range(n):
        t, p, m = s_tr[i], s_plus[i], s_minus[i]
        if t is None or p is None or m is None:
            continue
        plus_di[i] = 0.0 if t == 0 else 100 * p / t
        minus_di[i] = 0.0 if t == 0 else 100 * m / t
        total = plus_di[i] + minus_di[i]  # type: ignore[operator]
        dx.append(0.0 if total == 0 else 100 * abs(plus_di[i] - minus_di[i]) / total)  # type: ignore[operator]
        dx_index.append(i)
    smoothed = _wilder(dx, period)
    adx_line: Series = [None] * n
    for value, i in zip(smoothed, dx_index, strict=True):
        adx_line[i] = value
    return ADX(adx_line, plus_di, minus_di)


# ---------- Levels ----------


def pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Classic floor-trader pivot points from the previous session."""
    p = (high + low + close) / 3
    return {
        "P": p,
        "R1": 2 * p - low,
        "S1": 2 * p - high,
        "R2": p + (high - low),
        "S2": p - (high - low),
        "R3": high + 2 * (p - low),
        "S3": low - 2 * (high - p),
    }


def swing_levels(
    highs: Sequence[float], lows: Sequence[float], window: int = 3
) -> tuple[list[float], list[float]]:
    """(supports, resistances): swing lows and highs that are the extreme of their window."""
    supports: set[float] = set()
    resistances: set[float] = set()
    for i in range(window, len(highs) - window):
        span_h = highs[i - window : i + window + 1]
        span_l = lows[i - window : i + window + 1]
        if highs[i] == max(span_h) and span_h.count(highs[i]) == 1:
            resistances.add(round(highs[i], 2))
        if lows[i] == min(span_l) and span_l.count(lows[i]) == 1:
            supports.add(round(lows[i], 2))
    return sorted(supports), sorted(resistances)


# ---------- Candlesticks ----------


def candlestick_patterns(
    opens: Sequence[float], highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[str]:
    """Classic patterns completed by the last bar, from its body, wicks, and range."""
    if not closes:
        return []
    found: list[str] = []
    o, h, lo, c = opens[-1], highs[-1], lows[-1], closes[-1]
    span = h - lo
    body = abs(c - o)
    upper, lower = h - max(o, c), min(o, c) - lo
    if span > 0:
        if body <= 0.1 * span:
            found.append("doji")
        if lower >= 0.6 * span and upper <= 0.25 * span and lower >= 2 * body:
            found.append("hammer")
        if upper >= 0.6 * span and lower <= 0.25 * span and upper >= 2 * body:
            found.append("shooting_star")
        if body >= 0.9 * span:
            found.append("bullish_marubozu" if c > o else "bearish_marubozu")
    if len(closes) >= 2:
        po, ph, pl, pc = opens[-2], highs[-2], lows[-2], closes[-2]
        if pc < po and c > o and o <= pc and c >= po:
            found.append("bullish_engulfing")
        if pc > po and c < o and o >= pc and c <= po:
            found.append("bearish_engulfing")
        if h < ph and lo > pl:
            found.append("inside_bar")
    if len(closes) >= 3:
        o1, c1 = opens[-3], closes[-3]
        o2, c2 = opens[-2], closes[-2]
        first_body = abs(c1 - o1)
        small_middle = first_body > 0 and abs(c2 - o2) <= 0.3 * first_body
        midpoint = (o1 + c1) / 2
        if c1 < o1 and small_middle and max(o2, c2) <= c1 and c > o and c > midpoint:
            found.append("morning_star")
        if c1 > o1 and small_middle and min(o2, c2) >= c1 and c < o and c < midpoint:
            found.append("evening_star")
    return found
