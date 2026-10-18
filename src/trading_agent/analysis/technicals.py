"""The technical report (ADR 0020): every standard indicator for one stock, with readings.

Deterministic. The same bars always give the same report. Each reading is a fixed rule
from common technical practice ("RSI 70+: overbought, pullback risk"), so the analyst gets
conclusions it can cite instead of doing arithmetic in its head, and the critic can check
them against the same numbers.

Sections:
    trend              20/50/200-day SMAs, 9/21 EMAs, MACD, ADX with +DI/-DI, 50/200 crosses
    momentum           RSI, stochastic, Williams %R, CCI, rate of change, money flow index
    volatility         ATR, Bollinger bands (with squeeze), Keltner and Donchian channels
    volume             relative volume, on-balance volume trend, Chaikin money flow
    levels             pivot points, swing supports/resistances, 52-week high and low
    patterns           candlesticks, RSI divergence, the opening gap
    intraday           VWAP, the 30-minute opening range, 5-minute EMA trend and RSI
    relative_strength  20-day return against SPY and QQQ

Indicators describe the past. They are evidence to weigh, not predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.analysis import indicators as ta
from trading_agent.domain.models import Bar

Signal = Literal["bullish", "bearish", "neutral"]
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
OPENING_RANGE_MINUTES = 30
CROSS_LOOKBACK = 30  # sessions in which a 50/200 cross still counts as recent
BIAS_MARGIN = 3  # net readings needed to call the picture bullish or bearish
MACD_FLAT = 0.05  # a histogram within 5% of the MACD line counts as in line with it


class Reading(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    signal: Signal
    note: str


class TechnicalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: datetime
    price: Decimal
    trend: dict[str, Any]
    momentum: dict[str, Any]
    volatility: dict[str, Any]
    volume: dict[str, Any]
    levels: dict[str, Any]
    patterns: dict[str, Any]
    intraday: dict[str, Any] | None
    relative_strength: dict[str, Any]
    signals: list[Reading]
    bullish: int
    bearish: int
    neutral: int
    bias: Literal["bullish", "bearish", "mixed"]
    warnings: list[str]


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _last(series: Sequence[float | None]) -> float | None:
    return series[-1] if series else None


def _floats(bars: Sequence[Bar]) -> tuple[list[float], ...]:
    return (
        [float(b.open) for b in bars],
        [float(b.high) for b in bars],
        [float(b.low) for b in bars],
        [float(b.close) for b in bars],
        [float(b.volume) for b in bars],
    )


def _return(closes: Sequence[float], days: int) -> float | None:
    if len(closes) <= days or closes[-days - 1] == 0:
        return None
    return (closes[-1] / closes[-days - 1] - 1) * 100


def technical_report(
    symbol: str,
    daily: Sequence[Bar],
    *,
    now: datetime,
    intraday: Sequence[Bar] = (),
    benchmarks: Mapping[str, Sequence[Bar]] | None = None,
    timezone: str = "America/New_York",
) -> TechnicalReport:
    """Daily bars oldest first (ideally a year); intraday 5-minute bars for today."""
    if not daily:
        raise ValueError(f"no daily bars for {symbol}")
    zone = ZoneInfo(timezone)
    today: date = now.astimezone(zone).date()
    opens, highs, lows, closes, volumes = _floats(daily)
    signals: list[Reading] = []
    warnings: list[str] = []

    def say(name: str, signal: Signal, note: str) -> None:
        signals.append(Reading(name=name, signal=signal, note=note))

    # Today's regular-session bars, if any.
    session = [
        b
        for b in intraday
        if b.timestamp.astimezone(zone).date() == today
        and SESSION_OPEN <= b.timestamp.astimezone(zone).time() < SESSION_CLOSE
    ]
    price = float(session[-1].close) if session else closes[-1]
    # The last completed session (the latest daily bar may be today's, still forming).
    completed = (
        len(daily) - 2
        if daily[-1].timestamp.astimezone(zone).date() == today and len(daily) > 1
        else len(daily) - 1
    )

    # ---------- Trend ----------
    sma20, sma50, sma200 = (_last(ta.sma(closes, p)) for p in (20, 50, 200))
    ema9, ema21 = _last(ta.ema(closes, 9)), _last(ta.ema(closes, 21))
    macd = ta.macd(closes)
    trend_strength = ta.adx(highs, lows, closes)
    for label, average in (("20-day", sma20), ("50-day", sma50), ("200-day", sma200)):
        if average is None:
            warnings.append(f"not enough history for the {label} average")
            continue
        above = price > average
        say(
            f"{label} average",
            "bullish" if above else "bearish",
            f"price {'above' if above else 'below'} the {label} average ({average:.2f})",
        )
    sma50_series, sma200_series = ta.sma(closes, 50), ta.sma(closes, 200)
    if sma50 is not None and sma200 is not None:
        cross = None
        for back in range(1, min(CROSS_LOOKBACK, len(closes) - 1) + 1):
            a_now, b_now = sma50_series[-back], sma200_series[-back]
            a_was, b_was = sma50_series[-back - 1], sma200_series[-back - 1]
            if None in (a_now, b_now, a_was, b_was):
                break
            if a_was <= b_was < a_now:  # type: ignore[operator]
                cross = ("golden", back)
                break
            if a_was >= b_was > a_now:  # type: ignore[operator]
                cross = ("death", back)
                break
        if cross:
            kind, days_ago = cross
            say(
                "50/200-day cross",
                "bullish" if kind == "golden" else "bearish",
                f"{kind} cross {days_ago} sessions ago",
            )
        else:
            above = sma50 > sma200
            say(
                "50/200-day cross",
                "bullish" if above else "bearish",
                f"50-day {'above' if above else 'below'} the 200-day: long-term "
                f"{'uptrend' if above else 'downtrend'} structure",
            )
    line, signal_line, hist = (_last(s) for s in macd)
    if line is not None and signal_line is not None and hist is not None:
        flat = abs(hist) <= abs(line) * MACD_FLAT
        if line > 0 and (hist >= 0 or flat):
            say("MACD", "bullish", "MACD above zero and above its signal: upward momentum")
        elif line < 0 and (hist <= 0 or flat):
            say("MACD", "bearish", "MACD below zero and below its signal: downward momentum")
        else:
            say(
                "MACD",
                "neutral",
                "MACD and its signal disagree: momentum is "
                + ("fading" if line > 0 else "improving"),
            )
    adx_value = _last(trend_strength.adx)
    plus_di, minus_di = _last(trend_strength.plus_di), _last(trend_strength.minus_di)
    if adx_value is not None and plus_di is not None and minus_di is not None:
        if adx_value >= 25:
            up = plus_di > minus_di
            say(
                "ADX",
                "bullish" if up else "bearish",
                f"strong {'up' if up else 'down'}trend (ADX {adx_value:.0f})",
            )
        elif adx_value < 20:
            say("ADX", "neutral", f"no clear trend (ADX {adx_value:.0f}): choppy market")
        else:
            say("ADX", "neutral", f"a trend may be forming (ADX {adx_value:.0f})")

    # ---------- Momentum ----------
    rsi = _last(ta.rsi(closes))
    stoch = ta.stochastic(highs, lows, closes)
    k, d = _last(stoch.k), _last(stoch.d)
    mfi = _last(ta.mfi(highs, lows, closes, volumes))
    if rsi is not None:
        if rsi >= 70:
            say("RSI", "bearish", f"RSI {rsi:.0f}: overbought, pullback risk")
        elif rsi <= 30:
            say("RSI", "neutral", f"RSI {rsi:.0f}: oversold, may bounce but is still falling")
        elif rsi >= 50:
            say("RSI", "bullish", f"RSI {rsi:.0f}: buyers in control")
        else:
            say("RSI", "bearish", f"RSI {rsi:.0f}: sellers in control")
    if k is not None:
        if k >= 80:
            say("Stochastic", "bearish", f"stochastic {k:.0f}: overbought")
        elif k <= 20:
            say("Stochastic", "neutral", f"stochastic {k:.0f}: oversold")
        else:
            say("Stochastic", "neutral", f"stochastic {k:.0f}: mid-range")
    if mfi is not None:
        if mfi >= 80:
            say("MFI", "bearish", f"money flow {mfi:.0f}: overbought on volume")
        elif mfi <= 20:
            say("MFI", "neutral", f"money flow {mfi:.0f}: oversold on volume")

    # ---------- Volatility ----------
    atr = _last(ta.atr(highs, lows, closes))
    bands = ta.bollinger(closes)
    upper, middle, lower = (_last(s) for s in bands)
    percent_b = bandwidth = None
    squeeze = False
    if upper is not None and lower is not None and middle:
        percent_b = (price - lower) / (upper - lower) if upper > lower else 0.5
        bandwidth = (upper - lower) / middle
        widths = [
            (u - lo) / m
            for u, lo, m in zip(
                bands.upper[-120:], bands.lower[-120:], bands.middle[-120:], strict=True
            )
            if u is not None and lo is not None and m
        ]
        squeeze = len(widths) >= 60 and bandwidth <= min(widths) * 1.05
        if percent_b > 1:
            say("Bollinger", "bearish", "above the upper Bollinger band: stretched")
        elif percent_b < 0:
            say("Bollinger", "neutral", "below the lower Bollinger band: stretched down")
        if squeeze:
            say("Bollinger squeeze", "neutral", "bands at a 6-month low: a big move may come")
    keltner = ta.keltner(highs, lows, closes)
    donchian = ta.donchian(highs[:-1], lows[:-1]) if len(highs) > 20 else None

    # ---------- Volume ----------
    window = volumes[-21:-1] if len(volumes) > 20 else volumes[:-1]
    average_volume = sum(window) / len(window) if window else None
    relative = volumes[-1] / average_volume if average_volume else None
    if relative is not None:
        if relative >= 1.5:
            up_day = closes[-1] >= opens[-1]
            say(
                "Volume",
                "bullish" if up_day else "bearish",
                f"heavy volume ({relative:.1f}x average) on {'an up' if up_day else 'a down'} day",
            )
        elif relative <= 0.6:
            say("Volume", "neutral", f"light volume ({relative:.1f}x average): little conviction")
        else:
            say("Volume", "neutral", f"normal volume ({relative:.1f}x average)")
    on_balance = ta.obv(closes, volumes)
    obv_trend = "flat"
    if len(on_balance) > 20:
        obv_trend = (
            "rising"
            if on_balance[-1] > on_balance[-21]
            else "falling"
            if on_balance[-1] < on_balance[-21]
            else "flat"
        )
        if obv_trend != "flat":
            say(
                "OBV",
                "bullish" if obv_trend == "rising" else "bearish",
                f"on-balance volume {obv_trend} over 20 sessions",
            )
    money_flow = _last(ta.cmf(highs, lows, closes, volumes))
    if money_flow is not None and abs(money_flow) >= 0.05:
        say(
            "Money flow",
            "bullish" if money_flow > 0 else "bearish",
            f"Chaikin money flow {money_flow:+.2f}: "
            + ("accumulation" if money_flow > 0 else "distribution"),
        )

    # ---------- Levels ----------
    pivot = ta.pivots(highs[completed], lows[completed], closes[completed])
    supports, resistances = ta.swing_levels(highs[-60:], lows[-60:], 3)
    below = sorted((s for s in supports if s < price), reverse=True)[:3]
    above_levels = sorted(r for r in resistances if r > price)[:3]
    year = daily[-252:]
    high_52w = max(float(b.high) for b in year)
    low_52w = min(float(b.low) for b in year)
    if donchian is not None:
        prior_high, prior_low = _last(donchian.upper), _last(donchian.lower)
        if prior_high is not None and closes[-1] > prior_high:
            heavy = relative is not None and relative >= 1.5
            say(
                "Breakout",
                "bullish" if heavy else "neutral",
                f"closed above the 20-day high ({prior_high:.2f})"
                + (" on heavy volume" if heavy else " on ordinary volume: needs confirmation"),
            )
        elif prior_low is not None and closes[-1] < prior_low:
            say("Breakout", "bearish", f"closed below the 20-day low ({prior_low:.2f})")
    if above_levels and (above_levels[0] - price) / price <= 0.01:
        say("Resistance", "neutral", f"just under resistance at {above_levels[0]:.2f}")
    if below and (price - below[0]) / price <= 0.01:
        say("Support", "neutral", f"just above support at {below[0]:.2f}")
    if high_52w and (high_52w - price) / high_52w <= 0.02:
        say("52-week", "bullish", "within 2% of the 52-week high: strength")
    elif low_52w and (price - low_52w) / low_52w <= 0.02:
        say("52-week", "bearish", "within 2% of the 52-week low: weakness")

    # ---------- Patterns ----------
    candles = ta.candlestick_patterns(opens, highs, lows, closes)
    bullish_candles = {"hammer", "bullish_engulfing", "morning_star", "bullish_marubozu"}
    bearish_candles = {"shooting_star", "bearish_engulfing", "evening_star", "bearish_marubozu"}
    for name in candles:
        tone: Signal = (
            "bullish"
            if name in bullish_candles
            else "bearish"
            if name in bearish_candles
            else "neutral"
        )
        say("Candles", tone, f"{name.replace('_', ' ')} on the last daily bar")
    divergence = None
    rsi_series = ta.rsi(closes)
    recent_rsi = [v for v in rsi_series[-20:] if v is not None]
    if len(closes) >= 40 and len(recent_rsi) == 20:
        first, second = slice(0, 10), slice(10, None)
        recent = closes[-20:]
        price_hi = (max(recent[first]), max(recent[second]))
        rsi_hi = (max(recent_rsi[first]), max(recent_rsi[second]))
        price_lo = (min(recent[first]), min(recent[second]))
        rsi_lo = (min(recent_rsi[first]), min(recent_rsi[second]))
        if price_hi[1] > price_hi[0] and rsi_hi[1] < rsi_hi[0]:
            divergence = "bearish"
            say(
                "RSI divergence", "bearish", "higher price high but lower RSI high: momentum fading"
            )
        elif price_lo[1] < price_lo[0] and rsi_lo[1] > rsi_lo[0]:
            divergence = "bullish"
            say("RSI divergence", "bullish", "lower price low but higher RSI low: selling fading")
    gap = None
    if session:
        previous_close = closes[completed]
        gap = (float(session[0].open) / previous_close - 1) * 100 if previous_close else None

    # ---------- Intraday ----------
    intraday_view = None
    if session:
        _, s_highs, s_lows, s_closes, s_volumes = _floats(session)
        vwap = ta.vwap(s_highs, s_lows, s_closes, s_volumes)[-1]
        first_bar = session[0].timestamp.astimezone(zone)
        opening = [
            b
            for b in session
            if (b.timestamp.astimezone(zone) - first_bar).total_seconds()
            < OPENING_RANGE_MINUTES * 60
        ]
        or_high = max(float(b.high) for b in opening)
        or_low = min(float(b.low) for b in opening)
        fast, slow = _last(ta.ema(s_closes, 9)), _last(ta.ema(s_closes, 20))
        intraday_view = {
            "bars": len(session),
            "vwap": _r(vwap),
            "price_vs_vwap_pct": _r((price / vwap - 1) * 100) if vwap else None,
            "opening_range_high": _r(or_high, 4),
            "opening_range_low": _r(or_low, 4),
            "ema_9": _r(fast),
            "ema_20": _r(slow),
            "rsi_14": _r(_last(ta.rsi(s_closes))),
            "gap_pct": _r(gap),
        }
        say(
            "VWAP",
            "bullish" if price > vwap else "bearish",
            f"price {'above' if price > vwap else 'below'} VWAP ({vwap:.2f}): "
            + ("buyers in control today" if price > vwap else "sellers in control today"),
        )
        if len(session) > len(opening):
            if price > or_high:
                say(
                    "Opening range",
                    "bullish",
                    f"broke above the opening range high ({or_high:.2f})",
                )
            elif price < or_low:
                say("Opening range", "bearish", f"broke below the opening range low ({or_low:.2f})")
            else:
                say("Opening range", "neutral", "still inside the opening range")
        if fast is not None and slow is not None:
            say(
                "Intraday trend",
                "bullish" if fast > slow else "bearish",
                f"5-minute 9 EMA {'above' if fast > slow else 'below'} the 20 EMA",
            )
        if gap is not None and abs(gap) >= 2:
            say("Gap", "neutral", f"opened {gap:+.1f}% from the last close: gaps often partly fill")
    else:
        warnings.append("no intraday bars yet today")

    # ---------- Relative strength ----------
    strength: dict[str, Any] = {"return_20d": _r(_return(closes, 20))}
    for name, bars in (benchmarks or {}).items():
        own, theirs = _return(closes, 20), _return([float(b.close) for b in bars], 20)
        if own is None or theirs is None:
            continue
        gap_points = own - theirs
        strength[f"vs_{name}_20d"] = _r(gap_points)
        if gap_points >= 2:
            say(f"vs {name}", "bullish", f"beat {name} by {gap_points:.1f} points over 20 sessions")
        elif gap_points <= -2:
            say(
                f"vs {name}",
                "bearish",
                f"lagged {name} by {-gap_points:.1f} points over 20 sessions",
            )

    bullish = sum(1 for s in signals if s.signal == "bullish")
    bearish = sum(1 for s in signals if s.signal == "bearish")
    net = bullish - bearish
    return TechnicalReport(
        symbol=symbol,
        as_of=now.astimezone(zone),
        price=Decimal(str(round(price, 4))),
        trend={
            "sma_20": _r(sma20),
            "sma_50": _r(sma50),
            "sma_200": _r(sma200),
            "ema_9": _r(ema9),
            "ema_21": _r(ema21),
            "macd": _r(line, 3),
            "macd_signal": _r(signal_line, 3),
            "macd_histogram": _r(hist, 3),
            "adx": _r(adx_value),
            "plus_di": _r(plus_di),
            "minus_di": _r(minus_di),
        },
        momentum={
            "rsi_14": _r(rsi),
            "stoch_k": _r(k),
            "stoch_d": _r(d),
            "williams_r": _r(_last(ta.williams_r(highs, lows, closes))),
            "cci_20": _r(_last(ta.cci(highs, lows, closes))),
            "roc_10": _r(_last(ta.roc(closes))),
            "mfi_14": _r(mfi),
        },
        volatility={
            "atr_14": _r(atr),
            "atr_pct": _r(atr / price * 100) if atr is not None and price else None,
            "bollinger_upper": _r(upper),
            "bollinger_middle": _r(middle),
            "bollinger_lower": _r(lower),
            "bollinger_percent_b": _r(percent_b),
            "bollinger_bandwidth": _r(bandwidth, 4),
            "squeeze": squeeze,
            "keltner_upper": _r(_last(keltner.upper)),
            "keltner_lower": _r(_last(keltner.lower)),
            "donchian_high_20": _r(_last(donchian.upper)) if donchian else None,
            "donchian_low_20": _r(_last(donchian.lower)) if donchian else None,
        },
        volume={
            "last": int(volumes[-1]),
            "average_20d": int(average_volume) if average_volume else None,
            "relative_volume": _r(relative),
            "obv_trend": obv_trend,
            "chaikin_money_flow": _r(money_flow),
        },
        levels={
            "pivots": {name: _r(value) for name, value in pivot.items()},
            "supports": below,
            "resistances": above_levels,
            "high_52w": _r(high_52w),
            "low_52w": _r(low_52w),
            "pct_from_52w_high": _r((price / high_52w - 1) * 100) if high_52w else None,
        },
        patterns={"candles": candles, "rsi_divergence": divergence, "gap_pct": _r(gap)},
        intraday=intraday_view,
        relative_strength=strength,
        signals=signals,
        bullish=bullish,
        bearish=bearish,
        neutral=len(signals) - bullish - bearish,
        bias="bullish" if net >= BIAS_MARGIN else "bearish" if net <= -BIAS_MARGIN else "mixed",
        warnings=warnings,
    )
