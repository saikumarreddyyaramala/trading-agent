"""The technical report (ADR 0020): readings from constructed markets with known answers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from trading_agent.analysis.technicals import technical_report
from trading_agent.domain.models import Bar

NY = ZoneInfo("America/New_York")
DAY0 = datetime(2026, 1, 2, 21, tzinfo=UTC)


def daily(
    closes: list[float],
    *,
    spread: float = 0.01,
    volumes: list[int] | None = None,
    symbol: str = "AAPL",
    opens: list[float] | None = None,
) -> list[Bar]:
    vols = volumes or [1_000_000] * len(closes)
    bars = []
    for i, close in enumerate(closes):
        open_ = opens[i] if opens else (closes[i - 1] if i else close)
        high = max(open_, close) * (1 + spread)
        low = min(open_, close) * (1 - spread)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=DAY0 + timedelta(days=i),
                open=Decimal(str(round(open_, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(close, 4))),
                volume=vols[i],
            )
        )
    return bars


def session(prices: list[float], *, volume: int = 10_000, symbol: str = "AAPL") -> list[Bar]:
    """5-minute bars from 9:30 New York today, one per price."""
    start = datetime(2026, 10, 15, 9, 30, tzinfo=NY)
    return [
        Bar(
            symbol=symbol,
            timestamp=start + timedelta(minutes=5 * i),
            open=Decimal(str(p)),
            high=Decimal(str(p * 1.001)),
            low=Decimal(str(p * 0.999)),
            close=Decimal(str(p)),
            volume=volume,
        )
        for i, p in enumerate(prices)
    ]


def uptrend(n: int = 260, start: float = 50, step: float = 0.3) -> list[float]:
    # Rising with small pullbacks every 4th day, so oscillators are not pinned at 100.
    out, price = [], start
    for i in range(n):
        price += -step if i % 4 == 3 else step
        out.append(price)
    return out


def names(report: Any) -> set[str]:
    return {r.name for r in report.signals}


def reading(report: Any, name: str) -> Any:
    return next(r for r in report.signals if r.name == name)


# ---------- Trend ----------


def test_a_steady_uptrend_reads_bullish() -> None:
    report = technical_report("AAPL", daily(uptrend()), now=DAY0 + timedelta(days=260))
    assert report.bias == "bullish"
    assert reading(report, "200-day average").signal == "bullish"
    assert reading(report, "MACD").signal == "bullish"
    assert reading(report, "ADX").signal == "bullish"
    assert report.trend["sma_200"] is not None
    assert report.bullish > report.bearish


def test_a_steady_downtrend_reads_bearish() -> None:
    closes = list(reversed(uptrend()))
    report = technical_report("AAPL", daily(closes), now=DAY0 + timedelta(days=260))
    assert report.bias == "bearish"
    assert reading(report, "50-day average").signal == "bearish"


def test_a_golden_cross_is_spotted() -> None:
    closes = [100.0 - 0.2 * i for i in range(200)] + [60.0 + 1.2 * i for i in range(60)]
    report = technical_report("AAPL", daily(closes), now=DAY0 + timedelta(days=260))
    assert "golden cross" in reading(report, "50/200-day cross").note


# ---------- Momentum ----------


def test_overbought_is_called_out() -> None:
    closes = [100.0] * 40 + [100.0 + 2 * i for i in range(1, 16)]
    report = technical_report("AAPL", daily(closes), now=DAY0 + timedelta(days=60))
    rsi = reading(report, "RSI")
    assert report.momentum["rsi_14"] > 70
    assert rsi.signal == "bearish"
    assert "overbought" in rsi.note


# ---------- Volume, breakouts, patterns ----------


def test_a_breakout_on_heavy_volume() -> None:
    closes = [100.0 + (0.5 if i % 2 else -0.5) for i in range(59)] + [104.0]
    volumes = [1_000_000] * 59 + [3_000_000]
    report = technical_report("AAPL", daily(closes, volumes=volumes), now=DAY0 + timedelta(days=60))
    assert report.volume["relative_volume"] == pytest.approx(3, rel=0.1)
    assert reading(report, "Breakout").signal == "bullish"
    assert "Volume" in names(report)


def test_candlestick_patterns_of_the_last_day() -> None:
    closes = [100.0] * 40 + [99.0, 101.0]
    opens = [100.0] * 40 + [100.5, 98.8]  # a red day, then a green day engulfing it
    report = technical_report(
        "AAPL", daily(closes, opens=opens, spread=0.001), now=DAY0 + timedelta(days=42)
    )
    assert "bullish_engulfing" in report.patterns["candles"]


# ---------- Intraday ----------


def test_intraday_vwap_and_opening_range() -> None:
    prices = [100.0, 100.4, 100.2, 100.6, 100.5, 100.8] + [101.0 + 0.1 * i for i in range(10)]
    report = technical_report(
        "AAPL",
        daily(uptrend()),
        intraday=session(prices),
        now=datetime(2026, 10, 15, 11, 0, tzinfo=NY),
    )
    assert report.intraday is not None
    assert report.intraday["vwap"] < report.price
    assert report.intraday["opening_range_high"] == pytest.approx(100.8 * 1.001, rel=1e-6)
    assert reading(report, "VWAP").signal == "bullish"
    assert reading(report, "Opening range").signal == "bullish"


def test_no_intraday_bars_is_said_plainly() -> None:
    report = technical_report("AAPL", daily(uptrend()), now=DAY0 + timedelta(days=260))
    assert report.intraday is None
    assert "no intraday bars yet today" in report.warnings


# ---------- Relative strength ----------


def test_strength_against_the_market() -> None:
    spy = daily([400.0] * 260, symbol="SPY")
    report = technical_report(
        "AAPL", daily(uptrend()), benchmarks={"SPY": spy}, now=DAY0 + timedelta(days=260)
    )
    assert report.relative_strength["vs_SPY_20d"] > 0
    assert reading(report, "vs SPY").signal == "bullish"


# ---------- Short history and bad input ----------


def test_short_history_skips_what_it_cannot_measure() -> None:
    report = technical_report("AAPL", daily(uptrend(40)), now=DAY0 + timedelta(days=40))
    assert report.trend["sma_200"] is None
    assert "200-day average" not in names(report)
    assert any("200-day" in w for w in report.warnings)


def test_no_bars_is_an_error() -> None:
    with pytest.raises(ValueError, match="no daily bars"):
        technical_report("AAPL", [], now=DAY0)


def test_the_report_is_json_ready_and_rounded() -> None:
    report = technical_report("AAPL", daily(uptrend()), now=DAY0 + timedelta(days=260))
    data = report.model_dump(mode="json")
    assert data["symbol"] == "AAPL"
    assert isinstance(data["price"], str)
    rsi = data["momentum"]["rsi_14"]
    assert rsi == round(rsi, 2)
    assert set(data["levels"]) >= {"pivots", "supports", "resistances", "high_52w", "low_52w"}


def test_heavy_selling_reads_bearish_in_plain_english() -> None:
    closes = [100.0] * 59 + [97.0]
    opens = [100.0] * 59 + [100.0]
    volumes = [1_000_000] * 59 + [2_000_000]
    report = technical_report(
        "AAPL", daily(closes, opens=opens, volumes=volumes), now=DAY0 + timedelta(days=60)
    )
    volume = reading(report, "Volume")
    assert volume.signal == "bearish"
    assert volume.note == "heavy volume (2.0x average) on a down day"
