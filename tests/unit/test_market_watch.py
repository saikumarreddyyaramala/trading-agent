"""Pre-market and after-hours watch (ADR 0016): deterministic scans of what matters next."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FakeMarketData, FakeNews
from trading_agent.domain.models import Mover, NewsArticle, Snapshot
from trading_agent.settings.models import AppConfig
from trading_agent.watch import MarketWatch

NY = ZoneInfo("America/New_York")
DAY = date(2026, 10, 15)


def ny(hour: int, minute: int = 0, day: date = DAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=NY)


def snap(
    symbol: str,
    last: str | None,
    last_at: datetime | None,
    close: str | None,
    close_at: datetime | None,
    previous: str | None = None,
) -> Snapshot:
    return Snapshot(
        symbol=symbol,
        last_price=None if last is None else Decimal(last),
        last_trade_at=last_at,
        session_close=None if close is None else Decimal(close),
        session_at=close_at,
        previous_close=None if previous is None else Decimal(previous),
        session_volume=1_000,
    )


def article(symbol: str, headline: str, hours_ago: float, now: datetime) -> NewsArticle:
    at = now - timedelta(hours=hours_ago)
    return NewsArticle(
        id=f"{symbol}-{hours_ago}",
        headline=headline,
        summary="",
        source="benzinga",
        url=None,
        symbols=(symbol,),
        created_at=at,
        updated_at=at,
    )


YESTERDAY_CLOSE = ny(16, 0, DAY - timedelta(days=1))


def _pre_market(**changes: Any) -> FakeMarketData:
    values: dict[str, Any] = {
        "snapshots": {
            # Yesterday's session closed at 100; it trades at 104 before today's open.
            "AAPL": snap("AAPL", "104", ny(7, 55), "100", YESTERDAY_CLOSE, "98"),
            # A bar for today already exists: the reference is still the previous close.
            "MSFT": snap("MSFT", "402", ny(7, 50), "410", ny(4, 0), "400"),
            # Last trade was yesterday evening: nothing yet this morning.
            "IBM": snap("IBM", "150", ny(19, 0, DAY - timedelta(days=1)), "149", YESTERDAY_CLOSE),
            "SPY": snap("SPY", "601.80", ny(7, 59), "600", YESTERDAY_CLOSE),
            "UP": snap("UP", "13", ny(7, 40), "12", YESTERDAY_CLOSE),
        },
        "movers": (
            Mover(symbol="UP", price=Decimal("12"), change_pct=Decimal("20"), direction="gainer"),
            Mover(
                symbol="PENNY", price=Decimal("0.5"), change_pct=Decimal("80"), direction="gainer"
            ),
            Mover(symbol="AAPL", price=Decimal("104"), change_pct=Decimal("4"), direction="gainer"),
        ),
    }
    values.update(changes)
    return FakeMarketData(**values)


def _watch(
    market_data: FakeMarketData | None = None,
    news: FakeNews | None = None,
    *,
    now: datetime | None = None,
    config: AppConfig | None = None,
) -> MarketWatch:
    return MarketWatch(
        market_data or _pre_market(),
        news or FakeNews(),
        config or AppConfig(),
        clock=lambda: now or ny(8, 0),
    )


FOCUS = {"AAPL": ["plan"], "MSFT": ["held"], "IBM": ["proposed"]}


# ---------- Pre-market ----------


def test_pre_market_moves_are_measured_from_the_last_regular_close() -> None:
    report = _watch().scan("pre", DAY, FOCUS)
    items = {i.symbol: i for i in report.items}
    assert items["AAPL"].reference_close == Decimal("100")
    assert items["AAPL"].move_pct == Decimal("4.00")
    assert items["MSFT"].reference_close == Decimal("400")  # not today's partial bar
    assert items["MSFT"].move_pct == Decimal("0.50")
    assert items["IBM"].move_pct is None
    assert items["IBM"].stale
    assert report.market is not None
    assert report.market.symbol == "SPY"
    assert report.market.move_pct == Decimal("0.30")


def test_big_moves_raise_alerts_and_sort_first() -> None:
    report = _watch().scan("pre", DAY, FOCUS)
    assert report.items[0].symbol == "UP"  # +8.33%, the biggest move
    assert [i.symbol for i in report.items if i.alert] == ["UP", "AAPL"]
    assert report.alerts == (
        "UP +8.33% pre-market (gainer)",
        "AAPL +4.00% pre-market (plan, gainer)",
    )


def test_reasons_merge_and_movers_are_filtered_by_price() -> None:
    report = _watch().scan("pre", DAY, FOCUS)
    reasons = {i.symbol: i.reasons for i in report.items}
    assert reasons["AAPL"] == ("plan", "gainer")
    assert "PENNY" not in reasons  # below the $5 minimum price
    assert reasons["UP"] == ("gainer",)


def test_news_gives_counts_catalysts_and_short_headlines() -> None:
    now = ny(8, 0)
    news = FakeNews(
        [
            article("IBM", "IBM Q3 Earnings Beat Estimates " + "x" * 300, 2, now),
            article("IBM", "Analyst Upgrades IBM", 3, now),
            article("MSFT", "Microsoft Unveils New Chip", 1, now),
        ]
    )
    report = _watch(news=news).scan("pre", DAY, FOCUS)
    ibm = next(i for i in report.items if i.symbol == "IBM")
    assert ibm.news_count == 2
    assert ibm.catalysts == ("earnings", "analyst")
    assert len(ibm.headlines[0]) <= 140
    assert ibm.alert  # a major catalyst, even with no trades yet
    assert "IBM: earnings, analyst news (2 articles), no pre-market trades yet" in report.alerts
    assert news.calls[0]["since"] == now - timedelta(hours=16)


def test_the_symbol_list_is_capped_but_focus_always_stays() -> None:
    config = AppConfig.model_validate({"watch": {"max_symbols": 3}})
    report = _watch(config=config).scan("pre", DAY, FOCUS)
    assert [i.symbol for i in report.items if "gainer" not in i.reasons] == ["MSFT", "IBM"]
    assert len(report.items) == 3


# ---------- After hours ----------


def test_after_hours_moves_are_measured_from_todays_close() -> None:
    market = FakeMarketData(
        snapshots={
            "NVDA": snap("NVDA", "94", ny(16, 40), "100", ny(9, 30), "99"),
            "OLD": snap("OLD", "50", ny(16, 30), "49", YESTERDAY_CLOSE),  # no session today
            "SPY": snap("SPY", "600", ny(16, 44), "600", ny(9, 30)),
        }
    )
    report = _watch(market, now=ny(16, 45)).scan("post", DAY, {"NVDA": ["traded"], "OLD": ["held"]})
    items = {i.symbol: i for i in report.items}
    assert items["NVDA"].reference_close == Decimal("100")
    assert items["NVDA"].move_pct == Decimal("-6.00")
    assert "NVDA -6.00% after hours (traded)" in report.alerts
    assert items["OLD"].reference_close is None
    assert items["OLD"].move_pct is None


# ---------- Failures are contained ----------


def test_snapshot_outage_is_an_error_not_a_crash() -> None:
    report = _watch(_pre_market(fail_snapshots=True)).scan("pre", DAY, FOCUS)
    assert report.errors == ("snapshots unavailable: could not fetch snapshots: down",)
    assert all(i.move_pct is None for i in report.items)


def test_movers_and_news_outages_still_scan_the_focus() -> None:
    report = _watch(_pre_market(fail_movers=True), FakeNews(fail=True)).scan("pre", DAY, FOCUS)
    assert "movers unavailable: could not fetch market movers: down" in report.errors
    assert "news unavailable: could not fetch news: down" in report.errors
    assert {i.symbol for i in report.items} == {"AAPL", "MSFT", "IBM"}


def test_disabled_watch_scans_nothing() -> None:
    market = _pre_market()
    report = _watch(market, config=AppConfig.model_validate({"watch": {"enabled": False}})).scan(
        "pre", DAY, FOCUS
    )
    assert report.items == ()
    assert market.calls == []


def test_watch_config_is_validated() -> None:
    with pytest.raises(ValueError, match="watch"):
        AppConfig.model_validate({"watch": {"gap_alert_pct": 0}})
    with pytest.raises(ValueError, match="watch"):
        AppConfig.model_validate({"watch": {"max_symbols": 0}})


def test_report_round_trips_as_json() -> None:
    from trading_agent.watch import WatchReport

    report = _watch().scan("pre", DAY, FOCUS)
    assert WatchReport.model_validate_json(report.model_dump_json()) == report


def test_a_tiny_fall_rounds_to_zero_without_a_minus_sign() -> None:
    from trading_agent.watch.market_watch import _pct

    assert str(_pct(Decimal("-0.001"))) == "0.00"  # was "-0.00%" in the journal and alerts
    assert str(_pct(Decimal("-0.006"))) == "-0.01"
