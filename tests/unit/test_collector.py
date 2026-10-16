from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from tests.conftest import (
    NOW,
    FakeMarketData,
    FakeNews,
    healthy_market_data,
    make_article,
    make_quote,
)
from trading_agent.data.collector import MAX_SYMBOLS_PER_RUN, MarketDataCollector
from trading_agent.data.errors import MarketDataError
from trading_agent.settings.models import MarketDataConfig


def _collector(
    market_data: FakeMarketData, news: FakeNews | None = None, **config: Any
) -> MarketDataCollector:
    return MarketDataCollector(
        market_data,
        news or FakeNews(),
        MarketDataConfig(**config),
        max_data_age_seconds=300,
        clock=lambda: NOW,
    )


def test_happy_path_collects_everything() -> None:
    news = FakeNews([make_article("1", ("AAPL",)), make_article("2", ("MSFT",))])
    collector = _collector(healthy_market_data("AAPL", "MSFT"), news)

    snapshot = collector.collect(["aapl", "MSFT"], require_fresh_quotes=True)

    assert snapshot.collected_at == NOW
    assert snapshot.rejected == {}
    assert list(snapshot.symbols) == ["AAPL", "MSFT"]
    aapl = snapshot.symbols["AAPL"]
    assert aapl.quote is not None
    assert len(aapl.daily_bars) == 3
    assert [a.id for a in aapl.news] == ["1"]


def test_requests_use_config_values() -> None:
    market_data = healthy_market_data()
    news = FakeNews()
    collector = _collector(market_data, news, daily_bars=2, news_lookback_hours=12)

    snapshot = collector.collect(["AAPL"], require_fresh_quotes=False)

    assert len(snapshot.symbols["AAPL"].daily_bars) == 2
    assert news.calls[0]["since"] == NOW - timedelta(hours=12)
    assert news.calls[0]["limit"] == 50


def test_symbols_are_normalized_and_deduplicated() -> None:
    market_data = healthy_market_data()
    _collector(market_data).collect([" aapl", "AAPL", "Aapl"], require_fresh_quotes=False)
    assert ("quotes", ("AAPL",)) in market_data.calls


def test_invalid_symbol_is_rejected_before_any_call() -> None:
    market_data = healthy_market_data()
    with pytest.raises(ValueError, match="invalid symbol"):
        _collector(market_data).collect(["AAPL", "DROP TABLE"], require_fresh_quotes=False)
    assert market_data.calls == []


def test_empty_and_oversized_requests_are_rejected() -> None:
    collector = _collector(healthy_market_data())
    with pytest.raises(ValueError, match="at least one symbol"):
        collector.collect([], require_fresh_quotes=False)
    too_many = [f"A{i}" for i in range(MAX_SYMBOLS_PER_RUN + 1)]
    with pytest.raises(ValueError, match="too many symbols"):
        collector.collect(too_many, require_fresh_quotes=False)


def test_provider_outage_raises() -> None:
    with pytest.raises(MarketDataError, match="quotes"):
        _collector(FakeMarketData(fail=True)).collect(["AAPL"], require_fresh_quotes=False)
    with pytest.raises(MarketDataError, match="news"):
        _collector(healthy_market_data(), FakeNews(fail=True)).collect(
            ["AAPL"], require_fresh_quotes=False
        )


def test_stale_quote_rejects_symbol_when_freshness_required() -> None:
    market_data = healthy_market_data("AAPL", "MSFT")
    market_data.quotes["AAPL"] = make_quote("AAPL", age_seconds=600)

    snapshot = _collector(market_data).collect(["AAPL", "MSFT"], require_fresh_quotes=True)

    assert list(snapshot.symbols) == ["MSFT"]
    assert "stale quote" in snapshot.rejected["AAPL"]


def test_stale_quote_is_kept_for_pre_market_research() -> None:
    market_data = healthy_market_data()
    market_data.quotes["AAPL"] = make_quote("AAPL", age_seconds=50_000)
    snapshot = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)
    assert snapshot.symbols["AAPL"].quote is not None


def test_bad_quote_is_dropped_for_pre_market_research() -> None:
    market_data = healthy_market_data()
    market_data.quotes["AAPL"] = make_quote("AAPL", bid="0")
    snapshot = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)
    assert snapshot.symbols["AAPL"].quote is None
    assert snapshot.rejected == {}


def test_missing_quote_rejects_only_when_freshness_required() -> None:
    market_data = healthy_market_data()
    del market_data.quotes["AAPL"]

    strict = _collector(market_data).collect(["AAPL"], require_fresh_quotes=True)
    relaxed = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)

    assert strict.rejected == {"AAPL": "no quote available"}
    assert relaxed.symbols["AAPL"].quote is None


def test_missing_bars_reject_symbol() -> None:
    market_data = healthy_market_data()
    del market_data.bars["AAPL"]
    snapshot = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)
    assert snapshot.rejected == {"AAPL": "no daily bars"}


def test_inconsistent_bars_reject_symbol() -> None:
    market_data = healthy_market_data()
    first, *rest = market_data.bars["AAPL"]
    market_data.bars["AAPL"] = (first.model_copy(update={"high": first.low - 1}), *rest)
    snapshot = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)
    assert "inconsistent high/low" in snapshot.rejected["AAPL"]


def test_unknown_asset_rejects_symbol() -> None:
    market_data = healthy_market_data()
    del market_data.assets["AAPL"]
    snapshot = _collector(market_data).collect(["AAPL"], require_fresh_quotes=False)
    assert "asset lookup failed" in snapshot.rejected["AAPL"]


def test_news_is_capped_per_symbol() -> None:
    news = FakeNews([make_article(str(i), ("AAPL", "MSFT")) for i in range(8)])
    collector = _collector(healthy_market_data("AAPL", "MSFT"), news, news_per_symbol=3)

    snapshot = collector.collect(["AAPL", "MSFT"], require_fresh_quotes=False)

    assert [a.id for a in snapshot.symbols["AAPL"].news] == ["0", "1", "2"]
    assert len(snapshot.symbols["MSFT"].news) == 3


def test_rejections_are_logged() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def warning(self, event: str, **kw: Any) -> None:
            events.append((event, kw))

        def info(self, event: str, **kw: Any) -> None:
            events.append((event, kw))

    market_data = healthy_market_data()
    del market_data.bars["AAPL"]
    collector = MarketDataCollector(
        market_data,
        FakeNews(),
        MarketDataConfig(),
        max_data_age_seconds=300,
        clock=lambda: NOW,
        logger=_Log(),
    )
    collector.collect(["AAPL"], require_fresh_quotes=False)

    assert events[0] == (
        "market_data.symbol_rejected",
        {"symbol": "AAPL", "reason": "no daily bars"},
    )
    assert events[1][0] == "market_data.collected"
    assert events[1][1]["rejected"] == 1
