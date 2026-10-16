from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    FakeNews,
    healthy_market_data,
    make_article,
    make_quote,
)
from trading_agent.data.collector import MarketDataCollector
from trading_agent.domain.models import ActiveStock
from trading_agent.mcp_server.tools import (
    MAX_BARS,
    MAX_MOST_ACTIVE,
    MAX_NEWS_ARTICLES,
    MAX_NEWS_HOURS,
    MAX_SNAPSHOT_SYMBOLS,
    UNTRUSTED_NEWS_NOTICE,
    ResearchToolError,
    ResearchTools,
    ToolBudgetExceededError,
)
from trading_agent.settings.models import MarketDataConfig


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))


def _tools(
    market_data: FakeMarketData | None = None,
    news: FakeNews | None = None,
    *,
    max_calls: int = 100,
    logger: Any = None,
) -> ResearchTools:
    market_data = market_data or healthy_market_data("AAPL", "MSFT")
    news = news or FakeNews([make_article("1", ("AAPL",))])
    collector = MarketDataCollector(
        market_data, news, MarketDataConfig(), max_data_age_seconds=300, clock=lambda: NOW
    )
    return ResearchTools(
        market_data,
        news,
        collector,
        FakeBroker(market_open=False).get_clock,
        max_data_age_seconds=300,
        max_calls=max_calls,
        clock=lambda: NOW,
        logger=logger or _Log(),
    )


# ---------- Outputs ----------


def test_market_status() -> None:
    status = _tools().market_status()
    assert status == {
        "is_open": False,
        "timestamp": NOW.isoformat(),
        "next_open": NOW.isoformat(),
        "next_close": NOW.isoformat(),
    }


def test_get_quote_reports_prices_as_strings_and_freshness() -> None:
    quote = _tools().get_quote(" aapl ")
    assert quote["symbol"] == "AAPL"
    assert quote["bid"] == "100"  # never "1E+2"
    assert quote["ask"] == "100.1"
    assert quote["mid"] == "100.05"
    assert quote["spread_pct"] == "0.100"
    assert quote["age_seconds"] == 2
    assert quote["fresh"] is True
    assert quote["issues"] == []


def test_get_quote_flags_stale_data() -> None:
    market_data = healthy_market_data()
    market_data.quotes["AAPL"] = make_quote(age_seconds=900)
    quote = _tools(market_data).get_quote("AAPL")
    assert quote["fresh"] is False
    assert quote["issues"] == ["stale quote: 900s old (limit 300s)"]


def test_get_quote_without_data_is_an_error() -> None:
    with pytest.raises(ResearchToolError, match="no quote available for TSLA"):
        _tools().get_quote("TSLA")


def test_get_daily_bars_with_summary() -> None:
    result = _tools().get_daily_bars("AAPL", 3)
    assert [b["close"] for b in result["bars"]] == ["100", "102", "101"]
    assert result["bars"][0]["date"] == f"{NOW - timedelta(days=3):%Y-%m-%d}"
    assert result["summary"] == {
        "last_close": "101",
        "change_pct": "1.00",
        "avg_daily_volume": 2_000_000,
        "high": "103",
        "low": "99",
    }


def test_get_daily_bars_without_data_is_an_error() -> None:
    with pytest.raises(ResearchToolError, match="no daily bars available for TSLA"):
        _tools().get_daily_bars("TSLA", 5)


def test_get_news_is_labelled_untrusted() -> None:
    news = FakeNews([make_article("7", ("AAPL",), hours_ago=2)])
    result = _tools(news=news).get_news("AAPL", 12, 5)
    assert result["notice"] == UNTRUSTED_NEWS_NOTICE
    assert result["articles"][0]["headline"] == "Headline 7"
    assert news.calls[0] == {"symbols": ("AAPL",), "since": NOW - timedelta(hours=12), "limit": 5}


def test_get_asset() -> None:
    assert _tools().get_asset("MSFT") == {
        "symbol": "MSFT",
        "name": "MSFT Inc.",
        "exchange": "NASDAQ",
        "status": "active",
        "tradable": True,
        "fractionable": True,
    }


def test_get_most_active_skips_symbols_other_tools_would_reject() -> None:
    market_data = FakeMarketData(
        active=[
            ActiveStock(symbol="NVDA", volume=300, trade_count=30),
            ActiveStock(symbol="SPAC/WS", volume=200, trade_count=20),
            ActiveStock(symbol="F", volume=100, trade_count=10),
        ]
    )
    result = _tools(market_data).get_most_active(10)
    assert [s["symbol"] for s in result["stocks"]] == ["NVDA", "F"]
    assert result["stocks"][0] == {"symbol": "NVDA", "volume": 300, "trade_count": 30}


def test_research_snapshot_summarises_and_reports_rejections() -> None:
    result = _tools().research_snapshot(["AAPL", "MSFT", "TSLA"])
    assert result["notice"] == UNTRUSTED_NEWS_NOTICE
    assert [s["symbol"] for s in result["symbols"]] == ["AAPL", "MSFT"]
    aapl = result["symbols"][0]
    assert aapl["quote"] == {"bid": "100", "ask": "100.1"}
    assert aapl["change_pct"] == "1.00"
    assert aapl["headlines"] == ["Headline 1"]
    assert result["rejected"] == {"TSLA": "no daily bars"}


def test_research_snapshot_keeps_symbol_without_usable_quote() -> None:
    market_data = healthy_market_data()
    market_data.quotes.clear()
    result = _tools(market_data).research_snapshot(["AAPL"])
    assert result["symbols"][0]["quote"] is None


# ---------- Input validation ----------


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.get_quote("AAPL; rm -rf /"),
        lambda t: t.get_daily_bars("", 5),
        lambda t: t.get_news("../../etc", 24, 5),
        lambda t: t.get_asset("12345"),
        lambda t: t.research_snapshot(["AAPL", "not a ticker"]),
    ],
)
def test_invalid_symbols_are_rejected(call: Any) -> None:
    with pytest.raises(ResearchToolError, match="invalid symbol"):
        call(_tools())


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda t: t.get_daily_bars("AAPL", 0), "days must be between 1"),
        (lambda t: t.get_daily_bars("AAPL", MAX_BARS + 1), f"and {MAX_BARS}"),
        (lambda t: t.get_news("AAPL", MAX_NEWS_HOURS + 1, 5), "hours must be"),
        (lambda t: t.get_news("AAPL", 24, MAX_NEWS_ARTICLES + 1), "limit must be"),
        (lambda t: t.get_most_active(MAX_MOST_ACTIVE + 1), "limit must be"),
        (lambda t: t.get_most_active(0), "limit must be"),
        (lambda t: t.research_snapshot([]), "at least one symbol"),
        (
            lambda t: t.research_snapshot(["AAPL"] * (MAX_SNAPSHOT_SYMBOLS + 1)),
            f"at most {MAX_SNAPSHOT_SYMBOLS} symbols",
        ),
    ],
)
def test_counts_are_bounded(call: Any, message: str) -> None:
    with pytest.raises(ResearchToolError, match=message):
        call(_tools())


def test_rejected_input_never_reaches_the_provider() -> None:
    market_data = healthy_market_data()
    with pytest.raises(ResearchToolError):
        _tools(market_data).get_daily_bars("AAPL", 10_000)
    assert market_data.calls == []


# ---------- Errors, budget, and audit ----------


def test_provider_errors_become_tool_errors() -> None:
    with pytest.raises(ResearchToolError, match="could not fetch quotes: down"):
        _tools(FakeMarketData(fail=True)).get_quote("AAPL")


def test_budget_is_enforced() -> None:
    log = _Log()
    tools = _tools(max_calls=2, logger=log)
    tools.get_quote("AAPL")
    with pytest.raises(ResearchToolError):
        tools.get_quote("BAD SYMBOL")  # failed calls still count
    assert tools.calls_remaining == 0

    with pytest.raises(ToolBudgetExceededError, match="budget of 2 calls"):
        tools.market_status()
    assert tools.calls_used == 2
    assert log.events[-1] == ("tool.budget_exceeded", {"tool": "market_status", "max_calls": 2})


def test_every_call_is_audit_logged() -> None:
    log = _Log()
    tools = _tools(FakeMarketData(fail=True), logger=log)
    with pytest.raises(ResearchToolError):
        tools.get_quote("AAPL")
    with pytest.raises(ResearchToolError):
        tools.get_quote("NOT VALID")

    (first_event, first), (_, second) = log.events
    assert first_event == "tool.called"
    assert first["tool"] == "get_quote"
    assert first["args"] == {"symbol": "AAPL"}
    assert first["outcome"] == "data_error"
    assert first["calls_used"] == 1
    assert first["duration_ms"] >= 0
    assert second["outcome"] == "rejected"


def test_unexpected_errors_propagate_and_are_logged() -> None:
    def broken_clock() -> Any:
        raise RuntimeError("bug")

    log = _Log()
    tools = _tools(logger=log)
    tools._market_clock = broken_clock
    with pytest.raises(RuntimeError, match="bug"):
        tools.market_status()
    assert log.events[0][1]["outcome"] == "crash"
