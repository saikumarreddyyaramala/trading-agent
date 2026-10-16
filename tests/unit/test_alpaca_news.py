from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import NOW
from trading_agent.data.alpaca_news import (
    MAX_HEADLINE_CHARS,
    MAX_SUMMARY_CHARS,
    AlpacaNews,
    clean_text,
)
from trading_agent.data.errors import MarketDataError


def _item(article_id: int, hours_ago: float, **overrides: Any) -> Any:
    created = NOW - timedelta(hours=hours_ago)
    values: dict[str, Any] = {
        "id": article_id,
        "headline": f"  Headline\n {article_id}  ",
        "summary": "Short   summary.",
        "source": "benzinga",
        "url": f"https://example.com/{article_id}",
        "symbols": ["AAPL"],
        "created_at": created,
        "updated_at": created,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Client:
    def __init__(self, items: list[Any] | None = None, *, fail: bool = False) -> None:
        self.items = items or []
        self.fail = fail
        self.requests: list[Any] = []

    def get_news(self, request: Any) -> Any:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(data={"news": self.items})


def test_articles_are_translated_and_cleaned() -> None:
    client = _Client([_item(7, 1, url="", source=None, symbols=None)])
    article = AlpacaNews("k", "s", client=client).get_news(["AAPL"], since=NOW, limit=10)[0]
    assert article.id == "7"
    assert article.headline == "Headline 7"
    assert article.summary == "Short summary."
    assert article.url is None
    assert article.source == "unknown"
    assert article.symbols == ()


def test_request_parameters() -> None:
    client = _Client()
    since = NOW - timedelta(hours=24)
    AlpacaNews("k", "s", client=client).get_news(["AAPL", "MSFT"], since=since, limit=25)
    request = client.requests[0]
    assert request.symbols == "AAPL,MSFT"
    assert request.limit == 25
    assert request.include_content is False
    assert request.start is not None


def test_empty_symbols_requests_all_news() -> None:
    client = _Client()
    AlpacaNews("k", "s", client=client).get_news([], since=NOW, limit=1)
    assert client.requests[0].symbols is None


def test_results_are_newest_first_and_capped() -> None:
    client = _Client([_item(1, 5), _item(2, 1), _item(3, 3)])
    articles = AlpacaNews("k", "s", client=client).get_news(["AAPL"], since=NOW, limit=2)
    assert [a.id for a in articles] == ["2", "3"]


def test_errors_are_wrapped() -> None:
    with pytest.raises(MarketDataError, match="could not fetch news"):
        AlpacaNews("k", "s", client=_Client(fail=True)).get_news(["AAPL"], since=NOW, limit=1)


def test_clean_text_caps_length() -> None:
    long_text = "word " * 1000
    assert len(clean_text(long_text, MAX_SUMMARY_CHARS)) == MAX_SUMMARY_CHARS
    assert clean_text(long_text, MAX_HEADLINE_CHARS).endswith("…")
    assert clean_text(None, 10) == ""
    assert clean_text("exactly", 7) == "exactly"
