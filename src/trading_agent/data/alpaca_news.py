"""Alpaca (Benzinga) news adapter implementing `NewsPort`.

Only headlines and summaries are kept. Article bodies are skipped: they are long, may contain
HTML, and are untrusted text that will later be shown to an LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import NewsArticle

MAX_HEADLINE_CHARS = 300
MAX_SUMMARY_CHARS = 1000


def clean_text(value: str | None, limit: int) -> str:
    """Collapse whitespace and cap the length, marking any cut with an ellipsis."""
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class AlpacaNews:
    def __init__(self, api_key: str, secret_key: str, *, client: Any | None = None) -> None:
        self._client: Any = client or NewsClient(api_key, secret_key)

    def get_news(
        self, symbols: Sequence[str], *, since: datetime, limit: int
    ) -> tuple[NewsArticle, ...]:
        request = NewsRequest(
            symbols=",".join(symbols) if symbols else None,
            start=since,
            limit=limit,
            sort="desc",
            include_content=False,
        )
        try:
            raw = self._client.get_news(request)
        except Exception as exc:
            raise MarketDataError(f"could not fetch news: {type(exc).__name__}: {exc}") from exc

        articles = [
            NewsArticle(
                id=str(item.id),
                headline=clean_text(item.headline, MAX_HEADLINE_CHARS),
                summary=clean_text(item.summary, MAX_SUMMARY_CHARS),
                source=item.source or "unknown",
                url=item.url or None,
                symbols=tuple(item.symbols or ()),
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in raw.data.get("news", [])
        ]
        articles.sort(key=lambda article: article.created_at, reverse=True)
        return tuple(articles[:limit])
