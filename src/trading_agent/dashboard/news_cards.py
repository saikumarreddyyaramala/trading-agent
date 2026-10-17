"""News cards (ADR 0018): the key facts of an article, by rule, never a model's summary.

A card holds the headline, the first sentence of the summary (HTML removed), the catalyst
tags (ADR 0016), the symbols, how old it is, and the symbol's move today. Its importance
says why it is near the top: a major catalyst, a symbol the system cares about, a big move.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from trading_agent.data.catalysts import MAJOR, catalyst_tags
from trading_agent.domain.models import NewsArticle

HEADLINE_CHARS = 140
FACT_CHARS = 200
MAX_SYMBOLS = 5
BIG_MOVE_PCT = Decimal(3)
_TAGS = re.compile(r"<[^>]*>")
# A sentence ends at . ! or ? after a lowercase letter, digit, or closing mark, followed by a
# space or glued onto a capital ("grows.ASML"). Initials like "U.S." do not end one.
_SENTENCE = re.compile(r"(?<=[a-z0-9)%\"'][.!?])(?:\s|(?=[A-Z]))")


class NewsCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    headline: str
    key_fact: str
    source: str
    url: str | None
    symbols: tuple[str, ...]
    published_at: datetime
    age: str
    catalysts: tuple[str, ...]
    major: bool
    move_pct: Decimal | None  # the first symbol's move today
    watched: bool  # held, planned, traded, or proposed
    importance: int  # 0-5: major +2, watched +2, big move +1


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def age_text(age: timedelta) -> str:
    seconds = age.total_seconds()
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def news_card(
    article: NewsArticle,
    now: datetime,
    *,
    moves: Mapping[str, Decimal],
    watched: set[str],
) -> NewsCard:
    text = _TAGS.sub(" ", article.summary)
    first = _SENTENCE.split(" ".join(text.split()), maxsplit=1)[0] if text.strip() else ""
    symbols = article.symbols[:MAX_SYMBOLS]
    tags = catalyst_tags(article.headline, article.summary)
    major = any(t in MAJOR for t in tags)
    move = next((moves[s] for s in symbols if s in moves), None)
    is_watched = any(s in watched for s in symbols)
    importance = (
        (2 if major else 0)
        + (2 if is_watched else 0)
        + (1 if move is not None and abs(move) >= BIG_MOVE_PCT else 0)
    )
    return NewsCard(
        id=article.id,
        headline=_clip(article.headline, HEADLINE_CHARS),
        key_fact=_clip(first, FACT_CHARS),
        source=article.source,
        url=article.url,
        symbols=symbols,
        published_at=article.created_at,
        age=age_text(now - article.created_at),
        catalysts=tags,
        major=major,
        move_pct=move,
        watched=is_watched,
        importance=importance,
    )


def build_cards(
    articles: Iterable[NewsArticle],
    now: datetime,
    *,
    moves: Mapping[str, Decimal],
    watched: set[str],
    limit: int,
) -> list[NewsCard]:
    unique: dict[str, NewsArticle] = {}
    for article in articles:
        unique.setdefault(article.id, article)
    cards = [news_card(a, now, moves=moves, watched=watched) for a in unique.values()]
    cards.sort(key=lambda c: (-c.importance, -c.published_at.timestamp()))
    return cards[:limit]
