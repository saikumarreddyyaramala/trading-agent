"""News cards (ADR 0018): only the key facts, by rule, newest and most important first."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import NOW
from trading_agent.dashboard.news_cards import age_text, build_cards, news_card
from trading_agent.domain.models import NewsArticle


def article(
    aid: str = "1",
    headline: str = "Apple Q4 Earnings Beat Estimates",
    summary: str = "<p>Revenue rose 8%. Margins improved. More text.</p>",
    symbols: tuple[str, ...] = ("AAPL",),
    minutes_ago: float = 30,
) -> NewsArticle:
    at = NOW - timedelta(minutes=minutes_ago)
    return NewsArticle(
        id=aid,
        headline=headline,
        summary=summary,
        source="benzinga",
        url=f"https://example.com/{aid}",
        symbols=symbols,
        created_at=at,
        updated_at=at,
    )


def test_a_card_keeps_only_the_key_facts() -> None:
    card = news_card(article(), NOW, moves={"AAPL": Decimal("3.20")}, watched={"AAPL"})
    assert card.headline == "Apple Q4 Earnings Beat Estimates"
    assert card.key_fact == "Revenue rose 8%."  # first sentence, HTML removed
    assert card.catalysts == ("earnings",)
    assert card.major
    assert card.symbols == ("AAPL",)
    assert card.move_pct == Decimal("3.20")
    assert card.watched
    assert card.age == "30m"
    assert card.importance == 5  # major 2 + watched 2 + big move 1


def test_cards_are_safe_with_messy_text() -> None:
    card = news_card(
        article(headline="  " + "x" * 300, summary="", symbols=tuple(f"S{i}" for i in range(9))),
        NOW,
        moves={},
        watched=set(),
    )
    assert len(card.headline) <= 140
    assert card.key_fact == ""
    assert len(card.symbols) == 5
    assert card.move_pct is None
    assert card.importance == 0


def test_build_cards_dedupes_and_sorts_by_importance_then_time() -> None:
    articles = [
        article("1", "Quiet market update", "", ("SPY",), 5),
        article("2", "FDA Approves New Drug", "", ("PFE",), 60),
        article("2", "FDA Approves New Drug", "", ("PFE",), 60),  # duplicate id
        article("3", "Another quiet update", "", ("IBM",), 1),
    ]
    cards = build_cards(articles, NOW, moves={}, watched=set(), limit=10)
    assert [c.id for c in cards] == ["2", "3", "1"]
    assert len(build_cards(articles, NOW, moves={}, watched=set(), limit=2)) == 2


def test_age_text() -> None:
    assert age_text(timedelta(seconds=20)) == "now"
    assert age_text(timedelta(minutes=59)) == "59m"
    assert age_text(timedelta(hours=5, minutes=10)) == "5h"
    assert age_text(timedelta(days=3)) == "3d"
    assert age_text(timedelta(minutes=-5)) == "now"  # clock skew


@given(headline=st.text(max_size=500), summary=st.text(max_size=800))
def test_any_text_makes_a_bounded_card(headline: str, summary: str) -> None:
    card = news_card(
        article(headline=headline or "x", summary=summary), NOW, moves={}, watched=set()
    )
    assert len(card.headline) <= 140
    assert len(card.key_fact) <= 200
    assert "<" not in card.key_fact or ">" not in card.key_fact
