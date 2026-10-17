"""Rule-based catalyst tags for news (ADR 0016): deterministic, never sentiment."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trading_agent.data.catalysts import MAJOR, TAGS, catalyst_tags


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        ("Apple Q4 Earnings Beat Estimates; Revenue Up 8%", ("earnings",)),
        ("Nvidia Raises Full-Year Guidance", ("guidance",)),
        ("Morgan Stanley Upgrades Tesla to Overweight", ("analyst",)),
        ("Goldman Downgrades Intel, Cuts Price Target to $20", ("analyst",)),
        ("FDA Approves Pfizer's New Drug", ("fda",)),
        ("Microsoft to Acquire Gaming Studio in $2B Deal", ("merger",)),
        ("Company Announces $500M Public Offering of Common Stock", ("offering",)),
        ("SEC Opens Investigation Into Accounting Practices", ("legal",)),
        ("Board Approves $10B Share Buyback", ("buyback",)),
        ("Company Declares Quarterly Dividend of $0.25", ("dividend",)),
        ("CEO Steps Down Effective Immediately", ("management",)),
        ("Retailer Files for Chapter 11 Bankruptcy", ("bankruptcy",)),
        ("Announces 4-for-1 Stock Split", ("split",)),
        ("Wins $1.2B Defense Contract", ("contract",)),
        ("Unveils New AI Chip at Developer Conference", ("product",)),
        ("Shares trade higher in a quiet session", ()),
    ],
)
def test_headlines_get_their_catalyst(headline: str, expected: tuple[str, ...]) -> None:
    assert catalyst_tags(headline) == expected


def test_summaries_count_and_tags_are_ordered_and_unique() -> None:
    tags = catalyst_tags(
        "Q3 earnings beat; analyst upgrade follows", "The company also raised guidance."
    )
    assert tags == ("earnings", "guidance", "analyst")
    assert list(tags) == [t for t in TAGS if t in tags]


def test_major_catalysts_are_a_subset_of_tags() -> None:
    assert set(MAJOR) <= set(TAGS)
    assert {"earnings", "fda", "merger", "offering", "bankruptcy", "guidance"} <= set(MAJOR)


def test_words_inside_other_words_do_not_match() -> None:
    assert catalyst_tags("Learnings from the conference") == ()
    assert catalyst_tags("The split-screen app update") == ()


@given(text=st.text(max_size=400))
def test_any_text_is_safe_and_deterministic(text: str) -> None:
    first = catalyst_tags(text, text)
    assert first == catalyst_tags(text, text)
    assert set(first) <= set(TAGS)
    assert len(first) == len(set(first))
