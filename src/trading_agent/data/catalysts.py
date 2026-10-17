"""Rule-based catalyst tags for news (ADR 0016).

Deterministic keyword rules that say what kind of news an article is: earnings, an analyst
action, an FDA decision, a deal, and so on. They never guess sentiment: "beats estimates but
cuts guidance" is both good and bad, and keyword sentiment gets that wrong too often.
"""

from __future__ import annotations

import re

# Order matters: tags are returned in this order.
_RULES: tuple[tuple[str, str], ...] = (
    ("earnings", r"earnings|quarterly results|q[1-4] results|eps\b|revenue (?:beat|miss)"),
    ("guidance", r"guidance|outlook|forecast"),
    ("analyst", r"upgrade[sd]?|downgrade[sd]?|price target|initiates? coverage|reiterates?"),
    ("fda", r"fda|clinical trial|phase (?:1|2|3|i|ii|iii)\b"),
    ("merger", r"acquire[sd]?|acquisition|merger|buyout|takeover|to be acquired"),
    ("offering", r"offering|dilution|share sale|convertible notes"),
    ("legal", r"lawsuit|investigation|probe|\bsec\b|settlement|subpoena"),
    ("buyback", r"buyback|repurchase"),
    ("dividend", r"dividend"),
    ("management", r"\bceo\b|\bcfo\b|steps down|resigns?|appoints?|appointed"),
    ("bankruptcy", r"bankruptcy|chapter 11|insolven"),
    ("split", r"stock split|\d+-for-\d+ (?:stock )?split|reverse split"),
    ("contract", r"contract|awarded|wins? (?:a |an )?(?:\$[\d.,]+[bmk]? )?deal"),
    ("product", r"launch(?:es|ed)?|unveils?|introduces?|new product"),
)
_COMPILED = tuple(
    (tag, re.compile(rf"(?<![a-z])(?:{pattern})", re.IGNORECASE)) for tag, pattern in _RULES
)

TAGS: tuple[str, ...] = tuple(tag for tag, _ in _RULES)
# Catalysts that often move a stock sharply on their own.
MAJOR: tuple[str, ...] = ("earnings", "guidance", "fda", "merger", "offering", "bankruptcy")


def catalyst_tags(headline: str, summary: str = "") -> tuple[str, ...]:
    """Every catalyst the headline or summary mentions, in `TAGS` order, without repeats."""
    text = f"{headline}\n{summary}"
    return tuple(tag for tag, pattern in _COMPILED if pattern.search(text))
