"""Data quality and freshness rules.

Each rule returns a list of human-readable issues; an empty list means the data is usable.
Callers treat any issue as "do not use this data" (fail safe). The same rules are reused
by the risk engine before an order (Phase 4).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from trading_agent.domain.models import Bar, Quote

# Tolerated difference between our clock and the vendor's before a timestamp counts as bogus.
MAX_CLOCK_SKEW_SECONDS = 5

# US equity tickers: letters, digits, and a share-class separator (BRK.B, BF-B).
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z]{1,2})?$")


def normalize_symbol(raw: str) -> str:
    """Upper-case and validate a ticker. Rejects anything that is not a plausible US symbol."""
    symbol = raw.strip().upper()
    if not _SYMBOL.match(symbol):
        raise ValueError(f"invalid symbol: {raw!r}")
    return symbol


def quote_issues(quote: Quote, *, now: datetime, max_age_seconds: int | None) -> list[str]:
    """Problems that make a quote unusable. `max_age_seconds=None` skips the freshness rule."""
    issues: list[str] = []
    if quote.bid_price <= 0 or quote.ask_price <= 0:
        issues.append("one-sided or empty quote")
    elif quote.ask_price < quote.bid_price:
        issues.append("crossed quote (ask below bid)")

    age = quote.age_seconds(now)
    if age < -MAX_CLOCK_SKEW_SECONDS:
        issues.append("quote timestamp is in the future")
    elif max_age_seconds is not None and age > max_age_seconds:
        issues.append(f"stale quote: {age:.0f}s old (limit {max_age_seconds}s)")
    return issues


def bar_issues(bars: Sequence[Bar]) -> list[str]:
    """Problems that make a bar series unusable."""
    if not bars:
        return ["no daily bars"]
    issues: list[str] = []
    timestamps = [bar.timestamp for bar in bars]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        issues.append("bars are not in strictly ascending time order")
    for bar in bars:
        day = f"{bar.timestamp:%Y-%m-%d}"
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            issues.append(f"non-positive price on {day}")
        elif bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close):
            issues.append(f"inconsistent high/low on {day}")
        if bar.volume < 0:
            issues.append(f"negative volume on {day}")
    return issues
