from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import NOW, make_bars, make_quote
from trading_agent.data.quality import bar_issues, normalize_symbol, quote_issues

# ---------- Symbols ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("AAPL", "AAPL"), (" msft ", "MSFT"), ("BRK.B", "BRK.B"), ("BF-B", "BF-B"), ("F", "F")],
)
def test_valid_symbols_are_normalized(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "  ", "1ABC", "TOOLONGX", "AAPL;DROP", "A B", "AAPL.", "../etc", "$SPY"]
)
def test_invalid_symbols_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid symbol"):
        normalize_symbol(raw)


# ---------- Quotes ----------


def test_fresh_two_sided_quote_has_no_issues() -> None:
    assert quote_issues(make_quote(age_seconds=10), now=NOW, max_age_seconds=300) == []


def test_stale_quote_is_flagged() -> None:
    issues = quote_issues(make_quote(age_seconds=301), now=NOW, max_age_seconds=300)
    assert issues == ["stale quote: 301s old (limit 300s)"]


def test_freshness_is_skipped_when_no_limit_given() -> None:
    assert quote_issues(make_quote(age_seconds=86_400), now=NOW, max_age_seconds=None) == []


def test_future_quote_is_flagged() -> None:
    issues = quote_issues(make_quote(age_seconds=-60), now=NOW, max_age_seconds=None)
    assert issues == ["quote timestamp is in the future"]


def test_small_clock_skew_is_tolerated() -> None:
    assert quote_issues(make_quote(age_seconds=-3), now=NOW, max_age_seconds=300) == []


@pytest.mark.parametrize(("bid", "ask"), [("0", "100"), ("100", "0"), ("-1", "100")])
def test_one_sided_quote_is_flagged(bid: str, ask: str) -> None:
    issues = quote_issues(make_quote(bid=bid, ask=ask), now=NOW, max_age_seconds=300)
    assert issues == ["one-sided or empty quote"]


def test_crossed_quote_is_flagged() -> None:
    issues = quote_issues(make_quote(bid="101", ask="100"), now=NOW, max_age_seconds=300)
    assert issues == ["crossed quote (ask below bid)"]


def test_multiple_quote_issues_are_all_reported() -> None:
    issues = quote_issues(
        make_quote(bid="101", ask="100", age_seconds=999), now=NOW, max_age_seconds=300
    )
    assert len(issues) == 2


# ---------- Bars ----------


def test_valid_bars_have_no_issues() -> None:
    assert bar_issues(make_bars()) == []


def test_empty_bars_are_flagged() -> None:
    assert bar_issues(()) == ["no daily bars"]


def test_unsorted_bars_are_flagged() -> None:
    bars = make_bars()
    assert bar_issues(tuple(reversed(bars))) == ["bars are not in strictly ascending time order"]


def test_duplicate_bar_timestamps_are_flagged() -> None:
    bar = make_bars()[0]
    assert "strictly ascending" in bar_issues((bar, bar))[0]


def test_non_positive_price_is_flagged() -> None:
    bar = make_bars()[0].model_copy(update={"low": Decimal("0")})
    assert bar_issues((bar,)) == [f"non-positive price on {bar.timestamp:%Y-%m-%d}"]


def test_inconsistent_high_low_is_flagged() -> None:
    bar = make_bars()[0]
    broken = bar.model_copy(update={"high": bar.close - 2})
    assert "inconsistent high/low" in bar_issues((broken,))[0]


def test_negative_volume_is_flagged() -> None:
    bar = make_bars()[0].model_copy(update={"volume": -1})
    assert "negative volume" in bar_issues((bar,))[0]


def test_every_bad_bar_is_reported() -> None:
    first, second, third = make_bars()
    bad = [b.model_copy(update={"volume": -1}) for b in (first, third)]
    issues = bar_issues((bad[0], second, bad[1]))
    assert len(issues) == 2
    assert issues[1] == f"negative volume on {third.timestamp:%Y-%m-%d}"
