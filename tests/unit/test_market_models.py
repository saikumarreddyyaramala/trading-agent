from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import NOW, make_article, make_asset, make_bars, make_quote
from trading_agent.domain.models import SymbolData


def test_quote_mid_and_spread() -> None:
    quote = make_quote(bid="99.95", ask="100.05")
    assert quote.mid_price == Decimal("100.00")
    assert quote.spread_pct == Decimal("0.1")


def test_empty_quote_has_zero_spread() -> None:
    assert make_quote(bid="0", ask="0").spread_pct == Decimal("0")


def test_quote_age() -> None:
    assert make_quote(age_seconds=42).age_seconds(NOW) == pytest.approx(42)


def test_symbol_data_derived_values() -> None:
    data = SymbolData(
        asset=make_asset("MSFT"),
        quote=None,
        daily_bars=make_bars("MSFT", closes=("200", "210", "220")),
        news=(make_article(symbols=("MSFT",)),),
    )
    assert data.symbol == "MSFT"
    assert data.last_close == Decimal("220")
    assert data.change_pct == Decimal("10")
    assert data.avg_daily_volume == 2_000_000


def test_models_are_immutable() -> None:
    quote = make_quote()
    with pytest.raises(ValueError, match="frozen"):
        quote.bid_price = Decimal("1")  # type: ignore[misc]
