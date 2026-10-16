from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import make_asset
from trading_agent.risk.eligibility import (
    asset_issues,
    liquidity_issues,
    looks_like_fund,
    price_issues,
)
from trading_agent.settings.models import RiskConfig

RISK = RiskConfig()


@pytest.mark.parametrize(
    "name",
    [
        "SPDR S&P 500 ETF Trust",
        "Direxion Daily Semiconductor Bull 3X ETF",
        "GraniteShares 2x Short NVDA Daily ETF",
        "iPath Series B S&P 500 VIX Short-Term Futures ETN",
        "ProShares UltraPro QQQ",
        "ProShares UltraShort Bloomberg Crude Oil",
        "Tortoise Energy Infrastructure Fund",
        "Defiance Daily Target 1.5x Long TSLA",
        "MicroSectors Gold Miners -3X Inverse Leveraged",
    ],
)
def test_funds_and_leveraged_products_are_recognised(name: str) -> None:
    assert looks_like_fund(name)


@pytest.mark.parametrize(
    "name",
    [
        "Apple Inc. Common Stock",
        "Northern Trust Corporation Common Stock",  # "Trust" alone is not a fund
        "Ultra Clean Holdings, Inc. Common Stock",  # "Ultra" alone is not a fund
        "Fundamental Global Inc.",  # "Fund" only as a whole word
        "Exxon Mobil Corporation",
        None,
        "",
    ],
)
def test_operating_companies_are_not_mistaken_for_funds(name: str | None) -> None:
    assert not looks_like_fund(name)


def test_eligible_asset_has_no_issues() -> None:
    assert asset_issues(make_asset("AAPL"), RISK) == []


def test_every_asset_problem_is_reported() -> None:
    asset = make_asset(
        "SOXL",
        tradable=False,
        status="inactive",
        exchange="ARCA",
        name="Direxion Daily Semiconductor Bull 3X ETF",
    )
    assert asset_issues(asset, RISK) == [
        "not tradable at the broker",
        "asset status is inactive",
        "listed on ARCA (allowed: NYSE, NASDAQ)",
        "looks like a fund, ETF, or leveraged product (Direxion Daily Semiconductor Bull 3X ETF)",
    ]


def test_missing_exchange_is_reported_clearly() -> None:
    assert asset_issues(make_asset(exchange=""), RISK) == [
        "listed on unknown exchange (allowed: NYSE, NASDAQ)"
    ]


def test_funds_can_be_allowed_by_config() -> None:
    risk = RiskConfig(exclude_funds=False, allowed_exchanges=("ARCA",))
    asset = make_asset("SPY", exchange="ARCA", name="SPDR S&P 500 ETF Trust")
    assert asset_issues(asset, risk) == []


@pytest.mark.parametrize(("price", "ok"), [("5", True), ("4.99", False), ("250", True)])
def test_minimum_price(price: str, ok: bool) -> None:
    assert (price_issues(Decimal(price), RISK) == []) is ok


@pytest.mark.parametrize(("volume", "ok"), [(1_000_000, True), (999_999, False)])
def test_minimum_liquidity(volume: int, ok: bool) -> None:
    issues = liquidity_issues(volume, RISK)
    assert (issues == []) is ok
    if not ok:
        assert issues == ["average daily volume 999,999 is below the 1,000,000 minimum"]
