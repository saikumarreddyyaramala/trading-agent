"""Shared fixtures and fakes. Unit tests never touch the network or real credentials."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import (
    AccountSnapshot,
    Asset,
    Bar,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MarketClock,
    NewsArticle,
    Quote,
)
from trading_agent.llm.errors import LLMProviderError
from trading_agent.settings.secrets import Secrets

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"

# Fixed "now" for market data tests: 2026-10-15 10:00 New York time.
NOW = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A private copy of the real config folder, safe to modify per test."""
    target = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, target)
    return target


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test may read the developer's real .env or exported keys."""
    for name in (
        "APP_ENV",
        "CONFIG_DIR",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LIVE_TRADING_CONFIRMED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def make_secrets(**overrides: Any) -> Secrets:
    values: dict[str, Any] = {
        "alpaca_api_key": SecretStr("test-alpaca-key"),
        "alpaca_secret_key": SecretStr("test-alpaca-secret"),
        "openai_api_key": SecretStr("test-openai-key"),
        "anthropic_api_key": SecretStr("test-anthropic-key"),
        "live_trading_confirmed": False,
    }
    values.update(overrides)
    return Secrets(_env_file=None, **values)


class FakeBroker:
    def __init__(
        self,
        *,
        is_paper: bool = True,
        trading_blocked: bool = False,
        fail_account: bool = False,
        fail_clock: bool = False,
        market_open: bool = True,
    ) -> None:
        self._market_open = market_open
        self._is_paper = is_paper
        self._trading_blocked = trading_blocked
        self._fail_account = fail_account
        self._fail_clock = fail_clock

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def get_account(self) -> AccountSnapshot:
        if self._fail_account:
            raise RuntimeError("broker unreachable")
        return AccountSnapshot(
            account_number_masked="****1234",
            status="ACTIVE",
            currency="USD",
            cash=Decimal("100000"),
            equity=Decimal("100000"),
            buying_power=Decimal("100000"),
            pattern_day_trader=False,
            trading_blocked=self._trading_blocked,
            account_blocked=False,
            is_paper=self._is_paper,
        )

    def get_clock(self) -> MarketClock:
        if self._fail_clock:
            raise RuntimeError("clock unavailable")
        return MarketClock(timestamp=NOW, is_open=self._market_open, next_open=NOW, next_close=NOW)


class FakeProvider:
    """Scripted LLM provider: each call pops the next outcome (a text or an exception)."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        self.calls.append((model, max_tokens))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome,
            provider=self.name,
            model=model,
            stop_reason="end_turn",
            usage=LLMUsage(input_tokens=10, output_tokens=2),
            latency_ms=5.0,
        )


def provider_error(model: str = "m", *, fatal: bool = False) -> LLMProviderError:
    return LLMProviderError("fake", model, "boom", fatal=fatal)


# ---------- Market data builders and fakes ----------


def make_quote(
    symbol: str = "AAPL",
    *,
    bid: str = "100.00",
    ask: str = "100.10",
    age_seconds: float = 2,
    now: datetime = NOW,
) -> Quote:
    return Quote(
        symbol=symbol,
        timestamp=now - timedelta(seconds=age_seconds),
        bid_price=Decimal(bid),
        bid_size=Decimal("100"),
        ask_price=Decimal(ask),
        ask_size=Decimal("200"),
    )


def make_bars(
    symbol: str = "AAPL", closes: Sequence[str] = ("100", "102", "101"), volume: int = 2_000_000
) -> tuple[Bar, ...]:
    """One daily bar per close, oldest first, ending the day before NOW."""
    start = NOW - timedelta(days=len(closes))
    bars = []
    for offset, raw in enumerate(closes):
        close = Decimal(raw)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=start + timedelta(days=offset),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=volume,
            )
        )
    return tuple(bars)


def make_asset(symbol: str = "AAPL", *, tradable: bool = True) -> Asset:
    return Asset(
        symbol=symbol,
        name=f"{symbol} Inc.",
        exchange="NASDAQ",
        status="active",
        tradable=tradable,
        fractionable=True,
    )


def make_article(
    article_id: str = "1", symbols: Sequence[str] = ("AAPL",), *, hours_ago: float = 1
) -> NewsArticle:
    created = NOW - timedelta(hours=hours_ago)
    return NewsArticle(
        id=article_id,
        headline=f"Headline {article_id}",
        summary="Summary.",
        source="benzinga",
        url=f"https://example.com/{article_id}",
        symbols=tuple(symbols),
        created_at=created,
        updated_at=created,
    )


class FakeMarketData:
    """In-memory `MarketDataPort`. Records calls; any data not supplied is simply absent."""

    def __init__(
        self,
        *,
        quotes: dict[str, Quote] | None = None,
        bars: dict[str, tuple[Bar, ...]] | None = None,
        assets: dict[str, Asset] | None = None,
        fail: bool = False,
    ) -> None:
        self.quotes = quotes or {}
        self.bars = bars or {}
        self.assets = assets or {}
        self.fail = fail
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def get_asset(self, symbol: str) -> Asset:
        self.calls.append(("asset", (symbol,)))
        if symbol not in self.assets:
            raise MarketDataError(f"asset lookup failed for {symbol}: not found")
        return self.assets[symbol]

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        self.calls.append(("quotes", tuple(symbols)))
        if self.fail:
            raise MarketDataError("could not fetch quotes: down")
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    def get_daily_bars(self, symbols: Sequence[str], *, days: int) -> dict[str, tuple[Bar, ...]]:
        self.calls.append(("bars", tuple(symbols)))
        return {s: self.bars[s][-days:] for s in symbols if s in self.bars}


def healthy_market_data(*symbols: str) -> FakeMarketData:
    names = symbols or ("AAPL",)
    return FakeMarketData(
        quotes={s: make_quote(s) for s in names},
        bars={s: make_bars(s) for s in names},
        assets={s: make_asset(s) for s in names},
    )


class FakeNews:
    def __init__(self, articles: Sequence[NewsArticle] = (), *, fail: bool = False) -> None:
        self.articles = tuple(articles)
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def get_news(
        self, symbols: Sequence[str], *, since: datetime, limit: int
    ) -> tuple[NewsArticle, ...]:
        self.calls.append({"symbols": tuple(symbols), "since": since, "limit": limit})
        if self.fail:
            raise MarketDataError("could not fetch news: down")
        return self.articles[:limit]
