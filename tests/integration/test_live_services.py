"""Integration tests against the real Alpaca paper API and the real LLM.

Skipped unless the keys are present, so CI stays green without secrets.
Run locally with:  make test-integration
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_agent.bootstrap import Container
from trading_agent.data.quality import bar_issues
from trading_agent.domain.models import ChatMessage, LLMRequest
from trading_agent.settings import Secrets, load_config
from trading_agent.settings.secrets import LLM_SECRET_FIELDS

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _container() -> Container:
    return Container(
        config=load_config("paper", REPO / "config"),
        secrets=Secrets(_env_file=REPO / ".env"),
    )


@pytest.fixture(autouse=True)
def _restore_env_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unit-test fixture clears env vars; integration tests read keys from the repo's .env.
    monkeypatch.chdir(REPO)


def _require(*fields: str) -> Container:
    container = _container()
    missing = container.secrets.missing(*fields)
    if missing:
        pytest.skip(f"missing {', '.join(missing)}")
    return container


def test_alpaca_paper_account_is_reachable() -> None:
    broker = _require("alpaca_api_key", "alpaca_secret_key").build_broker()
    account = broker.get_account()
    assert account.is_paper
    assert account.status
    assert broker.get_clock().next_open is not None


def test_alpaca_market_data_for_a_liquid_symbol() -> None:
    container = _require("alpaca_api_key", "alpaca_secret_key")
    market_data = container.build_market_data()

    quote = market_data.get_latest_quotes(["SPY"])["SPY"]
    bars = market_data.get_daily_bars(["SPY"], days=5)["SPY"]
    asset = market_data.get_asset("SPY")

    assert quote.ask_price > 0
    assert len(bars) == 5
    assert bar_issues(bars) == []
    # SIP daily volume for SPY is tens of millions; IEX-only volume would be far lower.
    assert sum(b.volume for b in bars) // len(bars) > 10_000_000
    assert asset.tradable


def test_alpaca_news_is_reachable() -> None:
    news = _require("alpaca_api_key", "alpaca_secret_key").build_news()
    articles = news.get_news([], since=datetime.now(UTC) - timedelta(days=3), limit=3)
    assert articles
    assert all(a.headline for a in articles)


def test_collector_end_to_end() -> None:
    collector = _require("alpaca_api_key", "alpaca_secret_key").build_collector()
    snapshot = collector.collect(["AAPL", "MSFT"], require_fresh_quotes=False)
    assert set(snapshot.symbols) == {"AAPL", "MSFT"}, snapshot.rejected
    assert snapshot.symbols["AAPL"].asset.exchange == "NASDAQ"


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="never spend LLM tokens in CI")
def test_llm_round_trip() -> None:
    provider = _container().config.llm.provider
    gateway = _require(*LLM_SECRET_FIELDS[provider]).build_llm_gateway()
    response = gateway.complete(
        LLMRequest(
            system="Reply with exactly one word.",
            messages=(ChatMessage(role="user", content="Say OK"),),
            max_tokens=512,
        )
    )
    assert response.text.strip()
