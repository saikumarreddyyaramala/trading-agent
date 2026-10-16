"""Integration tests against the real Alpaca paper API and the real LLM.

Skipped unless the keys are present, so CI stays green without secrets.
Run locally with:  make test-integration
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp_types import CallToolResult

from trading_agent.bootstrap import Container
from trading_agent.data.quality import bar_issues
from trading_agent.domain.models import ChatMessage, Evidence, LLMRequest, TradeProposal
from trading_agent.mcp_server import ALLOWED_TOOLS
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


def test_research_tools_against_alpaca() -> None:
    tools = _require("alpaca_api_key", "alpaca_secret_key").build_research_tools()
    active = tools.get_most_active(10)["stocks"]
    assert active
    snapshot = tools.research_snapshot([active[0]["symbol"], "SPY"])
    assert snapshot["symbols"], snapshot["rejected"]
    assert tools.market_status()["next_open"]


def test_mcp_server_over_stdio() -> None:
    """Launch the real server as a subprocess, exactly as an agent host would."""
    _require("alpaca_api_key", "alpaca_secret_key")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "trading_agent.cli", "mcp-server"], cwd=str(REPO)
    )

    async def main() -> tuple[set[str], CallToolResult]:
        async with Client(params) as client:
            listed = await client.list_tools()
            quote = await client.call_tool("get_quote", {"symbol": "SPY"})
            return {t.name for t in listed.tools}, quote

    names, quote = anyio.run(main)
    assert names == ALLOWED_TOOLS
    assert quote.is_error is False
    assert quote.structured_content is not None
    assert quote.structured_content["symbol"] == "SPY"


def test_broker_positions_and_orders_are_readable() -> None:
    broker = _require("alpaca_api_key", "alpaca_secret_key").build_broker()
    assert isinstance(broker.get_positions(), tuple)
    assert broker.count_buy_orders_since(datetime.now(UTC) - timedelta(days=1)) >= 0


def test_risk_engine_with_live_data() -> None:
    """Every rule runs against real data. Approval depends on market hours, so only the
    record is asserted; ETFs must always be rejected."""
    container = _require("alpaca_api_key", "alpaca_secret_key")
    engine = container.build_risk_engine()
    quote = container.build_market_data().get_latest_quotes(["SOXL"])["SOXL"]
    proposal = TradeProposal(
        decision="buy",
        symbol="SOXL",
        thesis="integration test",
        reference_price=quote.ask_price if quote.ask_price > 0 else Decimal("1"),
        confidence="low",
        evidence=(
            Evidence(claim="x", source_call_id="t"),
            Evidence(claim="y", source_call_id="t"),
        ),
        risks=("test",),
    )
    decision = engine.evaluate(proposal)
    assert not decision.approved
    assert "Asset" in [c.name for c in decision.failed_checks]
    assert len(decision.checks) == 15


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


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="never spend LLM tokens in CI")
def test_analyst_end_to_end() -> None:
    """A real analyst run: real LLM, real Alpaca data. Costs tokens; places no order."""
    container = _container()
    provider = container.config.llm.provider
    agent = _require(
        "alpaca_api_key", "alpaca_secret_key", *LLM_SECRET_FIELDS[provider]
    ).build_analyst()
    result = anyio.run(agent.run)
    print(result.model_dump_json(indent=2))  # visible with -s, for a human to judge the reasoning
    assert result.status in {"proposed", "no_trade"}, result.reason
    assert result.tool_calls, "the analyst should research before answering"
