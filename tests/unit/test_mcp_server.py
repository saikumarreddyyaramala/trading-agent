"""The MCP server, exercised through a real MCP client connected in-process."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp_types import CallToolResult, TextContent

from tests.conftest import NOW, FakeBroker, FakeNews, healthy_market_data, make_article
from trading_agent.data.collector import MarketDataCollector
from trading_agent.mcp_server import ALLOWED_TOOLS, ResearchTools, build_server
from trading_agent.mcp_server.tools import MAX_BARS, MAX_SNAPSHOT_SYMBOLS
from trading_agent.settings.models import MarketDataConfig

# Words that would suggest a tool can change broker state.
FORBIDDEN_WORDS = ("order", "trade", "buy", "sell", "cancel", "submit", "close", "position")


def _server(max_calls: int = 100) -> tuple[MCPServer, ResearchTools]:
    market_data = healthy_market_data("AAPL", "MSFT")
    news = FakeNews([make_article("1", ("AAPL",))])
    collector = MarketDataCollector(
        market_data, news, MarketDataConfig(), max_data_age_seconds=300, clock=lambda: NOW
    )
    tools = ResearchTools(
        market_data,
        news,
        collector,
        FakeBroker().get_clock,
        max_data_age_seconds=300,
        max_calls=max_calls,
        clock=lambda: NOW,
    )
    return build_server(tools), tools


def _with_client[T](server: MCPServer, fn: Callable[[Client], Awaitable[T]]) -> T:
    async def main() -> T:
        async with Client(server) as client:
            return await fn(client)

    return anyio.run(main)


def _call(name: str, args: dict[str, Any], server: MCPServer | None = None) -> CallToolResult:
    return _with_client(server or _server()[0], lambda c: c.call_tool(name, args))


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


# ---------- Tool surface (the permission model) ----------


def test_only_allowlisted_tools_are_exposed() -> None:
    listed = _with_client(_server()[0], lambda c: c.list_tools())
    assert {tool.name for tool in listed.tools} == ALLOWED_TOOLS


def test_no_tool_name_suggests_trading() -> None:
    for name in ALLOWED_TOOLS:
        assert not any(word in name for word in FORBIDDEN_WORDS), name


def test_every_tool_is_annotated_read_only() -> None:
    listed = _with_client(_server()[0], lambda c: c.list_tools())
    for tool in listed.tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.destructive_hint is False, tool.name
        assert tool.description, tool.name


def test_schemas_carry_the_bounds() -> None:
    listed = _with_client(_server()[0], lambda c: c.list_tools())
    schemas = {tool.name: tool.input_schema for tool in listed.tools}
    assert schemas["get_daily_bars"]["properties"]["days"]["maximum"] == MAX_BARS
    assert schemas["get_quote"]["properties"]["symbol"]["maxLength"] == 10
    assert schemas["research_snapshot"]["properties"]["symbols"]["maxItems"] == (
        MAX_SNAPSHOT_SYMBOLS
    )


def test_server_instructions_warn_about_untrusted_news() -> None:
    server, _ = _server()
    assert "untrusted" in (server.instructions or "")


# ---------- Calls ----------


def test_successful_call_returns_structured_and_text_content() -> None:
    result = _call("get_quote", {"symbol": "aapl"})
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["symbol"] == "AAPL"
    assert json.loads(_text(result))["ask"] == "100.1"


def test_defaults_apply_when_arguments_are_omitted() -> None:
    result = _call("get_daily_bars", {"symbol": "AAPL"})
    assert result.is_error is False
    assert result.structured_content is not None
    assert len(result.structured_content["bars"]) == 3  # all the fake has, up to 30


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("market_status", {}),
        ("get_news", {"symbol": "AAPL"}),
        ("get_asset", {"symbol": "MSFT"}),
        ("get_most_active", {}),
        ("research_snapshot", {"symbols": ["AAPL", "MSFT"]}),
        ("technical_analysis", {"symbol": "AAPL"}),
    ],
)
def test_every_tool_answers(name: str, args: dict[str, Any]) -> None:
    result = _call(name, args)
    assert result.is_error is False, _text(result)


def test_anticipated_failures_reach_the_model_as_tool_errors() -> None:
    result = _call("get_quote", {"symbol": "BAD SYM"})
    assert result.is_error is True
    assert "invalid symbol" in _text(result)


def test_schema_rejects_out_of_range_arguments_before_the_tool_runs() -> None:
    server, tools = _server()
    result = _call("get_daily_bars", {"symbol": "AAPL", "days": MAX_BARS + 1}, server)
    assert result.is_error is True
    assert tools.calls_used == 0


def test_unknown_tools_cannot_be_called() -> None:
    result = _call("place_order", {"symbol": "AAPL", "qty": 1})
    assert result.is_error is True
    assert "Unknown tool" in _text(result)


def test_budget_exhaustion_is_reported_to_the_model() -> None:
    server, _ = _server(max_calls=1)

    async def two_calls(client: Client) -> CallToolResult:
        await client.call_tool("market_status", {})
        return await client.call_tool("market_status", {})

    result = _with_client(server, two_calls)
    assert result.is_error is True
    assert "budget" in _text(result)


def test_crash_details_are_not_leaked_to_the_model() -> None:
    server, tools = _server()

    def broken_clock() -> Any:
        raise RuntimeError("secret internal detail")

    tools._market_clock = broken_clock
    result = _call("market_status", {}, server)
    assert result.is_error is True
    assert "secret internal detail" not in _text(result)
