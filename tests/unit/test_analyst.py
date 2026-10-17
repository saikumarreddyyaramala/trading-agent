"""The analyst agent end to end: scripted model, real in-process MCP server, fake market data."""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeNews,
    FakeProvider,
    healthy_market_data,
    make_article,
    provider_error,
    tool_call,
)
from trading_agent.agents.analyst import MAX_TOOL_RESULT_CHARS, AnalystAgent, AnalystResult
from trading_agent.agents.prompts import PROMPT_VERSION
from trading_agent.data.collector import MarketDataCollector
from trading_agent.domain.models import ChatMessage, Lesson, ToolCall
from trading_agent.llm import LLMGateway
from trading_agent.mcp_server import ALLOWED_TOOLS, ResearchTools, build_server
from trading_agent.settings.models import AppConfig, LLMConfig, MarketDataConfig

QUOTE = tool_call("c1", "get_quote", symbol="AAPL")
BARS = tool_call("c2", "get_daily_bars", symbol="AAPL", days=3)


def _answer(**overrides: Any) -> str:
    values: dict[str, Any] = {
        "decision": "buy",
        "symbol": "AAPL",
        "thesis": "Tight spread, steady uptrend.",
        "reference_price": 100.05,
        "confidence": "medium",
        "evidence": [
            {"claim": "Spread is 0.1%", "source_call_id": "c1"},
            {"claim": "Up 1% over 3 days", "source_call_id": "c2"},
        ],
        "risks": ["Broad market reversal"],
    }
    values.update(overrides)
    return json.dumps(values)


NO_TRADE = json.dumps(
    {"decision": "no_trade", "thesis": "Nothing compelling.", "confidence": "high"}
)


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def bind(self, **_: Any) -> _Log:
        return self

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))


def _run(
    outcomes: list[Any],
    *,
    log: _Log | None = None,
    tool_budget: int = 100,
    lessons: tuple[Lesson, ...] = (),
    playbooks: tuple[Any, ...] = (),
    **analyst: Any,
) -> tuple[AnalystResult, FakeProvider]:
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
        max_calls=tool_budget,
        clock=lambda: NOW,
    )
    config = AppConfig.model_validate({"analyst": analyst})
    provider = FakeProvider(outcomes)
    gateway = LLMGateway(provider, LLMConfig(model="primary", fallback_model="backup"))
    agent = AnalystAgent(
        gateway,
        build_server(tools),
        config,
        clock=lambda: NOW,
        logger=log or _Log(),
        lessons=lessons,
        playbooks=playbooks,
    )
    return anyio.run(agent.run), provider


def _messages(provider: FakeProvider, request: int) -> tuple[ChatMessage, ...]:
    return provider.requests[request].messages


# ---------- Successful runs ----------


def test_research_then_verified_buy() -> None:
    result, _ = _run([(QUOTE, BARS), _answer()])

    assert result.status == "proposed"
    assert result.reason == "evidence verified"
    assert result.proposal is not None
    assert result.proposal.symbol == "AAPL"
    assert result.turns == 2
    assert [(c.call_id, c.tool, c.is_error) for c in result.tool_calls] == [
        ("c1", "get_quote", False),
        ("c2", "get_daily_bars", False),
    ]
    assert result.usage.input_tokens == 20
    assert result.prompt_version == PROMPT_VERSION
    assert result.started_at == NOW


def test_model_sees_every_allowlisted_tool_and_the_system_prompt() -> None:
    _, provider = _run([NO_TRADE])
    request = provider.requests[0]
    assert {tool.name for tool in request.tools} == ALLOWED_TOOLS
    assert all(tool.input_schema.get("type") == "object" for tool in request.tools)
    assert "AT MOST ONE trade" in request.system
    assert "Thursday 2026-10-15 10:00" in _messages(provider, 0)[0].content
    assert request.tags["agent"] == "analyst"


def test_tool_results_are_returned_with_their_call_ids() -> None:
    _, provider = _run([(QUOTE, BARS), _answer()])
    assistant, results = _messages(provider, 1)[1:]
    assert assistant.tool_calls == (QUOTE, BARS)
    first, second = results.tool_results
    assert first.call_id == "c1"
    assert first.content.startswith("call_id=c1\n")
    assert json.loads(first.content.split("\n", 1)[1])["symbol"] == "AAPL"
    assert second.is_error is False


def test_no_trade_is_a_valid_outcome() -> None:
    result, _ = _run([NO_TRADE])
    assert result.status == "no_trade"
    assert result.proposal is not None
    assert result.proposal.decision == "no_trade"


# ---------- Evidence enforcement and repair ----------


def test_invented_evidence_gets_one_repair_chance() -> None:
    bad = _answer(evidence=[{"claim": "Insider buying", "source_call_id": "nope"}] * 2)
    result, provider = _run([(QUOTE, BARS), bad, _answer()])

    assert result.status == "proposed"
    repair = _messages(provider, 2)[-1]
    assert "Your answer was rejected" in repair.content
    assert "unknown or failed tool call 'nope'" in repair.content


def test_repeated_bad_answers_fail_safe() -> None:
    result, _ = _run([(QUOTE, BARS), _answer(reference_price=150), _answer(reference_price=150)])
    assert result.status == "failed"
    assert result.proposal is None
    assert "away from every price" in result.reason


def test_unparseable_answer_without_repairs_fails() -> None:
    result, _ = _run(["I'd buy AAPL."], repair_attempts=0)
    assert result.status == "failed"
    assert result.reason == "answer rejected: the answer contains no JSON object"


def test_buy_without_any_research_is_rejected() -> None:
    result, _ = _run([_answer()], repair_attempts=0)
    assert result.status == "failed"
    assert "unknown or failed tool call 'c1'" in result.reason


# ---------- Limits and failures ----------


def test_model_outage_fails_safe() -> None:
    result, _ = _run([provider_error("primary"), provider_error("backup")])
    assert result.status == "failed"
    assert result.reason.startswith("model unavailable:")
    assert result.turns == 1


def test_turn_limit() -> None:
    result, _ = _run([(QUOTE,), (BARS,)], max_turns=2)
    assert result.status == "failed"
    assert result.reason == "turn limit of 2 reached"
    assert result.turns == 2


def test_token_budget() -> None:
    result, provider = _run([(QUOTE,), NO_TRADE], max_total_tokens=12)
    assert result.status == "failed"
    assert result.reason == "token budget for the run is spent"
    assert len(provider.requests) == 1


def test_tool_calls_per_turn_are_capped() -> None:
    calls = tuple(tool_call(f"c{i}", "market_status") for i in range(5))
    result, provider = _run([calls, NO_TRADE], max_tool_calls_per_turn=2)
    assert [c.call_id for c in result.tool_calls] == ["c0", "c1"]
    assert len(_messages(provider, 1)[-1].tool_results) == 2


def test_malformed_arguments_are_reported_to_the_model() -> None:
    broken = ToolCall(id="x1", name="get_quote", arguments_json="{symbol: AAPL")
    result, provider = _run([(broken,), NO_TRADE])
    assert result.tool_calls[0].is_error is True
    assert result.tool_calls[0].arguments is None
    error = _messages(provider, 1)[-1].tool_results[0]
    assert error.is_error is True
    assert "arguments must be a JSON object" in error.content


def test_unknown_tools_fail_and_count_for_nothing() -> None:
    trade = tool_call("t1", "place_order", symbol="AAPL", qty=10)
    result, provider = _run(
        [(trade,), _answer(evidence=[{"claim": "x", "source_call_id": "t1"}] * 2)],
        repair_attempts=0,
    )
    assert result.tool_calls[0].is_error is True
    assert "Unknown tool" in _messages(provider, 1)[-1].tool_results[0].content
    assert result.status == "failed"  # an errored call can never back a proposal


def test_tool_budget_spent_mid_run() -> None:
    result, provider = _run([(QUOTE, BARS), NO_TRADE], tool_budget=1)
    assert [c.is_error for c in result.tool_calls] == [False, True]
    assert "budget" in _messages(provider, 1)[-1].tool_results[1].content


def test_long_tool_results_are_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_agent.agents.analyst.MAX_TOOL_RESULT_CHARS", 50)
    _, provider = _run([(BARS,), NO_TRADE])
    content = _messages(provider, 1)[-1].tool_results[0].content
    assert content.endswith("[truncated]")
    assert MAX_TOOL_RESULT_CHARS == 20_000


def test_outcome_is_logged() -> None:
    log = _Log()
    _run([(QUOTE, BARS), _answer()], log=log)
    event, fields = log.events[-1]
    assert event == "analyst.finished"
    assert fields["status"] == "proposed"
    assert fields["symbol"] == "AAPL"
    assert fields["tool_calls"] == 2


def test_tool_outputs_are_kept_for_review() -> None:
    result, _ = _run([(QUOTE, tool_call("bad", "place_order")), NO_TRADE])
    quote, unknown = result.tool_calls
    assert quote.result is not None
    assert json.loads(quote.result)["symbol"] == "AAPL"
    assert unknown.is_error is True
    assert "Unknown tool" in (unknown.result or "")


def test_malformed_arguments_keep_their_message() -> None:
    broken = ToolCall(id="x1", name="get_quote", arguments_json="{oops")
    result, _ = _run([(broken,), NO_TRADE])
    assert result.tool_calls[0].result == "arguments must be a JSON object"


# ---------- Lessons (self-learning) ----------


def _lesson(lesson_id: int, text: str) -> Lesson:
    return Lesson(
        id=lesson_id,
        text=text,
        category="research",
        source_run_id=None,
        created_at=NOW,
        expires_at=NOW,
    )


def test_lessons_reach_the_prompt_and_are_recorded() -> None:
    lessons = (_lesson(3, "Distrust single-source news."), _lesson(5, "Check volume first."))
    result, provider = _run([NO_TRADE], lessons=lessons)
    system = provider.requests[0].system
    assert "Lessons from your own past results" in system
    assert "- [research, 2026-10-15] Distrust single-source news." in system
    assert result.lesson_ids == (3, 5)


def test_no_lessons_no_lessons_section() -> None:
    result, provider = _run([NO_TRADE])
    assert "Lessons from your own past results" not in provider.requests[0].system
    assert result.lesson_ids == ()


# ---------- Playbooks (ADR 0017) ----------


def test_a_named_playbook_is_kept_when_it_was_offered() -> None:
    from tests.unit.test_guidebook_store import playbook

    offered = (playbook("gap-check"),)
    result, provider = _run([(QUOTE, BARS), _answer(playbook="gap-check")], playbooks=offered)
    assert result.proposal is not None
    assert result.proposal.playbook == "gap-check"
    system = provider.requests[0].system
    assert "### gap-check: Earnings gap check (candidate)" in system
    assert '"playbook": "the name of the playbook you followed" or null' in system


def test_an_unknown_playbook_name_is_dropped() -> None:
    result, _ = _run([(QUOTE, BARS), _answer(playbook="made-up")])
    assert result.status == "proposed"
    assert result.proposal is not None
    assert result.proposal.playbook is None
