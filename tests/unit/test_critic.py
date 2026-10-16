from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from tests.conftest import NOW, FakeProvider, provider_error
from trading_agent.agents.analyst import ToolCallRecord
from trading_agent.agents.critic import (
    CRITIC_PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
    CriticAgent,
    CriticOutput,
    EvidenceCheck,
    enforce,
    evidence_message,
)
from trading_agent.domain.models import Evidence, TradeProposal
from trading_agent.llm import LLMGateway
from trading_agent.settings.models import AppConfig, LLMConfig

PROPOSAL = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="Tight spread and rising volume.",
    reference_price=Decimal("100.05"),
    confidence="medium",
    evidence=(
        Evidence(claim="Ask is $100.10", source_call_id="c1"),
        Evidence(claim="Up 1% over 3 days", source_call_id="c2"),
    ),
    risks=("Market reversal",),
)
CALLS = [
    ToolCallRecord(
        call_id="c0", tool="market_status", arguments={}, is_error=False, result='{"is_open": true}'
    ),
    ToolCallRecord(
        call_id="c2",
        tool="get_daily_bars",
        arguments={"symbol": "AAPL"},
        is_error=False,
        result='{"summary": {"change_pct": "1.00"}}',
    ),
    ToolCallRecord(
        call_id="c1",
        tool="get_quote",
        arguments={"symbol": "AAPL"},
        is_error=False,
        result='{"ask": "100.1"}',
    ),
    ToolCallRecord(call_id="cx", tool="get_news", arguments={}, is_error=True, result="error"),
]


def _answer(
    verdict: str = "approve", supported: tuple[bool, bool] = (True, True), **extra: Any
) -> str:
    values: dict[str, Any] = {
        "verdict": verdict,
        "summary": "The claims match the data.",
        "evidence_checks": [
            {"source_call_id": "c1", "supported": supported[0], "note": "ask matches"},
            {"source_call_id": "c2", "supported": supported[1], "note": "change matches"},
        ],
        "concerns": ["Low news coverage"],
    }
    values.update(extra)
    return json.dumps(values)


def _critic(outcomes: list[Any], **critic: Any) -> tuple[CriticAgent, FakeProvider]:
    provider = FakeProvider(outcomes)
    gateway = LLMGateway(provider, LLMConfig(model="judge", fallback_model="judge-backup"))
    config = AppConfig.model_validate({"critic": critic})
    return CriticAgent(gateway, config, clock=lambda: NOW), provider


# ---------- Verdicts ----------


def test_approval_when_every_claim_is_supported() -> None:
    critic, provider = _critic([_answer()])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "approved"
    assert result.model_verdict == "approve"
    assert result.reasons == ()
    assert result.concerns == ("Low news coverage",)
    assert result.model == "judge"
    assert result.prompt_version == CRITIC_PROMPT_VERSION
    assert result.usage.input_tokens == 10
    assert result.created_at == NOW
    request = provider.requests[0]
    assert request.system == CRITIC_SYSTEM_PROMPT
    assert request.tools == ()  # the critic gets no tools
    assert request.tags == {"run_id": "r1", "agent": "critic"}


def test_the_critics_veto_always_stands() -> None:
    critic, _ = _critic([_answer("veto")])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "vetoed"
    assert result.reasons == ("the critic vetoed the trade",)


def test_an_unsupported_claim_vetoes_even_an_approve() -> None:
    critic, _ = _critic([_answer("approve", supported=(True, False))])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "vetoed"
    assert result.model_verdict == "approve"
    assert result.reasons == ("evidence 'c2' is unsupported: change matches",)


def test_an_unchecked_claim_vetoes() -> None:
    answer = json.loads(_answer())
    answer["evidence_checks"] = answer["evidence_checks"][:1]
    critic, _ = _critic([json.dumps(answer)])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "vetoed"
    assert result.reasons == ("evidence 'c2' was not checked",)


def test_checks_for_uncited_calls_do_not_matter() -> None:
    answer = json.loads(_answer())
    answer["evidence_checks"].append({"source_call_id": "c0", "supported": False, "note": "n/a"})
    critic, _ = _critic([json.dumps(answer)])
    assert critic.review("r1", PROPOSAL, CALLS).status == "approved"


# ---------- Fail safe ----------


def test_model_outage_fails_and_means_no_trade() -> None:
    critic, _ = _critic([provider_error("judge"), provider_error("judge-backup")])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "failed"
    assert result.model_verdict is None
    assert result.summary.startswith("critic unavailable:")
    assert result.reasons == (result.summary,)


def test_invalid_answer_gets_one_repair() -> None:
    critic, provider = _critic(["Looks fine to me.", _answer()])
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "approved"
    repair = provider.requests[1].messages[-1].content
    assert "Your answer was rejected (the answer contains no JSON object)" in repair


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        json.dumps({"verdict": "maybe", "summary": "x"}),
        json.dumps({"verdict": "approve", "summary": "x", "extra": 1}),
        json.dumps({"verdict": "approve", "summary": ""}),
    ],
)
def test_invalid_answers_without_repairs_fail(bad: str) -> None:
    critic, _ = _critic([bad], repair_attempts=0)
    result = critic.review("r1", PROPOSAL, CALLS)
    assert result.status == "failed"
    assert result.summary.startswith("invalid critic answer:")


def test_only_buys_are_reviewed() -> None:
    critic, _ = _critic([])
    no_trade = TradeProposal(decision="no_trade", thesis="Quiet.", confidence="low")
    with pytest.raises(ValueError, match="buy proposals only"):
        critic.review("r1", no_trade, CALLS)


# ---------- What the critic sees ----------


def test_evidence_message_lists_cited_calls_first_and_skips_errors() -> None:
    message = evidence_message(PROPOSAL, CALLS, max_chars=60_000)
    assert message.startswith("PROPOSAL:")
    assert (
        json.loads(message.split("PROPOSAL:\n", 1)[1].split("\n\nTOOL OUTPUTS", 1)[0])["symbol"]
        == "AAPL"
    )
    order = [message.index(f"call_id={c}") for c in ("c1", "c2", "c0")]
    assert order == sorted(order)  # cited in evidence order, then the rest
    assert "call_id=cx" not in message  # failed calls are not evidence
    assert '{"ask": "100.1"}' in message


def test_evidence_message_respects_the_budget() -> None:
    big = [
        ToolCallRecord(
            call_id="c1", tool="get_news", arguments={}, is_error=False, result="x" * 9000
        )
    ]
    message = evidence_message(PROPOSAL, big, max_chars=5_000)
    assert len(message) <= 5_000
    assert "c1: omitted (evidence budget reached)" in message


def test_missing_outputs_are_marked() -> None:
    old = [ToolCallRecord(call_id="c1", tool="get_quote", arguments=None, is_error=False)]
    assert "(output not recorded)" in evidence_message(PROPOSAL, old, max_chars=60_000)


def test_prompt_demands_a_check_per_evidence_item_and_flags_injection() -> None:
    assert "For EVERY evidence item" in CRITIC_SYSTEM_PROMPT
    assert "instructions or text aimed at an AI" in CRITIC_SYSTEM_PROMPT
    assert "When in doubt, veto." in CRITIC_SYSTEM_PROMPT


def test_enforce_directly() -> None:
    output = CriticOutput(
        verdict="approve",
        summary="ok",
        evidence_checks=(
            EvidenceCheck(source_call_id="c1", supported=True),
            EvidenceCheck(source_call_id="c2", supported=True),
        ),
    )
    assert enforce(output, PROPOSAL) == ("approved", ())
