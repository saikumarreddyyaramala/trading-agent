from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from trading_agent.agents.evidence import (
    ProposalError,
    ResearchLedger,
    parse_proposal,
    verify_proposal,
)
from trading_agent.domain.models import Evidence, TradeProposal

LIMIT = Decimal("2")


def _buy(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "decision": "buy",
        "symbol": "AAPL",
        "thesis": "Strong volume and a positive catalyst.",
        "reference_price": 100.05,
        "confidence": "medium",
        "evidence": [
            {"claim": "Quote is tight", "source_call_id": "c1"},
            {"claim": "Up 1% over 3 days", "source_call_id": "c2"},
        ],
        "risks": ["Market-wide selloff"],
    }
    values.update(overrides)
    return values


def _ledger() -> ResearchLedger:
    ledger = ResearchLedger()
    ledger.record(
        "c1",
        "get_quote",
        {"symbol": "aapl"},
        {"symbol": "AAPL", "bid": "100", "ask": "100.1", "mid": "100.05"},
    )
    ledger.record(
        "c2",
        "get_daily_bars",
        {"symbol": "AAPL", "days": 3},
        {"symbol": "AAPL", "summary": {"last_close": "101"}},
    )
    return ledger


# ---------- Parsing ----------


def test_parses_plain_json() -> None:
    proposal = parse_proposal(json.dumps(_buy()))
    assert proposal.decision == "buy"
    assert proposal.reference_price == Decimal("100.05")
    assert proposal.evidence[0] == Evidence(claim="Quote is tight", source_call_id="c1")


def test_parses_fenced_json_with_surrounding_prose() -> None:
    text = f"Here is my answer:\n```json\n{json.dumps(_buy())}\n```\nGood luck."
    assert parse_proposal(text).symbol == "AAPL"


def test_parses_json_embedded_in_prose() -> None:
    assert parse_proposal(f"Final: {json.dumps(_buy())} done").symbol == "AAPL"


def test_symbol_is_normalised() -> None:
    assert parse_proposal(json.dumps(_buy(symbol=" aapl "))).symbol == "AAPL"


def test_no_trade_needs_no_symbol_or_evidence() -> None:
    text = json.dumps(
        {"decision": "no_trade", "thesis": "Nothing stands out.", "confidence": "high"}
    )
    proposal = parse_proposal(text)
    assert proposal.decision == "no_trade"
    assert proposal.symbol is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("I think AAPL looks good.", "no JSON object"),
        ("{not json}", "not valid JSON"),
        (json.dumps(_buy(decision="sell")), "decision"),
        (json.dumps(_buy(confidence="certain")), "confidence"),
        (json.dumps(_buy(reference_price=-1)), "reference_price"),
        (json.dumps(_buy(evidence=[_buy()["evidence"][0]])), "at least 2 evidence items"),
        (json.dumps(_buy(risks=[])), "at least 1 risk"),
        (json.dumps(_buy(symbol=None, reference_price=None)), "symbol, reference_price"),
        (json.dumps({**_buy(), "quantity": 100}), "quantity"),  # extra fields are rejected
    ],
)
def test_invalid_answers_are_rejected(text: str, message: str) -> None:
    with pytest.raises(ProposalError, match=message):
        parse_proposal(text)


# ---------- Ledger ----------


def test_ledger_extracts_prices_from_each_price_tool() -> None:
    ledger = _ledger()
    ledger.record(
        "c3",
        "research_snapshot",
        {"symbols": ["MSFT", "TSLA"]},
        {
            "symbols": [
                {"symbol": "MSFT", "last_close": "400", "quote": {"bid": "399", "ask": "401"}},
                {"symbol": "TSLA", "last_close": "250", "quote": None},
                "junk",
            ]
        },
    )
    assert ledger.prices["AAPL"] == [
        Decimal("100"),
        Decimal("100.1"),
        Decimal("100.05"),
        Decimal("101"),
    ]
    assert ledger.prices["MSFT"] == [Decimal("400"), Decimal("399"), Decimal("401")]
    assert ledger.prices["TSLA"] == [Decimal("250")]
    assert ledger.symbols_by_call["c3"] == {"MSFT", "TSLA"}


def test_ledger_tracks_symbols_without_prices() -> None:
    ledger = ResearchLedger()
    ledger.record("n1", "get_news", {"symbol": "nvda", "hours": 24}, {"symbol": "NVDA"})
    ledger.record("a1", "get_asset", {"symbol": "F"}, {"symbol": "F", "exchange": "NYSE"})
    assert ledger.symbols_by_call == {"n1": {"NVDA"}, "a1": {"F"}}
    assert ledger.prices == {}


@pytest.mark.parametrize("bad", [None, "", "abc", "NaN", "Infinity", "0", "-5"])
def test_ledger_ignores_unusable_prices(bad: Any) -> None:
    ledger = ResearchLedger()
    ledger.record("c1", "get_quote", {"symbol": "AAPL"}, {"symbol": "AAPL", "bid": bad})
    assert "AAPL" not in ledger.prices


def test_ledger_ignores_non_mapping_results() -> None:
    ledger = ResearchLedger()
    ledger.record("c1", "get_quote", {"symbol": "AAPL"}, None)
    assert ledger.symbols_by_call["c1"] == {"AAPL"}


# ---------- Verification ----------


def test_verified_buy_has_no_problems() -> None:
    proposal = TradeProposal.model_validate(_buy())
    assert verify_proposal(proposal, _ledger(), max_price_deviation_pct=LIMIT) == []


def test_no_trade_is_never_blocked() -> None:
    proposal = TradeProposal(decision="no_trade", thesis="Weak tape.", confidence="low")
    assert verify_proposal(proposal, ResearchLedger(), max_price_deviation_pct=LIMIT) == []


def test_invented_call_ids_are_rejected() -> None:
    evidence = [
        {"claim": "Quote is tight", "source_call_id": "c1"},
        {"claim": "CEO bought shares", "source_call_id": "made_up"},
    ]
    proposal = TradeProposal.model_validate(_buy(evidence=evidence))
    problems = verify_proposal(proposal, _ledger(), max_price_deviation_pct=LIMIT)
    assert problems == ["evidence cites unknown or failed tool call 'made_up'"]


def test_evidence_must_be_about_the_symbol() -> None:
    ledger = _ledger()
    ledger.record("m1", "get_quote", {"symbol": "MSFT"}, {"symbol": "MSFT", "bid": "400"})
    ledger.record("m2", "get_news", {"symbol": "MSFT"}, {"symbol": "MSFT"})
    evidence = [
        {"claim": "MSFT is strong", "source_call_id": "m1"},
        {"claim": "MSFT news", "source_call_id": "m2"},
    ]
    proposal = TradeProposal.model_validate(_buy(evidence=evidence))
    problems = verify_proposal(proposal, ledger, max_price_deviation_pct=LIMIT)
    assert problems == ["no evidence item cites a tool call about AAPL"]


def test_symbol_without_price_data_is_rejected() -> None:
    ledger = ResearchLedger()
    ledger.record("n1", "get_news", {"symbol": "AAPL"}, {"symbol": "AAPL"})
    ledger.record("n2", "get_asset", {"symbol": "AAPL"}, {"symbol": "AAPL"})
    evidence = [
        {"claim": "Good news", "source_call_id": "n1"},
        {"claim": "Tradable", "source_call_id": "n2"},
    ]
    proposal = TradeProposal.model_validate(_buy(evidence=evidence))
    problems = verify_proposal(proposal, ledger, max_price_deviation_pct=LIMIT)
    assert problems == ["no price data was retrieved for AAPL"]


@pytest.mark.parametrize(
    ("price", "ok"), [(101.0, True), (102.9, True), (103.1, False), (90, False)]
)
def test_reference_price_must_match_observed_prices(price: float, ok: bool) -> None:
    proposal = TradeProposal.model_validate(_buy(reference_price=price))
    problems = verify_proposal(proposal, _ledger(), max_price_deviation_pct=LIMIT)
    assert (problems == []) is ok
    if not ok:
        assert "away from every price the tools returned for AAPL (limit 2%)" in problems[0]


def test_buy_missing_fields_reports_instead_of_crashing() -> None:
    proposal = TradeProposal.model_construct(
        decision="buy", symbol=None, reference_price=None, evidence=(), risks=()
    )
    problems = verify_proposal(proposal, _ledger(), max_price_deviation_pct=LIMIT)
    assert problems == ["a buy proposal needs a symbol and a reference_price"]
