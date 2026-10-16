"""Turn the model's final answer into a verified `TradeProposal`.

The model's claims are not trusted. A buy is accepted only if:
- every evidence item cites a tool call that actually ran and succeeded in this session,
- at least one cited call is about the proposed symbol,
- price data for the symbol was actually retrieved, and
- the reference price is within `max_price_deviation_pct` of a price the tools returned.

This catches hallucinated sources, a symbol that was never researched, and invented prices.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from trading_agent.domain.models import TradeProposal

_FENCED = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class ProposalError(ValueError):
    """The final answer is not a valid proposal. The message is safe to show the model."""


def extract_json_object(text: str) -> Any:
    """The JSON object in a model's answer: fenced (```json), or the outermost braces."""
    fenced = _FENCED.search(text)
    if fenced:
        raw = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ProposalError("the answer contains no JSON object")
        raw = text[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"the answer is not valid JSON: {exc.msg}") from exc


def schema_problems(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}" for err in exc.errors()
    )


def parse_proposal(text: str) -> TradeProposal:
    """Extract and validate the JSON proposal from the model's final message."""
    data = extract_json_object(text)
    try:
        return TradeProposal.model_validate(data)
    except ValidationError as exc:
        raise ProposalError(
            f"the JSON does not match the proposal schema: {schema_problems(exc)}"
        ) from exc


def _decimal(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and number > 0 else None


@dataclass
class ResearchLedger:
    """What the tools actually returned in this session, keyed by tool call id."""

    calls: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    symbols_by_call: dict[str, set[str]] = field(default_factory=dict)
    prices: dict[str, list[Decimal]] = field(default_factory=dict)

    def record(self, call_id: str, tool: str, arguments: Mapping[str, Any], result: Any) -> None:
        """Record a successful call and the symbols and prices it touched."""
        self.calls[call_id] = (tool, dict(arguments))
        symbols: set[str] = set()
        for key in ("symbol", "symbols"):
            value = arguments.get(key)
            values = value if isinstance(value, list) else [value]
            symbols.update(str(v).strip().upper() for v in values if isinstance(v, str))
        if isinstance(result, Mapping):
            symbols.update(self._extract_prices(tool, result))
        self.symbols_by_call[call_id] = symbols

    def _extract_prices(self, tool: str, result: Mapping[str, Any]) -> set[str]:
        found: set[str] = set()

        def add(symbol: Any, *values: Any) -> None:
            if not isinstance(symbol, str):
                return
            found.add(symbol)
            for value in values:
                price = _decimal(value)
                if price is not None:
                    self.prices.setdefault(symbol, []).append(price)

        if tool == "get_quote":
            add(result.get("symbol"), result.get("bid"), result.get("ask"), result.get("mid"))
        elif tool == "get_daily_bars":
            summary = result.get("summary")
            if isinstance(summary, Mapping):
                add(result.get("symbol"), summary.get("last_close"))
        elif tool == "research_snapshot":
            items = result.get("symbols")
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, Mapping):
                    continue
                quote = item.get("quote")
                quote_prices = (
                    (quote.get("bid"), quote.get("ask")) if isinstance(quote, Mapping) else ()
                )
                add(item.get("symbol"), item.get("last_close"), *quote_prices)
        elif isinstance(result.get("symbol"), str):
            found.add(result["symbol"])
        return found

    def nearest_price_gap_pct(self, symbol: str, price: Decimal) -> Decimal | None:
        observed = self.prices.get(symbol)
        if not observed:
            return None
        return min(abs(price - seen) / seen * 100 for seen in observed)


def verify_proposal(
    proposal: TradeProposal, ledger: ResearchLedger, *, max_price_deviation_pct: Decimal
) -> list[str]:
    """Problems that make a buy proposal untrustworthy (empty list means accepted)."""
    if proposal.decision != "buy":
        return []
    if proposal.symbol is None or proposal.reference_price is None:
        return ["a buy proposal needs a symbol and a reference_price"]
    symbol = proposal.symbol.strip().upper()
    problems = [
        f"evidence cites unknown or failed tool call '{item.source_call_id}'"
        for item in proposal.evidence
        if item.source_call_id not in ledger.calls
    ]
    cited: Iterable[str] = (e.source_call_id for e in proposal.evidence)
    if not any(symbol in ledger.symbols_by_call.get(call_id, set()) for call_id in cited):
        problems.append(f"no evidence item cites a tool call about {symbol}")

    gap = ledger.nearest_price_gap_pct(symbol, proposal.reference_price)
    if gap is None:
        problems.append(f"no price data was retrieved for {symbol}")
    elif gap > max_price_deviation_pct:
        problems.append(
            f"reference_price {proposal.reference_price} is {gap:.2f}% away from every price "
            f"the tools returned for {symbol} (limit {max_price_deviation_pct}%)"
        )
    return problems
