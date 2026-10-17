"""Exact prices: Decimal arithmetic only, and no price means no number, never a guess."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal

from trading_agent.domain.models import LLMUsage
from trading_agent.settings.models import ModelPrice, PaidService

MICRO = Decimal("0.000001")  # LLM costs are kept to the micro-dollar
CENTI_CENT = Decimal("0.0001")
_MILLION = Decimal(1_000_000)
_DAYS_PER_MONTH = Decimal(365) / Decimal(12)


def price_for(prices: Mapping[str, ModelPrice], model: str) -> ModelPrice | None:
    """The price for `model`: an exact match, or the longest key it extends with a dash
    (providers answer "gpt-6.1-sol-2026-09-01" for "gpt-6.1-sol")."""
    if model in prices:
        return prices[model]
    matches = [key for key in prices if model.startswith(f"{key}-")]
    return prices[max(matches, key=len)] if matches else None


def llm_cost(prices: Mapping[str, ModelPrice], model: str, usage: LLMUsage) -> Decimal | None:
    price = price_for(prices, model)
    if price is None:
        return None
    total = (usage.input_tokens * price.input + usage.output_tokens * price.output) / _MILLION
    return total.quantize(MICRO, rounding=ROUND_HALF_UP)


def service_cost_per_day(services: Iterable[PaidService]) -> Decimal:
    """Monthly subscriptions spread evenly over every calendar day of the year."""
    monthly = sum((s.monthly_usd for s in services), Decimal(0))
    return (monthly / _DAYS_PER_MONTH).quantize(CENTI_CENT, rounding=ROUND_HALF_UP)
