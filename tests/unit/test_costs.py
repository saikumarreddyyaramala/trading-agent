"""Cost ledger (ADR 0014): exact pricing, per-run records, the daily budget, and summaries.

Written before the code, as failing tests: properties over random inputs and failure
injection (store down, prices missing, clock at midnight), not just the happy path.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import NOW
from trading_agent.costs import (
    CostLedger,
    cost_report,
    llm_cost,
    price_for,
    render_costs,
    service_cost_per_day,
)
from trading_agent.domain.models import LLMCallRecord, LLMUsage
from trading_agent.settings.models import AppConfig, ModelPrice, PaidService
from trading_agent.storage.store import AuditError, AuditRecorder, NullAuditStore

PRICES = {
    "gpt-6.1-sol": ModelPrice(input=Decimal("2.50"), output=Decimal("10")),
    "gpt-6": ModelPrice(input=Decimal("1"), output=Decimal("4")),
}
tokens = st.integers(min_value=0, max_value=50_000_000)


def _config(**costs: Any) -> AppConfig:
    values: dict[str, Any] = {
        "llm_prices": {k: v.model_dump() for k, v in PRICES.items()},
        "services": [],
        "daily_llm_budget_usd": "1.00",
    }
    values.update(costs)
    return AppConfig.model_validate({"costs": values})


def _call(
    model: str = "gpt-6.1-sol",
    input_tokens: int = 1_000_000,
    output_tokens: int = 100_000,
    *,
    agent: str = "analyst",
    run_id: str | None = "daily-20261015",
    status: str = "ok",
    at: datetime = NOW,
) -> LLMCallRecord:
    return LLMCallRecord(
        created_at=at,
        provider="openai",
        model=model,
        agent=agent,
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=None if status == "error" else 1200.0,
        error="boom" if status == "error" else None,
    )


# ---------- Pricing ----------


def test_cost_is_exact_in_micro_dollars() -> None:
    usage = LLMUsage(input_tokens=1_000_000, output_tokens=100_000)
    assert llm_cost(PRICES, "gpt-6.1-sol", usage) == Decimal("3.500000")
    assert llm_cost(PRICES, "gpt-6.1-sol", LLMUsage(input_tokens=1)) == Decimal("0.000003")
    assert llm_cost(PRICES, "gpt-6.1-sol", LLMUsage()) == Decimal("0")


def test_unknown_model_has_no_price_never_a_guess() -> None:
    assert price_for(PRICES, "claude-sonnet-5-5") is None
    assert llm_cost(PRICES, "claude-sonnet-5-5", LLMUsage(input_tokens=10)) is None
    assert llm_cost({}, "gpt-6.1-sol", LLMUsage(input_tokens=10)) is None


def test_dated_model_ids_use_the_longest_matching_price() -> None:
    # Providers often answer with a dated id ("gpt-6.1-sol-2026-09-01").
    assert price_for(PRICES, "gpt-6.1-sol-2026-09-01") == PRICES["gpt-6.1-sol"]
    assert price_for(PRICES, "gpt-6-2026-01-01") == PRICES["gpt-6"]
    assert price_for(PRICES, "gpt-6.1-sol") == PRICES["gpt-6.1-sol"]
    assert price_for(PRICES, "gpt-6.1") is None  # a prefix of a key is not a match


@given(a_in=tokens, a_out=tokens, b_in=tokens, b_out=tokens)
def test_cost_is_additive_within_rounding(a_in: int, a_out: int, b_in: int, b_out: int) -> None:
    a = llm_cost(PRICES, "gpt-6.1-sol", LLMUsage(input_tokens=a_in, output_tokens=a_out))
    b = llm_cost(PRICES, "gpt-6.1-sol", LLMUsage(input_tokens=b_in, output_tokens=b_out))
    both = llm_cost(
        PRICES, "gpt-6.1-sol", LLMUsage(input_tokens=a_in + b_in, output_tokens=a_out + b_out)
    )
    assert a is not None
    assert b is not None
    assert both is not None
    assert abs(both - (a + b)) <= Decimal("0.000001")
    assert both >= 0


@given(base=tokens, extra=st.integers(min_value=1, max_value=1_000_000))
def test_more_tokens_never_cost_less(base: int, extra: int) -> None:
    low = llm_cost(PRICES, "gpt-6", LLMUsage(input_tokens=base))
    high = llm_cost(PRICES, "gpt-6", LLMUsage(input_tokens=base + extra))
    assert low is not None
    assert high is not None
    assert high >= low


def test_services_are_spread_over_every_calendar_day() -> None:
    services = (
        PaidService(name="Market data plan", monthly_usd=Decimal("99")),
        PaidService(name="Free broker", monthly_usd=Decimal("0")),
    )
    assert service_cost_per_day(services) == Decimal("3.2548")  # 99 * 12 / 365
    assert service_cost_per_day(()) == Decimal("0")


def test_prices_and_services_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="costs"):
        AppConfig.model_validate({"costs": {"llm_prices": {"m": {"input": -1, "output": 1}}}})
    with pytest.raises(ValueError, match="costs"):
        AppConfig.model_validate({"costs": {"services": [{"name": "x", "monthly_usd": -5}]}})
    with pytest.raises(ValueError, match="costs"):
        AppConfig.model_validate({"costs": {"daily_llm_budget_usd": 0}})


def test_repo_config_costs(config_dir: Any) -> None:
    from trading_agent.settings import load_config

    costs = load_config("paper", config_dir).costs
    assert costs.daily_llm_budget_usd == Decimal("2.00")
    assert costs.services[0].monthly_usd == 0  # Alpaca's free plan
    assert costs.llm_prices == {}  # you set your own prices; none are guessed


# ---------- Ledger ----------


class _Store(NullAuditStore):
    """Keeps llm calls in memory; can be switched off to simulate an outage."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.down = False

    def record_llm_call(self, call: LLMCallRecord, cost_usd: Decimal | None) -> None:
        if self.down:
            raise AuditError("could not record llm call: database down")
        self.rows.append(
            {
                "created_at": call.created_at,
                "run_id": call.run_id,
                "agent": call.agent,
                "provider": call.provider,
                "model": call.model,
                "status": call.status,
                "input_tokens": call.usage.input_tokens,
                "output_tokens": call.usage.output_tokens,
                "latency_ms": call.latency_ms,
                "cost_usd": cost_usd,
                "error": call.error,
            }
        )

    def llm_calls_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        if self.down:
            raise AuditError("could not read llm calls: database down")
        return [r for r in self.rows if start <= r["created_at"] < end]


def _ledger(
    store: _Store, *, config: AppConfig | None = None, now: datetime = NOW, required: bool = False
) -> CostLedger:
    return CostLedger(
        AuditRecorder(store, required=required), config or _config(), clock=lambda: now
    )


def test_ledger_prices_each_call_when_it_happens() -> None:
    store = _Store()
    ledger = _ledger(store)
    ledger.record(_call())
    ledger.record(_call("unknown-model"))
    ledger.record(_call(status="error", input_tokens=0, output_tokens=0))
    assert [r["cost_usd"] for r in store.rows] == [Decimal("3.500000"), None, Decimal("0")]


def test_ledger_write_failure_never_breaks_the_caller() -> None:
    store = _Store()
    store.down = True
    _ledger(store).record(_call())  # optional store: logged, not raised
    _ledger(store, required=True).record(_call())  # cost records never block a decision
    assert store.rows == []


def test_spend_counts_only_today_in_market_time() -> None:
    store = _Store()
    # 2026-10-15 03:30 UTC is still 2026-10-14 in New York.
    yesterday = datetime(2026, 10, 15, 3, 30, tzinfo=UTC)
    store.rows = []
    ledger = _ledger(store)
    ledger.record(_call(at=yesterday))
    ledger.record(_call(at=NOW))
    ledger.record(_call("unknown-model", at=NOW))
    spend = ledger.spend_on(date(2026, 10, 15))
    assert spend.known_usd == Decimal("3.500000")
    assert spend.calls == 2
    assert spend.unpriced_calls == 1
    assert ledger.spend_on(date(2026, 10, 14)).known_usd == Decimal("3.500000")


def test_budget_refuses_once_todays_spend_reaches_it() -> None:
    store = _Store()
    ledger = _ledger(store, config=_config(daily_llm_budget_usd="3.50"))
    assert ledger.budget_reason() is None
    ledger.record(_call())  # exactly $3.50
    reason = ledger.budget_reason()
    assert reason is not None
    assert "daily LLM budget of $3.50 reached" in reason
    assert "$3.50 spent" in reason


def test_budget_resets_the_next_market_day() -> None:
    store = _Store()
    _ledger(store, config=_config(daily_llm_budget_usd="1")).record(_call())
    tomorrow = _ledger(store, config=_config(daily_llm_budget_usd="1"), now=NOW + timedelta(days=1))
    assert tomorrow.budget_reason() is None


def test_no_budget_configured_means_no_limit() -> None:
    store = _Store()
    ledger = _ledger(store, config=_config(daily_llm_budget_usd=None))
    for _ in range(5):
        ledger.record(_call())
    assert ledger.budget_reason() is None


def test_budget_check_fails_open_when_spend_is_unreadable() -> None:
    # The budget limits spending, not trading safety: a database outage must not stop
    # the analyst (the risk engine and the audit rules already handle that).
    store = _Store()
    store.down = True
    assert _ledger(store).budget_reason() is None


# ---------- Summaries ----------


def _rows(*calls: tuple[LLMCallRecord, Decimal | None]) -> list[dict[str, Any]]:
    store = _Store()
    for call, cost in calls:
        store.record_llm_call(call, cost)
    return store.rows


def test_cost_report_by_day_agent_and_model() -> None:
    day_one = datetime(2026, 10, 15, 13, tzinfo=UTC)
    day_two = datetime(2026, 10, 16, 13, tzinfo=UTC)
    rows = _rows(
        (_call(at=day_one), Decimal("3.5")),
        (_call(agent="critic", at=day_one), Decimal("0.25")),
        (_call(status="error", input_tokens=0, output_tokens=0, at=day_one), Decimal("0")),
        (_call(agent="reflector", at=day_two), Decimal("0.10")),
    )
    config = _config(services=[{"name": "Data plan", "monthly_usd": "99"}])
    report = cost_report(rows, config, date(2026, 10, 15), date(2026, 10, 16))

    one, two = report.days
    assert (one.day, one.calls, one.failed_calls) == (date(2026, 10, 15), 3, 1)
    assert one.llm_usd == Decimal("3.75")
    assert one.by_agent["analyst"].usd == Decimal("3.5")
    assert one.by_agent["critic"].usd == Decimal("0.25")
    assert one.services_usd == Decimal("3.2548")
    assert one.total_usd == Decimal("7.0048")
    assert two.llm_usd == Decimal("0.10")
    assert report.total.llm_usd == Decimal("3.85")
    assert report.total.services_usd == Decimal("6.5096")
    assert report.by_model == {"gpt-6.1-sol": Decimal("3.85")}
    assert report.unpriced_models == []


def test_unpriced_calls_make_the_total_unknown_not_low() -> None:
    rows = _rows((_call(), Decimal("3.5")), (_call("mystery"), None))
    report = cost_report(rows, _config(), date(2026, 10, 15), date(2026, 10, 15))
    (day,) = report.days
    assert day.llm_usd is None
    assert day.known_llm_usd == Decimal("3.5")
    assert day.unpriced_calls == 1
    assert day.total_usd is None
    assert report.unpriced_models == ["mystery"]


def test_days_without_calls_still_carry_services() -> None:
    config = _config(services=[{"name": "Data plan", "monthly_usd": "30.4167"}])
    report = cost_report([], config, date(2026, 10, 10), date(2026, 10, 12))
    assert [d.day for d in report.days] == [
        date(2026, 10, 10),
        date(2026, 10, 11),
        date(2026, 10, 12),
    ]
    assert all(d.calls == 0 and d.llm_usd == 0 for d in report.days)
    assert report.days[0].services_usd == Decimal("1.0000")


def test_render_costs_is_honest_about_missing_prices() -> None:
    rows = _rows((_call(), Decimal("3.5")), (_call("mystery"), None))
    text = render_costs(cost_report(rows, _config(), date(2026, 10, 15), date(2026, 10, 15)))
    assert "2026-10-15" in text
    assert "$3.50 + unpriced" in text
    assert "No price set for: mystery" in text
    assert "costs.llm_prices" in text


def test_render_costs_with_everything_priced() -> None:
    rows = _rows((_call(), Decimal("3.5")), (_call(agent="critic"), Decimal("0.5")))
    text = render_costs(cost_report(rows, _config(), date(2026, 10, 15), date(2026, 10, 15)))
    assert "analyst $3.50" in text
    assert "critic $0.50" in text
    assert "Total: $4.00" in text
    assert "No price set" not in text


@settings(max_examples=60)
@given(
    costs=st.lists(
        st.one_of(
            st.none(),
            st.decimals(min_value=0, max_value=100, places=6, allow_nan=False),
        ),
        max_size=40,
    )
)
def test_report_totals_equal_the_sum_of_their_parts(costs: list[Decimal | None]) -> None:
    rows = _rows(*[(_call(model="m" if c is not None else "unpriced"), c) for c in costs])
    report = cost_report(rows, _config(), date(2026, 10, 15), date(2026, 10, 15))
    (day,) = report.days
    known = sum((c for c in costs if c is not None), Decimal("0"))
    assert day.known_llm_usd == known
    assert day.calls == len(costs)
    assert day.unpriced_calls == sum(1 for c in costs if c is None)
    assert (day.llm_usd is None) == any(c is None for c in costs)
    if day.llm_usd is not None:
        assert day.llm_usd == known == report.total.llm_usd


@pytest.mark.parametrize(
    ("value", "shown"),
    [("0.065", "$0.07"), ("0.075", "$0.08"), ("0.004999", "$0.00"), ("1234.005", "$1,234.01")],
)
def test_money_rounds_half_up(value: str, shown: str) -> None:
    from trading_agent.costs.summary import usd

    assert usd(Decimal(value)) == shown
    assert usd(None, Decimal(value)) == f"{shown} + unpriced"
    assert usd(None) == "unpriced"
