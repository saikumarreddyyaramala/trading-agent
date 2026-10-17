from __future__ import annotations

import contextlib
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import FakeProvider, provider_error
from trading_agent.domain.models import ChatMessage, LLMCallRecord, LLMRequest
from trading_agent.llm import LLMBudgetExceededError, LLMGateway, LLMUnavailableError
from trading_agent.settings.models import LLMConfig

REQUEST = LLMRequest(
    system="sys",
    messages=(ChatMessage(role="user", content="hi"),),
    tags={"run_id": "r1"},
)


def _gateway(provider: FakeProvider, **config: object) -> LLMGateway:
    defaults: dict[str, object] = {
        "model": "primary",
        "fallback_model": "backup",
        "max_tokens": 100,
    }
    defaults.update(config)
    return LLMGateway(provider, LLMConfig.model_validate(defaults))


def test_uses_primary_model_when_it_works() -> None:
    provider = FakeProvider(["hello"])
    response = _gateway(provider).complete(REQUEST)
    assert response.text == "hello"
    assert provider.calls == [("primary", 100)]


def test_falls_back_when_primary_fails() -> None:
    provider = FakeProvider([provider_error("primary"), "from backup"])
    response = _gateway(provider).complete(REQUEST)
    assert response.text == "from backup"
    assert [model for model, _ in provider.calls] == ["primary", "backup"]


def test_fatal_error_skips_fallback() -> None:
    provider = FakeProvider([provider_error("primary", fatal=True), "never used"])
    with pytest.raises(LLMUnavailableError):
        _gateway(provider).complete(REQUEST)
    assert len(provider.calls) == 1


def test_raises_when_every_model_fails() -> None:
    provider = FakeProvider([provider_error("primary"), provider_error("backup")])
    with pytest.raises(LLMUnavailableError, match="all models failed"):
        _gateway(provider).complete(REQUEST)


def test_no_duplicate_call_when_fallback_equals_primary() -> None:
    provider = FakeProvider([provider_error("primary")])
    gateway = _gateway(provider, fallback_model="primary")
    assert gateway.models == ("primary",)
    with pytest.raises(LLMUnavailableError):
        gateway.complete(REQUEST)
    assert len(provider.calls) == 1


def test_caller_cannot_exceed_token_cap() -> None:
    provider = FakeProvider(["a", "b"])
    gateway = _gateway(provider, max_tokens=100)
    gateway.complete(REQUEST.model_copy(update={"max_tokens": 5000}))
    gateway.complete(REQUEST.model_copy(update={"max_tokens": 20}))
    assert [tokens for _, tokens in provider.calls] == [100, 20]


def test_tracks_cumulative_usage() -> None:
    provider = FakeProvider(["a", "b"])
    gateway = _gateway(provider)
    gateway.complete(REQUEST)
    gateway.complete(REQUEST)
    assert gateway.total_usage.input_tokens == 20
    assert gateway.total_usage.output_tokens == 4


# ---------- Cost ledger hooks (ADR 0014) ----------


AT = datetime(2026, 10, 15, 14, tzinfo=UTC)


def _hooked(
    provider: FakeProvider,
    *,
    listener: object = None,
    budget: object = None,
    **config: object,
) -> LLMGateway:
    defaults: dict[str, object] = {"model": "primary", "fallback_model": "backup"}
    defaults.update(config)
    return LLMGateway(
        provider,
        LLMConfig.model_validate(defaults),
        listener=listener,  # type: ignore[arg-type]
        budget=budget,  # type: ignore[arg-type]
        clock=lambda: AT,
    )


def test_every_call_is_reported_including_failed_attempts() -> None:
    calls: list[LLMCallRecord] = []
    provider = FakeProvider([provider_error("primary"), "from backup"])
    tagged = REQUEST.model_copy(update={"tags": {"run_id": "daily-1", "agent": "critic"}})
    _hooked(provider, listener=calls.append).complete(tagged)

    failed, ok = calls
    assert (failed.model, failed.status, failed.usage.input_tokens) == ("primary", "error", 0)
    assert failed.error is not None
    assert "boom" in failed.error
    assert (ok.model, ok.status, ok.usage.input_tokens, ok.usage.output_tokens) == (
        "backup",
        "ok",
        10,
        2,
    )
    assert all(c.agent == "critic" and c.run_id == "daily-1" for c in calls)
    assert ok.created_at == AT
    assert ok.provider == "fake"
    assert ok.latency_ms == 5.0


def test_untagged_calls_are_attributed_to_other() -> None:
    calls: list[LLMCallRecord] = []
    request = REQUEST.model_copy(update={"tags": {}})
    _hooked(FakeProvider(["hi"]), listener=calls.append).complete(request)
    assert (calls[0].agent, calls[0].run_id) == ("other", None)


def test_a_broken_listener_never_breaks_the_call() -> None:
    def explode(call: LLMCallRecord) -> None:
        raise RuntimeError("ledger down")

    response = _hooked(FakeProvider(["hello"]), listener=explode).complete(REQUEST)
    assert response.text == "hello"


def test_budget_refusal_stops_before_any_model_is_called() -> None:
    provider = FakeProvider(["never"])
    gateway = _hooked(provider, budget=lambda: "daily LLM budget of $1.00 reached")
    with pytest.raises(LLMBudgetExceededError, match=r"budget of \$1.00 reached"):
        gateway.complete(REQUEST)
    assert provider.calls == []


def test_budget_is_checked_on_every_call() -> None:
    reasons: list[str | None] = [None, "over budget"]
    provider = FakeProvider(["first", "second"])
    gateway = _hooked(provider, budget=lambda: reasons.pop(0))
    assert gateway.complete(REQUEST).text == "first"
    with pytest.raises(LLMBudgetExceededError):
        gateway.complete(REQUEST)
    assert len(provider.calls) == 1


def test_a_broken_budget_check_fails_open() -> None:
    def broken() -> str | None:
        raise RuntimeError("cannot read spend")

    assert _hooked(FakeProvider(["ok"]), budget=broken).complete(REQUEST).text == "ok"


def test_budget_error_is_an_llm_error_so_agents_fail_safe() -> None:
    from trading_agent.llm.errors import LLMError

    assert issubclass(LLMBudgetExceededError, LLMError)


@given(outcomes=st.lists(st.sampled_from(["ok", "retry", "fatal"]), min_size=1, max_size=12))
def test_ledger_sees_exactly_the_attempts_the_provider_saw(outcomes: list[str]) -> None:
    script: list[object] = [
        "text" if o == "ok" else provider_error("m", fatal=o == "fatal") for o in outcomes
    ] + ["text"]  # a retry may fall back to the next model, so the script ends with an answer
    provider = FakeProvider(script)  # type: ignore[arg-type]
    calls: list[LLMCallRecord] = []
    gateway = _hooked(provider, listener=calls.append)
    while provider.outcomes:
        with contextlib.suppress(LLMUnavailableError):
            gateway.complete(REQUEST)
    assert len(calls) == len(provider.calls)
    assert [c.model for c in calls] == [m for m, _ in provider.calls]
    ok = [c for c in calls if c.status == "ok"]
    assert gateway.total_usage.input_tokens == sum(c.usage.input_tokens for c in ok)
