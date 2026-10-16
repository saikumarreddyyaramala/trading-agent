from __future__ import annotations

import pytest

from tests.conftest import FakeProvider, provider_error
from trading_agent.domain.models import ChatMessage, LLMRequest
from trading_agent.llm import LLMGateway, LLMUnavailableError
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
