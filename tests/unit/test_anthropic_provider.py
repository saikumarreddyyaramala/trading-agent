from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from trading_agent.domain.models import ChatMessage, LLMRequest
from trading_agent.llm.anthropic_provider import AnthropicProvider
from trading_agent.llm.errors import LLMProviderError

REQUEST = LLMRequest(system="be brief", messages=(ChatMessage(role="user", content="hi"),))


def _sdk_error(cls: type[Exception], message: str = "boom") -> Exception:
    """Build an SDK exception without an HTTP response (constructors need live httpx objects)."""
    exc = cls.__new__(cls)
    Exception.__init__(exc, message)
    return exc


class _Messages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _provider(result: Any) -> tuple[AnthropicProvider, _Messages]:
    messages = _Messages(result)
    client = SimpleNamespace(messages=messages)
    return AnthropicProvider("key", timeout_seconds=5, max_retries=0, client=client), messages


def test_translates_request_and_response() -> None:
    sdk_message = Message.model_construct(
        content=[
            TextBlock.model_construct(type="text", text="Hello "),
            SimpleNamespace(type="tool_use"),  # non-text blocks are ignored
            TextBlock.model_construct(type="text", text="world"),
        ],
        model="claude-test",
        stop_reason="end_turn",
        usage=Usage.model_construct(input_tokens=12, output_tokens=3),
    )
    provider, messages = _provider(sdk_message)

    response = provider.complete(REQUEST, model="claude-test", max_tokens=50)

    assert messages.kwargs == {
        "model": "claude-test",
        "max_tokens": 50,
        "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert response.text == "Hello world"
    assert response.provider == "anthropic"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.stop_reason == "end_turn"


@pytest.mark.parametrize(
    ("error_cls", "fatal"),
    [
        (anthropic.AuthenticationError, True),
        (anthropic.PermissionDeniedError, True),
        (anthropic.RateLimitError, False),
        (anthropic.APIConnectionError, False),
        (anthropic.InternalServerError, False),
    ],
)
def test_sdk_errors_become_provider_errors(error_cls: type[Exception], fatal: bool) -> None:
    provider, _ = _provider(_sdk_error(error_cls))
    with pytest.raises(LLMProviderError) as caught:
        provider.complete(REQUEST, model="claude-test", max_tokens=10)
    assert caught.value.fatal is fatal
    assert caught.value.model == "claude-test"
