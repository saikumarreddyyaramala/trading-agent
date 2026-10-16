from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from trading_agent.domain.models import ChatMessage, LLMRequest, ToolCall, ToolResult, ToolSpec
from trading_agent.llm.errors import LLMProviderError
from trading_agent.llm.openai_provider import OpenAIProvider, input_items

REQUEST = LLMRequest(system="be brief", messages=(ChatMessage(role="user", content="hi"),))


def _sdk_error(cls: type[Exception], message: str = "boom") -> Exception:
    """Build an SDK exception without an HTTP response (constructors need live httpx objects)."""
    exc = cls.__new__(cls)
    Exception.__init__(exc, message)
    return exc


class _Responses:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _provider(result: Any, **kwargs: Any) -> tuple[OpenAIProvider, _Responses]:
    responses = _Responses(result)
    client = SimpleNamespace(responses=responses)
    provider = OpenAIProvider("key", timeout_seconds=5, max_retries=0, client=client, **kwargs)
    return provider, responses


def _result(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "output_text": "OK",
        "model": "gpt-6.1-sol-2026-09-29",
        "status": "completed",
        "incomplete_details": None,
        "usage": SimpleNamespace(input_tokens=20, output_tokens=5),
        "output": [SimpleNamespace(type="message")],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_translates_request_and_response() -> None:
    provider, responses = _provider(_result(), reasoning_effort="medium")

    response = provider.complete(REQUEST, model="gpt-6.1-sol", max_tokens=300)

    assert responses.kwargs == {
        "model": "gpt-6.1-sol",
        "instructions": "be brief",
        "input": [{"role": "user", "content": "hi"}],
        "max_output_tokens": 300,
        "store": False,
        "reasoning": {"effort": "medium"},
    }
    assert response.text == "OK"
    assert response.provider == "openai"
    assert response.stop_reason == "completed"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 5


def test_reasoning_is_omitted_when_not_configured() -> None:
    provider, responses = _provider(_result())
    provider.complete(REQUEST, model="m", max_tokens=10)
    assert "reasoning" not in responses.kwargs


def test_incomplete_response_reports_reason() -> None:
    result = _result(
        output_text="",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    provider, _ = _provider(result)
    response = provider.complete(REQUEST, model="m", max_tokens=10)
    assert response.stop_reason == "incomplete:max_output_tokens"
    assert response.text == ""


def test_missing_usage_counts_as_zero() -> None:
    provider, _ = _provider(_result(usage=None))
    assert provider.complete(REQUEST, model="m", max_tokens=10).usage.input_tokens == 0


@pytest.mark.parametrize(
    ("error_cls", "fatal"),
    [
        (openai.AuthenticationError, True),
        (openai.PermissionDeniedError, True),
        (openai.RateLimitError, False),
        (openai.APIConnectionError, False),
        (openai.NotFoundError, False),
    ],
)
def test_sdk_errors_become_provider_errors(error_cls: type[Exception], fatal: bool) -> None:
    provider, _ = _provider(_sdk_error(error_cls))
    with pytest.raises(LLMProviderError) as caught:
        provider.complete(REQUEST, model="gpt-6.1-sol", max_tokens=10)
    assert caught.value.fatal is fatal
    assert caught.value.provider == "openai"


# ---------- Tool calling ----------

TOOL = ToolSpec(
    name="get_quote",
    description="Latest quote",
    input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
)


def test_tools_are_sent_as_functions() -> None:
    provider, responses = _provider(_result())
    provider.complete(REQUEST.model_copy(update={"tools": (TOOL,)}), model="m", max_tokens=10)
    assert responses.kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_quote",
            "description": "Latest quote",
            "parameters": TOOL.input_schema,
            "strict": False,
        }
    ]


def test_no_tools_key_without_tools() -> None:
    provider, responses = _provider(_result())
    provider.complete(REQUEST, model="m", max_tokens=10)
    assert "tools" not in responses.kwargs


def test_function_calls_are_returned() -> None:
    output = [
        SimpleNamespace(type="reasoning"),
        SimpleNamespace(
            type="function_call", call_id="call_1", name="get_quote", arguments='{"symbol":"AAPL"}'
        ),
    ]
    provider, _ = _provider(_result(output_text="", output=output))
    response = provider.complete(REQUEST, model="m", max_tokens=10)
    assert response.tool_calls == (
        ToolCall(id="call_1", name="get_quote", arguments_json='{"symbol":"AAPL"}'),
    )


def test_tool_turns_become_function_items_in_order() -> None:
    call = ToolCall(id="c1", name="get_quote", arguments_json='{"symbol":"AAPL"}')
    messages = (
        ChatMessage(role="user", content="research"),
        ChatMessage(role="assistant", content="Looking.", tool_calls=(call,)),
        ChatMessage(
            role="user",
            content="continue",
            tool_results=(
                ToolResult(call_id="c1", content='{"bid":"1"}'),
                ToolResult(call_id="c2", content="bad symbol", is_error=True),
            ),
        ),
    )
    assert input_items(messages) == [
        {"role": "user", "content": "research"},
        {"role": "assistant", "content": "Looking."},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_quote",
            "arguments": '{"symbol":"AAPL"}',
        },
        {"type": "function_call_output", "call_id": "c1", "output": '{"bid":"1"}'},
        {"type": "function_call_output", "call_id": "c2", "output": "ERROR: bad symbol"},
        {"role": "user", "content": "continue"},
    ]
