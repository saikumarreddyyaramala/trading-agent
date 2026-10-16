from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from trading_agent.domain.models import ChatMessage, LLMRequest, ToolCall, ToolResult, ToolSpec
from trading_agent.llm.anthropic_provider import AnthropicProvider, message_params
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
            SimpleNamespace(type="thinking"),  # other blocks are ignored
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


# ---------- Tool calling ----------

TOOL = ToolSpec(
    name="get_quote",
    description="Latest quote",
    input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
)


def _message(*content: object, stop_reason: str = "tool_use") -> Message:
    return Message.model_construct(
        content=list(content),
        model="claude-test",
        stop_reason=stop_reason,
        usage=Usage.model_construct(input_tokens=1, output_tokens=1),
    )


def test_tools_are_sent_and_tool_use_is_returned() -> None:
    sdk_message = _message(
        TextBlock.model_construct(type="text", text="Checking."),
        ToolUseBlock.model_construct(
            type="tool_use", id="toolu_1", name="get_quote", input={"symbol": "AAPL"}
        ),
    )
    provider, messages = _provider(sdk_message)
    request = REQUEST.model_copy(update={"tools": (TOOL,)})

    response = provider.complete(request, model="claude-test", max_tokens=50)

    assert messages.kwargs["tools"] == [
        {"name": "get_quote", "description": "Latest quote", "input_schema": TOOL.input_schema}
    ]
    assert response.text == "Checking."
    assert response.tool_calls == (
        ToolCall(id="toolu_1", name="get_quote", arguments_json='{"symbol": "AAPL"}'),
    )
    assert response.stop_reason == "tool_use"


def test_no_tools_key_without_tools() -> None:
    provider, messages = _provider(_message(TextBlock.model_construct(type="text", text="x")))
    provider.complete(REQUEST, model="claude-test", max_tokens=50)
    assert "tools" not in messages.kwargs


def test_tool_turns_become_content_blocks() -> None:
    call = ToolCall(id="t1", name="get_quote", arguments_json='{"symbol": "AAPL"}')
    bad = ToolCall(id="t2", name="get_quote", arguments_json="not json")
    messages = (
        ChatMessage(role="user", content="research"),
        ChatMessage(role="assistant", content="Looking.", tool_calls=(call, bad)),
        ChatMessage(
            role="user",
            tool_results=(
                ToolResult(call_id="t1", content='{"bid":"1"}'),
                ToolResult(call_id="t2", content="invalid", is_error=True),
            ),
        ),
    )
    assert message_params(messages) == [
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "id": "t1", "name": "get_quote", "input": {"symbol": "AAPL"}},
                {"type": "tool_use", "id": "t2", "name": "get_quote", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": '{"bid":"1"}',
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "invalid",
                    "is_error": True,
                },
            ],
        },
    ]


def test_non_object_tool_arguments_become_empty_input() -> None:
    call = ToolCall(id="t1", name="get_quote", arguments_json="[1, 2]")
    params = message_params((ChatMessage(role="assistant", tool_calls=(call,)),))
    assert params[0]["content"][0]["input"] == {}
