"""Anthropic (Claude) adapter for the LLM gateway."""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from trading_agent.domain.models import ChatMessage, LLMRequest, LLMResponse, LLMUsage, ToolCall
from trading_agent.llm.errors import LLMProviderError

_FATAL_ERRORS: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
)


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def message_params(messages: tuple[ChatMessage, ...]) -> list[dict[str, Any]]:
    """Conversation turns as Messages API params. Tool results must lead a user turn."""
    params: list[dict[str, Any]] = []
    for message in messages:
        if not message.tool_calls and not message.tool_results:
            params.append({"role": message.role, "content": message.content})
            continue
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        blocks.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": _json_object(call.arguments_json),
            }
            for call in message.tool_calls
        )
        params.append({"role": message.role, "content": blocks})
    return params


class AnthropicProvider:
    """Implements `LLMProviderPort` on top of the official Anthropic SDK.

    The SDK already retries transient failures (`max_retries`) with backoff; this adapter
    only translates requests, responses, and errors into vendor-neutral types.
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self._client: Any = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        options: dict[str, Any] = {}
        if request.tools:
            options["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        started = time.perf_counter()
        try:
            result = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=request.system,
                messages=message_params(request.messages),
                **options,
            )
        except anthropic.AnthropicError as exc:
            raise LLMProviderError(
                self.name,
                model,
                f"{type(exc).__name__}: {exc}",
                fatal=isinstance(exc, _FATAL_ERRORS),
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        text = "".join(block.text for block in result.content if isinstance(block, TextBlock))
        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(result.model),
            stop_reason=None if result.stop_reason is None else str(result.stop_reason),
            usage=LLMUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
            latency_ms=round(latency_ms, 1),
            tool_calls=tuple(
                ToolCall(id=block.id, name=block.name, arguments_json=json.dumps(block.input))
                for block in result.content
                if isinstance(block, ToolUseBlock)
            ),
        )
