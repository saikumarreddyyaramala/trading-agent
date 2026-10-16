"""Anthropic (Claude) adapter for the LLM gateway."""

from __future__ import annotations

import time
from typing import Any

import anthropic
from anthropic.types import MessageParam, TextBlock

from trading_agent.domain.models import LLMRequest, LLMResponse, LLMUsage
from trading_agent.llm.errors import LLMProviderError

_FATAL_ERRORS: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
)


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
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        messages: list[MessageParam] = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        started = time.perf_counter()
        try:
            result = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=request.system,
                messages=messages,
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
        )
