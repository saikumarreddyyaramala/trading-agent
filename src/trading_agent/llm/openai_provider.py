"""OpenAI (GPT) adapter for the LLM gateway, using the Responses API."""

from __future__ import annotations

import time
from typing import Any, Literal

import openai
from openai.types.responses import EasyInputMessageParam

from trading_agent.domain.models import LLMRequest, LLMResponse, LLMUsage
from trading_agent.llm.errors import LLMProviderError

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

_FATAL_ERRORS: tuple[type[Exception], ...] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)


class OpenAIProvider:
    """Implements `LLMProviderPort` on top of the official OpenAI SDK.

    GPT reasoning models spend part of `max_output_tokens` on hidden reasoning, so callers
    should leave headroom. The SDK retries transient failures (`max_retries`) itself.
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float,
        max_retries: int,
        reasoning_effort: ReasoningEffort | None = None,
        client: Any | None = None,
    ) -> None:
        self._reasoning_effort = reasoning_effort
        self._client: Any = client or openai.OpenAI(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def name(self) -> str:
        return "openai"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        messages: list[EasyInputMessageParam] = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        options: dict[str, Any] = {}
        if self._reasoning_effort is not None:
            options["reasoning"] = {"effort": self._reasoning_effort}

        started = time.perf_counter()
        try:
            result = self._client.responses.create(
                model=model,
                instructions=request.system,
                input=messages,
                max_output_tokens=max_tokens,
                store=False,  # do not keep request data on the vendor side
                **options,
            )
        except openai.OpenAIError as exc:
            raise LLMProviderError(
                self.name,
                model,
                f"{type(exc).__name__}: {exc}",
                fatal=isinstance(exc, _FATAL_ERRORS),
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        stop_reason: str | None
        if result.status == "incomplete":
            reason = getattr(result.incomplete_details, "reason", None) or "unknown"
            stop_reason = f"incomplete:{reason}"
        else:
            stop_reason = None if result.status is None else str(result.status)

        usage = result.usage
        return LLMResponse(
            text=result.output_text,
            provider=self.name,
            model=str(result.model),
            stop_reason=stop_reason,
            usage=LLMUsage(
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            ),
            latency_ms=round(latency_ms, 1),
        )
