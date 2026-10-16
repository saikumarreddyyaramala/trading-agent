"""LLM gateway: the single entry point every agent uses to call a model.

Why a gateway instead of calling the SDK directly (the enterprise pattern):
- Vendor independence: agents depend on `LLMGateway`, not on Anthropic or OpenAI.
- Model pinning and fallback live in config, not scattered through agent code.
- One place for usage logging, token caps, and (later) cost budgets and tracing.
"""

from __future__ import annotations

from typing import Any

from trading_agent.domain.models import LLMRequest, LLMResponse, LLMUsage
from trading_agent.domain.ports import LLMProviderPort
from trading_agent.llm.errors import LLMProviderError, LLMUnavailableError
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import LLMConfig


class LLMGateway:
    def __init__(self, provider: LLMProviderPort, config: LLMConfig, logger: Any = None) -> None:
        self._provider = provider
        self._config = config
        self._log = logger or get_logger(__name__)
        self._usage = LLMUsage()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def models(self) -> tuple[str, ...]:
        """Primary model first, then the fallback (if configured and different)."""
        fallback = self._config.fallback_model
        if fallback and fallback != self._config.model:
            return (self._config.model, fallback)
        return (self._config.model,)

    @property
    def total_usage(self) -> LLMUsage:
        """Tokens used by this gateway instance (one instance per run gives per-run cost)."""
        return self._usage

    def complete(self, request: LLMRequest) -> LLMResponse:
        # A caller may ask for fewer tokens than the configured cap, never more.
        max_tokens = min(request.max_tokens or self._config.max_tokens, self._config.max_tokens)
        last_error: LLMProviderError | None = None

        for attempt, model in enumerate(self.models):
            try:
                response = self._provider.complete(request, model=model, max_tokens=max_tokens)
            except LLMProviderError as exc:
                last_error = exc
                self._log.warning(
                    "llm.call_failed",
                    provider=self._provider.name,
                    model=model,
                    attempt=attempt + 1,
                    fatal=exc.fatal,
                    error=str(exc),
                    **request.tags,
                )
                if exc.fatal:
                    break
                continue

            self._usage = LLMUsage(
                input_tokens=self._usage.input_tokens + response.usage.input_tokens,
                output_tokens=self._usage.output_tokens + response.usage.output_tokens,
            )
            self._log.info(
                "llm.call_succeeded",
                provider=response.provider,
                model=response.model,
                fallback_used=attempt > 0,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=response.latency_ms,
                stop_reason=response.stop_reason,
                **request.tags,
            )
            return response

        raise LLMUnavailableError(
            f"all models failed ({', '.join(self.models)}); last error: {last_error}"
        ) from last_error
