"""LLM gateway: the single entry point every agent uses to call a model.

Why a gateway instead of calling the SDK directly (the enterprise pattern):
- Vendor independence: agents depend on `LLMGateway`, not on Anthropic or OpenAI.
- Model pinning and fallback live in config, not scattered through agent code.
- One place for usage logging, token caps, the cost ledger, and the daily budget.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from trading_agent.domain.models import LLMCallRecord, LLMRequest, LLMResponse, LLMUsage
from trading_agent.domain.ports import LLMProviderPort
from trading_agent.llm.errors import (
    LLMBudgetExceededError,
    LLMProviderError,
    LLMUnavailableError,
)
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import LLMConfig


class LLMGateway:
    def __init__(
        self,
        provider: LLMProviderPort,
        config: LLMConfig,
        logger: Any = None,
        *,
        listener: Callable[[LLMCallRecord], None] | None = None,
        budget: Callable[[], str | None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """`listener` receives every attempt (the cost ledger). `budget` returns a reason to
        refuse the call, or None. Neither can break a call: their failures are logged."""
        self._provider = provider
        self._config = config
        self._log = logger or get_logger(__name__)
        self._usage = LLMUsage()
        self._listener = listener
        self._budget = budget
        self._clock = clock or (lambda: datetime.now(UTC))

    def _refusal(self) -> str | None:
        if self._budget is None:
            return None
        try:
            return self._budget()
        except Exception as exc:  # fail open: the budget limits spend, not trading safety
            self._log.warning("llm.budget_check_failed", error=str(exc))
            return None

    def _report(
        self,
        request: LLMRequest,
        model: str,
        *,
        response: LLMResponse | None = None,
        error: str | None = None,
    ) -> None:
        if self._listener is None:
            return
        try:
            self._listener(
                LLMCallRecord(
                    created_at=self._clock(),
                    provider=response.provider if response else self._provider.name,
                    model=response.model if response else model,
                    agent=request.tags.get("agent", "other"),
                    run_id=request.tags.get("run_id"),
                    status="ok" if response else "error",
                    usage=response.usage if response else LLMUsage(),
                    latency_ms=response.latency_ms if response else None,
                    error=error,
                )
            )
        except Exception as exc:  # the ledger must never break a model call
            self._log.warning("llm.ledger_failed", error=str(exc))

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
        refusal = self._refusal()
        if refusal:
            self._log.warning("llm.budget_refused", reason=refusal, **request.tags)
            raise LLMBudgetExceededError(refusal)

        for attempt, model in enumerate(self.models):
            try:
                response = self._provider.complete(request, model=model, max_tokens=max_tokens)
            except LLMProviderError as exc:
                last_error = exc
                self._report(request, model, error=str(exc))
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

            self._report(request, model, response=response)
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
                tool_calls=len(response.tool_calls),
                **request.tags,
            )
            return response

        raise LLMUnavailableError(
            f"all models failed ({', '.join(self.models)}); last error: {last_error}"
        ) from last_error
