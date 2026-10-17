"""Vendor-neutral LLM errors. Adapters translate SDK exceptions into these."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for all LLM failures."""


class LLMProviderError(LLMError):
    """A single provider call failed.

    `fatal` marks errors that another model cannot fix (bad credentials, no permission),
    so the gateway stops instead of wasting a fallback attempt.
    """

    def __init__(self, provider: str, model: str, message: str, *, fatal: bool = False) -> None:
        super().__init__(f"[{provider}:{model}] {message}")
        self.provider = provider
        self.model = model
        self.fatal = fatal


class LLMUnavailableError(LLMError):
    """Every configured model failed. Callers must treat this as 'no decision' (fail safe)."""


class LLMBudgetExceededError(LLMError):
    """The daily LLM budget is spent (ADR 0014). No call was made; callers fail safe."""
