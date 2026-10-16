"""Ports: the interfaces the core depends on (hexagonal / ports-and-adapters architecture).

Business logic talks only to these protocols. Concrete adapters live in `execution/`,
`llm/`, `data/` and can be swapped (another broker, another LLM vendor, a fake in tests)
without touching the core.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_agent.domain.models import AccountSnapshot, LLMRequest, LLMResponse, MarketClock


@runtime_checkable
class BrokerPort(Protocol):
    """Read-only broker access for now.

    Order placement is added in Phase 5, behind the risk engine.
    """

    @property
    def is_paper(self) -> bool: ...

    def get_account(self) -> AccountSnapshot: ...

    def get_clock(self) -> MarketClock: ...


@runtime_checkable
class LLMProviderPort(Protocol):
    """One LLM vendor. The gateway composes providers and adds fallback, limits, and logging."""

    @property
    def name(self) -> str: ...

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse: ...
