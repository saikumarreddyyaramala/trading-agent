"""Composition root: the only place that wires concrete adapters to ports.

Everything else receives its dependencies as arguments, which keeps business logic
testable with fakes and lets us swap vendors through config.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent.domain.ports import BrokerPort
from trading_agent.execution.alpaca_broker import AlpacaBroker
from trading_agent.llm.anthropic_provider import AnthropicProvider
from trading_agent.llm.gateway import LLMGateway
from trading_agent.settings.loader import ConfigError
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import BROKER_SECRET_FIELDS, LLM_SECRET_FIELDS, Secrets


@dataclass(frozen=True)
class Container:
    config: AppConfig
    secrets: Secrets

    def build_broker(self) -> BrokerPort:
        missing = self.secrets.missing(*BROKER_SECRET_FIELDS[self.config.broker.provider])
        if missing:
            raise ConfigError(f"missing broker credentials: {', '.join(missing)}")
        return AlpacaBroker(
            self.secrets.require("alpaca_api_key"),
            self.secrets.require("alpaca_secret_key"),
            paper=self.config.broker.paper,
        )

    def build_llm_gateway(self) -> LLMGateway:
        llm = self.config.llm
        missing = self.secrets.missing(*LLM_SECRET_FIELDS[llm.provider])
        if missing:
            raise ConfigError(f"missing LLM credentials: {', '.join(missing)}")
        provider = AnthropicProvider(
            self.secrets.require("anthropic_api_key"),
            timeout_seconds=llm.timeout_seconds,
            max_retries=llm.max_retries,
        )
        return LLMGateway(provider, llm)
