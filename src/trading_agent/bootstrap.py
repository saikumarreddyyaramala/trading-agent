"""Composition root: the only place that wires concrete adapters to ports.

Everything else receives its dependencies as arguments, which keeps business logic
testable with fakes and lets us swap vendors through config.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent.agents.analyst import AnalystAgent
from trading_agent.data.alpaca_market_data import AlpacaMarketData
from trading_agent.data.alpaca_news import AlpacaNews
from trading_agent.data.collector import MarketDataCollector
from trading_agent.domain.ports import BrokerPort, LLMProviderPort, MarketDataPort, NewsPort
from trading_agent.execution.alpaca_broker import AlpacaBroker
from trading_agent.llm.anthropic_provider import AnthropicProvider
from trading_agent.llm.gateway import LLMGateway
from trading_agent.llm.openai_provider import OpenAIProvider
from trading_agent.mcp_server.server import build_server
from trading_agent.mcp_server.tools import ResearchTools
from trading_agent.settings.loader import ConfigError
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import (
    BROKER_SECRET_FIELDS,
    LLM_SECRET_FIELDS,
    MARKET_DATA_SECRET_FIELDS,
    Secrets,
)


@dataclass(frozen=True)
class Container:
    config: AppConfig
    secrets: Secrets

    def _require(self, what: str, fields: tuple[str, ...]) -> None:
        missing = self.secrets.missing(*fields)
        if missing:
            raise ConfigError(f"missing {what} credentials: {', '.join(missing)}")

    def build_broker(self) -> BrokerPort:
        self._require("broker", BROKER_SECRET_FIELDS[self.config.broker.provider])
        return AlpacaBroker(
            self.secrets.require("alpaca_api_key"),
            self.secrets.require("alpaca_secret_key"),
            paper=self.config.broker.paper,
        )

    def build_market_data(self) -> MarketDataPort:
        market_data = self.config.market_data
        self._require("market data", MARKET_DATA_SECRET_FIELDS[market_data.provider])
        return AlpacaMarketData(
            self.secrets.require("alpaca_api_key"),
            self.secrets.require("alpaca_secret_key"),
            paper=self.config.broker.paper,
            quote_feed=market_data.quote_feed,
            bars_feed=market_data.bars_feed,
        )

    def build_news(self) -> NewsPort:
        self._require("news", MARKET_DATA_SECRET_FIELDS[self.config.market_data.provider])
        return AlpacaNews(
            self.secrets.require("alpaca_api_key"), self.secrets.require("alpaca_secret_key")
        )

    def build_collector(
        self, market_data: MarketDataPort | None = None, news: NewsPort | None = None
    ) -> MarketDataCollector:
        return MarketDataCollector(
            market_data or self.build_market_data(),
            news or self.build_news(),
            self.config.market_data,
            max_data_age_seconds=self.config.risk.max_data_age_seconds,
        )

    def build_research_tools(self) -> ResearchTools:
        """Read-only tools for agents. The broker is used only for the market clock."""
        market_data = self.build_market_data()
        news = self.build_news()
        return ResearchTools(
            market_data,
            news,
            self.build_collector(market_data, news),
            self.build_broker().get_clock,
            max_data_age_seconds=self.config.risk.max_data_age_seconds,
            max_calls=self.config.tools.max_calls_per_session,
        )

    def build_analyst(self) -> AnalystAgent:
        """A fresh agent per run: its own tool budget, token usage, and audit trail."""
        return AnalystAgent(
            self.build_llm_gateway(), build_server(self.build_research_tools()), self.config
        )

    def build_llm_gateway(self) -> LLMGateway:
        llm = self.config.llm
        self._require("LLM", LLM_SECRET_FIELDS[llm.provider])
        provider: LLMProviderPort
        if llm.provider == "openai":
            provider = OpenAIProvider(
                self.secrets.require("openai_api_key"),
                timeout_seconds=llm.timeout_seconds,
                max_retries=llm.max_retries,
                reasoning_effort=llm.reasoning_effort,
            )
        else:
            provider = AnthropicProvider(
                self.secrets.require("anthropic_api_key"),
                timeout_seconds=llm.timeout_seconds,
                max_retries=llm.max_retries,
            )
        return LLMGateway(provider, llm)
