"""Typed configuration (YAML per environment) and secrets (environment variables)."""

from trading_agent.settings.loader import (
    ConfigError,
    assert_live_trading_allowed,
    load_config,
)
from trading_agent.settings.models import (
    AnalystConfig,
    AppConfig,
    BrokerConfig,
    CriticConfig,
    Environment,
    ExecutionConfig,
    LLMConfig,
    MarketDataConfig,
    OrchestrationConfig,
    ReportingConfig,
    RiskConfig,
    ScheduleConfig,
    StorageConfig,
    ToolsConfig,
)
from trading_agent.settings.secrets import Secrets

__all__ = [
    "AnalystConfig",
    "AppConfig",
    "BrokerConfig",
    "ConfigError",
    "CriticConfig",
    "Environment",
    "ExecutionConfig",
    "LLMConfig",
    "MarketDataConfig",
    "OrchestrationConfig",
    "ReportingConfig",
    "RiskConfig",
    "ScheduleConfig",
    "Secrets",
    "StorageConfig",
    "ToolsConfig",
    "assert_live_trading_allowed",
    "load_config",
]
