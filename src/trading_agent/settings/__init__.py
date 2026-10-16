"""Typed configuration (YAML per environment) and secrets (environment variables)."""

from trading_agent.settings.loader import (
    ConfigError,
    assert_live_trading_allowed,
    load_config,
)
from trading_agent.settings.models import (
    AppConfig,
    BrokerConfig,
    Environment,
    LLMConfig,
    RiskConfig,
    ScheduleConfig,
)
from trading_agent.settings.secrets import Secrets

__all__ = [
    "AppConfig",
    "BrokerConfig",
    "ConfigError",
    "Environment",
    "LLMConfig",
    "RiskConfig",
    "ScheduleConfig",
    "Secrets",
    "assert_live_trading_allowed",
    "load_config",
]
