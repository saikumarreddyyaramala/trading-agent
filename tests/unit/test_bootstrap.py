from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_secrets
from trading_agent.bootstrap import Container
from trading_agent.execution import AlpacaBroker
from trading_agent.llm import LLMGateway
from trading_agent.settings import ConfigError, load_config


def test_builds_paper_broker_and_gateway(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    broker = container.build_broker()
    assert isinstance(broker, AlpacaBroker)
    assert broker.is_paper
    gateway = container.build_llm_gateway()
    assert isinstance(gateway, LLMGateway)
    assert gateway.models == ("claude-sonnet-5-5", "claude-haiku-5-5")


def test_missing_broker_keys_raise(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_secret_key=None)
    )
    with pytest.raises(ConfigError, match="ALPACA_SECRET_KEY"):
        container.build_broker()


def test_missing_llm_key_raises(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(anthropic_api_key=None)
    )
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        container.build_llm_gateway()
