from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_secrets
from trading_agent.bootstrap import Container
from trading_agent.data import AlpacaMarketData, AlpacaNews, MarketDataCollector
from trading_agent.execution import AlpacaBroker
from trading_agent.llm import LLMGateway
from trading_agent.settings import ConfigError, load_config


def _use_claude(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\n"
        "broker:\n  paper: true\n"
        "llm:\n  provider: anthropic\n  model: claude-sonnet-5-5\n  fallback_model: null\n",
        encoding="utf-8",
    )


def test_builds_paper_broker(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    broker = container.build_broker()
    assert isinstance(broker, AlpacaBroker)
    assert broker.is_paper


def test_builds_market_data_news_and_collector(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert isinstance(container.build_market_data(), AlpacaMarketData)
    assert isinstance(container.build_news(), AlpacaNews)
    assert isinstance(container.build_collector(), MarketDataCollector)


@pytest.mark.parametrize("builder", ["build_market_data", "build_news", "build_collector"])
def test_market_data_needs_alpaca_keys(config_dir: Path, builder: str) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_api_key=None)
    )
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        getattr(container, builder)()


def test_default_llm_is_openai(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    gateway = container.build_llm_gateway()
    assert isinstance(gateway, LLMGateway)
    assert gateway.provider_name == "openai"
    assert gateway.models == ("gpt-6.1-sol", "gpt-6-luna")


def test_llm_vendor_is_switched_by_config_only(config_dir: Path) -> None:
    _use_claude(config_dir)
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    gateway = container.build_llm_gateway()
    assert gateway.provider_name == "anthropic"
    assert gateway.models == ("claude-sonnet-5-5",)


def test_missing_broker_keys_raise(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_secret_key=None)
    )
    with pytest.raises(ConfigError, match="missing broker credentials: ALPACA_SECRET_KEY"):
        container.build_broker()


def test_missing_openai_key_raises(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(openai_api_key=None)
    )
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        container.build_llm_gateway()


def test_only_the_selected_vendor_key_is_required(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(anthropic_api_key=None)
    )
    assert container.build_llm_gateway().provider_name == "openai"
