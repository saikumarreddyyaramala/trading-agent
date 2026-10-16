from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from tests.conftest import FakeBroker, FakeMarketData, make_secrets
from trading_agent.agents import AnalystAgent
from trading_agent.bootstrap import Container
from trading_agent.data import AlpacaMarketData, AlpacaNews, MarketDataCollector
from trading_agent.execution import AlpacaBroker
from trading_agent.execution.executor import ExecutionDisabledError, Executor
from trading_agent.llm import LLMGateway
from trading_agent.mcp_server import ResearchTools
from trading_agent.risk import RiskEngine
from trading_agent.settings import ConfigError, load_config
from trading_agent.storage.store import NullAuditStore, SqlAuditStore


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


def test_builds_research_tools_with_configured_budget(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    tools = container.build_research_tools()
    assert isinstance(tools, ResearchTools)
    assert tools.calls_remaining == 100


@pytest.mark.parametrize(
    "builder", ["build_market_data", "build_news", "build_collector", "build_research_tools"]
)
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


def test_builds_analyst(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert isinstance(container.build_analyst(), AnalystAgent)


@pytest.mark.parametrize("missing", ["alpaca_api_key", "openai_api_key"])
def test_analyst_needs_data_and_llm_keys(config_dir: Path, missing: str) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(**{missing: None})
    )
    with pytest.raises(ConfigError, match=missing.upper()):
        container.build_analyst()


def test_builds_risk_engine(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert isinstance(container.build_risk_engine(), RiskEngine)


def test_risk_engine_needs_broker_keys(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_secret_key=None)
    )
    with pytest.raises(ConfigError, match="ALPACA_SECRET_KEY"):
        container.build_risk_engine()


def test_builds_paper_executor(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert isinstance(container.build_executor(), Executor)


def test_live_executor_is_refused_even_when_confirmed(config_dir: Path) -> None:
    container = Container(
        config=load_config("live", config_dir), secrets=make_secrets(live_trading_confirmed=True)
    )
    with pytest.raises(ExecutionDisabledError, match="paper-only"):
        container.build_executor()


def test_executor_needs_broker_keys(config_dir: Path) -> None:
    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_api_key=None)
    )
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        container.build_executor()


def test_audit_store_without_url_is_null(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.database_url() is None
    assert isinstance(container.build_audit_store(), NullAuditStore)
    assert container.build_recorder().required is False


def test_audit_store_with_url_is_sql(config_dir: Path) -> None:
    secrets = make_secrets(database_url=SecretStr("sqlite://"))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    assert isinstance(container.build_audit_store(), SqlAuditStore)


def test_required_audit_store_without_url_raises(config_dir: Path) -> None:
    container = Container(
        config=load_config("live", config_dir), secrets=make_secrets(live_trading_confirmed=True)
    )
    with pytest.raises(ConfigError, match="DATABASE_URL is not set"):
        container.build_audit_store()


def test_builds_cycle_services(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    services = container.build_cycle_services()
    assert services.recorder.required is False
    assert callable(services.analyze)
    assert callable(services.health_check)


def test_cycle_services_refuse_live(config_dir: Path) -> None:
    secrets = make_secrets(live_trading_confirmed=True, database_url=SecretStr("sqlite://"))
    container = Container(config=load_config("live", config_dir), secrets=secrets)
    with pytest.raises(ExecutionDisabledError):
        container.build_cycle_services()


def test_publish_report_saves_and_sends(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_agent import bootstrap as module
    from trading_agent.reporting.report import ReportBuilder

    secrets = make_secrets(report_webhook_url=SecretStr("https://hooks.example/x"))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    builder = ReportBuilder(NullAuditStore(), FakeMarketData(), FakeBroker(), container.config)
    monkeypatch.setattr(Container, "build_report_builder", lambda self, store=None: builder)
    sent: list[str] = []
    monkeypatch.setattr(module, "post_webhook", lambda url, text: sent.append(url))

    detail = container.publish_report(date(2026, 10, 15))
    assert detail == f"saved {Path('reports') / '2026-10-15.md'}; sent to the webhook"
    assert sent == ["https://hooks.example/x"]
    assert (tmp_path / "reports" / "2026-10-15.json").exists()

    assert container.publish_report(date(2026, 10, 15), send=False).endswith("2026-10-15.md")
    assert len(sent) == 1


def test_report_webhook_url(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.report_webhook_url() is None
