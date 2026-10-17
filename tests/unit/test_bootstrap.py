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
from trading_agent.execution.executor import Executor
from trading_agent.health import CheckResult, CheckStatus
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


def _gates(monkeypatch: pytest.MonkeyPatch, *, passing: bool) -> None:
    status = CheckStatus.PASS if passing else CheckStatus.FAIL
    results = [CheckResult("Paper evaluation", status, "GO" if passing else "says NO-GO")]
    monkeypatch.setattr(Container, "live_checks", lambda self: results)


def test_live_executor_is_blocked_when_a_gate_fails(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(monkeypatch, passing=False)
    container = Container(
        config=load_config("live", config_dir), secrets=make_secrets(live_trading_confirmed=True)
    )
    executor = container.build_executor()
    assert executor._entries_blocked == "live gates failed: Paper evaluation: says NO-GO"


def test_live_executor_is_authorized_only_when_every_gate_passes(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(monkeypatch, passing=True)
    container = Container(
        config=load_config("live", config_dir), secrets=make_secrets(live_trading_confirmed=True)
    )
    assert container.build_executor()._entries_blocked is None


def test_paper_executor_skips_live_gates(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(self: Container) -> None:
        raise AssertionError("live gates must not run for paper")

    monkeypatch.setattr(Container, "live_checks", explode)
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.build_executor()._entries_blocked is None


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


def test_live_cycle_services_block_entries_until_the_gates_pass(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(monkeypatch, passing=False)
    secrets = make_secrets(live_trading_confirmed=True, database_url=SecretStr("sqlite://"))
    container = Container(config=load_config("live", config_dir), secrets=secrets)
    services = container.build_cycle_services()
    assert services.executor._entries_blocked is not None  # type: ignore[attr-defined]


def test_live_checks_run_without_network_when_keys_are_missing(config_dir: Path) -> None:
    container = Container(
        config=load_config("live", config_dir), secrets=make_secrets(alpaca_api_key=None)
    )
    results = {r.name: r for r in container.live_checks()}
    assert results["Cash account"].status is CheckStatus.FAIL
    assert "missing broker credentials" in results["Cash account"].detail


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


def test_critic_uses_its_own_model_when_configured(config_dir: Path) -> None:
    from trading_agent.agents.critic import CriticAgent

    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\ncritic:\n  model: gpt-6-luna\n",
        encoding="utf-8",
    )
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    critic = container.build_critic()
    assert isinstance(critic, CriticAgent)
    assert critic._gateway.models == ("gpt-6-luna",)


def test_critic_defaults_to_the_analysts_model(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    critic = container.build_critic()
    assert critic is not None
    assert critic._gateway.models == ("gpt-6.1-sol", "gpt-6-luna")


def test_disabled_critic_is_none(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\ncritic:\n  enabled: false\n",
        encoding="utf-8",
    )
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.build_critic() is None


# ---------- Self-learning ----------


def _learning_db(tmp_path: Path) -> SecretStr:
    from trading_agent.storage.database import make_engine, upgrade

    url = f"sqlite:///{tmp_path / 'audit.db'}"
    upgrade(make_engine(url))
    return SecretStr(url)


def _disable_learning(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nlearning:\n  enabled: false\n",
        encoding="utf-8",
    )


def test_analyst_gets_the_active_lessons(config_dir: Path, tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from trading_agent.storage.store import NewLessonRow

    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    container.build_audit_store().record_lessons(
        None,
        [NewLessonRow(text="Check volume first.", category="research", evidence={})],
        expires_at=datetime.now(UTC) + timedelta(days=5),
        max_active=15,
    )
    [lesson] = container.active_lessons()
    assert lesson.text == "Check volume first."
    assert container.build_analyst()._lessons == (lesson,)


def test_no_lessons_without_a_database_or_when_disabled(config_dir: Path, tmp_path: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.active_lessons() == []
    unmigrated = make_secrets(database_url=SecretStr(f"sqlite:///{tmp_path / 'empty.db'}"))
    assert Container(container.config, unmigrated).active_lessons() == []  # read error: none
    _disable_learning(config_dir)
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    assert Container(load_config("paper", config_dir), secrets).active_lessons() == []


def test_builds_reflector_and_learning_service(config_dir: Path) -> None:
    from trading_agent.agents.reflection import Reflector
    from trading_agent.learning import LearningService

    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    reflector = container.build_reflector()
    assert isinstance(reflector, Reflector)
    assert reflector._max_new == 2
    assert isinstance(container.build_learning_service(), LearningService)


def test_learn_needs_a_database(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.learn(date(2026, 10, 15)).startswith("skipped: no database")


def test_learn_reviews_the_day(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_agent.learning import LearningService

    seen: list[date] = []
    monkeypatch.setattr(LearningService, "learn", lambda self, day: seen.append(day) or "1 new")
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    assert container.learn(date(2026, 10, 15)) == "1 new"
    assert seen == [date(2026, 10, 15)]


def test_cycle_services_reflect_only_when_learning_is_enabled(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.build_cycle_services().reflect == container.learn
    _disable_learning(config_dir)
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.build_cycle_services().reflect is None


# ---------- Cost ledger (ADR 0014) ----------


def test_every_gateway_records_to_the_ledger_and_obeys_the_budget(
    config_dir: Path, tmp_path: Path
) -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    from trading_agent.costs import CostLedger

    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\n"
        "costs:\n  llm_prices:\n    gpt-6.1-sol: {input: 1, output: 1}\n"
        "  daily_llm_budget_usd: 0.50\n",
        encoding="utf-8",
    )
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    gateway = container.build_llm_gateway()
    assert isinstance(gateway._listener.__self__, CostLedger)  # type: ignore[union-attr]
    assert gateway._budget is not None
    assert gateway._budget() is None

    store = container.build_audit_store()
    from trading_agent.domain.models import LLMCallRecord, LLMUsage

    store.record_llm_call(
        LLMCallRecord(
            created_at=datetime.now(UTC),
            provider="openai",
            model="gpt-6.1-sol",
            agent="analyst",
            run_id=None,
            status="ok",
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            latency_ms=1.0,
        ),
        Decimal("0.50"),
    )
    reason = container.build_llm_gateway()._budget()  # type: ignore[misc]
    assert reason is not None
    assert "budget of $0.50 reached" in reason


def test_cost_report_from_the_container(config_dir: Path, tmp_path: Path) -> None:
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    report = container.cost_report(days=3)
    assert len(report.days) == 3
    assert report.total.calls == 0
