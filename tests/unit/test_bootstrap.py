from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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


# ---------- Trade slots and score history (ADR 0015) ----------


def _paper3(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\n"
        "risk:\n  max_trades_per_day: 3\n  max_open_positions: 3\n",
        encoding="utf-8",
    )


def test_slot_capacity(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import make_position

    _paper3(config_dir)
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    broker = FakeBroker()
    monkeypatch.setattr(Container, "build_broker", lambda self: broker)
    assert container.slot_capacity() is None

    broker.orders_today = 3
    assert container.slot_capacity() == "3 buy order(s) today (max 3)"

    broker.orders_today = 1
    broker.positions = tuple(make_position(s) for s in ("A", "B", "C"))
    assert container.slot_capacity() == "3 position(s) open (max 3)"


def test_slot_capacity_respects_the_kill_switch(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nkill_switch: true\n", encoding="utf-8"
    )
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.slot_capacity() == "kill switch is on"


def test_trade_history_comes_from_closed_system_trades(config_dir: Path, tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from trading_agent.domain.models import BrokerOrder

    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    assert container.trade_history() == []

    store = container.build_audit_store()
    run = "daily-20261015"
    store.start_run(run, "daily")
    # Midday yesterday: the buy and the sell must fall on the same market day.
    when = datetime.now(UTC).replace(hour=15, minute=0, second=0, microsecond=0) - timedelta(days=1)

    def order(cid: str, side: str, price: str, minutes: int) -> BrokerOrder:
        return BrokerOrder(
            id=f"id-{cid}",
            client_order_id=cid,
            symbol="AAPL",
            side=side,  # type: ignore[arg-type]
            order_type="limit" if side == "buy" else "market",
            status="filled",
            quantity=Decimal("1"),
            filled_quantity=Decimal("1"),
            filled_avg_price=Decimal(price),
            submitted_at=when + timedelta(minutes=minutes),
        )

    store.record_orders(
        run, [order(f"ta-{run}-entry", "buy", "100", 0), order("x", "sell", "102", 30)]
    )
    assert container.trade_history() == [Decimal("2.00")]


def test_risk_engine_gets_the_history(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    engine = container.build_risk_engine()
    assert engine._history is not None
    assert engine._history() == []  # no database: no history, which scores as neutral


def test_cycle_services_pass_context_and_capacity(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str | None, str | None]] = []

    async def fake_run(
        self: AnalystAgent, run_id: str | None = None, context: str | None = None
    ) -> str:
        seen.append((run_id, context))
        return "result"

    monkeypatch.setattr(AnalystAgent, "run", fake_run)
    monkeypatch.setattr(Container, "build_analyst", lambda self: AnalystAgent.__new__(AnalystAgent))
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    services = container.build_cycle_services()
    assert services.analyze("daily-1-2", "slot 2") == "result"
    assert seen == [("daily-1-2", "slot 2")]
    assert services.capacity == container.slot_capacity


# ---------- Market watch and plan (ADR 0016) ----------


def _watch_container(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, broker: Any = None
) -> Container:
    from tests.conftest import FakeNews
    from tests.unit.test_market_watch import _pre_market

    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    market = _pre_market()
    monkeypatch.setattr(Container, "build_market_data", lambda self: market)
    monkeypatch.setattr(Container, "build_news", lambda self: FakeNews())
    monkeypatch.setattr(Container, "build_broker", lambda self: broker or FakeBroker())
    return container


def test_premarket_scan_focuses_on_the_plan_and_positions(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from tests.conftest import make_position
    from trading_agent.domain.models import DayPlan, PlanItem

    container = _watch_container(
        config_dir, tmp_path, monkeypatch, broker=FakeBroker(positions=(make_position("MSFT"),))
    )
    store = container.build_audit_store()
    store.record_plan(
        None,
        DayPlan(
            source_day=date(2026, 10, 14),
            for_day=date(2026, 10, 15),
            created_at=datetime.now(UTC),
            items=(PlanItem(kind="watch", symbol="AAPL", text="AAPL moved", priority=1),),
            summary="1 item",
        ),
    )
    detail = container.premarket(date(2026, 10, 15))
    assert detail.startswith("scanned ")
    report = store.get_watch(date(2026, 10, 15), "pre")
    assert report is not None
    reasons = {i.symbol: i.reasons for i in report.items}
    assert reasons["AAPL"][0] == "plan"
    assert reasons["MSFT"][0] == "held"


def test_postmarket_scan_focuses_on_the_days_ideas(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_report import _record_trading_day

    container = _watch_container(config_dir, tmp_path, monkeypatch)
    store = container.build_audit_store()
    _record_trading_day(store)  # AAPL proposed and traded on 2026-10-15
    container.postmarket(date(2026, 10, 15))
    report = store.get_watch(date(2026, 10, 15), "post")
    assert report is not None
    reasons = {i.symbol: i.reasons for i in report.items}
    assert reasons["AAPL"][:2] == ("traded", "proposed")


def test_make_plan_and_brief(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_report import _record_trading_day

    container = _watch_container(config_dir, tmp_path, monkeypatch)
    _record_trading_day(container.build_audit_store())
    detail = container.make_plan(date(2026, 10, 15))
    assert detail.startswith("1 item(s) for ")
    assert container.brief(date(2026, 10, 15)) is None  # the plan is for the next session


def test_watch_and_plan_need_a_database(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.make_plan(date(2026, 10, 15)).startswith("skipped: no database")
    assert container.brief(date(2026, 10, 15)) is None


def test_cycle_services_include_watch_plan_and_brief(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    services = container.build_cycle_services()
    assert services.premarket == container.premarket
    assert services.postmarket == container.postmarket
    assert services.plan == container.make_plan
    assert services.brief == container.brief


def test_scans_survive_a_broker_outage(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _watch_container(
        config_dir, tmp_path, monkeypatch, broker=FakeBroker(fail_positions=True)
    )
    assert container.premarket(date(2026, 10, 15)).startswith("scanned ")
    assert container.postmarket(date(2026, 10, 15)).startswith("scanned ")


def test_plan_uses_the_brokers_calendar(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_report import _record_trading_day

    container = _watch_container(config_dir, tmp_path, monkeypatch)
    store = container.build_audit_store()
    _record_trading_day(store)
    container.make_plan(date(2026, 10, 15))
    plan = store.plan_from(date(2026, 10, 15))
    assert plan is not None
    assert plan["for_day"] > date(2026, 10, 15)


# ---------- Guidebook (ADR 0017) ----------


def test_curate_builds_and_saves_the_guidebook(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    monkeypatch.setattr(Container, "build_playbook_writer", lambda self: None)
    detail = container.curate(date(2026, 10, 15))
    assert "guidebook v1" in detail
    saved = tmp_path / "reports" / "guidebook.md"
    assert saved.read_text(encoding="utf-8").startswith("# Guidebook")
    assert "guidebook v" not in container.curate(date(2026, 10, 15))  # nothing changed


def test_curate_needs_a_database_and_can_be_disabled(config_dir: Path, tmp_path: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.curate(date(2026, 10, 15)).startswith("skipped: no database")
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nguidebook:\n  enabled: false\n",
        encoding="utf-8",
    )
    secrets = make_secrets(database_url=_learning_db(tmp_path))
    off = Container(config=load_config("paper", config_dir), secrets=secrets)
    assert off.curate(date(2026, 10, 15)) == "disabled (guidebook.enabled: false)"
    assert off.build_cycle_services().curate is None


def test_the_analyst_is_offered_playbooks(config_dir: Path, tmp_path: Path) -> None:
    from tests.unit.test_guidebook_store import playbook

    secrets = make_secrets(database_url=_learning_db(tmp_path))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    store = container.build_audit_store()
    store.record_playbook(playbook("trial", status="candidate"))
    store.record_playbook(playbook("proven", status="active"))
    assert [p.name for p in container.offered_playbooks()] == ["trial", "proven"]
    assert [p.name for p in container.build_analyst()._playbooks] == ["trial", "proven"]


def test_live_is_offered_only_active_playbooks(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_guidebook_store import playbook

    secrets = make_secrets(database_url=_learning_db(tmp_path), live_trading_confirmed=True)
    container = Container(config=load_config("live", config_dir), secrets=secrets)
    store = container.build_audit_store()
    store.record_playbook(playbook("trial", status="candidate"))
    store.record_playbook(playbook("proven", status="active"))
    assert [p.name for p in container.offered_playbooks()] == ["proven"]


def test_no_playbooks_without_a_database(config_dir: Path, tmp_path: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.offered_playbooks() == []
    unmigrated = make_secrets(database_url=SecretStr(f"sqlite:///{tmp_path / 'empty.db'}"))
    assert Container(container.config, unmigrated).offered_playbooks() == []


def test_builds_the_playbook_writer(config_dir: Path) -> None:
    from trading_agent.agents.playbooks import PlaybookWriter

    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert isinstance(container.build_playbook_writer(), PlaybookWriter)


# ---------- Dashboard (ADR 0018) ----------


def test_dashboard_data_without_keys_still_builds(config_dir: Path) -> None:
    from trading_agent.dashboard.data import DashboardData

    container = Container(
        config=load_config("paper", config_dir), secrets=make_secrets(alpaca_api_key=None)
    )
    data = container.build_dashboard_data()
    assert isinstance(data, DashboardData)
    assert data._broker is None
    assert data._market_data is None
    assert "broker unavailable: no broker credentials" in data.overview()["errors"]


def test_dashboard_data_with_keys(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    data = container.build_dashboard_data()
    assert data._broker is not None
    assert data._news is not None


# ---------- Phone alerts (ADR 0021) ----------


def test_alerts_are_off_without_telegram_secrets(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert not container.alerter.configured


def test_alerts_need_both_the_token_and_the_chat(config_dir: Path) -> None:
    token_only = Container(
        config=load_config("paper", config_dir),
        secrets=make_secrets(telegram_bot_token=SecretStr("1:abc")),
    )
    assert not token_only.alerter.configured
    both = Container(
        config=load_config("paper", config_dir),
        secrets=make_secrets(
            telegram_bot_token=SecretStr("1:abc"), telegram_chat_id=SecretStr("42")
        ),
    )
    assert both.alerter.configured
    assert both.alerter is both.alerter  # one alerter: one dedupe and rate limit


def test_the_cycle_alerts_through_the_containers_alerter(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    services = container.build_cycle_services()
    assert services.alert == container.alerter.notify


# ---------- Chat (ADR 0022) ----------


def _telegram(config_dir: Path, **config: Any) -> Container:
    loaded = load_config("paper", config_dir)
    if config:
        loaded = loaded.model_copy(update={"chat": loaded.chat.model_copy(update=config)})
    return Container(
        config=loaded,
        secrets=make_secrets(
            telegram_bot_token=SecretStr("1:abc"), telegram_chat_id=SecretStr("42")
        ),
    )


def test_no_chat_without_telegram(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    assert container.build_chat_listener() is None


def test_chat_can_be_switched_off(config_dir: Path) -> None:
    assert _telegram(config_dir, enabled=False).build_chat_listener() is None


def test_chat_listener_answers_only_the_configured_chat(config_dir: Path) -> None:
    listener = _telegram(config_dir).build_chat_listener()
    assert listener is not None
    assert listener._owner == "42"


def test_the_cycle_checks_the_chat_pause(config_dir: Path, tmp_path: Path) -> None:
    container = _telegram(config_dir, pause_file=str(tmp_path / "pause.json"))
    services = container.build_cycle_services()
    assert services.paused is not None
    assert services.paused() is None
    today = datetime.now(ZoneInfo(container.config.schedule.timezone)).date()
    container.pause_store().pause(today, by="telegram")
    assert services.paused() == f"paused from telegram for {today.isoformat()}"


def test_chat_data_finds_the_next_plan(config_dir: Path, tmp_path: Path) -> None:
    from trading_agent.domain.models import DayPlan, PlanItem
    from trading_agent.storage.database import make_engine, upgrade

    url = f"sqlite:///{tmp_path / 'audit.db'}"
    upgrade(make_engine(url))
    secrets = make_secrets(database_url=SecretStr(url))
    container = Container(config=load_config("paper", config_dir), secrets=secrets)
    data = container.build_chat_data()
    friday, monday = date(2026, 10, 16), date(2026, 10, 19)
    assert data.plan(date(2026, 10, 17)) is None
    data._store.record_plan(  # type: ignore[attr-defined]
        None,
        DayPlan(
            source_day=friday,
            for_day=monday,
            created_at=datetime(2026, 10, 16, 21, tzinfo=UTC),
            items=(PlanItem(kind="ops", text="Fix the thing.", priority=1),),
            summary="1 item",
        ),
    )
    plan = data.plan(date(2026, 10, 17))  # Saturday: the plan Friday made for Monday
    assert plan is not None
    assert plan["for_day"] == "2026-10-19"
    assert plan["items"][0]["text"] == "Fix the thing."
    assert data.plan(monday)["for_day"] == "2026-10-19"


# ---------- Market journal (ADR 0023) ----------


def test_the_journal_is_built_with_every_free_source(config_dir: Path) -> None:
    container = Container(config=load_config("paper", config_dir), secrets=make_secrets())
    recorder = container.build_journal_recorder()
    assert recorder is not None
    names = {s.name for s in container.build_journal_sources()}
    assert names == {"company_news", "federal_reserve", "bls", "federal_register"}


def test_journal_sources_follow_the_config(config_dir: Path) -> None:
    loaded = load_config("paper", config_dir)
    journal = loaded.journal.model_copy(
        update={"sources": loaded.journal.sources.model_copy(update={"bls": False})}
    )
    container = Container(
        config=loaded.model_copy(update={"journal": journal}), secrets=make_secrets()
    )
    assert "bls" not in {s.name for s in container.build_journal_sources()}


def test_no_journal_when_switched_off_or_without_market_data(config_dir: Path) -> None:
    loaded = load_config("paper", config_dir)
    off = loaded.model_copy(
        update={"journal": loaded.journal.model_copy(update={"enabled": False})}
    )
    assert Container(config=off, secrets=make_secrets()).build_journal_recorder() is None
    no_keys = make_secrets(alpaca_api_key=None, alpaca_secret_key=None)
    assert Container(config=loaded, secrets=no_keys).build_journal_recorder() is None


def test_the_journal_records_without_ai_when_no_llm_key(config_dir: Path) -> None:
    secrets = make_secrets(openai_api_key=None)
    recorder = Container(
        config=load_config("paper", config_dir), secrets=secrets
    ).build_journal_recorder()
    assert recorder is not None
    assert recorder._writer is None
