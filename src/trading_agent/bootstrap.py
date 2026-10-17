"""Composition root: the only place that wires concrete adapters to ports.

Everything else receives its dependencies as arguments, which keeps business logic
testable with fakes and lets us swap vendors through config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import anyio

from trading_agent.agents.analyst import AnalystAgent, ToolCallRecord
from trading_agent.agents.critic import CriticAgent, CriticResult
from trading_agent.agents.reflection import Reflector
from trading_agent.costs import CostLedger, CostReport, cost_report
from trading_agent.data.alpaca_market_data import AlpacaMarketData
from trading_agent.data.alpaca_news import AlpacaNews
from trading_agent.data.collector import MarketDataCollector
from trading_agent.domain.models import Lesson, TradeProposal
from trading_agent.domain.ports import (
    BrokerPort,
    LLMProviderPort,
    MarketDataPort,
    NewsPort,
    OrderExecutionPort,
)
from trading_agent.execution.alpaca_broker import AlpacaBroker
from trading_agent.execution.alpaca_orders import AlpacaOrderExecutor
from trading_agent.execution.executor import Executor
from trading_agent.execution.live import blocking_reason, live_readiness
from trading_agent.health import CheckResult, run_healthchecks
from trading_agent.learning.service import LearningService
from trading_agent.llm.anthropic_provider import AnthropicProvider
from trading_agent.llm.gateway import LLMGateway
from trading_agent.llm.openai_provider import OpenAIProvider
from trading_agent.mcp_server.server import build_server
from trading_agent.mcp_server.tools import ResearchTools
from trading_agent.orchestration.cycle import CycleServices, DailyCycle
from trading_agent.planning import Planner, morning_brief
from trading_agent.reporting.delivery import post_webhook, save_report
from trading_agent.reporting.performance import trades_from_orders
from trading_agent.reporting.report import ReportBuilder, render_markdown
from trading_agent.reporting.slots import slot_runs
from trading_agent.risk.engine import RiskEngine
from trading_agent.settings.loader import ConfigError
from trading_agent.settings.models import AppConfig, LLMConfig
from trading_agent.settings.secrets import (
    BROKER_SECRET_FIELDS,
    LLM_SECRET_FIELDS,
    MARKET_DATA_SECRET_FIELDS,
    Secrets,
)
from trading_agent.storage.database import make_engine
from trading_agent.storage.store import (
    AuditError,
    AuditRecorder,
    AuditStore,
    NullAuditStore,
    SqlAuditStore,
)
from trading_agent.watch import MarketWatch, Phase


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
            risk=self.config.risk,
        )

    def build_analyst(self) -> AnalystAgent:
        """A fresh agent per run: its own tool budget, token usage, and audit trail."""
        return AnalystAgent(
            self.build_llm_gateway(),
            build_server(self.build_research_tools()),
            self.config,
            lessons=self.active_lessons(),
        )

    def active_lessons(self) -> list[Lesson]:
        """What the analyst has learned (ADR 0013). Empty when learning is off or the store
        cannot be read: lessons are advisory, so their absence never blocks a run."""
        learning = self.config.learning
        if not learning.enabled:
            return []
        try:
            return self.build_audit_store().active_lessons(datetime.now(UTC), learning.max_active)
        except AuditError:
            return []

    def build_reflector(self) -> Reflector:
        """The daily reviewer. Its own gateway, so its usage is counted separately."""
        learning = self.config.learning
        return Reflector(
            self.build_llm_gateway(),
            max_new=learning.max_new_per_day,
            repair_attempts=learning.repair_attempts,
        )

    def build_learning_service(self, store: AuditStore | None = None) -> LearningService:
        return LearningService(
            store or self.build_audit_store(),
            self.build_market_data(),
            self.build_reflector(),
            self.config,
        )

    def learn(self, day: date) -> str:
        """Review `day` and update the lessons. Needs a database: lessons must persist."""
        store = self.build_audit_store()
        if isinstance(store, NullAuditStore):
            return "skipped: no database, so there is nowhere to keep lessons"
        return self.build_learning_service(store).learn(day)

    def build_risk_engine(self) -> RiskEngine:
        return RiskEngine(
            self.build_broker(),
            self.build_market_data(),
            self.config,
            live_trading_confirmed=self.secrets.live_trading_confirmed,
            history=self.trade_history,
        )

    def trade_history(self) -> list[Decimal]:
        """Returns (%) of the daily system's closed trades over the last year, oldest first,
        for the trade score's history part (ADR 0015). Empty without a database."""
        zone = ZoneInfo(self.config.schedule.timezone)
        end = datetime.now(zone) + timedelta(days=1)
        orders = self.build_audit_store().orders_between(end - timedelta(days=366), end)
        return [
            t.return_pct
            for t in trades_from_orders(orders, zone)
            if t.closed and t.return_pct is not None
        ]

    def slot_capacity(self) -> str | None:
        """Why an extra trade slot cannot trade now, checked before spending LLM tokens."""
        if self.config.kill_switch:
            return "kill switch is on"
        risk = self.config.risk
        broker = self.build_broker()
        positions = broker.get_positions()
        if len(positions) >= risk.max_open_positions:
            return f"{len(positions)} position(s) open (max {risk.max_open_positions})"
        zone = ZoneInfo(self.config.schedule.timezone)
        day_start = datetime.combine(datetime.now(zone).date(), time(0), tzinfo=zone)
        bought = broker.count_buy_orders_since(day_start)
        if bought >= risk.max_trades_per_day:
            return f"{bought} buy order(s) today (max {risk.max_trades_per_day})"
        return None

    def build_order_port(self) -> OrderExecutionPort:
        self._require("broker", BROKER_SECRET_FIELDS[self.config.broker.provider])
        return AlpacaOrderExecutor(
            self.secrets.require("alpaca_api_key"),
            self.secrets.require("alpaca_secret_key"),
            paper=self.config.broker.paper,
        )

    def live_checks(self) -> list[CheckResult]:
        """Every real-money gate (ADR 0012). Read-only; never raises."""
        return live_readiness(
            self.config,
            self.secrets,
            account=lambda: self.build_broker().get_account(),
            audit_environments=lambda: self.build_audit_store().environments(),
            evaluation_dir=Path(self.config.reporting.output_dir),
        )

    def build_executor(self) -> Executor:
        """Order execution. Live entries are authorized only if every live gate passes now;
        otherwise the executor can still manage and exit positions, but not enter."""
        orders = self.build_order_port()
        blocked = blocking_reason(self.live_checks()) if self.config.is_live else None
        return Executor(
            orders,
            self.build_broker(),
            self.build_market_data(),
            self.config,
            live_authorized=self.config.is_live and blocked is None,
            entries_blocked=blocked,
        )

    def database_url(self) -> str | None:
        value = self.secrets.database_url
        return value.get_secret_value() if value and value.get_secret_value() else None

    def build_audit_store(self) -> AuditStore:
        url = self.database_url()
        if url is None:
            if self.config.storage.required:
                raise ConfigError("storage.required is true but DATABASE_URL is not set")
            return NullAuditStore()
        return SqlAuditStore(make_engine(url), self.config)

    def build_recorder(self) -> AuditRecorder:
        return AuditRecorder(self.build_audit_store(), required=self.config.storage.required)

    def build_cycle_services(self) -> CycleServices:
        """Everything the daily cycle needs. Agents, tools, and gateways are built fresh per
        call, so each day gets its own tool budget and token accounting."""
        url = self.database_url()
        executor = self.build_executor()  # refuses a live configuration
        broker = self.build_broker()

        def health_check() -> list[Any]:
            return run_healthchecks(
                self.config,
                self.secrets,
                broker_factory=self.build_broker,
                market_data_factory=self.build_market_data,
                news_factory=self.build_news,
                llm_factory=self.build_llm_gateway,
                engine_factory=None if url is None else (lambda: make_engine(url)),
                live_checks=self.live_checks if self.config.is_live else None,
            )

        def analyze(run_id: str, context: str | None = None) -> Any:
            return anyio.run(self.build_analyst().run, run_id, context)

        def critique(
            run_id: str, proposal: TradeProposal, calls: list[ToolCallRecord]
        ) -> CriticResult | None:
            critic = self.build_critic()
            return None if critic is None else critic.review(run_id, proposal, calls)

        return CycleServices(
            health_check=health_check,
            analyze=analyze,
            evaluate=lambda proposal: self.build_risk_engine().evaluate(proposal),
            executor=executor,
            market_clock=broker.get_clock,
            orders_since=self.build_order_port().get_orders_since,
            recorder=self.build_recorder(),
            report=self.publish_report,
            critique=critique,
            reflect=self.learn if self.config.learning.enabled else None,
            capacity=self.slot_capacity,
            premarket=self.premarket,
            postmarket=self.postmarket,
            plan=self.make_plan,
            brief=self.brief,
        )

    # ---------- Market watch and the next-day plan (ADR 0016) ----------

    def build_market_watch(self) -> MarketWatch:
        return MarketWatch(self.build_market_data(), self.build_news(), self.config)

    def _held(self) -> list[str]:
        try:
            return [p.symbol for p in self.build_broker().get_positions()]
        except Exception:  # the watch still runs without positions
            return []

    def _scan(self, phase: Phase, day: date, focus: dict[str, list[str]]) -> str:
        store = self.build_audit_store()
        report = self.build_market_watch().scan(phase, day, focus)
        store.record_watch(DailyCycle.run_id(day), report)
        detail = f"scanned {len(report.items)} symbols, {len(report.alerts)} alert(s)"
        return detail + (f"; {'; '.join(report.errors)}" if report.errors else "")

    def premarket(self, day: date) -> str:
        """Scan before the open: yesterday's plan symbols, open positions, top movers."""
        focus: dict[str, list[str]] = {}
        plan = self.build_audit_store().plan_for(day)
        for item in (plan or {}).get("items", []):
            if item.get("symbol"):
                focus.setdefault(item["symbol"], []).append("plan")
        for symbol in self._held():
            focus.setdefault(symbol, []).append("held")
        return self._scan("pre", day, focus)

    def postmarket(self, day: date) -> str:
        """Scan after the close: today's trades and ideas, open positions, top movers."""
        store = self.build_audit_store()
        zone = ZoneInfo(self.config.schedule.timezone)
        start = datetime.combine(day, time(0), tzinfo=zone)
        focus: dict[str, list[str]] = {}
        for trade in trades_from_orders(
            store.orders_between(start, start + timedelta(days=1)), zone
        ):
            if trade.day == day:
                focus.setdefault(trade.symbol, []).append("traded")
        main = store.get_run(DailyCycle.run_id(day))
        for run in [main, *slot_runs(store, DailyCycle.run_id(day))] if main else []:
            proposal = (run["analysis"] or {}).get("proposal") or {}
            if proposal.get("decision") == "buy" and proposal.get("symbol"):
                reasons = focus.setdefault(proposal["symbol"], [])
                if "proposed" not in reasons:
                    reasons.append("proposed")
        for symbol in self._held():
            focus.setdefault(symbol, []).append("held")
        return self._scan("post", day, focus)

    def make_plan(self, day: date) -> str:
        """The to-do list for the next session. Needs a database: plans must persist."""
        store = self.build_audit_store()
        if isinstance(store, NullAuditStore):
            return "skipped: no database, so there is nowhere to keep the plan"
        zone = ZoneInfo(self.config.schedule.timezone)

        def next_session() -> date:
            return self.build_broker().get_clock().next_open.astimezone(zone).date()

        return Planner(store, self.config, next_session=next_session).plan(day).summary

    def brief(self, day: date) -> str | None:
        return morning_brief(self.build_audit_store(), day)

    def publish_report(self, day: date, *, send: bool = True) -> str:
        """Build the day's report, save it, and post it to the webhook if one is set."""
        report = self.build_report_builder().build(day)
        markdown = render_markdown(report)
        md_path, _ = save_report(report, markdown, Path(self.config.reporting.output_dir))
        url = self.report_webhook_url()
        if send and url:
            post_webhook(url, markdown)
            return f"saved {md_path}; sent to the webhook"
        return f"saved {md_path}"

    def report_webhook_url(self) -> str | None:
        value = self.secrets.report_webhook_url
        return value.get_secret_value() if value and value.get_secret_value() else None

    def build_report_builder(self, store: AuditStore | None = None) -> ReportBuilder:
        return ReportBuilder(
            store or self.build_audit_store(),
            self.build_market_data(),
            self.build_broker(),
            self.config,
        )

    def build_critic(self) -> CriticAgent | None:
        """The critic, or None when it is disabled. Its own gateway: own model, own usage."""
        critic = self.config.critic
        if not critic.enabled:
            return None
        llm = self.config.llm
        if critic.model:
            llm = llm.model_copy(update={"model": critic.model, "fallback_model": None})
        return CriticAgent(self.build_llm_gateway(llm), self.config)

    def build_llm_gateway(self, llm: LLMConfig | None = None) -> LLMGateway:
        llm = llm or self.config.llm
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
        ledger = self.build_cost_ledger()
        return LLMGateway(provider, llm, listener=ledger.record, budget=ledger.budget_reason)

    def build_cost_ledger(self) -> CostLedger:
        """Prices and records every model call; refuses calls over the daily budget."""
        return CostLedger(self.build_recorder(), self.config)

    def cost_report(self, days: int, store: AuditStore | None = None) -> CostReport:
        """Costs for the last `days` calendar days, today included, in market time."""
        zone = ZoneInfo(self.config.schedule.timezone)
        end = datetime.now(zone).date()
        start = end - timedelta(days=days - 1)
        begin = datetime.combine(start, time(0), tzinfo=zone)
        finish = datetime.combine(end + timedelta(days=1), time(0), tzinfo=zone)
        rows = (store or self.build_audit_store()).llm_calls_between(begin, finish)
        return cost_report(rows, self.config, start, end)
