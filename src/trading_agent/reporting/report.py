"""The daily report: what the system did and why, the result, and how it compares with SPY.

Reports are derived views. They are rebuilt from the audit trail plus a broker snapshot and
benchmark bars, so any past day can be regenerated, and nothing here is a source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.costs.summary import DayCost, cost_report, usd
from trading_agent.domain.models import Bar
from trading_agent.domain.ports import BrokerPort, MarketDataPort
from trading_agent.observability.logging import get_logger
from trading_agent.reporting.performance import (
    SYSTEM_RUN_PREFIX,
    Performance,
    TradeResult,
    intraday_return_pct,
    summarize,
    trades_from_orders,
)
from trading_agent.reporting.slots import SlotSummary, score_text, slot_runs, summarize_slot
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditError, AuditStore

MAX_BENCHMARK_BARS = 250  # the data provider's practical limit for one request


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    reason: str
    decision: str | None
    symbol: str | None
    confidence: str | None
    thesis: str | None
    evidence: list[dict[str, str]]
    risks: list[str]
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None


class CriticSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    summary: str
    reasons: list[str]
    concerns: list[str]
    unsupported: list[str]
    model: str
    input_tokens: int
    output_tokens: int


class RiskSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    failed_checks: list[str]
    score: str | None = None  # the parametric trade score (ADR 0015)
    score_detail: str | None = None


class LearningSummary(BaseModel):
    """What the day's review taught (ADR 0013)."""

    model_config = ConfigDict(frozen=True)

    lessons_used: int  # lessons the analyst had in its prompt that morning
    added: list[str]
    retired: list[str]
    active: int


class LlmTotals(BaseModel):
    """Tokens to date. Dollars come from the cost ledger (`costs`), never from a guess."""

    model_config = ConfigDict(frozen=True)

    analyses: int
    critiques: int = 0
    input_tokens: int
    output_tokens: int


class DailyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    run_id: str
    generated_at: datetime
    environment: str
    cycle_status: str | None
    cycle_summary: str | None
    steps: dict[str, str]
    analysis: AnalysisSummary | None
    critic: CriticSummary | None
    risk: RiskSummary | None
    trades: list[TradeResult]  # every trade of the day, across all slots
    exits: list[str]
    benchmark_symbol: str
    benchmark_pct: Decimal | None
    excess_pct: Decimal | None
    account_equity: Decimal | None
    performance: Performance
    llm: LlmTotals
    warnings: list[str]
    learning: LearningSummary | None = None
    costs: DayCost | None = None  # from the cost ledger (ADR 0014)
    unpriced_models: list[str] = []
    slots: list[SlotSummary] = []  # extra trade slots (ADR 0015)


class ReportBuilder:
    def __init__(
        self,
        store: AuditStore,
        market_data: MarketDataPort,
        broker: BrokerPort,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._market_data = market_data
        self._broker = broker
        self._config = config
        self._settings = config.reporting
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def build(self, day: date) -> DailyReport:
        run_id = f"{SYSTEM_RUN_PREFIX}{day:%Y%m%d}"
        warnings: list[str] = []

        try:
            run = self._store.get_run(run_id)
        except AuditError as exc:
            run, warnings = None, [*warnings, f"audit store unavailable: {exc}"]

        day_end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=self._zone)
        history_start = day_end - timedelta(days=self._settings.history_days)
        try:
            orders = self._store.orders_between(history_start, day_end)
        except AuditError as exc:
            orders = []
            warnings.append(f"orders unavailable: {exc}")
        trades = trades_from_orders(orders, self._zone)
        today = [t for t in trades if t.day == day]

        bars: tuple[Bar, ...] = ()
        symbol = self._settings.benchmark_symbol
        # The provider returns the most recent N bars, so the window must reach back from
        # today to the earliest day needed (the report day, or the oldest trade).
        today_local = self._clock().astimezone(self._zone).date()
        earliest = min([t.day for t in trades] + [day])
        span = min(MAX_BENCHMARK_BARS, max((today_local - earliest).days, 0) * 5 // 7 + 5)
        try:
            bars = self._market_data.get_daily_bars([symbol], days=span).get(symbol, ())
        except Exception as exc:
            warnings.append(f"benchmark data unavailable: {exc}")
        benchmark = {t.day: intraday_return_pct(bars, t.day, self._zone) for t in trades}
        benchmark_today = intraday_return_pct(bars, day, self._zone)
        if benchmark_today is None and bars and day < today_local and day.weekday() < 5:
            warnings.append(
                f"no {symbol} bar for {day} (a market holiday, or older than the data window)"
            )

        try:
            equity: Decimal | None = self._broker.get_account().equity
        except Exception as exc:
            equity = None
            warnings.append(f"account unavailable: {exc}")

        ledger: list[dict[str, Any]] | None
        try:
            ledger = self._store.llm_calls_between(day_end - timedelta(days=1), day_end)
        except AuditError as exc:
            ledger = None
            warnings.append(f"cost ledger unavailable: {exc}")
        costs = None if ledger is None else cost_report(ledger, self._config, day, day)

        try:
            totals = self._store.analysis_totals(SYSTEM_RUN_PREFIX)
        except AuditError:
            totals = {"analyses": 0, "critiques": 0, "input_tokens": 0, "output_tokens": 0}

        excess = None
        returns = [t.return_pct for t in today if t.return_pct is not None]
        if returns and benchmark_today is not None:
            excess = (sum(returns, Decimal(0)) / len(returns) - benchmark_today).quantize(
                Decimal("0.01")
            )

        slots: list[SlotSummary] = []
        if run is not None:  # slots exist only on days the cycle ran
            try:
                slots = [summarize_slot(r) for r in slot_runs(self._store, run_id)]
            except AuditError as exc:
                warnings.append(f"trade slots unavailable: {exc}")

        report = DailyReport(
            day=day,
            run_id=run_id,
            generated_at=self._clock(),
            environment=self._config.environment.value,
            cycle_status=run["run"]["status"] if run else None,
            cycle_summary=run["run"]["summary"] if run else None,
            steps={s["step"]: s["status"] for s in run["steps"]} if run else {},
            analysis=self._analysis(run, ledger or []),
            critic=self._critic(run),
            risk=self._risk(run),
            trades=today,
            exits=[f"{e['action']}: {e['detail']}" for e in run["position_events"]] if run else [],
            benchmark_symbol=symbol,
            benchmark_pct=benchmark_today,
            excess_pct=excess,
            account_equity=equity,
            performance=summarize(trades, benchmark),
            llm=LlmTotals(
                analyses=totals["analyses"],
                critiques=totals.get("critiques", 0),
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
            ),
            warnings=warnings,
            learning=self._learning(day, run_id, run),
            costs=costs.days[0] if costs else None,
            slots=slots,
            unpriced_models=costs.unpriced_models if costs else [],
        )
        self._log.info("report.built", day=str(day), run_id=run_id, warnings=len(warnings))
        return report

    def _analysis(
        self, run: dict[str, Any] | None, ledger: list[dict[str, Any]]
    ) -> AnalysisSummary | None:
        if not run or not run["analysis"]:
            return None
        row = run["analysis"]
        calls = [r for r in ledger if r["run_id"] == row["run_id"] and r["agent"] == "analyst"]
        cost = (
            sum((r["cost_usd"] for r in calls), Decimal(0))
            if calls and all(r["cost_usd"] is not None for r in calls)
            else None
        )
        proposal = row["proposal"] or {}
        return AnalysisSummary(
            status=row["status"],
            reason=row["reason"],
            decision=row["decision"],
            symbol=row["symbol"],
            confidence=row["confidence"],
            thesis=proposal.get("thesis"),
            evidence=[
                {"claim": e["claim"], "source": e["source_call_id"]}
                for e in proposal.get("evidence", [])
            ],
            risks=list(proposal.get("risks", [])),
            tool_calls=len(run["tool_calls"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=cost,
        )

    def _learning(
        self, day: date, run_id: str, run: dict[str, Any] | None
    ) -> LearningSummary | None:
        if not self._config.learning.enabled:
            return None
        try:
            lessons = self._store.list_lessons(include_retired=True)
        except AuditError:
            return None
        analysis = run["analysis"] if run else None
        used = len(analysis.get("lesson_ids") or []) if analysis else 0
        added = [f"[{x['category']}] {x['text']}" for x in lessons if x["source_run_id"] == run_id]
        retired = [
            f"[{x['category']}] {x['text']} ({x['retired_reason']})"
            for x in lessons
            if x["retired_at"] is not None and x["retired_at"].astimezone(self._zone).date() == day
        ]
        now = self._clock()
        active = sum(1 for x in lessons if x["status"] == "active" and x["expires_at"] > now)
        return LearningSummary(lessons_used=used, added=added, retired=retired, active=active)

    @staticmethod
    def _critic(run: dict[str, Any] | None) -> CriticSummary | None:
        row = run.get("critique") if run else None
        if not row:
            return None
        return CriticSummary(
            status=row["status"],
            summary=row["summary"],
            reasons=list(row["reasons"]),
            concerns=list(row["concerns"]),
            unsupported=[
                f"{c['source_call_id']}: {c['note']}"
                for c in row["evidence_checks"]
                if not c["supported"]
            ],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    @staticmethod
    def _risk(run: dict[str, Any] | None) -> RiskSummary | None:
        if not run or not run["risk_decisions"]:
            return None
        latest = run["risk_decisions"][-1]
        score, detail = score_text(latest.get("score"))
        return RiskSummary(
            approved=latest["approved"],
            failed_checks=list(latest["failed_checks"]),
            score=score,
            score_detail=detail,
        )


# ---------- Rendering ----------


def _money(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _signed(value: Decimal | None, unit: str = "%") -> str:
    return "n/a" if value is None else f"{value:+.2f}{unit}"


def _tokens(input_tokens: int, output_tokens: int, cost: Decimal | None) -> str:
    text = f"{input_tokens + output_tokens:,} tokens"
    return text if cost is None else f"{text} (~${cost:,.2f})"


def render_markdown(report: DailyReport) -> str:
    lines = [f"# Daily report: {report.day:%A %Y-%m-%d}", ""]
    status = (report.cycle_status or "no cycle recorded").upper()
    summary = f": {report.cycle_summary}" if report.cycle_summary else ""
    lines += [
        f"**{status}**{summary}  ",
        f"Run `{report.run_id}`, {report.environment} account",
        "",
    ]

    if report.steps:
        lines += ["| Step | Status |", "| --- | --- |"]
        lines += [f"| {step} | {status} |" for step, status in report.steps.items()]
        lines.append("")

    lines.append("## Analysis")
    analysis = report.analysis
    if analysis is None:
        lines += ["No analysis recorded.", ""]
    else:
        if analysis.decision == "buy":
            headline = f"**BUY {analysis.symbol}** (confidence {analysis.confidence})"
        elif analysis.decision == "no_trade":
            headline = f"**NO TRADE** (confidence {analysis.confidence})"
        else:
            headline = f"**{analysis.status.upper()}**: {analysis.reason}"
        lines += [
            f"{headline}. {analysis.tool_calls} tool calls, "
            f"{_tokens(analysis.input_tokens, analysis.output_tokens, analysis.cost_usd)}.",
            "",
        ]
        if analysis.thesis:
            lines += [f"> {analysis.thesis}", ""]
        if analysis.evidence:
            lines.append("Evidence:")
            lines += [f"- {e['claim']} (`{e['source']}`)" for e in analysis.evidence]
            lines.append("")
        if analysis.risks:
            lines.append("Risks:")
            lines += [f"- {risk}" for risk in analysis.risks]
            lines.append("")

    critic = report.critic
    if critic is not None:
        lines += [
            "## Critic",
            f"**{critic.status.upper()}** by {critic.model}, "
            f"{_tokens(critic.input_tokens, critic.output_tokens, None)}. {critic.summary}",
            "",
        ]
        if critic.unsupported:
            lines.append("Unsupported evidence:")
            lines += [f"- {item}" for item in critic.unsupported]
            lines.append("")
        if critic.status != "approved" and critic.reasons:
            lines.append("Why it did not pass:")
            lines += [f"- {reason}" for reason in critic.reasons]
            lines.append("")
        if critic.concerns:
            lines.append("Concerns:")
            lines += [f"- {concern}" for concern in critic.concerns]
            lines.append("")

    lines.append("## Risk check")
    if report.risk is None:
        lines += ["No risk check ran.", ""]
    elif report.risk.approved:
        lines += ["Approved: every check passed.", ""]
    else:
        lines += [f"Rejected. Failed: {', '.join(report.risk.failed_checks)}.", ""]
    if report.risk is not None and report.risk.score:
        lines += [f"Trade score: {report.risk.score} (weakest: {report.risk.score_detail}).", ""]

    if report.slots:
        lines += [
            "## Trade slots",
            "",
            "| Slot | Run | Outcome | Symbol | Score | Detail |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {s.number} | `{s.run_id}` | {(s.status or 'incomplete').upper()} | "
            f"{s.symbol or '-'} | {s.score or 'n/a'} | {s.summary or ''} |"
            for s in report.slots
        ]
        lines.append("")

    lines.append("## Trades" if len(report.trades) > 1 else "## Trade")
    if not report.trades:
        lines += ["No trade today.", ""]
    else:
        lines += [
            "| Symbol | Quantity | Entry | Exit | P&L | Return | Exit via |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for trade in report.trades:
            exit_price = "n/a" if trade.exit_price is None else f"${trade.exit_price:,.4f}"
            lines.append(
                f"| {trade.symbol} | {trade.quantity} | ${trade.entry_price:,.4f} | {exit_price} "
                f"| {_money(trade.pnl_usd)} | {_signed(trade.return_pct)} | {trade.exit_via} |"
            )
        lines.append("")
    for event in report.exits:
        lines.append(f"- {event}")
    if report.exits:
        lines.append("")
    lines += [
        f"Benchmark ({report.benchmark_symbol} open to close): {_signed(report.benchmark_pct)}. "
        f"Excess return: {_signed(report.excess_pct, ' pts')}.",
        "",
    ]

    learning = report.learning
    if learning is not None:
        lines += [
            "## Learning",
            f"The analyst used {learning.lessons_used} lesson(s) this morning; "
            f"{learning.active} active now.",
            "",
        ]
        if learning.added:
            lines.append("Learned today:")
            lines += [f"- {text}" for text in learning.added]
            lines.append("")
        if learning.retired:
            lines.append("Retired today:")
            lines += [f"- {text}" for text in learning.retired]
            lines.append("")

    costs = report.costs
    if costs is not None:
        agents = ", ".join(
            f"{name} {usd(cost.usd, cost.known_usd)}" for name, cost in costs.by_agent.items()
        )
        tokens = costs.input_tokens + costs.output_tokens
        lines += [
            "## Costs today",
            f"LLM {usd(costs.llm_usd, costs.known_llm_usd)}"
            + (f" ({agents}; {costs.calls} calls, {tokens:,} tokens)" if costs.calls else "")
            + f". Services {usd(costs.services_usd)}. "
            f"Total {usd(costs.total_usd, costs.known_llm_usd + costs.services_usd)}.",
            "",
        ]
        if report.unpriced_models:
            lines += [
                f"No price set for {', '.join(report.unpriced_models)}: add it under "
                "costs.llm_prices to see its cost.",
                "",
            ]

    perf = report.performance
    lines += [
        "## Performance to date (system trades only)",
        "",
        "| Trades | Closed | Win rate | Total P&L | Avg return | "
        f"Avg {report.benchmark_symbol} | Avg excess |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {perf.trades} | {perf.closed_trades} | {_signed(perf.win_rate_pct).lstrip('+')} "
        f"| {_money(perf.total_pnl_usd)} | {_signed(perf.avg_return_pct)} "
        f"| {_signed(perf.avg_benchmark_pct)} | {_signed(perf.avg_excess_pct, ' pts')} |",
        "",
        f"LLM usage to date: {report.llm.analyses} analyses, {report.llm.critiques} critiques, "
        f"{_tokens(report.llm.input_tokens, report.llm.output_tokens, None)}.",
        "",
        f"Account equity: {_money(report.account_equity)} ({report.environment}).",
        "",
    ]
    if report.warnings:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in report.warnings]
        lines.append("")
    lines.append(
        f"_Generated {report.generated_at:%Y-%m-%d %H:%M} UTC. "
        "Educational project, not financial advice._"
    )
    return "\n".join(lines) + "\n"
