"""The evaluation scorecard: the paper record measured against the go/no-go bar.

Everything comes from the audit trail (daily runs, orders) plus daily bars for the
benchmark and for counterfactuals. Only the daily cycle's own runs and trades count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.domain.models import Bar
from trading_agent.domain.ports import MarketDataPort
from trading_agent.evaluation.simulate import simulate_intraday
from trading_agent.evaluation.stats import (
    bootstrap_mean_interval,
    max_drawdown,
    mean,
    profit_factor,
)
from trading_agent.observability.logging import get_logger
from trading_agent.reporting.performance import (
    SYSTEM_RUN_PREFIX,
    intraday_return_pct,
    trades_from_orders,
)
from trading_agent.reporting.slots import slot_runs
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditError, AuditStore

MAX_BARS = 250


class Funnel(BaseModel):
    model_config = ConfigDict(frozen=True)

    calendar_days: int
    cycle_days: int
    statuses: dict[str, int]
    analyses: int
    buy_proposals: int
    no_trade_decisions: int
    analysis_failures: int
    critic: dict[str, int]
    risk_approved: int
    risk_rejected: int
    risk_failed_checks: dict[str, int]
    entries: dict[str, int]
    closed_trades: int


class Performance(BaseModel):
    model_config = ConfigDict(frozen=True)

    closed_trades: int
    wins: int
    win_rate_pct: Decimal | None
    total_pnl_usd: Decimal
    mean_return_pct: Decimal | None
    mean_benchmark_pct: Decimal | None
    mean_excess_pct: Decimal | None
    excess_interval_pct: tuple[Decimal, Decimal] | None
    max_drawdown_usd: Decimal
    profit_factor: Decimal | None
    best_trade_pct: Decimal | None
    worst_trade_pct: Decimal | None


class Counterfactual(BaseModel):
    """Simulated results of proposals a gatekeeper stopped."""

    model_config = ConfigDict(frozen=True)

    gate: Literal["critic", "risk engine"]
    stopped: int
    simulated: int
    mean_return_pct: Decimal | None
    would_have_won: int
    would_have_lost: int


class Cost(BaseModel):
    model_config = ConfigDict(frozen=True)

    analyses: int
    critiques: int
    input_tokens: int
    output_tokens: int
    total_usd: Decimal | None
    per_cycle_day_usd: Decimal | None
    calls: int = 0  # model calls in the cost ledger (daily runs only)
    unknown_reason: str | None = None  # why total_usd is None


class Criterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    threshold: str
    actual: str
    passed: bool | None  # None: could not be evaluated, which counts as a fail


class Scorecard(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: date
    end: date
    generated_at: datetime
    environment: str
    funnel: Funnel
    performance: Performance
    counterfactuals: list[Counterfactual]
    cost: Cost
    criteria: list[Criterion]
    decision: Literal["GO", "NO-GO"]
    warnings: list[str]


def _fmt(value: Decimal | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


class ScorecardBuilder:
    def __init__(
        self,
        store: AuditStore,
        market_data: MarketDataPort,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._market_data = market_data
        self._config = config
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def first_cycle_day(self) -> date | None:
        """The earliest day with a recorded daily run, or None."""
        try:
            runs = self._store.list_runs(5000)
        except AuditError:
            return None
        days = [
            datetime.strptime(r.run_id.removeprefix(SYSTEM_RUN_PREFIX), "%Y%m%d").date()
            for r in runs
            if r.kind == "daily" and r.run_id.startswith(SYSTEM_RUN_PREFIX)
        ]
        return min(days) if days else None

    def build(self, start: date, end: date) -> Scorecard:
        if end < start:
            raise ValueError("the end date is before the start date")
        warnings: list[str] = []
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        reports: list[dict[str, Any]] = []
        for day in days:
            try:
                report = self._store.get_run(f"{SYSTEM_RUN_PREFIX}{day:%Y%m%d}")
            except AuditError as exc:
                warnings.append(f"audit store unavailable: {exc}")
                break
            if report is not None:
                reports.append(report)
        runs = list(reports)  # every run, extra trade slots included (ADR 0015)
        for report in reports:
            try:
                runs += slot_runs(self._store, report["run"]["run_id"])
            except AuditError as exc:
                warnings.append(f"trade slots unavailable: {exc}")
                break

        window_start = datetime.combine(start, time(0), tzinfo=self._zone)
        window_end = datetime.combine(end + timedelta(days=1), time(0), tzinfo=self._zone)
        try:
            orders = self._store.orders_between(window_start, window_end)
        except AuditError as exc:
            orders = []
            warnings.append(f"orders unavailable: {exc}")
        trades = [t for t in trades_from_orders(orders, self._zone) if t.closed]

        stopped = self._stopped_proposals(runs)
        bars = self._bars(
            {self._config.reporting.benchmark_symbol} | {s for s, _, _, _ in stopped},
            start,
            warnings,
        )

        funnel = self._funnel(days, reports, runs, len(trades))
        performance = self._performance(trades, bars)
        counterfactuals = self._counterfactuals(stopped, bars)
        try:
            ledger = [
                row
                for row in self._store.llm_calls_between(window_start, window_end)
                if (row["run_id"] or "").startswith(SYSTEM_RUN_PREFIX)
            ]
        except AuditError as exc:
            ledger = None
            warnings.append(f"cost ledger unavailable: {exc}")
        cost = self._cost(runs, ledger, funnel.cycle_days)
        criteria = self._criteria(funnel, performance, cost)
        decision: Literal["GO", "NO-GO"] = (
            "GO" if criteria and all(c.passed is True for c in criteria) else "NO-GO"
        )
        self._log.info("evaluation.scored", start=str(start), end=str(end), decision=decision)
        return Scorecard(
            start=start,
            end=end,
            generated_at=self._clock(),
            environment=self._config.environment.value,
            funnel=funnel,
            performance=performance,
            counterfactuals=counterfactuals,
            cost=cost,
            criteria=criteria,
            decision=decision,
            warnings=warnings,
        )

    # ---------- Pieces ----------

    def _bars(
        self, symbols: set[str], start: date, warnings: list[str]
    ) -> dict[str, tuple[Bar, ...]]:
        today = self._clock().astimezone(self._zone).date()
        span = min(MAX_BARS, max((today - start).days, 0) * 5 // 7 + 5)
        try:
            return self._market_data.get_daily_bars(sorted(symbols), days=span)
        except Exception as exc:
            warnings.append(f"market data unavailable: {exc}")
            return {}

    @staticmethod
    def _stopped_proposals(
        reports: list[dict[str, Any]],
    ) -> list[tuple[str, date, Decimal, Literal["critic", "risk engine"]]]:
        """Buy proposals that the critic or the risk engine stopped: (symbol, day, price, gate)."""
        stopped: list[tuple[str, date, Decimal, Literal["critic", "risk engine"]]] = []
        for report in reports:
            analysis = report["analysis"]
            proposal = (analysis or {}).get("proposal") or {}
            if (
                not analysis
                or analysis["status"] != "proposed"
                or proposal.get("decision") != "buy"
            ):
                continue
            day = datetime.strptime(
                report["run"]["run_id"].removeprefix(SYSTEM_RUN_PREFIX)[:8], "%Y%m%d"
            ).date()
            symbol, price = proposal["symbol"], Decimal(str(proposal["reference_price"]))
            critique = report.get("critique")
            if critique and critique["status"] != "approved":
                stopped.append((symbol, day, price, "critic"))
            elif report["risk_decisions"] and not report["risk_decisions"][-1]["approved"]:
                stopped.append((symbol, day, price, "risk engine"))
        return stopped

    @staticmethod
    def _funnel(
        days: list[date],
        reports: list[dict[str, Any]],
        runs: list[dict[str, Any]],
        closed: int,
    ) -> Funnel:
        """Day outcomes come from the daily runs; every other count includes the slots."""
        statuses: Counter[str] = Counter(
            report["run"]["status"] or "incomplete" for report in reports
        )
        critic: Counter[str] = Counter()
        entries: Counter[str] = Counter()
        failed_checks: Counter[str] = Counter()
        analyses = buys = no_trades = failures = approved = rejected = 0
        for report in runs:
            analysis = report["analysis"]
            if analysis:
                analyses += 1
                proposal = analysis.get("proposal") or {}
                if analysis["status"] == "proposed" and proposal.get("decision") == "buy":
                    buys += 1
                elif analysis["status"] == "no_trade":
                    no_trades += 1
                elif analysis["status"] == "failed":
                    failures += 1
            if report.get("critique"):
                critic[report["critique"]["status"]] += 1
            if report["risk_decisions"]:
                latest = report["risk_decisions"][-1]
                if latest["approved"]:
                    approved += 1
                else:
                    rejected += 1
                    failed_checks.update(latest["failed_checks"])
            entry = next((s for s in report["steps"] if s["step"] == "entry"), None)
            if entry and entry["status"] not in ("skipped", "rejected"):
                entries[entry["status"]] += 1
        return Funnel(
            calendar_days=len(days),
            cycle_days=len(reports),
            statuses=dict(statuses),
            analyses=analyses,
            buy_proposals=buys,
            no_trade_decisions=no_trades,
            analysis_failures=failures,
            critic=dict(critic),
            risk_approved=approved,
            risk_rejected=rejected,
            risk_failed_checks=dict(failed_checks.most_common()),
            entries=dict(entries),
            closed_trades=closed,
        )

    def _performance(self, trades: list[Any], bars: dict[str, tuple[Bar, ...]]) -> Performance:
        benchmark = bars.get(self._config.reporting.benchmark_symbol, ())
        returns = [t.return_pct for t in trades if t.return_pct is not None]
        pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
        excess: list[Decimal] = []
        benchmarks: list[Decimal] = []
        for trade in trades:
            spy = intraday_return_pct(benchmark, trade.day, self._zone)
            if spy is not None and trade.return_pct is not None:
                benchmarks.append(spy)
                excess.append(trade.return_pct - spy)
        wins = sum(1 for p in pnls if p > 0)
        return Performance(
            closed_trades=len(trades),
            wins=wins,
            win_rate_pct=(Decimal(wins) / len(pnls) * 100).quantize(Decimal("0.01"))
            if pnls
            else None,
            total_pnl_usd=sum(pnls, Decimal("0")).quantize(Decimal("0.01")),
            mean_return_pct=mean(returns),
            mean_benchmark_pct=mean(benchmarks),
            mean_excess_pct=mean(excess),
            excess_interval_pct=bootstrap_mean_interval(
                excess, samples=self._config.evaluation.bootstrap_samples
            ),
            max_drawdown_usd=max_drawdown(pnls),
            profit_factor=profit_factor(pnls),
            best_trade_pct=max(returns) if returns else None,
            worst_trade_pct=min(returns) if returns else None,
        )

    def _counterfactuals(
        self,
        stopped: list[tuple[str, date, Decimal, Literal["critic", "risk engine"]]],
        bars: dict[str, tuple[Bar, ...]],
    ) -> list[Counterfactual]:
        risk = self._config.risk
        results: list[Counterfactual] = []
        for gate in ("critic", "risk engine"):
            simulated: list[Decimal] = []
            group = [s for s in stopped if s[3] == gate]
            for symbol, day, price, _ in group:
                bar = next(
                    (
                        b
                        for b in bars.get(symbol, ())
                        if b.timestamp.astimezone(self._zone).date() == day
                    ),
                    None,
                )
                if bar is not None:
                    simulated.append(
                        simulate_intraday(
                            bar,
                            price,
                            stop_loss_pct=risk.stop_loss_pct,
                            take_profit_pct=risk.take_profit_pct,
                        ).return_pct
                    )
            results.append(
                Counterfactual(
                    gate=gate,
                    stopped=len(group),
                    simulated=len(simulated),
                    mean_return_pct=mean(simulated),
                    would_have_won=sum(1 for r in simulated if r > 0),
                    would_have_lost=sum(1 for r in simulated if r < 0),
                )
            )
        return results

    def _cost(
        self,
        reports: list[dict[str, Any]],
        ledger: list[dict[str, Any]] | None,
        cycle_days: int,
    ) -> Cost:
        """Dollars come only from the cost ledger, priced when each call happened."""
        analyses = [r["analysis"] for r in reports if r["analysis"]]
        critiques = [r["critique"] for r in reports if r.get("critique")]
        rows = ledger or []
        if rows:
            inputs = sum(r["input_tokens"] for r in rows)
            outputs = sum(r["output_tokens"] for r in rows)
        else:  # history from before the ledger: tokens are known, dollars are not
            inputs = sum(a["input_tokens"] for a in analyses) + sum(
                c["input_tokens"] for c in critiques
            )
            outputs = sum(a["output_tokens"] for a in analyses) + sum(
                c["output_tokens"] for c in critiques
            )

        unpriced = sorted({r["model"] for r in rows if r["cost_usd"] is None})
        logged_runs = {r["run_id"] for r in rows}
        unlogged = [a for a in analyses if a["run_id"] not in logged_runs]
        reason: str | None = None
        if ledger is None:
            reason = "the cost ledger could not be read"
        elif unpriced:
            reason = f"no price for {', '.join(unpriced)} (costs.llm_prices)"
        elif unlogged:
            reason = f"no cost records for {len(unlogged)} analyses (recorded before the ledger)"
        elif not rows:
            reason = "no cost records in this window"

        total = None
        if reason is None:
            total = sum((r["cost_usd"] for r in rows), Decimal(0)).quantize(Decimal("0.01"))
        return Cost(
            analyses=len(analyses),
            critiques=len(critiques),
            input_tokens=inputs,
            output_tokens=outputs,
            total_usd=total,
            per_cycle_day_usd=(total / cycle_days).quantize(Decimal("0.01"))
            if total is not None and cycle_days
            else None,
            calls=len(rows),
            unknown_reason=reason,
        )

    def _criteria(self, funnel: Funnel, perf: Performance, cost: Cost) -> list[Criterion]:
        bar = self._config.evaluation

        def at_least(
            name: str, actual: Decimal | int | None, threshold: Decimal | int, unit: str = ""
        ) -> Criterion:
            return Criterion(
                name=name,
                threshold=f">= {threshold}{unit}",
                actual=_fmt(None if actual is None else Decimal(actual), unit),
                passed=None if actual is None else actual >= threshold,
            )

        error_days = funnel.statuses.get("error", 0)
        error_pct = (
            (Decimal(error_days) / funnel.cycle_days * 100).quantize(Decimal("0.01"))
            if funnel.cycle_days
            else None
        )
        interval_low = perf.excess_interval_pct[0] if perf.excess_interval_pct else None
        criteria = [
            at_least("Cycle days", funnel.cycle_days, bar.min_cycle_days),
            at_least("Closed trades", perf.closed_trades, bar.min_closed_trades),
            at_least("Win rate", perf.win_rate_pct, bar.min_win_rate_pct, "%"),
            Criterion(
                name="Mean excess return vs SPY",
                threshold=f"> {bar.min_mean_excess_pct}%",
                actual=_fmt(perf.mean_excess_pct, "%"),
                passed=None
                if perf.mean_excess_pct is None
                else perf.mean_excess_pct > bar.min_mean_excess_pct,
            ),
            Criterion(
                name="Excess return, 90% interval low end",
                threshold=f"> {bar.min_excess_ci_lower_pct}%",
                actual=_fmt(interval_low, "%"),
                passed=None if interval_low is None else interval_low > bar.min_excess_ci_lower_pct,
            ),
            Criterion(
                name="Total P&L",
                threshold="> $0",
                actual=f"${perf.total_pnl_usd}",
                passed=perf.total_pnl_usd > 0 if perf.closed_trades else None,
            ),
            Criterion(
                name="Max drawdown",
                threshold=f"<= ${bar.max_drawdown_usd:.2f}",
                actual=f"${perf.max_drawdown_usd}",
                passed=perf.max_drawdown_usd <= bar.max_drawdown_usd
                if perf.closed_trades
                else None,
            ),
            Criterion(
                name="Cycle days with errors",
                threshold=f"<= {bar.max_error_day_pct}%",
                actual=_fmt(error_pct, "%"),
                passed=None if error_pct is None else error_pct <= bar.max_error_day_pct,
            ),
        ]
        if bar.max_llm_cost_per_day_usd is not None:
            criteria.append(
                Criterion(
                    name="LLM cost per cycle day",
                    threshold=f"<= ${bar.max_llm_cost_per_day_usd:.2f}",
                    actual=f"unknown: {cost.unknown_reason or 'no cycle days'}"
                    if cost.per_cycle_day_usd is None
                    else f"${cost.per_cycle_day_usd}",
                    passed=None
                    if cost.per_cycle_day_usd is None
                    else cost.per_cycle_day_usd <= bar.max_llm_cost_per_day_usd,
                )
            )
        return criteria


# ---------- Rendering ----------


def _top(counts: dict[str, int], n: int = 3) -> str:
    return ", ".join(f"{name} {count}" for name, count in list(counts.items())[:n])


def render_scorecard(card: Scorecard) -> str:
    f, p, c = card.funnel, card.performance, card.cost
    interval = (
        "n/a"
        if p.excess_interval_pct is None
        else f"{p.excess_interval_pct[0]}% to {p.excess_interval_pct[1]}%"
    )
    lines = [
        f"# Evaluation: {card.start} to {card.end}",
        "",
        f"## Decision: **{card.decision}**",
        "",
        "| Criterion | Required | Actual | Result |",
        "| --- | --- | --- | --- |",
    ]
    for criterion in card.criteria:
        result = {True: "pass", False: "FAIL", None: "FAIL (no data)"}[criterion.passed]
        lines.append(
            f"| {criterion.name} | {criterion.threshold} | {criterion.actual} | {result} |"
        )
    lines += [
        "",
        "## Funnel",
        "",
        f"- {f.cycle_days} cycle days out of {f.calendar_days} calendar days. "
        f"Outcomes: {', '.join(f'{k} {v}' for k, v in sorted(f.statuses.items())) or 'none'}.",
        f"- Analyses: {f.analyses} ({f.buy_proposals} buy proposals, "
        f"{f.no_trade_decisions} no-trade, {f.analysis_failures} failed).",
        f"- Critic: {', '.join(f'{k} {v}' for k, v in sorted(f.critic.items())) or 'no reviews'}.",
        f"- Risk engine: {f.risk_approved} approved, {f.risk_rejected} rejected"
        + (f" (most failed: {_top(f.risk_failed_checks)})" if f.risk_failed_checks else "")
        + ".",
        f"- Entries: {', '.join(f'{k} {v}' for k, v in sorted(f.entries.items())) or 'none'}. "
        f"Closed trades: {f.closed_trades}.",
        "",
        "## Performance (system trades only)",
        "",
        "| Closed | Win rate | Total P&L | Mean return | Mean SPY | Mean excess | 90% interval | "
        "Max drawdown | Profit factor | Best | Worst |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {p.closed_trades} | {_fmt(p.win_rate_pct, '%')} | ${p.total_pnl_usd} | "
        f"{_fmt(p.mean_return_pct, '%')} | {_fmt(p.mean_benchmark_pct, '%')} | "
        f"{_fmt(p.mean_excess_pct, '%')} | {interval} | ${p.max_drawdown_usd} | "
        f"{_fmt(p.profit_factor)} | {_fmt(p.best_trade_pct, '%')} | "
        f"{_fmt(p.worst_trade_pct, '%')} |",
        "",
        "## What the gatekeepers stopped (simulated, conservative)",
        "",
        "| Gate | Stopped | Simulated | Mean simulated return | Would have won | Would have lost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cf in card.counterfactuals:
        lines.append(
            f"| {cf.gate} | {cf.stopped} | {cf.simulated} | {_fmt(cf.mean_return_pct, '%')} | "
            f"{cf.would_have_won} | {cf.would_have_lost} |"
        )
    lines += [
        "",
        "A negative mean means the gate saved money; a positive one means it blocked winners.",
        "",
        "## Cost",
        "",
        f"{c.analyses} analyses and {c.critiques} critiques used "
        f"{c.input_tokens + c.output_tokens:,} tokens"
        + (
            f" (${c.total_usd}, ${c.per_cycle_day_usd} per cycle day)."
            if c.total_usd is not None
            else f". Dollars unknown: {c.unknown_reason}."
        ),
        "",
    ]
    if card.warnings:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in card.warnings]
        lines.append("")
    lines.append(
        f"_Generated {card.generated_at:%Y-%m-%d %H:%M} UTC on the {card.environment} record. "
        "Past paper results do not guarantee live results. Not financial advice._"
    )
    return "\n".join(lines) + "\n"
