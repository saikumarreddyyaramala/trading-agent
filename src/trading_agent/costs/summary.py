"""Cost summaries for the CLI, the daily report, and (later) the dashboard."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.costs.pricing import service_cost_per_day
from trading_agent.settings.models import AppConfig

_ZERO = Decimal(0)


class AgentCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    calls: int
    input_tokens: int
    output_tokens: int
    usd: Decimal | None  # None: some calls had no price
    known_usd: Decimal


class DayCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date | None  # None for the period total
    calls: int
    failed_calls: int
    input_tokens: int
    output_tokens: int
    llm_usd: Decimal | None  # None when any call had tokens but no price
    known_llm_usd: Decimal
    unpriced_calls: int
    by_agent: dict[str, AgentCost]
    services_usd: Decimal
    total_usd: Decimal | None


class CostReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: date
    end: date
    days: list[DayCost]
    total: DayCost
    by_model: dict[str, Decimal | None]
    unpriced_models: list[str]
    services: list[dict[str, Any]]
    daily_budget_usd: Decimal | None


def _summarize(day: date | None, rows: list[Mapping[str, Any]], services: Decimal) -> DayCost:
    agents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        agents[row["agent"]].append(row)

    def known(items: Iterable[Mapping[str, Any]]) -> Decimal:
        return sum((r["cost_usd"] for r in items if r["cost_usd"] is not None), _ZERO)

    def complete(items: list[Mapping[str, Any]]) -> bool:
        return all(r["cost_usd"] is not None for r in items)

    by_agent = {
        name: AgentCost(
            calls=len(items),
            input_tokens=sum(r["input_tokens"] for r in items),
            output_tokens=sum(r["output_tokens"] for r in items),
            usd=known(items) if complete(items) else None,
            known_usd=known(items),
        )
        for name, items in sorted(agents.items())
    }
    llm = known(rows) if complete(rows) else None
    return DayCost(
        day=day,
        calls=len(rows),
        failed_calls=sum(1 for r in rows if r["status"] != "ok"),
        input_tokens=sum(r["input_tokens"] for r in rows),
        output_tokens=sum(r["output_tokens"] for r in rows),
        llm_usd=llm,
        known_llm_usd=known(rows),
        unpriced_calls=sum(1 for r in rows if r["cost_usd"] is None),
        by_agent=by_agent,
        services_usd=services,
        total_usd=None if llm is None else llm + services,
    )


def cost_report(
    rows: Iterable[Mapping[str, Any]], config: AppConfig, start: date, end: date
) -> CostReport:
    """Every calendar day from `start` to `end`, inclusive, in market time."""
    zone = ZoneInfo(config.schedule.timezone)
    per_day = service_cost_per_day(config.costs.services)
    by_day: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[row["created_at"].astimezone(zone).date()].append(row)

    days = []
    current = start
    while current <= end:
        days.append(_summarize(current, by_day.get(current, []), per_day))
        current += timedelta(days=1)
    in_range = [r for d in days if d.day for r in by_day.get(d.day, [])]

    models: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in in_range:
        models[row["model"]].append(row)
    by_model = {
        name: sum((r["cost_usd"] for r in items), _ZERO)
        if all(r["cost_usd"] is not None for r in items)
        else None
        for name, items in sorted(models.items())
    }
    return CostReport(
        start=start,
        end=end,
        days=days,
        total=_summarize(None, in_range, per_day * len(days)),
        by_model=by_model,
        unpriced_models=[name for name, usd in by_model.items() if usd is None],
        services=[s.model_dump(mode="json") for s in config.costs.services],
        daily_budget_usd=config.costs.daily_llm_budget_usd,
    )


def _cents(value: Decimal) -> str:
    # Money rounds half up ($0.065 -> $0.07), not half-to-even as Decimal formatting does.
    return f"${value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,.2f}"


def usd(value: Decimal | None, known: Decimal | None = None) -> str:
    if value is not None:
        return _cents(value)
    return f"{_cents(known)} + unpriced" if known is not None else "unpriced"


def render_costs(report: CostReport) -> str:
    lines = [
        f"Costs {report.start} to {report.end}",
        "",
        f"{'Day':<11} {'Calls':>5} {'Failed':>6} {'Tokens':>11} {'LLM':>18} "
        f"{'Services':>9}  By agent",
    ]
    for day in report.days:
        agents = ", ".join(
            f"{name} {usd(cost.usd, cost.known_usd)}" for name, cost in day.by_agent.items()
        )
        lines.append(
            f"{day.day!s:<11} {day.calls:>5} {day.failed_calls:>6} "
            f"{day.input_tokens + day.output_tokens:>11,} "
            f"{usd(day.llm_usd, day.known_llm_usd):>18} {usd(day.services_usd):>9}  {agents}"
        )
    total = report.total
    lines += [
        "",
        f"Total: {usd(total.total_usd, total.known_llm_usd + total.services_usd)} "
        f"(LLM {usd(total.llm_usd, total.known_llm_usd)}, "
        f"services {usd(total.services_usd)}, {total.calls} calls)",
    ]
    if report.by_model:
        lines.append(
            "By model: "
            + ", ".join(f"{name} {usd(cost)}" for name, cost in report.by_model.items())
        )
    if report.daily_budget_usd is not None:
        lines.append(f"Daily LLM budget: {usd(report.daily_budget_usd)}")
    if report.unpriced_models:
        lines += [
            "",
            f"No price set for: {', '.join(report.unpriced_models)}. Add them under "
            "costs.llm_prices in config/base.yaml to see their cost; until then totals that "
            "include them are a lower bound, and the daily budget cannot count them.",
        ]
    return "\n".join(lines) + "\n"
