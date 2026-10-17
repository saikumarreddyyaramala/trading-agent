"""The cost ledger: prices every model call as it happens and enforces the daily budget."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.costs.pricing import llm_cost
from trading_agent.domain.models import LLMCallRecord
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditError, AuditRecorder


class Spend(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    known_usd: Decimal  # priced calls only
    calls: int
    unpriced_calls: int  # calls with tokens but no price: the true total is higher


class CostLedger:
    def __init__(
        self,
        recorder: AuditRecorder,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._recorder = recorder
        self._costs = config.costs
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def cost_of(self, call: LLMCallRecord) -> Decimal | None:
        if call.usage.input_tokens == 0 and call.usage.output_tokens == 0:
            return Decimal(0)  # a failed attempt, or nothing used: free whatever the price
        return llm_cost(self._costs.llm_prices, call.model, call.usage)

    def record(self, call: LLMCallRecord) -> None:
        """Write the call with its cost. Never raises: a cost record never blocks a decision."""
        cost = self.cost_of(call)
        with contextlib.suppress(AuditError):  # already logged by the recorder
            self._recorder.write("llm call", lambda s: s.record_llm_call(call, cost))
        if cost is None:
            self._log.info("costs.unpriced_call", model=call.model, agent=call.agent)

    def spend_on(self, day: date) -> Spend:
        start = datetime.combine(day, time(0), tzinfo=self._zone)
        rows = self._recorder.store.llm_calls_between(start, start + timedelta(days=1))
        priced = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        return Spend(
            day=day,
            known_usd=sum(priced, Decimal(0)),
            calls=len(rows),
            unpriced_calls=len(rows) - len(priced),
        )

    def budget_reason(self) -> str | None:
        """Why a new LLM call must be refused now, or None. Fails open when spend is
        unreadable: the budget limits spending; trading safety lives elsewhere."""
        budget = self._costs.daily_llm_budget_usd
        if budget is None:
            return None
        today = self._clock().astimezone(self._zone).date()
        try:
            spend = self.spend_on(today)
        except AuditError as exc:
            self._log.warning("costs.spend_unreadable", error=str(exc))
            return None
        if spend.known_usd >= budget:
            return (
                f"daily LLM budget of ${budget:.2f} reached: ${spend.known_usd:.2f} spent "
                f"on {today} (costs.daily_llm_budget_usd)"
            )
        return None
