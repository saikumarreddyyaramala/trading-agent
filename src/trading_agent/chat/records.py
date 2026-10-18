"""The chat bot's read-only view of the records (ADR 0022), built on the dashboard's data
layer and the daily report builder, so chat and dashboard always agree."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from trading_agent.dashboard.data import DashboardData, jsonable
from trading_agent.reporting.report import ReportBuilder
from trading_agent.storage.store import AuditStore

PLAN_LOOKBACK_DAYS = 7


class ChatRecords:
    def __init__(self, store: AuditStore, dashboard: DashboardData, reports: ReportBuilder) -> None:
        self._store = store
        self._dashboard = dashboard
        self._reports = reports

    def overview(self) -> dict[str, Any]:
        return self._dashboard.overview()

    def day_report(self, day: date) -> dict[str, Any] | None:
        data: dict[str, Any] = jsonable(self._reports.build(day))
        return data

    def trades(self, days: int) -> list[dict[str, Any]]:
        return self._dashboard.trades(days)

    def costs(self, days: int) -> dict[str, Any]:
        return self._dashboard.costs(days)

    def plan(self, day: date) -> dict[str, Any] | None:
        """The plan for `day`, or else the latest one made for a later session."""
        found = self._store.plan_for(day)
        for back in range(PLAN_LOOKBACK_DAYS + 1):
            if found is not None:
                break
            made = self._store.plan_from(day - timedelta(days=back))
            if made is not None and made["for_day"] >= day:
                found = made
        result: dict[str, Any] | None = None if found is None else jsonable(found)
        return result
