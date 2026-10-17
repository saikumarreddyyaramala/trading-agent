"""Extra trade slots of a daily run (ADR 0015): `daily-YYYYMMDD-2`, `-3`, and so on."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from trading_agent.storage.store import AuditStore

MAX_SLOTS = 10  # risk.max_trades_per_day is at most 10


class SlotSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int
    run_id: str
    status: str | None
    summary: str | None
    decision: str | None
    symbol: str | None
    critic: str | None
    risk_approved: bool | None
    failed_checks: list[str]
    score: str | None  # "72.5/100 (needs 60)"
    score_detail: str | None  # the two weakest parts


def slot_runs(store: AuditStore, run_id: str) -> list[dict[str, Any]]:
    """Every recorded slot run of `run_id`, in slot order. Skipped slots have no run."""
    runs = []
    for number in range(2, MAX_SLOTS + 1):
        run = store.get_run(f"{run_id}-{number}")
        if run is not None:
            runs.append(run)
    return runs


def score_text(score: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """(summary, weakest parts) of a stored trade score."""
    if not score:
        return None, None
    needs = f"{Decimal(str(score['min_score'])).normalize():f}"
    weakest = sorted(score["components"], key=lambda c: Decimal(str(c["value"])))[:2]
    detail = "; ".join(
        f"{c['name'].lower()} {Decimal(str(c['value'])).quantize(Decimal('0.01'))}" for c in weakest
    )
    return f"{score['total']}/100 (needs {needs})", detail or None


def summarize_slot(run: dict[str, Any]) -> SlotSummary:
    run_id: str = run["run"]["run_id"]
    analysis = run["analysis"] or {}
    critique = run.get("critique")
    decision = run["risk_decisions"][-1] if run["risk_decisions"] else None
    score, detail = score_text(decision.get("score") if decision else None)
    return SlotSummary(
        number=int(run_id.rsplit("-", 1)[1]),
        run_id=run_id,
        status=run["run"]["status"],
        summary=run["run"]["summary"],
        decision=analysis.get("decision"),
        symbol=analysis.get("symbol"),
        critic=critique["status"] if critique else None,
        risk_approved=None if decision is None else decision["approved"],
        failed_checks=list(decision["failed_checks"]) if decision else [],
        score=score,
        score_detail=detail,
    )
