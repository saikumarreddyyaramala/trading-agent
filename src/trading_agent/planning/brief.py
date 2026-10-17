"""The morning brief: yesterday's plan and today's pre-market alerts, for the analyst.

Advisory context only. It carries rule-made sentences, numbers, and catalyst tags; never a
headline or other third-party text, so news cannot slip instructions into the prompt.
Operational items (open positions, failed steps, prices) are for the operator, not the
analyst, and are left out.
"""

from __future__ import annotations

from datetime import date

from trading_agent.observability.logging import get_logger
from trading_agent.storage.store import AuditError, AuditStore

MAX_PLAN_LINES = 8
MAX_ALERT_LINES = 8
_LOG = get_logger(__name__)


def morning_brief(store: AuditStore, day: date) -> str | None:
    """The brief for `day`, or None when there is nothing to say (or nothing readable)."""
    try:
        plan = store.plan_for(day)
        pre = store.get_watch(day, "pre")
    except AuditError as exc:
        _LOG.warning("brief.unavailable", day=str(day), error=str(exc))
        return None

    items = [i for i in (plan or {}).get("items", []) if i["kind"] != "ops"][:MAX_PLAN_LINES]
    if not items and pre is None:
        return None
    lines = ["Morning brief (advisory: verify everything with your tools)."]
    if items:
        lines.append("Plan from the last session:")
        lines += [f"- {i['text']}" for i in items]
    if pre is not None:
        if pre.alerts:
            lines.append("Pre-market alerts:")
            lines += [f"- {a}" for a in pre.alerts[:MAX_ALERT_LINES]]
        if pre.market is not None and pre.market.move_pct is not None:
            lines.append(f"Market: {pre.market.symbol} {pre.market.move_pct:+.2f}% pre-market.")
    return "\n".join(lines)
