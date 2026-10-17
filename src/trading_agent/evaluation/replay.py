"""Replay the critic over recorded proposals, to compare a prompt or model change.

Possible because each analyst tool call's output is stored (migration 0003): the critic
sees exactly what it saw at the time. The analyst itself cannot be replayed honestly; its
tools return live data, which has moved on since.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from trading_agent.agents.critic import CriticAgent
from trading_agent.storage.store import AuditStore


class ReplayItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    symbol: str | None
    recorded: str  # the critic status at the time, or "none" if there was no review
    replayed: str
    changed: bool
    summary: str


class ReplayResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reviewed: int
    agreed: int
    now_vetoed: int  # approved (or unreviewed) then, not approved now
    now_approved: int  # not approved then, approved now
    skipped: dict[str, str]  # run id -> why it was not replayed
    items: list[ReplayItem]


def _passed(status: str) -> bool:
    return status in ("approved", "none", "disabled")


def replay_critic(store: AuditStore, critic: CriticAgent, run_ids: Iterable[str]) -> ReplayResult:
    items: list[ReplayItem] = []
    skipped: dict[str, str] = {}
    for run_id in run_ids:
        proposal = store.get_proposal(run_id)
        if proposal is None:
            continue  # nothing to review: no verified buy proposal that day
        calls = store.tool_calls(run_id)
        if not any(call.result for call in calls):
            skipped[run_id] = "no tool outputs recorded (run predates migration 0003)"
            continue
        recorded_row: dict[str, Any] | None = store.get_critique(run_id)
        recorded = recorded_row["status"] if recorded_row else "none"
        result = critic.review(run_id, proposal, calls)
        items.append(
            ReplayItem(
                run_id=run_id,
                symbol=proposal.symbol,
                recorded=recorded,
                replayed=result.status,
                changed=_passed(recorded) != (result.status == "approved"),
                summary=result.summary,
            )
        )
    return ReplayResult(
        reviewed=len(items),
        agreed=sum(1 for i in items if not i.changed),
        now_vetoed=sum(1 for i in items if i.changed and _passed(i.recorded)),
        now_approved=sum(1 for i in items if i.changed and not _passed(i.recorded)),
        skipped=skipped,
        items=items,
    )
