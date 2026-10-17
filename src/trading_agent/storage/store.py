"""The audit store: what the system decided and did, and why, keyed by run id."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Protocol

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict

from trading_agent import __version__
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.agents.critic import CriticResult
from trading_agent.domain.models import (
    BrokerOrder,
    DayPlan,
    Lesson,
    LLMCallRecord,
    Playbook,
    RiskDecision,
    TradeProposal,
)
from trading_agent.execution.executor import ManageResult
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig
from trading_agent.storage import schema
from trading_agent.watch.market_watch import WatchReport

_RUN_ORDER = re.compile(r"^ta-(?P<run>[A-Za-z0-9-]+)-(?P<role>entry|stop)$")


class NewLessonRow(BaseModel):
    """A lesson to store: what was learned, its category, and the evidence behind it."""

    model_config = ConfigDict(frozen=True)

    text: str
    category: str
    evidence: dict[str, Any]


class AuditError(RuntimeError):
    """The audit store could not be written or read."""


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    kind: str
    status: str | None
    started_at: datetime
    finished_at: datetime | None
    summary: str | None


def order_role(order: BrokerOrder) -> tuple[str, str | None]:
    """(role, run id) for an order, read from the client order id the executor assigned."""
    match = _RUN_ORDER.match(order.client_order_id)
    if match:
        return match["role"], match["run"]
    if order.client_order_id.startswith("ta-restore-"):
        return "restore_stop", None
    return ("exit" if order.side == "sell" else "external"), None


class AuditStore(Protocol):
    def start_run(self, run_id: str, kind: str) -> None: ...

    def finish_run(self, run_id: str, status: str, summary: str) -> None: ...

    def record_analysis(self, result: AnalystResult) -> None: ...

    def record_risk_decision(self, run_id: str, decision: RiskDecision) -> None: ...

    def record_orders(self, run_id: str | None, orders: Iterable[BrokerOrder]) -> None: ...

    def record_position_event(self, run_id: str | None, result: ManageResult) -> None: ...

    def record_step(
        self, run_id: str, step: str, status: str, detail: str, started_at: datetime
    ) -> None: ...

    def completed_steps(self, run_id: str) -> dict[str, str]:
        """Step name to status for the steps a run has already finished."""
        ...

    def get_proposal(self, run_id: str) -> TradeProposal | None:
        """The verified buy proposal recorded for a run, if there is one."""
        ...

    def list_runs(self, limit: int) -> list[RunSummary]: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def orders_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Orders submitted in [start, end), oldest first."""
        ...

    def record_critique(self, result: CriticResult) -> None: ...

    def get_critique(self, run_id: str) -> dict[str, Any] | None: ...

    def tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        """The analyst's tool calls for a run, with their outputs, in call order."""
        ...

    def analysis_totals(self, run_id_prefix: str) -> dict[str, int]:
        """Analyses and critiques, and their combined token usage, for runs with a prefix."""
        ...

    def environments(self) -> set[str]:
        """The environments (paper, live) this database holds runs for."""
        ...

    def record_lessons(
        self,
        run_id: str | None,
        lessons: Sequence[NewLessonRow],
        *,
        expires_at: datetime,
        max_active: int,
    ) -> list[int]:
        """Store new lessons; retire the oldest active ones beyond `max_active`."""
        ...

    def active_lessons(self, now: datetime, limit: int) -> list[Lesson]:
        """Unretired, unexpired lessons, newest first."""
        ...

    def retire_lesson(self, lesson_id: int, reason: str) -> bool: ...

    def list_lessons(self, *, include_retired: bool) -> list[dict[str, Any]]: ...

    def record_llm_call(self, call: LLMCallRecord, cost_usd: Decimal | None) -> None: ...

    def llm_calls_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]: ...

    def record_watch(self, run_id: str | None, report: WatchReport) -> None: ...

    def get_watch(self, day: date, phase: str) -> WatchReport | None:
        """The latest scan of that phase for that day."""
        ...

    def record_plan(self, run_id: str | None, plan: DayPlan) -> None: ...

    def plan_for(self, day: date) -> dict[str, Any] | None:
        """The latest plan made for `day`."""
        ...

    def plan_from(self, day: date) -> dict[str, Any] | None:
        """The latest plan made at the end of `day`."""
        ...

    def promote_lesson(self, lesson_id: int) -> bool: ...

    def record_playbook(self, playbook: Playbook) -> int: ...

    def list_playbooks(self, *, include_retired: bool) -> list[Playbook]: ...

    def set_playbook_status(self, name: str, status: str, reason: str) -> bool: ...

    def record_guidebook(self, content: str, changes: Sequence[str]) -> int: ...

    def latest_guidebook(self) -> dict[str, Any] | None: ...

    def guidebook(self, version: int) -> dict[str, Any] | None: ...

    def analysis_contexts(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """run_id, started_at, lesson_ids, and playbook of analyses in [start, end)."""
        ...


class NullAuditStore:
    """Used when no database is configured: records nothing, finds nothing."""

    def start_run(self, run_id: str, kind: str) -> None:
        return None

    def finish_run(self, run_id: str, status: str, summary: str) -> None:
        return None

    def record_analysis(self, result: AnalystResult) -> None:
        return None

    def record_risk_decision(self, run_id: str, decision: RiskDecision) -> None:
        return None

    def record_orders(self, run_id: str | None, orders: Iterable[BrokerOrder]) -> None:
        return None

    def record_position_event(self, run_id: str | None, result: ManageResult) -> None:
        return None

    def record_step(
        self, run_id: str, step: str, status: str, detail: str, started_at: datetime
    ) -> None:
        return None

    def completed_steps(self, run_id: str) -> dict[str, str]:
        return {}

    def get_proposal(self, run_id: str) -> TradeProposal | None:
        return None

    def list_runs(self, limit: int) -> list[RunSummary]:
        return []

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return None

    def orders_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return []

    def record_critique(self, result: CriticResult) -> None:
        return None

    def get_critique(self, run_id: str) -> dict[str, Any] | None:
        return None

    def tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        return []

    def analysis_totals(self, run_id_prefix: str) -> dict[str, int]:
        return {"analyses": 0, "critiques": 0, "input_tokens": 0, "output_tokens": 0}

    def environments(self) -> set[str]:
        return set()

    def record_lessons(
        self,
        run_id: str | None,
        lessons: Sequence[NewLessonRow],
        *,
        expires_at: datetime,
        max_active: int,
    ) -> list[int]:
        return []

    def active_lessons(self, now: datetime, limit: int) -> list[Lesson]:
        return []

    def retire_lesson(self, lesson_id: int, reason: str) -> bool:
        return False

    def list_lessons(self, *, include_retired: bool) -> list[dict[str, Any]]:
        return []

    def record_llm_call(self, call: LLMCallRecord, cost_usd: Decimal | None) -> None:
        return None

    def llm_calls_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return []

    def record_watch(self, run_id: str | None, report: WatchReport) -> None:
        return None

    def get_watch(self, day: date, phase: str) -> WatchReport | None:
        return None

    def record_plan(self, run_id: str | None, plan: DayPlan) -> None:
        return None

    def plan_for(self, day: date) -> dict[str, Any] | None:
        return None

    def plan_from(self, day: date) -> dict[str, Any] | None:
        return None

    def promote_lesson(self, lesson_id: int) -> bool:
        return False

    def record_playbook(self, playbook: Playbook) -> int:
        return 0

    def list_playbooks(self, *, include_retired: bool) -> list[Playbook]:
        return []

    def set_playbook_status(self, name: str, status: str, reason: str) -> bool:
        return False

    def record_guidebook(self, content: str, changes: Sequence[str]) -> int:
        return 0

    def latest_guidebook(self) -> dict[str, Any] | None:
        return None

    def guidebook(self, version: int) -> dict[str, Any] | None:
        return None

    def analysis_contexts(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return []


class SqlAuditStore:
    def __init__(
        self,
        engine: sa.Engine,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))

    def _write(self, what: str, fn: Callable[[sa.Connection], None]) -> None:
        try:
            with self._engine.begin() as connection:
                fn(connection)
        except sa.exc.SQLAlchemyError as exc:
            raise AuditError(f"could not record {what}: {type(exc).__name__}: {exc}") from exc

    def _run_exists(self, connection: sa.Connection, run_id: str | None) -> bool:
        if run_id is None:
            return False
        query = sa.select(schema.runs.c.run_id).where(schema.runs.c.run_id == run_id)
        return connection.execute(query).first() is not None

    # ---------- Writes ----------

    def start_run(self, run_id: str, kind: str) -> None:
        def write(connection: sa.Connection) -> None:
            if self._run_exists(connection, run_id):
                return  # a retried run keeps its original record
            connection.execute(
                schema.runs.insert().values(
                    run_id=run_id,
                    kind=kind,
                    environment=self._config.environment.value,
                    app_version=__version__,
                    started_at=self._clock(),
                    config=self._config.model_dump(mode="json"),
                )
            )

        self._write(f"run {run_id}", write)

    def finish_run(self, run_id: str, status: str, summary: str) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.runs.update()
                .where(schema.runs.c.run_id == run_id)
                .values(finished_at=self._clock(), status=status, summary=summary[:2000])
            )

        self._write(f"run {run_id} result", write)

    def record_analysis(self, result: AnalystResult) -> None:
        proposal = result.proposal

        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.analyses.insert().values(
                    run_id=result.run_id,
                    status=result.status,
                    reason=result.reason,
                    decision=proposal.decision if proposal else None,
                    symbol=proposal.symbol if proposal else None,
                    confidence=proposal.confidence if proposal else None,
                    reference_price=proposal.reference_price if proposal else None,
                    prompt_version=result.prompt_version,
                    turns=result.turns,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    duration_seconds=result.duration_seconds,
                    started_at=result.started_at,
                    proposal=proposal.model_dump(mode="json") if proposal else None,
                    lesson_ids=list(result.lesson_ids),
                )
            )
            if result.tool_calls:
                connection.execute(
                    schema.tool_calls.insert(),
                    [
                        {
                            "run_id": result.run_id,
                            "seq": seq,
                            "call_id": call.call_id,
                            "tool": call.tool,
                            "arguments": call.arguments,
                            "is_error": call.is_error,
                            "result": call.result,
                        }
                        for seq, call in enumerate(result.tool_calls, start=1)
                    ],
                )

        self._write(f"analysis for run {result.run_id}", write)

    def record_risk_decision(self, run_id: str, decision: RiskDecision) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.risk_decisions.insert().values(
                    run_id=run_id,
                    symbol=decision.symbol,
                    approved=decision.approved,
                    evaluated_at=decision.evaluated_at,
                    failed_checks=[c.name for c in decision.failed_checks],
                    checks=[c.model_dump(mode="json") for c in decision.checks],
                    plan=decision.plan.model_dump(mode="json") if decision.plan else None,
                    score=decision.score.model_dump(mode="json") if decision.score else None,
                )
            )

        self._write(f"risk decision for run {run_id}", write)

    def record_orders(self, run_id: str | None, orders: Iterable[BrokerOrder]) -> None:
        batch = list(orders)
        if not batch:
            return

        def write(connection: sa.Connection) -> None:
            now = self._clock()
            table = schema.orders
            for order in batch:
                role, parsed_run = order_role(order)
                owner = run_id or parsed_run
                values: dict[str, Any] = {
                    "status": order.status,
                    "filled_quantity": order.filled_quantity,
                    "filled_avg_price": order.filled_avg_price,
                    "updated_at": now,
                }
                exists = connection.execute(
                    sa.select(table.c.id).where(table.c.id == order.id)
                ).first()
                if exists:
                    connection.execute(table.update().where(table.c.id == order.id).values(values))
                    continue
                connection.execute(
                    table.insert().values(
                        id=order.id,
                        client_order_id=order.client_order_id,
                        run_id=owner if self._run_exists(connection, owner) else None,
                        role=role,
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        submitted_at=order.submitted_at,
                        recorded_at=now,
                        **values,
                    )
                )

        self._write(f"{len(batch)} order(s)", write)

    def record_position_event(self, run_id: str | None, result: ManageResult) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.position_events.insert().values(
                    run_id=run_id if self._run_exists(connection, run_id) else None,
                    at=self._clock(),
                    symbol=result.symbol,
                    action=result.action,
                    detail=result.detail,
                )
            )

        self._write(f"position event {result.action}", write)
        self.record_orders(run_id, result.orders)

    def record_step(
        self, run_id: str, step: str, status: str, detail: str, started_at: datetime
    ) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.run_steps.insert().values(
                    run_id=run_id,
                    step=step,
                    status=status,
                    detail=detail[:2000],
                    started_at=started_at,
                    finished_at=self._clock(),
                )
            )

        self._write(f"step {step} of run {run_id}", write)

    # ---------- Reads ----------

    def completed_steps(self, run_id: str) -> dict[str, str]:
        steps = schema.run_steps

        def read(connection: sa.Connection) -> dict[str, str]:
            query = sa.select(steps.c.step, steps.c.status).where(steps.c.run_id == run_id)
            return {row.step: row.status for row in connection.execute(query)}

        return self._read(f"steps of run {run_id}", read)

    def get_proposal(self, run_id: str) -> TradeProposal | None:
        analyses = schema.analyses

        def read(connection: sa.Connection) -> TradeProposal | None:
            query = sa.select(analyses.c.proposal).where(
                analyses.c.run_id == run_id, analyses.c.status == "proposed"
            )
            raw = connection.execute(query).scalar_one_or_none()
            return None if raw is None else TradeProposal.model_validate(raw)

        return self._read(f"proposal of run {run_id}", read)

    def _read[T](self, what: str, fn: Callable[[sa.Connection], T]) -> T:
        try:
            with self._engine.connect() as connection:
                return fn(connection)
        except sa.exc.SQLAlchemyError as exc:
            raise AuditError(f"could not read {what}: {type(exc).__name__}: {exc}") from exc

    def list_runs(self, limit: int) -> list[RunSummary]:
        runs = schema.runs

        def read(connection: sa.Connection) -> list[RunSummary]:
            rows = connection.execute(
                sa.select(runs).order_by(runs.c.started_at.desc()).limit(limit)
            ).mappings()
            return [
                RunSummary(
                    run_id=row["run_id"],
                    kind=row["kind"],
                    status=row["status"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    summary=row["summary"],
                )
                for row in rows
            ]

        return self._read("runs", read)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Everything recorded for one run, as plain data (decimals, datetimes, JSON)."""

        def rows(
            connection: sa.Connection, table: sa.Table, *order_by: Any
        ) -> list[dict[str, Any]]:
            query = sa.select(table).where(table.c.run_id == run_id).order_by(*order_by)
            return [dict(row) for row in connection.execute(query).mappings()]

        def read(connection: sa.Connection) -> dict[str, Any] | None:
            run = (
                connection.execute(sa.select(schema.runs).where(schema.runs.c.run_id == run_id))
                .mappings()
                .first()
            )
            if run is None:
                return None
            analyses = rows(connection, schema.analyses, schema.analyses.c.id)
            return {
                "run": dict(run),
                "analysis": analyses[0] if analyses else None,
                "tool_calls": rows(connection, schema.tool_calls, schema.tool_calls.c.seq),
                "risk_decisions": rows(
                    connection, schema.risk_decisions, schema.risk_decisions.c.id
                ),
                # Orders recorded in one batch share a timestamp; break ties deterministically.
                "orders": rows(
                    connection,
                    schema.orders,
                    schema.orders.c.recorded_at,
                    schema.orders.c.submitted_at,
                    schema.orders.c.client_order_id,
                ),
                "position_events": rows(
                    connection, schema.position_events, schema.position_events.c.id
                ),
                "steps": rows(connection, schema.run_steps, schema.run_steps.c.id),
                "critique": next(
                    iter(rows(connection, schema.critiques, schema.critiques.c.id)), None
                ),
            }

        return self._read(f"run {run_id}", read)

    def orders_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        table = schema.orders

        def read(connection: sa.Connection) -> list[dict[str, Any]]:
            query = (
                sa.select(table)
                .where(table.c.submitted_at >= start, table.c.submitted_at < end)
                .order_by(table.c.submitted_at, table.c.client_order_id)
            )
            return [dict(row) for row in connection.execute(query).mappings()]

        return self._read("orders", read)

    def analysis_totals(self, run_id_prefix: str) -> dict[str, int]:
        def totals(connection: sa.Connection, table: sa.Table) -> tuple[int, int, int]:
            query = sa.select(
                sa.func.count(table.c.id),
                sa.func.coalesce(sa.func.sum(table.c.input_tokens), 0),
                sa.func.coalesce(sa.func.sum(table.c.output_tokens), 0),
            ).where(table.c.run_id.startswith(run_id_prefix, autoescape=True))
            count, inputs, outputs = connection.execute(query).one()
            return int(count), int(inputs), int(outputs)

        def read(connection: sa.Connection) -> dict[str, int]:
            analyses = totals(connection, schema.analyses)
            critiques = totals(connection, schema.critiques)
            return {
                "analyses": analyses[0],
                "critiques": critiques[0],
                "input_tokens": analyses[1] + critiques[1],
                "output_tokens": analyses[2] + critiques[2],
            }

        return self._read("analysis totals", read)

    def record_lessons(
        self,
        run_id: str | None,
        lessons: Sequence[NewLessonRow],
        *,
        expires_at: datetime,
        max_active: int,
    ) -> list[int]:
        table = schema.lessons
        added: list[int] = []

        def write(connection: sa.Connection) -> None:
            now = self._clock()
            owner = run_id if self._run_exists(connection, run_id) else None
            for lesson in lessons:
                result = connection.execute(
                    table.insert().values(
                        source_run_id=owner,
                        created_at=now,
                        expires_at=expires_at,
                        category=lesson.category,
                        text=lesson.text,
                        evidence=lesson.evidence,
                        status="active",
                    )
                )
                key = result.inserted_primary_key
                if key is None:  # pragma: no cover - every insert here has a primary key
                    raise sa.exc.InvalidRequestError("lesson insert returned no id")
                added.append(int(key[0]))
            # Keep the memory small: retire the oldest active lessons beyond the cap.
            active = connection.execute(
                sa.select(table.c.id)
                .where(
                    table.c.status == "active",
                    table.c.expires_at > now,
                    table.c.promoted_at.is_(None),
                )
                .order_by(table.c.created_at.desc(), table.c.id.desc())
            ).scalars()
            surplus = list(active)[max_active:]
            if surplus:
                connection.execute(
                    table.update()
                    .where(table.c.id.in_(surplus))
                    .values(
                        status="retired",
                        retired_at=now,
                        retired_reason="capacity: replaced by newer lessons",
                    )
                )

        self._write(f"{len(lessons)} lesson(s)", write)
        return added

    def active_lessons(self, now: datetime, limit: int) -> list[Lesson]:
        table = schema.lessons

        def read(connection: sa.Connection) -> list[Lesson]:
            query = (
                sa.select(table)
                .where(
                    table.c.status == "active",
                    sa.or_(table.c.expires_at > now, table.c.promoted_at.is_not(None)),
                )
                .order_by(
                    table.c.promoted_at.is_(None), table.c.created_at.desc(), table.c.id.desc()
                )
                .limit(limit)
            )
            return [
                Lesson(
                    id=row["id"],
                    text=row["text"],
                    category=row["category"],
                    source_run_id=row["source_run_id"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    promoted=row["promoted_at"] is not None,
                )
                for row in connection.execute(query).mappings()
            ]

        return self._read("lessons", read)

    def retire_lesson(self, lesson_id: int, reason: str) -> bool:
        table = schema.lessons
        changed: list[int] = []

        def write(connection: sa.Connection) -> None:
            result = connection.execute(
                table.update()
                .where(table.c.id == lesson_id, table.c.status == "active")
                .values(status="retired", retired_at=self._clock(), retired_reason=reason[:500])
            )
            changed.append(result.rowcount)

        self._write(f"lesson {lesson_id}", write)
        return bool(changed and changed[0])

    def list_lessons(self, *, include_retired: bool) -> list[dict[str, Any]]:
        table = schema.lessons

        def read(connection: sa.Connection) -> list[dict[str, Any]]:
            query = sa.select(table).order_by(table.c.created_at.desc(), table.c.id.desc())
            if not include_retired:
                query = query.where(table.c.status == "active")
            return [dict(row) for row in connection.execute(query).mappings()]

        return self._read("lessons", read)

    def record_llm_call(self, call: LLMCallRecord, cost_usd: Decimal | None) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.llm_calls.insert().values(
                    run_id=call.run_id if self._run_exists(connection, call.run_id) else None,
                    created_at=call.created_at,
                    agent=call.agent[:32],
                    provider=call.provider[:32],
                    model=call.model[:64],
                    status=call.status,
                    input_tokens=call.usage.input_tokens,
                    output_tokens=call.usage.output_tokens,
                    latency_ms=call.latency_ms,
                    cost_usd=cost_usd,
                    error=None if call.error is None else call.error[:2000],
                )
            )

        self._write("llm call", write)

    def llm_calls_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        table = schema.llm_calls

        def read(connection: sa.Connection) -> list[dict[str, Any]]:
            query = (
                sa.select(table)
                .where(table.c.created_at >= start, table.c.created_at < end)
                .order_by(table.c.created_at, table.c.id)
            )
            return [dict(row) for row in connection.execute(query).mappings()]

        return self._read("llm calls", read)

    def record_watch(self, run_id: str | None, report: WatchReport) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.market_watch.insert().values(
                    run_id=run_id if self._run_exists(connection, run_id) else None,
                    day=report.day,
                    phase=report.phase,
                    created_at=report.created_at,
                    report=report.model_dump(mode="json"),
                )
            )

        self._write(f"{report.phase}-market watch", write)

    def get_watch(self, day: date, phase: str) -> WatchReport | None:
        table = schema.market_watch

        def read(connection: sa.Connection) -> WatchReport | None:
            row = connection.execute(
                sa.select(table.c.report)
                .where(table.c.day == day, table.c.phase == phase)
                .order_by(table.c.created_at.desc(), table.c.id.desc())
                .limit(1)
            ).first()
            return None if row is None else WatchReport.model_validate(row[0])

        return self._read("market watch", read)

    def record_plan(self, run_id: str | None, plan: DayPlan) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.day_plans.insert().values(
                    source_run_id=run_id if self._run_exists(connection, run_id) else None,
                    source_day=plan.source_day,
                    for_day=plan.for_day,
                    created_at=plan.created_at,
                    items=[i.model_dump(mode="json") for i in plan.items],
                    summary=plan.summary,
                )
            )

        self._write(f"plan for {plan.for_day}", write)

    def plan_for(self, day: date) -> dict[str, Any] | None:
        return self._latest_plan(schema.day_plans.c.for_day == day)

    def plan_from(self, day: date) -> dict[str, Any] | None:
        return self._latest_plan(schema.day_plans.c.source_day == day)

    def _latest_plan(self, condition: Any) -> dict[str, Any] | None:
        table = schema.day_plans

        def read(connection: sa.Connection) -> dict[str, Any] | None:
            row = (
                connection.execute(
                    sa.select(table)
                    .where(condition)
                    .order_by(table.c.created_at.desc(), table.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            return None if row is None else dict(row)

        return self._read("plans", read)

    def promote_lesson(self, lesson_id: int) -> bool:
        table = schema.lessons
        changed: list[int] = []

        def write(connection: sa.Connection) -> None:
            result = connection.execute(
                table.update()
                .where(
                    table.c.id == lesson_id,
                    table.c.status == "active",
                    table.c.promoted_at.is_(None),
                )
                .values(promoted_at=self._clock())
            )
            changed.append(result.rowcount)

        self._write(f"promotion of lesson {lesson_id}", write)
        return bool(changed and changed[0])

    def record_playbook(self, playbook: Playbook) -> int:
        ids: list[int] = []

        def write(connection: sa.Connection) -> None:
            now = self._clock()
            result = connection.execute(
                schema.playbooks.insert().values(
                    name=playbook.name,
                    title=playbook.title,
                    when_to_use=playbook.when_to_use,
                    steps=list(playbook.steps),
                    category=playbook.category,
                    status=playbook.status,
                    source_lesson_ids=list(playbook.source_lesson_ids),
                    created_at=playbook.created_at,
                    updated_at=now,
                )
            )
            key = result.inserted_primary_key
            ids.append(int(key[0]) if key else 0)

        self._write(f"playbook {playbook.name}", write)
        return ids[0]

    def list_playbooks(self, *, include_retired: bool) -> list[Playbook]:
        table = schema.playbooks

        def read(connection: sa.Connection) -> list[Playbook]:
            query = sa.select(table).order_by(table.c.created_at, table.c.id)
            if not include_retired:
                query = query.where(table.c.status != "retired")
            return [
                Playbook(
                    id=row["id"],
                    name=row["name"],
                    title=row["title"],
                    when_to_use=row["when_to_use"],
                    steps=tuple(row["steps"]),
                    category=row["category"],
                    status=row["status"],
                    source_lesson_ids=tuple(row["source_lesson_ids"]),
                    created_at=row["created_at"],
                    retired_reason=row["status_reason"] if row["status"] == "retired" else None,
                )
                for row in connection.execute(query).mappings()
            ]

        return self._read("playbooks", read)

    def set_playbook_status(self, name: str, status: str, reason: str) -> bool:
        table = schema.playbooks
        changed: list[int] = []

        def write(connection: sa.Connection) -> None:
            result = connection.execute(
                table.update()
                .where(table.c.name == name, table.c.status != "retired")
                .values(status=status, status_reason=reason[:500], updated_at=self._clock())
            )
            changed.append(result.rowcount)

        self._write(f"playbook {name}", write)
        return bool(changed and changed[0])

    def record_guidebook(self, content: str, changes: Sequence[str]) -> int:
        table = schema.guidebook_versions
        versions: list[int] = []

        def write(connection: sa.Connection) -> None:
            latest = connection.execute(sa.select(sa.func.max(table.c.version))).scalar()
            version = int(latest or 0) + 1
            connection.execute(
                table.insert().values(
                    version=version,
                    created_at=self._clock(),
                    content=content,
                    changes=list(changes),
                )
            )
            versions.append(version)

        self._write("guidebook", write)
        return versions[0]

    def _guidebook_row(self, version: int | None) -> dict[str, Any] | None:
        table = schema.guidebook_versions

        def read(connection: sa.Connection) -> dict[str, Any] | None:
            query = sa.select(table)
            query = (
                query.order_by(table.c.version.desc()).limit(1)
                if version is None
                else query.where(table.c.version == version)
            )
            row = connection.execute(query).mappings().first()
            return None if row is None else dict(row)

        return self._read("guidebook", read)

    def latest_guidebook(self) -> dict[str, Any] | None:
        return self._guidebook_row(None)

    def guidebook(self, version: int) -> dict[str, Any] | None:
        return self._guidebook_row(version)

    def analysis_contexts(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        table = schema.analyses

        def read(connection: sa.Connection) -> list[dict[str, Any]]:
            query = (
                sa.select(table.c.run_id, table.c.started_at, table.c.lesson_ids, table.c.proposal)
                .where(table.c.started_at >= start, table.c.started_at < end)
                .order_by(table.c.started_at, table.c.id)
            )
            return [
                {
                    "run_id": row["run_id"],
                    "started_at": row["started_at"],
                    "lesson_ids": list(row["lesson_ids"] or []),
                    "playbook": (row["proposal"] or {}).get("playbook"),
                }
                for row in connection.execute(query).mappings()
            ]

        return self._read("analyses", read)

    def environments(self) -> set[str]:
        def read(connection: sa.Connection) -> set[str]:
            query = sa.select(schema.runs.c.environment).distinct()
            return {row[0] for row in connection.execute(query)}

        return self._read("environments", read)

    def record_critique(self, result: CriticResult) -> None:
        def write(connection: sa.Connection) -> None:
            connection.execute(
                schema.critiques.insert().values(
                    run_id=result.run_id,
                    status=result.status,
                    model_verdict=result.model_verdict,
                    summary=result.summary,
                    reasons=list(result.reasons),
                    concerns=list(result.concerns),
                    evidence_checks=[c.model_dump(mode="json") for c in result.evidence_checks],
                    model=result.model,
                    prompt_version=result.prompt_version,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    created_at=result.created_at,
                )
            )

        self._write(f"critique for run {result.run_id}", write)

    def get_critique(self, run_id: str) -> dict[str, Any] | None:
        table = schema.critiques

        def read(connection: sa.Connection) -> dict[str, Any] | None:
            row = (
                connection.execute(sa.select(table).where(table.c.run_id == run_id))
                .mappings()
                .first()
            )
            return None if row is None else dict(row)

        return self._read(f"critique of run {run_id}", read)

    def tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        table = schema.tool_calls

        def read(connection: sa.Connection) -> list[ToolCallRecord]:
            query = sa.select(table).where(table.c.run_id == run_id).order_by(table.c.seq)
            return [
                ToolCallRecord(
                    call_id=row["call_id"],
                    tool=row["tool"],
                    arguments=row["arguments"],
                    is_error=row["is_error"],
                    result=row["result"],
                )
                for row in connection.execute(query).mappings()
            ]

        return self._read(f"tool calls of run {run_id}", read)


class AuditRecorder:
    """Applies the storage policy around a store.

    Required: a failed write raises, and callers must not trade without a record.
    Optional: a failed write is logged as an error and the caller carries on.
    """

    def __init__(self, store: AuditStore, *, required: bool, logger: Any = None) -> None:
        self.store = store
        self.required = required
        self._log = logger or get_logger(__name__)

    def write(self, what: str, fn: Callable[[AuditStore], None]) -> bool:
        try:
            fn(self.store)
        except AuditError as exc:
            self._log.error("audit.write_failed", what=what, error=str(exc), required=self.required)
            if self.required:
                raise
            return False
        return True
