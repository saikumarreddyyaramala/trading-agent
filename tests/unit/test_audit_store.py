from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa

from tests.conftest import NOW
from trading_agent.agents.analyst import AnalystResult, ToolCallRecord
from trading_agent.domain.models import (
    BrokerOrder,
    Evidence,
    LLMUsage,
    OrderPlan,
    RiskCheck,
    RiskDecision,
    TradeProposal,
)
from trading_agent.execution.executor import ManageResult
from trading_agent.settings.models import AppConfig
from trading_agent.storage import schema
from trading_agent.storage.store import (
    AuditError,
    AuditRecorder,
    NewLessonRow,
    NullAuditStore,
    SqlAuditStore,
    order_role,
)


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _store(engine: sa.Engine) -> SqlAuditStore:
    return SqlAuditStore(engine, AppConfig(), clock=_Clock())


PROPOSAL = TradeProposal(
    decision="buy",
    symbol="AAPL",
    thesis="Tight spread.",
    reference_price=Decimal("100.05"),
    confidence="medium",
    evidence=(
        Evidence(claim="a", source_call_id="c1"),
        Evidence(claim="b", source_call_id="c2"),
    ),
    risks=("reversal",),
)


def _analysis(run_id: str = "r1", status: str = "proposed") -> AnalystResult:
    return AnalystResult(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        reason="evidence verified",
        proposal=PROPOSAL,
        prompt_version="p2",
        turns=3,
        tool_calls=(
            ToolCallRecord(
                call_id="c1", tool="get_quote", arguments={"symbol": "AAPL"}, is_error=False
            ),
            ToolCallRecord(call_id="c2", tool="get_news", arguments=None, is_error=True),
        ),
        usage=LLMUsage(input_tokens=1000, output_tokens=200),
        started_at=NOW,
        duration_seconds=12.5,
    )


PLAN = OrderPlan(
    symbol="AAPL",
    quantity=Decimal("0.4899"),
    fractional=True,
    notional_usd=Decimal("49.03"),
    entry_reference_price=Decimal("100.10"),
    limit_price=Decimal("102.05"),
    stop_loss_price=Decimal("95.09"),
    take_profit_price=Decimal("110.11"),
)


def _decision(approved: bool = True) -> RiskDecision:
    checks = (
        RiskCheck(name="Kill switch", passed=True, detail="off"),
        RiskCheck(name="Spread", passed=approved, detail="0.1%" if approved else "too wide"),
    )
    return RiskDecision(
        approved=approved,
        symbol="AAPL",
        checks=checks,
        plan=PLAN if approved else None,
        evaluated_at=NOW,
    )


def _order(client_id: str = "ta-r1-entry", **changes: Any) -> BrokerOrder:
    values: dict[str, Any] = {
        "id": f"id-{client_id}",
        "client_order_id": client_id,
        "symbol": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "status": "new",
        "quantity": Decimal("0.4899"),
        "filled_quantity": Decimal("0"),
        "limit_price": Decimal("102.05"),
        "submitted_at": NOW,
    }
    values.update(changes)
    return BrokerOrder(**values)


# ---------- Runs ----------


def test_run_lifecycle_with_config_snapshot(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "analyze")
    store.start_run("r1", "analyze")  # a retry keeps the original record
    store.finish_run("r1", "proposed", "buy AAPL")

    (run,) = store.list_runs(10)
    assert run.run_id == "r1"
    assert run.kind == "analyze"
    assert run.status == "proposed"
    assert run.started_at.tzinfo is not None
    assert run.finished_at is not None
    assert run.finished_at > run.started_at

    report = store.get_run("r1")
    assert report is not None
    assert report["run"]["environment"] == "paper"
    assert report["run"]["config"]["risk"]["max_position_usd"] == "50"
    assert "api_key" not in str(report["run"]["config"]).lower()


def test_list_runs_is_newest_first_and_limited(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    for run_id in ("a", "b", "c"):
        store.start_run(run_id, "risk_check")
    assert [r.run_id for r in store.list_runs(2)] == ["c", "b"]


def test_unknown_run_is_none(db_engine: sa.Engine) -> None:
    assert _store(db_engine).get_run("nope") is None


# ---------- Analyses and risk decisions ----------


def test_analysis_with_tool_calls(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "analyze")
    store.record_analysis(_analysis())

    report = store.get_run("r1")
    assert report is not None
    analysis = report["analysis"]
    assert analysis["status"] == "proposed"
    assert analysis["symbol"] == "AAPL"
    assert analysis["reference_price"] == Decimal("100.05")
    assert analysis["input_tokens"] == 1000
    assert analysis["proposal"]["evidence"][1]["source_call_id"] == "c2"
    calls = report["tool_calls"]
    assert [(c["seq"], c["tool"], c["is_error"]) for c in calls] == [
        (1, "get_quote", False),
        (2, "get_news", True),
    ]
    assert calls[0]["arguments"] == {"symbol": "AAPL"}


def test_failed_analysis_without_proposal(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r2", "analyze")
    failed = _analysis("r2", "failed").model_copy(update={"proposal": None, "tool_calls": ()})
    store.record_analysis(failed)
    report = store.get_run("r2")
    assert report is not None
    assert report["analysis"]["decision"] is None
    assert report["tool_calls"] == []


def test_one_analysis_per_run(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "analyze")
    store.record_analysis(_analysis())
    with pytest.raises(AuditError, match="could not record analysis for run r1"):
        store.record_analysis(_analysis())


@pytest.mark.parametrize("approved", [True, False])
def test_risk_decision(db_engine: sa.Engine, approved: bool) -> None:
    store = _store(db_engine)
    store.start_run("r1", "manual_buy")
    store.record_risk_decision("r1", _decision(approved))
    report = store.get_run("r1")
    assert report is not None
    (decision,) = report["risk_decisions"]
    assert decision["approved"] is approved
    assert decision["failed_checks"] == ([] if approved else ["Spread"])
    assert len(decision["checks"]) == 2
    assert (decision["plan"] is not None) is approved
    if approved:
        assert decision["plan"]["limit_price"] == "102.05"


def test_writes_for_unknown_runs_fail(db_engine: sa.Engine) -> None:
    with pytest.raises(AuditError):
        _store(db_engine).record_risk_decision("never-started", _decision())


# ---------- Orders and events ----------


def test_orders_are_linked_and_updated_in_place(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "manual_buy")
    store.record_orders("r1", [_order(), _order("ta-r1-stop", side="sell", order_type="stop")])
    filled = _order(
        status="filled", filled_quantity=Decimal("0.4899"), filled_avg_price=Decimal("102")
    )
    store.record_orders(None, [filled])  # a later sync, without a run id

    report = store.get_run("r1")
    assert report is not None
    orders = {o["role"]: o for o in report["orders"]}
    entry, stop = orders["entry"], orders["stop"]
    assert (entry["role"], entry["status"], entry["filled_avg_price"]) == (
        "entry",
        "filled",
        Decimal("102"),
    )
    assert entry["filled_quantity"] == Decimal("0.4899")
    assert entry["updated_at"] > entry["recorded_at"]
    assert (stop["role"], stop["run_id"]) == ("stop", "r1")


def test_synced_orders_find_their_run_from_the_client_id(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r9", "daily")
    store.record_orders(None, [_order("ta-r9-entry"), _order("ta-gone-entry")])
    with db_engine.connect() as connection:
        rows = dict(
            connection.execute(
                sa.select(schema.orders.c.client_order_id, schema.orders.c.run_id)
            ).all()
        )
    assert rows == {"ta-r9-entry": "r9", "ta-gone-entry": None}


def test_empty_order_batch_is_a_no_op(db_engine: sa.Engine) -> None:
    _store(db_engine).record_orders("r1", [])


def test_position_event_records_its_orders(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    close = _order("close-1", side="sell", order_type="market", status="filled")
    store.record_position_event(
        None,
        ManageResult(
            action="force_close", detail="force close at 15:50", symbol="AAPL", orders=(close,)
        ),
    )
    with db_engine.connect() as connection:
        event = connection.execute(sa.select(schema.position_events)).mappings().one()
        order = connection.execute(sa.select(schema.orders)).mappings().one()
    assert (event["action"], event["symbol"], event["run_id"]) == ("force_close", "AAPL", None)
    assert (order["role"], order["run_id"]) == ("exit", None)


@pytest.mark.parametrize(
    ("client_id", "side", "role", "run_id"),
    [
        ("ta-r1-entry", "buy", "entry", "r1"),
        ("ta-manual-abc123-stop", "sell", "stop", "manual-abc123"),
        ("ta-restore-AAPL-20261015195000", "sell", "restore_stop", None),
        ("random-close", "sell", "exit", None),
        ("someone-else", "buy", "external", None),
    ],
)
def test_order_role(client_id: str, side: str, role: str, run_id: str | None) -> None:
    assert order_role(_order(client_id, side=side)) == (role, run_id)


# ---------- Failure handling and policy ----------


def test_unmigrated_database_raises_audit_errors(tmp_path: Any) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    store = _store(engine)
    with pytest.raises(AuditError, match="could not record run r1"):
        store.start_run("r1", "analyze")
    with pytest.raises(AuditError, match="could not read runs"):
        store.list_runs(5)


def test_null_store_records_nothing() -> None:
    store = NullAuditStore()
    store.start_run("r1", "analyze")
    store.finish_run("r1", "ok", "")
    store.record_analysis(_analysis())
    store.record_risk_decision("r1", _decision())
    store.record_orders("r1", [_order()])
    store.record_position_event(None, ManageResult(action="flat", detail=""))
    assert store.list_runs(5) == []
    assert store.get_run("r1") is None


class _Log:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []

    def error(self, event: str, **kw: Any) -> None:
        self.errors.append(kw)


def _failing(_: Any) -> None:
    raise AuditError("database down")


def test_required_recorder_raises() -> None:
    log = _Log()
    recorder = AuditRecorder(NullAuditStore(), required=True, logger=log)
    with pytest.raises(AuditError, match="database down"):
        recorder.write("risk decision", _failing)
    assert log.errors[0]["required"] is True


def test_optional_recorder_logs_and_carries_on() -> None:
    log = _Log()
    recorder = AuditRecorder(NullAuditStore(), required=False, logger=log)
    assert recorder.write("risk decision", _failing) is False
    assert recorder.write("risk decision", lambda store: None) is True
    assert log.errors[0]["what"] == "risk decision"


def test_naive_timestamps_are_read_as_utc(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "analyze")
    assert store.list_runs(1)[0].started_at.utcoffset() == UTC.utcoffset(None)


def test_orders_in_one_batch_come_back_in_a_stable_order(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "manual_buy")
    store.record_orders("r1", [_order("ta-r1-stop", side="sell"), _order("ta-r1-entry")])
    report = store.get_run("r1")
    assert report is not None
    assert [o["client_order_id"] for o in report["orders"]] == ["ta-r1-entry", "ta-r1-stop"]


def test_prices_read_back_without_exponents(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "manual_buy")
    store.record_orders("r1", [_order(filled_avg_price=Decimal("102.000000000"))])
    report = store.get_run("r1")
    assert report is not None
    assert str(report["orders"][0]["filled_avg_price"]) == "102"


# ---------- Report queries and timestamps ----------


def test_orders_between_is_a_half_open_range(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    at = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)
    store.record_orders(
        "r1",
        [
            _order("ta-r1-entry", submitted_at=at),
            _order("late", side="sell", submitted_at=at + timedelta(hours=10)),
        ],
    )
    rows = store.orders_between(at, at + timedelta(hours=10))
    assert [r["client_order_id"] for r in rows] == ["ta-r1-entry"]
    assert rows[0]["submitted_at"] == at


def test_analysis_totals_by_run_prefix(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    for run_id in ("daily-1", "daily-2", "manual-1"):
        store.start_run(run_id, "x")
        store.record_analysis(_analysis(run_id).model_copy(update={"tool_calls": ()}))
    assert store.analysis_totals("daily-") == {
        "analyses": 2,
        "critiques": 0,
        "input_tokens": 2000,
        "output_tokens": 400,
    }
    assert store.analysis_totals("none-")["analyses"] == 0


def test_timestamps_are_stored_in_utc_whatever_zone_they_come_in(db_engine: sa.Engine) -> None:
    from zoneinfo import ZoneInfo

    store = _store(db_engine)
    store.start_run("r1", "daily")
    local = datetime(2026, 10, 15, 8, 30, tzinfo=ZoneInfo("America/New_York"))
    store.record_step("r1", "preflight", "ok", "", local)
    report = store.get_run("r1")
    assert report is not None
    started = report["steps"][0]["started_at"]
    assert started == local
    assert started.tzinfo is not None
    assert started.hour == 12  # 08:30 New York is 12:30 UTC


def test_naive_timestamps_are_refused(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    with pytest.raises(AuditError, match="timezone-aware"):
        store.record_step("r1", "preflight", "ok", "", datetime(2026, 10, 15, 8, 30))


# ---------- Critiques and tool outputs ----------


def _critique_result(run_id: str = "r1") -> Any:
    from trading_agent.agents.critic import CriticResult, EvidenceCheck

    return CriticResult(
        run_id=run_id,
        status="vetoed",
        model_verdict="approve",
        summary="Claim c2 has no support.",
        reasons=("evidence 'c2' is unsupported: no news in the output",),
        concerns=("thin volume",),
        evidence_checks=(
            EvidenceCheck(source_call_id="c1", supported=True, note="ok"),
            EvidenceCheck(source_call_id="c2", supported=False, note="no news in the output"),
        ),
        model="judge",
        prompt_version="critic-p",
        usage=LLMUsage(input_tokens=700, output_tokens=80),
        created_at=NOW,
    )


def test_critique_round_trip(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    store.record_critique(_critique_result())
    row = store.get_critique("r1")
    assert row is not None
    assert (row["status"], row["model_verdict"], row["model"]) == ("vetoed", "approve", "judge")
    assert row["evidence_checks"][1] == {
        "source_call_id": "c2",
        "supported": False,
        "note": "no news in the output",
    }
    assert row["reasons"] == ["evidence 'c2' is unsupported: no news in the output"]
    report = store.get_run("r1")
    assert report is not None
    assert report["critique"]["status"] == "vetoed"
    assert store.get_critique("other") is None
    totals = store.analysis_totals("r")
    assert (totals["critiques"], totals["input_tokens"]) == (1, 700)


def test_one_critique_per_run(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    store.record_critique(_critique_result())
    with pytest.raises(AuditError):
        store.record_critique(_critique_result())


def test_tool_outputs_are_stored_and_read_back(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "analyze")
    with_outputs = _analysis().model_copy(
        update={
            "tool_calls": (
                ToolCallRecord(
                    call_id="c1",
                    tool="get_quote",
                    arguments={"symbol": "AAPL"},
                    is_error=False,
                    result='{"ask": "100.1"}',
                ),
            )
        }
    )
    store.record_analysis(with_outputs)
    (call,) = store.tool_calls("r1")
    assert call.result == '{"ask": "100.1"}'
    assert call.arguments == {"symbol": "AAPL"}
    assert store.tool_calls("none") == []


def test_null_store_has_no_critiques() -> None:
    store = NullAuditStore()
    store.record_critique(_critique_result())
    assert store.get_critique("r1") is None
    assert store.tool_calls("r1") == []
    assert store.analysis_totals("x")["critiques"] == 0


# ---------- Lessons (self-learning) ----------


def _lesson(text: str, category: str = "research") -> NewLessonRow:
    return NewLessonRow(text=text, category=category, evidence={"run_id": "r1"})


def test_lessons_round_trip_with_their_run(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    store.record_analysis(_analysis())
    expires = NOW + timedelta(days=60)
    ids = store.record_lessons(
        "r1", [_lesson("Distrust single-source news.")], expires_at=expires, max_active=5
    )
    [lesson] = store.active_lessons(NOW, 10)
    assert lesson.id == ids[0]
    assert lesson.text == "Distrust single-source news."
    assert lesson.category == "research"
    assert lesson.source_run_id == "r1"
    assert lesson.expires_at == expires
    [row] = store.list_lessons(include_retired=False)
    assert row["evidence"] == {"run_id": "r1"}
    assert row["status"] == "active"


def test_lessons_from_an_unknown_run_are_kept_without_it(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.record_lessons(
        "nope", [_lesson("Check volume first.")], expires_at=NOW + timedelta(days=5), max_active=5
    )
    assert store.active_lessons(NOW, 10)[0].source_run_id is None


def test_active_lessons_are_newest_first_limited_and_unexpired(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    later = NOW + timedelta(days=60)
    store.record_lessons(None, [_lesson("Old lesson one.")], expires_at=later, max_active=10)
    store.record_lessons(None, [_lesson("New lesson two.")], expires_at=later, max_active=10)
    store.record_lessons(
        None, [_lesson("Expiring lesson.")], expires_at=NOW + timedelta(days=1), max_active=10
    )
    assert [x.text for x in store.active_lessons(NOW, 2)] == ["Expiring lesson.", "New lesson two."]
    future = [x.text for x in store.active_lessons(NOW + timedelta(days=2), 10)]
    assert future == ["New lesson two.", "Old lesson one."]


def test_capacity_retires_the_oldest_lessons(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    later = NOW + timedelta(days=60)
    store.record_lessons(
        None, [_lesson("First lesson."), _lesson("Second lesson.")], expires_at=later, max_active=3
    )
    store.record_lessons(
        None, [_lesson("Third lesson."), _lesson("Fourth lesson.")], expires_at=later, max_active=3
    )
    assert [x.text for x in store.active_lessons(NOW, 10)] == [
        "Fourth lesson.",
        "Third lesson.",
        "Second lesson.",
    ]
    retired = [x for x in store.list_lessons(include_retired=True) if x["status"] == "retired"]
    assert [x["text"] for x in retired] == ["First lesson."]
    assert retired[0]["retired_reason"] == "capacity: replaced by newer lessons"
    assert retired[0]["retired_at"] is not None


def test_retire_lesson_only_once(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    [lesson_id] = store.record_lessons(
        None, [_lesson("Check volume first.")], expires_at=NOW + timedelta(days=5), max_active=5
    )
    assert store.retire_lesson(lesson_id, "contradicted")
    assert not store.retire_lesson(lesson_id, "again")
    assert not store.retire_lesson(9999, "missing")
    assert store.active_lessons(NOW, 10) == []
    assert store.list_lessons(include_retired=False) == []
    [row] = store.list_lessons(include_retired=True)
    assert row["retired_reason"] == "contradicted"


def test_analysis_records_the_lessons_it_used(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    store.record_analysis(_analysis().model_copy(update={"lesson_ids": (3, 5)}))
    run = store.get_run("r1")
    assert run is not None
    assert run["analysis"]["lesson_ids"] == [3, 5]


def test_lesson_reads_and_writes_raise_audit_errors_when_unmigrated(tmp_path: Any) -> None:
    store = _store(sa.create_engine(f"sqlite:///{tmp_path / 'empty.db'}"))
    with pytest.raises(AuditError, match="could not read lessons"):
        store.active_lessons(NOW, 5)
    with pytest.raises(AuditError, match="could not record 1 lesson"):
        store.record_lessons(None, [_lesson("Check volume first.")], expires_at=NOW, max_active=5)


def test_null_store_has_no_lessons() -> None:
    store = NullAuditStore()
    assert (
        store.record_lessons(None, [_lesson("Check volume first.")], expires_at=NOW, max_active=5)
        == []
    )
    assert store.active_lessons(NOW, 5) == []
    assert not store.retire_lesson(1, "x")
    assert store.list_lessons(include_retired=True) == []


# ---------- LLM cost ledger (ADR 0014) ----------


def _llm_call(run_id: str | None = "r1", at: datetime = NOW, **changes: Any) -> Any:
    from trading_agent.domain.models import LLMCallRecord

    values: dict[str, Any] = {
        "created_at": at,
        "provider": "openai",
        "model": "gpt-6.1-sol",
        "agent": "analyst",
        "run_id": run_id,
        "status": "ok",
        "usage": LLMUsage(input_tokens=1200, output_tokens=300),
        "latency_ms": 850.5,
    }
    values.update(changes)
    return LLMCallRecord(**values)


def test_llm_calls_round_trip_with_exact_costs(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.start_run("r1", "daily")
    store.record_llm_call(_llm_call(), Decimal("0.006000"))
    store.record_llm_call(_llm_call(run_id="unknown-run", model="mystery"), None)
    store.record_llm_call(
        _llm_call(status="error", usage=LLMUsage(), latency_ms=None, error="timeout"),
        Decimal("0"),
    )
    rows = store.llm_calls_between(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    assert [(r["run_id"], r["model"], r["status"]) for r in rows] == [
        ("r1", "gpt-6.1-sol", "ok"),
        (None, "mystery", "ok"),  # unknown runs are kept without the link
        ("r1", "gpt-6.1-sol", "error"),
    ]
    assert rows[0]["cost_usd"] == Decimal("0.006")
    assert rows[0]["input_tokens"] == 1200
    assert rows[0]["latency_ms"] == 850.5
    assert rows[1]["cost_usd"] is None
    assert rows[2]["error"] == "timeout"


def test_llm_calls_between_is_half_open(db_engine: sa.Engine) -> None:
    store = _store(db_engine)
    store.record_llm_call(_llm_call(run_id=None, at=NOW), Decimal("1"))
    assert len(store.llm_calls_between(NOW, NOW + timedelta(seconds=1))) == 1
    assert store.llm_calls_between(NOW - timedelta(seconds=1), NOW) == []


def test_many_llm_calls_sum_exactly(db_engine: sa.Engine) -> None:
    # A stress test: hundreds of tiny costs must add up to the cent, with no float drift.
    store = _store(db_engine)
    for i in range(300):
        store.record_llm_call(
            _llm_call(run_id=None, at=NOW + timedelta(seconds=i)), Decimal("0.000123")
        )
    rows = store.llm_calls_between(NOW, NOW + timedelta(hours=1))
    assert len(rows) == 300
    assert sum(r["cost_usd"] for r in rows) == Decimal("0.036900")


def test_llm_ledger_on_unmigrated_database_raises(tmp_path: Any) -> None:
    store = _store(sa.create_engine(f"sqlite:///{tmp_path / 'empty.db'}"))
    with pytest.raises(AuditError, match="could not record llm call"):
        store.record_llm_call(_llm_call(), None)
    with pytest.raises(AuditError, match="could not read llm calls"):
        store.llm_calls_between(NOW, NOW)


def test_null_store_has_no_llm_calls() -> None:
    store = NullAuditStore()
    store.record_llm_call(_llm_call(), None)
    assert store.llm_calls_between(NOW, NOW + timedelta(days=1)) == []
