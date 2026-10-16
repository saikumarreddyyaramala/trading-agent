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
