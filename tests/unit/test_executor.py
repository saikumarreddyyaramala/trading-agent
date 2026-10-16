from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    FakeOrders,
    FakeTime,
    healthy_market_data,
    make_position,
    make_quote,
)
from trading_agent.domain.models import OrderPlan, RiskCheck, RiskDecision
from trading_agent.execution.executor import ExecutionDisabledError, Executor
from trading_agent.settings.models import AppConfig

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
APPROVED = RiskDecision(
    approved=True,
    symbol="AAPL",
    checks=(RiskCheck(name="All", passed=True, detail=""),),
    plan=PLAN,
    evaluated_at=NOW,
)


class _Log:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __getattr__(self, level: str) -> Any:
        return lambda event, **_: self.events.append(f"{level}:{event}")


def _executor(
    orders: FakeOrders | None = None,
    *,
    broker: FakeBroker | None = None,
    market_data: Any = None,
    config: dict[str, Any] | None = None,
    time: FakeTime | None = None,
    log: _Log | None = None,
) -> tuple[Executor, FakeOrders, FakeTime]:
    orders = orders or FakeOrders()
    time = time or FakeTime()
    executor = Executor(
        orders,
        broker or FakeBroker(),
        market_data or healthy_market_data("AAPL"),
        AppConfig.model_validate(config or {}),
        clock=time.clock,
        sleep=time.sleep,
        logger=log or _Log(),
    )
    return executor, orders, time


# ---------- Guards ----------


def test_live_configuration_is_refused_outright() -> None:
    live = {"environment": "live", "broker": {"paper": False}, "storage": {"required": True}}
    with pytest.raises(ExecutionDisabledError, match="paper-only until Phase 11"):
        _executor(config=live)


@pytest.mark.parametrize(
    ("decision", "config", "run_id", "age", "reason"),
    [
        (
            RiskDecision(approved=False, symbol="AAPL", checks=(), plan=None, evaluated_at=NOW),
            {},
            "run1",
            0,
            "not an approval",
        ),
        (APPROVED, {"kill_switch": True}, "run1", 0, "kill switch is on"),
        (APPROVED, {}, "run 1; drop", 0, "run id must be"),
        (APPROVED, {}, "run1", 61, "approval is 61s old (limit 60s)"),
    ],
)
def test_refusals_place_no_order(
    decision: RiskDecision, config: dict[str, Any], run_id: str, age: int, reason: str
) -> None:
    time = FakeTime(NOW + timedelta(seconds=age))
    executor, orders, _ = _executor(config=config, time=time)
    result = executor.enter(decision, run_id)
    assert result.status == "refused"
    assert reason in result.reason
    assert orders.log == []


# ---------- Entries ----------


def test_filled_entry_gets_a_protective_stop() -> None:
    log = _Log()
    executor, orders, _ = _executor(log=log)
    result = executor.enter(APPROVED, "run1")

    assert result.status == "entered"
    assert result.filled_quantity == Decimal("0.4899")
    assert result.avg_fill_price == Decimal("102.00")
    assert orders.log == [
        ("buy", "0.4899 AAPL @ 102.05 [ta-run1-entry]"),
        ("stop", "0.4899 AAPL @ 95.09 [ta-run1-stop]"),
    ]
    assert result.stop_order is not None
    assert result.stop_order.stop_price == Decimal("95.09")
    assert "info:execution.entered" in log.events


def test_instant_fill_needs_no_polling() -> None:
    executor, _, time = _executor(FakeOrders("instant"))
    assert executor.enter(APPROVED, "run1").status == "entered"
    assert time.slept == 0


def test_retry_reuses_the_existing_order() -> None:
    executor, orders, _ = _executor(FakeOrders("instant"))
    executor.enter(APPROVED, "run1")
    second = executor.enter(APPROVED, "run1")
    assert second.status == "entered"
    assert [action for action, _ in orders.log] == ["buy", "stop"]  # nothing submitted twice


def test_rejected_entry() -> None:
    executor, _, _ = _executor(FakeOrders(reject_entry=True))
    result = executor.enter(APPROVED, "run1")
    assert result.status == "failed"
    assert "insufficient buying power" in result.reason


def test_unfilled_entry_is_cancelled_after_the_timeout() -> None:
    executor, orders, time = _executor(FakeOrders("none"))
    result = executor.enter(APPROVED, "run1")
    assert result.status == "not_filled"
    assert result.entry_order is not None
    assert result.entry_order.status == "canceled"
    assert ("cancel", "ta-run1-entry") in orders.log
    assert time.slept >= 120
    assert not any(action == "stop" for action, _ in orders.log)


def test_partial_fill_is_protected_for_the_filled_part() -> None:
    executor, orders, _ = _executor(FakeOrders("partial"))
    result = executor.enter(APPROVED, "run1")
    assert result.status == "entered"
    assert result.filled_quantity == Decimal("0.24495")
    assert ("stop", "0.24495 AAPL @ 95.09 [ta-run1-stop]") in orders.log


def test_fill_that_races_the_cancel_is_honoured() -> None:
    executor, orders, _ = _executor(FakeOrders("late"))
    result = executor.enter(APPROVED, "run1")
    assert result.status == "entered"
    assert result.filled_quantity == Decimal("0.4899")
    assert orders.log[-1][0] == "stop"


def test_poll_errors_are_tolerated() -> None:
    executor, _, _ = _executor(FakeOrders("full", fail_polls=2))
    assert executor.enter(APPROVED, "run1").status == "entered"


def test_position_is_closed_if_the_stop_cannot_be_placed() -> None:
    log = _Log()
    executor, orders, _ = _executor(FakeOrders("full", reject_stop=True), log=log)
    result = executor.enter(APPROVED, "run1")
    assert result.status == "closed_unprotected"
    assert "stop-loss failed" in result.reason
    assert orders.log[-1] == ("close", "AAPL")
    assert "error:execution.stop_failed" in log.events


# ---------- Managing open positions ----------


def _holding(**broker: Any) -> FakeBroker:
    return FakeBroker(positions=(make_position("AAPL", "0.4899", "100"),), **broker)


def _with_stop(orders: FakeOrders) -> FakeOrders:
    orders.submit_stop_sell("AAPL", Decimal("0.4899"), Decimal("95"), "ta-run1-stop")
    orders.log.clear()
    return orders


def test_flat_when_nothing_is_held() -> None:
    executor, _, _ = _executor()
    (result,) = executor.manage()
    assert result.action == "flat"


def test_hold_while_protected_and_below_target() -> None:
    executor, orders, _ = _executor(_with_stop(FakeOrders()), broker=_holding())
    (result,) = executor.manage()
    assert result.action == "hold"
    assert result.detail == "bid $100.00, target $110.00, stop $95.00"
    assert orders.log == []


def test_missing_stop_is_restored() -> None:
    executor, orders, _ = _executor(broker=_holding())
    (result,) = executor.manage()
    assert result.action == "stop_restored"
    action, detail = orders.log[0]
    assert action == "stop"
    assert detail.startswith("0.4899 AAPL @ 95.00 [ta-restore-AAPL-")


def test_unrestorable_stop_closes_the_position() -> None:
    executor, orders, _ = _executor(FakeOrders(reject_stop=True), broker=_holding())
    (result,) = executor.manage()
    assert result.action == "force_close"
    assert "could not be restored" in result.detail
    assert orders.log[-1] == ("close", "AAPL")


def test_take_profit_cancels_the_stop_before_closing() -> None:
    market_data = healthy_market_data("AAPL")
    market_data.quotes["AAPL"] = make_quote(bid="110.00", ask="110.05")
    executor, orders, _ = _executor(
        _with_stop(FakeOrders()), broker=_holding(), market_data=market_data
    )
    (result,) = executor.manage()
    assert result.action == "take_profit"
    assert orders.log == [("cancel", "ta-run1-stop"), ("close", "AAPL")]


@pytest.mark.parametrize("kill_switch", [False, True])
def test_force_close_at_the_end_of_the_day(kill_switch: bool) -> None:
    late = FakeTime(datetime(2026, 10, 15, 19, 50, tzinfo=UTC))  # 15:50 New York
    executor, orders, _ = _executor(
        _with_stop(FakeOrders()),
        broker=_holding(),
        time=late,
        config={"kill_switch": kill_switch},  # exits must work with the kill switch on
    )
    (result,) = executor.manage()
    assert result.action == "force_close"
    assert result.detail == "force close at 15:50"
    assert orders.log[-1] == ("close", "AAPL")


def test_manage_reports_errors() -> None:
    executor, _, _ = _executor(broker=FakeBroker(fail_positions=True))
    assert executor.manage()[0].action == "error"

    executor, _, _ = _executor(broker=_holding(), market_data=FakeMarketData(fail=True))
    (result,) = executor.manage()
    assert result.action == "error"
    assert "data unavailable" in result.detail


def test_failed_close_is_an_error() -> None:
    late = FakeTime(datetime(2026, 10, 15, 19, 55, tzinfo=UTC))
    executor, _, _ = _executor(FakeOrders(fail_close=True), broker=_holding(), time=late)
    (result,) = executor.manage()
    assert result.action == "error"
    assert "close failed: market closed" in result.detail


def test_cancel_that_never_settles_still_attempts_the_close() -> None:
    late = FakeTime(datetime(2026, 10, 15, 19, 55, tzinfo=UTC))
    orders = _with_stop(FakeOrders(sticky_cancel=True))
    executor, _, time = _executor(orders, broker=_holding(), time=late)
    (result,) = executor.manage()
    assert result.action == "force_close"
    assert "open orders did not cancel in time" in result.detail
    assert time.slept >= 30


# ---------- Flatten ----------


def test_flatten_cancels_everything_and_closes_everything() -> None:
    broker = FakeBroker(positions=(make_position("AAPL"), make_position("MSFT")))
    executor, orders, _ = _executor(_with_stop(FakeOrders()), broker=broker)
    result = executor.flatten()
    assert result.action == "flattened"
    assert result.detail == "closed 2 position(s)"
    assert orders.log == [("cancel", "ta-run1-stop"), ("close", "AAPL"), ("close", "MSFT")]


def test_flatten_reports_failures() -> None:
    executor, _, _ = _executor(FakeOrders(fail_close=True), broker=_holding())
    result = executor.flatten()
    assert result.action == "error"
    assert "close AAPL failed" in result.detail


def test_flatten_without_position_data() -> None:
    executor, _, _ = _executor(broker=FakeBroker(fail_positions=True))
    result = executor.flatten()
    assert result.action == "error"
    assert "positions unavailable" in result.detail
