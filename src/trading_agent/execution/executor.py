"""Order execution: turn an approved risk decision into a protected position, and get out.

Safety rules enforced here:
- Live entries need explicit authorization, granted only after every live gate passes
  (ADR 0012), and are capped at LIVE_HARD_CAP_USD whatever the config says.
- Only a fresh, approved `RiskDecision` can open a position, and only within its plan.
- The kill switch blocks entries, never exits.
- Idempotent: order ids derive from the run id, so a retry finds the existing order
  instead of submitting a second one.
- Never unprotected: a broker-side stop goes on as soon as the entry fills. If that fails,
  the position is closed. `manage()` restores a missing stop on every pass.
"""

from __future__ import annotations

import re
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime, time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.domain.models import BrokerOrder, Position, RiskDecision
from trading_agent.domain.ports import BrokerPort, MarketDataPort, OrderExecutionPort
from trading_agent.execution.live import LIVE_HARD_CAP_USD
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig

CENT = Decimal("0.01")
_RUN_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class EntryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["entered", "not_filled", "refused", "failed", "closed_unprotected"]
    reason: str
    symbol: str | None = None
    entry_order: BrokerOrder | None = None
    stop_order: BrokerOrder | None = None
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None


class ManageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal[
        "flat", "hold", "take_profit", "force_close", "stop_restored", "flattened", "error"
    ]
    detail: str
    symbol: str | None = None
    orders: tuple[BrokerOrder, ...] = ()


def _cents_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_DOWN)


class Executor:
    def __init__(
        self,
        orders: OrderExecutionPort,
        broker: BrokerPort,
        market_data: MarketDataPort,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        logger: Any = None,
        live_authorized: bool = False,
        entries_blocked: str | None = None,
    ) -> None:
        """`live_authorized` must be True for any live entry; only the composition root sets it,
        after every live gate passes (ADR 0012). Exits never depend on it."""
        if config.is_live and not live_authorized and entries_blocked is None:
            entries_blocked = "live entries require every live gate to pass (`live-check`)"
        self._entries_blocked = entries_blocked
        self._orders = orders
        self._broker = broker
        self._market_data = market_data
        self._config = config
        self._exec = config.execution
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or _time.sleep
        self._log = logger or get_logger(__name__)

    # ---------- Entry ----------

    def enter(self, decision: RiskDecision, run_id: str) -> EntryResult:
        plan = decision.plan
        refusal = self._refusal(decision, run_id)
        if refusal or plan is None:
            reason = refusal or "no order plan"
            self._log.warning("execution.refused", reason=reason, symbol=decision.symbol)
            return EntryResult(status="refused", reason=reason, symbol=decision.symbol)

        symbol = plan.symbol
        entry_id = f"ta-{run_id}-entry"
        try:
            order = self._orders.get_order(entry_id)
            if order is not None:
                self._log.warning("execution.entry_exists", client_order_id=entry_id)
            else:
                order = self._orders.submit_limit_buy(
                    symbol, plan.quantity, plan.limit_price, entry_id
                )
                self._log.info(
                    "execution.order_submitted",
                    role="entry",
                    symbol=symbol,
                    quantity=str(plan.quantity),
                    limit_price=str(plan.limit_price),
                    client_order_id=entry_id,
                )
        except Exception as exc:
            self._log.error("execution.entry_failed", symbol=symbol, error=str(exc))
            return EntryResult(status="failed", reason=f"entry order failed: {exc}", symbol=symbol)

        order = self._await_fill(order)
        filled = order.filled_quantity
        if filled <= 0:
            return EntryResult(
                status="not_filled",
                reason=f"entry order {order.status} without a fill",
                symbol=symbol,
                entry_order=order,
            )

        stop_id = f"ta-{run_id}-stop"
        try:
            stop = self._orders.get_order(stop_id) or self._orders.submit_stop_sell(
                symbol, filled, plan.stop_loss_price, stop_id
            )
        except Exception as exc:
            # Never hold a position without broker-side protection.
            self._log.error("execution.stop_failed", symbol=symbol, error=str(exc))
            close = self._exit(symbol, "stop could not be placed", "force_close")
            return EntryResult(
                status="closed_unprotected",
                reason=f"stop-loss failed ({exc}); {close.detail}",
                symbol=symbol,
                entry_order=order,
                filled_quantity=filled,
                avg_fill_price=order.filled_avg_price,
            )

        self._log.info(
            "execution.entered",
            symbol=symbol,
            filled_quantity=str(filled),
            avg_fill_price=str(order.filled_avg_price),
            stop_price=str(plan.stop_loss_price),
        )
        return EntryResult(
            status="entered",
            reason="filled and protected by a stop-loss",
            symbol=symbol,
            entry_order=order,
            stop_order=stop,
            filled_quantity=filled,
            avg_fill_price=order.filled_avg_price,
        )

    def _refusal(self, decision: RiskDecision, run_id: str) -> str | None:
        if not decision.approved or decision.plan is None:
            return "the risk decision is not an approval"
        if self._config.kill_switch:
            return "kill switch is on"
        if self._entries_blocked:
            return self._entries_blocked
        if self._config.is_live and decision.plan.notional_usd > LIVE_HARD_CAP_USD:
            return (
                f"order value ${decision.plan.notional_usd} exceeds the "
                f"${LIVE_HARD_CAP_USD} live hard cap"
            )
        if not _RUN_ID.match(run_id):
            return "run id must be 1-64 letters, digits, or dashes"
        age = (self._clock() - decision.evaluated_at).total_seconds()
        if age > self._exec.max_decision_age_seconds:
            return (
                f"the risk approval is {age:.0f}s old "
                f"(limit {self._exec.max_decision_age_seconds}s); re-run the checks"
            )
        return None

    def _poll(self, order: BrokerOrder, timeout_seconds: float) -> BrokerOrder:
        """Refresh the order until it is terminal or the timeout passes."""
        started = self._clock()
        while not order.is_terminal:
            if (self._clock() - started).total_seconds() >= timeout_seconds:
                break
            self._sleep(self._exec.poll_interval_seconds)
            try:
                order = self._orders.get_order(order.client_order_id) or order
            except Exception as exc:  # keep polling; the timeout bounds this
                self._log.warning("execution.poll_failed", order=order.id, error=str(exc))
        return order

    def _await_fill(self, order: BrokerOrder) -> BrokerOrder:
        order = self._poll(order, self._exec.entry_timeout_seconds)
        if order.is_terminal:
            return order
        try:
            self._orders.cancel_order(order.id)
        except Exception as exc:
            self._log.warning("execution.cancel_failed", order=order.id, error=str(exc))
        # Fills can still arrive while the cancel settles; the final state decides.
        return self._poll(order, self._exec.cancel_timeout_seconds)

    # ---------- Exits ----------

    def manage(self, force_close_at: time | None = None) -> tuple[ManageResult, ...]:
        """One management pass over open positions: force-close, take profit, keep a stop.

        `force_close_at` overrides the configured close time, e.g. on an early-close day.
        """
        try:
            positions = self._broker.get_positions()
        except Exception as exc:
            return (ManageResult(action="error", detail=f"positions unavailable: {exc}"),)
        if not positions:
            return (ManageResult(action="flat", detail="no open positions"),)
        close_at = force_close_at or self._config.schedule.force_close
        return tuple(self._manage_one(position, close_at) for position in positions)

    def _manage_one(self, position: Position, force_close: time) -> ManageResult:
        risk = self._config.risk
        symbol = position.symbol
        schedule = self._config.schedule
        local_now = self._clock().astimezone(ZoneInfo(schedule.timezone))
        if local_now.time() >= force_close:
            return self._exit(symbol, f"force close at {force_close:%H:%M}", "force_close")

        entry = position.avg_entry_price
        target = _cents_down(entry * (1 + risk.take_profit_pct / 100))
        stop_price = _cents_down(entry * (1 - risk.stop_loss_pct / 100))
        try:
            quote = self._market_data.get_latest_quotes([symbol]).get(symbol)
            open_orders = self._orders.get_open_orders(symbol)
        except Exception as exc:
            return ManageResult(action="error", detail=f"data unavailable: {exc}", symbol=symbol)

        if quote is not None and quote.bid_price >= target:
            return self._exit(
                symbol, f"take profit: bid ${quote.bid_price} >= ${target}", "take_profit"
            )

        if not any(o.side == "sell" for o in open_orders):
            stop_id = f"ta-restore-{symbol}-{local_now:%Y%m%d%H%M%S}"
            try:
                stop = self._orders.submit_stop_sell(symbol, position.quantity, stop_price, stop_id)
            except Exception as exc:
                self._log.error("execution.stop_restore_failed", symbol=symbol, error=str(exc))
                return self._exit(
                    symbol, f"no stop and it could not be restored ({exc})", "force_close"
                )
            self._log.warning("execution.stop_restored", symbol=symbol, stop_price=str(stop_price))
            return ManageResult(
                action="stop_restored",
                detail=f"placed missing stop at ${stop_price}",
                symbol=symbol,
                orders=(stop,),
            )

        bid = "n/a" if quote is None else f"${quote.bid_price}"
        return ManageResult(
            action="hold",
            detail=f"bid {bid}, target ${target}, stop ${stop_price}",
            symbol=symbol,
        )

    def _cancel_open(self, symbol: str | None) -> list[str]:
        """Cancel open orders and wait for them to settle. Returns problems, if any."""
        problems: list[str] = []
        try:
            for order in self._orders.get_open_orders(symbol):
                try:
                    self._orders.cancel_order(order.id)
                except Exception as exc:
                    problems.append(f"cancel {order.id} failed: {exc}")
            started = self._clock()
            while self._orders.get_open_orders(symbol):
                if (self._clock() - started).total_seconds() >= self._exec.cancel_timeout_seconds:
                    problems.append("open orders did not cancel in time")
                    break
                self._sleep(self._exec.poll_interval_seconds)
        except Exception as exc:
            problems.append(f"open orders unavailable: {exc}")
        return problems

    def _exit(
        self, symbol: str, reason: str, action: Literal["take_profit", "force_close"]
    ) -> ManageResult:
        # The stop holds the shares; cancel it first or the broker rejects the close.
        problems = self._cancel_open(symbol)
        try:
            order = self._orders.close_position(symbol)
        except Exception as exc:
            problems.append(f"close failed: {exc}")
            self._log.error("execution.exit_failed", symbol=symbol, problems=problems)
            return ManageResult(
                action="error", detail=f"{reason}; " + "; ".join(problems), symbol=symbol
            )
        self._log.info("execution.exit", symbol=symbol, reason=reason, order=order.id)
        detail = reason if not problems else f"{reason} (warnings: {'; '.join(problems)})"
        return ManageResult(action=action, detail=detail, symbol=symbol, orders=(order,))

    def flatten(self) -> ManageResult:
        """Emergency exit: cancel every open order and close every position."""
        problems = self._cancel_open(None)
        closed: list[BrokerOrder] = []
        try:
            positions = self._broker.get_positions()
        except Exception as exc:
            problems.append(f"positions unavailable: {exc}")
            positions = ()
        for position in positions:
            try:
                closed.append(self._orders.close_position(position.symbol))
            except Exception as exc:
                problems.append(f"close {position.symbol} failed: {exc}")
        self._log.warning("execution.flatten", closed=len(closed), problems=problems)
        detail = f"closed {len(closed)} position(s)"
        if problems:
            return ManageResult(
                action="error", detail=f"{detail}; " + "; ".join(problems), orders=tuple(closed)
            )
        return ManageResult(action="flattened", detail=detail, orders=tuple(closed))
