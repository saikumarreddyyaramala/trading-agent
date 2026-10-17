from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    healthy_market_data,
    make_asset,
    make_bars,
    make_quote,
)
from trading_agent.domain.models import (
    Evidence,
    OrderPlan,
    Position,
    RiskCheck,
    RiskDecision,
    TradeProposal,
)
from trading_agent.risk.engine import RiskEngine
from trading_agent.settings.models import AppConfig

CHECK_NAMES = [
    "Kill switch",
    "Trading mode",
    "Proposal",
    "Market open",
    "Trade window",
    "Trades today",
    "Open positions",
    "Account",
    "Asset",
    "Price",
    "Quote",
    "Spread",
    "Price vs proposal",
    "Liquidity",
    "Position size",
]


def _proposal(symbol: str | None = "AAPL", price: str | None = "100.05") -> TradeProposal:
    return TradeProposal(
        decision="buy",
        symbol=symbol,
        thesis="Tight spread, steady uptrend.",
        reference_price=None if price is None else Decimal(price),
        confidence="medium",
        evidence=(
            Evidence(claim="a", source_call_id="c1"),
            Evidence(claim="b", source_call_id="c2"),
        ),
        risks=("reversal",),
    )


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))


def _evaluate(
    proposal: TradeProposal | None = None,
    *,
    broker: FakeBroker | None = None,
    market_data: FakeMarketData | None = None,
    config: dict[str, Any] | None = None,
    live_confirmed: bool = False,
    now: datetime = NOW,
    log: _Log | None = None,
) -> RiskDecision:
    engine = RiskEngine(
        broker or FakeBroker(),
        market_data or healthy_market_data("AAPL"),
        AppConfig.model_validate(config or {}),
        live_trading_confirmed=live_confirmed,
        clock=lambda: now,
        logger=log or _Log(),
    )
    return engine.evaluate(proposal or _proposal())


def _check(decision: RiskDecision, name: str) -> RiskCheck:
    return next(c for c in decision.checks if c.name == name)


def _failed(decision: RiskDecision) -> list[str]:
    return [c.name for c in decision.failed_checks]


# ---------- Approval ----------


def test_clean_proposal_is_approved_and_sized() -> None:
    log = _Log()
    decision = _evaluate(log=log)

    assert decision.approved
    assert [c.name for c in decision.checks] == CHECK_NAMES
    assert decision.failed_checks == ()
    assert decision.evaluated_at == NOW
    assert decision.plan == OrderPlan(
        symbol="AAPL",
        quantity=Decimal("0.4899"),  # $50 / $102.05 limit, rounded down
        fractional=True,
        notional_usd=Decimal("49.03"),  # at the $100.10 ask
        entry_reference_price=Decimal("100.10"),
        limit_price=Decimal("102.05"),  # $100.05 + 2% allowed deviation
        stop_loss_price=Decimal("95.09"),
        take_profit_price=Decimal("110.11"),
    )
    event, fields = log.events[-1]
    assert event == "risk.decision"
    assert fields == {"approved": True, "symbol": "AAPL", "failed": [], "notional_usd": "49.03"}


def test_trades_are_counted_from_midnight_market_time() -> None:
    broker = FakeBroker()
    _evaluate(broker=broker)
    expected = datetime(2026, 10, 15, tzinfo=ZoneInfo("America/New_York"))
    assert broker.orders_since == [expected]


def test_details_explain_each_passed_check() -> None:
    decision = _evaluate()
    assert _check(decision, "Trade window").detail == "10:00 within 09:45-15:30"
    assert _check(decision, "Liquidity").detail == "2,000,000 shares/day over 3 days"
    assert _check(decision, "Price vs proposal").detail == "0.05% from $100.05"


# ---------- Rejections, one rule at a time ----------


@pytest.mark.parametrize(
    ("kwargs", "failed", "detail"),
    [
        ({"config": {"kill_switch": True}}, "Kill switch", "kill switch is on"),
        ({"broker": FakeBroker(market_open=False)}, "Market open", "the market is closed"),
        (
            {"now": datetime(2026, 10, 15, 13, 30, tzinfo=UTC)},
            "Trade window",
            "09:30 is outside the 09:45-15:30 trade window",
        ),
        (
            {"now": datetime(2026, 10, 15, 19, 45, tzinfo=UTC)},
            "Trade window",
            "15:45 is outside",
        ),
        ({"broker": FakeBroker(orders_today=1)}, "Trades today", "1 buy order(s) today"),
        (
            {
                "broker": FakeBroker(
                    positions=(
                        Position(
                            symbol="MSFT",
                            quantity=Decimal("1"),
                            market_value=Decimal("400"),
                            avg_entry_price=Decimal("399"),
                        ),
                    )
                )
            },
            "Open positions",
            "already holding MSFT",
        ),
        ({"broker": FakeBroker(trading_blocked=True)}, "Account", "account is blocked"),
        ({"broker": FakeBroker(is_paper=False)}, "Account", "account type does not match"),
        ({"broker": FakeBroker(equity="69.99")}, "Account", "below the $70 floor"),
    ],
)
def test_account_and_schedule_rules(kwargs: dict[str, Any], failed: str, detail: str) -> None:
    decision = _evaluate(**kwargs)
    assert not decision.approved
    assert decision.plan is None
    assert failed in _failed(decision)
    assert detail in _check(decision, failed).detail
    assert _check(decision, "Position size").detail == "not sized: earlier checks failed"


def _data(**changes: Any) -> FakeMarketData:
    market_data = healthy_market_data("AAPL")
    for key, value in changes.items():
        getattr(market_data, key)["AAPL"] = value
    return market_data


@pytest.mark.parametrize(
    ("market_data", "failed", "detail"),
    [
        (
            _data(assets=make_asset("AAPL", name="Some 3x Leveraged ETF", exchange="ARCA")),
            "Asset",
            "listed on ARCA",
        ),
        (_data(assets=make_asset("AAPL", tradable=False)), "Asset", "not tradable"),
        (_data(quotes=make_quote(bid="3.00", ask="3.01")), "Price", "below the $5 minimum"),
        (_data(quotes=make_quote(age_seconds=301)), "Quote", "stale quote"),
        (_data(quotes=make_quote(bid="99", ask="100.1")), "Spread", "exceeds 0.5%"),
        (
            _data(quotes=make_quote(bid="104", ask="104.05")),
            "Price vs proposal",
            "is 4.00% from the proposal's $100.05",
        ),
        (_data(bars=make_bars(volume=999_999)), "Liquidity", "below the 1,000,000 minimum"),
    ],
)
def test_instrument_rules(market_data: FakeMarketData, failed: str, detail: str) -> None:
    decision = _evaluate(market_data=market_data)
    assert not decision.approved
    assert failed in _failed(decision)
    assert detail in _check(decision, failed).detail


def test_live_trading_needs_confirmation() -> None:
    live = {"environment": "live", "broker": {"paper": False}, "storage": {"required": True}}
    rejected = _evaluate(broker=FakeBroker(is_paper=False), config=live)
    approved = _evaluate(broker=FakeBroker(is_paper=False), config=live, live_confirmed=True)
    assert "Trading mode" in _failed(rejected)
    assert approved.approved
    assert _check(approved, "Trading mode").detail == "live"


@pytest.mark.parametrize(
    ("proposal", "detail"),
    [
        (
            TradeProposal(decision="no_trade", thesis="Quiet.", confidence="low"),
            "the proposal is not a buy",
        ),
        (_proposal(symbol="NOT VALID"), "invalid symbol"),
        (
            TradeProposal.model_construct(
                decision="buy", symbol=None, reference_price=None, evidence=(), risks=()
            ),
            "a buy needs a symbol and a reference price",
        ),
    ],
)
def test_invalid_proposals_never_reach_market_data(proposal: TradeProposal, detail: str) -> None:
    market_data = healthy_market_data("AAPL")
    decision = _evaluate(proposal, market_data=market_data)
    assert not decision.approved
    assert decision.symbol is None
    assert detail in _check(decision, "Proposal").detail
    assert _check(decision, "Asset").detail == "no valid symbol to check"
    assert market_data.calls == []
    # Account-level checks are still recorded for the audit trail.
    assert _check(decision, "Account").passed


# ---------- Fail safe on missing data ----------


@pytest.mark.parametrize(
    ("broker", "failed", "detail"),
    [
        (FakeBroker(fail_clock=True), "Market open", "market clock unavailable"),
        (FakeBroker(fail_account=True), "Account", "account unavailable"),
        (FakeBroker(fail_positions=True), "Open positions", "positions unavailable"),
        (FakeBroker(fail_orders=True), "Trades today", "today's orders unavailable"),
    ],
)
def test_broker_failures_reject(broker: FakeBroker, failed: str, detail: str) -> None:
    decision = _evaluate(broker=broker)
    assert not decision.approved
    assert detail in _check(decision, failed).detail


def test_market_data_outage_rejects() -> None:
    decision = _evaluate(market_data=FakeMarketData(fail=True))
    assert not decision.approved
    assert "quote unavailable: could not fetch quotes: down" in _check(decision, "Quote").detail
    assert "asset unavailable" in _check(decision, "Asset").detail


def test_missing_quote_and_bars_reject() -> None:
    market_data = healthy_market_data("AAPL")
    market_data.quotes.clear()
    market_data.bars.clear()
    decision = _evaluate(market_data=market_data)
    for name in ("Price", "Quote", "Spread", "Price vs proposal"):
        assert _check(decision, name).detail == "no quote available for AAPL"
    assert _check(decision, "Liquidity").detail == "no daily bars for AAPL"


# ---------- Sizing ----------


def test_whole_shares_when_not_fractionable() -> None:
    market_data = _data(
        assets=make_asset("AAPL", fractionable=False), quotes=make_quote(bid="19.99", ask="20")
    )
    decision = _evaluate(_proposal(price="20"), market_data=market_data)
    assert decision.plan is not None
    assert decision.plan.quantity == Decimal("2")  # $50 / $20.40 limit = 2 whole shares
    assert decision.plan.notional_usd == Decimal("40.00")
    assert decision.plan.fractional is False


def test_expensive_non_fractionable_stock_cannot_be_sized() -> None:
    market_data = _data(assets=make_asset("AAPL", fractionable=False))
    decision = _evaluate(market_data=market_data)
    assert not decision.approved
    assert "$50 buys no whole share of AAPL at $102.05" in _check(decision, "Position size").detail


def test_cash_caps_the_position_never_margin() -> None:
    decision = _evaluate(broker=FakeBroker(equity="100", cash="30"))
    assert decision.plan is not None
    assert decision.plan.quantity == Decimal("0.2939")  # $30 / $102.05
    assert decision.plan.notional_usd < Decimal("30")


def test_stop_loss_must_not_breach_the_equity_floor() -> None:
    decision = _evaluate(broker=FakeBroker(equity="72", cash="72"))
    assert not decision.approved
    assert "breach the $70 equity floor" in _check(decision, "Position size").detail


def test_tiny_orders_are_rejected() -> None:
    decision = _evaluate(broker=FakeBroker(equity="100", cash="0.50"))
    assert "below the $1 minimum" in _check(decision, "Position size").detail


def test_limit_uses_the_lower_of_ask_and_reference() -> None:
    # A stale-high reference price must not loosen the limit.
    decision = _evaluate(_proposal(price="101.9"))
    assert decision.plan is not None
    assert decision.plan.limit_price == Decimal("102.10")  # $100.10 ask + 2%, not $101.90 + 2%


# ---------- The decision record ----------


def test_approval_without_plan_cannot_be_built() -> None:
    passed = RiskCheck(name="x", passed=True, detail="")
    with pytest.raises(ValueError, match="approved decision needs a plan"):
        RiskDecision(approved=True, symbol="AAPL", checks=(passed,), plan=None, evaluated_at=NOW)


def test_approval_with_a_failed_check_cannot_be_built() -> None:
    plan = _evaluate().plan
    failed = RiskCheck(name="x", passed=False, detail="no")
    with pytest.raises(ValueError, match="only passed checks"):
        RiskDecision(approved=True, symbol="AAPL", checks=(failed,), plan=plan, evaluated_at=NOW)


def test_rejection_cannot_carry_a_plan() -> None:
    plan = _evaluate().plan
    failed = RiskCheck(name="x", passed=False, detail="no")
    with pytest.raises(ValueError, match="cannot carry an order plan"):
        RiskDecision(approved=False, symbol="AAPL", checks=(failed,), plan=plan, evaluated_at=NOW)


def test_sizing_uses_the_most_conservative_money_figure() -> None:
    # A cash account with $100 cash but only $20 settled: size from the $20.
    broker = FakeBroker(equity="100", cash="100")
    original = broker.get_account

    def settled_only() -> Any:
        return original().model_copy(update={"non_marginable_buying_power": Decimal("20")})

    broker.get_account = settled_only  # type: ignore[method-assign]
    decision = _evaluate(broker=broker)
    assert decision.plan is not None
    assert decision.plan.quantity == Decimal("0.1959")  # $20 / $102.05
