"""The risk engine: code, not AI, decides whether a proposed trade happens and how big it is.

Principles:
- Trust nothing from the analyst except the symbol and its reference price. Account,
  positions, asset, quote, and volume are fetched fresh here.
- Fail safe: a data error fails the checks that needed that data, so the trade is rejected.
- Every check runs and is recorded, pass or fail, so a decision can be audited afterwards.
- Approval and an order plan cannot be separated (see `RiskDecision`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from decimal import ROUND_DOWN, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from trading_agent.data.quality import normalize_symbol, quote_issues
from trading_agent.domain.models import (
    AccountSnapshot,
    Asset,
    Bar,
    MarketClock,
    OrderPlan,
    Position,
    Quote,
    RiskCheck,
    RiskDecision,
    TradeProposal,
)
from trading_agent.domain.ports import BrokerPort, MarketDataPort
from trading_agent.observability.logging import get_logger
from trading_agent.risk.eligibility import asset_issues, liquidity_issues, price_issues
from trading_agent.settings.models import AppConfig

CENT = Decimal("0.01")
FRACTIONAL_STEP = Decimal("0.0001")
MIN_NOTIONAL_USD = Decimal("1")  # Alpaca's minimum for fractional orders


@dataclass
class _Checks:
    items: list[RiskCheck] = field(default_factory=list)

    def add(self, name: str, issues: list[str], ok_detail: str) -> bool:
        passed = not issues
        self.items.append(
            RiskCheck(name=name, passed=passed, detail=ok_detail if passed else "; ".join(issues))
        )
        return passed


def _cents_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_DOWN)


class RiskEngine:
    def __init__(
        self,
        broker: BrokerPort,
        market_data: MarketDataPort,
        config: AppConfig,
        *,
        live_trading_confirmed: bool,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._broker = broker
        self._market_data = market_data
        self._config = config
        self._risk = config.risk
        self._live_confirmed = live_trading_confirmed
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def _fetch[T](self, what: str, fn: Callable[[], T], errors: dict[str, str]) -> T | None:
        try:
            return fn()
        except Exception as exc:  # any failure to get data means the dependent checks fail
            errors[what] = f"{what} unavailable: {exc}"
            return None

    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        now = self._clock()
        checks = _Checks()
        errors: dict[str, str] = {}
        risk = self._risk

        # ---- Global switches ----
        checks.add(
            "Kill switch",
            ["kill switch is on: no new trades"] if self._config.kill_switch else [],
            "off",
        )
        live_issue = (
            ["live config but LIVE_TRADING_CONFIRMED is not true"]
            if self._config.is_live and not self._live_confirmed
            else []
        )
        checks.add("Trading mode", live_issue, "live" if self._config.is_live else "paper")

        # ---- Proposal ----
        symbol: str | None = None
        proposal_issues: list[str] = []
        if proposal.decision != "buy":
            proposal_issues.append("the proposal is not a buy")
        elif proposal.symbol is None or proposal.reference_price is None:
            proposal_issues.append("a buy needs a symbol and a reference price")
        else:
            try:
                symbol = normalize_symbol(proposal.symbol)
            except ValueError as exc:
                proposal_issues.append(str(exc))
        checks.add("Proposal", proposal_issues, f"buy {symbol} near ${proposal.reference_price}")

        # ---- Fresh data (fetched even for a bad proposal, so account checks are recorded) ----
        clock = self._fetch("market clock", self._broker.get_clock, errors)
        account = self._fetch("account", self._broker.get_account, errors)
        positions = self._fetch("positions", self._broker.get_positions, errors)
        day_start = self._day_start(now)
        orders_today = self._fetch(
            "today's orders", lambda: self._broker.count_buy_orders_since(day_start), errors
        )
        asset = quote = None
        bars: tuple[Bar, ...] = ()
        if symbol is not None:
            asset = self._fetch("asset", lambda: self._market_data.get_asset(symbol), errors)
            quote = self._fetch(
                "quote", lambda: self._market_data.get_latest_quotes([symbol]).get(symbol), errors
            )
            bars = (
                self._fetch(
                    "daily bars",
                    lambda: self._market_data.get_daily_bars(
                        [symbol], days=risk.volume_lookback_days
                    ).get(symbol, ()),
                    errors,
                )
                or ()
            )

        # ---- Market and schedule ----
        self._check_market(checks, clock, now, errors)

        # ---- Account state ----
        self._check_trades_today(checks, orders_today, errors)
        self._check_positions(checks, positions, errors)
        self._check_account(checks, account, errors)

        # ---- The instrument ----
        if symbol is None:
            for name in ("Asset", "Price", "Liquidity", "Quote", "Spread", "Price vs proposal"):
                checks.add(name, ["no valid symbol to check"], "")
        else:
            self._check_instrument(checks, symbol, proposal, asset, quote, bars, now, errors)

        # ---- Sizing (only meaningful when everything else passed) ----
        plan: OrderPlan | None = None
        if (
            symbol is not None
            and account is not None
            and asset is not None
            and quote is not None
            and all(c.passed for c in checks.items)
        ):
            plan = self._size(checks, symbol, proposal, account, asset, quote)
        else:
            checks.add("Position size", ["not sized: earlier checks failed"], "")

        approved = plan is not None and all(c.passed for c in checks.items)
        decision = RiskDecision(
            approved=approved,
            symbol=symbol,
            checks=tuple(checks.items),
            plan=plan if approved else None,
            evaluated_at=now,
        )
        self._log.info(
            "risk.decision",
            approved=decision.approved,
            symbol=symbol,
            failed=[c.name for c in decision.failed_checks],
            notional_usd=str(plan.notional_usd) if plan and approved else None,
        )
        return decision

    # ---------- Individual checks ----------

    def _day_start(self, now: datetime) -> datetime:
        zone = ZoneInfo(self._config.schedule.timezone)
        local = now.astimezone(zone)
        return datetime.combine(local.date(), time(0), tzinfo=zone)

    def _check_market(
        self, checks: _Checks, clock: MarketClock | None, now: datetime, errors: dict[str, str]
    ) -> None:
        if clock is None:
            checks.add("Market open", [errors["market clock"]], "")
        else:
            checks.add("Market open", [] if clock.is_open else ["the market is closed"], "open")

        schedule = self._config.schedule
        local = now.astimezone(ZoneInfo(schedule.timezone)).time()
        window = f"{schedule.trade_window_start:%H:%M}-{schedule.trade_window_end:%H:%M}"
        inside = schedule.trade_window_start <= local <= schedule.trade_window_end
        checks.add(
            "Trade window",
            [] if inside else [f"{local:%H:%M} is outside the {window} trade window"],
            f"{local:%H:%M} within {window}",
        )

    def _check_trades_today(
        self, checks: _Checks, orders_today: int | None, errors: dict[str, str]
    ) -> None:
        limit = self._risk.max_trades_per_day
        if orders_today is None:
            checks.add("Trades today", [errors["today's orders"]], "")
            return
        checks.add(
            "Trades today",
            [f"{orders_today} buy order(s) today; the limit is {limit}"]
            if orders_today >= limit
            else [],
            f"{orders_today} of {limit}",
        )

    def _check_positions(
        self, checks: _Checks, positions: tuple[Position, ...] | None, errors: dict[str, str]
    ) -> None:
        if positions is None:
            checks.add("Open positions", [errors["positions"]], "")
            return
        held = ", ".join(p.symbol for p in positions)
        checks.add(
            "Open positions",
            [f"already holding {held}; one position at a time"] if positions else [],
            "none",
        )

    def _check_account(
        self, checks: _Checks, account: AccountSnapshot | None, errors: dict[str, str]
    ) -> None:
        if account is None:
            checks.add("Account", [errors["account"]], "")
            return
        issues: list[str] = []
        if account.is_paper != self._config.broker.paper:
            issues.append("account type does not match config")
        if account.trading_blocked or account.account_blocked:
            issues.append(f"account is blocked (status {account.status})")
        if account.status.upper() != "ACTIVE":
            issues.append(f"account status is {account.status}")
        if account.equity < self._risk.account_floor_usd:
            issues.append(
                f"equity ${account.equity} is below the ${self._risk.account_floor_usd} floor"
            )
        checks.add("Account", issues, f"{account.status}, equity ${account.equity}")

    def _check_instrument(
        self,
        checks: _Checks,
        symbol: str,
        proposal: TradeProposal,
        asset: Asset | None,
        quote: Quote | None,
        bars: tuple[Bar, ...],
        now: datetime,
        errors: dict[str, str],
    ) -> None:
        risk = self._risk
        if asset is None:
            checks.add("Asset", [errors.get("asset", f"no asset data for {symbol}")], "")
        else:
            checks.add("Asset", asset_issues(asset, risk), f"{asset.exchange}, tradable")

        if quote is None:
            missing = errors.get("quote", f"no quote available for {symbol}")
            for name in ("Price", "Quote", "Spread", "Price vs proposal"):
                checks.add(name, [missing], "")
        else:
            checks.add("Price", price_issues(quote.ask_price, risk), f"ask ${quote.ask_price}")
            checks.add(
                "Quote",
                quote_issues(quote, now=now, max_age_seconds=risk.max_data_age_seconds),
                f"{quote.age_seconds(now):.0f}s old",
            )
            spread = quote.spread_pct
            checks.add(
                "Spread",
                [f"spread {spread:.3f}% exceeds {risk.max_spread_pct}%"]
                if spread > risk.max_spread_pct
                else [],
                f"{spread:.3f}%",
            )
            reference = proposal.reference_price or Decimal("0")
            gap = abs(quote.ask_price - reference) / reference * 100 if reference > 0 else None
            checks.add(
                "Price vs proposal",
                [f"ask ${quote.ask_price} is {gap:.2f}% from the proposal's ${reference}"]
                if gap is None or gap > risk.max_price_deviation_pct
                else [],
                f"{gap:.2f}% from ${reference}" if gap is not None else "",
            )

        if not bars:
            checks.add("Liquidity", [errors.get("daily bars", f"no daily bars for {symbol}")], "")
        else:
            average = sum(b.volume for b in bars) // len(bars)
            checks.add(
                "Liquidity",
                liquidity_issues(average, risk),
                f"{average:,} shares/day over {len(bars)} days",
            )

    def _size(
        self,
        checks: _Checks,
        symbol: str,
        proposal: TradeProposal,
        account: AccountSnapshot,
        asset: Asset,
        quote: Quote,
    ) -> OrderPlan | None:
        risk = self._risk
        price = quote.ask_price
        reference = proposal.reference_price or price
        # Allow the configured deviation above the lower of the live ask and the analyst's
        # price, so a stale-high reference can never loosen the limit.
        limit_price = _cents_down(min(price, reference) * (1 + risk.max_price_deviation_pct / 100))
        # Cash, not buying power: no margin, ever.
        budget = min(risk.max_position_usd, account.cash)

        if asset.fractionable:
            quantity = (budget / limit_price).quantize(FRACTIONAL_STEP, rounding=ROUND_DOWN)
        else:
            quantity = (budget / limit_price).to_integral_value(rounding=ROUND_DOWN)
        notional = _cents_down(quantity * price)

        issues: list[str] = []
        if quantity <= 0:
            issues.append(f"${budget} buys no whole share of {symbol} at ${limit_price}")
        elif notional < MIN_NOTIONAL_USD:
            issues.append(f"order value ${notional} is below the ${MIN_NOTIONAL_USD} minimum")
        worst_loss = _cents_down(quantity * limit_price * risk.stop_loss_pct / 100)
        if account.equity - worst_loss < risk.account_floor_usd:
            issues.append(
                f"a stop-loss exit could lose ${worst_loss} and breach the "
                f"${risk.account_floor_usd} equity floor"
            )
        detail = (
            f"{quantity} {symbol} (~${notional}) limit ${limit_price}, "
            f"stop -{risk.stop_loss_pct}%, target +{risk.take_profit_pct}%"
        )
        if not checks.add("Position size", issues, detail):
            return None
        return OrderPlan(
            symbol=symbol,
            quantity=quantity,
            fractional=asset.fractionable,
            notional_usd=notional,
            entry_reference_price=price,
            limit_price=limit_price,
            stop_loss_price=_cents_down(price * (1 - risk.stop_loss_pct / 100)),
            take_profit_price=_cents_down(price * (1 + risk.take_profit_pct / 100)),
        )
