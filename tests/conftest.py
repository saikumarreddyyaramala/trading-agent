"""Shared fixtures and fakes. Unit tests never touch the network or real credentials."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import (
    AccountSnapshot,
    ActiveStock,
    Asset,
    Bar,
    BrokerOrder,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MarketClock,
    NewsArticle,
    Position,
    Quote,
    ToolCall,
)
from trading_agent.llm.errors import LLMProviderError
from trading_agent.settings.secrets import Secrets

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"

# Fixed "now" for market data tests: 2026-10-15 10:00 New York time.
NOW = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A private copy of the real config folder, safe to modify per test."""
    target = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, target)
    return target


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test may read the developer's real .env or exported keys."""
    for name in (
        "APP_ENV",
        "CONFIG_DIR",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LIVE_TRADING_CONFIRMED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def make_secrets(**overrides: Any) -> Secrets:
    values: dict[str, Any] = {
        "alpaca_api_key": SecretStr("test-alpaca-key"),
        "alpaca_secret_key": SecretStr("test-alpaca-secret"),
        "openai_api_key": SecretStr("test-openai-key"),
        "anthropic_api_key": SecretStr("test-anthropic-key"),
        "live_trading_confirmed": False,
    }
    values.update(overrides)
    return Secrets(_env_file=None, **values)


class FakeBroker:
    def __init__(
        self,
        *,
        is_paper: bool = True,
        trading_blocked: bool = False,
        fail_account: bool = False,
        fail_clock: bool = False,
        market_open: bool = True,
        equity: str = "100000",
        cash: str = "100000",
        positions: Sequence[Position] = (),
        orders_today: int = 0,
        fail_positions: bool = False,
        fail_orders: bool = False,
    ) -> None:
        self._market_open = market_open
        self._equity = Decimal(equity)
        self._cash = Decimal(cash)
        self.positions = tuple(positions)
        self.orders_today = orders_today
        self._fail_positions = fail_positions
        self._fail_orders = fail_orders
        self.orders_since: list[datetime] = []
        self._is_paper = is_paper
        self._trading_blocked = trading_blocked
        self._fail_account = fail_account
        self._fail_clock = fail_clock

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def get_account(self) -> AccountSnapshot:
        if self._fail_account:
            raise RuntimeError("broker unreachable")
        return AccountSnapshot(
            account_number_masked="****1234",
            status="ACTIVE",
            currency="USD",
            cash=self._cash,
            equity=self._equity,
            buying_power=Decimal("100000"),
            pattern_day_trader=False,
            trading_blocked=self._trading_blocked,
            account_blocked=False,
            is_paper=self._is_paper,
        )

    def get_clock(self) -> MarketClock:
        if self._fail_clock:
            raise RuntimeError("clock unavailable")
        return MarketClock(timestamp=NOW, is_open=self._market_open, next_open=NOW, next_close=NOW)

    def get_positions(self) -> tuple[Position, ...]:
        if self._fail_positions:
            raise RuntimeError("positions unavailable")
        return self.positions

    def count_buy_orders_since(self, since: datetime) -> int:
        self.orders_since.append(since)
        if self._fail_orders:
            raise RuntimeError("orders unavailable")
        return self.orders_today


class FakeProvider:
    """Scripted LLM provider: each call pops the next outcome.

    An outcome is a final text, a tuple of tool calls to request, or an exception to raise.
    """

    def __init__(self, outcomes: Sequence[str | tuple[ToolCall, ...] | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []
        self.requests: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        self.calls.append((model, max_tokens))
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        calls = outcome if isinstance(outcome, tuple) else ()
        return LLMResponse(
            text=outcome if isinstance(outcome, str) else "",
            provider=self.name,
            model=model,
            stop_reason="tool_use" if calls else "end_turn",
            usage=LLMUsage(input_tokens=10, output_tokens=2),
            latency_ms=5.0,
            tool_calls=calls,
        )


def tool_call(call_id: str, name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments))


def provider_error(model: str = "m", *, fatal: bool = False) -> LLMProviderError:
    return LLMProviderError("fake", model, "boom", fatal=fatal)


# ---------- Market data builders and fakes ----------


def make_quote(
    symbol: str = "AAPL",
    *,
    bid: str = "100.00",
    ask: str = "100.10",
    age_seconds: float = 2,
    now: datetime = NOW,
) -> Quote:
    return Quote(
        symbol=symbol,
        timestamp=now - timedelta(seconds=age_seconds),
        bid_price=Decimal(bid),
        bid_size=Decimal("100"),
        ask_price=Decimal(ask),
        ask_size=Decimal("200"),
    )


def make_bars(
    symbol: str = "AAPL", closes: Sequence[str] = ("100", "102", "101"), volume: int = 2_000_000
) -> tuple[Bar, ...]:
    """One daily bar per close, oldest first, ending the day before NOW."""
    start = NOW - timedelta(days=len(closes))
    bars = []
    for offset, raw in enumerate(closes):
        close = Decimal(raw)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=start + timedelta(days=offset),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=volume,
            )
        )
    return tuple(bars)


def make_asset(
    symbol: str = "AAPL",
    *,
    tradable: bool = True,
    name: str | None = None,
    exchange: str = "NASDAQ",
    fractionable: bool = True,
    status: str = "active",
) -> Asset:
    return Asset(
        symbol=symbol,
        name=name or f"{symbol} Inc.",
        exchange=exchange,
        status=status,
        tradable=tradable,
        fractionable=fractionable,
    )


def make_article(
    article_id: str = "1", symbols: Sequence[str] = ("AAPL",), *, hours_ago: float = 1
) -> NewsArticle:
    created = NOW - timedelta(hours=hours_ago)
    return NewsArticle(
        id=article_id,
        headline=f"Headline {article_id}",
        summary="Summary.",
        source="benzinga",
        url=f"https://example.com/{article_id}",
        symbols=tuple(symbols),
        created_at=created,
        updated_at=created,
    )


class FakeMarketData:
    """In-memory `MarketDataPort`. Records calls; any data not supplied is simply absent."""

    def __init__(
        self,
        *,
        quotes: dict[str, Quote] | None = None,
        bars: dict[str, tuple[Bar, ...]] | None = None,
        assets: dict[str, Asset] | None = None,
        active: Sequence[ActiveStock] = (),
        fail: bool = False,
    ) -> None:
        self.quotes = quotes or {}
        self.bars = bars or {}
        self.assets = assets or {}
        self.active = tuple(active)
        self.fail = fail
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def get_asset(self, symbol: str) -> Asset:
        self.calls.append(("asset", (symbol,)))
        if symbol not in self.assets:
            raise MarketDataError(f"asset lookup failed for {symbol}: not found")
        return self.assets[symbol]

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        self.calls.append(("quotes", tuple(symbols)))
        if self.fail:
            raise MarketDataError("could not fetch quotes: down")
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    def get_daily_bars(self, symbols: Sequence[str], *, days: int) -> dict[str, tuple[Bar, ...]]:
        self.calls.append(("bars", tuple(symbols)))
        return {s: self.bars[s][-days:] for s in symbols if s in self.bars}

    def get_most_active(self, limit: int) -> tuple[ActiveStock, ...]:
        self.calls.append(("most_active", (str(limit),)))
        if self.fail:
            raise MarketDataError("could not fetch most active stocks: down")
        return self.active[:limit]


def healthy_market_data(*symbols: str) -> FakeMarketData:
    names = symbols or ("AAPL",)
    return FakeMarketData(
        quotes={s: make_quote(s) for s in names},
        bars={s: make_bars(s) for s in names},
        assets={s: make_asset(s) for s in names},
    )


class FakeNews:
    def __init__(self, articles: Sequence[NewsArticle] = (), *, fail: bool = False) -> None:
        self.articles = tuple(articles)
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def get_news(
        self, symbols: Sequence[str], *, since: datetime, limit: int
    ) -> tuple[NewsArticle, ...]:
        self.calls.append({"symbols": tuple(symbols), "since": since, "limit": limit})
        if self.fail:
            raise MarketDataError("could not fetch news: down")
        return self.articles[:limit]


# ---------- Simulated order execution ----------


class FakeTime:
    """A clock whose `sleep` advances time instantly, for polling loops."""

    def __init__(self, start: datetime = NOW) -> None:
        self.now = start
        self.slept = 0.0

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += timedelta(seconds=seconds)


class FakeOrders:
    """In-memory `OrderExecutionPort` that simulates how an entry order fills.

    fill: "instant" (filled on submit), "full" (filled on the first poll), "partial" (half
    fills, the rest never does), "none" (never fills), "late" (fills while being cancelled).
    """

    def __init__(
        self,
        fill: str = "full",
        *,
        reject_entry: bool = False,
        reject_stop: bool = False,
        fail_close: bool = False,
        sticky_cancel: bool = False,
        fail_polls: int = 0,
    ) -> None:
        self.fill = fill
        self.reject_entry = reject_entry
        self.reject_stop = reject_stop
        self.fail_close = fail_close
        self.sticky_cancel = sticky_cancel
        self.fail_polls = fail_polls
        self.orders: dict[str, BrokerOrder] = {}
        self.log: list[tuple[str, str]] = []  # (action, detail), in call order

    def _new(
        self, client_id: str, symbol: str, side: str, kind: str, qty: Decimal, **px: Any
    ) -> BrokerOrder:
        order = BrokerOrder(
            id=f"id-{client_id}",
            client_order_id=client_id,
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            order_type=kind,
            status="new",
            quantity=qty,
            filled_quantity=Decimal("0"),
            submitted_at=NOW,
            **px,
        )
        self.orders[client_id] = order
        return order

    def _update(self, client_id: str, **changes: Any) -> BrokerOrder:
        self.orders[client_id] = self.orders[client_id].model_copy(update=changes)
        return self.orders[client_id]

    def _filled(
        self, order: BrokerOrder, qty: Decimal | None = None, status: str = "filled"
    ) -> BrokerOrder:
        price = (order.limit_price or Decimal("100")) - Decimal("0.05")
        return self._update(
            order.client_order_id,
            status=status,
            filled_quantity=order.quantity if qty is None else qty,
            filled_avg_price=price,
        )

    def submit_limit_buy(
        self, symbol: str, quantity: Decimal, limit_price: Decimal, client_order_id: str
    ) -> BrokerOrder:
        self.log.append(("buy", f"{quantity} {symbol} @ {limit_price} [{client_order_id}]"))
        if self.reject_entry:
            raise RuntimeError("insufficient buying power")
        order = self._new(
            client_order_id, symbol, "buy", "limit", quantity, limit_price=limit_price
        )
        return self._filled(order) if self.fill == "instant" else order

    def submit_stop_sell(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> BrokerOrder:
        self.log.append(("stop", f"{quantity} {symbol} @ {stop_price} [{client_order_id}]"))
        if self.reject_stop:
            raise RuntimeError("stop rejected")
        return self._new(client_order_id, symbol, "sell", "stop", quantity, stop_price=stop_price)

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        if self.fail_polls > 0 and client_order_id in self.orders:
            self.fail_polls -= 1
            raise RuntimeError("timeout")
        order = self.orders.get(client_order_id)
        if order is None or order.side != "buy" or order.is_terminal:
            return order
        if self.fill == "full":
            return self._filled(order)
        if self.fill == "partial" and order.status == "new":
            return self._filled(order, order.quantity / 2, status="partially_filled")
        return order

    def get_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]:
        return tuple(
            o
            for o in self.orders.values()
            if (not o.is_terminal or (self.sticky_cancel and o.status == "canceled"))
            and (symbol is None or o.symbol == symbol)
        )

    def cancel_order(self, order_id: str) -> None:
        order = next(o for o in self.orders.values() if o.id == order_id)
        self.log.append(("cancel", order.client_order_id))
        if order.is_terminal:
            return
        if order.side == "buy" and self.fill == "late":
            self._filled(order)
        else:
            self._update(order.client_order_id, status="canceled")

    def close_position(self, symbol: str) -> BrokerOrder:
        self.log.append(("close", symbol))
        if self.fail_close:
            raise RuntimeError("market closed")
        order = self._new(f"close-{symbol}-{len(self.log)}", symbol, "sell", "market", Decimal("1"))
        return self._update(order.client_order_id, status="filled", filled_quantity=Decimal("1"))


def make_position(symbol: str = "AAPL", quantity: str = "0.4899", avg: str = "100") -> Position:
    qty, price = Decimal(quantity), Decimal(avg)
    return Position(symbol=symbol, quantity=qty, market_value=qty * price, avg_entry_price=price)
