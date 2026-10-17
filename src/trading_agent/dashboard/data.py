"""The dashboard's data layer (ADR 0018): everything the UI shows, read-only.

Each section degrades on its own. A broker outage leaves the account empty and says why,
but the cycle, the plan, and the costs still show. Broker and market-data answers are
cached for a few seconds so a busy UI can never hammer the APIs.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from functools import partial
from typing import Any, TypeVar, cast
from zoneinfo import ZoneInfo

import anyio

from trading_agent.costs.summary import cost_report
from trading_agent.dashboard.news_cards import build_cards
from trading_agent.domain.models import MarketClock, Snapshot
from trading_agent.domain.ports import BrokerPort, MarketDataPort, NewsPort
from trading_agent.execution.alpaca_broker import HISTORY_SPANS
from trading_agent.observability.logging import get_logger
from trading_agent.reporting.performance import SYSTEM_RUN_PREFIX, summarize, trades_from_orders
from trading_agent.reporting.slots import slot_runs, summarize_slot
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditError, AuditStore

T = TypeVar("T")
CENT = Decimal("0.01")
NEWS_LOOKBACK = timedelta(hours=24)


def jsonable(value: Any) -> Any:
    """Plain JSON types: Decimals as exact strings, dates and times as ISO 8601."""

    def default(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime | date | time):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if isinstance(obj, set | tuple):
            return list(obj)
        raise TypeError(f"not JSON serializable: {type(obj).__name__}")

    return json.loads(json.dumps(value, default=default))


def _obj(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], jsonable(value))


def _rows(value: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], jsonable(value))


def _pct(part: Decimal, whole: Decimal) -> Decimal | None:
    if not whole:
        return None
    value = (part / whole * 100).quantize(CENT, ROUND_HALF_UP)
    return value if value else Decimal("0.00")  # never "-0.00"


class DashboardData:
    def __init__(
        self,
        store: AuditStore,
        config: AppConfig,
        *,
        broker: BrokerPort | None,
        market_data: MarketDataPort | None,
        news: NewsPort | None,
        clock: Callable[[], datetime] | None = None,
        cache_seconds: float = 5,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._config = config
        self._broker = broker
        self._market_data = market_data
        self._news = news
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ttl = cache_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._source_errors: dict[str, str] = {}
        self._log = logger or get_logger(__name__)

    # ---------- Helpers ----------

    def _today(self) -> date:
        return self._clock().astimezone(self._zone).date()

    def _cached(self, key: str, fn: Callable[[], T]) -> T:
        hit = self._cache.get(key)
        if hit is not None and _time.monotonic() - hit[0] < self._ttl:
            return hit[1]  # type: ignore[no-any-return]
        value = fn()
        if self._ttl > 0:
            self._cache[key] = (_time.monotonic(), value)
        return value

    def _try(self, source: str, fn: Callable[[], T], errors: list[str]) -> T | None:
        try:
            value = self._cached(source, fn)
        except Exception as exc:
            errors.append(f"{source} unavailable: {exc}")
            return None
        return value

    def _one(self, fn: Callable[[], T]) -> T | None:
        """A store read that degrades to None when the database is unreadable."""
        try:
            return fn()
        except AuditError as exc:
            self._log.warning("dashboard.store_unavailable", error=str(exc))
            return None

    def _many(self, fn: Callable[[], list[Any]]) -> list[Any]:
        """A store read that degrades to an empty list when the database is unreadable."""
        found = self._one(fn)
        return [] if found is None else found

    def _day_window(self, day: date) -> tuple[datetime, datetime]:
        start = datetime.combine(day, time(0), tzinfo=self._zone)
        return start, start + timedelta(days=1)

    # ---------- Overview ----------

    def overview(self) -> dict[str, Any]:
        errors: list[str] = []
        account = positions = clock = None
        if self._broker is None:
            errors.append("broker unavailable: no broker credentials")
        else:
            broker = self._broker
            account = self._try("account", broker.get_account, errors)
            positions = self._try("positions", broker.get_positions, errors)
            clock = self._try("market clock", broker.get_clock, errors)

        risk = self._config.risk
        rows = []
        for p in positions or ():
            price = p.market_value / p.quantity if p.quantity else p.avg_entry_price
            pnl = p.market_value - p.quantity * p.avg_entry_price
            rows.append(
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_entry_price": p.avg_entry_price,
                    "price": price.quantize(CENT, ROUND_HALF_UP),
                    "market_value": p.market_value,
                    "unrealized_pnl": pnl.quantize(CENT, ROUND_HALF_UP),
                    "unrealized_pct": _pct(pnl, p.quantity * p.avg_entry_price),
                    "stop_price": (p.avg_entry_price * (1 - risk.stop_loss_pct / 100)).quantize(
                        CENT, ROUND_DOWN
                    ),
                    "target_price": (p.avg_entry_price * (1 + risk.take_profit_pct / 100)).quantize(
                        CENT, ROUND_DOWN
                    ),
                }
            )
        account_view = None
        if account is not None:
            change = None if account.last_equity is None else account.equity - account.last_equity
            account_view = {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "last_equity": account.last_equity,
                "day_change": change,
                "day_change_pct": None
                if change is None or account.last_equity is None
                else _pct(change, account.last_equity),
                "account": account.account_number_masked,
            }
        errors += [message for message in self._source_errors.values() if message not in errors]
        return _obj(
            {
                "generated_at": self._clock(),
                "environment": self._config.environment.value,
                "paper": not self._config.is_live,
                "kill_switch": self._config.kill_switch,
                "account": account_view,
                "positions": rows,
                "market": None
                if clock is None
                else {
                    "is_open": clock.is_open,
                    "next_open": clock.next_open,
                    "next_close": clock.next_close,
                },
                "cycle": (cycle := self._cycle(self._today())),
                "next_step": self._next_step(clock, ran_today=cycle is not None),
                "today": self._today_costs(),
                "errors": errors,
            }
        )

    def _cycle(self, day: date) -> dict[str, Any] | None:
        run_id = f"{SYSTEM_RUN_PREFIX}{day:%Y%m%d}"
        run = self._one(lambda: self._store.get_run(run_id))
        if run is None:
            return None
        slots = self._many(lambda: slot_runs(self._store, run_id))
        return {
            "run_id": run_id,
            "status": run["run"]["status"],
            "summary": run["run"]["summary"],
            "steps": [
                {
                    "step": s["step"],
                    "status": s["status"],
                    "detail": s["detail"],
                    "finished_at": s["finished_at"],
                }
                for s in run["steps"]
            ],
            "slots": [summarize_slot(slot) for slot in slots],
        }

    def _next_step(
        self, clock: MarketClock | None = None, *, ran_today: bool = False
    ) -> dict[str, Any]:
        """The next scheduled step. On a day without a session (a weekend or a holiday, per
        the broker's clock, and no cycle ran), that is the next trading day's pre-market scan."""
        schedule, risk = self._config.schedule, self._config.risk
        now = self._clock().astimezone(self._zone)
        if clock is not None and not clock.is_open and not ran_today:
            next_day = clock.next_open.astimezone(self._zone).date()
            if next_day > now.date():
                closed_today = clock.next_close.astimezone(self._zone).date() != now.date()
                if closed_today:
                    return {
                        "step": "premarket",
                        "at": datetime.combine(
                            next_day, schedule.premarket_watch, tzinfo=self._zone
                        ),
                    }
        steps = [
            ("premarket", schedule.premarket_watch),
            ("preflight", schedule.data_collection),
            ("analysis", schedule.analysis),
            ("entry", schedule.trade_window_start),
            *[
                (f"slot{n}", at)
                for n, at in enumerate(
                    schedule.extra_analyses[: max(risk.max_trades_per_day - 1, 0)], start=2
                )
            ],
            ("close", schedule.force_close),
            ("reconcile", schedule.report),
            ("postmarket", schedule.post_market),
        ]
        for name, at in steps:
            when = datetime.combine(now.date(), at, tzinfo=self._zone)
            if when > now:
                return {"step": name, "at": when}
        tomorrow = now.date() + timedelta(days=1)
        return {
            "step": "premarket",
            "at": datetime.combine(tomorrow, schedule.premarket_watch, tzinfo=self._zone),
        }

    def _today_costs(self) -> dict[str, Any]:
        start, end = self._day_window(self._today())
        rows = self._many(lambda: self._store.llm_calls_between(start, end))
        known = sum((r["cost_usd"] for r in rows if r["cost_usd"] is not None), Decimal(0))
        unpriced = sum(1 for r in rows if r["cost_usd"] is None)
        return {
            "calls": len(rows),
            "llm_usd": None if unpriced else known,
            "llm_known_usd": known,
            "unpriced_calls": unpriced,
            "budget_usd": self._config.costs.daily_llm_budget_usd,
        }

    # ---------- Charts and trades ----------

    def history(self, span: str) -> dict[str, Any]:
        if span not in HISTORY_SPANS:
            raise ValueError(f"unknown span {span!r}: use one of {', '.join(HISTORY_SPANS)}")
        if self._broker is None:
            return {"span": span, "points": [], "error": "no broker credentials"}
        broker = self._broker
        history = self._cached(f"history:{span}", lambda: broker.get_portfolio_history(span))
        # Alpaca reports zero equity for the days before the account was funded.
        points = history.points
        while points and points[0].equity == 0:
            points = points[1:]
        base = history.base_value or (points[0].equity if points else None)
        if points and len(points) < len(history.points):
            base = points[0].equity  # measure from funding, not from the empty account
        change = None if not points or base is None else points[-1].equity - base
        return _obj(
            {
                "span": span,
                "base_value": base,
                "points": [{"at": p.at, "equity": p.equity} for p in points],
                "change": change,
                "change_pct": None if change is None or base is None else _pct(change, base),
            }
        )

    def _trades(self, days: int) -> list[Any]:
        end = self._clock() + timedelta(days=1)
        orders = self._many(lambda: self._store.orders_between(end - timedelta(days=days + 1), end))
        return trades_from_orders(orders, self._zone)

    def trades(self, days: int) -> list[dict[str, Any]]:
        return _rows([t.model_dump() for t in reversed(self._trades(days))])

    def performance(self, days: int) -> dict[str, Any]:
        return _obj(summarize(self._trades(days), {}))

    # ---------- News ----------

    def _watched(self) -> set[str]:
        symbols = {p.symbol for p in self._cached_positions()}
        today = self._today()
        plan = self._one(lambda: self._store.plan_for(today))
        symbols |= {i["symbol"] for i in (plan or {}).get("items", []) if i.get("symbol")}
        cycle = self._cycle(today)
        for slot in (cycle or {}).get("slots", []):
            if slot.symbol:
                symbols.add(slot.symbol)
        return symbols

    def _cached_positions(self) -> tuple[Any, ...]:
        if self._broker is None:
            return ()
        broker = self._broker
        try:
            return self._cached("positions", broker.get_positions)
        except Exception:
            return ()

    def _moves(self, symbols: list[str]) -> dict[str, Decimal]:
        if self._market_data is None or not symbols:
            return {}
        market_data = self._market_data
        try:
            snaps: dict[str, Snapshot] = self._cached(
                "snapshots:" + ",".join(sorted(symbols)), lambda: market_data.get_snapshots(symbols)
            )
        except Exception as exc:
            self._source_errors["snapshots"] = f"snapshots unavailable: {exc}"
            return {}
        today = self._today()
        moves = {}
        for symbol, snap in snaps.items():
            session_day = (
                None if snap.session_at is None else snap.session_at.astimezone(self._zone).date()
            )
            reference = snap.previous_close if session_day == today else snap.session_close
            if snap.last_price is not None and reference:
                moves[symbol] = ((snap.last_price - reference) / reference * 100).quantize(
                    CENT, ROUND_HALF_UP
                )
        return moves

    def news(self, limit: int) -> list[dict[str, Any]]:
        if self._news is None:
            return []
        news = self._news
        watched = self._watched()
        since = self._clock() - NEWS_LOOKBACK
        try:
            articles = list(
                self._cached("news:all", lambda: news.get_news([], since=since, limit=50))
            )
            if watched:
                articles += self._cached(
                    "news:watched:" + ",".join(sorted(watched)),
                    lambda: news.get_news(sorted(watched), since=since, limit=50),
                )
        except Exception as exc:
            self._source_errors["news"] = f"news unavailable: {exc}"
            return []
        self._source_errors.pop("news", None)
        symbols = sorted({s for a in articles for s in a.symbols[:1]})[:30]
        cards = build_cards(
            articles, self._clock(), moves=self._moves(symbols), watched=watched, limit=limit
        )
        return _rows([c.model_dump() for c in cards])

    # ---------- Watch, plan, learning, costs, activity ----------

    def watch(self, day: date) -> dict[str, Any]:
        pre = self._one(lambda: self._store.get_watch(day, "pre"))
        post = self._one(lambda: self._store.get_watch(day, "post"))
        return _obj({"day": day, "pre": pre, "post": post})

    def plan(self, day: date) -> dict[str, Any]:
        return _obj(
            {
                "day": day,
                "today": self._one(lambda: self._store.plan_for(day)),
                "next": self._one(lambda: self._store.plan_from(day)),
            }
        )

    def learning(self) -> dict[str, Any]:
        now = self._clock()
        lessons = self._many(lambda: self._store.active_lessons(now, 50))
        playbooks = self._many(lambda: self._store.list_playbooks(include_retired=True))
        guidebook = self._one(self._store.latest_guidebook)
        return _obj(
            {
                "principles": [lesson for lesson in lessons if lesson.promoted],
                "lessons": [lesson for lesson in lessons if not lesson.promoted],
                "playbooks": playbooks,
                "guidebook": None
                if guidebook is None
                else {k: guidebook[k] for k in ("version", "created_at", "changes", "content")},
            }
        )

    def costs(self, days: int) -> dict[str, Any]:
        end_day = self._today()
        start_day = end_day - timedelta(days=days - 1)
        start, _ = self._day_window(start_day)
        _, end = self._day_window(end_day)
        rows = self._many(lambda: self._store.llm_calls_between(start, end))
        return _obj(cost_report(rows, self._config, start_day, end_day))

    def activity(self, limit: int, since: datetime | None = None) -> list[dict[str, Any]]:
        since = since or self._clock() - timedelta(days=7)
        return _rows(self._many(lambda: self._store.activity_since(since, limit)))

    def run(self, run_id: str) -> dict[str, Any] | None:
        found = self._one(lambda: self._store.get_run(run_id))
        return None if found is None else _obj(found)


async def stream(
    data: DashboardData,
    *,
    interval: float,
    ticks: int | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    since: datetime | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Live updates: the overview whenever it changes, and each new activity item."""
    cursor = since or datetime.now(UTC)
    last_hash = None
    tick = 0
    while ticks is None or tick < ticks:
        overview = await anyio.to_thread.run_sync(data.overview)
        digest = hashlib.sha256(
            json.dumps(
                {k: v for k, v in overview.items() if k != "generated_at"}, sort_keys=True
            ).encode()
        ).hexdigest()
        if digest != last_hash:
            last_hash = digest
            yield {"event": "overview", "data": overview}
        items = await anyio.to_thread.run_sync(partial(data.activity, 100, since=cursor))
        for item in reversed(items):  # oldest first
            yield {"event": "activity", "data": item}
        if items:
            cursor = max(datetime.fromisoformat(i["at"]) for i in items)
        tick += 1
        if ticks is None or tick < ticks:
            await sleep(interval)
