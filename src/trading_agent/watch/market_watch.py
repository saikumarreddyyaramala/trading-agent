"""The market watch: what moved outside regular hours, and why (ADR 0016).

Two deterministic scans, no LLM:

- pre-market (before the pre-flight): each symbol's latest trade against the last regular
  close before today, plus overnight news;
- after hours (once the session is over): each symbol's latest trade against today's
  close, plus the day's late news.

The symbols are the ones that matter next (yesterday's plan, open positions, today's trades
and ideas), then the day's biggest movers. A big move or a major catalyst raises an alert.
Headlines are kept for the operator but never leave this record as instructions: the
morning brief for the analyst uses only numbers and catalyst tags.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.data.catalysts import MAJOR, TAGS, catalyst_tags
from trading_agent.domain.models import Snapshot
from trading_agent.domain.ports import MarketDataPort, NewsPort
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig

Phase = Literal["pre", "post"]
_PHASE_WORDS = {"pre": "pre-market", "post": "after hours"}
_WINDOW_START = {"pre": time(4, 0), "post": time(16, 0)}  # extended sessions, market time
MAX_HEADLINES = 3
HEADLINE_CHARS = 140


class WatchItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    reasons: tuple[str, ...]  # plan, held, traded, proposed, gainer, loser, ...
    last_price: Decimal | None
    last_trade_at: datetime | None
    reference_close: Decimal | None  # the regular close the move is measured from
    move_pct: Decimal | None
    stale: bool  # no trade yet in this extended session
    news_count: int
    catalysts: tuple[str, ...]
    headlines: tuple[str, ...]  # for the operator only: untrusted third-party text
    alert: bool


class WatchReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: Phase
    day: date
    created_at: datetime
    market: WatchItem | None  # the benchmark (SPY)
    items: tuple[WatchItem, ...]
    alerts: tuple[str, ...]  # one sentence per alerting item, numbers and tags only
    errors: tuple[str, ...]


def _pct(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _headline(text: str) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= HEADLINE_CHARS else clean[: HEADLINE_CHARS - 1] + "…"


class MarketWatch:
    def __init__(
        self,
        market_data: MarketDataPort,
        news: NewsPort,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._market_data = market_data
        self._news = news
        self._config = config
        self._settings = config.watch
        self._zone = ZoneInfo(config.schedule.timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def scan(self, phase: Phase, day: date, focus: Mapping[str, Sequence[str]]) -> WatchReport:
        """Scan `focus` (symbol -> why it matters) plus the day's biggest movers."""
        now = self._clock()
        if not self._settings.enabled:
            return WatchReport(
                phase=phase, day=day, created_at=now, market=None, items=(), alerts=(), errors=()
            )
        errors: list[str] = []
        reasons: dict[str, list[str]] = {s.upper(): list(r) for s, r in focus.items()}
        movers: list[str] = []
        if self._settings.movers:
            try:
                for mover in self._market_data.get_movers(self._settings.movers):
                    if mover.price < self._config.risk.min_price_usd:
                        continue
                    if mover.symbol not in reasons:
                        movers.append(mover.symbol)
                    reasons.setdefault(mover.symbol, []).append(mover.direction)
            except Exception as exc:
                errors.append(f"movers unavailable: {exc}")
        room = max(self._settings.max_symbols - len(focus), 0)
        symbols = [s.upper() for s in focus][: self._settings.max_symbols] + movers[:room]

        benchmark = self._config.reporting.benchmark_symbol
        snapshots: dict[str, Snapshot] = {}
        try:
            snapshots = self._market_data.get_snapshots([*symbols, benchmark])
        except Exception as exc:
            errors.append(f"snapshots unavailable: {exc}")

        since = now - timedelta(hours=self._settings.news_lookback_hours)
        articles: dict[str, list[Any]] = {s: [] for s in symbols}
        if symbols:
            try:
                for article in self._news.get_news(symbols, since=since, limit=50):
                    for symbol in article.symbols:
                        if symbol in articles:
                            articles[symbol].append(article)
            except Exception as exc:
                errors.append(f"news unavailable: {exc}")

        window = datetime.combine(day, _WINDOW_START[phase], tzinfo=self._zone)
        items = [
            self._item(phase, day, window, s, tuple(reasons[s]), snapshots.get(s), articles[s])
            for s in symbols
        ]
        items.sort(
            key=lambda i: (
                not i.alert,
                i.move_pct is None,
                -abs(i.move_pct or Decimal(0)),
                i.symbol,
            )
        )
        market = (
            self._item(phase, day, window, benchmark, ("market",), snapshots[benchmark], [])
            if benchmark in snapshots
            else None
        )
        report = WatchReport(
            phase=phase,
            day=day,
            created_at=now,
            market=market,
            items=tuple(items),
            alerts=tuple(self._alert(phase, i) for i in items if i.alert),
            errors=tuple(errors),
        )
        self._log.info(
            "watch.scanned",
            phase=phase,
            day=str(day),
            symbols=len(items),
            alerts=len(report.alerts),
        )
        return report

    def _reference(self, phase: Phase, day: date, snap: Snapshot) -> Decimal | None:
        session_day = (
            None if snap.session_at is None else snap.session_at.astimezone(self._zone).date()
        )
        if phase == "pre":
            # The latest daily bar may already be today's (built from pre-market trades):
            # then the last regular close is the bar before it.
            return snap.previous_close if session_day == day else snap.session_close
        return snap.session_close if session_day == day else None

    def _item(
        self,
        phase: Phase,
        day: date,
        window: datetime,
        symbol: str,
        reasons: tuple[str, ...],
        snap: Snapshot | None,
        articles: list[Any],
    ) -> WatchItem:
        reference = None if snap is None else self._reference(phase, day, snap)
        fresh = (
            snap is not None
            and snap.last_trade_at is not None
            and snap.last_price is not None
            and snap.last_trade_at >= window
        )
        move = None
        if fresh and snap is not None and snap.last_price is not None and reference:
            move = _pct((snap.last_price - reference) / reference * 100)
        tags: list[str] = []
        for article in articles:
            tags += [t for t in catalyst_tags(article.headline, article.summary) if t not in tags]
        ordered = tuple(t for t in TAGS if t in tags)
        alert = (move is not None and abs(move) >= self._settings.gap_alert_pct) or any(
            t in MAJOR for t in ordered
        )
        return WatchItem(
            symbol=symbol,
            reasons=reasons,
            last_price=None if snap is None else snap.last_price,
            last_trade_at=None if snap is None else snap.last_trade_at,
            reference_close=reference,
            move_pct=move,
            stale=snap is not None and not fresh,
            news_count=len(articles),
            catalysts=ordered,
            headlines=tuple(_headline(a.headline) for a in articles[:MAX_HEADLINES]),
            alert=alert,
        )

    @staticmethod
    def _alert(phase: Phase, item: WatchItem) -> str:
        words = _PHASE_WORDS[phase]
        count = f"{item.news_count} article{'s' if item.news_count != 1 else ''}"
        news = f"{', '.join(item.catalysts)} news ({count})" if item.catalysts else ""
        if item.move_pct is None:
            return f"{item.symbol}: {news}, no {words} trades yet"
        text = f"{item.symbol} {item.move_pct:+.2f}% {words} ({', '.join(item.reasons)})"
        return f"{text}; {news}" if news else text
