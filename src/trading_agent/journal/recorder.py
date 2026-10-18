"""The all-day journal recorder (ADR 0023): watches and records; it never trades.

On each trading day (holidays and early closes come from the broker's calendar):

    04:00        pre-market opens: a snapshot every 30 minutes
    09:30        regular session (the bot trades here, elsewhere): note on the pre-market
    16:00        post-market: note on the regular session
    20:00        post-market ends (close + 4 h): note on the post-market
    20:20        the day notebook connects the dots

A snapshot holds the market indexes, the bot's stocks and the day's movers (the market
watch measures them: pre-market and regular against the last close, post-market against
today's close), new company news and official releases, and in regular hours the bot's
trades. It runs on its own thread; every failure is recorded and contained, so it can
never stop or slow the trading cycle. After a restart it continues from the day's files.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from trading_agent.domain.models import MarketDay
from trading_agent.journal.facts import day_facts, facts_headline
from trading_agent.journal.models import (
    SESSIONS,
    DayNotebook,
    JournalSnapshot,
    Move,
    Session,
    SessionRecord,
)
from trading_agent.journal.sources import Source, fetch_all
from trading_agent.journal.store import JournalStore
from trading_agent.journal.writer import JournalError, JournalWriter
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import JournalConfig
from trading_agent.watch.market_watch import Phase, WatchItem, WatchReport

OVERNIGHT = timedelta(hours=8)  # the first fetch of the day looks back this far
OVERLAP = timedelta(minutes=5)  # later fetches overlap the last one a little
RETRY_AFTER = timedelta(minutes=30)
MAX_ATTEMPTS = 3


class DayTimes(NamedTuple):
    day: date
    pre_start: datetime
    open: datetime
    close: datetime
    post_end: datetime
    notebook_at: datetime


def day_times(market: MarketDay, config: JournalConfig, zone: ZoneInfo) -> DayTimes:
    post_end = market.close + timedelta(hours=config.post_market_hours)
    return DayTimes(
        day=market.day,
        pre_start=datetime.combine(market.day, config.pre_market_start, tzinfo=zone),
        open=market.open,
        close=market.close,
        post_end=post_end,
        notebook_at=post_end + timedelta(minutes=config.notebook_delay_minutes),
    )


def session_at(now: datetime, times: DayTimes) -> Session | None:
    if times.pre_start <= now < times.open:
        return "pre"
    if times.open <= now < times.close:
        return "regular"
    if times.close <= now < times.post_end:
        return "post"
    return None


def _move(item: WatchItem) -> Move:
    return Move(
        symbol=item.symbol,
        reasons=item.reasons,
        price=item.last_price,
        reference=item.reference_close,
        move_pct=item.move_pct,
        stale=item.stale,
        catalysts=item.catalysts,
        alert=item.alert,
    )


class JournalRecorder:
    def __init__(
        self,
        store: JournalStore,
        *,
        scan: Callable[[Phase, date, dict[str, list[str]]], WatchReport],
        sources: Sequence[Source],
        writer: JournalWriter | None,
        market_day: Callable[[date], MarketDay | None],
        focus: Callable[[date], dict[str, list[str]]],
        trades: Callable[[date], list[dict[str, Any]]],
        config: JournalConfig,
        timezone: str,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._scan = scan
        self._sources = sources
        self._writer = writer
        self._market_day = market_day
        self._focus = focus
        self._trades = trades
        self._config = config
        self._zone = ZoneInfo(timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)
        self._times: dict[date, DayTimes | None] = {}
        self._done: set[tuple[date, str]] = set()
        self._attempts: dict[tuple[date, str], tuple[int, datetime]] = {}

    # ---------- One pass ----------

    def _day(self, day: date) -> DayTimes | None:
        if day not in self._times:
            market = self._market_day(day)  # may raise: retried on the next pass
            self._times[day] = (
                None if market is None else day_times(market, self._config, self._zone)
            )
        return self._times[day]

    def step(self) -> list[str]:
        """Do whatever is due now. Returns what was done, e.g. ["note:pre", "snapshot:regular"]."""
        if not self._config.enabled:
            return []
        now = self._clock()
        local = now.astimezone(self._zone)
        day = local.date()
        try:
            times = self._day(day)
        except Exception as exc:
            self._log.warning("journal.calendar_failed", error=str(exc))
            return []
        if times is None:
            return []
        actions: list[str] = []
        for session, end in (
            ("pre", times.open),
            ("regular", times.close),
            ("post", times.post_end),
        ):
            if local >= end:
                actions += self._note(day, session, now)  # type: ignore[arg-type]
        if local >= times.notebook_at:
            actions += self._notebook(day, now)
        current = session_at(local, times)
        if current is not None:
            actions += self._snapshot(day, current, times, now)
        return actions

    # ---------- Snapshots ----------

    def _snapshot(self, day: date, session: Session, times: DayTimes, now: datetime) -> list[str]:
        opens, closes = {
            "pre": (times.pre_start, times.open),
            "regular": (times.open, times.close),
            "post": (times.close, times.post_end),
        }[session]
        record = self._store.load(day, session) or SessionRecord(
            day=day, session=session, opens=opens, closes=closes
        )
        every = timedelta(minutes=self._config.snapshot_minutes)
        if record.snapshots and now - record.snapshots[-1].at < every:
            return []
        errors: list[str] = []
        market_symbols = [s.upper() for s in self._config.market_symbols]
        focus: dict[str, list[str]] = {s: ["market"] for s in market_symbols}
        try:
            for symbol, reasons in self._focus(day).items():
                focus.setdefault(symbol.upper(), []).extend(r for r in reasons if r)
        except Exception as exc:
            errors.append(f"watch list unavailable: {exc}")
        report: WatchReport | None = None
        try:
            report = self._scan("post" if session == "post" else "pre", day, focus)
        except Exception as exc:
            errors.append(f"market scan failed: {exc}")

        # Sources: once per day each, fetched since the day's previous snapshot.
        day_records = [self._store.load(day, s) for s in SESSIONS if s != session]
        others = [r for r in day_records if r is not None]
        seen = {i.id for r in [*others, record] for i in r.sources}
        last = max((s.at for r in [*others, record] for s in r.snapshots), default=None)
        since = times.pre_start - OVERNIGHT if last is None else last - OVERLAP
        items, source_errors = fetch_all(self._sources, since)
        record.add_sources(items, seen)
        if session == "regular":
            try:
                record.trades = self._trades(day)
            except Exception as exc:
                errors.append(f"trades unavailable: {exc}")

        moves = [_move(i) for i in (report.items if report else ())]
        record.snapshots.append(
            JournalSnapshot(
                at=now,
                market=tuple(m for m in moves if m.symbol in market_symbols),
                stocks=tuple(m for m in moves if m.symbol not in market_symbols),
                alerts=report.alerts if report else (),
                errors=tuple(errors + list(report.errors if report else ()) + source_errors),
            )
        )
        self._store.save(record)
        return [f"snapshot:{session}"]

    # ---------- Notes and the notebook ----------

    def _can_try(self, key: tuple[date, str], now: datetime) -> bool:
        tries, next_at = self._attempts.get(key, (0, now))
        return tries < MAX_ATTEMPTS and now >= next_at

    def _failed(self, key: tuple[date, str], now: datetime) -> None:
        tries, _ = self._attempts.get(key, (0, now))
        self._attempts[key] = (tries + 1, now + RETRY_AFTER)

    def _note(self, day: date, session: Session, now: datetime) -> list[str]:
        key = (day, f"note:{session}")
        if self._writer is None or not self._config.ai_notes or key in self._done:
            return []
        if not self._can_try(key, now):
            return []
        record = self._store.load(day, session)
        if record is None or not record.snapshots or record.note is not None:
            self._done.add(key)
            return []
        try:
            record.note = self._writer.session_note(record)
        except JournalError as exc:
            self._failed(key, now)
            record.errors.append(f"{session} note failed: {exc}")
            self._store.save(record)
            self._log.warning("journal.note_failed", session=session, error=str(exc))
            return []
        self._store.save(record)
        self._done.add(key)
        return [f"note:{session}"]

    def _notebook(self, day: date, now: datetime) -> list[str]:
        key = (day, "notebook")
        if key in self._done:
            return []
        if self._store.load_notebook(day) is not None:
            self._done.add(key)
            return []
        records = {
            s: r for s in SESSIONS if (r := self._store.load(day, s)) is not None and r.snapshots
        }
        self._done.add(key)
        if not records:
            return []
        facts = day_facts(day, records)
        headline = facts_headline(day, facts, (self._config.market_symbols or ("SPY",))[0])
        notebook = DayNotebook(day=day, headline=headline, facts=facts, created_at=now)
        if self._writer is not None and self._config.ai_notes:
            try:
                notebook = self._writer.notebook(day, records, facts)
            except JournalError as exc:
                notebook = notebook.model_copy(update={"note": f"AI story unavailable: {exc}"})
                self._log.warning("journal.notebook_ai_failed", error=str(exc))
        self._store.save_notebook(notebook)
        self._log.info("journal.notebook_saved", day=str(day), ai=notebook.ai)
        return ["notebook"]

    # ---------- Background thread ----------

    def run(self, stop: threading.Event, tick_seconds: float = 60) -> None:
        while not stop.is_set():
            try:
                for action in self.step():
                    self._log.info("journal.step", action=action)
            except Exception as exc:  # the journal must never take the bot down
                self._log.warning("journal.step_failed", error=f"{type(exc).__name__}: {exc}")
            stop.wait(tick_seconds)

    def start(self) -> threading.Event:
        stop = threading.Event()
        threading.Thread(target=self.run, args=(stop,), name="market-journal", daemon=True).start()
        self._log.info("journal.started")
        return stop
