"""The all-day market journal and the day notebook (ADR 0023). Written first.

Timeline on a normal day (New York): pre-market 04:00-09:30, regular 09:30-16:00,
post-market 16:00-20:00, notebook at 20:20. Snapshots every 30 minutes while a session is
open, an AI note when each session ends, and a notebook that connects the dots. It only
watches and records: nothing here can trade.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FakeBroker
from trading_agent.domain.models import LLMRequest, LLMResponse, LLMUsage
from trading_agent.journal.models import Connection, SessionRecord
from trading_agent.journal.recorder import JournalRecorder, day_times, session_at
from trading_agent.journal.sources import SourceItem
from trading_agent.journal.store import JournalStore
from trading_agent.journal.writer import JournalError, JournalWriter
from trading_agent.llm.errors import LLMError
from trading_agent.settings.models import JournalConfig
from trading_agent.watch.market_watch import WatchItem, WatchReport

NY = ZoneInfo("America/New_York")
MONDAY = date(2026, 10, 19)


def at(hour: int, minute: int = 0, day: date = MONDAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=NY)


# ---------- Fakes ----------


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _item(symbol: str, move: str | None, reasons: tuple[str, ...] = ("plan",)) -> WatchItem:
    return WatchItem(
        symbol=symbol,
        reasons=reasons,
        last_price=Decimal("100"),
        last_trade_at=None,
        reference_close=Decimal("99"),
        move_pct=None if move is None else Decimal(move),
        stale=move is None,
        news_count=1,
        catalysts=("earnings",) if symbol == "NVDA" else (),
        headlines=(),
        alert=symbol == "NVDA",
    )


class FakeScan:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date, dict[str, list[str]]]] = []
        self.fail = False

    def __call__(self, phase: Any, day: date, focus: Any) -> WatchReport:
        self.calls.append((phase, day, dict(focus)))
        if self.fail:
            raise RuntimeError("snapshots down")
        items = [_item(s, "0.40", tuple(r)) for s, r in focus.items()]
        items.append(_item("NVDA", "4.20", ("gainer",)))
        return WatchReport(
            phase=phase,
            day=day,
            created_at=at(4),
            market=None,
            items=tuple(items),
            alerts=("NVDA +4.20% pre-market (gainer); earnings news (1 article)",),
            errors=(),
        )


def _source(item_id: str, title: str, kind: str = "company_news") -> SourceItem:
    return SourceItem(
        id=item_id,
        source="benzinga" if kind == "company_news" else "federal_register",
        kind=kind,  # type: ignore[arg-type]
        title=title,
        summary="Summary text.",
        url=f"https://example.com/{item_id}",
        published_at=at(7, 15),
    )


class FakeSource:
    name = "fake"

    def __init__(self, *batches: list[SourceItem]) -> None:
        self.batches = list(batches)
        self.since: list[datetime] = []

    def fetch(self, since: datetime) -> list[SourceItem]:
        self.since.append(since)
        return self.batches.pop(0) if self.batches else []


NOTE_JSON = json.dumps(
    {
        "summary": "Chip stocks led the pre-market after an export-policy change.",
        "connections": [
            {
                "effect": "NVDA +4.2% pre-market",
                "cause": "A presidential document eased chip export limits",
                "source_ids": ["federal_register:2026-21432", "news:999-invented"],
                "confidence": "medium",
            },
            {
                "effect": "SPY flat",
                "cause": "No broad news",
                "source_ids": [],
                "confidence": "low",
            },
        ],
        "watch_next": ["Does NVDA hold its gap at the open?"],
    }
)

NOTEBOOK_JSON = json.dumps(
    {
        "headline": "Chips rallied on an export-policy change; the bot made one trade.",
        "story": "Pre-market, NVDA gapped up after the policy news. It held all day.",
        "connections": [
            {
                "effect": "NVDA +4.2% pre-market, +5.1% at the close",
                "cause": "Eased chip export limits",
                "source_ids": ["federal_register:2026-21432"],
                "confidence": "medium",
            }
        ],
        "lessons": ["Policy news before the open moved chips all day."],
        "watch_tomorrow": ["NVDA follow-through"],
        "open_questions": ["Was the move priced in by the close?"],
    }
)


class FakeAI:
    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[LLMRequest] = []

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else NOTE_JSON
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome,
            provider="fake",
            model="fake-model",
            stop_reason="end",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
        )


def _recorder(
    tmp_path: Path,
    clock: Clock,
    *,
    ai: FakeAI | None = None,
    sources: list[Any] | None = None,
    broker: FakeBroker | None = None,
    scan: FakeScan | None = None,
    **config: Any,
) -> tuple[JournalRecorder, JournalStore, FakeScan]:
    store = JournalStore(tmp_path / "journal")
    scan = scan or FakeScan()
    broker = broker or FakeBroker()
    writer = None if ai is None else JournalWriter(ai, clock=clock)
    recorder = JournalRecorder(
        store,
        scan=scan,
        sources=sources if sources is not None else [FakeSource()],
        writer=writer,
        market_day=broker.get_market_day,
        focus=lambda day: {"AAPL": ["held"]},
        trades=lambda day: [{"symbol": "AAPL", "pnl_usd": "0.54", "closed": True}],
        config=JournalConfig.model_validate(config),
        timezone="America/New_York",
        clock=clock,
    )
    return recorder, store, scan


def _run_until(
    recorder: JournalRecorder, clock: Clock, end: datetime, step_minutes: int = 1
) -> list[str]:
    actions: list[str] = []
    while clock.now < end:
        actions += recorder.step()
        clock.now += timedelta(minutes=step_minutes)
    return actions


# ---------- The day's times ----------


def test_a_normal_day() -> None:
    times = day_times(FakeBroker().get_market_day(MONDAY), JournalConfig(), NY)  # type: ignore[arg-type]
    assert (times.pre_start, times.open, times.close, times.post_end, times.notebook_at) == (
        at(4),
        at(9, 30),
        at(16),
        at(20),
        at(20, 20),
    )
    assert session_at(at(3, 59), times) is None
    assert session_at(at(4), times) == "pre"
    assert session_at(at(9, 30), times) == "regular"
    assert session_at(at(16), times) == "post"
    assert session_at(at(20), times) is None


def test_an_early_close_ends_the_post_market_four_hours_later() -> None:
    friday = date(2026, 11, 27)
    broker = FakeBroker(market_days={friday: "13:00"})
    times = day_times(broker.get_market_day(friday), JournalConfig(), NY)  # type: ignore[arg-type]
    assert times.close == at(13, day=friday)
    assert times.post_end == at(17, day=friday)
    assert times.notebook_at == at(17, 20, day=friday)


# ---------- Recording ----------


def test_snapshots_every_30_minutes_in_each_session(tmp_path: Path) -> None:
    clock = Clock(at(3, 50))
    recorder, store, scan = _recorder(tmp_path, clock)
    actions = _run_until(recorder, clock, at(20, 10))
    pre = store.load(MONDAY, "pre")
    regular = store.load(MONDAY, "regular")
    post = store.load(MONDAY, "post")
    assert pre is not None
    assert regular is not None
    assert post is not None
    assert len(pre.snapshots) == 11  # 04:00, 04:30, ... 09:00
    assert len(regular.snapshots) == 13  # 09:30 ... 15:30
    assert len(post.snapshots) == 8  # 16:00 ... 19:30
    assert pre.snapshots[0].at == at(4)
    assert actions.count("snapshot:pre") == 11
    # Phases: pre-market and regular measure from the last close, post from today's close.
    assert {c[0] for c in scan.calls[:11]} == {"pre"}
    assert {c[0] for c in scan.calls[-8:]} == {"post"}


def test_the_market_indexes_and_the_bots_stocks_are_watched(tmp_path: Path) -> None:
    clock = Clock(at(4))
    recorder, store, scan = _recorder(tmp_path, clock)
    recorder.step()
    _, _, focus = scan.calls[0]
    assert focus["SPY"] == ["market"]
    assert set(focus) >= {"SPY", "QQQ", "IWM", "DIA", "AAPL"}
    snapshot = store.load(MONDAY, "pre").snapshots[0]  # type: ignore[union-attr]
    assert {m.symbol for m in snapshot.market} == {"SPY", "QQQ", "IWM", "DIA"}
    assert {m.symbol for m in snapshot.stocks} == {"AAPL", "NVDA"}
    assert snapshot.alerts


def test_nothing_is_recorded_on_weekends_or_holidays(tmp_path: Path) -> None:
    thanksgiving = date(2026, 11, 26)
    broker = FakeBroker(market_days={thanksgiving: None})
    for day in (date(2026, 10, 17), thanksgiving):
        clock = Clock(at(4, day=day))
        recorder, _, scan = _recorder(tmp_path, clock, broker=broker)
        assert _run_until(recorder, clock, at(21, day=day), step_minutes=10) == []
        assert scan.calls == []
        assert not (tmp_path / "journal" / day.isoformat()).exists()


def test_sources_are_kept_once_with_the_session_they_arrived_in(tmp_path: Path) -> None:
    policy = _source("federal_register:2026-21432", "Adjusting chip exports", "government")
    news = _source("news:1", "NVDA jumps")
    source = FakeSource([policy, news], [news], [])
    clock = Clock(at(4))
    recorder, store, _ = _recorder(tmp_path, clock, sources=[source])
    recorder.step()
    clock.now = at(4, 30)
    recorder.step()
    pre = store.load(MONDAY, "pre")
    assert pre is not None
    assert [s.id for s in pre.sources] == ["federal_register:2026-21432", "news:1"]
    # The first fetch looks back overnight; later ones from the last snapshot.
    assert source.since[0] <= at(4) - timedelta(hours=8)
    assert source.since[1] >= at(4) - timedelta(minutes=5)


def test_the_regular_session_records_the_trades(tmp_path: Path) -> None:
    clock = Clock(at(10))
    recorder, store, _ = _recorder(tmp_path, clock)
    recorder.step()
    regular = store.load(MONDAY, "regular")
    assert regular is not None
    assert regular.trades == [{"symbol": "AAPL", "pnl_usd": "0.54", "closed": True}]
    assert store.load(MONDAY, "pre") is None  # started late: no pre-market record


def test_a_restart_picks_up_the_days_files(tmp_path: Path) -> None:
    clock = Clock(at(4))
    first, store, _ = _recorder(tmp_path, clock)
    first.step()
    clock.now = at(4, 10)
    again, _, scan = _recorder(tmp_path, clock)
    assert again.step() == []  # the 04:00 snapshot is on disk: next one at 04:30
    assert scan.calls == []
    clock.now = at(4, 30)
    assert again.step() == ["snapshot:pre"]
    assert len(store.load(MONDAY, "pre").snapshots) == 2  # type: ignore[union-attr]


def test_a_failing_scan_is_recorded_not_raised(tmp_path: Path) -> None:
    scan = FakeScan()
    scan.fail = True
    clock = Clock(at(4))
    recorder, store, _ = _recorder(tmp_path, clock, scan=scan)
    assert recorder.step() == ["snapshot:pre"]
    snapshot = store.load(MONDAY, "pre").snapshots[0]  # type: ignore[union-attr]
    assert any("snapshots down" in e for e in snapshot.errors)


def test_a_failing_calendar_skips_and_retries(tmp_path: Path) -> None:
    clock = Clock(at(4))
    recorder, _, scan = _recorder(tmp_path, clock, broker=FakeBroker(fail_clock=True))
    assert recorder.step() == []
    assert scan.calls == []


def test_the_journal_can_be_switched_off(tmp_path: Path) -> None:
    clock = Clock(at(4))
    recorder, _, _ = _recorder(tmp_path, clock, enabled=False)
    assert recorder.step() == []


# ---------- AI notes and the notebook ----------


def _with_policy_source() -> list[Any]:
    return [FakeSource([_source("federal_register:2026-21432", "Chip exports", "government")])]


def test_a_note_when_each_session_ends_and_a_notebook_at_the_end(tmp_path: Path) -> None:
    clock = Clock(at(3, 50))
    ai = FakeAI(NOTE_JSON, NOTE_JSON, NOTE_JSON, NOTEBOOK_JSON)
    recorder, store, _ = _recorder(tmp_path, clock, ai=ai, sources=_with_policy_source())
    actions = _run_until(recorder, clock, at(20, 30))
    assert [a for a in actions if not a.startswith("snapshot")] == [
        "note:pre",
        "note:regular",
        "note:post",
        "notebook",
    ]
    assert actions.index("note:pre") < actions.index("snapshot:regular")
    notebook = store.load_notebook(MONDAY)
    assert notebook is not None
    assert notebook.ai
    assert notebook.headline.startswith("Chips rallied")
    assert notebook.connections[0].verified
    assert all(r.tags == {"agent": "journal"} for r in ai.requests)


def test_only_real_sources_count_as_evidence(tmp_path: Path) -> None:
    clock = Clock(at(4))
    recorder, store, _ = _recorder(
        tmp_path, clock, ai=FakeAI(NOTE_JSON), sources=_with_policy_source()
    )
    _run_until(recorder, clock, at(9, 31))
    note = store.load(MONDAY, "pre").note  # type: ignore[union-attr]
    assert note is not None
    sourced, unsourced = note.connections
    assert sourced.source_ids == ("federal_register:2026-21432",)  # the invented id is dropped
    assert sourced.verified
    assert not unsourced.verified


def test_the_note_prompt_marks_sources_as_untrusted_and_asks_for_citations(tmp_path: Path) -> None:
    clock = Clock(at(4))
    ai = FakeAI(NOTE_JSON)
    recorder, _, _ = _recorder(tmp_path, clock, ai=ai, sources=_with_policy_source())
    _run_until(recorder, clock, at(9, 31))
    (request,) = ai.requests
    assert "untrusted" in request.system.lower()
    assert "source_ids" in request.system
    assert "not proof" in request.system
    content = request.messages[0].content
    assert "[federal_register:2026-21432]" in content
    assert "NVDA" in content
    assert "Pre-market" in content


def test_bad_ai_output_gets_one_repair(tmp_path: Path) -> None:
    clock = Clock(at(4))
    ai = FakeAI("not json at all", NOTE_JSON)
    recorder, store, _ = _recorder(tmp_path, clock, ai=ai, sources=_with_policy_source())
    _run_until(recorder, clock, at(9, 31))
    assert store.load(MONDAY, "pre").note is not None  # type: ignore[union-attr]
    assert len(ai.requests) == 2


def test_a_failed_note_is_recorded_and_retried_later_a_few_times(tmp_path: Path) -> None:
    clock = Clock(at(4))
    ai = FakeAI(*[LLMError("budget reached")] * 10)
    recorder, store, _ = _recorder(tmp_path, clock, ai=ai)
    _run_until(recorder, clock, at(12))
    pre = store.load(MONDAY, "pre")
    assert pre is not None
    assert pre.note is None
    assert any("budget reached" in e for e in pre.errors)
    assert len(ai.requests) == 3  # 09:30, then twice more 30 minutes apart, then it stops


def test_without_ai_the_notebook_still_has_the_facts(tmp_path: Path) -> None:
    clock = Clock(at(3, 50))
    recorder, store, _ = _recorder(tmp_path, clock, sources=_with_policy_source())
    actions = _run_until(recorder, clock, at(20, 30), step_minutes=10)
    assert "notebook" in actions
    assert not any(a.startswith("note:") for a in actions)
    notebook = store.load_notebook(MONDAY)
    assert notebook is not None
    assert not notebook.ai
    assert "SPY" in notebook.headline
    assert notebook.facts["sources"]["total"] == 1
    assert notebook.facts["trades"]["count"] == 1


def test_a_failed_ai_notebook_falls_back_to_the_facts(tmp_path: Path) -> None:
    clock = Clock(at(19))
    ai = FakeAI(NOTE_JSON, LLMError("down"), LLMError("down"))
    recorder, store, _ = _recorder(tmp_path, clock, ai=ai)
    _run_until(recorder, clock, at(20, 30))
    notebook = store.load_notebook(MONDAY)
    assert notebook is not None
    assert not notebook.ai
    assert notebook.note is not None
    assert "down" in notebook.note


def test_no_notebook_for_a_day_with_no_records(tmp_path: Path) -> None:
    clock = Clock(at(20, 25))
    recorder, store, _ = _recorder(tmp_path, clock, ai=FakeAI(NOTEBOOK_JSON))
    assert recorder.step() == []
    assert store.load_notebook(MONDAY) is None


# ---------- Files ----------


def test_files_for_people_and_for_machines(tmp_path: Path) -> None:
    clock = Clock(at(3, 50))
    ai = FakeAI(NOTE_JSON, NOTE_JSON, NOTE_JSON, NOTEBOOK_JSON)
    recorder, _, _ = _recorder(tmp_path, clock, ai=ai, sources=_with_policy_source())
    _run_until(recorder, clock, at(20, 30), step_minutes=5)
    folder = tmp_path / "journal" / "2026-10-19"
    names = sorted(p.name for p in folder.iterdir())
    assert names == [
        "day.json",
        "day.md",
        "market.json",
        "market.md",
        "postmarket.json",
        "postmarket.md",
        "premarket.json",
        "premarket.md",
    ]
    day_md = (folder / "day.md").read_text(encoding="utf-8")
    assert "# Day notebook: Mon 2026-10-19" in day_md
    assert "Connecting the dots" in day_md
    assert "federal_register:2026-21432" in day_md
    pre_md = (folder / "premarket.md").read_text(encoding="utf-8")
    assert "# Pre-market: Mon 2026-10-19" in pre_md
    assert "untrusted" in pre_md.lower()
    data = json.loads((folder / "premarket.json").read_text(encoding="utf-8"))
    assert data["session"] == "pre"
    assert data["snapshots"][0]["market"]


def test_a_damaged_file_starts_the_session_fresh(tmp_path: Path) -> None:
    store = JournalStore(tmp_path / "journal")
    folder = tmp_path / "journal" / "2026-10-19"
    folder.mkdir(parents=True)
    (folder / "premarket.json").write_text("{broken", encoding="utf-8")
    assert store.load(MONDAY, "pre") is None
    assert (folder / "premarket.json.damaged").exists()  # kept for a look, not lost


def test_writes_are_atomic(tmp_path: Path) -> None:
    store = JournalStore(tmp_path / "journal")
    record = SessionRecord(day=MONDAY, session="pre", opens=at(4), closes=at(9, 30))
    store.save(record)
    assert not list((tmp_path / "journal" / "2026-10-19").glob("*.tmp"))


# ---------- The writer on its own ----------


def test_the_writer_rejects_output_it_cannot_repair() -> None:
    writer = JournalWriter(FakeAI("nope", "still nope"), clock=Clock(at(9, 30)))
    record = SessionRecord(day=MONDAY, session="pre", opens=at(4), closes=at(9, 30))
    with pytest.raises(JournalError, match="invalid"):
        writer.session_note(record)


def test_connections_are_bounded() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Connection(effect="x", cause="y", source_ids=(), confidence="certain")  # type: ignore[arg-type]


# ---------- Background thread ----------


def test_the_thread_survives_a_crashing_step(tmp_path: Path) -> None:
    clock = Clock(at(4))
    recorder, _, _ = _recorder(tmp_path, clock)
    calls: list[int] = []
    stop = threading.Event()

    def broken_step() -> list[str]:
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
        raise RuntimeError("bug")

    recorder.step = broken_step  # type: ignore[method-assign]
    recorder.run(stop, tick_seconds=0)
    assert len(calls) == 2
