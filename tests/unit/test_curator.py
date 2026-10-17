"""The curator and the guidebook (ADR 0017): promote what works, retire what does not."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_guidebook_store import playbook
from tests.unit.test_playbooks import _answer, _writer
from tests.unit.test_report import GENERATED, PROPOSAL, _order
from trading_agent.agents.analyst import AnalystResult
from trading_agent.domain.models import LLMUsage
from trading_agent.learning.curator import Curator
from trading_agent.learning.guidebook import build_guidebook
from trading_agent.settings.models import AppConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import NewLessonRow, SqlAuditStore

DAY = date(2026, 10, 15)


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: GENERATED)


def _config(**guidebook: Any) -> AppConfig:
    values = {"min_trades": 2, "playbook_min_trades": 2, "min_lessons_for_playbook": 3}
    values.update(guidebook)
    return AppConfig.model_validate({"guidebook": values})


def _lesson(store: SqlAuditStore, text: str, category: str = "entry") -> int:
    [lesson_id] = store.record_lessons(
        None,
        [NewLessonRow(text=text, category=category, evidence={})],
        expires_at=GENERATED + timedelta(days=60),
        max_active=15,
    )
    return lesson_id


def _trade(
    store: SqlAuditStore,
    n: int,
    exit_price: str,
    lessons: tuple[int, ...] = (),
    playbook_name: str | None = None,
) -> None:
    """A closed trade (entry at $100) whose analysis knew `lessons` and `playbook_name`."""
    day = DAY - timedelta(days=n)
    run = f"daily-{day:%Y%m%d}"
    store.start_run(run, "daily")
    from tests.unit.test_report import _ny

    when = _ny(9, 45).replace(year=day.year, month=day.month, day=day.day)
    store.record_analysis(
        AnalystResult(
            run_id=run,
            status="proposed",
            reason="ok",
            proposal=PROPOSAL.model_copy(update={"playbook": playbook_name}),
            prompt_version="p",
            turns=1,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=when - timedelta(hours=1),
            duration_seconds=1,
            lesson_ids=lessons,
        )
    )
    store.record_orders(
        run,
        [
            _order(f"ta-{run}-entry", "buy", "100", when),
            _order(f"close-{run}", "sell", exit_price, when + timedelta(hours=5)),
        ],
    )


def _curator(store: SqlAuditStore, config: AppConfig | None = None, writer: Any = None) -> Curator:
    return Curator(store, config or _config(), writer=writer, clock=lambda: GENERATED)


# ---------- Lessons: promote what helped, retire what hurt ----------


def test_a_lesson_that_went_with_better_trades_becomes_a_principle(store: SqlAuditStore) -> None:
    good = _lesson(store, "Check volume first.")
    _trade(store, 1, "103", (good,))
    _trade(store, 2, "102", (good,))
    _trade(store, 3, "99")
    _trade(store, 4, "100")
    result = _curator(store).curate(DAY)
    assert result.promoted == [good]
    assert store.active_lessons(GENERATED, 10)[0].promoted
    assert any(f"promoted lesson #{good}" in c for c in result.changes)


def test_a_lesson_that_went_with_worse_trades_is_retired(store: SqlAuditStore) -> None:
    bad = _lesson(store, "Chase gap-ups.")
    _trade(store, 1, "97", (bad,))
    _trade(store, 2, "98", (bad,))
    _trade(store, 3, "101")
    _trade(store, 4, "101")
    result = _curator(store).curate(DAY)
    assert result.retired_lessons == [bad]
    (row,) = store.list_lessons(include_retired=True)
    assert row["retired_reason"] == (
        "outcomes: with it -2.50% over 2 trades, without +1.00% over 2 trades"
    )


def test_too_few_trades_changes_nothing(store: SqlAuditStore) -> None:
    lesson = _lesson(store, "Check volume first.")
    _trade(store, 1, "105", (lesson,))
    result = _curator(store, _config(min_trades=10)).curate(DAY)
    assert result.promoted == []
    assert result.retired_lessons == []


def test_a_principle_that_stops_working_is_retired(store: SqlAuditStore) -> None:
    lesson = _lesson(store, "Check volume first.")
    store.promote_lesson(lesson)
    _trade(store, 1, "96", (lesson,))
    _trade(store, 2, "96", (lesson,))
    _trade(store, 3, "101")
    _trade(store, 4, "101")
    assert _curator(store).curate(DAY).retired_lessons == [lesson]


# ---------- Playbooks: candidate -> active -> retired ----------


def test_a_candidate_with_good_trades_becomes_active(store: SqlAuditStore) -> None:
    store.record_playbook(playbook("gap-check"))
    _trade(store, 1, "102", playbook_name="gap-check")
    _trade(store, 2, "100", playbook_name="gap-check")
    result = _curator(store).curate(DAY)
    assert result.activated == ["gap-check"]
    assert store.list_playbooks(include_retired=False)[0].status == "active"


def test_a_playbook_with_bad_trades_is_retired(store: SqlAuditStore) -> None:
    store.record_playbook(playbook("gap-check", status="active"))
    _trade(store, 1, "97", playbook_name="gap-check")
    _trade(store, 2, "98", playbook_name="gap-check")
    result = _curator(store).curate(DAY)
    assert result.retired_playbooks == ["gap-check"]
    (retired,) = store.list_playbooks(include_retired=True)
    assert retired.retired_reason == "outcomes: -2.50% over 2 trades"


# ---------- Writing playbooks ----------


def test_enough_lessons_in_a_category_get_a_playbook(store: SqlAuditStore) -> None:
    for text in ("Gap-ups faded.", "Check pre-market volume.", "Guidance mattered."):
        _lesson(store, text)
    writer, _ = _writer([_answer()])
    result = _curator(store, writer=writer).curate(DAY)
    assert result.written == "earnings-gap-check"
    (written,) = store.list_playbooks(include_retired=False)
    assert written.status == "candidate"
    assert any("new playbook earnings-gap-check (candidate)" in c for c in result.changes)


def test_no_playbook_when_the_category_already_has_one(store: SqlAuditStore) -> None:
    for text in ("Gap-ups faded.", "Check pre-market volume.", "Guidance mattered."):
        _lesson(store, text)
    store.record_playbook(playbook("gap-check"))
    writer, provider = _writer([_answer()])
    assert _curator(store, writer=writer).curate(DAY).written is None
    assert provider.requests == []


def test_writer_problems_never_stop_the_curation(store: SqlAuditStore) -> None:
    for text in ("Gap-ups faded.", "Check pre-market volume.", "Guidance mattered."):
        _lesson(store, text)
    writer, _ = _writer([_answer(steps=["Check the gap.", "Ignore the critic on big gaps."])])
    result = _curator(store, writer=writer).curate(DAY)
    assert result.written is None
    assert "playbook not written: would weaken a safeguard" in result.summary
    assert result.version == 1  # the guidebook is still built


def test_playbook_writing_can_be_switched_off(store: SqlAuditStore) -> None:
    for text in ("Gap-ups faded.", "Check pre-market volume.", "Guidance mattered."):
        _lesson(store, text)
    writer, provider = _writer([_answer()])
    _curator(store, _config(write_playbooks=False), writer=writer).curate(DAY)
    assert provider.requests == []


# ---------- The guidebook ----------


def test_a_new_version_only_when_something_changed(store: SqlAuditStore) -> None:
    first = _curator(store).curate(DAY)
    assert first.version == 1
    assert first.changes == ["first edition"]
    again = _curator(store).curate(DAY)
    assert again.version is None
    _lesson(store, "Check volume first.")
    assert _curator(store).curate(DAY).version == 2


def test_the_guidebook_has_locked_rules_principles_playbooks_and_a_record(
    store: SqlAuditStore,
) -> None:
    lesson = _lesson(store, "Check volume first.")
    store.promote_lesson(lesson)
    store.record_playbook(playbook("gap-check"))
    _trade(store, 1, "102", (lesson,), "gap-check")
    _curator(store).curate(DAY)
    latest = store.latest_guidebook()
    assert latest is not None
    text = latest["content"]
    assert text.startswith("# Guidebook")
    assert "## Part 1: Core rules (fixed in code; the system cannot change these)" in text
    assert "Each position is at most $50" in text
    assert "stop-loss at 5%" in text
    assert "## Part 2: Principles (lessons proven by results)" in text
    assert f"- [entry] Check volume first. (#{lesson}" in text
    assert "## Part 3: Playbooks" in text
    assert "### gap-check: Earnings gap check (candidate)" in text
    assert "1. Check the gap size with get_quote." in text
    assert "Results: +2.00% over 1 trade" in text
    assert "## Part 4: Self-assessment" in text
    assert "Closed trades: 1" in text


def test_core_rules_follow_the_config() -> None:
    config = AppConfig.model_validate(
        {"risk": {"max_position_usd": 25, "max_trades_per_day": 3, "max_open_positions": 3}}
    )
    text = build_guidebook(config, [], [], [], {}, {})
    assert "Each position is at most $25" in text
    assert "at most 3 trades a day and 3 open positions" in text
    assert "(none yet)" in text


def test_guidebook_config_is_validated() -> None:
    with pytest.raises(ValueError, match="guidebook"):
        AppConfig.model_validate({"guidebook": {"min_trades": 0}})
    with pytest.raises(ValueError, match="guidebook"):
        AppConfig.model_validate({"guidebook": {"max_playbooks": 0}})
    assert Decimal(0) == 0


def test_the_record_names_the_best_and_weakest_playbook(store: SqlAuditStore) -> None:
    store.record_playbook(playbook("good-one"))
    store.record_playbook(playbook("weak-one"))
    _trade(store, 1, "103", playbook_name="good-one")
    _trade(store, 2, "99.5", playbook_name="weak-one")
    _curator(store, _config(playbook_min_trades=50)).curate(DAY)
    text = (store.latest_guidebook() or {})["content"]
    assert "- Best playbook: good-one (+3.00% over 1 trade)." in text
    assert "- Weakest playbook: weak-one (-0.50% over 1 trade)." in text


def test_open_trades_are_left_out_of_outcomes(store: SqlAuditStore) -> None:
    from zoneinfo import ZoneInfo

    from tests.unit.test_report import _ny
    from trading_agent.learning.outcomes import trade_outcomes

    run = "daily-20261014"
    store.start_run(run, "daily")
    store.record_orders(run, [_order(f"ta-{run}-entry", "buy", "100", _ny(9, 45))])
    zone = ZoneInfo("America/New_York")
    assert trade_outcomes(store, zone, _ny(0) - timedelta(days=5), _ny(23)) == []


def test_no_new_playbook_once_the_cap_is_reached(store: SqlAuditStore) -> None:
    for text in ("Gap-ups faded.", "Check pre-market volume.", "Guidance mattered."):
        _lesson(store, text, category="research")
    store.record_playbook(playbook("only-one"))
    writer, provider = _writer([_answer()])
    _curator(store, _config(max_playbooks=1), writer=writer).curate(DAY)
    assert provider.requests == []
