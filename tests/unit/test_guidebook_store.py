"""Storage for the guidebook (ADR 0017): principles, playbooks, versions, and usage."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_report import DAY, GENERATED, PROPOSAL, RUN, _ny, _record_trading_day
from trading_agent.agents.analyst import AnalystResult
from trading_agent.domain.models import LLMUsage, Playbook
from trading_agent.settings.models import AppConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import NewLessonRow, NullAuditStore, SqlAuditStore


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: GENERATED)


def _lesson(store: SqlAuditStore, text: str, *, days: int = 60) -> int:
    [lesson_id] = store.record_lessons(
        None,
        [NewLessonRow(text=text, category="entry", evidence={})],
        expires_at=GENERATED + timedelta(days=days),
        max_active=3,
    )
    return lesson_id


def playbook(name: str = "earnings-gap", status: str = "candidate", **changes: Any) -> Playbook:
    values: dict[str, Any] = {
        "name": name,
        "title": "Earnings gap check",
        "when_to_use": "A stock gaps on earnings before the open.",
        "steps": ("Check the gap size with get_quote.", "Read the news with get_news."),
        "category": "entry",
        "status": status,
        "source_lesson_ids": (1, 2),
        "created_at": GENERATED,
    }
    values.update(changes)
    return Playbook(**values)


# ---------- Principles: promoted lessons ----------


def test_promoted_lessons_never_expire_or_get_pushed_out(store: SqlAuditStore) -> None:
    principle = _lesson(store, "Check volume first.", days=1)
    assert store.promote_lesson(principle)
    assert not store.promote_lesson(principle)  # already a principle
    for i in range(4):
        _lesson(store, f"Newer lesson {i}.")
    later = GENERATED + timedelta(days=5)
    active = store.active_lessons(later, 10)
    assert principle in [lesson.id for lesson in active]  # past its expiry and over capacity
    assert next(lesson for lesson in active if lesson.id == principle).promoted


def test_a_principle_can_still_be_retired(store: SqlAuditStore) -> None:
    principle = _lesson(store, "Check volume first.")
    store.promote_lesson(principle)
    assert store.retire_lesson(principle, "outcomes turned negative")
    assert principle not in [lesson.id for lesson in store.active_lessons(GENERATED, 10)]


# ---------- Playbooks ----------


def test_playbooks_round_trip_and_change_status(store: SqlAuditStore) -> None:
    pid = store.record_playbook(playbook())
    assert pid > 0
    [found] = store.list_playbooks(include_retired=False)
    assert found.id == pid
    assert found.steps == ("Check the gap size with get_quote.", "Read the news with get_news.")
    assert found.source_lesson_ids == (1, 2)
    assert store.set_playbook_status("earnings-gap", "active", "5 trades, +1.2%")
    assert store.list_playbooks(include_retired=False)[0].status == "active"
    assert store.set_playbook_status("earnings-gap", "retired", "mean -2%")
    assert store.list_playbooks(include_retired=False) == []
    assert store.list_playbooks(include_retired=True)[0].retired_reason == "mean -2%"
    assert not store.set_playbook_status("missing", "active", "x")


def test_playbook_names_are_unique(store: SqlAuditStore) -> None:
    from trading_agent.storage.store import AuditError

    store.record_playbook(playbook())
    with pytest.raises(AuditError, match="playbook"):
        store.record_playbook(playbook())


def test_playbook_names_and_sizes_are_validated() -> None:
    with pytest.raises(ValueError, match="name"):
        playbook(name="Bad Name!")
    with pytest.raises(ValueError, match="steps"):
        playbook(steps=())
    with pytest.raises(ValueError, match="steps"):
        playbook(steps=tuple(f"step {i}" for i in range(7)))


# ---------- Guidebook versions ----------


def test_guidebook_versions(store: SqlAuditStore) -> None:
    assert store.latest_guidebook() is None
    assert store.record_guidebook("# v1", ["first edition"]) == 1
    assert store.record_guidebook("# v2", ["promoted lesson #3"]) == 2
    latest = store.latest_guidebook()
    assert latest is not None
    assert (latest["version"], latest["content"], latest["changes"]) == (
        2,
        "# v2",
        ["promoted lesson #3"],
    )
    first = store.guidebook(1)
    assert first is not None
    assert first["content"] == "# v1"
    assert store.guidebook(9) is None


# ---------- What each analysis knew ----------


def test_analysis_contexts_carry_lessons_and_playbook(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    slot = f"{RUN}-2"
    store.start_run(slot, "daily")
    store.record_analysis(
        AnalystResult(
            run_id=slot,
            status="proposed",
            reason="ok",
            proposal=PROPOSAL.model_copy(update={"playbook": "earnings-gap"}),
            prompt_version="p",
            turns=1,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=_ny(10, 30),
            duration_seconds=1,
            lesson_ids=(4, 5),
        )
    )
    contexts = {c["run_id"]: c for c in store.analysis_contexts(_ny(0), _ny(23))}
    assert contexts[RUN]["playbook"] is None
    assert contexts[RUN]["lesson_ids"] == []
    assert contexts[slot]["playbook"] == "earnings-gap"
    assert contexts[slot]["lesson_ids"] == [4, 5]


def test_null_store_has_no_guidebook() -> None:
    store = NullAuditStore()
    assert not store.promote_lesson(1)
    assert store.record_playbook(playbook()) == 0
    assert store.list_playbooks(include_retired=True) == []
    assert not store.set_playbook_status("x", "active", "")
    assert store.record_guidebook("x", []) == 0
    assert store.latest_guidebook() is None
    assert store.guidebook(1) is None
    assert store.analysis_contexts(GENERATED, GENERATED) == []
    assert Decimal(0) == 0  # keep the import honest
    assert DAY
