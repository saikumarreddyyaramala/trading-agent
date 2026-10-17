"""The curator: keeps the guidebook honest with the system's own results (ADR 0017).

Run once a day, after the evening review:

1. Lessons whose trades did measurably better (or worse) than trades without them are
   promoted to principles (or retired). Needs `guidebook.min_trades` on both sides.
2. Candidate playbooks with enough non-negative trades become active; any playbook whose
   trades fall below `guidebook.playbook_retire_below_pct` is retired.
3. When a category has enough lessons and no playbook, the writer (an LLM, guarded) drafts
   one, as a candidate.
4. The guidebook is rebuilt; a new version is stored only if its text changed.

Steps 1, 2, and 4 are deterministic. A failing writer never stops the rest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from trading_agent.agents.playbooks import PlaybookWriter
from trading_agent.domain.models import Lesson
from trading_agent.learning.guidebook import build_guidebook
from trading_agent.learning.outcomes import lesson_outcome, playbook_outcomes, trade_outcomes
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig
from trading_agent.storage.store import AuditStore

MAX_LESSONS = 50


class CurationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    promoted: list[int]
    retired_lessons: list[int]
    activated: list[str]
    retired_playbooks: list[str]
    written: str | None
    changes: list[str]
    version: int | None  # the new guidebook version, if the text changed
    summary: str


class Curator:
    def __init__(
        self,
        store: AuditStore,
        config: AppConfig,
        *,
        writer: PlaybookWriter | None = None,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._store = store
        self._config = config
        self._settings = config.guidebook
        self._zone = ZoneInfo(config.schedule.timezone)
        self._writer = writer
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def curate(self, day: date) -> CurationResult:
        settings = self._settings
        now = self._clock()
        trades = trade_outcomes(
            self._store,
            self._zone,
            now - timedelta(days=settings.history_days),
            now + timedelta(days=1),
        )
        changes: list[str] = []
        notes: list[str] = []

        # ---- Lessons ----
        promoted: list[int] = []
        retired_lessons: list[int] = []
        for lesson in self._store.active_lessons(now, MAX_LESSONS):
            outcome = lesson_outcome(lesson.id, trades)
            if (
                outcome.lift_pct is None
                or outcome.with_ is None
                or outcome.without is None
                or min(outcome.with_.n, outcome.without.n) < settings.min_trades
            ):
                continue
            evidence = f"with it {outcome.with_.describe()}, without {outcome.without.describe()}"
            if outcome.lift_pct <= -settings.retire_lift_pct:
                if self._store.retire_lesson(lesson.id, f"outcomes: {evidence}"):
                    retired_lessons.append(lesson.id)
                    kind = "principle" if lesson.promoted else "lesson"
                    changes.append(f"retired {kind} #{lesson.id} ({evidence}): {lesson.text}")
            elif (
                not lesson.promoted
                and outcome.lift_pct >= settings.promote_lift_pct
                and outcome.with_.mean_pct > 0
                and self._store.promote_lesson(lesson.id)
            ):
                promoted.append(lesson.id)
                changes.append(
                    f"promoted lesson #{lesson.id} to a principle ({evidence}): {lesson.text}"
                )

        # ---- Playbooks ----
        activated: list[str] = []
        retired_playbooks: list[str] = []
        results = playbook_outcomes(trades)
        for playbook in self._store.list_playbooks(include_retired=False):
            result = results.get(playbook.name)
            if result is None or result.n < settings.playbook_min_trades:
                continue
            if result.mean_pct < settings.playbook_retire_below_pct:
                if self._store.set_playbook_status(
                    playbook.name, "retired", f"outcomes: {result.describe()}"
                ):
                    retired_playbooks.append(playbook.name)
                    changes.append(f"retired playbook {playbook.name} ({result.describe()})")
            elif (
                playbook.status == "candidate"
                and result.mean_pct >= 0
                and self._store.set_playbook_status(
                    playbook.name, "active", f"outcomes: {result.describe()}"
                )
            ):
                activated.append(playbook.name)
                changes.append(f"playbook {playbook.name} is now active ({result.describe()})")

        # ---- Writing a new playbook ----
        written = None
        lessons = self._store.active_lessons(now, MAX_LESSONS)
        playbooks = self._store.list_playbooks(include_retired=False)
        if settings.write_playbooks and self._writer is not None:
            written, note = self._write(lessons, playbooks)
            if written:
                changes.append(f"new playbook {written} (candidate)")
                playbooks = self._store.list_playbooks(include_retired=False)
            if note:
                notes.append(note)

        # ---- The guidebook ----
        content = build_guidebook(
            self._config,
            lessons,
            playbooks,
            trades,
            {lesson.id: lesson_outcome(lesson.id, trades) for lesson in lessons},
            playbook_outcomes(trades),
        )
        latest = self._store.latest_guidebook()
        version = None
        if latest is None or latest["content"] != content:
            recorded = changes or (["first edition"] if latest is None else ["record updated"])
            version = self._store.record_guidebook(content, recorded) or None
            changes = recorded

        parts = list(changes) if version or changes else ["no changes"]
        if version:
            parts.append(f"guidebook v{version}")
        summary = "; ".join(parts + notes)
        self._log.info("curation.done", day=str(day), version=version, changes=len(changes))
        return CurationResult(
            promoted=promoted,
            retired_lessons=retired_lessons,
            activated=activated,
            retired_playbooks=retired_playbooks,
            written=written,
            changes=changes,
            version=version,
            summary=summary,
        )

    def _write(self, lessons: list[Lesson], playbooks: list[Any]) -> tuple[str | None, str | None]:
        if self._writer is None or len(playbooks) >= self._settings.max_playbooks:
            return None, None
        covered = {p.category for p in playbooks}
        grouped: dict[str, list[Lesson]] = defaultdict(list)
        for lesson in lessons:
            grouped[lesson.category].append(lesson)
        ready = sorted(
            (
                (category, group)
                for category, group in grouped.items()
                if category not in covered and len(group) >= self._settings.min_lessons_for_playbook
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
        if not ready:
            return None, None
        category, group = ready[0]
        result = self._writer.write(
            category, group, self._store.list_playbooks(include_retired=True)
        )
        if result.status != "ok" or result.playbook is None:
            return None, f"playbook not written: {result.reason}"
        self._store.record_playbook(result.playbook)
        return result.playbook.name, None
