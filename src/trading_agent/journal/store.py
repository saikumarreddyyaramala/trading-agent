"""The journal on disk (ADR 0023): one folder per day, JSON for machines, Markdown for people.

    journal/2026-10-19/premarket.json + premarket.md
                       market.json    + market.md
                       postmarket.json + postmarket.md
                       day.json       + day.md        (the notebook)

Every write goes to a temporary file first and then replaces the old one, so a crash or a
power cut never leaves half a file. A damaged file is set aside, not deleted.
"""

from __future__ import annotations

import contextlib
import os
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

from trading_agent.journal.models import FILE_NAMES, DayNotebook, Session, SessionRecord
from trading_agent.journal.render import render_notebook, render_session
from trading_agent.observability.logging import get_logger

_LOG = get_logger(__name__)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class JournalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def folder(self, day: date) -> Path:
        return self.root / day.isoformat()

    def _read[T: BaseModel](self, path: Path, model: type[T]) -> T | None:
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            _LOG.warning("journal.damaged_file", path=str(path), error=type(exc).__name__)
            with contextlib.suppress(OSError):
                os.replace(path, path.with_name(path.name + ".damaged"))
            return None

    def load(self, day: date, session: Session) -> SessionRecord | None:
        return self._read(self.folder(day) / f"{FILE_NAMES[session]}.json", SessionRecord)

    def save(self, record: SessionRecord) -> None:
        base = self.folder(record.day) / FILE_NAMES[record.session]
        _write(base.with_suffix(".json"), record.model_dump_json(indent=2))
        _write(base.with_suffix(".md"), render_session(record))

    def load_notebook(self, day: date) -> DayNotebook | None:
        return self._read(self.folder(day) / "day.json", DayNotebook)

    def save_notebook(self, notebook: DayNotebook) -> None:
        base = self.folder(notebook.day) / "day"
        _write(base.with_suffix(".json"), notebook.model_dump_json(indent=2))
        _write(base.with_suffix(".md"), render_notebook(notebook))
