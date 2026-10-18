"""Pause new trades for the rest of a day, from chat (ADR 0022).

A small JSON file holds the paused day. It only ever makes the bot safer: it blocks new
analyses and entries, never exits, and it ends by itself when the day changes. A damaged
or unreadable file means "not paused" (the kill switch in the config is the hard control).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path


class PauseStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict[str, str] | None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def paused_reason(self, day: date) -> str | None:
        data = self._read()
        if data is None or data.get("day") != day.isoformat():
            return None
        return f"paused from {data.get('by', 'chat')} for {day.isoformat()}"

    def pause(self, day: date, *, by: str) -> None:
        """Raises OSError if the file cannot be written."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"day": day.isoformat(), "by": by, "at": datetime.now(UTC).isoformat()}
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    def resume(self) -> bool:
        """Clear any pause. True if there was one."""
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        return True
