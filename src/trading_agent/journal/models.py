"""What the journal records (ADR 0023). Everything is JSON-ready and has a Markdown twin."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trading_agent.journal.sources import SourceItem

Session = Literal["pre", "regular", "post"]
SESSIONS: tuple[Session, ...] = ("pre", "regular", "post")
FILE_NAMES: dict[Session, str] = {"pre": "premarket", "regular": "market", "post": "postmarket"}
TITLES: dict[Session, str] = {
    "pre": "Pre-market",
    "regular": "Regular market",
    "post": "Post-market",
}
Confidence = Literal["low", "medium", "high"]


class Move(BaseModel):
    """One symbol at one moment: its last price against the close it is measured from."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    reasons: tuple[str, ...] = ()
    price: Decimal | None = None
    reference: Decimal | None = None
    move_pct: Decimal | None = None
    stale: bool = False
    catalysts: tuple[str, ...] = ()
    alert: bool = False


class JournalSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    at: datetime
    market: tuple[Move, ...] = ()
    stocks: tuple[Move, ...] = ()
    alerts: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class Connection(BaseModel):
    """One dot connected: an effect, its likely cause, and the sources that back it.
    `verified` means at least one cited source really is in the record (code checks it)."""

    model_config = ConfigDict(frozen=True)

    effect: str = Field(min_length=1, max_length=300)
    cause: str = Field(min_length=1, max_length=400)
    source_ids: tuple[str, ...] = Field(default=(), max_length=5)
    confidence: Confidence
    verified: bool = False


class SessionNote(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    connections: tuple[Connection, ...] = ()
    watch_next: tuple[str, ...] = ()
    model: str | None = None
    created_at: datetime


class SessionRecord(BaseModel):
    """Everything recorded in one session of one day."""

    day: date
    session: Session
    opens: datetime
    closes: datetime
    snapshots: list[JournalSnapshot] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    note: SessionNote | None = None
    errors: list[str] = Field(default_factory=list)

    def add_sources(self, items: list[SourceItem], seen: set[str]) -> int:
        """Keep each item once across the day (`seen` holds ids already recorded)."""
        added = 0
        for item in items:
            if item.id not in seen:
                seen.add(item.id)
                self.sources.append(item)
                added += 1
        return added


class DayNotebook(BaseModel):
    """The day, from the start of pre-market to the end of post-market, with the dots
    connected. Without the AI it still holds the facts."""

    day: date
    headline: str
    story: str = ""
    connections: tuple[Connection, ...] = ()
    lessons: tuple[str, ...] = ()
    watch_tomorrow: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    facts: dict[str, Any] = Field(default_factory=dict)
    ai: bool = False
    model: str | None = None
    note: str | None = None  # why the AI story is missing, if it is
    created_at: datetime
