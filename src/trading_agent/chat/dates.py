"""Days and periods in plain words: "last wednesday", "yesterday", "oct 12", "this week".

Fixed rules, no AI. A weekday name means the most recent such day (today counts);
"last <weekday>" means the one before today. A month and day without a year that would be
in the future means last year. Numeric dates are ISO (2026-10-12) or US style (10/12).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_US = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b")
_WEEKDAY = re.compile(rf"\b(last\s+|this\s+|past\s+)?({'|'.join(WEEKDAYS)})\b")
_LAST_N = re.compile(r"\b(?:last|past|previous)\s+(\d{1,3})\s+days?\b")


def _safe(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _recent(month: int, day: int, today: date) -> date | None:
    found = _safe(today.year, month, day)
    if found is not None and found > today:
        found = _safe(today.year - 1, month, day)
    return found


def parse_day(text: str, today: date) -> date | None:
    """The one day `text` refers to, or None."""
    lowered = text.lower()
    if match := _ISO.search(lowered):
        return _safe(*(int(g) for g in match.groups()))
    if re.search(r"\btoday\b", lowered):
        return today
    if re.search(r"\byesterday\b", lowered):
        return today - timedelta(days=1)
    if match := _WEEKDAY.search(lowered):
        target = WEEKDAYS.index(match.group(2))
        back = (today.weekday() - target) % 7
        if back == 0 and (match.group(1) or "").strip() == "last":
            back = 7
        return today - timedelta(days=back)
    if match := _MONTH_DAY.search(lowered):
        return _recent(MONTHS[match.group(1)], int(match.group(2)), today)
    if match := _DAY_MONTH.search(lowered):
        return _recent(MONTHS[match.group(2)], int(match.group(1)), today)
    if match := _US.search(lowered):
        month, day, year = match.groups()
        if year:
            full = int(year) + (2000 if len(year) == 2 else 0)
            return _safe(full, int(month), int(day))
        return _recent(int(month), int(day), today)
    return None


def parse_span(text: str, today: date) -> int | None:
    """How many days back (today included) `text` asks about, or None."""
    lowered = text.lower()
    if match := _LAST_N.search(lowered):
        return max(1, min(int(match.group(1)), 365))
    if re.search(r"\bthis\s+week\b", lowered):
        return today.weekday() + 1
    if re.search(r"\b(?:last|past|previous)\s+week\b", lowered):
        return 7
    if re.search(r"\bthis\s+month\b", lowered):
        return today.day
    if re.search(r"\b(?:last|past|previous)\s+month\b", lowered):
        return 30
    if re.search(r"\btoday\b", lowered):
        return 1
    return None


def span_label(text: str, days: int) -> str:
    lowered = text.lower()
    if re.search(r"\bthis\s+week\b", lowered):
        return "This week"
    if re.search(r"\bthis\s+month\b", lowered):
        return "This month"
    return "Today" if days == 1 else f"Last {days} days"
