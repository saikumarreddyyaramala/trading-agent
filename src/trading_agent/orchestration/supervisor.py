"""Restart the cycle after an unexpected crash (ADR 0021).

Each step already contains its own failures; this catches what escapes the cycle itself
(a locked database, a bug). It alerts, waits (1, 2, 4 ... minutes, capped), and starts
again; restarting is safe because finished steps are never repeated. Too many crashes in
an hour means something is really wrong: it alerts once more and stops. Open positions
keep their broker-side stop-loss either way.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from trading_agent.observability.logging import get_logger

FIRST_BACKOFF_SECONDS = 60
MAX_BACKOFF_SECONDS = 900
CRASH_WINDOW = timedelta(hours=1)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def supervise(
    run: Callable[[], None],
    *,
    alert: Callable[[str, str], object],
    sleep: Callable[[float], None] = _time.sleep,
    clock: Callable[[], datetime] | None = None,
    max_crashes: int = 5,
    first_backoff_seconds: float = FIRST_BACKOFF_SECONDS,
    max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
    logger: Any = None,
) -> int:
    """Run `run` until it returns (exit code 0) or crashes too often (exit code 1).
    Ctrl+C (KeyboardInterrupt) is passed straight through."""
    now = clock or (lambda: datetime.now(UTC))
    log = logger or get_logger(__name__)
    crashes: list[datetime] = []
    backoff = first_backoff_seconds

    def safe_alert(text: str) -> None:
        try:
            alert("crash", text)
        except Exception as exc:  # the restart matters more than the alert
            log.warning("supervisor.alert_failed", error=type(exc).__name__)

    while True:
        started = now()
        try:
            run()
            return 0
        except Exception as exc:
            crashed = now()
            reason = f"{type(exc).__name__}: {exc}"
            log.error("supervisor.crash", error=reason)
            crashes = [c for c in crashes if crashed - c < CRASH_WINDOW] + [crashed]
            if crashed - started >= CRASH_WINDOW:
                backoff = first_backoff_seconds  # it ran fine for a while: start over
            if len(crashes) >= max_crashes:
                safe_alert(
                    f"The trading cycle stopped after {len(crashes)} crashes within an hour. "
                    f"Last error: {reason}. Open positions keep their broker-side stop-loss. "
                    "Restart `trading-agent run` once it is fixed."
                )
                return 1
            safe_alert(f"The trading cycle crashed ({reason}); restarting in {_minutes(backoff)}.")
            sleep(backoff)
            backoff = min(backoff * 2, max_backoff_seconds)
