"""The alerter: decides which alerts go out, and makes sure sending can never hurt trading.

- Only the kinds listed in `alerts.events` are sent.
- The same text is not repeated within `dedupe_minutes`.
- At most `max_per_hour` alerts go out in any hour, so a failure loop cannot flood a phone.
- Sending never raises: a failure is logged and the cycle carries on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from trading_agent.alerts.telegram import AlertError
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AlertsConfig


class AlertSender(Protocol):
    def send(self, text: str) -> None: ...


class Alerter:
    def __init__(
        self,
        sender: AlertSender | None,
        config: AlertsConfig,
        *,
        label: str,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._sender = sender
        self._config = config
        self._label = label
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)
        self._last_sent: dict[str, datetime] = {}
        self._recent: deque[datetime] = deque()

    @property
    def configured(self) -> bool:
        return self._sender is not None and self._config.enabled

    def notify(self, kind: str, text: str) -> bool:
        """Send one alert if it is wanted, new, and within the hourly cap. True when sent."""
        sender = self._sender
        if sender is None or not self._config.enabled or kind not in self._config.events:
            return False
        now = self._clock()
        last = self._last_sent.get(text)
        if last is not None and now - last < timedelta(minutes=self._config.dedupe_minutes):
            return False
        while self._recent and now - self._recent[0] >= timedelta(hours=1):
            self._recent.popleft()
        if len(self._recent) >= self._config.max_per_hour:
            self._log.warning("alert.rate_limited", kind=kind, max_per_hour=len(self._recent))
            return False
        try:
            sender.send(f"[{self._label}] {text}")
        except AlertError as exc:
            self._log.warning("alert.failed", kind=kind, error=str(exc))
            return False
        except Exception as exc:  # an alert must never break the cycle
            self._log.warning("alert.failed", kind=kind, error=type(exc).__name__)
            return False
        self._last_sent[text] = now
        self._recent.append(now)
        self._log.info("alert.sent", kind=kind)
        return True
