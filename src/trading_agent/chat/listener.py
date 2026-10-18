"""The Telegram chat listener (ADR 0022): long-polls for messages and answers the owner.

- Only the owner's chat (TELEGRAM_CHAT_ID) is answered. Anyone else who finds the bot is
  ignored and logged once, without their text.
- Every update is confirmed (the offset moves past it), so nothing is answered twice.
- Messages older than `max_age` (sent while the bot was off) are skipped.
- It runs on its own thread and survives every failure: a bad reply becomes an apology, a
  Telegram outage becomes a growing pause. Nothing here can stop the trading cycle.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from trading_agent.observability.logging import get_logger

POLL_TIMEOUT_SECONDS = 30
RETRY_SECONDS = 10.0
MAX_RETRY_SECONDS = 300.0
NON_TEXT_REPLY = "I can only read text messages. Send help to see what I can do."
SORRY_REPLY = "Sorry, something went wrong answering that."


class UpdatesClient(Protocol):
    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]: ...

    def send(self, text: str) -> None: ...


class Replier(Protocol):
    def reply(self, text: str) -> str: ...


class ChatListener:
    def __init__(
        self,
        client: UpdatesClient,
        owner_chat_id: str,
        bot: Replier,
        *,
        max_age: timedelta,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = _time.sleep,
        logger: Any = None,
    ) -> None:
        self._client = client
        self._owner = owner_chat_id
        self._bot = bot
        self._max_age = max_age
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._log = logger or get_logger(__name__)
        self._offset: int | None = None
        self._strangers: set[str] = set()

    def _send(self, text: str) -> None:
        try:
            self._client.send(text)
        except Exception as exc:
            self._log.warning("chat.send_failed", error=str(exc))

    def poll_once(self, timeout: int = 0) -> int:
        """Fetch and answer one batch of messages. Returns how many were answered."""
        answered = 0
        for update in self._client.get_updates(self._offset, timeout):
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = max(self._offset or 0, update_id + 1)
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat")
            chat_id = str(chat.get("id")) if isinstance(chat, dict) else None
            if chat_id != self._owner:
                if chat_id is not None and chat_id not in self._strangers:
                    self._strangers.add(chat_id)
                    self._log.warning("chat.stranger_ignored", chat_id=chat_id)
                continue
            sent = message.get("date")
            if isinstance(sent, int | float):
                age = self._clock() - datetime.fromtimestamp(sent, UTC)
                if age > self._max_age:
                    self._log.info(
                        "chat.stale_skipped", age_hours=round(age.total_seconds() / 3600, 1)
                    )
                    continue
            text = message.get("text")
            if not isinstance(text, str):
                self._send(NON_TEXT_REPLY)
                answered += 1
                continue
            try:
                answer = self._bot.reply(text)
            except Exception as exc:
                self._log.warning("chat.reply_failed", error=f"{type(exc).__name__}: {exc}")
                answer = SORRY_REPLY
            self._send(answer)
            answered += 1
        return answered

    def run(self, stop: threading.Event) -> None:
        """Poll until `stop` is set, pausing longer after each failure in a row."""
        retry = RETRY_SECONDS
        while not stop.is_set():
            try:
                self.poll_once(POLL_TIMEOUT_SECONDS)
                retry = RETRY_SECONDS
            except Exception as exc:
                self._log.warning("chat.poll_failed", error=f"{type(exc).__name__}: {exc}")
                self._sleep(retry)
                retry = min(retry * 2, MAX_RETRY_SECONDS)

    def start(self) -> threading.Event:
        """Run on a background (daemon) thread. Set the returned event to stop it."""
        stop = threading.Event()
        threading.Thread(target=self.run, args=(stop,), name="telegram-chat", daemon=True).start()
        self._log.info("chat.listening")
        return stop
