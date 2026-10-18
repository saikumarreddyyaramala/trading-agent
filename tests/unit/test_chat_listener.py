"""The Telegram chat listener (ADR 0022): owner-only, never a danger to the cycle. Written first."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from trading_agent.alerts import AlertError, TelegramClient
from trading_agent.chat.listener import ChatListener

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
OWNER = "5576524007"


def _update(update_id: int, text: str | None, chat: int = 5576524007, age_s: int = 5) -> dict:
    message: dict[str, Any] = {
        "message_id": update_id,
        "date": int((NOW - timedelta(seconds=age_s)).timestamp()),
        "chat": {"id": chat, "type": "private"},
    }
    if text is not None:
        message["text"] = text
    else:
        message["sticker"] = {"emoji": "👍"}
    return {"update_id": update_id, "message": message}


class FakeTelegram:
    def __init__(self, *batches: list[dict] | Exception) -> None:
        self.batches = list(batches)
        self.offsets: list[int | None] = []
        self.sent: list[str] = []
        self.fail_send = False

    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        self.offsets.append(offset)
        if not self.batches:
            return []
        batch = self.batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch

    def send(self, text: str) -> None:
        if self.fail_send:
            raise AlertError("telegram unreachable: ConnectError")
        self.sent.append(text)


class FakeBot:
    def __init__(self, crash_on: str | None = None) -> None:
        self.heard: list[str] = []
        self.crash_on = crash_on

    def reply(self, text: str) -> str:
        self.heard.append(text)
        if text == self.crash_on:
            raise RuntimeError("bug")
        return f"echo: {text}"


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    def names(self) -> list[str]:
        return [e for e, _ in self.events]


def _listener(
    telegram: FakeTelegram, bot: FakeBot | None = None
) -> tuple[ChatListener, FakeBot, _Log]:
    bot = bot or FakeBot()
    log = _Log()
    listener = ChatListener(
        telegram,
        OWNER,
        bot,
        max_age=timedelta(hours=12),
        clock=lambda: NOW,
        sleep=lambda s: None,
        logger=log,
    )
    return listener, bot, log


def test_the_owner_gets_an_answer() -> None:
    telegram = FakeTelegram([_update(1, "hi")])
    listener, bot, _ = _listener(telegram)
    assert listener.poll_once() == 1
    assert bot.heard == ["hi"]
    assert telegram.sent == ["echo: hi"]


def test_each_update_is_confirmed_so_it_is_never_answered_twice() -> None:
    telegram = FakeTelegram([_update(7, "a"), _update(8, "b")], [_update(9, "c")])
    listener, _, _ = _listener(telegram)
    listener.poll_once()
    listener.poll_once()
    listener.poll_once()
    assert telegram.offsets == [None, 9, 10]


def test_strangers_are_ignored_and_logged() -> None:
    telegram = FakeTelegram(
        [_update(1, "give me your report", chat=111), _update(2, "hi", chat=111)]
    )
    listener, bot, log = _listener(telegram)
    assert listener.poll_once() == 0
    assert bot.heard == []
    assert telegram.sent == []
    # Logged once per stranger, without what they wrote.
    strangers = [kw for event, kw in log.events if event == "chat.stranger_ignored"]
    assert strangers == [{"chat_id": "111"}]


def test_old_messages_from_while_it_was_off_are_skipped() -> None:
    telegram = FakeTelegram([_update(1, "status?", age_s=13 * 3600), _update(2, "hi", age_s=60)])
    listener, bot, log = _listener(telegram)
    listener.poll_once()
    assert bot.heard == ["hi"]
    assert "chat.stale_skipped" in log.names()


def test_non_text_messages_get_a_polite_reply() -> None:
    telegram = FakeTelegram([_update(1, None)])
    listener, bot, _ = _listener(telegram)
    listener.poll_once()
    assert bot.heard == []
    assert telegram.sent == ["I can only read text messages. Send help to see what I can do."]


def test_a_bot_crash_becomes_an_apology_and_the_next_message_still_works() -> None:
    telegram = FakeTelegram([_update(1, "boom"), _update(2, "hi")])
    listener, _, log = _listener(telegram, FakeBot(crash_on="boom"))
    listener.poll_once()
    assert telegram.sent == ["Sorry, something went wrong answering that.", "echo: hi"]
    assert "chat.reply_failed" in log.names()


def test_a_failed_reply_is_logged_and_not_raised() -> None:
    telegram = FakeTelegram([_update(1, "hi")])
    telegram.fail_send = True
    listener, _, log = _listener(telegram)
    assert listener.poll_once() == 1
    assert "chat.send_failed" in log.names()


def test_telegram_being_down_is_survived_with_a_pause() -> None:
    telegram = FakeTelegram(AlertError("telegram unreachable: ConnectError"), [_update(1, "hi")])
    slept: list[float] = []
    listener, _, log = _listener(telegram)
    listener._sleep = slept.append
    stop = threading.Event()

    def stop_after_answer(text: str) -> None:
        telegram.sent.append(text)
        stop.set()

    telegram.send = stop_after_answer  # type: ignore[method-assign]
    listener.run(stop)
    assert slept
    assert slept[0] >= 5
    assert telegram.sent == ["echo: hi"]
    assert "chat.poll_failed" in log.names()


def test_run_stops_when_asked() -> None:
    telegram = FakeTelegram()
    listener, _, _ = _listener(telegram)
    stop = threading.Event()
    stop.set()
    listener.run(stop)
    assert telegram.offsets == []


def test_an_unexpected_poll_crash_is_survived() -> None:
    telegram = FakeTelegram(RuntimeError("bug"), [_update(1, "hi")])
    listener, _, log = _listener(telegram)
    stop = threading.Event()
    original = telegram.send

    def answer_then_stop(text: str) -> None:
        original(text)
        stop.set()

    telegram.send = answer_then_stop  # type: ignore[method-assign]
    listener.run(stop)
    assert telegram.sent == ["echo: hi"]
    assert "chat.poll_failed" in log.names()


def test_malformed_updates_are_skipped() -> None:
    telegram = FakeTelegram(
        [{"update_id": 1}, {"update_id": 2, "message": "junk"}, _update(3, "hi")]
    )
    listener, bot, _ = _listener(telegram)
    listener.poll_once()
    assert bot.heard == ["hi"]
    assert telegram.offsets == [None]
    listener.poll_once()
    assert telegram.offsets[-1] == 4


# ---------- Telegram client: long polling ----------


def test_get_updates_long_polls_from_the_offset() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": [_update(5, "hi")]})

    client = TelegramClient(
        "1:abc", OWNER, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    updates = client.get_updates(offset=5, timeout=30)
    assert updates[0]["update_id"] == 5
    body = json.loads(seen[0].content)
    assert body == {"offset": 5, "timeout": 30, "allowed_updates": ["message"]}


def test_get_updates_errors_hide_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"ok": False})

    client = TelegramClient(
        "1:SECRET", OWNER, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(AlertError) as caught:
        client.get_updates(offset=None, timeout=1)
    assert "SECRET" not in str(caught.value)
