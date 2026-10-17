"""Phone alerts over Telegram (ADR 0021): delivery, secrecy, filtering, and rate limits.

Written first. The bot token is part of the Telegram URL, so every error path is checked
for leaking it. Alerts must never break the trading cycle.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import httpx
import pytest

from tests.conftest import NOW
from trading_agent.alerts import Alerter, AlertError, TelegramClient
from trading_agent.settings.models import AlertsConfig

TOKEN = "123456:SECRET-token"
CHAT = "987654"


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))


def _client(handler: Any) -> TelegramClient:
    return TelegramClient(TOKEN, CHAT, client=httpx.Client(transport=httpx.MockTransport(handler)))


# ---------- Telegram client ----------


def test_send_posts_plain_text_to_the_chat() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    _client(handler).send("Bought 1 AAPL")
    (request,) = seen
    assert str(request.url) == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    body = json.loads(request.content)
    assert body == {"chat_id": CHAT, "text": "Bought 1 AAPL", "disable_web_page_preview": True}
    assert "parse_mode" not in body  # plain text: nothing in a message is read as markup


def test_long_messages_are_cut_to_telegrams_limit() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["text"])
        return httpx.Response(200, json={"ok": True})

    _client(handler).send("x" * 5000)
    assert len(sent[0]) <= 4096
    assert sent[0].endswith("…")


def test_a_rejected_alert_never_reveals_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

    with pytest.raises(AlertError) as caught:
        _client(handler).send("hi")
    assert "HTTP 401" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert "SECRET" not in repr(caught.value)


def test_an_unreachable_telegram_never_reveals_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(AlertError) as caught:
        _client(handler).send("hi")
    assert "ConnectError" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None  # the original exception carries the URL


def test_find_chats_lists_who_messaged_the_bot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/getUpdates")
        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {"message": {"chat": {"id": 987654, "first_name": "Sai", "type": "private"}}},
                    {"message": {"chat": {"id": 987654, "first_name": "Sai", "type": "private"}}},
                    {"message": {"chat": {"id": -100, "title": "Group", "type": "group"}}},
                    {"edited_message": {"chat": {"id": 5, "username": "x", "type": "private"}}},
                    {"update_id": 1},
                ],
            },
        )

    chats = TelegramClient(TOKEN, None, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert chats.find_chats() == [("987654", "Sai"), ("-100", "Group"), ("5", "x")]


def test_sending_needs_a_chat_id() -> None:
    client = TelegramClient(TOKEN, None)
    with pytest.raises(AlertError, match="TELEGRAM_CHAT_ID"):
        client.send("hi")


# ---------- Alerter ----------


class _Sender:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    def send(self, text: str) -> None:
        if self.fail:
            raise AlertError("telegram unreachable: ConnectError")
        self.sent.append(text)


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> Any:
        return self.now


def _alerter(
    sender: _Sender | None = None, clock: _Clock | None = None, **config: Any
) -> tuple[Alerter, _Sender, _Log]:
    sender = sender if sender is not None else _Sender()
    log = _Log()
    alerter = Alerter(
        sender,
        AlertsConfig.model_validate(config),
        label="paper",
        clock=clock or _Clock(),
        logger=log,
    )
    return alerter, sender, log


def test_alerts_are_labelled_with_the_account() -> None:
    alerter, sender, _ = _alerter()
    assert alerter.notify("trade", "Bought 1 AAPL at $100")
    assert sender.sent == ["[paper] Bought 1 AAPL at $100"]


def test_unwanted_kinds_are_not_sent() -> None:
    alerter, sender, _ = _alerter(events=["failure"])
    assert not alerter.notify("trade", "Bought 1 AAPL")
    assert alerter.notify("failure", "close failed")
    assert sender.sent == ["[paper] close failed"]


def test_disabled_alerts_send_nothing() -> None:
    alerter, sender, _ = _alerter(enabled=False)
    assert not alerter.notify("crash", "boom")
    assert sender.sent == []


def test_without_telegram_set_up_alerts_are_a_no_op() -> None:
    log = _Log()
    alerter = Alerter(None, AlertsConfig(), label="paper", logger=log)
    assert not alerter.configured
    assert not alerter.notify("crash", "boom")
    assert log.events == []


def test_repeats_are_dropped_until_the_dedupe_window_passes() -> None:
    clock = _Clock()
    alerter, sender, _ = _alerter(clock=clock, dedupe_minutes=30)
    assert alerter.notify("failure", "manage failed")
    clock.now += timedelta(minutes=29)
    assert not alerter.notify("failure", "manage failed")
    assert alerter.notify("failure", "something else")
    clock.now += timedelta(minutes=2)
    assert alerter.notify("failure", "manage failed")
    assert len(sender.sent) == 3


def test_a_flood_is_capped_per_hour_and_logged() -> None:
    clock = _Clock()
    alerter, sender, log = _alerter(clock=clock, max_per_hour=3)
    results = [alerter.notify("failure", f"problem {i}") for i in range(6)]
    assert results == [True, True, True, False, False, False]
    assert len(sender.sent) == 3
    assert sum(1 for event, _ in log.events if event == "alert.rate_limited") == 3
    clock.now += timedelta(minutes=61)
    assert alerter.notify("failure", "problem 9")


def test_a_failed_send_is_logged_and_never_raised() -> None:
    alerter, _, log = _alerter(_Sender(fail=True))
    assert not alerter.notify("crash", "boom")
    event, fields = log.events[-1]
    assert event == "alert.failed"
    assert fields["error"] == "telegram unreachable: ConnectError"


def test_an_unexpected_sender_crash_is_contained() -> None:
    class _Broken:
        def send(self, text: str) -> None:
            raise RuntimeError("bug")

    log = _Log()
    alerter = Alerter(_Broken(), AlertsConfig(), label="paper", clock=_Clock(), logger=log)
    assert not alerter.notify("trade", "x")
    assert log.events[-1][0] == "alert.failed"


def test_alerts_config_is_validated() -> None:
    with pytest.raises(ValueError, match="events"):
        AlertsConfig.model_validate({"events": ["lunch"]})
    with pytest.raises(ValueError, match="max_per_hour"):
        AlertsConfig.model_validate({"max_per_hour": 0})
