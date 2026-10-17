"""A minimal Telegram Bot API client: send a plain-text message, find who wrote to the bot.

The bot token is part of every request URL, so no error raised here carries the URL, the
token, or the original exception (whose text can include the URL).
"""

from __future__ import annotations

from typing import Any

import httpx

API = "https://api.telegram.org"
TIMEOUT_SECONDS = 10.0
MAX_CHARS = 4096  # Telegram's limit for one message


class AlertError(RuntimeError):
    """An alert could not be sent. The message never contains the bot token."""


class TelegramClient:
    def __init__(self, token: str, chat_id: str | None, *, client: Any | None = None) -> None:
        self._token = token
        self._chat_id = chat_id
        self._client = client

    def __repr__(self) -> str:  # never show the token
        return "TelegramClient(token=***)"

    def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        http = self._client or httpx.Client(timeout=TIMEOUT_SECONDS)
        try:
            response = http.post(f"{API}/bot{self._token}/{method}", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise AlertError(f"telegram rejected the alert: HTTP {code}") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise AlertError(f"telegram unreachable: {type(exc).__name__}") from None
        finally:
            if self._client is None:
                http.close()
        return data

    def send(self, text: str) -> None:
        """Send `text` as plain text (no Markdown or HTML parsing) to the configured chat."""
        if not self._chat_id:
            raise AlertError("TELEGRAM_CHAT_ID is not set: run `trading-agent alerts-setup`")
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 1] + "…"
        self._call(
            "sendMessage",
            {"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True},
        )

    def find_chats(self) -> list[tuple[str, str]]:
        """Chats that recently messaged the bot, as (chat id, name), first seen first."""
        data = self._call("getUpdates", {})
        found: dict[str, str] = {}
        for update in data.get("result") or []:
            if not isinstance(update, dict):
                continue
            message = update.get("message") or update.get("edited_message") or {}
            chat = message.get("chat") if isinstance(message, dict) else None
            if not isinstance(chat, dict) or "id" not in chat:
                continue
            name = chat.get("title") or chat.get("first_name") or chat.get("username") or "?"
            found.setdefault(str(chat["id"]), str(name))
        return list(found.items())
