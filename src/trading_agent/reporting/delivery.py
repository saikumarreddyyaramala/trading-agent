"""Saving and sending daily reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from trading_agent.reporting.report import DailyReport

WEBHOOK_TIMEOUT_SECONDS = 10.0
MAX_WEBHOOK_CHARS = 39_000  # Slack's message limit is 40k characters


class DeliveryError(RuntimeError):
    """The report could not be saved or sent. The message never contains the webhook URL."""


def save_report(report: DailyReport, markdown: str, directory: Path) -> tuple[Path, Path]:
    """Write `<day>.md` and `<day>.json`, replacing an earlier version for the same day."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        md_path = directory / f"{report.day:%Y-%m-%d}.md"
        json_path = directory / f"{report.day:%Y-%m-%d}.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise DeliveryError(f"could not save the report: {exc}") from exc
    return md_path, json_path


def post_webhook(url: str, markdown: str, *, client: Any | None = None) -> None:
    """Post the report to a Slack-compatible incoming webhook (`{"text": ...}`)."""
    text = markdown if len(markdown) <= MAX_WEBHOOK_CHARS else markdown[:MAX_WEBHOOK_CHARS] + "\n…"
    http = client or httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS)
    try:
        response = http.post(url, json={"text": text})
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DeliveryError(
            f"webhook rejected the report: HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        # The exception text can include the URL, which is a secret; report the type only.
        raise DeliveryError(f"webhook unreachable: {type(exc).__name__}") from exc
    finally:
        if client is None:
            http.close()
