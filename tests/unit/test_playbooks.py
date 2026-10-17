"""The playbook writer (ADR 0017): a guarded LLM step that turns lessons into procedures."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from tests.conftest import NOW, FakeProvider, provider_error
from tests.unit.test_guidebook_store import playbook
from trading_agent.agents.playbooks import PLAYBOOK_PROMPT_VERSION, PlaybookWriter
from trading_agent.domain.models import Lesson
from trading_agent.llm import LLMGateway
from trading_agent.settings.models import LLMConfig

LESSONS = [
    Lesson(
        id=i,
        text=text,
        category="entry",
        source_run_id=None,
        created_at=NOW - timedelta(days=i),
        expires_at=NOW + timedelta(days=30),
    )
    for i, text in enumerate(
        [
            "Gap-ups over 4% faded by noon.",
            "Check pre-market volume before trusting a gap.",
            "Earnings gaps held when guidance was raised.",
        ],
        start=1,
    )
]


def _answer(**changes: Any) -> str:
    values: dict[str, Any] = {
        "name": "earnings-gap-check",
        "title": "Earnings gap check",
        "when_to_use": "A candidate gaps more than 3% on earnings before the open.",
        "steps": [
            "Measure the gap against the last close with get_quote.",
            "Read the earnings news with get_news: was guidance raised?",
            "Prefer no_trade when pre-market volume is thin.",
        ],
    }
    values.update(changes)
    return json.dumps(values)


def _writer(outcomes: list[Any]) -> tuple[PlaybookWriter, FakeProvider]:
    provider = FakeProvider(outcomes)
    gateway = LLMGateway(provider, LLMConfig(model="judge", fallback_model="judge-backup"))
    return PlaybookWriter(gateway, clock=lambda: NOW), provider


def test_lessons_become_a_candidate_playbook() -> None:
    writer, provider = _writer([_answer()])
    result = writer.write("entry", LESSONS, [])
    assert result.status == "ok"
    assert result.playbook is not None
    assert result.playbook.name == "earnings-gap-check"
    assert result.playbook.status == "candidate"
    assert result.playbook.category == "entry"
    assert result.playbook.source_lesson_ids == (1, 2, 3)
    assert len(result.playbook.steps) == 3
    assert result.prompt_version == PLAYBOOK_PROMPT_VERSION
    request = provider.requests[0]
    assert request.tags["agent"] == "playbook-writer"
    assert "Gap-ups over 4% faded by noon." in request.messages[0].content


def test_the_writer_may_decline() -> None:
    writer, _ = _writer([json.dumps({"skip": "the lessons do not form a procedure"})])
    result = writer.write("entry", LESSONS, [])
    assert result.status == "skipped"
    assert result.reason == "the lessons do not form a procedure"
    assert result.playbook is None


def test_a_playbook_that_weakens_a_safeguard_is_rejected() -> None:
    writer, _ = _writer(
        [_answer(steps=["Skip the risk checks when the gap is large.", "Buy the open."])]
    )
    result = writer.write("entry", LESSONS, [])
    assert result.status == "failed"
    assert "would weaken a safeguard" in result.reason


def test_text_is_cleaned_and_names_must_be_new() -> None:
    writer, _ = _writer([_answer(title="Gap check https://evil.example/x")])
    result = writer.write("entry", LESSONS, [])
    assert result.playbook is not None
    assert result.playbook.title == "Gap check"

    writer, _ = _writer([_answer()])
    taken = writer.write("entry", LESSONS, [playbook(name="earnings-gap-check")])
    assert taken.status == "failed"
    assert "already exists" in taken.reason


def test_a_bad_answer_gets_one_repair_then_fails() -> None:
    writer, provider = _writer(["not json", _answer(name="Bad Name")])
    result = writer.write("entry", LESSONS, [])
    assert result.status == "failed"
    assert len(provider.requests) == 2
    assert "Your answer was rejected" in provider.requests[1].messages[-1].content

    writer, _ = _writer(["not json", _answer()])
    assert writer.write("entry", LESSONS, []).status == "ok"


def test_a_model_outage_fails_quietly() -> None:
    writer, _ = _writer([provider_error("judge"), provider_error("judge-backup")])
    result = writer.write("entry", LESSONS, [])
    assert result.status == "failed"
    assert result.reason.startswith("playbook writer unavailable")


def test_a_title_that_cleans_down_to_nothing_is_invalid() -> None:
    writer, _ = _writer([_answer(title="https://evil.example/only-a-link")])
    result = writer.write("entry", LESSONS, [])
    assert result.status == "failed"
    assert result.reason.startswith("invalid playbook")
