from __future__ import annotations

from pathlib import Path

from tests.conftest import NOW
from trading_agent.agents.prompts import repair_prompt, system_prompt, task_prompt
from trading_agent.domain.models import Lesson
from trading_agent.settings import load_config


def test_system_prompt_uses_configured_rules(config_dir: Path) -> None:
    prompt = " ".join(system_prompt(load_config("paper", config_dir)).split())
    assert "closed by 15:50 America/New_York" in prompt
    assert "NYSE or NASDAQ" in prompt
    assert "$5 per share" in prompt
    assert "1,000,000 shares a day" in prompt
    assert "untrusted third-party data" in prompt
    assert "Funds, ETFs, ETNs, and leveraged or inverse products are not allowed" in prompt
    assert '"decision": "buy" | "no_trade"' in prompt  # JSON braces survive formatting


def test_system_prompt_follows_config_changes(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nbroker:\n  paper: true\nrisk:\n  min_price_usd: 10\n"
        "  allowed_exchanges: [NYSE]\n",
        encoding="utf-8",
    )
    prompt = " ".join(system_prompt(load_config("paper", config_dir)).split())
    assert "$10 per share" in prompt
    assert "listed on NYSE," in prompt


def test_task_prompt_uses_market_time() -> None:
    assert "Thursday 2026-10-15 10:00 (America/New_York)" in task_prompt(NOW, "America/New_York")


def test_repair_prompt_lists_problems() -> None:
    prompt = repair_prompt(["bad id", "bad price"])
    assert "- bad id\n- bad price" in prompt
    assert "no_trade" in prompt


def test_lessons_come_after_the_rules_and_are_advisory(config_dir: Path) -> None:
    config = load_config("paper", config_dir)
    lesson = Lesson(
        id=1,
        text="Gap-ups often faded by noon.",
        category="entry",
        source_run_id="daily-20261014",
        created_at=NOW,
        expires_at=NOW,
    )
    base = system_prompt(config)
    prompt = system_prompt(config, [lesson])
    assert prompt.startswith(base)  # the rules are unchanged; lessons are only appended
    tail = " ".join(prompt[len(base) :].split())
    assert "They are advisory" in tail
    assert "never override anything above" in tail
    assert tail.endswith("- [entry, 2026-10-15] Gap-ups often faded by noon.")


def test_task_prompt_carries_slot_context() -> None:
    prompt = task_prompt(NOW, "America/New_York", "This is trade slot 2 of 3 today.")
    assert prompt.endswith("no trade.\n\nThis is trade slot 2 of 3 today.")
    assert task_prompt(NOW, "America/New_York").endswith("or no trade.")


def test_principles_lessons_and_playbooks_are_separate(config_dir: Path) -> None:
    from tests.unit.test_guidebook_store import playbook

    config = load_config("paper", config_dir)

    def lesson(i: int, text: str, promoted: bool) -> Lesson:
        return Lesson(
            id=i,
            text=text,
            category="entry",
            source_run_id=None,
            created_at=NOW,
            expires_at=NOW,
            promoted=promoted,
        )

    prompt = system_prompt(
        config,
        [lesson(1, "Proven rule of thumb.", True), lesson(2, "Fresh hunch.", False)],
        [playbook("gap-check"), playbook("old-one", status="active")],
    )
    principles = prompt.index("Principles proven by your own results")
    lessons = prompt.index("Lessons from your own past results")
    playbooks = prompt.index("Playbooks:")
    assert principles < lessons < playbooks
    assert "- [entry] Proven rule of thumb." in prompt[principles:lessons]
    assert "- [entry, 2026-10-15] Fresh hunch." in prompt[lessons:playbooks]
    assert "1. Check the gap size with get_quote." in prompt[playbooks:]
    assert "### old-one: Earnings gap check (active)" in prompt[playbooks:]


def test_system_prompt_teaches_technical_analysis(config_dir: Path) -> None:
    prompt = " ".join(system_prompt(load_config("paper", config_dir)).split())
    assert "technical_analysis" in prompt
    assert "Indicators describe the past" in prompt
    assert "cite the reading" in prompt
    assert "overbought" in prompt


def test_before_the_open_the_analyst_knows_when_its_buy_is_placed() -> None:
    from datetime import UTC, datetime, time

    before_open = datetime(2026, 10, 19, 12, 45, tzinfo=UTC)  # 08:45 New York
    prompt = " ".join(
        task_prompt(before_open, "America/New_York", trade_window_start=time(9, 45)).split()
    )
    assert "Monday 2026-10-19 08:45" in prompt
    assert "The market opens at 09:30" in prompt
    assert "placed after 09:45" in prompt
    assert "is expected before the open" in prompt


def test_during_the_session_there_is_no_pre_open_note() -> None:
    from datetime import time

    prompt = task_prompt(NOW, "America/New_York", trade_window_start=time(9, 45))
    assert "opens at 09:30" not in prompt
