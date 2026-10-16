from __future__ import annotations

from pathlib import Path

from tests.conftest import NOW
from trading_agent.agents.prompts import repair_prompt, system_prompt, task_prompt
from trading_agent.settings import load_config


def test_system_prompt_uses_configured_rules(config_dir: Path) -> None:
    prompt = " ".join(system_prompt(load_config("paper", config_dir)).split())
    assert "closed by 15:50 America/New_York" in prompt
    assert "NYSE or NASDAQ" in prompt
    assert "$5 per share" in prompt
    assert "1,000,000 shares a day" in prompt
    assert "untrusted third-party data" in prompt
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
