from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import FakeBroker, FakeProvider, make_secrets, provider_error
from trading_agent.health import CheckResult, CheckStatus, overall_ok, run_healthchecks
from trading_agent.llm import LLMGateway
from trading_agent.settings import load_config
from trading_agent.settings.models import AppConfig


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def _gateway(config: AppConfig, outcomes: list[str | Exception]) -> LLMGateway:
    return LLMGateway(FakeProvider(outcomes), config.llm)


@pytest.fixture
def paper(config_dir: Path) -> AppConfig:
    return load_config("paper", config_dir)


def test_all_green(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=FakeBroker,
        llm_factory=lambda: _gateway(paper, ["OK"]),
    )
    assert overall_ok(results), results
    checks = _by_name(results)
    assert "equity $100,000.00" in checks["Broker account"].detail
    assert checks["LLM"].status is CheckStatus.PASS


def test_missing_secrets_fail_and_skip_dependent_checks(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(alpaca_api_key=None),
        broker_factory=FakeBroker,
        llm_factory=None,
    )
    checks = _by_name(results)
    assert checks["Secrets"].status is CheckStatus.FAIL
    assert "ALPACA_API_KEY" in checks["Secrets"].detail
    assert checks["Broker account"].status is CheckStatus.SKIP
    assert not overall_ok(results)


def test_skip_llm_does_not_require_llm_key(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(anthropic_api_key=None),
        broker_factory=FakeBroker,
        llm_factory=None,
    )
    assert overall_ok(results)
    assert _by_name(results)["LLM"].status is CheckStatus.SKIP


def test_unreachable_broker_fails(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=lambda: FakeBroker(fail_account=True, fail_clock=True),
        llm_factory=None,
    )
    checks = _by_name(results)
    assert checks["Broker account"].status is CheckStatus.FAIL
    assert checks["Market clock"].status is CheckStatus.FAIL
    assert not overall_ok(results)


def test_paper_config_with_live_account_fails(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=lambda: FakeBroker(is_paper=False),
        llm_factory=None,
    )
    assert _by_name(results)["Broker account"].status is CheckStatus.FAIL


def test_blocked_account_fails(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=lambda: FakeBroker(trading_blocked=True),
        llm_factory=None,
    )
    assert "blocked" in _by_name(results)["Broker account"].detail


def test_llm_failure_fails(paper: AppConfig) -> None:
    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=FakeBroker,
        llm_factory=lambda: _gateway(paper, [provider_error(), provider_error()]),
    )
    assert _by_name(results)["LLM"].status is CheckStatus.FAIL


def test_live_without_confirmation_fails(config_dir: Path) -> None:
    live = load_config("live", config_dir)
    results = run_healthchecks(
        live,
        make_secrets(),
        broker_factory=lambda: FakeBroker(is_paper=False),
        llm_factory=None,
    )
    assert _by_name(results)["Trading mode"].status is CheckStatus.FAIL
    assert not overall_ok(results)


def test_live_with_confirmation_warns_but_passes(config_dir: Path) -> None:
    live = load_config("live", config_dir)
    results = run_healthchecks(
        live,
        make_secrets(live_trading_confirmed=True),
        broker_factory=lambda: FakeBroker(is_paper=False),
        llm_factory=None,
    )
    assert _by_name(results)["Trading mode"].status is CheckStatus.WARN
    assert overall_ok(results)


def test_kill_switch_is_reported(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nkill_switch: true\nbroker:\n  paper: true\n", encoding="utf-8"
    )
    config = load_config("paper", config_dir)
    results = run_healthchecks(config, make_secrets(), broker_factory=FakeBroker, llm_factory=None)
    assert _by_name(results)["Kill switch"].status is CheckStatus.WARN
