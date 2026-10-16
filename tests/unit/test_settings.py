from __future__ import annotations

from datetime import time
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from tests.conftest import make_secrets
from trading_agent.settings import (
    AppConfig,
    ConfigError,
    Environment,
    assert_live_trading_allowed,
    load_config,
)
from trading_agent.settings.loader import deep_merge


def _write(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_repo_config_loads_paper_by_default(config_dir: Path) -> None:
    config = load_config(config_dir=config_dir)
    assert config.environment is Environment.PAPER
    assert config.broker.paper is True
    assert config.risk.max_position_usd == Decimal("50")
    assert config.schedule.trade_window_start == time(9, 45)


def test_repo_config_loads_live(config_dir: Path) -> None:
    config = load_config("live", config_dir)
    assert config.is_live
    assert config.broker.paper is False


def test_app_env_variable_selects_environment(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "live")
    assert load_config(config_dir=config_dir).is_live


def test_unknown_environment_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="unknown environment"):
        load_config("staging", config_dir)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("paper", tmp_path)


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    _write(config_dir / "paper.yaml", {"broker": {"paper": True}, "risk": {"max_postion": 10}})
    with pytest.raises(ConfigError, match="max_postion"):
        load_config("paper", config_dir)


def test_paper_environment_cannot_point_at_real_money(config_dir: Path) -> None:
    _write(config_dir / "paper.yaml", {"broker": {"paper": False}})
    with pytest.raises(ConfigError, match=r"requires broker\.paper = true"):
        load_config("paper", config_dir)


def test_live_environment_cannot_silently_use_paper(config_dir: Path) -> None:
    _write(config_dir / "live.yaml", {"broker": {"paper": True}})
    with pytest.raises(ConfigError, match=r"requires broker\.paper = false"):
        load_config("live", config_dir)


def test_schedule_must_be_chronological() -> None:
    with pytest.raises(ValueError, match="chronological"):
        AppConfig.model_validate({"schedule": {"analysis": "08:00", "data_collection": "08:30"}})


def test_invalid_timezone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        AppConfig.model_validate({"schedule": {"timezone": "Mars/Olympus"}})


def test_negative_position_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_position_usd"):
        AppConfig.model_validate({"risk": {"max_position_usd": -1}})


def test_config_is_immutable(config_dir: Path) -> None:
    config = load_config(config_dir=config_dir)
    with pytest.raises(ValueError, match="frozen"):
        config.kill_switch = True  # type: ignore[misc]


def test_deep_merge_overrides_nested_values_only() -> None:
    base = {"risk": {"a": 1, "b": 2}, "x": 1}
    merged = deep_merge(base, {"risk": {"b": 3}})
    assert merged == {"risk": {"a": 1, "b": 3}, "x": 1}
    assert base["risk"] == {"a": 1, "b": 2}  # inputs are not mutated


def test_repo_config_market_data_defaults(config_dir: Path) -> None:
    market_data = load_config("paper", config_dir).market_data
    assert market_data.provider == "alpaca"
    assert market_data.quote_feed == "iex"
    assert market_data.bars_feed == "sip"
    assert market_data.daily_bars == 30


@pytest.mark.parametrize(
    "override",
    [
        {"quote_feed": "bloomberg"},
        {"bars_feed": "delayed_sip"},
        {"daily_bars": 1},
        {"news_lookback_hours": 0},
        {"max_news_articles": 500},
    ],
)
def test_invalid_market_data_settings_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="market_data"):
        AppConfig.model_validate({"market_data": override})


def test_repo_config_risk_additions(config_dir: Path) -> None:
    risk = load_config("paper", config_dir).risk
    assert risk.max_spread_pct == Decimal("0.5")
    assert risk.exclude_funds is True
    assert risk.volume_lookback_days == 20


@pytest.mark.parametrize(
    "override", [{"max_spread_pct": 0}, {"volume_lookback_days": 4}, {"volume_lookback_days": 121}]
)
def test_invalid_risk_additions_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="risk"):
        AppConfig.model_validate({"risk": override})


def test_repo_config_execution(config_dir: Path) -> None:
    execution = load_config("paper", config_dir).execution
    assert execution.entry_timeout_seconds == 120
    assert execution.poll_interval_seconds == 2
    assert execution.max_decision_age_seconds == 60
    assert execution.cancel_timeout_seconds == 30


@pytest.mark.parametrize(
    "override",
    [
        {"entry_timeout_seconds": 1},
        {"poll_interval_seconds": 0},
        {"max_decision_age_seconds": 601},
        {"cancel_timeout_seconds": 4},
    ],
)
def test_invalid_execution_settings_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="execution"):
        AppConfig.model_validate({"execution": override})


def test_storage_is_optional_for_paper_and_required_for_live(config_dir: Path) -> None:
    assert load_config("paper", config_dir).storage.required is False
    assert load_config("live", config_dir).storage.required is True


def test_live_without_required_storage_is_rejected(config_dir: Path) -> None:
    _write(config_dir / "live.yaml", {"broker": {"paper": False}, "storage": {"required": False}})
    with pytest.raises(ConfigError, match=r"requires storage\.required = true"):
        load_config("live", config_dir)


def test_repo_config_orchestration(config_dir: Path) -> None:
    orchestration = load_config("paper", config_dir).orchestration
    assert orchestration.manage_interval_seconds == 60
    assert orchestration.early_close_buffer_minutes == 10
    assert orchestration.close_settle_seconds == 30


@pytest.mark.parametrize(
    "override",
    [
        {"manage_interval_seconds": 5},
        {"early_close_buffer_minutes": 61},
        {"close_settle_seconds": 0},
    ],
)
def test_invalid_orchestration_settings_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="orchestration"):
        AppConfig.model_validate({"orchestration": override})


def test_repo_config_tool_budget(config_dir: Path) -> None:
    assert load_config("paper", config_dir).tools.max_calls_per_session == 100


@pytest.mark.parametrize("value", [0, -1, 1001])
def test_tool_budget_must_be_sane(value: int) -> None:
    with pytest.raises(ValueError, match="max_calls_per_session"):
        AppConfig.model_validate({"tools": {"max_calls_per_session": value}})


def test_repo_config_analyst_limits(config_dir: Path) -> None:
    analyst = load_config("paper", config_dir).analyst
    assert analyst.max_turns == 15
    assert analyst.max_tool_calls_per_turn == 8
    assert analyst.max_total_tokens == 300_000
    assert analyst.repair_attempts == 1


@pytest.mark.parametrize(
    "override",
    [
        {"max_turns": 1},
        {"max_tool_calls_per_turn": 0},
        {"max_total_tokens": 0},
        {"repair_attempts": 4},
    ],
)
def test_invalid_analyst_limits_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="analyst"):
        AppConfig.model_validate({"analyst": override})


def test_live_trading_requires_explicit_confirmation(config_dir: Path) -> None:
    live = load_config("live", config_dir)
    with pytest.raises(ConfigError, match="LIVE_TRADING_CONFIRMED"):
        assert_live_trading_allowed(live, make_secrets(live_trading_confirmed=False))
    assert_live_trading_allowed(live, make_secrets(live_trading_confirmed=True))


def test_paper_trading_needs_no_confirmation(config_dir: Path) -> None:
    assert_live_trading_allowed(load_config("paper", config_dir), make_secrets())


def test_repo_config_reporting(config_dir: Path) -> None:
    reporting = load_config("paper", config_dir).reporting
    assert reporting.benchmark_symbol == "SPY"
    assert reporting.output_dir == "reports"
    assert reporting.llm_usd_per_million_input_tokens is None


@pytest.mark.parametrize(
    "override", [{"history_days": 3}, {"llm_usd_per_million_input_tokens": -1}]
)
def test_invalid_reporting_settings_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="reporting"):
        AppConfig.model_validate({"reporting": override})


def test_repo_config_critic(config_dir: Path) -> None:
    critic = load_config("paper", config_dir).critic
    assert critic.enabled is True
    assert critic.model is None
    assert critic.repair_attempts == 1


@pytest.mark.parametrize("override", [{"repair_attempts": 3}, {"max_evidence_chars": 100}])
def test_invalid_critic_settings_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="critic"):
        AppConfig.model_validate({"critic": override})
