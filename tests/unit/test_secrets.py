from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from tests.conftest import make_secrets
from trading_agent.settings.secrets import MissingSecretError, Secrets


def test_secrets_are_hidden_in_repr_and_str() -> None:
    secrets = make_secrets(anthropic_api_key=SecretStr("sk-very-secret"))
    assert "sk-very-secret" not in repr(secrets)
    assert "sk-very-secret" not in str(secrets)


def test_missing_reports_env_var_names() -> None:
    secrets = make_secrets(alpaca_api_key=None, anthropic_api_key=SecretStr(""))
    assert secrets.missing("alpaca_api_key", "alpaca_secret_key", "anthropic_api_key") == [
        "ALPACA_API_KEY",
        "ANTHROPIC_API_KEY",
    ]


def test_require_returns_value_or_raises() -> None:
    secrets = make_secrets(alpaca_api_key=None)
    assert secrets.require("anthropic_api_key") == "test-anthropic-key"
    with pytest.raises(MissingSecretError, match="ALPACA_API_KEY"):
        secrets.require("alpaca_api_key")


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    secrets = Secrets(_env_file=None)
    assert secrets.require("anthropic_api_key") == "from-env"
    assert secrets.live_trading_confirmed is True


def test_reads_from_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY=from-file\n", encoding="utf-8")
    assert Secrets(_env_file=env_file).require("alpaca_api_key") == "from-file"


def test_live_confirmation_defaults_to_false() -> None:
    assert Secrets(_env_file=None).live_trading_confirmed is False
