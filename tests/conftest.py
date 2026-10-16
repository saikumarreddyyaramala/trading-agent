"""Shared fixtures and fakes. Unit tests never touch the network or real credentials."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from trading_agent.domain.models import (
    AccountSnapshot,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MarketClock,
)
from trading_agent.llm.errors import LLMProviderError
from trading_agent.settings.secrets import Secrets

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A private copy of the real config folder, safe to modify per test."""
    target = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, target)
    return target


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test may read the developer's real .env or exported keys."""
    for name in (
        "APP_ENV",
        "CONFIG_DIR",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ANTHROPIC_API_KEY",
        "LIVE_TRADING_CONFIRMED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def make_secrets(**overrides: Any) -> Secrets:
    values: dict[str, Any] = {
        "alpaca_api_key": SecretStr("test-alpaca-key"),
        "alpaca_secret_key": SecretStr("test-alpaca-secret"),
        "anthropic_api_key": SecretStr("test-anthropic-key"),
        "live_trading_confirmed": False,
    }
    values.update(overrides)
    return Secrets(_env_file=None, **values)


class FakeBroker:
    def __init__(
        self,
        *,
        is_paper: bool = True,
        trading_blocked: bool = False,
        fail_account: bool = False,
        fail_clock: bool = False,
    ) -> None:
        self._is_paper = is_paper
        self._trading_blocked = trading_blocked
        self._fail_account = fail_account
        self._fail_clock = fail_clock

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def get_account(self) -> AccountSnapshot:
        if self._fail_account:
            raise RuntimeError("broker unreachable")
        return AccountSnapshot(
            account_number_masked="****1234",
            status="ACTIVE",
            currency="USD",
            cash=Decimal("100000"),
            equity=Decimal("100000"),
            buying_power=Decimal("100000"),
            pattern_day_trader=False,
            trading_blocked=self._trading_blocked,
            account_blocked=False,
            is_paper=self._is_paper,
        )

    def get_clock(self) -> MarketClock:
        if self._fail_clock:
            raise RuntimeError("clock unavailable")
        now = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)
        return MarketClock(timestamp=now, is_open=True, next_open=now, next_close=now)


class FakeProvider:
    """Scripted LLM provider: each call pops the next outcome (a text or an exception)."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, request: LLMRequest, *, model: str, max_tokens: int) -> LLMResponse:
        self.calls.append((model, max_tokens))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome,
            provider=self.name,
            model=model,
            stop_reason="end_turn",
            usage=LLMUsage(input_tokens=10, output_tokens=2),
            latency_ms=5.0,
        )


def provider_error(model: str = "m", *, fatal: bool = False) -> LLMProviderError:
    return LLMProviderError("fake", model, "boom", fatal=fatal)
