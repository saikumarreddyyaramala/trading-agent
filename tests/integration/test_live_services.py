"""Integration tests against the real Alpaca paper API and the real LLM.

Skipped unless the keys are present, so CI stays green without secrets.
Run locally with:  make test-integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trading_agent.bootstrap import Container
from trading_agent.domain.models import ChatMessage, LLMRequest
from trading_agent.settings import Secrets, load_config

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _container() -> Container:
    return Container(
        config=load_config("paper", REPO / "config"),
        secrets=Secrets(_env_file=REPO / ".env"),
    )


@pytest.fixture(autouse=True)
def _restore_env_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unit-test fixture clears env vars; integration tests read keys from the repo's .env.
    monkeypatch.chdir(REPO)


def _require(*fields: str) -> Container:
    container = _container()
    missing = container.secrets.missing(*fields)
    if missing:
        pytest.skip(f"missing {', '.join(missing)}")
    return container


def test_alpaca_paper_account_is_reachable() -> None:
    broker = _require("alpaca_api_key", "alpaca_secret_key").build_broker()
    account = broker.get_account()
    assert account.is_paper
    assert account.status
    assert broker.get_clock().next_open is not None


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="never spend LLM tokens in CI")
def test_llm_round_trip() -> None:
    gateway = _require("anthropic_api_key").build_llm_gateway()
    response = gateway.complete(
        LLMRequest(
            system="Reply with exactly one word.",
            messages=(ChatMessage(role="user", content="Say OK"),),
            max_tokens=16,
        )
    )
    assert response.text.strip()
