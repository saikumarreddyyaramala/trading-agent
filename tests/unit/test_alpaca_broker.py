from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from trading_agent.execution.alpaca_broker import AlpacaBroker, BrokerError, mask_account_number


class _Status(Enum):
    ACTIVE = "ACTIVE"


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get_account(self) -> Any:
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(
            account_number="PA3XYZ9876",
            status=_Status.ACTIVE,
            currency="USD",
            cash="100000.5",
            equity="100250",
            buying_power="200000",
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
        )

    def get_clock(self) -> Any:
        if self.fail:
            raise ConnectionError("network down")
        now = datetime(2026, 10, 15, 13, 30, tzinfo=UTC)
        return SimpleNamespace(timestamp=now, is_open=False, next_open=now, next_close=now)


def test_account_is_translated_and_masked() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_Client())
    account = broker.get_account()
    assert account.account_number_masked == "****9876"
    assert account.status == "ACTIVE"
    assert account.cash == Decimal("100000.5")
    assert account.equity == Decimal("100250")
    assert account.is_paper is True


def test_clock_is_translated() -> None:
    clock = AlpacaBroker("k", "s", paper=True, client=_Client()).get_clock()
    assert clock.is_open is False


def test_errors_are_wrapped() -> None:
    broker = AlpacaBroker("k", "s", paper=True, client=_Client(fail=True))
    with pytest.raises(BrokerError, match="could not fetch account"):
        broker.get_account()
    with pytest.raises(BrokerError, match="could not fetch market clock"):
        broker.get_clock()


@pytest.mark.parametrize(
    ("raw", "masked"),
    [("PA3XYZ9876", "****9876"), (None, "unknown"), ("", "unknown"), ("12", "****12")],
)
def test_mask_account_number(raw: str | None, masked: str) -> None:
    assert mask_account_number(raw) == masked
