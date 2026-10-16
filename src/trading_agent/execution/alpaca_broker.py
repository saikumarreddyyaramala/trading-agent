"""Alpaca adapter implementing `BrokerPort`.

Phase 0 is read-only: account and market clock. Order placement arrives in Phase 5 and
will only ever be called by the orchestration layer *after* the risk engine approves.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from alpaca.trading.client import TradingClient

from trading_agent.domain.models import AccountSnapshot, MarketClock


class BrokerError(RuntimeError):
    """The broker could not be reached or returned an unusable response."""


def mask_account_number(value: str | None) -> str:
    """Show only the last 4 characters, enough to tell accounts apart in logs."""
    if not value:
        return "unknown"
    return f"****{value[-4:]}"


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value not in (None, "") else Decimal("0")


def _to_text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


class AlpacaBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool,
        client: Any | None = None,
    ) -> None:
        self._paper = paper
        # Typed as Any: the SDK's return types are unions with raw dicts (raw_data mode),
        # and we always use model mode, so we translate attributes explicitly below.
        self._client: Any = client or TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )

    @property
    def is_paper(self) -> bool:
        return self._paper

    def get_account(self) -> AccountSnapshot:
        try:
            account = self._client.get_account()
        except Exception as exc:  # SDK raises APIError and network errors
            raise BrokerError(f"could not fetch account: {type(exc).__name__}: {exc}") from exc

        return AccountSnapshot(
            account_number_masked=mask_account_number(account.account_number),
            status=_to_text(account.status),
            currency=account.currency or "USD",
            cash=_to_decimal(account.cash),
            equity=_to_decimal(account.equity),
            buying_power=_to_decimal(account.buying_power),
            pattern_day_trader=bool(account.pattern_day_trader),
            trading_blocked=bool(account.trading_blocked),
            account_blocked=bool(account.account_blocked),
            is_paper=self._paper,
        )

    def get_clock(self) -> MarketClock:
        try:
            clock = self._client.get_clock()
        except Exception as exc:
            raise BrokerError(f"could not fetch market clock: {type(exc).__name__}: {exc}") from exc

        return MarketClock(
            timestamp=clock.timestamp,
            is_open=bool(clock.is_open),
            next_open=clock.next_open,
            next_close=clock.next_close,
        )
