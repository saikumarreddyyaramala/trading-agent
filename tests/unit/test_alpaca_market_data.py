from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrameUnit

from tests.conftest import NOW
from trading_agent.data.alpaca_market_data import SIP_HISTORY_DELAY, AlpacaMarketData
from trading_agent.data.errors import MarketDataError


class _Exchange(Enum):
    NASDAQ = "NASDAQ"


class _Status(Enum):
    ACTIVE = "active"


class _DataClient:
    def __init__(self, *, fail: bool = False, bars: dict[str, list[Any]] | None = None) -> None:
        self.fail = fail
        self.bars = bars
        self.requests: list[Any] = []

    def get_stock_latest_quote(self, request: Any) -> Any:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        return {
            "AAPL": SimpleNamespace(
                timestamp=NOW - timedelta(seconds=3),
                bid_price=229.1,
                bid_size=3.0,
                ask_price=229.14,
                ask_size=1.0,
            )
        }

    def get_stock_bars(self, request: Any) -> Any:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(data=self.bars if self.bars is not None else {})


class _TradingClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get_asset(self, symbol: str) -> Any:
        if self.fail:
            raise RuntimeError("asset not found")
        return SimpleNamespace(
            symbol=symbol,
            name="Apple Inc. Common Stock",
            exchange=_Exchange.NASDAQ,
            status=_Status.ACTIVE,
            tradable=True,
            fractionable=True,
        )


def _bar(day: int, close: float) -> Any:
    return SimpleNamespace(
        timestamp=NOW - timedelta(days=day),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1_234_567.0,
        vwap=None if day % 2 else close,
    )


def _adapter(**kwargs: Any) -> tuple[AlpacaMarketData, _DataClient]:
    data = kwargs.pop("data_client", _DataClient())
    adapter = AlpacaMarketData(
        "k",
        "s",
        paper=True,
        data_client=data,
        trading_client=kwargs.pop("trading_client", _TradingClient()),
        clock=lambda: NOW,
        **kwargs,
    )
    return adapter, data


def test_quotes_are_translated_to_decimals() -> None:
    adapter, client = _adapter()
    quote = adapter.get_latest_quotes(["AAPL"])["AAPL"]
    assert quote.bid_price == Decimal("229.1")
    assert quote.ask_price == Decimal("229.14")
    assert quote.age_seconds(NOW) == 3
    assert client.requests[0].feed is DataFeed.IEX


def test_quote_feed_comes_from_config() -> None:
    adapter, client = _adapter(quote_feed="sip")
    adapter.get_latest_quotes(["AAPL"])
    assert client.requests[0].feed is DataFeed.SIP


def test_no_symbols_means_no_call() -> None:
    adapter, client = _adapter()
    assert adapter.get_latest_quotes([]) == {}
    assert adapter.get_daily_bars([], days=5) == {}
    assert client.requests == []


def test_bars_are_sorted_trimmed_and_translated() -> None:
    client = _DataClient(bars={"AAPL": [_bar(1, 103.0), _bar(4, 100.0), _bar(2, 102.0)]})
    adapter, _ = _adapter(data_client=client)

    bars = adapter.get_daily_bars(["AAPL"], days=2)["AAPL"]

    assert [b.close for b in bars] == [Decimal("102.0"), Decimal("103.0")]
    assert bars[0].volume == 1_234_567
    assert bars[0].vwap == Decimal("102.0")
    assert bars[1].vwap is None


def test_sip_bars_end_before_the_free_plan_delay() -> None:
    adapter, client = _adapter()
    adapter.get_daily_bars(["AAPL"], days=30)
    request = client.requests[0]
    assert request.feed is DataFeed.SIP
    assert request.adjustment is Adjustment.ALL
    assert request.timeframe.unit is TimeFrameUnit.Day
    # The SDK stores request times as naive UTC.
    assert request.end == (NOW - SIP_HISTORY_DELAY).replace(tzinfo=None)
    # The window must cover 30 trading days of calendar time.
    assert request.end - request.start >= timedelta(days=42)


def test_iex_bars_end_now() -> None:
    adapter, client = _adapter(bars_feed="iex")
    adapter.get_daily_bars(["AAPL"], days=5)
    assert client.requests[0].end == NOW.replace(tzinfo=None)


def test_symbols_without_bars_are_omitted() -> None:
    client = _DataClient(bars={"AAPL": [], "MSFT": [_bar(1, 400.0)]})
    adapter, _ = _adapter(data_client=client)
    assert set(adapter.get_daily_bars(["AAPL", "MSFT"], days=5)) == {"MSFT"}


def test_asset_is_translated() -> None:
    adapter, _ = _adapter()
    asset = adapter.get_asset("AAPL")
    assert asset.exchange == "NASDAQ"
    assert asset.status == "active"
    assert asset.tradable is True


def test_errors_are_wrapped() -> None:
    adapter, _ = _adapter(
        data_client=_DataClient(fail=True), trading_client=_TradingClient(fail=True)
    )
    with pytest.raises(MarketDataError, match="could not fetch quotes"):
        adapter.get_latest_quotes(["AAPL"])
    with pytest.raises(MarketDataError, match="could not fetch bars"):
        adapter.get_daily_bars(["AAPL"], days=5)
    with pytest.raises(MarketDataError, match="asset lookup failed for AAPL"):
        adapter.get_asset("AAPL")


# ---------- Most active screener ----------


class _Screener:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[Any] = []

    def get_most_actives(self, request: Any) -> Any:
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(
            most_actives=[
                SimpleNamespace(symbol="F", volume=10.0, trade_count=5.0),
                SimpleNamespace(symbol="NVDA", volume=900.0, trade_count=80.0),
            ]
        )


def test_most_active_is_translated_and_sorted() -> None:
    screener = _Screener()
    adapter, _ = _adapter(screener_client=screener)
    stocks = adapter.get_most_active(2)
    assert [s.symbol for s in stocks] == ["NVDA", "F"]
    assert stocks[0].volume == 900
    assert stocks[0].trade_count == 80
    assert screener.requests[0].top == 2


@pytest.mark.parametrize("limit", [0, 101])
def test_most_active_limit_is_bounded(limit: int) -> None:
    adapter, _ = _adapter(screener_client=_Screener())
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        adapter.get_most_active(limit)


def test_most_active_errors_are_wrapped() -> None:
    adapter, _ = _adapter(screener_client=_Screener(fail=True))
    with pytest.raises(MarketDataError, match="could not fetch most active stocks"):
        adapter.get_most_active(5)
