"""Alpaca adapter implementing `MarketDataPort` (quotes, daily bars, asset reference data)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from alpaca.data.enums import Adjustment, DataFeed, MostActivesBy
from alpaca.data.historical import ScreenerClient, StockHistoricalDataClient
from alpaca.data.requests import MostActivesRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import ActiveStock, Asset, Bar, Quote

QuoteFeed = Literal["iex", "sip", "delayed_sip"]
BarsFeed = Literal["iex", "sip"]

# Alpaca's free plan serves SIP history only when it is at least 15 minutes old.
SIP_HISTORY_DELAY = timedelta(minutes=16)
# Largest list the most-actives screener returns.
MAX_MOST_ACTIVE = 100


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value not in (None, "") else Decimal("0")


def _text(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _calendar_window(trading_days: int) -> timedelta:
    # ~252 trading days per 365 calendar days, plus slack for holidays and long weekends.
    return timedelta(days=math.ceil(trading_days * 365 / 252) + 7)


class AlpacaMarketData:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool,
        quote_feed: QuoteFeed = "iex",
        bars_feed: BarsFeed = "sip",
        data_client: Any | None = None,
        trading_client: Any | None = None,
        screener_client: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._quote_feed = DataFeed(quote_feed)
        self._bars_feed = DataFeed(bars_feed)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._data: Any = data_client or StockHistoricalDataClient(api_key, secret_key)
        self._screener: Any = screener_client or ScreenerClient(api_key, secret_key)
        # Asset reference data lives on the trading API, which differs for paper and live.
        self._trading: Any = trading_client or TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )

    def get_asset(self, symbol: str) -> Asset:
        try:
            asset = self._trading.get_asset(symbol)
        except Exception as exc:  # SDK raises APIError (incl. 404) and network errors
            raise MarketDataError(f"asset lookup failed for {symbol}: {exc}") from exc
        return Asset(
            symbol=str(asset.symbol),
            name=asset.name,
            exchange=_text(asset.exchange),
            status=_text(asset.status),
            tradable=bool(asset.tradable),
            fractionable=bool(asset.fractionable),
        )

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        if not symbols:
            return {}
        request = StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=self._quote_feed)
        try:
            raw = self._data.get_stock_latest_quote(request)
        except Exception as exc:
            raise MarketDataError(f"could not fetch quotes: {type(exc).__name__}: {exc}") from exc
        return {
            symbol: Quote(
                symbol=symbol,
                timestamp=quote.timestamp,
                bid_price=_decimal(quote.bid_price),
                bid_size=_decimal(quote.bid_size),
                ask_price=_decimal(quote.ask_price),
                ask_size=_decimal(quote.ask_size),
            )
            for symbol, quote in raw.items()
        }

    def get_daily_bars(self, symbols: Sequence[str], *, days: int) -> dict[str, tuple[Bar, ...]]:
        if not symbols:
            return {}
        now = self._clock()
        end = now - SIP_HISTORY_DELAY if self._bars_feed is DataFeed.SIP else now
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=end - _calendar_window(days),
            end=end,
            adjustment=Adjustment.ALL,
            feed=self._bars_feed,
        )
        try:
            raw = self._data.get_stock_bars(request)
        except Exception as exc:
            raise MarketDataError(f"could not fetch bars: {type(exc).__name__}: {exc}") from exc

        result: dict[str, tuple[Bar, ...]] = {}
        for symbol, bars in raw.data.items():
            translated = sorted(
                (
                    Bar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=_decimal(bar.open),
                        high=_decimal(bar.high),
                        low=_decimal(bar.low),
                        close=_decimal(bar.close),
                        volume=int(bar.volume),
                        vwap=None if bar.vwap is None else _decimal(bar.vwap),
                    )
                    for bar in bars
                ),
                key=lambda bar: bar.timestamp,
            )
            if translated:
                result[symbol] = tuple(translated[-days:])
        return result

    def get_most_active(self, limit: int) -> tuple[ActiveStock, ...]:
        if not 1 <= limit <= MAX_MOST_ACTIVE:
            raise ValueError(f"limit must be between 1 and {MAX_MOST_ACTIVE}")
        try:
            request = MostActivesRequest(top=limit, by=MostActivesBy.VOLUME)
            raw = self._screener.get_most_actives(request)
        except Exception as exc:
            raise MarketDataError(
                f"could not fetch most active stocks: {type(exc).__name__}: {exc}"
            ) from exc
        stocks = [
            ActiveStock(
                symbol=str(item.symbol),
                volume=int(item.volume),
                trade_count=int(item.trade_count),
            )
            for item in raw.most_actives
        ]
        stocks.sort(key=lambda stock: stock.volume, reverse=True)
        return tuple(stocks[:limit])
