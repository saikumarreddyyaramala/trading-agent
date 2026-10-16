"""Market data collection: one call gathers quotes, daily bars, asset info, and news.

Fail-safe rules:
- A provider outage (quotes, bars, or news call fails) raises: the run has no data.
- A single symbol with missing or bad data is rejected with a reason; the rest continue.
- With `require_fresh_quotes=True` (trading decisions), a stale or bad quote rejects the symbol.
  Without it (pre-market research), an unusable quote is dropped and the symbol is kept.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from trading_agent.data.errors import DataQualityError, MarketDataError
from trading_agent.data.quality import bar_issues, normalize_symbol, quote_issues
from trading_agent.domain.models import Bar, MarketDataSnapshot, NewsArticle, Quote, SymbolData
from trading_agent.domain.ports import MarketDataPort, NewsPort
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import MarketDataConfig

MAX_SYMBOLS_PER_RUN = 100


class MarketDataCollector:
    def __init__(
        self,
        market_data: MarketDataPort,
        news: NewsPort,
        config: MarketDataConfig,
        *,
        max_data_age_seconds: int,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
    ) -> None:
        self._market_data = market_data
        self._news = news
        self._config = config
        self._max_age = max_data_age_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    def collect(self, symbols: Sequence[str], *, require_fresh_quotes: bool) -> MarketDataSnapshot:
        wanted = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
        if not wanted:
            raise ValueError("at least one symbol is required")
        if len(wanted) > MAX_SYMBOLS_PER_RUN:
            raise ValueError(f"too many symbols: {len(wanted)} (max {MAX_SYMBOLS_PER_RUN})")

        now = self._clock()
        quotes = self._market_data.get_latest_quotes(wanted)
        bars = self._market_data.get_daily_bars(wanted, days=self._config.daily_bars)
        news = self._news.get_news(
            wanted,
            since=now - timedelta(hours=self._config.news_lookback_hours),
            limit=self._config.max_news_articles,
        )

        collected: dict[str, SymbolData] = {}
        rejected: dict[str, str] = {}
        for symbol in wanted:
            try:
                collected[symbol] = self._build(
                    symbol,
                    quotes.get(symbol),
                    bars.get(symbol, ()),
                    news,
                    now,
                    require_fresh_quotes,
                )
            except MarketDataError as exc:
                rejected[symbol] = str(exc)
                self._log.warning("market_data.symbol_rejected", symbol=symbol, reason=str(exc))

        self._log.info(
            "market_data.collected",
            requested=len(wanted),
            collected=len(collected),
            rejected=len(rejected),
            news_articles=len(news),
            require_fresh_quotes=require_fresh_quotes,
        )
        return MarketDataSnapshot(collected_at=now, symbols=collected, rejected=rejected)

    def _build(
        self,
        symbol: str,
        quote: Quote | None,
        bars: tuple[Bar, ...],
        news: tuple[NewsArticle, ...],
        now: datetime,
        require_fresh: bool,
    ) -> SymbolData:
        problems = bar_issues(bars)
        if problems:
            raise DataQualityError("; ".join(problems))

        if quote is None:
            if require_fresh:
                raise DataQualityError("no quote available")
        else:
            problems = quote_issues(
                quote, now=now, max_age_seconds=self._max_age if require_fresh else None
            )
            if problems and require_fresh:
                raise DataQualityError("; ".join(problems))
            if problems:
                quote = None

        asset = self._market_data.get_asset(symbol)
        related = tuple(a for a in news if symbol in a.symbols)[: self._config.news_per_symbol]
        return SymbolData(asset=asset, quote=quote, daily_bars=bars, news=related)
