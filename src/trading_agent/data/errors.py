"""Vendor-neutral market data errors. Adapters translate SDK exceptions into these."""

from __future__ import annotations


class MarketDataError(RuntimeError):
    """A data provider could not be reached or returned an unusable response."""


class DataQualityError(MarketDataError):
    """Data was returned but fails a quality or freshness rule, so it must not be used."""
