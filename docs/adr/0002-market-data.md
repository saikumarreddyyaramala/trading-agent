# ADR 0002: Market data provider and data freshness rules

- Status: Accepted
- Date: 2026-10-15

## Context

The analyst agent (Phase 3) and the risk engine (Phase 4) need prices, recent history,
reference data (exchange, tradability), and news. Decisions made on stale or malformed data
are the most likely way for an automated trader to lose money, so the rules for "usable data"
must be explicit, testable, and shared by every component.

## Decisions

### 1. Alpaca for prices, reference data, and news
Alpaca's market data API (and its Benzinga news feed) uses the same keys as the broker, so
Phase 1 adds no vendor, contract, or secret. The data sits behind two new ports,
`MarketDataPort` and `NewsPort`. A different vendor (Polygon, Tiingo, ...) is a new adapter
in `data/` plus a `market_data.provider` value; no business code changes.

### 2. Feed per data type
| Data | Feed | Why |
| --- | --- | --- |
| Latest quote | `iex` (free, real-time) | Good enough for a top-of-book sanity check on liquid stocks |
| Daily bars | `sip` (free when >15 min old) | All US exchanges; IEX-only volume understates liquidity ~30x |
| News | Benzinga via Alpaca | Headline and summary only; bodies are skipped |

Bars requests end 16 minutes before "now" on the SIP feed to stay within the free plan. Both
feeds are config values, so a paid subscription only needs a YAML change.

### 3. Quality and freshness rules (`data/quality.py`)
- Quote: both sides positive, not crossed, timestamp not in the future (5 s skew allowed), and
  no older than `risk.max_data_age_seconds` when freshness is required.
- Bars: present, strictly ascending in time, positive prices, consistent high/low, volume >= 0.
- Symbols: validated against a strict ticker pattern before any request. Symbols will later
  come from an LLM, so input is never trusted.

### 4. Fail safe, per symbol
`MarketDataCollector` rejects a single symbol with a recorded reason and continues with the
rest. A provider outage (quotes, bars, or news call fails) fails the whole run.
Freshness is enforced only when a decision is being made (`require_fresh_quotes=True`).
Pre-market research keeps the symbol and drops an unusable quote instead.

### 5. News is untrusted text
Headlines and summaries are whitespace-normalised and length-capped. They will reach an LLM
in Phase 3, which must treat them as data, never as instructions.

## Consequences

- No new costs or keys. The free IEX quote is thin outside regular hours, which the
  pre-market mode tolerates by design.
- The risk engine reuses `quote_issues` before every order, so collection and execution apply
  the same definition of "fresh".
- The health check now proves the data feeds work (an SPY quote and bars, plus the news feed),
  and fails on a stale quote while the market is open.
