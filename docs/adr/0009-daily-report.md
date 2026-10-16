# ADR 0009: Daily report contents and delivery

- Status: Accepted
- Date: 2026-10-16

## Context

The system runs unattended (Phase 7). A person needs to see each day what it did, why, and
whether it is any good, without reading logs or querying the database. "Any good" needs a
fair benchmark and an honest count of cost.

## Decisions

### 1. Reports are derived views
A report is rebuilt from the audit trail, a broker account snapshot, and benchmark bars.
Nothing new is stored, so any past day can be regenerated (`trading-agent report --date`).
A report can never disagree with the records it summarises.

### 2. Contents
- The cycle: status, summary, and each step's outcome.
- The analysis: decision, thesis, cited evidence, risks, tool calls, tokens.
- The risk check: approved, or which checks failed.
- The trade: quantity, entry and exit fills, P&L, return, and how it exited (stop-loss or
  exit order), plus exit events.
- Benchmark and performance to date: trade count, win rate, total P&L, and average return
  against the benchmark.
- Account equity, and any warnings about missing data.

### 3. A fair benchmark
Positions are intraday, so each trade is compared with SPY's **open-to-close** move on the
same day. "Excess return" is the trade's return minus that move. The benchmark symbol is
configurable (`reporting.benchmark_symbol`).

### 4. Only the system's own record counts
Trades are rebuilt from filled orders. Only entries placed by the daily cycle (`daily-*` run
ids) count toward performance. Manual `buy` commands and test round trips are excluded.

### 5. Cost without guessing
Token usage is always shown. A dollar cost appears only if the operator configures their
actual per-token prices. The report never shows an assumed price.

### 6. Graceful with missing data
An unreachable broker, data provider, or database becomes a warning in the report, never a
failed report. A missing benchmark bar for a past trading day is flagged explicitly.

### 7. Delivery
Each report is saved as `reports/<date>.md` (readable on GitHub or any Markdown viewer) and
`reports/<date>.json` (for tooling). If `REPORT_WEBHOOK_URL` is set, the Markdown is posted
to a Slack-compatible incoming webhook. The URL is a secret: it is never logged, and error
messages carry only the HTTP status or exception type. The daily cycle runs this as its
last step, after recording the day's result.

## Consequences

- The report is only as complete as the audit trail. Without `DATABASE_URL` it shows the
  benchmark and equity but no cycle details.
- Performance statistics become meaningful only after weeks of daily runs. Phase 10
  (evaluation) builds on the same queries.

## Fixed along the way
SQLite stored timezone-aware timestamps as local wall time with the zone dropped. Times
recorded in market time were read back four hours off. Timestamps are now converted to UTC
before storage on every backend, and naive timestamps are refused.
