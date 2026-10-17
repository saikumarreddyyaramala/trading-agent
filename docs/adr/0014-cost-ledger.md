# ADR 0014: Cost ledger, daily budget, and test-first stress tests

- Status: Accepted
- Date: 2026-10-17

## Context

Running the system costs money: LLM tokens for the analyst, critic, and reflector, and
possibly paid data plans. Until now the report estimated cost from the analysis's tokens
and one global price. That estimate missed the critic and reflector, mispriced any second
model, and could not cap spend. The operator wants every cost visible per run, and later in
the dashboard's settings.

## Decisions

### 1. Every model call is recorded and priced when it happens
`LLMGateway` reports every attempt to a listener, failed ones included. The cost ledger
(`costs/ledger.py`) prices each attempt with `costs.llm_prices` and writes it to the
`llm_calls` table (migration 0005) with:
- the run, agent, provider, and model;
- tokens, latency, and status;
- the cost in exact decimals, to the micro-dollar.

Pricing at call time keeps history correct when prices change later.

### 2. Never a guess
A model with no configured price has a NULL cost. Any total that includes it is "unknown",
shown as "$X + unpriced" with the missing models named. It is never shown as a lower
number. Dated model ids (`gpt-6.1-sol-2026-09-01`) match their base price.

### 3. A daily LLM budget
New calls are refused (`LLMBudgetExceededError`, an `LLMError`) once today's priced spend in
market time reaches `costs.daily_llm_budget_usd` ($2.00 by default). Agents already treat
an LLM error as "no decision", so a refusal means no trade, never a blind one.

The budget check fails open: if spend cannot be read, the call goes ahead. The budget
limits spending. Trading safety is the risk engine's and audit store's job. Without prices
the budget cannot count, and the health check warns about that.

### 4. The ledger never breaks a decision
A failing listener, ledger write, or budget check is logged and ignored. A cost record is
bookkeeping, unlike the audit records that gate trades (ADR 0007).

### 5. Paid services
`costs.services` lists subscriptions, such as Alpaca's free plan or a paid data plan, with
their monthly price. Each is spread evenly over every calendar day.

### 6. Where costs are shown
- `trading-agent costs`: per day, agent, and model, with services, totals, and the budget.
- Daily report: a "Costs today" section, and the analysis cost from that run's own calls.
- Scorecard: LLM cost per cycle day from the ledger, daily runs only. History recorded
  before the ledger has tokens but unknown dollars, so that criterion fails as "no data".
- Health check: an "LLM pricing" line, WARN when a configured model has no price.
- The dashboard (Phase 17-18) will read the same ledger.

### 7. Failing tests first
From this phase on, every feature starts with failing tests:
- property-based tests with Hypothesis, such as "costs add up exactly" and "the ledger
  sees every attempt the provider saw";
- stress tests, such as hundreds of micro-costs summing to the cent;
- failure injection: store down, listener raising, budget check raising, unknown models,
  and market-time midnight.

The code is written to make those tests pass. This found two real defects in this phase:
plural safeguard words in ADR 0013's filter, and half-to-even rounding of dollar amounts.

## Consequences

- The operator must enter their own prices under `costs.llm_prices`. Until then, dollars
  show as unknown and the budget cannot act, and the health check says so.
- The scorecard's cost criterion needs ledger records, so it covers days from Phase 13 on.
- `reporting.llm_usd_per_million_*` is removed. An old config using it fails at startup
  with an unknown-key error, by design.
