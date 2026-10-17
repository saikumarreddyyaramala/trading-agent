# ADR 0006: Order execution (paper)

- Status: Accepted
- Date: 2026-10-16

## Context

Phase 5 adds the first code that changes the broker account. It must do exactly what an
approved risk decision allows. It must survive retries, partial fills, and broker errors
without leaving a position unprotected. And it must not touch real money before Phase 11.

## Decisions

### 1. A separate write port
`OrderExecutionPort` (submit limit buy, submit stop sell, get, list open, cancel, close) is
separate from the read-only `BrokerPort`. Research tools and agents are built only from
read-only ports. The guarantee that "the AI can never place an order" holds by construction,
not by convention.

### 2. Paper only until Phase 11 (superseded by ADR 0012)
`Executor` raises `ExecutionDisabledError` for any live configuration, even with
`LIVE_TRADING_CONFIRMED=true`. Enabling real money will be a reviewed code change with its
own ADR, not a config edit.

### 3. Entry guards
`enter(decision, run_id)` refuses, before any order, when the decision is not an approval,
the kill switch is on, the run id is malformed, or the approval is older than
`execution.max_decision_age_seconds`, since prices may have moved. It submits exactly the
plan's quantity at the plan's limit price, as a DAY limit order with no extended hours.

### 4. Idempotency
Client order ids derive from the run id (`ta-<run>-entry`, `ta-<run>-stop`). Before
submitting, the executor looks the id up. A retried run finds its own order and continues;
it never submits a second one. The broker also enforces client-id uniqueness.

### 5. Fills, timeouts, and cancels
The entry is polled until it is terminal or `entry_timeout_seconds` passes. Then the rest is
cancelled, and the final state after the cancel settles is used, because fills can race
cancels. Partial fills are kept and protected for the filled quantity only.

### 6. Never unprotected
Alpaca does not support bracket or OCO orders for fractional shares. So the exits are:
- a broker-side stop-loss, placed right after the entry fills. If it cannot be placed, the
  position is closed at once (`closed_unprotected`);
- `manage()`, run periodically, which force-closes at `schedule.force_close`, takes profit
  when the bid reaches the target, and re-places a missing stop;
- `flatten()`, the emergency exit: cancel all open orders, close all positions.
Exit levels come from each position's average entry price and the config, so they need no
stored state before Phase 6. Stops are cancelled before a close, because they hold the shares.

### 7. Exits ignore the kill switch
The kill switch blocks new entries only. Exits, stop restoration, and flatten always work.

## Consequences

- Between `manage()` passes, take-profit is checked only periodically; the stop is always
  live at the broker. Phase 7 schedules `manage()` every minute or so during the session.
- Real paper orders are exercised by an opt-in test (`make test-paper-orders`), never in CI.
