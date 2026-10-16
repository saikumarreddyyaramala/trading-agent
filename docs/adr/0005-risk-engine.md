# ADR 0005: Risk engine rules and order gating

- Status: Accepted
- Date: 2026-10-16

## Context

The analyst (Phase 3) proposes a trade. Something deterministic must decide whether it happens
and how big it is, so that no LLM output can move money directly. That component must be
explainable after the fact and must say no whenever it is unsure.

## Decisions

### 1. Code decides, with its own data
`RiskEngine.evaluate(proposal)` uses only the proposal's symbol and reference price. Account,
positions, today's orders, market clock, asset, quote, and volume are fetched fresh from the
broker and data provider. The analyst cannot influence size, limits, or any check result.

### 2. Every rule runs, every result is recorded
| Check | Rule (from `config/*.yaml`) |
| --- | --- |
| Kill switch | `kill_switch` must be off |
| Trading mode | live config requires `LIVE_TRADING_CONFIRMED=true` |
| Proposal | a buy with a valid symbol and reference price |
| Market open | broker clock says open |
| Trade window | `schedule.trade_window_start`–`trade_window_end`, market time |
| Trades today | buy orders since midnight below `risk.max_trades_per_day` |
| Open positions | none: one position at a time |
| Account | active, not blocked, paper/live matches, equity ≥ `account_floor_usd` |
| Asset | tradable, active, on `allowed_exchanges`, not a fund/ETF/leveraged product |
| Price | ask ≥ `min_price_usd` |
| Quote | two-sided, not crossed, ≤ `max_data_age_seconds` old |
| Spread | ≤ `max_spread_pct` |
| Price vs proposal | ask within `max_price_deviation_pct` of the analyst's price |
| Liquidity | average volume over `volume_lookback_days` ≥ `min_avg_daily_volume` |
| Position size | sizable, ≥ $1, and a stop-out cannot breach the equity floor |

Checks do not short-circuit. A rejection shows every reason at once, and a data failure
shows up as the failed check that needed the data.

### 3. Sizing
- Budget = min(`max_position_usd`, account cash). Cash, never buying power: no margin.
- Limit price = the lower of the live ask and the analyst's price, plus
  `max_price_deviation_pct`. A stale-high reference cannot loosen it.
- Quantity is sized at the limit price, rounded down: fractional (4 decimals) when the asset
  allows it, otherwise whole shares.
- Stop-loss and take-profit prices come from the live ask with `stop_loss_pct` and
  `take_profit_pct`, rounded down to the cent.

### 4. Approval and plan are inseparable
`RiskDecision` refuses to build an approval without an `OrderPlan`, an approval with a failed
check, or a rejection that carries a plan. Execution (Phase 5) takes only an approved
decision's plan and must not exceed its quantity, notional, or limit price.

### 5. One definition of "eligible"
`risk/eligibility.py` (exchange, tradable, fund/ETF name heuristic, minimum price,
liquidity) is shared by the risk engine and the `get_most_active` research tool. The analyst
is shown only candidates the risk engine could approve. This fixes the penny stocks and
leveraged ETFs seen in the first real analyst run.

## Consequences

- A rejected trade is fully explained by its check list. An approved one carries the exact
  limits execution must respect.
- The fund heuristic is name-based and deliberately narrow. Most ETFs are also stopped by
  exchange (NYSE Arca). A misclassified operating company can be allowed by adjusting
  `exclude_funds` or the pattern, with a test.
- Day-trade limits for small live accounts (see ADR 0004) remain a Phase 11 concern. The
  one-trade-per-day rule already bounds them.
