# ADR 0012: Going live with $100, and the gates in front of it

- Status: Accepted
- Date: 2026-10-16
- Supersedes: ADR 0006 section 2 ("Paper only until Phase 11")

## Context

Phases 0-10 built and measured the system on paper. Real money adds risks that paper
hides. A config mistake becomes a real loss. Margin accounts restrict same-day round
trips (the pattern-day-trader rule). Cash accounts restrict reusing unsettled proceeds
(good-faith violations). And the system must never be pointed at an account it was not
built for.

## Decisions

### 1. Live entries need every gate to pass
`execution/live.py` checks, and `trading-agent live-check --env live` shows, each gate as
PASS or FAIL:

| Gate | Passes when |
| --- | --- |
| Live switches | live config, `broker.paper: false`, `LIVE_TRADING_CONFIRMED=true`, `storage.required: true`, kill switch off |
| Position cap | `risk.max_position_usd` ≤ the $100 hard cap |
| Paper evaluation | the newest `reports/evaluation-*.json` says GO, scores the paper record, and is ≤ `live.max_evaluation_age_days` old |
| Cash account | the live account, multiplier 1 (no margin), not blocked |
| Account size | equity ≤ `live.max_account_equity_usd` ($150) and ≥ the $70 floor |
| Settled cash | the broker reports non-marginable buying power of at least $1 |
| Live audit database | `DATABASE_URL` holds no paper runs: live gets its own database |

The gates run when the executor is built (`Container.build_executor`) and in every daily
pre-flight. A live executor built any other way cannot enter.

### 2. A hard cap in code
No live order may be worth more than `LIVE_HARD_CAP_USD` ($100), whatever the config says.
A typo such as `max_position_usd: 5000` cannot become a $5,000 order.

### 3. Size from settled, non-margin money
Every position, paper or live, is sized from the smallest of cash, buying power, and
non-marginable buying power. That never uses margin, and in a cash account it avoids
spending unsettled proceeds, which would risk a good-faith violation. One trade per day
and the end-of-day force-close are unchanged.

### 4. Failed gates block entries, never exits
A live system whose gates fail, for example because the evaluation went stale, still
manages, force-closes, and flattens positions. Only new entries stop. The daily pre-flight
fails, so the day is halted and the reason is in the report.

### 5. Separate credentials and databases
Live keys belong in their own git-ignored file, selected with `ENV_FILE=.env.live`. Live
runs go to their own `DATABASE_URL`. The paper record that justified going live stays
untouched, and the two can never be mixed in a scorecard.

### 6. The operator, not the software, takes the step
The code makes live trading possible only when the gates pass. Opening and funding the
account, creating live keys, setting `LIVE_TRADING_CONFIRMED=true`, and starting the
cycle are deliberate human actions (see the README checklist).

## Consequences

- Going live takes about two months of paper history, then about ten minutes of setup.
- An evaluation must be re-run at least weekly to keep live entries enabled. That is a
  deliberate prompt to keep looking at results.
- Rollback is immediate: `flatten --confirm`, then stop the process or set the kill switch.
