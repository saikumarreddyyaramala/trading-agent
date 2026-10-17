# ADR 0015: The parametric trade score, and up to three paper trades a day

- Status: Accepted
- Date: 2026-10-17

## Context

The operator wants the system to focus on profitable trades: every execution should pass
a parametric evaluation, not only the safety checks. They also want more paper data faster
than one trade a day gives, while live stays at one trade.

No rule set can promise profit. A deterministic score can still do two useful things. It
keeps out trades whose structure is poor, before any model's opinion matters. And it lets
the system's own results count against the kind of trade it keeps losing on.

## Decisions

### 1. Every buy is scored by fixed rules
`risk/scoring.py` scores a buy from 0 to 100. The score is a "Trade score" check inside
the risk engine, so it is audited like every other check. It is stored with the decision
(`risk_decisions.score`, migration 0006).

| Part | Measures | 0 | 1 |
| --- | --- | --- | --- |
| Reward/risk | (target − 2×spread) ÷ (stop + 2×spread) | ≤ 1 | ≥ 2 |
| Stop vs volatility | stop distance in ATRs (14-day average true range) | 0.75 | ≥ 1.5 |
| Trend | price above its 20-day average; 5-day move up | neither | both |
| Extension | (price − 20-day average) ÷ ATR | ≥ 3 | ≤ 1 |
| Confidence | the analyst's own label (small weight) | low 0.2 | high 1 |
| History | mean return of the system's last 50 closed trades | −stop% | +stop% |

The weights are 20/20/25/15/10/10, set in `scoring.weights`. A trade needs a total of at
least `scoring.min_score` (60).

### 2. Hard fails override any total
A trade is rejected whatever its score when:
- reward/risk after the spread is below 1.5;
- the stop is less than 0.75 daily ranges away, so normal noise would hit it;
- the price is more than 3 ranges above its 20-day average;
- there are fewer than 21 daily bars, or no daily range at all.

### 3. History counts only once it means something
Before 10 closed trades, history is neutral (0.5). Unreadable history is also neutral,
and logged. It never rejects a trade, and it never passes one by default. Once there is
enough, the expected value shown is the plain mean return of recent trades. That number
describes the past. It is not a forecast.

### 4. Up to three paper trades a day, as separate slots
- On paper, `risk.max_trades_per_day` and `risk.max_open_positions` are 3.
- `schedule.extra_analyses` (10:30 and 12:00) add analysis-and-entry slots after the
  08:45 one. The number of slots is capped by `max_trades_per_day`.
- Each slot is its own run, `daily-YYYYMMDD-2` and `-3`, with its own analysis, critique,
  risk check, score, and entry. So every existing audit rule applies per trade, and order
  ids stay unique and idempotent.
- A later slot is told what was already bought and asked for a different symbol.
- A slot is skipped before spending tokens when positions or trades are at their cap, the
  kill switch is on, or the day is halted.
- Positions are managed between slots. A slot's own failure is contained, and the next
  slot still runs.
- On restart, finished slots are skipped, and a slot that was cut short resumes its own
  steps.
- The risk engine never adds to a symbol already held. Its equity-floor check now counts
  the stop-loss risk of every open position.

### 5. Live stays at one
The config refuses `environment: live` with more than one trade a day or more than one
position. More trades are for gathering paper data only. Changing that for live needs a
new decision.

### 6. Reporting
- The daily report lists every trade, has a "Trade slots" table, and shows the score with
  its weakest parts.
- The scorecard counts days from the daily runs and everything else across all slots,
  including the stopped-idea counterfactuals.
- The learning review sees every slot.
- `trading-agent risk-check` prints the full score breakdown.

## Consequences

- Up to three analyses a day means about three times the LLM spend on paper. The daily
  budget from ADR 0014 still caps it.
- The fixed 5% stop and 10% target are often far from the typical daily range of liquid
  stocks. The score's details make that visible; tuning them is future work.
- Paper results now mix up to three concurrent positions, while live will hold one.
  Per-trade statistics carry over; daily P&L and drawdown are not directly comparable.
- The score's parameters are judgement calls and are documented here. Change them in
  `config/base.yaml` and keep the change in git.
