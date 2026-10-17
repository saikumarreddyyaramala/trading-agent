# ADR 0016: Pre-market and after-hours watch, and the next-day plan

- Status: Accepted
- Date: 2026-10-17

## Context

The cycle used to start cold at 08:45 and stop thinking at the close. A good trader
checks what moved overnight before the open, looks at after-hours news once the session
ends, and leaves a to-do list for the next morning. The operator asked for exactly that:
a pre- and post-market watch, and an end-of-day list based on what the day showed.

## Decisions

### 1. Two deterministic scans, no LLM
`watch/market_watch.py` scans a short list of symbols:
- **Pre-market**, at 08:00 ET, before the pre-flight: the latest trade against the last
  regular close before today. If today's daily bar already exists, the close before it is
  used.
- **After hours**, at 16:45 ET, or 30 minutes after an early close: the latest trade
  against today's close.

A trade from before the extended session started (04:00 or 16:00) is stale and gives no
move.

The symbol list, in this order:
1. the symbols that matter next: yesterday's plan, open positions, and today's trades and
   ideas;
2. the day's top gainers and losers (Alpaca's movers screener), at or above the minimum
   price.

It is capped at `watch.max_symbols`. A move of at least `watch.gap_alert_pct` (3%) either
way, or a major catalyst, raises an alert.

### 2. Catalysts by rule, never sentiment
`data/catalysts.py` tags news by keyword rules: earnings, guidance, analyst action, FDA,
merger, offering, legal, buyback, dividend, management, bankruptcy, split, contract, and
product. Earnings, guidance, FDA, merger, offering, and bankruptcy count as major.

There is no sentiment. A headline like "beats estimates but cuts guidance" defeats keyword
sentiment, and a wrong label is worse than none. These tags will also drive the dashboard's
news cards (Phase 17).

### 3. The next-day plan is made by rules from recorded facts
`planning/planner.py` writes a to-do list after the day's review. Each item has a kind, an
optional symbol, a sentence, and a priority. Every item comes from a recorded fact:

| Kind | Comes from |
| --- | --- |
| review | trades that hit their stop, lost, or won; ideas the risk engine rejected |
| watch | ideas the critic vetoed; after-hours alerts |
| ops | positions still open after the close, failed steps, models without a price |
| apply | lessons the daily review learned that day |

Model-written fragments (critic reasons, lesson text) are cleaned and length-capped.

The plan is for the next trading day, taken from the broker's calendar, with the next
weekday as a fallback. It is capped at `watch.max_plan_items` and stored in `day_plans`
(migration 0007).

### 4. The morning brief feeds the analyst, safely
Each analysis gets a brief with:
- the plan for today, without ops items, which are for the operator;
- the pre-market alerts and the market's move.

It holds only rule-made sentences, numbers, and catalyst tags, never a headline, so news
text cannot inject instructions. It says it is advisory, to be verified with tools. A brief
that fails to build is skipped, never a reason to skip the analysis.

### 5. Closing the loop
The evening review (ADR 0013) now sees the plan that was made for the day and the
after-hours alerts. The system can learn whether its own to-do lists help.

### 6. Bookkeeping never blocks trading
The new steps (`premarket`, `postmarket`, `plan`) are recorded and resumable like every
step. A failure is logged as an error, and trading, exits, and the report go on. A halted
day still scans and plans, because the plan is where its failures get followed up.

### 7. Where to see it
- `trading-agent watch [SYMBOLS] --phase pre|post`: scans now and records nothing.
- `trading-agent plan --date D`: the plan for a session.
- The daily report's "Market watch" and "Plan for …" sections.
- The `market_watch` table keeps every scan, headlines included, for the operator and the
  dashboard.

## Consequences

- Free IEX data has thin pre- and after-hours activity. Many symbols show no fresh trade,
  and that is reported honestly as stale, not as a zero move.
- After-hours moves are measured from the latest daily bar's close as Alpaca reports it,
  which can differ slightly from the official closing auction price.
- The plan only lists what the records support. Judgement about the plan's value comes
  from the evening review and, later, from the operator.
