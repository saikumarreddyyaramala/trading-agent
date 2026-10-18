# ADR 0023: The all-day market journal and the day notebook

- Status: Accepted
- Date: 2026-10-18

## Context

The operator wants the bot active from the start of pre-market (04:00 New York) to the end
of post-market (20:00). Outside regular hours it should **watch and record only**, never
trade: more data now means better analysis later. It should keep separate records for
pre-market, the regular session, and post-market, plus an overall report. Those records
should be readable by people and by machines, and should connect the dots: "this went up
because of that news, or that policy change". About 20 to 30 minutes after post-market
ends, it writes the day's notebook. Free sources come first; a paid source may come later.
AI work runs after each session. The output is files on the PC.

## Decisions

### 1. A recorder on its own thread, separate from trading
`journal/recorder.py` runs inside `trading-agent run` on a background thread, like the
Telegram chat. It never touches orders. Every failure is recorded in the day's files and
contained. After a restart it continues from those files.

### 2. The day's timeline comes from the broker's calendar
`BrokerPort.get_market_day` (Alpaca's calendar) gives each day's regular session, so
holidays have no journal and early closes are handled. On a normal day:

| Time (New York) | What happens |
| --- | --- |
| 04:00 to 09:30 | pre-market: a snapshot every 30 minutes |
| 09:30 | note on the pre-market; regular-session snapshots begin |
| 16:00 | note on the regular session; post-market snapshots begin |
| 20:00 (close + 4 h) | note on the post-market |
| 20:20 | the day notebook |
| 20:30 | the cycle's evening steps (after-hours scan, lessons, guidebook, plan, report) |

On an early close at 13:00, post-market ends at 17:00 and the notebook comes at 17:20.

### 3. What a snapshot holds
- The market indexes (SPY, QQQ, IWM, DIA).
- The bot's stocks (the plan, positions, today's trades and ideas) and the day's movers.
  The market watch from ADR 0016 measures them: pre-market and regular hours against the
  last close, post-market against today's close.
- In regular hours, the bot's trades.
- New items from the sources, each kept once per day and fetched since the previous
  snapshot. The first fetch of the day looks back overnight.

### 4. Free sources, with room for a paid one
`journal/sources.py` defines one `fetch(since)` per source:
- company news from Alpaca/Benzinga, for the market and for the watched stocks;
- Federal Reserve press releases (RSS);
- the Bureau of Labor Statistics latest numbers (RSS);
- the Federal Register (JSON API): presidential documents and significant rules only.

XML is parsed with defusedxml, so a hostile feed is refused. A paid source later is one
more class. All source text is untrusted third-party data.

### 5. AI notes that must cite their sources
After each session, `journal/writer.py` asks the LLM for a note: a summary, connections,
and what to watch next. At the notebook time it asks for the day's story, connections,
lessons, what to watch tomorrow, and open questions.
- Each connection names its effect, likely cause, source IDs, and confidence.
- Code checks every cited ID against the record and drops invented ones. A connection
  with no real source is marked unverified.
- The prompt says a move and news at the same moment is a likely explanation, not proof.
- One repair is allowed for a malformed answer. A failed note is retried twice more,
  30 minutes apart.
- With no LLM key, or when the AI fails, the notebook still holds the day's facts, built
  by fixed rules.
- About 4 AI calls a day, recorded in the cost ledger as agent `journal`.

### 6. Files for people and for machines
Each day has a folder, `journal/<day>/`, holding `premarket`, `market`, `postmarket`, and
`day` (the notebook). Each comes as `.md` to read and `.json` for data. Writes are atomic,
going to a temporary file first and then replacing the old one, and a damaged file is
set aside rather than lost. `trading-agent journal [--date]` prints the notebook.
`trading-agent journal-check` is a dry run of every source and the market scan, with
nothing saved and no AI used.

### 7. The notebook does not reach the trading agents
The analyst's morning brief carries no third-party text, by design (ADR 0016). The
notebook is built from news, so it stays a record for the operator. Feeding it into the
trading prompts would open a path for injected instructions.

### 8. The evening steps move after post-market
`schedule.post_market` is now 20:30, so the after-hours scan, lessons, guidebook, next-day
plan, and daily report cover the whole day. The day's trading result is still sent to
Telegram when the session's result is recorded, at about 16:15.

## Consequences

- The bot now runs about 16 hours a day. The laptop must stay awake (ADR 0021).
- The journal adds a few free API calls every 30 minutes and about 4 AI calls a day.
- Explanations are evidence-backed hypotheses, labelled with confidence. They are not
  proof of cause.
