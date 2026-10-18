# ADR 0022: Chatting with the bot on Telegram

- Status: Accepted
- Date: 2026-10-17

## Context

The operator wants to message the bot and get answers: "hi", "give me the report of last
Wednesday and how many trades we did", and so on. The alert bot from ADR 0021 already
exists. The bot's username is public on Telegram, so anyone can find it and write to it.
Any chat channel into a trading system is also a way someone could try to make it act.

## Decisions

### 1. Only the owner is answered
The listener long-polls Telegram's `getUpdates` on a background thread inside
`trading-agent run`.
- It answers only the chat in `TELEGRAM_CHAT_ID`. Messages from anyone else are ignored
  and logged once per chat, without their text.
- Each update is confirmed, so nothing is answered twice.
- Messages older than `chat.max_message_age_hours`, sent while the bot was off, are
  skipped.
- A failure becomes an apology or a growing pause. The listener never stops the cycle.

### 2. Free answers by fixed rules
Common questions are answered from the bot's own records, with no AI and no cost:
- hi and status;
- positions;
- a day's report;
- trades over a period;
- costs;
- the plan.

`chat/dates.py` turns plain words into days and periods: "last wednesday", "yesterday",
"oct 12", "10/12", "this week", "last 30 days". The records come through the dashboard's
data layer and the report builder, so chat and dashboard always agree.

### 3. AI for everything else, within a cap
Other questions go to the configured LLM, recorded in the cost ledger as agent `chat`.
- Each question is sent with a compact set of records: the status, any day it mentions,
  recent trades, and the plan.
- The prompt says to answer only from those records and to admit gaps.
- At most `chat.max_ai_answers_per_day` AI answers go out per day.
- If the AI is unreachable or over budget, the bot falls back to the free answers.

### 4. Chat cannot trade
- Buy and sell requests get a fixed refusal.
- The one action is **pause**, which can only make the bot safer. It writes the day to
  `chat.pause_file`. The cycle then skips the analysis (spending no tokens), the entry,
  and the extra slots. Position management and the force-close still run.
- A pause ends by itself the next day.
- Resuming early is possible only on the computer (`trading-agent resume`), so a lost
  phone cannot restart trading.
- A damaged pause file means "not paused". The config kill switch remains the hard
  control.

### 5. Commands
- `trading-agent chat <message>` asks the same bot locally.
- `trading-agent pause` and `trading-agent resume` work from the computer.

## Consequences

- The operator can check on the bot from anywhere.
- Answers only come while `trading-agent run` is running and the PC is awake.
- Free-form answers cost a little each, and the dollar amount shows only once
  `costs.llm_prices` is set.
- Telegram sees the conversation. That is acceptable for a paper account; no keys or
  account numbers are ever sent.
