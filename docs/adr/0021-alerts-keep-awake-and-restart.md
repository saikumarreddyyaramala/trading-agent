# ADR 0021: Phone alerts, keep-awake, and crash restart

- Status: Accepted
- Date: 2026-10-17

## Context

The bot runs on the operator's Windows laptop. Three things could stop paper trading
without anyone noticing:
- the computer falls asleep;
- the process crashes;
- something goes wrong while the operator is away.

The operator chose Telegram for phone alerts and decided against auto-start at logon.

## Decisions

### 1. Telegram alerts, off unless both secrets are set
`alerts/telegram.py` is a small client for the Telegram Bot API. It uses `sendMessage` in
plain text, so nothing in a message is read as markup, and `getUpdates` to find the chat
id.
- Messages are cut at Telegram's 4096-character limit.
- The bot token is part of every request URL, so no error carries the URL, the token, or
  the original exception.
- Without `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, alerts are silently off.
- `trading-agent alerts-setup` finds the chat id and `trading-agent alerts-test` sends one
  message.

### 2. What is sent
The alert kinds come from `alerts.events`:

| Kind | When |
| --- | --- |
| start | the cycle starts |
| trade | a buy filled |
| exit | take-profit, force-close, or last-resort flatten |
| failure | a step errors; a position cannot be closed; a stop-loss had to be restored |
| halt | the pre-flight check failed, so no new trades today |
| report | the day's result, once per day (not again after a restart) |
| crash | the cycle crashed and is restarting, or gave up |

Every message is labelled with the account, for example `[paper]`.

### 3. Alerts can never hurt trading
The alerter drops kinds that are not wanted. It also drops repeats of the same text
within `dedupe_minutes`, and caps alerts at `max_per_hour` so a failure loop cannot flood
the phone. It never raises: a failed send is logged and the cycle carries on. The cycle
guards the alert call too. There is one alerter per process, so the dedupe and the cap
cover every alert.

### 4. Keep-awake for this process only
While `trading-agent run` runs on Windows, it asks the system not to sleep when idle, using
`SetThreadExecutionState` with `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`. The request is
released when the program stops, even after a crash.
- No power setting is changed.
- The display may still turn off.
- A sleep the user chooses, such as closing the lid or pressing sleep, still happens.
- After waking, the cycle's waits check the clock again, so it carries on.
- Setting: `orchestration.keep_awake`.

### 5. A crash supervisor
Each step already contains its own failures. `supervise()` is the last line for anything
that escapes the cycle, such as a locked database or a bug.
- It alerts and waits, starting at 1 minute and doubling up to 15 minutes, then starts the
  cycle again. A run that lasted an hour resets the wait.
- Restarting is safe because finished steps are never repeated.
- After `orchestration.max_crashes_per_hour` crashes in an hour, it sends a final alert
  and exits with code 1.
- Ctrl+C is not a crash.
- Open positions keep their broker-side stop-loss throughout.

### 6. One cycle at a time
A weekend audit found two copies of `trading-agent run` running at once. On a trading day
each would analyse and enter trades, and Telegram rejected their competing listeners
(HTTP 409). `run` now holds an exclusive operating-system lock on
`orchestration.lock_file`. The system releases it when the process ends, even on a crash,
so a stale lock is impossible. A second copy refuses to start and names the holder.

### 7. No auto-start
The operator starts `trading-agent run` by hand. Nothing here changes Windows settings or
adds startup entries.

## Consequences

- The operator hears about trades and problems wherever they are, at no cost.
- A laptop with its lid closed still sleeps. The bot cannot trade while asleep, but it
  resumes on wake, and broker-side stops protect open positions.
- Telegram's servers see the alert text (symbols, prices, statuses). That is acceptable
  for a paper account; alerts contain no keys or account numbers.
