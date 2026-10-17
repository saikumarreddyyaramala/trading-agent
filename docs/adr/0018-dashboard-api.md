# ADR 0018: The dashboard backend: a local, read-only API and a live stream

- Status: Accepted
- Date: 2026-10-17

## Context

The operator wants a dashboard that shows what the system is doing, with live charts,
portfolio, news, costs, and learning, inspired by Robinhood's clarity. Phase 18 builds the
interface. This phase builds what it reads from. That data is about real accounts, so how
it is exposed matters as much as what it shows.

## Decisions

### 1. Local and read-only
- `trading-agent dashboard` serves on `127.0.0.1:8765`.
- Every route is GET. Any other method returns 405.
- No route places, changes, or cancels an order, or changes any setting.
- Secrets never appear: the account number is masked, as everywhere else.
- There is no login, so the command refuses a non-local address unless `--allow-network` is
  given explicitly.
- Responses carry `cache-control: no-store`, `x-content-type-options: nosniff`, and
  `referrer-policy: no-referrer`.

### 2. A thin web layer over a testable data layer
`dashboard/data.py` gathers each section from the audit store, the broker, market data,
and news. `dashboard/app.py` is a small Starlette app that turns those sections into JSON.
Starlette, uvicorn, and sse-starlette were already installed for the MCP server and are now
declared directly. Blocking reads run in worker threads.

| Endpoint | Shows |
| --- | --- |
| `/api/overview` | account and day change, positions with live P&L, stop and target, market clock, today's cycle steps and slots, the next scheduled step, today's LLM spend against the budget, and any source errors |
| `/api/history?span=1D,1W,1M,3M,1Y` | the equity curve, from Alpaca's portfolio history |
| `/api/trades`, `/api/performance` | system trades and their summary |
| `/api/news` | news cards (section 4) |
| `/api/watch`, `/api/plan` | the market watch and the next-day plan (ADR 0016) |
| `/api/learning` | principles, lessons, playbooks, and the guidebook (ADR 0017) |
| `/api/costs` | the cost ledger by day and agent, services, and the budget (ADR 0014), for the Settings panel |
| `/api/activity` | steps, orders, exits, lessons, and guidebook editions, newest first |
| `/api/runs/{id}` | everything recorded for one run |
| `/api/stream` | live updates as server-sent events |

### 3. Partial, never broken
Each section degrades on its own. With no broker keys, or with the broker down, the
account and positions are empty and `errors` says why, while the cycle, plan, and costs
still show. An unreadable database gives empty sections, not a 500.

Broker and market-data answers are cached for 5 seconds, so a busy page or several open
tabs can never hammer Alpaca's rate limits.

### 4. News cards: key facts by rule
`dashboard/news_cards.py` builds each card from:
- the cleaned headline;
- the first sentence of the summary, with HTML removed;
- the catalyst tags (ADR 0016);
- the symbols;
- the article's age;
- the first symbol's move today.

There is no LLM summary and no sentiment. An importance score sorts the cards: +2 for a
major catalyst, +2 for a symbol the system holds, plans, or traded, and +1 for a move of 3%
or more.

### 5. Live updates by server-sent events
`/api/stream` checks every 3 seconds. It sends the overview only when it changed, and each
new activity item once. It also sends a keep-alive ping. Server-sent events are one-way,
which fits a read-only page; WebSockets would add a write channel the dashboard does not
need.

### 6. SQLite in write-ahead mode
File-based SQLite databases now use WAL journaling, so the dashboard can read while the
cycle writes, without "database is locked" errors.

## Consequences

- Phase 18 can build the interface on a stable, tested API without touching trading code.
- The dashboard is a second process next to `trading-agent run`. Stopping it never affects
  trading.
- A catalyst tag describes the article, not each symbol in it. An article about one
  company's bankruptcy that mentions another company tags both.
- Opening the dashboard to a network needs a reverse proxy with authentication. That is
  outside this project's scope and deliberately not the default.
