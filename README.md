# Trading Agent

An autonomous AI trading agent built to enterprise engineering standards. Each weekday it
researches US stocks, proposes one trade with evidence, passes it through a code-enforced
risk engine, executes on a broker, and reports the result against the S&P 500.

> Educational engineering project, not financial advice. Paper trading (simulated money)
> is the default; real money requires two separate, deliberate switches.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Foundations: config, secrets, logging, LLM gateway, broker connection, health check, CI | Done |
| 1 | Market data and news: quotes, daily bars, assets, news, freshness rules | Done |
| 2 | MCP server: 7 read-only research tools, input bounds, call budget, audit log | Done |
| 3 | Analyst agent: tool-calling research over MCP, evidence-verified proposals | Done |
| 4 | Risk engine: 15 code-enforced checks, cash-only sizing, eligible-only candidates | Done |
| 5 | Order execution (paper): idempotent entries, broker-side stops, managed exits | Done |
| 6 | Audit trail: runs, analyses, tool calls, risk checks, orders, events (PostgreSQL) | Done |
| 7 | Daily cycle: scheduled steps, resume after restart, holidays and early closes | Done |
| 8 | Daily report: P&L vs SPY, performance to date, Markdown/JSON, webhook | Done |
| 9 | Critic agent: independent review of every buy, veto only, code-enforced | Done |
| 10 | Evaluation: scorecard vs SPY, gatekeeper counterfactuals, go/no-go bar | Done |
| 11 | Go live ($100): live gates, hard cap, cash-account checks (operator switches on) | Done |
| 12 | Self-learning: daily review turns results into advisory, guarded lessons | Done |

## Architecture

```
            ┌────────────── domain/ports.py (interfaces) ──────────────┐
            │                                                          │
 analyst ──► MCP client ──► research tools (read-only)
 critic  ──► reviews the analyst's proposal and tool outputs (veto only)
 reflector ► reviews the finished day; writes lessons the analyst reads next morning
   │
   └──────► LLMGateway ──► LLMProviderPort ◄── llm/openai_provider.py (default)
                                             ◄── llm/anthropic_provider.py
            │
 orchestration ──► risk engine ──► BrokerPort ◄── execution/alpaca_broker.py
            │
 MarketDataCollector ──► MarketDataPort ◄── data/alpaca_market_data.py
                     └─► NewsPort       ◄── data/alpaca_news.py
            │
 bootstrap.py (composition root) wires adapters to ports from config + secrets
```

Key rules (see [ADR 0001](docs/adr/0001-architecture.md), [ADR 0002](docs/adr/0002-market-data.md),
[ADR 0003](docs/adr/0003-mcp-tools.md), [ADR 0004](docs/adr/0004-analyst-agent.md),
[ADR 0005](docs/adr/0005-risk-engine.md), [ADR 0006](docs/adr/0006-order-execution.md),
[ADR 0007](docs/adr/0007-audit-trail.md), [ADR 0008](docs/adr/0008-daily-cycle.md),
[ADR 0009](docs/adr/0009-daily-report.md), [ADR 0010](docs/adr/0010-critic-agent.md),
[ADR 0011](docs/adr/0011-evaluation.md), [ADR 0012](docs/adr/0012-going-live.md), and
[ADR 0013](docs/adr/0013-self-learning.md)):
- The AI has read-only tools; it can never place an order directly. Tool inputs are
  validated and bounded, and each session has a call budget.
- Every model call goes through the LLM gateway (pinned model, fallback, token caps, logs).
- The analyst's proposal is checked by code: every evidence item must cite a real tool call,
  and its price must match what the tools returned. Anything else means no trade.
- A second LLM, the critic, reviews every buy against the exact tool outputs. It can only
  veto; any unsupported claim, invalid answer, or outage also means no trade.
- Config is versioned YAML; secrets come only from the environment and are redacted.
- Any failed or erroring check means "no trade". The risk engine runs 15 checks with its
  own fresh data and sizes positions from cash only (no margin); an approval cannot exist
  without an order plan.
- Orders go through a separate write port that agents never receive. Execution is
  paper-only until Phase 11, every entry gets a broker-side stop immediately, and retries
  cannot double-submit.
- Every run is recorded (analysis, tool calls, risk checks, orders, exits) with a config
  snapshot. For live trading the audit store is required: no record, no trade.
- The daily cycle never lets a failure block an exit: a failed step stops new trades,
  but management, the force-close, and reconciliation always run.
- Market data must pass quality and freshness rules; a symbol with bad data is rejected.
- The system learns from its own results as advisory lessons only. Lessons can never
  change a rule, limit, or check, and code rejects any lesson that tries to.

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and optionally Docker.

```bash
git clone <your-repo-url> && cd trading-agent
make install                 # dependencies + git hooks
cp .env.example .env         # then fill in your keys (paper keys!)
make healthcheck             # verifies config, keys, Alpaca paper account, and the LLM
```

You need:
1. An [Alpaca](https://alpaca.markets) account. Use the **paper trading** API key and secret.
2. An [OpenAI API](https://platform.openai.com) key (the default LLM). To use Claude instead,
   set `llm.provider: anthropic` in `config/base.yaml` and provide `ANTHROPIC_API_KEY`.

Expected output when everything is set up:

```
PASS  Configuration    env=paper, broker=alpaca (paper), model=gpt-6.1-sol
PASS  Trading mode     paper trading (simulated money)
PASS  Kill switch      off
PASS  Secrets          all required keys present (values hidden)
PASS  Broker account   ACTIVE ****1234, equity $100,000.00, cash $100,000.00
PASS  Market clock     market closed; next open 2026-10-16 09:30 EDT
PASS  Market data      SPY bid $772.07 / ask $772.16, quote 7h 24m old, last close $771.79 (quotes=iex, bars=sip)
PASS  News             latest article 11m ago (benzinga)
PASS  LLM              gpt-6.1-sol replied in 1200 ms

Ready
```

## Commands

| Command | What it does |
| --- | --- |
| `make check` | Lint, type check, and unit tests (same as CI) |
| `make test-integration` | Tests against real Alpaca paper and the LLM (uses `.env`) |
| `make healthcheck` | Pre-flight check with your `.env` |
| `make docker-healthcheck` | Build the container and run the health check inside it |
| `uv run trading-agent market-data AAPL MSFT` | Quotes, daily bars, and news for symbols (`--fresh` enforces quote freshness, `--json` for machines) |
| `uv run trading-agent analyze` | Run the analyst, and the critic on any buy (uses LLM tokens, never trades) |
| `uv run trading-agent critique RUN_ID` | Re-review a recorded proposal with the critic |
| `uv run trading-agent risk-check AAPL --price 230` | Run every risk rule on a hypothetical buy with live data (never trades) |
| `uv run trading-agent buy AAPL --price 230 --confirm` | Risk-check and, if approved, enter on the PAPER account with a stop-loss |
| `uv run trading-agent manage` | One exit pass: force-close, take profit, keep a stop on every position |
| `uv run trading-agent flatten --confirm` | Emergency: cancel all orders, close all positions |
| `make test-paper-orders` | Round-trip a real order on the PAPER account (market hours) |
| `uv run trading-agent run` | Run the daily cycle on the PAPER account, every trading day (`--once` for today only) |
| `uv run trading-agent cycle-status` | Today's cycle: each step's status and detail |
| `uv run trading-agent report` | Today's report (`--date`, `--json`, `--send`); saved to `reports/` |
| `uv run trading-agent evaluate` | Score the paper record against the go/no-go bar (exit 0 = GO) |
| `uv run trading-agent evaluate-critic` | Replay the current critic over recorded proposals (uses tokens) |
| `uv run trading-agent lessons` | What the agent has learned (`--all` includes retired ones) |
| `uv run trading-agent retire-lesson ID --reason TEXT` | Stop the analyst from seeing a lesson |
| `uv run trading-agent learn --date 2026-10-15` | Review a finished day now (uses tokens) |
| `uv run trading-agent live-check --env live` | Check every real-money gate (read-only, places no order) |
| `uv run trading-agent db upgrade` | Create or migrate the audit database (`DATABASE_URL`) |
| `uv run trading-agent runs` | Recent runs from the audit trail |
| `uv run trading-agent show-run RUN_ID` | Everything recorded for one run (JSON) |
| `uv run trading-agent sync-orders` | Refresh recorded orders with final statuses and fills |
| `uv run trading-agent mcp-server` | Serve the read-only research tools over MCP (stdio) |
| `uv run trading-agent show-config` | Print the effective config (never secrets) |

## Project layout

```
config/                  base.yaml + paper.yaml / live.yaml (versioned, validated)
src/trading_agent/
  domain/                core types and ports (no vendor SDKs)
  settings/              typed config loader and secrets
  observability/         structured logging with secret redaction
  llm/                   LLM gateway + OpenAI and Anthropic adapters
  execution/             read-only broker adapter, order adapter, executor (paper)
  data/                  market data + news adapters, quality/freshness rules, collector
  mcp_server/            read-only research tools + MCP server (stdio)
  agents/                analyst (tool-calling research, evidence checks) and critic (veto-only review)
  risk/                  eligibility rules, risk engine, position sizing
  storage/               audit trail: schema, Alembic migrations, store
  orchestration/         the daily cycle: schedule, steps, resume, market calendar
  reporting/             daily report: trade P&L, SPY benchmark, Markdown/JSON, webhook
  evaluation/            scorecard, go/no-go criteria, counterfactuals, critic replay
  learning/              daily review: day record, lessons kept, retired, and expired
  bootstrap.py           composition root
  health.py              pre-flight checks
  cli.py                 `trading-agent` command
tests/unit/              fast tests, no network, no keys
tests/integration/       opt-in tests against real services
docs/adr/                architecture decision records
```

## The daily cycle

`trading-agent run` (or `docker compose up -d`) runs one cycle per trading day, in market time:

| Step | When | What |
| --- | --- | --- |
| preflight | 08:30 | health check; a failure halts new trades for the day |
| analysis | 08:45 | the analyst researches and proposes one trade or none |
| critique | after analysis | the critic reviews a buy against the evidence; it can only veto |
| entry | 09:45 | risk check with fresh data; paper entry with a stop-loss if approved |
| manage | every minute | take profit, keep the stop in place |
| close | 15:50 | force-close and verify the account is flat (earlier on half-days) |
| reconcile | 16:15 | sync final order statuses and fills |
| reflect | after reconcile | review the day and update the analyst's lessons (never affects trades) |
| report | after reflect | daily report saved to `reports/`, and sent to `REPORT_WEBHOOK_URL` if set |

Each step is recorded once, so a restarted process resumes where it left off.

## Daily report

Each trading day ends with a report (also on demand: `trading-agent report --date 2026-10-15`):
the analysis and its evidence, the risk decision, the trade's fills and P&L, the same-day
SPY open-to-close move for comparison, and performance to date. Only trades placed by the
daily cycle count toward performance. Set your LLM prices under `reporting:` in
`config/base.yaml` to see cost in dollars as well as tokens.

## Self-learning

After each trading day the reflect step reviews what actually happened: the proposal,
the critic's verdict, the risk checks, the trade against SPY, and what a stopped idea
would have returned. It writes at most two short lessons, for example "check pre-market
volume before trusting a gap-up". The analyst reads the active lessons the next morning,
after its rules, as advice.

The guardrails (ADR 0013):
- Lessons cannot change rules, limits, sizing, stops, or the critic. Code rejects any
  lesson that tries to.
- The reviewer never sees raw news, so a headline cannot plant a lasting instruction.
- Memory stays small: at most 15 active lessons, each expiring after 60 days.
  Contradicted lessons are retired.

Every analysis records which lessons it used, and the daily report shows what was
learned and retired. Review them with `trading-agent lessons`. Turn learning off with
`learning.enabled: false`.

## Evaluation: the bar for real money

`trading-agent evaluate` scores the paper record: the funnel from analyses to trades,
performance against SPY's same-day move (with a bootstrap confidence interval), what the
critic and risk engine stopped and what that would have returned, and LLM cost. It checks
the go/no-go bar written in advance in `config/base.yaml` (see ADR 0011): at least 40 cycle
days and 20 trades, a positive mean excess return over SPY, a drawdown within $15, and more.
Every criterion must pass, and missing data counts as a fail.

## Going live (you take these steps, not the software)

Live entries are possible only when every gate in ADR 0012 passes. Exits always work.

1. **Earn it on paper.** Run the daily cycle on paper for about two months, then
   `uv run trading-agent evaluate`. Continue only if it says **GO** (exit code 0).
2. **Open a cash account** at Alpaca (not margin) and fund it with about $100.
3. **Keep live credentials separate:** create `.env.live` with the live API keys, a separate
   `DATABASE_URL` for live, and `LIVE_TRADING_CONFIRMED=true`. Use it with `ENV_FILE=.env.live`.
4. **Create the live database:** `ENV_FILE=.env.live uv run trading-agent db upgrade --env live`.
5. **Check every gate:** `ENV_FILE=.env.live uv run trading-agent live-check --env live`.
   All must PASS.
6. **Start:** `ENV_FILE=.env.live uv run trading-agent run --env live`. Watch the daily reports.
7. **Stay current:** re-run `evaluate` on the paper record at least weekly. A stale GO
   blocks new live entries.
8. **Stop at any time:** `ENV_FILE=.env.live uv run trading-agent flatten --confirm --env live`,
   then stop the process or set `kill_switch: true`.

Guardrails in code: one trade a day, at most $100 per order whatever the config says,
sizing from settled non-margin cash only, a forced close before the bell, and every
decision recorded. This is still an educational project, not financial advice. Only risk
money you can afford to lose.

## Audit database

Set `DATABASE_URL` in `.env`, then create the schema:

```bash
uv run trading-agent db upgrade
```

- With Docker: `docker compose up db` and `postgresql://trading:trading@localhost:5432/trading`.
- Without Docker: `sqlite:///data/audit.db` works for local use (PostgreSQL is the production target).

The audit store is optional for paper trading and required for live trading.

## Research tools over MCP

The agent's research tools are also an MCP server, so you can explore them from any MCP host.
For example, to add them to Claude Code (run from the repo root, with `.env` filled in):

```bash
claude mcp add trading-research -- uv run --directory "$(pwd)" trading-agent mcp-server
```

Tools: `market_status`, `get_quote`, `get_daily_bars`, `get_news`, `get_asset`,
`get_most_active`, `research_snapshot`. All are read-only; none can trade.

## Configuration

Change behaviour in `config/*.yaml`, not in code. Examples:

```yaml
kill_switch: true            # research and report, but never place orders
llm:
  provider: openai           # or anthropic; no code changes needed
  model: gpt-6.1-sol
risk:
  max_position_usd: 50
  stop_loss_pct: 5
```

Unknown keys, out-of-order schedule times, or a paper/live mismatch fail at startup.
