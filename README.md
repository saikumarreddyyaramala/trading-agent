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
| 5 | Order execution (paper) | Next |
| 6 | PostgreSQL audit trail | Planned |
| 7 | Orchestration and scheduling | Planned |
| 8 | Daily report | Planned |
| 9 | Critic agent | Planned |
| 10 | Evaluation | Planned |
| 11 | Go live ($100) | Planned |

## Architecture

```
            ┌────────────── domain/ports.py (interfaces) ──────────────┐
            │                                                          │
 agents ──► MCP client ──► research tools (read-only)
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
and [ADR 0005](docs/adr/0005-risk-engine.md)):
- The AI has read-only tools; it can never place an order directly. Tool inputs are
  validated and bounded, and each session has a call budget.
- Every model call goes through the LLM gateway (pinned model, fallback, token caps, logs).
- The analyst's proposal is checked by code: every evidence item must cite a real tool call,
  and its price must match what the tools returned. Anything else means no trade.
- Config is versioned YAML; secrets come only from the environment and are redacted.
- Any failed or erroring check means "no trade". The risk engine runs 15 checks with its
  own fresh data and sizes positions from cash only (no margin); an approval cannot exist
  without an order plan.
- Market data must pass quality and freshness rules; a symbol with bad data is rejected.

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
| `uv run trading-agent analyze` | Run the analyst: research and propose one trade or none (uses LLM tokens, never trades) |
| `uv run trading-agent risk-check AAPL --price 230` | Run every risk rule on a hypothetical buy with live data (never trades) |
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
  execution/             Alpaca adapter (orders in Phase 5)
  data/                  market data + news adapters, quality/freshness rules, collector
  mcp_server/            read-only research tools + MCP server (stdio)
  agents/                analyst agent: prompts, tool-calling loop, evidence checks
  risk/                  eligibility rules, risk engine, position sizing
  storage/ orchestration/ reporting/   (upcoming phases)
  bootstrap.py           composition root
  health.py              pre-flight checks
  cli.py                 `trading-agent` command
tests/unit/              fast tests, no network, no keys
tests/integration/       opt-in tests against real services
docs/adr/                architecture decision records
```

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
