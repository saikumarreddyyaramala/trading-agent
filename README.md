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
| 1 | Market data and news | Next |
| 2 | MCP server (read-only tools) | Planned |
| 3 | Analyst agent | Planned |
| 4 | Risk engine | Planned |
| 5 | Order execution (paper) | Planned |
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
 agents ──► LLMGateway ──► LLMProviderPort ◄── llm/anthropic_provider.py
            │
 orchestration ──► risk engine ──► BrokerPort ◄── execution/alpaca_broker.py
            │
 bootstrap.py (composition root) wires adapters to ports from config + secrets
```

Key rules (see [ADR 0001](docs/adr/0001-architecture.md)):
- The AI has read-only tools; it can never place an order directly.
- Every model call goes through the LLM gateway (pinned model, fallback, token caps, logs).
- Config is versioned YAML; secrets come only from the environment and are redacted.
- Any failed or erroring check means "no trade".

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
2. An [Anthropic API](https://console.anthropic.com) key.

Expected output when everything is set up:

```
PASS  Configuration    env=paper, broker=alpaca (paper), model=claude-sonnet-5-5
PASS  Trading mode     paper trading (simulated money)
PASS  Kill switch      off
PASS  Secrets          all required keys present (values hidden)
PASS  Broker account   ACTIVE ****1234, equity $100,000.00, cash $100,000.00
PASS  Market clock     market closed; next open 2026-10-16 09:30 EDT
PASS  LLM              claude-sonnet-5-5 replied in 850 ms

Ready
```

## Commands

| Command | What it does |
| --- | --- |
| `make check` | Lint, type check, and unit tests (same as CI) |
| `make test-integration` | Tests against real Alpaca paper and the LLM (uses `.env`) |
| `make healthcheck` | Pre-flight check with your `.env` |
| `make docker-healthcheck` | Build the container and run the health check inside it |
| `uv run trading-agent show-config` | Print the effective config (never secrets) |

## Project layout

```
config/                  base.yaml + paper.yaml / live.yaml (versioned, validated)
src/trading_agent/
  domain/                core types and ports (no vendor SDKs)
  settings/              typed config loader and secrets
  observability/         structured logging with secret redaction
  llm/                   LLM gateway + Anthropic adapter
  execution/             Alpaca adapter (orders in Phase 5)
  data/ mcp_server/ agents/ risk/ storage/ orchestration/ reporting/   (upcoming phases)
  bootstrap.py           composition root
  health.py              pre-flight checks
  cli.py                 `trading-agent` command
tests/unit/              fast tests, no network, no keys
tests/integration/       opt-in tests against real services
docs/adr/                architecture decision records
```

## Configuration

Change behaviour in `config/*.yaml`, not in code. Examples:

```yaml
kill_switch: true            # research and report, but never place orders
risk:
  max_position_usd: 50
  stop_loss_pct: 5
```

Unknown keys, out-of-order schedule times, or a paper/live mismatch fail at startup.
