# ADR 0003: MCP tool design and permissions

- Status: Accepted
- Date: 2026-10-16

## Context

From Phase 3 on, LLM agents research stocks by calling tools. A model can be wrong, can loop,
and can be steered by text it reads (prompt injection through news). So the tools must be
safe no matter what the model asks for. The safety must come from code, not from the prompt.

## Decisions

### 1. One tool layer, two ways to reach it
`mcp_server/tools.py` (`ResearchTools`) is plain Python built on the read-only ports.
`mcp_server/server.py` adapts it to the Model Context Protocol, so any MCP host (Claude Code,
Claude Desktop, an agent SDK) can use it. Phase 3 agents can also call it in-process. Both
paths share the same validation, budget, and audit logging.

### 2. Read-only by construction
| Tool | Purpose |
| --- | --- |
| `market_status` | Open/closed, next open and close |
| `get_quote` | Bid/ask with age and freshness issues |
| `get_daily_bars` | OHLCV history plus a summary |
| `get_news` | Headlines and summaries, labelled untrusted |
| `get_asset` | Exchange, status, tradability |
| `get_most_active` | Liquid candidates by share volume |
| `research_snapshot` | One-call overview of up to 10 symbols |

- The tools receive read-only ports and a market clock function. They never get the broker
  object, so there is nothing to call that could place an order.
- Only names in `ALLOWED_TOOLS` may be registered. A test fails if the list changes or if a
  name suggests trading. Adding a tool is a reviewed code change.
- Every tool carries MCP annotations `readOnlyHint=true`, `destructiveHint=false`.
- The account (equity, cash) is deliberately not exposed: position sizing belongs to the
  risk engine, not the model (least privilege).

### 3. Validate everything the model sends
- Symbols must match a strict ticker pattern before any request leaves the process.
- Every count has a hard upper bound (bars ≤ 120, news ≤ 20 articles over ≤ 168 h,
  most active ≤ 50, snapshot ≤ 10 symbols). The bounds appear in the JSON schema, so the
  SDK rejects bad arguments early, and the tool layer checks them again.

### 4. Budget per session
Each `ResearchTools` instance (one per agent run or MCP session) allows
`tools.max_calls_per_session` calls (default 100). Failed calls count too. A looping agent
gets a clear "budget spent" error instead of draining the data provider's rate limit.

### 5. Errors the model can act on, crashes it cannot see
Anticipated failures (bad input, no data, provider error, budget) come back to the model as
tool errors with a clear message. Unexpected exceptions reach it only as a generic
"Error executing tool". Internal details stay in the server logs.

### 6. Audit trail
Every call logs `tool.called` with tool name, arguments, outcome
(`ok`, `rejected`, `data_error`, `crash`), duration, and calls used. Phase 6 persists
these to PostgreSQL alongside the decisions they informed.

### 7. stdio transport only
The host launches `trading-agent mcp-server` as a subprocess. There is no network listener,
so there is no authentication surface to secure. stdout carries the protocol, so all logs
go to stderr. An HTTP transport would need its own ADR (authentication, TLS, rate limits).

## Consequences

- The model can research freely but cannot affect the account. The worst it can do is waste
  its own call budget.
- New capabilities require a code change, a test update, and review. That friction is
  intentional.
- Prices are returned as strings to avoid float rounding. The model reads them fine, and
  later code parses them back to `Decimal`.
