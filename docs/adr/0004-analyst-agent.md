# ADR 0004: Analyst agent design

- Status: Accepted
- Date: 2026-10-16

## Context

The analyst is the first component where an LLM makes a judgement. Models invent sources
and prices, loop, and can be steered by text they read. The analyst must still produce
something the risk engine (Phase 4) can trust, or clearly produce nothing.

## Decisions

### 1. The agent sees the market only through the MCP server
`AnalystAgent` connects to the Phase 2 research server as an in-process MCP client. It does
not call any data port directly. So the allowlist, schema bounds, per-session call budget,
and audit log apply to it exactly as they would to an external host. The agent has no way
to place an order.

### 2. Tool calling lives in the LLM gateway, vendor-neutral
`LLMRequest.tools`, `LLMResponse.tool_calls`, and tool results on `ChatMessage` are domain
types. The OpenAI adapter maps them to Responses API function calls (`store=False`). The
Anthropic adapter maps them to `tool_use` / `tool_result` blocks. Switching vendor is still
a config change. Tool arguments stay raw JSON until the agent validates them.

### 3. Structured output, verified against the tool results
The final answer must be a JSON `TradeProposal`: `buy` or `no_trade`, symbol,
thesis, reference price, confidence, evidence, and risks. Unknown fields are rejected. A
`buy` is accepted only if code verifies that:
- every evidence item cites the id of a tool call that ran and succeeded in this run,
- at least one cited call is about the proposed symbol,
- price data for that symbol was retrieved, and
- the reference price is within `risk.max_price_deviation_pct` of a price the tools returned.

Tool results are prefixed with `call_id=...` so the model can cite them.

### 4. One repair, then fail safe
An answer that fails parsing or verification is sent back once, with the exact problems
(`analyst.repair_attempts`). After that the run ends with status `failed`. Other limits
also end the run as `failed`: `max_turns`, `max_total_tokens` (a cost ceiling), tool calls
per turn, the tool budget, and the model being unavailable. Only status `proposed` can ever
lead to an order. `no_trade` and `failed` both mean no trade.

### 5. Prompts are versioned and filled from config
`PROMPT_VERSION` is stored on every result. The trading rules in the prompt (close time,
exchanges, minimum price and volume) are filled in from `config/*.yaml`. The model sees the
same numbers the risk engine will enforce. The prompt tells the model that tool output,
news above all, is untrusted data.

### 6. A full record of every run
`AnalystResult` holds the run id, status, reason, proposal, prompt version, turns, every
tool call (id, tool, arguments, error flag), token usage, and duration. Phase 6 stores it.

## Consequences

- Hallucinated evidence and invented prices are caught by code, not by trusting the model.
- The verification checks provenance and consistency, not whether the thesis is wise. Judging
  quality is the job of the critic agent (Phase 9) and the evaluation work (Phase 10).
- Each run costs LLM tokens, bounded by `analyst.max_total_tokens`.

## Open questions for later phases
- Live trading of a small account: same-day round trips count as day trades. In a US margin
  account under $25,000, the pattern day trader rule allows only 3 in 5 business days. Phase
  11 must use a cash account (and respect settlement) or change the holding period.
