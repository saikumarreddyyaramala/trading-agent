# ADR 0001: Architecture and engineering standards

- Status: Accepted
- Date: 2026-10-15

## Context

The system researches US stocks with LLM agents and places at most one small trade per day.
It handles real money (eventually), calls external vendors that fail, and makes decisions
that must be explainable afterwards. It is built the way a regulated enterprise team would
build it, not as a script.

## Decisions

### 1. Ports and adapters (hexagonal architecture)
Business logic depends only on interfaces in `domain/ports.py` (`BrokerPort`,
`LLMProviderPort`, and later market data and storage ports). Vendor SDKs live only in
adapters (`execution/`, `llm/`, `data/`). `bootstrap.py` is the single composition root.

*Why:* vendors can be swapped through config, and every component is testable with fakes.

### 2. The AI can never place an order directly
Agents get read-only tools. Orders are placed by the orchestration layer only after the
code-based risk engine approves. Any error in a check means "no trade" (fail safe).

### 3. LLM gateway
All model calls go through `LLMGateway`: pinned primary model, fallback model, token caps,
usage accounting, and structured logs. Agents never import a vendor SDK.

### 4. Configuration vs secrets
- Configuration: versioned YAML (`config/base.yaml` + `config/<env>.yaml`), strictly
  validated, immutable at runtime. Unknown keys fail at startup.
- Secrets: environment variables only (a secrets manager in the cloud, `.env` locally),
  wrapped in `SecretStr`, never committed, redacted from logs.

### 5. Two independent switches for real money
Live trading requires both `APP_ENV=live` (whose config must set `broker.paper: false`)
and `LIVE_TRADING_CONFIRMED=true`. Config validation rejects paper/live mismatches.

### 6. Observability from day one
Structured logs (JSON in production) with per-run context. Tracing and metrics are added
with the orchestration layer (Phase 7).

### 7. Quality gates
`ruff` (lint + format), `mypy --strict`, unit tests with a 90% coverage floor, secret
scanning, and a container build all run in CI on every push. Integration tests against
real services are opt-in and never run in CI.

## Consequences

- More upfront structure than a script, but each later phase plugs into a known place.
- Every decision is reproducible from logs and stored data, which is what auditors and
  interviewers ask about.

## Planned decisions
- 0002: Market data provider and data freshness rules (Phase 1)
- 0003: MCP tool design and permissions (Phase 2)
- 0004: Workflow engine for the daily cycle (Phase 7)
