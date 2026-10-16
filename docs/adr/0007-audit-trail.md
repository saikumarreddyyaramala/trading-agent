# ADR 0007: Audit trail storage

- Status: Accepted
- Date: 2026-10-16

## Context

Until Phase 6 the system's decisions lived only in logs. To answer "why did it buy X on
Tuesday, with what evidence, and what happened?", and to evaluate the agent later (Phases
8-10), every run must be stored in a queryable, durable form.

## Decisions

### 1. PostgreSQL, through SQLAlchemy Core, with SQLite for local use
PostgreSQL is the production store (`docker-compose.yml`, a CI service container). The
schema uses portable types: an exact decimal type (NUMERIC on PostgreSQL, text on SQLite),
JSON that becomes JSONB on PostgreSQL, and UTC timestamps. So `sqlite:///data/audit.db`
works with no setup. SQLite foreign keys are switched on per connection, so both backends
enforce the same integrity.

### 2. Alembic migrations are the schema of record
Migrations ship inside the package and run with `trading-agent db upgrade`. In compose,
a one-shot `migrate` service runs before the agent. A test fails if the code's schema and
the migrations drift apart, on both SQLite and PostgreSQL.

### 3. One run id ties everything together
| Table | Holds |
| --- | --- |
| `runs` | kind, environment, app version, timings, outcome, **config snapshot** |
| `analyses` | status, proposal, prompt version, turns, token usage |
| `tool_calls` | every research call: tool, arguments, error flag |
| `risk_decisions` | approval, every check, the order plan |
| `orders` | broker order id, client id, role, status, fills |
| `position_events` | exits, stop restores, flattens |

Orders find their run from their client order id (`ta-<run>-entry` / `-stop`). Orders the
system did not place, such as a market close, are kept without a run.

### 4. Append-only, except order status
Records are inserted and never rewritten. Orders are the exception: their status and fills
change at the broker, so `sync-orders` updates them in place with an `updated_at`.

### 5. No record, no trade (when required)
`storage.required` decides what a failed write means. If true, the command stops before
placing an order. If false, the failure is logged as an error and the command carries on.
It is false for paper, so local use needs no database. The config validator refuses a live
environment unless it is true.

### 6. Secrets stay out
`DATABASE_URL` is a secret. The stored config snapshot is the validated `AppConfig`, which
never contains credentials.

## Consequences

- A run can be fully reconstructed with `trading-agent show-run <id>`. Reporting (Phase 8)
  and evaluation (Phase 10) query the same tables.
- Only state-changing `manage` passes are recorded; minute-by-minute "hold" results are not.
- Phase 7's scheduler must also call `sync-orders` after the close, so exit fills are final.
