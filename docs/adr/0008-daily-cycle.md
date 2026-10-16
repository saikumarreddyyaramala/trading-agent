# ADR 0008: Workflow engine for the daily cycle

- Status: Accepted
- Date: 2026-10-16

## Context

Phases 1-6 built every piece: health checks, the analyst, the risk engine, execution, and
the audit trail. They now have to run unattended, once per trading day, on schedule. That
has to survive restarts, weekends, holidays, half-days, and failures in any one piece.

## Options considered

| Option | Verdict |
| --- | --- |
| Temporal / Airflow / Prefect | Durable and observable, but a cluster of infrastructure for one cycle a day |
| OS cron or Kubernetes CronJobs per step | Simple, but state between steps would live outside the app |
| **In-process state machine + audit store** | One container; durability from the database we already have |

## Decisions

### 1. One long-lived process, one run per trading day
`trading-agent run` loops over days. Each day is one run, `daily-YYYYMMDD`, with steps at the
configured times (market time):

`preflight 08:30 -> analysis 08:45 -> entry 09:45 -> manage each minute -> close 15:50 -> reconcile 16:15`

### 2. Durable progress, and resume
Each step's outcome is a row in `run_steps` (migration 0002), recorded once. On restart the
cycle skips finished steps and recovers the morning's verified proposal from the store, so
a restart never pays for a second analysis. Without a database a restart still cannot
double-buy: the entry's client order id derives from the run id, and the risk engine
counts today's orders.

### 3. Failures stop new trades, never the exits
Every step is guarded. A failed pre-flight, an analysis error, a risk rejection, or an
unavailable required audit store all mean no entry. Position management, the force-close
(verified flat, with `flatten` as the last resort), and order reconciliation always run.

### 4. The broker's calendar decides
The market clock determines whether today trades and when it closes. Weekends and holidays
are skipped. On an early close, the force-close moves to
`close - orchestration.early_close_buffer_minutes` and the trade window ends 20 minutes
before that.

### 5. Cheap when idle
Positions are managed every `manage_interval_seconds` only while one may exist. Once the
entry step is over and the account is flat, the loop stops until the close.

### 6. Stopping
Ctrl+C stops the process cleanly. Open positions keep their broker-side stop-loss.
Restarting resumes the day, and `trading-agent flatten --confirm` exits immediately.

## Consequences

- The deployable unit stays one container plus PostgreSQL (`docker compose up`).
- Config is read at start-up (ADR 0001), so changing the kill switch needs a restart.
  `flatten` is the immediate emergency action.
- A process started after the day's close does nothing until the next trading day. A
  crash after the close skips that day's reconcile, and `trading-agent sync-orders` covers it.
