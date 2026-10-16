# ADR 0010: Critic agent

- Status: Accepted
- Date: 2026-10-16

## Context

Phase 3's evidence checks are mechanical: a cited call must exist, be about the symbol,
and the reference price must match the data. They cannot tell whether a claim is actually
*true* of the cited output, or whether the thesis makes sense. The first real analyst run
showed the gap: a quote price was cited against the daily-bars call. A human trader would
want a second opinion before risking money, so the system gets one.

## Decisions

### 1. A reviewer, not a researcher
The critic gets no tools. It sees the proposal and the exact tool outputs the analyst was
shown, cited calls first, within `critic.max_evidence_chars`. It judges the record rather
than redoing the research. That keeps it cheap, repeatable (`trading-agent critique <run>`
re-reviews any recorded proposal), and auditable.

### 2. Veto only
The critic answers `approve` or `veto`. No code path lets it create, change, or enlarge a
trade. An approved proposal still goes through the full risk engine.

### 3. Code has the last word
- Every cited evidence item must appear in the critic's `evidence_checks`, and every one
  must be `supported`. Otherwise the result is `vetoed`, whatever the critic's verdict.
- The critic's own `veto` always stands.
- Output that is not the required JSON, after `critic.repair_attempts`, is `failed`.
- An unavailable model is `failed`.
Only `approved` lets a trade continue. `vetoed`, `failed`, and an exception are all no trade.

### 4. Biased toward no
The prompt states that vetoing a good trade costs little and approving a bad one loses
money: "When in doubt, veto." It also tells the critic to treat text aimed at an AI inside
tool outputs (prompt injection, typically via news) as a red flag and veto.

### 5. Independence
`critic.model` can name a different model, which gives a more independent view at no extra
plumbing. It has its own gateway, so its token usage is accounted separately.

### 6. Evidence the critic needs is part of the audit trail
Migration 0003 stores each tool call's output (`tool_calls.result`) and adds a `critiques`
table. The critic gets the outputs in memory during the cycle, and from the store when a
restarted cycle resumes. So the critic works without a database, and a restart never
re-reviews.

### 7. Place in the cycle
`preflight -> analysis -> critique -> entry -> ...` The critique step is recorded like any
other. A disabled critic (`critic.enabled: false`) is recorded as `disabled`.

## Consequences

- One extra LLM call per proposed buy (none on no-trade days), shown in the report.
- The critic can veto good trades. That is accepted: Phase 10 (evaluation) measures how
  often and at what cost.
- Two LLMs from the same vendor can share blind spots. A different model, or a different
  vendor (a later config option), reduces that.
