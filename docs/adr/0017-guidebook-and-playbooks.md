# ADR 0017: The guidebook, outcome-scored lessons, and playbooks

- Status: Accepted
- Date: 2026-10-17

## Context

ADR 0013 gave the system a short-term memory: lessons written after each day, which expire.
The operator wants more: a system that knows what it has learned, keeps a reference of that
knowledge (its own constitution or guide book), writes reusable skills when they matter,
and improves recursively.

There are two risks. A system that edits its own rules or code can learn its way out of its
safeguards. And a system that trusts its own lessons without checking them against results
only accumulates opinions.

## Decisions

### 1. A guidebook in four parts, rebuilt daily, versioned
`learning/guidebook.py` renders the guidebook. A new version is stored only when the text
changes (`guidebook_versions`, migration 0008), with a list of what changed.

1. **Core rules**, generated from the config: long only and intraday, the critic, the trade
   score and the risk checks, size and stops, the trade limits, paper until the operator
   decides, and that advice never changes rules. The system cannot edit this part. It is a
   view of the code, not a policy the model holds.
2. **Principles**: lessons that went with measurably better trades.
3. **Playbooks**: research procedures, each with its results.
4. **Self-assessment**: the closed-trade record, the best and weakest playbook, and how
   much of what it knows is proven or still unproven.

The text has no timestamps, so the version history means something. It is saved to
`reports/guidebook.md` and shown by `trading-agent guidebook`.

### 2. Lessons are judged by results, conservatively
`learning/outcomes.py` joins every closed trade with what its analysis knew: the lessons
in the prompt (`analyses.lesson_ids`) and the playbook it followed.

Every lesson active at a time is in every prompt, so lessons cannot be credited
individually. A lesson is compared on the trades made with it in context against the
trades made without it. That comparison also picks up whatever else changed over time.

So the curator acts only with `guidebook.min_trades` (10) on both sides:
- it **promotes** a lesson to a principle when trades with it did at least 0.5 points better
  and were profitable on average;
- it **retires** a lesson, or a principle, when trades with it did 0.5 points worse.

Principles never expire and are never pushed out by newer lessons. They can still be
retired when their results turn.

### 3. Playbooks: written by a guarded model, judged by their own trades
When one category holds `min_lessons_for_playbook` (3) active lessons and no playbook, the
playbook writer drafts one: a name, a title, when to use it, and 2 to 6 steps. It uses the
same guardrails as lessons (ADR 0013):
- text is cleaned and capped;
- anything that would weaken a safeguard is rejected;
- names must be new;
- it gets one repair, and an outage only means no playbook.

A playbook is text for the analyst, never code. It starts as a **candidate**:
- the analyst sees candidates only on paper; live sees only active playbooks;
- the analyst names the playbook it followed (`TradeProposal.playbook`), and unknown names
  are dropped;
- after `playbook_min_trades` (5) trades with a non-negative mean it becomes **active**;
- below `playbook_retire_below_pct` (−1%) it is **retired**.

At most `max_playbooks` (8) are live. The operator can retire one with
`trading-agent retire-playbook`.

### 4. The analyst reads what was learned, in order of trust
The system prompt keeps the rules first, then:
- principles ("proven by your own results");
- recent, unproven lessons;
- the playbooks.

Every section says it is advice and never overrides the rules. The risk engine, the trade
score, and the critic enforce the rules anyway.

### 5. The loop
Every trading day runs this sequence:
1. The evening review (ADR 0013) writes lessons.
2. The `curate` step judges lessons and playbooks against the trades, writes a playbook
   when one is due, and rebuilds the guidebook.
3. The plan (ADR 0016) carries the day forward.
4. The next morning the analyst works from the updated guidebook.

Curation never blocks trading. A failure is recorded as an error, and the plan and the
report still run.

## Consequences

- Nothing is promoted or retired for results until there are 10+ closed trades on each side.
  At up to 3 paper trades a day, that is weeks of data. Until then, the guidebook mostly
  shows the core rules and an honest "not enough trades".
- With-vs-without comparisons are weak evidence. They decide which advice the analyst sees;
  they never change a rule.
- The playbook writer costs one LLM call when a category fills up. It shows in the cost
  ledger as `playbook-writer`.
- The system writes procedures, not software. Changing what it can do (new tools, new
  rules) stays a human, reviewed change in this repository.
