# ADR 0013: Self-learning from the system's own results

- Status: Accepted
- Date: 2026-10-17

## Context

Until now each trading day started from zero. The analyst never saw how its past ideas
turned out, which vetoes were right, or which patterns came before stop-outs. The audit
trail already holds every proposal, critic review, risk decision, fill, and exit. The
system should learn from that record.

Learning that changes behaviour is also a risk. A model could "learn" to ignore the
critic or to size up. A malicious news headline could plant a lasting instruction
(memory poisoning). And one lucky day could harden into a rule.

## Decisions

### 1. Learning is memory, not self-modification
After each day a reviewer (`agents/reflection.py`) reads the day's record and writes at
most `learning.max_new_per_day` (2) short lessons. The analyst reads the active lessons
the next morning, appended after its rules as **advisory** experience. Lessons can never
change code, config, risk limits, position size, stops, the schedule, or the critic. Those
stay fixed, and the risk engine and critic still enforce them on every trade.

We do not fine-tune models or let the system edit its own prompts or config. Those changes
cannot be reviewed, are hard to undo, and are easy to overfit on a few days of data.

### 2. Guardrails in code, not in the prompt
Every proposed lesson goes through `guard()` before it is stored:
- **Safeguard filter**: text that would weaken a safeguard is rejected. Examples: "ignore
  the critic", "skip the checks", "bigger positions", "use margin".
- **Sanitized**: URLs and control characters are removed, and the length is capped at
  300 characters.
- **No repeats**: duplicates of active lessons are rejected.
- **Daily cap**: at most `max_new_per_day` lessons per day.
- **Retire only active ids**: the reviewer can retire only lessons that are active now.

Rejections are logged and counted in the step detail.

### 3. The reviewer never sees raw news or tool output
Its input is a structured record built by `learning/service.py`. The record holds the
proposal, the critic's verdict and reasons, the failed risk checks, the trade's result,
SPY's open-to-close move, and what a stopped idea would have done (the same simulation as
the scorecard). Untrusted third-party text cannot reach the lessons directly. Model-written
text in the record is marked as data, not instructions.

### 4. Memory is small, dated, and decays
- At most `learning.max_active` (15) lessons are active. When new lessons arrive, the
  oldest are retired.
- Each lesson expires after `lesson_ttl_days` (60) unless it is learned again, because
  markets change.
- The reviewer retires lessons that the day's evidence contradicts.
- The operator can list (`trading-agent lessons`) and retire (`trading-agent
  retire-lesson`) lessons at any time.

Retired lessons stay in the table with a reason.

### 5. Every decision records what it knew
Each analysis stores the ids of the lessons in its prompt (`analyses.lesson_ids`), and each
lesson stores the facts of the day that produced it (`lessons.evidence`). The scorecard and
the audit trail can show whether the lessons helped. The daily report has a "Learning"
section with the lessons the analyst used, learned, and retired that day.

### 6. Learning can never cost a trade or an exit
The `reflect` step runs after the day's result is recorded and before the report. A failed
review is recorded as an error, and the day, its trades, and the report are unaffected.
The step is skipped on resume once finished. Lessons need a database. Without one, or when
`learning.enabled: false`, the step does nothing and the analyst runs without lessons.

## Consequences

- One extra LLM call per trading day. Its tokens are logged but not yet included in the
  scorecard's LLM cost criterion.
- Lessons change the analyst's inputs, so results before and after learning are not
  strictly comparable. `lesson_ids` makes the split visible, and `learning.enabled: false`
  turns learning off.
- The safeguard filter is a pattern list. It can miss a reworded attempt, which is why the
  rules themselves are enforced in code and not in the prompt.
