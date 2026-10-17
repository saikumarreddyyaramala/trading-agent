# ADR 0011: Evaluation method and the go/no-go bar for live trading

- Status: Accepted
- Date: 2026-10-16

## Context

Phase 11 puts real money ($100) behind the system. Before that, the paper record has to
show, with evidence, that the system is reliable and does better than holding the index.
A judgement made after looking at results tends to find reasons to say yes. So the bar is
fixed now, before any history exists, and checked by code.

## Decisions

### 1. The scorecard
`trading-agent evaluate [--from] [--to]` scores the daily cycle's record (default: from
the first daily cycle to today) and saves `reports/evaluation-<date>.md` and `.json`:
- **Funnel**: days → analyses → buy proposals → critic → risk engine → entries → closed
  trades, with the most frequent risk rejections.
- **Performance**, system trades only: win rate, total P&L, mean return, mean SPY
  open-to-close move on the same days, mean **excess** return, its **90% bootstrap
  interval** (seeded, so reproducible), max drawdown, profit factor, best and worst trade.
- **Gatekeepers**: what the critic's vetoes and the risk engine's rejections would have
  returned, simulated conservatively from the day's bar (entry at the analyst's price; if
  the low reached the stop, it counts as stopped out even if the target was also reached).
  A negative mean means the gate saved money; a positive one means it blocked winners.
- **Cost**: tokens for analyses and critiques, and dollars when prices are configured.

### 2. The bar (config `evaluation:`)
| Criterion | Required |
| --- | --- |
| Cycle days | ≥ 40 (about two months) |
| Closed trades | ≥ 20 |
| Win rate | ≥ 45% |
| Mean excess return vs SPY | > 0% |
| Low end of its 90% interval | > -0.5% |
| Total P&L | > $0 |
| Max drawdown | ≤ $15 (half the $30 cushion above the $70 floor) |
| Cycle days with errors | ≤ 5% |
| LLM cost per cycle day | ≤ $1.00 (needs configured prices) |

**GO** requires every criterion to pass. A criterion that cannot be evaluated (no trades
yet, no prices configured) is a **fail**: real money needs complete information. The
command exits 0 on GO and 1 on NO-GO, so Phase 11 can require a GO in code.

### 3. Changing the bar
Thresholds live in `config/base.yaml`, which is versioned. Loosening one is a reviewed
commit with a reason, made before looking at the scorecard it would change.

### 4. Replay: the critic, not the analyst
`trading-agent evaluate-critic` re-runs the current critic over recorded proposals and
reports agreement and flips. This is honest because the exact tool outputs it reviews are
stored (migration 0003). The analyst is not replayed: its tools return live data, so a
"replay" would really be a new run on different information. Proposals recorded before
migration 0003 are skipped and listed, never judged on empty evidence.

## Consequences

- No result is meaningful until there are weeks of paper history. Until then `evaluate`
  says NO-GO, correctly.
- Twenty trades is a small sample. The interval criterion exists so that a lucky streak
  does not pass on its average alone. Passing the bar means "allowed to try $100", not
  "proven profitable".
- The counterfactual simulation is approximate, using daily bars and the analyst's price.
  It is informative only and is not a GO criterion.
