"""Outcome scoring (ADR 0017): which lessons and playbooks went with better trades."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from trading_agent.learning.outcomes import (
    TradeOutcome,
    lesson_outcome,
    playbook_outcomes,
    stats,
)

D = Decimal


def trade(ret: str, lessons: tuple[int, ...] = (), playbook: str | None = None) -> TradeOutcome:
    return TradeOutcome(
        run_id="daily-20261015",
        day=date(2026, 10, 15),
        symbol="AAPL",
        return_pct=D(ret),
        lesson_ids=lessons,
        playbook=playbook,
    )


def test_stats() -> None:
    s = stats([D("2"), D("-1"), D("1.5")])
    assert s is not None
    assert (s.n, s.mean_pct, s.wins, s.win_rate_pct) == (3, D("0.83"), 2, D("66.67"))
    assert stats([]) is None


def test_a_lesson_is_judged_with_against_without() -> None:
    trades = [trade("2", (7,)), trade("1", (7, 8)), trade("-1", (8,)), trade("-3")]
    outcome = lesson_outcome(7, trades)
    assert outcome.with_ is not None
    assert outcome.without is not None
    assert (outcome.with_.n, outcome.with_.mean_pct) == (2, D("1.50"))
    assert (outcome.without.n, outcome.without.mean_pct) == (2, D("-2.00"))
    assert outcome.lift_pct == D("3.50")


def test_no_lift_without_both_sides() -> None:
    assert lesson_outcome(7, [trade("2", (7,))]).lift_pct is None
    assert lesson_outcome(7, [trade("2")]).with_ is None


def test_playbooks_are_attributed_by_name() -> None:
    result = playbook_outcomes(
        [
            trade("2", playbook="gap"),
            trade("-1", playbook="gap"),
            trade("5"),
            trade("1", playbook="x"),
        ]
    )
    assert set(result) == {"gap", "x"}
    assert (result["gap"].n, result["gap"].mean_pct) == (2, D("0.50"))


returns = st.lists(st.decimals(min_value=-20, max_value=20, places=2), max_size=40)


@given(returns=returns, flags=st.lists(st.booleans(), max_size=40))
def test_with_and_without_partition_every_trade(returns: list[Decimal], flags: list[bool]) -> None:
    trades = [trade(str(r), (1,) if f else ()) for r, f in zip(returns, flags, strict=False)]
    outcome = lesson_outcome(1, trades)
    n_with = outcome.with_.n if outcome.with_ else 0
    n_without = outcome.without.n if outcome.without else 0
    assert n_with + n_without == len(trades)
    if outcome.lift_pct is not None:
        assert outcome.with_ is not None
        assert outcome.without is not None
        assert outcome.lift_pct == outcome.with_.mean_pct - outcome.without.mean_pct
