"""Small, dependency-free statistics for the evaluation scorecard. No I/O."""

from __future__ import annotations

import random
from collections.abc import Sequence
from decimal import Decimal

_CENT = Decimal("0.01")


def _q(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(_CENT)


def mean(values: Sequence[Decimal]) -> Decimal | None:
    return _q(sum(values, Decimal("0")) / len(values)) if values else None


def bootstrap_mean_interval(
    values: Sequence[Decimal], *, confidence: float = 0.90, samples: int = 5000, seed: int = 7
) -> tuple[Decimal, Decimal] | None:
    """Percentile bootstrap interval for the mean. Seeded, so the same data gives the same
    answer every time. None with fewer than two values: there is nothing to resample."""
    if len(values) < 2:
        return None
    data = [float(v) for v in values]
    # Statistics, not security: a seeded generator makes the interval reproducible.
    rng = random.Random(seed)  # noqa: S311
    n = len(data)
    means = sorted(sum(rng.choices(data, k=n)) / n for _ in range(samples))
    tail = (1 - confidence) / 2
    low = means[int(tail * (samples - 1))]
    high = means[int((1 - tail) * (samples - 1))]
    return _q(low), _q(high)


def max_drawdown(pnls: Sequence[Decimal]) -> Decimal:
    """Largest peak-to-trough fall of cumulative P&L, as a positive amount (0 if none)."""
    peak = total = worst = Decimal("0")
    for pnl in pnls:
        total += pnl
        peak = max(peak, total)
        worst = max(worst, peak - total)
    return worst.quantize(_CENT)


def profit_factor(pnls: Sequence[Decimal]) -> Decimal | None:
    """Gross profit divided by gross loss. None when there are no losses to divide by."""
    gains = sum((p for p in pnls if p > 0), Decimal("0"))
    losses = -sum((p for p in pnls if p < 0), Decimal("0"))
    return None if losses == 0 else _q(gains / losses)
