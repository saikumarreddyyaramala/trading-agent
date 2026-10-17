"""What a proposal would have returned had it been traded, from that day's daily bar.

Used only for proposals that were NOT traded (vetoed by the critic, rejected by the risk
engine) to judge whether those gatekeepers saved or cost money. It is deliberately
conservative, because a daily bar cannot say in which order prices were hit:
- the entry is the analyst's reference price;
- if the day's low reached the stop, the trade is assumed stopped out, even if the target
  was also reached (stop first);
- otherwise, if the high reached the target, it is assumed to have taken profit;
- otherwise it is closed at the day's close.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from trading_agent.domain.models import Bar

Outcome = Literal["stop", "target", "close"]


class SimulatedTrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    return_pct: Decimal
    outcome: Outcome


def simulate_intraday(
    bar: Bar, entry: Decimal, *, stop_loss_pct: Decimal, take_profit_pct: Decimal
) -> SimulatedTrade:
    if entry <= 0:
        raise ValueError("entry price must be positive")
    stop = entry * (1 - stop_loss_pct / 100)
    target = entry * (1 + take_profit_pct / 100)
    if bar.low <= stop:
        return SimulatedTrade(return_pct=(-stop_loss_pct).quantize(Decimal("0.01")), outcome="stop")
    if bar.high >= target:
        return SimulatedTrade(
            return_pct=take_profit_pct.quantize(Decimal("0.01")), outcome="target"
        )
    move = (bar.close - entry) / entry * 100
    return SimulatedTrade(return_pct=move.quantize(Decimal("0.01")), outcome="close")
