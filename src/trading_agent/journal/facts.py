"""The day in numbers (ADR 0023): fixed rules, no AI, so the notebook always has facts."""

from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_agent.journal.models import SESSIONS, Move, Session, SessionRecord

TOP_MOVERS = 8


def _dec(value: Any) -> Decimal | None:
    try:
        return None if value is None else Decimal(str(value))
    except InvalidOperation:
        return None


def _last_moves(record: SessionRecord | None) -> tuple[list[Move], list[Move]]:
    if record is None or not record.snapshots:
        return [], []
    last = record.snapshots[-1]
    return list(last.market), list(last.stocks)


def day_facts(day: date, records: dict[Session, SessionRecord]) -> dict[str, Any]:
    market: dict[str, dict[str, str | None]] = {}
    for session in SESSIONS:
        moves, _ = _last_moves(records.get(session))
        for move in moves:
            entry = market.setdefault(move.symbol, {})
            entry[session] = None if move.move_pct is None else f"{move.move_pct:+.2f}"

    movers: dict[str, Move] = {}
    order: tuple[Session, ...] = ("regular", "pre", "post")  # the regular close counts first
    for session in order:
        _, stocks = _last_moves(records.get(session))
        for move in stocks:
            if move.move_pct is not None and move.symbol not in movers:
                movers[move.symbol] = move
    top = sorted(movers.values(), key=lambda m: -abs(m.move_pct or Decimal(0)))[:TOP_MOVERS]

    regular = records.get("regular")
    trades = regular.trades if regular else []
    closed = [t for t in trades if t.get("closed")]
    pnl = sum((_dec(t.get("pnl_usd")) or Decimal(0) for t in closed), Decimal(0))
    sources = [s for r in records.values() for s in r.sources]
    return {
        "day": day.isoformat(),
        "market": market,
        "top_movers": [
            {"symbol": m.symbol, "move_pct": f"{m.move_pct:+.2f}", "catalysts": list(m.catalysts)}
            for m in top
        ],
        "trades": {
            "count": len(trades),
            "closed": len(closed),
            "won": sum(1 for t in closed if (_dec(t.get("pnl_usd")) or 0) > 0),
            "pnl_usd": f"{pnl:+.2f}",
        },
        "sources": {"total": len(sources), "by_kind": dict(Counter(s.kind for s in sources))},
        "alerts": sum(len(s.alerts) for r in records.values() for s in r.snapshots),
        "snapshots": {s: len(r.snapshots) for s, r in records.items()},
    }


def facts_headline(day: date, facts: dict[str, Any], benchmark: str = "SPY") -> str:
    spy = facts["market"].get(benchmark, {})
    parts = [f"{day:%a} {day.isoformat()}:"]
    if spy.get("regular") is not None:
        parts.append(f"{benchmark} {spy['regular']}% at the close")
    elif spy.get("pre") is not None:
        parts.append(f"{benchmark} {spy['pre']}% pre-market")
    else:
        parts.append(f"{benchmark} move unknown")
    count = facts["trades"]["count"]
    tail = "no trades" if count == 0 else f"{count} trade{'s' if count != 1 else ''}"
    return f"{' '.join(parts)}; {tail}"
