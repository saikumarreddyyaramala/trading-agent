from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeMarketData, FakeProvider, provider_error
from tests.unit.test_report import (
    DAY,
    GENERATED,
    PROPOSAL,
    RUN,
    _ny,
    _record_trading_day,
)
from trading_agent.agents.analyst import AnalystResult
from trading_agent.agents.critic import CriticResult
from trading_agent.agents.reflection import Reflector
from trading_agent.domain.models import Bar, LLMUsage
from trading_agent.learning import LearningError, LearningService
from trading_agent.llm import LLMGateway
from trading_agent.settings.models import AppConfig, LLMConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import NewLessonRow, SqlAuditStore


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: GENERATED)


def _bar(symbol: str, open_: str, high: str, low: str, close: str) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2026, 10, 15, 4, tzinfo=UTC),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=1,
    )


def _market_data(**extra: Bar) -> FakeMarketData:
    bars = {"SPY": (_bar("SPY", "600", "610", "590", "603"),)}
    bars.update({symbol: (bar,) for symbol, bar in extra.items()})
    return FakeMarketData(bars=bars)


def _answer(lessons: list[dict[str, str]] | None = None, retire: list[int] | None = None) -> str:
    return json.dumps(
        {
            "summary": "A clean exit at the force-close.",
            "new_lessons": [
                {"text": "Earnings beats held up into the close.", "category": "research"}
            ]
            if lessons is None
            else lessons,
            "retire": retire or [],
        }
    )


def _service(
    store: Any, outcomes: list[Any], *, market_data: Any = None, **learning: Any
) -> tuple[LearningService, FakeProvider]:
    provider = FakeProvider(outcomes)
    config = AppConfig.model_validate({"learning": learning})
    reflector = Reflector(
        LLMGateway(provider, LLMConfig(model="judge", fallback_model="judge-backup")),
        max_new=config.learning.max_new_per_day,
        clock=lambda: GENERATED,
    )
    service = LearningService(
        store, market_data or _market_data(), reflector, config, clock=lambda: GENERATED
    )
    return service, provider


def _record_vetoed_day(store: SqlAuditStore) -> None:
    store.start_run(RUN, "daily")
    store.record_analysis(
        AnalystResult(
            run_id=RUN,
            status="proposed",
            reason="evidence verified",
            proposal=PROPOSAL,
            prompt_version="p",
            turns=2,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=_ny(8, 45),
            duration_seconds=1,
        )
    )
    store.record_critique(
        CriticResult(
            run_id=RUN,
            status="vetoed",
            model_verdict="veto",
            summary="Old news.",
            reasons=("the catalyst is a week old",),
            concerns=(),
            evidence_checks=(),
            model="judge",
            prompt_version="c",
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            created_at=GENERATED,
        )
    )
    store.finish_run(RUN, "no_trade", "the critic did not approve the proposal")


# ---------- The day record ----------


def test_day_record_of_a_traded_day(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    service, _ = _service(store, [])
    run = store.get_run(RUN)
    assert run is not None
    record = service.day_record(DAY, run)
    assert record["cycle_status"] == "traded"
    assert record["steps"] == {"preflight": "ok", "entry": "entered", "close": "ok"}
    assert record["analysis"]["symbol"] == "AAPL"
    assert record["analysis"]["evidence_claims"] == ["Beat estimates", "Spread 0.02%"]
    assert record["risk"] == {"approved": True, "failed_checks": []}
    assert record["critic"] is None
    assert record["trade"]["return_pct"] == "2.00"
    assert record["trade"]["exit_via"] == "exit order"
    assert record["benchmark_spy_open_to_close_pct"] == "0.50"
    assert record["stopped_idea_simulation"] is None
    assert record["position_events"] == ["force_close: force close at 15:50"]
    # Raw tool output and news text never reach the reviewer.
    assert "tool_calls" not in json.dumps(record)


def test_day_record_simulates_a_vetoed_idea(store: SqlAuditStore) -> None:
    _record_vetoed_day(store)
    service, _ = _service(
        store, [], market_data=_market_data(AAPL=_bar("AAPL", "100", "101", "94", "96"))
    )
    run = store.get_run(RUN)
    assert run is not None
    record = service.day_record(DAY, run)
    assert record["critic"]["status"] == "vetoed"
    assert record["critic"]["reasons"] == ["the catalyst is a week old"]
    assert record["trade"] is None
    assert record["stopped_idea_simulation"] == {
        "gate": "critic",
        "simulated_return_pct": "-5.00",
        "outcome": "stop",
    }


def test_day_record_simulates_a_risk_rejection(store: SqlAuditStore) -> None:
    from trading_agent.domain.models import RiskCheck, RiskDecision

    store.start_run(RUN, "daily")
    store.record_analysis(
        AnalystResult(
            run_id=RUN,
            status="proposed",
            reason="ok",
            proposal=PROPOSAL,
            prompt_version="p",
            turns=1,
            tool_calls=(),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            started_at=_ny(8, 45),
            duration_seconds=1,
        )
    )
    store.record_risk_decision(
        RUN,
        RiskDecision(
            approved=False,
            symbol="AAPL",
            checks=(RiskCheck(name="Spread", passed=False, detail="too wide"),),
            plan=None,
            evaluated_at=_ny(9, 45),
        ),
    )
    service, _ = _service(
        store, [], market_data=_market_data(AAPL=_bar("AAPL", "100", "111", "99", "105"))
    )
    run = store.get_run(RUN)
    assert run is not None
    record = service.day_record(DAY, run)
    assert record["risk"] == {"approved": False, "failed_checks": ["Spread"]}
    assert record["stopped_idea_simulation"]["gate"] == "risk engine"
    assert record["stopped_idea_simulation"]["outcome"] == "target"


def test_day_record_survives_missing_market_data(store: SqlAuditStore) -> None:
    _record_vetoed_day(store)
    service, _ = _service(store, [], market_data=_Broken())
    run = store.get_run(RUN)
    assert run is not None
    record = service.day_record(DAY, run)
    assert record["benchmark_spy_open_to_close_pct"] is None
    assert record["stopped_idea_simulation"] is None


class _Broken:
    def get_daily_bars(self, symbols: Any, *, days: int) -> Any:
        raise RuntimeError("data down")


# ---------- Learning ----------


def test_learn_stores_new_lessons_with_their_evidence(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    service, provider = _service(store, [_answer()])
    detail = service.learn(DAY)
    assert detail == "1 new, 0 retired: A clean exit at the force-close."
    [lesson] = store.active_lessons(GENERATED, 10)
    assert lesson.text == "Earnings beats held up into the close."
    assert lesson.source_run_id == RUN
    assert lesson.expires_at == GENERATED + timedelta(days=60)
    [row] = store.list_lessons(include_retired=False)
    assert row["evidence"] == {
        "run_id": RUN,
        "symbol": "AAPL",
        "decision": "buy",
        "critic": None,
        "trade_return_pct": "2.00",
        "spy_pct": "0.50",
    }
    assert '"symbol": "AAPL"' in provider.requests[0].messages[0].content


def test_learn_retires_contradicted_lessons_and_reports_rejections(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    [old] = store.record_lessons(
        None,
        [NewLessonRow(text="Earnings beats fade by noon.", category="research", evidence={})],
        expires_at=GENERATED + timedelta(days=30),
        max_active=15,
    )
    answer = _answer(
        lessons=[{"text": "Use bigger positions on earnings beats.", "category": "entry"}],
        retire=[old, 12345],
    )
    service, provider = _service(store, [answer])
    detail = service.learn(DAY)
    assert detail.startswith("0 new, 1 retired, 1 rejected by guardrails")
    assert store.active_lessons(GENERATED, 10) == []
    [row] = store.list_lessons(include_retired=True)
    assert row["retired_reason"] == f"contradicted by the review of {RUN}"
    assert f'"id": {old}' in provider.requests[0].messages[0].content


def test_learn_raises_when_the_review_fails(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    service, _ = _service(store, [provider_error("judge"), provider_error("judge-backup")])
    with pytest.raises(LearningError, match="reflector unavailable"):
        service.learn(DAY)
    assert store.list_lessons(include_retired=True) == []


def test_nothing_to_learn_without_an_analysis(store: SqlAuditStore) -> None:
    service, provider = _service(store, [])
    assert service.learn(DAY) == "nothing to learn: no analysis that day"
    store.start_run(RUN, "daily")
    assert service.learn(DAY) == "nothing to learn: no analysis that day"
    assert provider.requests == []
