"""The dashboard's data layer and API (ADR 0018): read-only, local, partial on failure."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest
from starlette.testclient import TestClient

from tests.conftest import FakeBroker, FakeMarketData, FakeNews, make_position
from tests.unit.test_market_watch import FOCUS, _pre_market, _watch
from tests.unit.test_news_cards import article
from tests.unit.test_report import DAY, GENERATED, RUN, _ny, _record_trading_day
from trading_agent.dashboard.app import create_app
from trading_agent.dashboard.data import DashboardData, stream
from trading_agent.domain.models import (
    LLMCallRecord,
    LLMUsage,
    PortfolioHistory,
    PortfolioPoint,
    Snapshot,
)
from trading_agent.settings.models import AppConfig
from trading_agent.storage.database import make_engine, upgrade
from trading_agent.storage.store import NullAuditStore, SqlAuditStore

NOW = _ny(11, 0)  # Thursday 2026-10-15, 11:00 New York


@pytest.fixture
def store(tmp_path: Path) -> SqlAuditStore:
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    upgrade(engine)
    return SqlAuditStore(engine, AppConfig(), clock=lambda: GENERATED)


class _HistoryBroker(FakeBroker):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.spans: list[str] = []

    def get_portfolio_history(self, span: str) -> PortfolioHistory:
        self.spans.append(span)
        start = _ny(9, 30)
        return PortfolioHistory(
            span=span,
            base_value=Decimal("100000"),
            points=(
                PortfolioPoint(at=start, equity=Decimal("100000")),
                PortfolioPoint(at=start + timedelta(minutes=5), equity=Decimal("100004")),
                PortfolioPoint(at=start + timedelta(minutes=10), equity=Decimal("100003")),
            ),
        )


def _data(store: Any, **kw: Any) -> DashboardData:
    values: dict[str, Any] = {
        "broker": _HistoryBroker(
            equity="100003", positions=(make_position("AAPL", quantity="0.5", avg="100"),)
        ),
        "market_data": FakeMarketData(
            snapshots={
                "AAPL": Snapshot(
                    symbol="AAPL",
                    last_price=Decimal("103"),
                    last_trade_at=NOW,
                    session_close=Decimal("103"),
                    session_at=_ny(9, 30),
                    previous_close=Decimal("100"),
                )
            }
        ),
        "news": FakeNews([article("1", minutes_ago=10), article("2", "Quiet day", "", ("SPY",))]),
    }
    values.update(kw)
    return DashboardData(store, values.pop("config", AppConfig()), clock=lambda: NOW, **values)


# ---------- Overview ----------


def test_overview_has_account_positions_market_and_cycle(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    overview = _data(store).overview()
    assert overview["environment"] == "paper"
    account = overview["account"]
    assert account["equity"] == "100003"
    (position,) = overview["positions"]
    assert position["symbol"] == "AAPL"
    assert position["unrealized_pnl"] == "0.00"  # FakeBroker values it at cost
    assert position["stop_price"] == "95.00"
    assert position["target_price"] == "110.00"
    assert overview["market"]["is_open"] is True
    cycle = overview["cycle"]
    assert cycle["run_id"] == RUN
    assert [s["step"] for s in cycle["steps"]] == ["preflight", "entry", "close"]
    assert overview["errors"] == []


def test_overview_without_a_broker_is_partial_not_broken(store: SqlAuditStore) -> None:
    overview = _data(store, broker=None).overview()
    assert overview["account"] is None
    assert overview["positions"] == []
    assert "broker unavailable: no broker credentials" in overview["errors"]


def test_overview_survives_broker_errors(store: SqlAuditStore) -> None:
    broker = _HistoryBroker(fail_account=True, fail_positions=True, fail_clock=True)
    overview = _data(store, broker=broker).overview()
    assert overview["account"] is None
    assert any(e.startswith("account unavailable") for e in overview["errors"])
    assert any(e.startswith("positions unavailable") for e in overview["errors"])
    assert any(e.startswith("market clock unavailable") for e in overview["errors"])


def test_next_step_follows_the_schedule(store: SqlAuditStore) -> None:
    next_step = _data(store).overview()["next_step"]
    assert next_step == {"step": "close", "at": _ny(15, 50).isoformat()}


def test_today_shows_spend_against_the_budget(store: SqlAuditStore) -> None:
    store.record_llm_call(
        LLMCallRecord(
            created_at=_ny(9),
            provider="openai",
            model="gpt-6.1-sol",
            agent="analyst",
            run_id=None,
            status="ok",
            usage=LLMUsage(input_tokens=1000, output_tokens=100),
            latency_ms=1.0,
        ),
        Decimal("0.25"),
    )
    today = _data(store).overview()["today"]
    assert (today["llm_usd"], today["budget_usd"], today["calls"]) == ("0.25", "2.00", 1)


# ---------- Charts, trades, news ----------


def test_history_and_its_change(store: SqlAuditStore) -> None:
    broker = _HistoryBroker()
    history = _data(store, broker=broker).history("1D")
    assert [p["equity"] for p in history["points"]] == ["100000", "100004", "100003"]
    assert history["change"] == "3"
    assert history["change_pct"] == "0.00"
    assert broker.spans == ["1D"]
    with pytest.raises(ValueError, match="span"):
        _data(store).history("5Y")


def test_trades_list(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    (trade,) = _data(store).trades(30)
    assert (trade["symbol"], trade["return_pct"], trade["run_id"]) == ("AAPL", "2.00", RUN)


def test_news_cards_mark_what_we_watch(store: SqlAuditStore) -> None:
    cards = _data(store).news(10)
    assert cards[0]["id"] == "1"  # earnings, a held symbol, and a 3% move
    assert cards[0]["watched"] is True
    assert cards[0]["move_pct"] == "3.00"
    assert cards[0]["catalysts"] == ["earnings"]


def test_news_outage_is_an_empty_list_with_an_error(store: SqlAuditStore) -> None:
    data = _data(store, news=FakeNews(fail=True))
    assert data.news(10) == []
    assert any(e.startswith("news unavailable") for e in data.overview()["errors"])


# ---------- Learning, plan, watch, costs, activity ----------


def test_learning_plan_watch_and_costs(store: SqlAuditStore) -> None:
    from trading_agent.planning import Planner

    _record_trading_day(store)
    store.record_watch(RUN, _watch(_pre_market()).scan("pre", DAY, FOCUS))
    Planner(store, AppConfig(), next_session=lambda: date(2026, 10, 16)).plan(DAY)
    store.record_guidebook("# Guidebook", ["first edition"])
    data = _data(store)
    assert data.watch(DAY)["pre"]["alerts"][0] == "UP +8.33% pre-market (gainer)"
    assert data.plan(DAY)["next"]["for_day"] == "2026-10-16"
    learning = data.learning()
    assert learning["guidebook"]["version"] == 1
    assert learning["principles"] == []
    costs = data.costs(7)
    assert len(costs["days"]) == 7
    assert costs["daily_budget_usd"] == "2.00"


def test_activity_feed(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    items = _data(store).activity(10)
    assert items
    assert {"at", "kind", "title"} <= set(items[0])


def test_no_database_means_empty_sections() -> None:
    data = _data(NullAuditStore())
    assert data.trades(30) == []
    assert data.learning()["lessons"] == []
    assert data.activity(10) == []
    assert data.overview()["cycle"] is None


def test_answers_are_cached_briefly(store: SqlAuditStore) -> None:
    broker = _HistoryBroker()
    data = DashboardData(
        store,
        AppConfig(),
        broker=broker,
        market_data=None,
        news=None,
        clock=lambda: NOW,
        cache_seconds=60,
    )
    data.history("1D")
    data.history("1D")
    assert broker.spans == ["1D"]


# ---------- The live stream ----------


def test_the_stream_sends_the_overview_then_new_activity(store: SqlAuditStore) -> None:
    data = _data(store)
    events: list[dict[str, Any]] = []
    ticks = {"n": 0}

    async def fake_sleep(_: float) -> None:
        ticks["n"] += 1
        if ticks["n"] == 1:  # something happens between the first and second tick
            store.start_run("r9", "daily")
            store.record_step("r9", "preflight", "ok", "fine", GENERATED + timedelta(days=1))

    async def collect() -> None:
        start = GENERATED - timedelta(seconds=1)
        async for event in stream(data, interval=1, ticks=2, sleep=fake_sleep, since=start):
            events.append(event)

    anyio.run(collect)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "overview"
    assert "activity" in kinds
    assert kinds.count("overview") == 1  # unchanged overviews are not resent


# ---------- The HTTP API ----------


def _client(store: Any, **kw: Any) -> TestClient:
    return TestClient(create_app(_data(store, **kw), stream_ticks=1))


@pytest.mark.parametrize(
    "path",
    [
        "/api/health",
        "/api/overview",
        "/api/history?span=1W",
        "/api/trades?days=10",
        "/api/performance?days=10",
        "/api/news?limit=5",
        "/api/watch",
        "/api/plan",
        "/api/learning",
        "/api/costs?days=3",
        "/api/activity?limit=5",
    ],
)
def test_every_endpoint_answers_json(store: SqlAuditStore, path: str) -> None:
    response = _client(store).get(path)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["cache-control"] == "no-store"


def test_bad_parameters_are_400s(store: SqlAuditStore) -> None:
    client = _client(store)
    assert client.get("/api/history?span=5Y").status_code == 400
    assert client.get("/api/trades?days=abc").status_code == 400
    assert client.get("/api/costs?days=0").status_code == 400
    assert client.get("/api/watch?date=soon").status_code == 400


def test_runs_can_be_inspected(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    client = _client(store)
    assert client.get(f"/api/runs/{RUN}").json()["run"]["status"] == "traded"
    assert client.get("/api/runs/nope").status_code == 404


def test_the_api_is_read_only(store: SqlAuditStore) -> None:
    client = _client(store)
    for method in ("post", "put", "delete", "patch"):
        assert getattr(client, method)("/api/overview").status_code == 405


def test_the_stream_endpoint_speaks_sse(store: SqlAuditStore) -> None:
    with _client(store).stream("GET", "/api/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "event: overview" in body
    payload = json.loads(next(line for line in body.splitlines() if line.startswith("data: "))[6:])
    assert payload["environment"] == "paper"


def test_the_home_page_points_to_the_api(store: SqlAuditStore) -> None:
    response = _client(store).get("/")
    assert response.status_code == 200
    assert "/api/overview" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert datetime.now(UTC) > NOW


def test_next_step_skips_to_the_next_trading_day(store: SqlAuditStore) -> None:
    from trading_agent.domain.models import MarketClock

    class _Weekend(_HistoryBroker):
        def get_clock(self) -> MarketClock:
            saturday = datetime(2026, 10, 17, 14, tzinfo=UTC)
            monday_open = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)
            return MarketClock(
                timestamp=saturday,
                is_open=False,
                next_open=monday_open,
                next_close=monday_open + timedelta(hours=6, minutes=30),
            )

    saturday = DashboardData(
        store,
        AppConfig(),
        broker=_Weekend(),
        market_data=None,
        news=None,
        clock=lambda: datetime(2026, 10, 17, 14, tzinfo=UTC),
    )
    assert saturday.overview()["next_step"] == {
        "step": "premarket",
        "at": "2026-10-19T08:00:00-04:00",
    }


def test_tiny_losses_never_show_as_negative_zero(store: SqlAuditStore) -> None:
    class _Flat(_HistoryBroker):
        def get_portfolio_history(self, span: str) -> PortfolioHistory:
            return PortfolioHistory(
                span=span,
                base_value=Decimal("100000"),
                points=(PortfolioPoint(at=NOW, equity=Decimal("99999.94")),),
            )

    history = _data(store, broker=_Flat()).history("1M")
    assert history["change"] == "-0.06"
    assert history["change_pct"] == "0.00"


def test_after_the_close_the_post_market_scan_is_next(store: SqlAuditStore) -> None:
    from trading_agent.domain.models import MarketClock

    class _Closed(_HistoryBroker):
        def get_clock(self) -> MarketClock:
            friday_open = datetime(2026, 10, 16, 13, 30, tzinfo=UTC)
            return MarketClock(
                timestamp=_ny(16, 30),
                is_open=False,
                next_open=friday_open,
                next_close=friday_open + timedelta(hours=6, minutes=30),
            )

    _record_trading_day(store)  # the cycle ran today
    data = DashboardData(
        store, AppConfig(), broker=_Closed(), market_data=None, news=None, clock=lambda: _ny(16, 30)
    )
    assert data.overview()["next_step"] == {"step": "postmarket", "at": _ny(16, 45).isoformat()}


# ---------- Failure paths (ADR 0018) ----------


def test_jsonable_handles_every_type() -> None:
    from trading_agent.dashboard.data import jsonable

    assert jsonable({"s": {1}, "t": (1, 2), "d": date(2026, 1, 2)}) == {
        "s": [1],
        "t": [1, 2],
        "d": "2026-01-02",
    }
    with pytest.raises(TypeError, match="not JSON serializable"):
        jsonable({"x": object()})


def test_an_unreadable_database_gives_empty_sections() -> None:
    from trading_agent.storage.store import AuditError

    class _Down(NullAuditStore):
        def __getattribute__(self, name: str) -> Any:
            if name.startswith(
                ("get_", "list_", "active_", "llm_", "plan_", "orders_", "activity")
            ):

                def fail(*args: Any, **kwargs: Any) -> Any:
                    raise AuditError("db down")

                return fail
            return super().__getattribute__(name)

    data = _data(_Down())
    overview = data.overview()
    assert overview["cycle"] is None
    assert overview["today"]["calls"] == 0
    assert data.learning()["guidebook"] is None
    assert data.watch(DAY)["pre"] is None
    assert data.activity(5) == []
    assert data.run("x") is None


def test_after_the_last_step_the_next_is_tomorrows_premarket(store: SqlAuditStore) -> None:
    late = DashboardData(
        store, AppConfig(), broker=None, market_data=None, news=None, clock=lambda: _ny(20)
    )
    assert late.overview()["next_step"] == {"step": "premarket", "at": "2026-10-16T08:00:00-04:00"}


def test_history_without_a_broker() -> None:
    data = _data(NullAuditStore(), broker=None)
    assert data.history("1D") == {"span": "1D", "points": [], "error": "no broker credentials"}


def test_news_survives_missing_parts(store: SqlAuditStore) -> None:
    _record_trading_day(store)
    slot = _data(store)
    assert slot.news(5)  # watched symbols come from positions, the plan, and the slots
    no_feed = _data(store, news=None)
    assert no_feed.news(5) == []
    broken_positions = _data(store, broker=_HistoryBroker(fail_positions=True))
    assert broken_positions.news(5)
    no_snapshots = _data(store, market_data=FakeMarketData(fail_snapshots=True))
    cards = no_snapshots.news(5)
    assert all(c["move_pct"] is None for c in cards)
    assert any(e.startswith("snapshots unavailable") for e in no_snapshots.overview()["errors"])
    no_market = _data(store, market_data=None)
    assert all(c["move_pct"] is None for c in no_market.news(5))


def test_stale_snapshots_give_no_move(store: SqlAuditStore) -> None:
    stale = FakeMarketData(
        snapshots={
            "AAPL": Snapshot(
                symbol="AAPL",
                last_price=None,
                last_trade_at=None,
                session_close=Decimal("100"),
                session_at=_ny(9, 30) - timedelta(days=1),
                previous_close=None,
            )
        }
    )
    cards = _data(store, market_data=stale).news(5)
    assert all(c["move_pct"] is None for c in cards)
