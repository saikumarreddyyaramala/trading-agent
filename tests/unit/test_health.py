from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import SecretStr

from tests.conftest import (
    NOW,
    FakeBroker,
    FakeMarketData,
    FakeNews,
    FakeProvider,
    healthy_market_data,
    make_article,
    make_quote,
    make_secrets,
    provider_error,
)
from trading_agent.health import (
    PROBE_SYMBOL,
    CheckResult,
    CheckStatus,
    format_age,
    overall_ok,
    run_healthchecks,
)
from trading_agent.llm import LLMGateway
from trading_agent.settings import load_config
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import Secrets
from trading_agent.storage.database import head_revision, make_engine, upgrade


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def _gateway(config: AppConfig, outcomes: list[str | Exception]) -> LLMGateway:
    return LLMGateway(FakeProvider(outcomes), config.llm)


def _run(
    config: AppConfig,
    secrets: Secrets | None = None,
    *,
    broker: FakeBroker | None = None,
    market_data: FakeMarketData | None = None,
    news: FakeNews | None = None,
    llm_factory: Any = None,
    engine_factory: Any = None,
) -> list[CheckResult]:
    return run_healthchecks(
        config,
        secrets or make_secrets(),
        broker_factory=lambda: broker or FakeBroker(),
        market_data_factory=lambda: market_data or healthy_market_data(PROBE_SYMBOL),
        news_factory=lambda: news or FakeNews([make_article(symbols=(PROBE_SYMBOL,))]),
        llm_factory=llm_factory,
        clock=lambda: NOW,
        engine_factory=engine_factory,
    )


def _migrated_engine() -> sa.Engine:
    engine = make_engine("sqlite://")
    upgrade(engine)
    return engine


@pytest.fixture
def paper(config_dir: Path) -> AppConfig:
    return load_config("paper", config_dir)


def test_all_green(paper: AppConfig) -> None:
    results = _run(paper, llm_factory=lambda: _gateway(paper, ["OK"]))
    assert overall_ok(results), results
    checks = _by_name(results)
    assert "equity $100,000.00" in checks["Broker account"].detail
    assert checks["LLM"].status is CheckStatus.PASS
    assert checks["Market data"].status is CheckStatus.PASS
    assert "SPY bid $100.00 / ask $100.10, quote 2s old" in checks["Market data"].detail
    assert "quotes=iex, bars=sip" in checks["Market data"].detail
    assert checks["News"].detail == "latest article 1h 0m ago (benzinga)"


def test_missing_secrets_fail_and_skip_dependent_checks(paper: AppConfig) -> None:
    results = _run(paper, make_secrets(alpaca_api_key=None))
    checks = _by_name(results)
    assert checks["Secrets"].status is CheckStatus.FAIL
    assert checks["Secrets"].detail == "missing: ALPACA_API_KEY"  # listed once, not twice
    assert checks["Broker account"].status is CheckStatus.SKIP
    assert checks["Market data"].status is CheckStatus.SKIP
    assert checks["News"].status is CheckStatus.SKIP
    assert not overall_ok(results)


def test_skip_llm_does_not_require_llm_key(paper: AppConfig) -> None:
    results = _run(paper, make_secrets(openai_api_key=None, anthropic_api_key=None))
    assert overall_ok(results)
    assert _by_name(results)["LLM"].status is CheckStatus.SKIP


def test_unreachable_broker_fails(paper: AppConfig) -> None:
    results = _run(paper, broker=FakeBroker(fail_account=True, fail_clock=True))
    checks = _by_name(results)
    assert checks["Broker account"].status is CheckStatus.FAIL
    assert checks["Market clock"].status is CheckStatus.FAIL
    assert not overall_ok(results)


def test_broker_factory_error_fails(paper: AppConfig) -> None:
    def broken() -> FakeBroker:
        raise RuntimeError("bad client")

    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=broken,
        market_data_factory=lambda: healthy_market_data(PROBE_SYMBOL),
        news_factory=FakeNews,
        llm_factory=None,
        clock=lambda: NOW,
    )
    assert _by_name(results)["Broker account"].detail == "bad client"


def test_paper_config_with_live_account_fails(paper: AppConfig) -> None:
    results = _run(paper, broker=FakeBroker(is_paper=False))
    assert _by_name(results)["Broker account"].status is CheckStatus.FAIL


def test_blocked_account_fails(paper: AppConfig) -> None:
    results = _run(paper, broker=FakeBroker(trading_blocked=True))
    assert "blocked" in _by_name(results)["Broker account"].detail


def test_llm_failure_fails(paper: AppConfig) -> None:
    results = _run(paper, llm_factory=lambda: _gateway(paper, [provider_error(), provider_error()]))
    assert _by_name(results)["LLM"].status is CheckStatus.FAIL


def test_live_without_confirmation_fails(config_dir: Path) -> None:
    live = load_config("live", config_dir)
    results = _run(live, broker=FakeBroker(is_paper=False))
    assert _by_name(results)["Trading mode"].status is CheckStatus.FAIL
    assert not overall_ok(results)


def test_live_with_confirmation_warns_but_passes(config_dir: Path) -> None:
    live = load_config("live", config_dir)
    results = _run(
        live,
        make_secrets(live_trading_confirmed=True, database_url=SecretStr("sqlite://")),
        broker=FakeBroker(is_paper=False),
        engine_factory=_migrated_engine,
    )
    assert _by_name(results)["Trading mode"].status is CheckStatus.WARN
    assert overall_ok(results)


def test_kill_switch_is_reported(config_dir: Path) -> None:
    (config_dir / "paper.yaml").write_text(
        "environment: paper\nkill_switch: true\nbroker:\n  paper: true\n", encoding="utf-8"
    )
    results = _run(load_config("paper", config_dir))
    assert _by_name(results)["Kill switch"].status is CheckStatus.WARN


# ---------- Market data and news ----------


def test_stale_quote_fails_while_market_is_open(paper: AppConfig) -> None:
    market_data = healthy_market_data(PROBE_SYMBOL)
    market_data.quotes[PROBE_SYMBOL] = make_quote(PROBE_SYMBOL, age_seconds=900)
    results = _run(paper, market_data=market_data)
    check = _by_name(results)["Market data"]
    assert check.status is CheckStatus.FAIL
    assert "stale quote" in check.detail
    assert not overall_ok(results)


def test_stale_quote_is_expected_while_market_is_closed(paper: AppConfig) -> None:
    market_data = healthy_market_data(PROBE_SYMBOL)
    market_data.quotes[PROBE_SYMBOL] = make_quote(PROBE_SYMBOL, age_seconds=50_000)
    results = _run(paper, broker=FakeBroker(market_open=False), market_data=market_data)
    assert _by_name(results)["Market data"].status is CheckStatus.PASS
    assert "quote 13h 53m old" in _by_name(results)["Market data"].detail


def test_bad_quote_warns_while_market_is_closed(paper: AppConfig) -> None:
    market_data = healthy_market_data(PROBE_SYMBOL)
    market_data.quotes[PROBE_SYMBOL] = make_quote(PROBE_SYMBOL, bid="0")
    results = _run(paper, broker=FakeBroker(market_open=False), market_data=market_data)
    check = _by_name(results)["Market data"]
    assert check.status is CheckStatus.WARN
    assert check.detail.startswith("one-sided or empty quote (market closed)")


def test_unknown_market_state_does_not_enforce_freshness(paper: AppConfig) -> None:
    market_data = healthy_market_data(PROBE_SYMBOL)
    market_data.quotes[PROBE_SYMBOL] = make_quote(PROBE_SYMBOL, age_seconds=900)
    results = _run(paper, broker=FakeBroker(fail_clock=True), market_data=market_data)
    checks = _by_name(results)
    assert checks["Market data"].status is CheckStatus.PASS
    assert checks["Market clock"].status is CheckStatus.FAIL  # the run still fails overall


@pytest.mark.parametrize(
    ("market_data", "detail"),
    [
        (FakeMarketData(fail=True), "could not fetch quotes: down"),
        (FakeMarketData(), "no quote for SPY"),
        (FakeMarketData(quotes={PROBE_SYMBOL: make_quote(PROBE_SYMBOL)}), "no daily bars for SPY"),
    ],
)
def test_market_data_failures(paper: AppConfig, market_data: FakeMarketData, detail: str) -> None:
    check = _by_name(_run(paper, market_data=market_data))["Market data"]
    assert check.status is CheckStatus.FAIL
    assert check.detail == detail


def test_market_data_factory_error_fails(paper: AppConfig) -> None:
    def broken() -> FakeMarketData:
        raise RuntimeError("bad data client")

    results = run_healthchecks(
        paper,
        make_secrets(),
        broker_factory=FakeBroker,
        market_data_factory=broken,
        news_factory=FakeNews,
        llm_factory=None,
        clock=lambda: NOW,
    )
    assert _by_name(results)["Market data"].detail == "bad data client"
    assert not overall_ok(results)


def test_news_probe_asks_for_all_recent_news(paper: AppConfig) -> None:
    news = FakeNews([make_article()])
    _run(paper, news=news)
    assert news.calls[0]["symbols"] == ()
    assert news.calls[0]["limit"] == 1


def test_news_outage_fails(paper: AppConfig) -> None:
    check = _by_name(_run(paper, news=FakeNews(fail=True)))["News"]
    assert check.status is CheckStatus.FAIL


def test_empty_news_feed_warns(paper: AppConfig) -> None:
    results = _run(paper, news=FakeNews([]))
    assert _by_name(results)["News"].status is CheckStatus.WARN
    assert overall_ok(results)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(-5, "0s"), (42, "42s"), (425, "7m 5s"), (11_520, "3h 12m"), (187_200, "2d 4h")],
)
def test_format_age(seconds: float, text: str) -> None:
    assert format_age(seconds) == text


# ---------- Audit store ----------


def _db_secrets() -> Secrets:
    return make_secrets(database_url=SecretStr("sqlite://"))


def test_audit_store_is_optional_for_paper(paper: AppConfig) -> None:
    check = _by_name(_run(paper))["Audit store"]
    assert check.status is CheckStatus.SKIP
    assert check.detail == "not configured (optional for paper)"


def test_required_audit_store_without_url_fails(paper: AppConfig) -> None:
    required = paper.model_copy(
        update={"storage": paper.storage.model_copy(update={"required": True})}
    )
    results = _run(required)
    assert _by_name(results)["Audit store"].status is CheckStatus.FAIL
    assert not overall_ok(results)


def test_migrated_audit_store_passes(paper: AppConfig) -> None:
    check = _by_name(_run(paper, _db_secrets(), engine_factory=_migrated_engine))["Audit store"]
    assert check.status is CheckStatus.PASS
    assert check.detail == f"sqlite, schema {head_revision()}"


def test_unmigrated_audit_store_warns_or_fails(paper: AppConfig) -> None:
    def empty() -> sa.Engine:
        return make_engine("sqlite://")

    check = _by_name(_run(paper, _db_secrets(), engine_factory=empty))["Audit store"]
    assert check.status is CheckStatus.WARN
    assert "run `trading-agent db upgrade`" in check.detail

    required = paper.model_copy(
        update={"storage": paper.storage.model_copy(update={"required": True})}
    )
    check = _by_name(_run(required, _db_secrets(), engine_factory=empty))["Audit store"]
    assert check.status is CheckStatus.FAIL


def test_unreachable_audit_store(paper: AppConfig) -> None:
    def broken() -> sa.Engine:
        raise RuntimeError("connection refused")

    check = _by_name(_run(paper, _db_secrets(), engine_factory=broken))["Audit store"]
    assert check.status is CheckStatus.WARN
    assert check.detail == "unreachable: connection refused"
