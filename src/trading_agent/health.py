"""Pre-flight health checks: is it safe and possible to run today?

Run before every scheduled cycle and on demand (`trading-agent healthcheck`).
Any FAIL means the cycle must not trade.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from trading_agent.data.quality import quote_issues
from trading_agent.domain.models import AccountSnapshot, ChatMessage, LLMRequest
from trading_agent.domain.ports import BrokerPort, MarketDataPort, NewsPort
from trading_agent.llm.gateway import LLMGateway
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import (
    BROKER_SECRET_FIELDS,
    LLM_SECRET_FIELDS,
    MARKET_DATA_SECRET_FIELDS,
    Secrets,
)
from trading_agent.storage.database import current_revision, head_revision

# A liquid, always-quoted symbol used to prove the market data feed works.
PROBE_SYMBOL = "SPY"
NEWS_PROBE_WINDOW = timedelta(hours=72)


class CheckStatus(StrEnum):
    PASS = "PASS"  # noqa: S105 (a status label, not a password)
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


def _money(value: object) -> str:
    return f"${float(str(value)):,.2f}"


def format_age(seconds: float) -> str:
    """Human-readable age: 42s, 7m 5s, 3h 12m, 2d 4h."""
    total = max(int(seconds), 0)
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _check_config(config: AppConfig) -> CheckResult:
    mode = "paper" if config.broker.paper else "LIVE"
    return CheckResult(
        "Configuration",
        CheckStatus.PASS,
        f"env={config.environment.value}, broker={config.broker.provider} ({mode}), "
        f"model={config.llm.model}",
    )


def _check_live_safety(config: AppConfig, secrets: Secrets) -> CheckResult:
    if not config.is_live:
        return CheckResult("Trading mode", CheckStatus.PASS, "paper trading (simulated money)")
    if not secrets.live_trading_confirmed:
        return CheckResult(
            "Trading mode", CheckStatus.FAIL, "live config but LIVE_TRADING_CONFIRMED is not true"
        )
    return CheckResult("Trading mode", CheckStatus.WARN, "LIVE trading with real money is enabled")


def _check_kill_switch(config: AppConfig) -> CheckResult:
    if config.kill_switch:
        return CheckResult("Kill switch", CheckStatus.WARN, "ON: research only, no orders")
    return CheckResult("Kill switch", CheckStatus.PASS, "off")


def _check_secrets(config: AppConfig, secrets: Secrets, skip_llm: bool) -> CheckResult:
    required = (
        BROKER_SECRET_FIELDS[config.broker.provider]
        + MARKET_DATA_SECRET_FIELDS[config.market_data.provider]
    )
    if not skip_llm:
        required += LLM_SECRET_FIELDS[config.llm.provider]
    missing = secrets.missing(*dict.fromkeys(required))
    if missing:
        return CheckResult("Secrets", CheckStatus.FAIL, f"missing: {', '.join(missing)}")
    return CheckResult("Secrets", CheckStatus.PASS, "all required keys present (values hidden)")


def _check_broker(config: AppConfig, broker: BrokerPort) -> tuple[list[CheckResult], bool | None]:
    """Broker results plus whether the market is open (None when the clock is unavailable)."""
    results: list[CheckResult] = []
    market_open: bool | None = None
    try:
        account = broker.get_account()
    except Exception as exc:
        results.append(CheckResult("Broker account", CheckStatus.FAIL, str(exc)))
    else:
        results.append(_account_result(config, account))

    try:
        clock = broker.get_clock()
    except Exception as exc:
        results.append(CheckResult("Market clock", CheckStatus.FAIL, str(exc)))
    else:
        market_open = clock.is_open
        state = "open" if clock.is_open else "closed"
        results.append(
            CheckResult(
                "Market clock",
                CheckStatus.PASS,
                f"market {state}; next open {clock.next_open:%Y-%m-%d %H:%M %Z}",
            )
        )
    return results, market_open


def _account_result(config: AppConfig, account: AccountSnapshot) -> CheckResult:
    if account.is_paper != config.broker.paper:
        return CheckResult("Broker account", CheckStatus.FAIL, "account type does not match config")
    if account.trading_blocked or account.account_blocked:
        return CheckResult("Broker account", CheckStatus.FAIL, f"blocked (status {account.status})")
    return CheckResult(
        "Broker account",
        CheckStatus.PASS,
        f"{account.status} {account.account_number_masked}, "
        f"equity {_money(account.equity)}, cash {_money(account.cash)}",
    )


def _check_market_data(
    config: AppConfig, market_data: MarketDataPort, *, market_open: bool | None, now: datetime
) -> CheckResult:
    name = "Market data"
    try:
        quote = market_data.get_latest_quotes([PROBE_SYMBOL]).get(PROBE_SYMBOL)
        bars = market_data.get_daily_bars([PROBE_SYMBOL], days=2).get(PROBE_SYMBOL, ())
    except Exception as exc:
        return CheckResult(name, CheckStatus.FAIL, str(exc))
    if quote is None:
        return CheckResult(name, CheckStatus.FAIL, f"no quote for {PROBE_SYMBOL}")
    if not bars:
        return CheckResult(name, CheckStatus.FAIL, f"no daily bars for {PROBE_SYMBOL}")

    feeds = f"quotes={config.market_data.quote_feed}, bars={config.market_data.bars_feed}"
    detail = (
        f"{PROBE_SYMBOL} bid {_money(quote.bid_price)} / ask {_money(quote.ask_price)}, "
        f"quote {format_age(quote.age_seconds(now))} old, "
        f"last close {_money(bars[-1].close)} ({feeds})"
    )
    # Quotes are only expected to be fresh while the market is open.
    max_age = config.risk.max_data_age_seconds if market_open else None
    issues = "; ".join(quote_issues(quote, now=now, max_age_seconds=max_age))
    if issues and market_open:
        return CheckResult(name, CheckStatus.FAIL, f"{issues}: {detail}")
    if issues:
        return CheckResult(name, CheckStatus.WARN, f"{issues} (market closed): {detail}")
    return CheckResult(name, CheckStatus.PASS, detail)


def _check_news(news: NewsPort, *, now: datetime) -> CheckResult:
    try:
        articles = news.get_news((), since=now - NEWS_PROBE_WINDOW, limit=1)
    except Exception as exc:
        return CheckResult("News", CheckStatus.FAIL, str(exc))
    if not articles:
        return CheckResult("News", CheckStatus.WARN, "feed reachable but no articles in 72h")
    latest = articles[0]
    age = format_age((now - latest.created_at).total_seconds())
    return CheckResult("News", CheckStatus.PASS, f"latest article {age} ago ({latest.source})")


def _check_storage(
    config: AppConfig, secrets: Secrets, engine_factory: Callable[[], Any] | None
) -> CheckResult:
    name = "Audit store"
    url = secrets.database_url
    if url is None or not url.get_secret_value() or engine_factory is None:
        if config.storage.required:
            return CheckResult(name, CheckStatus.FAIL, "required but DATABASE_URL is not set")
        return CheckResult(name, CheckStatus.SKIP, "not configured (optional for paper)")
    try:
        engine = engine_factory()
        revision = current_revision(engine)
        backend = engine.dialect.name
    except Exception as exc:
        status = CheckStatus.FAIL if config.storage.required else CheckStatus.WARN
        return CheckResult(name, status, f"unreachable: {exc}")
    head = head_revision()
    if revision != head:
        status = CheckStatus.FAIL if config.storage.required else CheckStatus.WARN
        return CheckResult(
            name,
            status,
            f"{backend} schema at {revision or 'nothing'}, expected {head}: "
            "run `trading-agent db upgrade`",
        )
    return CheckResult(name, CheckStatus.PASS, f"{backend}, schema {revision}")


def _check_llm(gateway: LLMGateway) -> CheckResult:
    request = LLMRequest(
        system="You are a health check. Follow the instruction exactly.",
        messages=(ChatMessage(role="user", content="Reply with the single word: OK"),),
        # Reasoning models spend hidden tokens before answering, so leave headroom.
        max_tokens=512,
        tags={"purpose": "healthcheck"},
    )
    try:
        response = gateway.complete(request)
    except Exception as exc:
        return CheckResult("LLM", CheckStatus.FAIL, str(exc))
    if not response.text.strip():
        return CheckResult(
            "LLM", CheckStatus.FAIL, f"{response.model} returned no text ({response.stop_reason})"
        )
    return CheckResult(
        "LLM",
        CheckStatus.PASS,
        f"{response.model} replied in {response.latency_ms:.0f} ms",
    )


def run_healthchecks(
    config: AppConfig,
    secrets: Secrets,
    *,
    broker_factory: Callable[[], BrokerPort],
    market_data_factory: Callable[[], MarketDataPort],
    news_factory: Callable[[], NewsPort],
    llm_factory: Callable[[], LLMGateway] | None,
    clock: Callable[[], datetime] | None = None,
    engine_factory: Callable[[], Any] | None = None,
) -> list[CheckResult]:
    now = clock() if clock else datetime.now(UTC)
    skip_llm = llm_factory is None
    results = [
        _check_config(config),
        _check_live_safety(config, secrets),
        _check_kill_switch(config),
        _check_secrets(config, secrets, skip_llm),
    ]

    market_open: bool | None = None
    if secrets.missing(*BROKER_SECRET_FIELDS[config.broker.provider]):
        results.append(CheckResult("Broker account", CheckStatus.SKIP, "no broker credentials"))
    else:
        try:
            broker = broker_factory()
        except Exception as exc:
            results.append(CheckResult("Broker account", CheckStatus.FAIL, str(exc)))
        else:
            broker_results, market_open = _check_broker(config, broker)
            results.extend(broker_results)

    if secrets.missing(*MARKET_DATA_SECRET_FIELDS[config.market_data.provider]):
        results.append(CheckResult("Market data", CheckStatus.SKIP, "no market data credentials"))
        results.append(CheckResult("News", CheckStatus.SKIP, "no market data credentials"))
    else:
        try:
            market_data = market_data_factory()
            news = news_factory()
        except Exception as exc:
            results.append(CheckResult("Market data", CheckStatus.FAIL, str(exc)))
        else:
            results.append(
                _check_market_data(config, market_data, market_open=market_open, now=now)
            )
            results.append(_check_news(news, now=now))

    results.append(_check_storage(config, secrets, engine_factory))

    if llm_factory is None:
        results.append(CheckResult("LLM", CheckStatus.SKIP, "skipped by request"))
    elif secrets.missing(*LLM_SECRET_FIELDS[config.llm.provider]):
        results.append(CheckResult("LLM", CheckStatus.SKIP, "no LLM credentials"))
    else:
        try:
            gateway = llm_factory()
        except Exception as exc:
            results.append(CheckResult("LLM", CheckStatus.FAIL, str(exc)))
        else:
            results.append(_check_llm(gateway))

    return results


def overall_ok(results: list[CheckResult]) -> bool:
    return not any(r.status is CheckStatus.FAIL for r in results)
