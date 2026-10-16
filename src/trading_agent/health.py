"""Pre-flight health checks: is it safe and possible to run today?

Run before every scheduled cycle and on demand (`trading-agent healthcheck`).
Any FAIL means the cycle must not trade.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from trading_agent.domain.models import AccountSnapshot, ChatMessage, LLMRequest
from trading_agent.domain.ports import BrokerPort
from trading_agent.llm.gateway import LLMGateway
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import BROKER_SECRET_FIELDS, LLM_SECRET_FIELDS, Secrets


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
    required = BROKER_SECRET_FIELDS[config.broker.provider]
    if not skip_llm:
        required += LLM_SECRET_FIELDS[config.llm.provider]
    missing = secrets.missing(*required)
    if missing:
        return CheckResult("Secrets", CheckStatus.FAIL, f"missing: {', '.join(missing)}")
    return CheckResult("Secrets", CheckStatus.PASS, "all required keys present (values hidden)")


def _check_broker(config: AppConfig, broker: BrokerPort) -> list[CheckResult]:
    results: list[CheckResult] = []
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
        state = "open" if clock.is_open else "closed"
        results.append(
            CheckResult(
                "Market clock",
                CheckStatus.PASS,
                f"market {state}; next open {clock.next_open:%Y-%m-%d %H:%M %Z}",
            )
        )
    return results


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
    llm_factory: Callable[[], LLMGateway] | None,
) -> list[CheckResult]:
    skip_llm = llm_factory is None
    results = [
        _check_config(config),
        _check_live_safety(config, secrets),
        _check_kill_switch(config),
        _check_secrets(config, secrets, skip_llm),
    ]

    if secrets.missing(*BROKER_SECRET_FIELDS[config.broker.provider]):
        results.append(CheckResult("Broker account", CheckStatus.SKIP, "no broker credentials"))
    else:
        try:
            broker = broker_factory()
        except Exception as exc:
            results.append(CheckResult("Broker account", CheckStatus.FAIL, str(exc)))
        else:
            results.extend(_check_broker(config, broker))

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
