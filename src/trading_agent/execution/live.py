"""Gates for real money (ADR 0012). Live entries happen only when every gate passes.

Exits never depend on these: a live position can always be managed, force-closed, and
flattened. A failed gate only blocks new entries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading_agent.domain.models import AccountSnapshot
from trading_agent.health import CheckResult, CheckStatus
from trading_agent.settings.models import AppConfig
from trading_agent.settings.secrets import Secrets

# No live order may exceed this, whatever the config says (protects against config typos).
LIVE_HARD_CAP_USD = Decimal("100")
MIN_SETTLED_CASH_USD = Decimal("1")


def _check(name: str, problems: list[str], ok: str) -> CheckResult:
    if problems:
        return CheckResult(name, CheckStatus.FAIL, "; ".join(problems))
    return CheckResult(name, CheckStatus.PASS, ok)


def _switches(config: AppConfig, secrets: Secrets) -> CheckResult:
    problems = []
    if not config.is_live:
        problems.append("environment is not live")
    if config.broker.paper:
        problems.append("broker.paper is true")
    if not secrets.live_trading_confirmed:
        problems.append("LIVE_TRADING_CONFIRMED is not true")
    if not config.storage.required:
        problems.append("storage.required is false")
    if config.kill_switch:
        problems.append("kill switch is on")
    return _check("Live switches", problems, "live config, confirmed, audit required")


def _position_cap(config: AppConfig) -> CheckResult:
    cap = config.risk.max_position_usd
    problems = (
        [f"risk.max_position_usd ${cap} exceeds the ${LIVE_HARD_CAP_USD} live hard cap"]
        if cap > LIVE_HARD_CAP_USD
        else []
    )
    return _check("Position cap", problems, f"${cap} per trade (hard cap ${LIVE_HARD_CAP_USD})")


def latest_evaluation(directory: Path) -> Path | None:
    files = sorted(directory.glob("evaluation-*.json")) if directory.is_dir() else []
    return files[-1] if files else None


def _evaluation(config: AppConfig, directory: Path, now: datetime) -> CheckResult:
    name = "Paper evaluation"
    path = latest_evaluation(directory)
    if path is None:
        return CheckResult(
            name, CheckStatus.FAIL, f"no evaluation in {directory}: run `trading-agent evaluate`"
        )
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(card["generated_at"])
        decision, environment = card["decision"], card["environment"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return CheckResult(name, CheckStatus.FAIL, f"unreadable {path.name}: {exc}")
    age_days = (now - generated).total_seconds() / 86_400
    problems = []
    if environment != "paper":
        problems.append(f"{path.name} scores the {environment} record, not paper")
    if decision != "GO":
        problems.append(f"{path.name} says {decision}")
    if age_days > config.live.max_evaluation_age_days:
        problems.append(
            f"{path.name} is {age_days:.0f} days old (limit {config.live.max_evaluation_age_days})"
        )
    return _check(name, problems, f"{path.name}: GO, {age_days:.1f} days old")


def _account(
    config: AppConfig, account: AccountSnapshot | None, error: str | None
) -> list[CheckResult]:
    if account is None:
        reason = f"account unavailable: {error}"
        return [
            CheckResult(name, CheckStatus.FAIL, reason)
            for name in ("Cash account", "Account size", "Settled cash")
        ]
    cash_problems = []
    if account.is_paper:
        cash_problems.append("this is the paper account")
    if account.multiplier != 1:
        cash_problems.append(
            f"margin account (multiplier {account.multiplier}); use a cash account"
        )
    if account.trading_blocked or account.account_blocked:
        cash_problems.append(f"account is blocked (status {account.status})")

    size_problems = []
    if account.equity > config.live.max_account_equity_usd:
        size_problems.append(
            f"equity ${account.equity} exceeds the ${config.live.max_account_equity_usd} limit"
        )
    if account.equity < config.risk.account_floor_usd:
        size_problems.append(
            f"equity ${account.equity} is below the ${config.risk.account_floor_usd} floor"
        )

    settled = account.non_marginable_buying_power
    settled_problems = (
        ["the broker did not report non-marginable buying power"]
        if settled is None
        else [f"only ${settled} settled cash available"]
        if settled < MIN_SETTLED_CASH_USD
        else []
    )
    return [
        _check("Cash account", cash_problems, f"cash account {account.account_number_masked}"),
        _check("Account size", size_problems, f"equity ${account.equity}"),
        _check("Settled cash", settled_problems, f"${settled} available without margin"),
    ]


def _audit(environments: set[str] | None, error: str | None) -> CheckResult:
    if environments is None:
        return CheckResult("Live audit database", CheckStatus.FAIL, f"unavailable: {error}")
    other = sorted(environments - {"live"})
    problems = (
        [f"this database also holds {', '.join(other)} runs; give live its own DATABASE_URL"]
        if other
        else []
    )
    return _check("Live audit database", problems, "dedicated to live runs")


def live_readiness(
    config: AppConfig,
    secrets: Secrets,
    *,
    account: Callable[[], AccountSnapshot],
    audit_environments: Callable[[], set[str]],
    evaluation_dir: Path,
    clock: Callable[[], datetime] | None = None,
) -> list[CheckResult]:
    """Every gate, each as a PASS or FAIL with the reason. Never raises."""
    now = clock() if clock else datetime.now(UTC)
    try:
        snapshot, account_error = account(), None
    except Exception as exc:
        snapshot, account_error = None, str(exc)
    try:
        environments, audit_error = audit_environments(), None
    except Exception as exc:
        environments, audit_error = None, str(exc)
    return [
        _switches(config, secrets),
        _position_cap(config),
        _evaluation(config, evaluation_dir, now),
        *_account(config, snapshot, account_error),
        _audit(environments, audit_error),
    ]


def blocking_reason(results: list[CheckResult]) -> str | None:
    """None when every gate passed; otherwise why live entries are blocked."""
    failed = [f"{r.name}: {r.detail}" for r in results if r.status is not CheckStatus.PASS]
    return None if not failed else "live gates failed: " + " | ".join(failed)
