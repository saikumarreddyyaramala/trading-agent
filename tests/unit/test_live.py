"""The real-money gates (ADR 0012). Nothing here talks to a broker."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import make_secrets
from trading_agent.domain.models import AccountSnapshot
from trading_agent.execution.live import (
    LIVE_HARD_CAP_USD,
    blocking_reason,
    latest_evaluation,
    live_readiness,
)
from trading_agent.health import CheckResult, CheckStatus
from trading_agent.settings.models import AppConfig

NOW = datetime(2026, 12, 15, 14, tzinfo=UTC)
LIVE: dict[str, Any] = {
    "environment": "live",
    "broker": {"paper": False},
    "storage": {"required": True},
}


def _account(**changes: Any) -> AccountSnapshot:
    values: dict[str, Any] = {
        "account_number_masked": "****9999",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": Decimal("100"),
        "equity": Decimal("100"),
        "buying_power": Decimal("100"),
        "pattern_day_trader": False,
        "trading_blocked": False,
        "account_blocked": False,
        "is_paper": False,
        "multiplier": Decimal("1"),
        "non_marginable_buying_power": Decimal("100"),
    }
    values.update(changes)
    return AccountSnapshot(**values)


def _evaluation(
    folder: Path,
    *,
    decision: str = "GO",
    environment: str = "paper",
    age_days: float = 1,
    name: str = "evaluation-2026-12-14.json",
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    generated = NOW - timedelta(days=age_days)
    path.write_text(
        json.dumps(
            {
                "decision": decision,
                "environment": environment,
                "generated_at": generated.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _check(
    tmp_path: Path,
    *,
    config: dict[str, Any] | None = None,
    confirmed: bool = True,
    account: Any = None,
    environments: Any = None,
    evaluation: bool = True,
) -> dict[str, CheckResult]:
    folder = tmp_path / "reports"
    if evaluation:
        _evaluation(folder)

    def get_account() -> AccountSnapshot:
        if isinstance(account, Exception):
            raise account
        return account or _account()

    def get_environments() -> set[str]:
        if isinstance(environments, Exception):
            raise environments
        return {"live"} if environments is None else environments

    results = live_readiness(
        AppConfig.model_validate(LIVE if config is None else config),
        make_secrets(live_trading_confirmed=confirmed),
        account=get_account,
        audit_environments=get_environments,
        evaluation_dir=folder,
        clock=lambda: NOW,
    )
    return {r.name: r for r in results}


def _failed(checks: dict[str, CheckResult]) -> dict[str, str]:
    return {name: r.detail for name, r in checks.items() if r.status is not CheckStatus.PASS}


def test_every_gate_passes_for_a_ready_live_setup(tmp_path: Path) -> None:
    checks = _check(tmp_path)
    assert list(checks) == [
        "Live switches",
        "Position cap",
        "Paper evaluation",
        "Cash account",
        "Account size",
        "Settled cash",
        "Live audit database",
    ]
    assert _failed(checks) == {}
    assert blocking_reason(list(checks.values())) is None
    assert checks["Paper evaluation"].detail == "evaluation-2026-12-14.json: GO, 1.0 days old"


def test_switches(tmp_path: Path) -> None:
    assert (
        "LIVE_TRADING_CONFIRMED is not true"
        in _failed(_check(tmp_path, confirmed=False))["Live switches"]
    )
    paper = _failed(_check(tmp_path, config={}))["Live switches"]
    assert "environment is not live" in paper
    assert "broker.paper is true" in paper
    assert "storage.required is false" in paper
    killed = _failed(_check(tmp_path, config={**LIVE, "kill_switch": True}))
    assert killed["Live switches"] == "kill switch is on"


def test_position_cap_cannot_be_raised_past_the_hard_cap(tmp_path: Path) -> None:
    failed = _failed(_check(tmp_path, config={**LIVE, "risk": {"max_position_usd": 500}}))
    assert (
        failed["Position cap"]
        == f"risk.max_position_usd $500 exceeds the ${LIVE_HARD_CAP_USD} live hard cap"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decision": "NO-GO"}, "says NO-GO"),
        ({"environment": "live"}, "scores the live record, not paper"),
        ({"age_days": 8}, "is 8 days old (limit 7)"),
    ],
)
def test_the_paper_evaluation_must_be_a_fresh_go(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    _evaluation(tmp_path / "reports", name="evaluation-2026-12-15.json", **kwargs)  # the latest
    assert message in _failed(_check(tmp_path))["Paper evaluation"]


def test_missing_or_unreadable_evaluation(tmp_path: Path) -> None:
    assert (
        "run `trading-agent evaluate`"
        in _failed(_check(tmp_path, evaluation=False))["Paper evaluation"]
    )
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "evaluation-2026-12-15.json").write_text("{broken", encoding="utf-8")
    assert (
        "unreadable evaluation-2026-12-15.json"
        in _failed(_check(tmp_path, evaluation=False))["Paper evaluation"]
    )


def test_latest_evaluation_is_chosen_by_date(tmp_path: Path) -> None:
    assert latest_evaluation(tmp_path / "nowhere") is None
    _evaluation(tmp_path, name="evaluation-2026-11-01.json")
    newest = _evaluation(tmp_path, name="evaluation-2026-12-01.json")
    assert latest_evaluation(tmp_path) == newest


@pytest.mark.parametrize(
    ("account", "message"),
    [
        (_account(is_paper=True), "this is the paper account"),
        (_account(multiplier=Decimal("2")), "margin account (multiplier 2); use a cash account"),
        (_account(trading_blocked=True), "account is blocked"),
    ],
)
def test_only_an_unblocked_live_cash_account(
    tmp_path: Path, account: AccountSnapshot, message: str
) -> None:
    assert message in _failed(_check(tmp_path, account=account))["Cash account"]


@pytest.mark.parametrize(
    ("equity", "message"),
    [("5000", "exceeds the $150 limit"), ("60", "below the $70 floor")],
)
def test_the_account_must_be_small_and_above_the_floor(
    tmp_path: Path, equity: str, message: str
) -> None:
    failed = _failed(_check(tmp_path, account=_account(equity=Decimal(equity))))
    assert message in failed["Account size"]


@pytest.mark.parametrize(
    ("settled", "message"),
    [
        (None, "did not report non-marginable buying power"),
        (Decimal("0.50"), "only $0.50 settled cash"),
    ],
)
def test_settled_cash_is_required(tmp_path: Path, settled: Decimal | None, message: str) -> None:
    failed = _failed(_check(tmp_path, account=_account(non_marginable_buying_power=settled)))
    assert message in failed["Settled cash"]


def test_unreachable_account_fails_every_account_gate(tmp_path: Path) -> None:
    failed = _failed(_check(tmp_path, account=RuntimeError("401 unauthorized")))
    for name in ("Cash account", "Account size", "Settled cash"):
        assert failed[name] == "account unavailable: 401 unauthorized"


def test_live_needs_its_own_audit_database(tmp_path: Path) -> None:
    shared = _failed(_check(tmp_path, environments={"live", "paper"}))
    assert shared["Live audit database"] == (
        "this database also holds paper runs; give live its own DATABASE_URL"
    )
    assert "Live audit database" not in _failed(_check(tmp_path, environments=set()))  # a fresh one
    down = _failed(_check(tmp_path, environments=RuntimeError("no database")))
    assert down["Live audit database"] == "unavailable: no database"


def test_blocking_reason_lists_every_failure(tmp_path: Path) -> None:
    checks = _check(tmp_path, confirmed=False, account=_account(multiplier=Decimal("4")))
    reason = blocking_reason(list(checks.values()))
    assert reason is not None
    assert reason.startswith("live gates failed: Live switches: LIVE_TRADING_CONFIRMED is not true")
    assert "Cash account: margin account" in reason
