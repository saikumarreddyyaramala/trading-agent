"""The crash supervisor and keep-awake (ADR 0021). Written first.

The cycle already contains failures inside each step; the supervisor is the last line for
anything that escapes: restart with a growing pause, alert, and give up on a crash loop.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from tests.conftest import NOW
from trading_agent.orchestration.power import (
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    keep_awake,
)
from trading_agent.orchestration.supervisor import supervise


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> Any:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)

    slept: list[float]


def _setup() -> tuple[_Clock, list[tuple[str, str]]]:
    clock = _Clock()
    clock.slept = []
    return clock, []


def _runner(*outcomes: Exception | None, clock: _Clock | None = None, gap: float = 0) -> Any:
    """A fake cycle: each call raises the next outcome, or returns on None."""
    queue = list(outcomes)
    calls: list[int] = []

    def run() -> None:
        calls.append(1)
        if clock is not None:
            clock.now += timedelta(seconds=gap)
        outcome = queue.pop(0)
        if outcome is not None:
            raise outcome

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_a_clean_exit_is_not_a_crash() -> None:
    clock, alerts = _setup()
    run = _runner(None)
    code = supervise(run, alert=lambda k, t: alerts.append((k, t)), sleep=clock.sleep, clock=clock)
    assert code == 0
    assert alerts == []
    assert clock.slept == []


def test_a_crash_is_alerted_and_the_cycle_restarts() -> None:
    clock, alerts = _setup()
    run = _runner(RuntimeError("database is locked"), None)
    code = supervise(run, alert=lambda k, t: alerts.append((k, t)), sleep=clock.sleep, clock=clock)
    assert code == 0
    assert len(run.calls) == 2
    assert clock.slept == [60]
    ((kind, text),) = alerts
    assert kind == "crash"
    assert "RuntimeError: database is locked" in text
    assert "restarting in 1 min" in text


def test_the_pause_grows_and_is_capped() -> None:
    clock, alerts = _setup()
    crash = RuntimeError("x")
    run = _runner(crash, crash, crash, crash, crash, None)
    supervise(
        run,
        alert=lambda k, t: alerts.append((k, t)),
        sleep=clock.sleep,
        clock=clock,
        max_crashes=10,
        max_backoff_seconds=300,
    )
    assert clock.slept == [60, 120, 240, 300, 300]


def test_a_crash_loop_gives_up_with_a_final_alert() -> None:
    clock, alerts = _setup()
    crash = RuntimeError("x")
    run = _runner(*([crash] * 10))
    code = supervise(
        run,
        alert=lambda k, t: alerts.append((k, t)),
        sleep=clock.sleep,
        clock=clock,
        max_crashes=3,
        first_backoff_seconds=1,
    )
    assert code == 1
    assert len(run.calls) == 3
    assert alerts[-1][0] == "crash"
    assert "stopped after 3 crashes" in alerts[-1][1]
    assert "broker-side stop-loss" in alerts[-1][1]


def test_crashes_far_apart_do_not_add_up() -> None:
    clock, alerts = _setup()
    crash = RuntimeError("x")
    # Each run lasts two hours before crashing: never 3 crashes within an hour.
    run = _runner(crash, crash, crash, crash, None, clock=clock, gap=7200)
    code = supervise(
        run,
        alert=lambda k, t: alerts.append((k, t)),
        sleep=clock.sleep,
        clock=clock,
        max_crashes=3,
    )
    assert code == 0
    assert clock.slept == [60, 60, 60, 60]  # a long healthy run resets the pause too


def test_ctrl_c_is_not_a_crash() -> None:
    clock, alerts = _setup()
    run = _runner(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        supervise(run, alert=lambda k, t: alerts.append((k, t)), sleep=clock.sleep, clock=clock)
    assert alerts == []


def test_a_failing_alert_does_not_stop_the_restart() -> None:
    clock, _ = _setup()

    def bad_alert(kind: str, text: str) -> None:
        raise RuntimeError("alert down")

    run = _runner(RuntimeError("x"), None)
    assert supervise(run, alert=bad_alert, sleep=clock.sleep, clock=clock) == 0
    assert len(run.calls) == 2


# ---------- Keep-awake ----------


def test_keep_awake_asks_windows_and_releases_on_exit() -> None:
    calls: list[int] = []

    def setter(flags: int) -> int:
        calls.append(flags)
        return 1

    with keep_awake(True, platform="win32", setter=setter) as active:
        assert active
        assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert calls[-1] == ES_CONTINUOUS


def test_keep_awake_releases_even_after_a_crash() -> None:
    calls: list[int] = []

    def setter(flags: int) -> int:
        calls.append(flags)
        return 1

    with pytest.raises(RuntimeError), keep_awake(True, platform="win32", setter=setter):
        raise RuntimeError("boom")
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


@pytest.mark.parametrize(("enabled", "platform"), [(False, "win32"), (True, "linux")])
def test_keep_awake_does_nothing_when_off_or_not_windows(enabled: bool, platform: str) -> None:
    calls: list[int] = []

    def setter(flags: int) -> int:
        calls.append(flags)
        return 1

    with keep_awake(enabled, platform=platform, setter=setter) as active:
        assert not active
    assert calls == []


def test_keep_awake_reports_when_windows_refuses() -> None:
    with keep_awake(True, platform="win32", setter=lambda flags: 0) as active:
        assert not active
