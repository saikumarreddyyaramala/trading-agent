"""Keep the computer awake while the cycle runs (ADR 0021).

On Windows this asks the system, for this process only, not to fall asleep when idle
(`SetThreadExecutionState`). It changes no power settings and ends when the cycle stops.
It cannot stop a sleep the user asks for, such as closing a laptop lid or pressing sleep.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from trading_agent.observability.logging import get_logger

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_setter() -> Callable[[int], int]:  # pragma: no cover - Windows only
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint
    return lambda flags: int(kernel32.SetThreadExecutionState(flags))


@contextmanager
def keep_awake(
    enabled: bool = True,
    *,
    platform: str = sys.platform,
    setter: Callable[[int], int] | None = None,
    logger: Any = None,
) -> Iterator[bool]:
    """Yields True while Windows has agreed to stay awake."""
    log = logger or get_logger(__name__)
    if not enabled or platform != "win32":
        yield False
        return
    request = setter or _windows_setter()
    active = request(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    log.info("power.keep_awake", active=active)
    try:
        yield active
    finally:
        if active:
            request(ES_CONTINUOUS)  # back to normal sleep rules
