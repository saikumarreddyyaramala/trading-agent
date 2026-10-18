"""One trading cycle at a time (ADR 0021).

Two copies of `trading-agent run` would each analyse and enter trades, and would fight over
Telegram (HTTP 409). `run` therefore takes an exclusive operating-system lock on a file. The
operating system releases it when the process ends, even after a crash or a power cut, so
there is never a stale lock to clean up. The holder's process id is kept in a side file
(Windows does not let other processes read a locked byte) to say who holds it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


class AlreadyRunningError(RuntimeError):
    """Another trading cycle holds the lock."""


def _try_lock(handle: IO[str]) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - exercised on Linux CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[str]) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised on Linux CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:  # pragma: no cover - closing the file releases it anyway
        pass


@contextmanager
def single_instance(path: Path) -> Iterator[None]:
    """Hold the cycle lock for the duration, or raise AlreadyRunningError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = path.with_suffix(".pid")
    handle = path.open("a+", encoding="utf-8")
    try:
        if not _try_lock(handle):
            try:
                holder = pid_file.read_text(encoding="utf-8").strip() or "unknown"
            except OSError:
                holder = "unknown"
            raise AlreadyRunningError(
                f"another trading cycle is already running (process {holder}). Stop it first "
                "(Ctrl+C in its window), then start this one."
            )
        try:
            pid_file.write_text(str(os.getpid()), encoding="utf-8")
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()
