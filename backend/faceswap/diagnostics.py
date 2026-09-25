from __future__ import annotations

from collections import deque
from datetime import datetime
import threading


_lock = threading.Lock()
_entries: deque[str] = deque(maxlen=120)


def record(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    line = f"{stamp} {message}"
    with _lock:
        _entries.append(line)


def entries() -> list[str]:
    with _lock:
        return list(_entries)


def clear() -> None:
    with _lock:
        _entries.clear()
