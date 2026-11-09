"""Small in-memory LRU cache for answers.

Key = (data fingerprint, conversation context, normalised question), so a cached answer is
only reused when the same question is asked about the same data in the same context.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


def make_key(*parts: str) -> str:
    normalised = [re.sub(r"\s+", " ", p.strip().lower()) for p in parts]
    return hashlib.sha256("\x1f".join(normalised).encode()).hexdigest()


class LRUCache(Generic[T]):
    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, T] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def set(self, key: str, value: T) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
