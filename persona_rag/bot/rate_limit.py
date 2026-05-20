from __future__ import annotations

import time
from collections import defaultdict


class TokenBucket:
    """Per-user token bucket. Refills continuously."""

    def __init__(self, *, rate_per_minute: int) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = float(rate_per_minute)
        self._tokens: dict[int, float] = defaultdict(lambda: self.capacity)
        self._last: dict[int, float] = defaultdict(lambda: time.monotonic())

    def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        elapsed = now - self._last[user_id]
        self._tokens[user_id] = min(self.capacity, self._tokens[user_id] + elapsed * self.rate)
        self._last[user_id] = now
        if self._tokens[user_id] >= 1.0:
            self._tokens[user_id] -= 1.0
            return True
        return False
