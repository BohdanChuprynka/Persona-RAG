import time

from persona_rag.bot.rate_limit import TokenBucket


def test_allows_within_budget():
    b = TokenBucket(rate_per_minute=6)
    for _ in range(6):
        assert b.allow(user_id=1)


def test_blocks_over_budget():
    b = TokenBucket(rate_per_minute=2)
    assert b.allow(1)
    assert b.allow(1)
    assert not b.allow(1)


def test_refills_after_time(monkeypatch):
    b = TokenBucket(rate_per_minute=60)  # 1 token/sec
    fake_time = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])
    assert b.allow(1)
    fake_time[0] += 2.0
    assert b.allow(1)
    assert b.allow(1)
