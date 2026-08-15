"""
Per-User Rate Limiting

flask-limiter only observes HTTP routes. The WebSocket chat path and the
background planner both invoke capabilities without passing through a route,
so budgets that must hold everywhere live here and are called from inside the
capability rather than from a decorator.

A budget is a (limit, window_seconds) pair. Fixed-window counters in Redis,
keyed by scope + user + window.
"""

from typing import Sequence

from services import redis_client

# Exact budgets from the design spec §2.2.
SCOUT_BUDGETS: list[tuple[int, int]] = [(3, 3600)]        # 3 per hour
PLAN_BUDGETS: list[tuple[int, int]] = [(5, 86400)]        # 5 per day
CHAT_BUDGETS: list[tuple[int, int]] = [(20, 60), (200, 86400)]


def _key(scope: str, user_id: int, window_seconds: int) -> str:
    return f"ratelimit:{scope}:{user_id}:{window_seconds}"


def allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool:
    """Consume one unit against every budget. False if any is exhausted.

    Note: when an earlier budget passes and a later one denies, the earlier
    counter has still been incremented. That over-counts rather than
    under-counts, which is the safe direction for a spend limit.
    """
    redis = redis_client.get_redis()
    for limit, window_seconds in budgets:
        key = _key(scope, user_id, window_seconds)

        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()

        # ttl < 0 means the key has no expiry (-1) or vanished between the two
        # ops (-2). Re-apply it so a counter can never become permanent and
        # lock the user out for good.
        if ttl is None or ttl < 0:
            redis.expire(key, window_seconds)

        if int(count) > limit:
            return False
    return True


def would_allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool:
    """Read-only check. Does not consume budget.

    Used to avoid offering the user a confirmation button for an action that
    would be refused the moment they click it.
    """
    redis = redis_client.get_redis()
    for limit, window_seconds in budgets:
        raw = redis.get(_key(scope, user_id, window_seconds))
        if raw is not None and int(raw) >= limit:
            return False
    return True
