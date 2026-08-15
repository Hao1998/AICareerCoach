"""
Pending Action Store

Backs the chat confirmation gate. A gated capability does not execute when the
model calls it — it proposes, storing the capability name and its
server-resolved arguments under a single-use nonce. The client is handed only
the nonce, so nothing downstream can alter what eventually runs.
"""

import json
import secrets

# Import the MODULE, not the function. `from services.redis_client import
# get_redis` binds the name at import time, which defeats the test fixture's
# monkeypatch of services.redis_client.get_redis — tests would then write to
# the developer's real Redis instead of the in-memory double, and silently
# pass or fail on real state. Resolve the attribute at call time instead.
from services import redis_client

TTL_SECONDS = 300

_PREFIX = "pending_action:"


def propose(user_id: int, capability: str, args: dict, label: str) -> str:
    """Store a proposed action and return its single-use nonce."""
    nonce = secrets.token_urlsafe(24)
    payload = json.dumps({
        "user_id": user_id,
        "capability": capability,
        "args": args,
        "label": label,
    })
    redis_client.get_redis().set(f"{_PREFIX}{nonce}", payload, ex=TTL_SECONDS)
    return nonce


def claim(user_id: int, nonce: str) -> dict | None:
    """Consume a pending action. Returns None if unknown, expired, or not this user's.

    A nonce belonging to another user is left in place deliberately. Deleting
    it on a failed ownership check would let any authenticated user burn every
    pending action issued to anyone else.
    """
    redis = redis_client.get_redis()
    key = f"{_PREFIX}{nonce}"

    raw = redis.get(key)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        redis.delete(key)
        return None

    if data.get("user_id") != user_id:
        return None

    redis.delete(key)
    return data
