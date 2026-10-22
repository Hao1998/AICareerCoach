"""
Gunicorn configuration for production deployment.

Uses gevent-websocket workers so WebSocket connections and streaming responses
are lightweight coroutines — allowing thousands of concurrent connections per
instance with minimal memory overhead.

Scaling: exactly ONE worker per gunicorn instance. Flask-SocketIO does not
support multiple gunicorn workers — gunicorn cannot pin a client to one worker,
so the Socket.IO handshake (HTTP long-polling before the WebSocket upgrade)
lands on different workers and fails with "session ID unknown". To use more
cores, run several instances on different ports (PORT=5001, 5002, ...) behind
nginx with `ip_hash` (see nginx.conf), or more containers behind a load
balancer with sticky sessions. Redis `message_queue` (factory.py) relays emits
between instances.
"""

import os

# ── Workers ───────────────────────────────────────────────────────────────────
worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
workers = 1                     # never raise — see module docstring
worker_connections = 1000       # max concurrent greenlets per worker

# ── Timeouts ──────────────────────────────────────────────────────────────────
# WebSocket/streaming responses can run 30-60s for LLM completions.
timeout = 300
keepalive = 5

# ── Binding ───────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"
errorlog = "-"
loglevel = "info"


# ── Post-fork ────────────────────────────────────────────────────────────────
def post_fork(server, worker):
    """Pre-warm the embedding model in the worker after fork."""
    from jobs.utils import get_embeddings
    get_embeddings()
