"""
Gunicorn configuration for production deployment.

Uses gevent-websocket workers so WebSocket connections and streaming responses
are lightweight coroutines — allowing thousands of concurrent connections per
instance with minimal memory overhead.
"""

import multiprocessing

# ── Workers ───────────────────────────────────────────────────────────────────
worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
workers = multiprocessing.cpu_count() * 2 + 1
worker_connections = 1000       # max concurrent greenlets per worker

# ── Timeouts ──────────────────────────────────────────────────────────────────
# WebSocket/streaming responses can run 30-60s for LLM completions.
timeout = 300
keepalive = 5

# ── Binding ───────────────────────────────────────────────────────────────────
bind = "0.0.0.0:5001"

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"
errorlog = "-"
loglevel = "info"


# ── Post-fork ────────────────────────────────────────────────────────────────
def post_fork(server, worker):
    """Pre-warm the embedding model in each worker after fork."""
    from jobs.utils import get_embeddings
    get_embeddings()
