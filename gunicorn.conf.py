"""
Gunicorn configuration for production deployment.

Uses gevent workers so each SSE stream is a lightweight coroutine instead of
an OS thread — allowing thousands of concurrent streams per instance with
minimal memory overhead.
"""

import multiprocessing

# ── Workers ───────────────────────────────────────────────────────────────────
worker_class = "gevent"
workers = multiprocessing.cpu_count() * 2 + 1
worker_connections = 1000       # max concurrent greenlets per worker

# ── Timeouts ──────────────────────────────────────────────────────────────────
# SSE streams stay open for the full LLM response duration (can be 30-60s).
# Default gunicorn timeout (30s) would kill mid-stream responses.
timeout = 300
keepalive = 5

# ── Binding ───────────────────────────────────────────────────────────────────
bind = "0.0.0.0:5001"

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"
errorlog = "-"
loglevel = "info"
