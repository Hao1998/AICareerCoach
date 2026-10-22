"""Guards for gunicorn.conf.py.

Flask-SocketIO does not support multiple gunicorn workers: gunicorn's load
balancing cannot pin a client to one worker, so the Socket.IO handshake (a few
HTTP long-polling requests before the WebSocket upgrade) lands on different
workers and fails with "session ID unknown". Scale out by running several
single-worker instances behind a sticky load balancer instead.
"""

import runpy
from pathlib import Path

CONF_PATH = str(Path(__file__).resolve().parent.parent / "gunicorn.conf.py")


def test_single_worker_per_instance():
    conf = runpy.run_path(CONF_PATH)
    assert conf["workers"] == 1


def test_bind_defaults_to_port_5001(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    conf = runpy.run_path(CONF_PATH)
    assert conf["bind"] == "0.0.0.0:5001"


def test_bind_honours_port_env(monkeypatch):
    monkeypatch.setenv("PORT", "5003")
    conf = runpy.run_path(CONF_PATH)
    assert conf["bind"] == "0.0.0.0:5003"
