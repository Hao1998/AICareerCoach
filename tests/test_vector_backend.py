"""
tests/test_vector_backend.py
============================
The VECTOR_BACKEND switch. Read from os.environ on every call so it works in
threads without an app context, and so tests can toggle it.
"""

from services.vector_backend import is_redis_vectors


def test_off_when_unset(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    assert is_redis_vectors() is False


def test_on_when_redis(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "redis")
    assert is_redis_vectors() is True


def test_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", " Redis ")
    assert is_redis_vectors() is True


def test_other_values_are_off(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    assert is_redis_vectors() is False
