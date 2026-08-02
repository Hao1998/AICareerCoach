"""
tests/test_memory_pgvector_search.py
====================================
Postgres-backed memory search.

The isolation test is the important one: the sqlite-vec path fetched top_k*3
candidates globally and filtered by user afterwards, so a user's own memories
could be crowded out by other users'. The pgvector path filters in SQL, so a
user surrounded by many closer foreign chunks still gets their own results.
"""
from datetime import datetime

import pytest

from models import db, User, UserMemoryChunk

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _fake_embeddings(val: float):
    class _FakeEmb:
        def embed_query(self, q):
            return _vec(val)
    return _FakeEmb()


def _ensure_user(user_id):
    """Create a User row with this explicit id if it doesn't already exist.

    UserMemoryChunk.user_id has a FK to users.id, enforced by Postgres (but
    not by SQLite by default), so any chunk referencing a user_id needs a
    real User row backing it here.
    """
    if db.session.get(User, user_id) is None:
        user = User(
            id=user_id,
            email=f"user{user_id}@example.com",
            username=f"user{user_id}",
            password_hash="x",
        )
        db.session.add(user)
        db.session.commit()


def _insert_chunk(user_id, content, vec_val):
    _ensure_user(user_id)
    chunk = UserMemoryChunk(
        user_id=user_id,
        content=content,
        memory_type="fact",
        embedding=_vec(vec_val),
        session_date=datetime(2026, 1, 1),
    )
    db.session.add(chunk)
    db.session.commit()
    return chunk


@pytest.mark.postgres
def test_pgvector_search_returns_nearest_first(pg_app, monkeypatch):
    with pg_app.app_context():
        _insert_chunk(1, "near chunk", 0.1)
        _insert_chunk(1, "far chunk", 0.9)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.1),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "test query", top_k=2)

        assert "near chunk" in result
        assert result.strip().split("\n")[0].endswith("near chunk")


@pytest.mark.postgres
def test_pgvector_search_isolates_by_user(pg_app, monkeypatch):
    with pg_app.app_context():
        _insert_chunk(1, "user1 chunk", 0.5)
        _insert_chunk(2, "user2 chunk", 0.5)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.5),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "anything", top_k=4)

        assert "user1 chunk" in result
        assert "user2 chunk" not in result


@pytest.mark.postgres
def test_pgvector_search_not_crowded_out_by_other_users(pg_app, monkeypatch):
    """20 closer foreign chunks must not displace the user's own match."""
    with pg_app.app_context():
        _insert_chunk(1, "my only chunk", 0.9)
        for i in range(20):
            _insert_chunk(2, f"foreign chunk {i}", 0.1)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.1),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "anything", top_k=4)

        assert "my only chunk" in result
        assert "foreign chunk" not in result


@pytest.mark.postgres
def test_pgvector_search_empty_for_unknown_user(pg_app, monkeypatch):
    with pg_app.app_context():
        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.5),
        )
        from chatbot.memory import search_memories
        assert "No long-term memories" in search_memories(999, "anything", top_k=4)
