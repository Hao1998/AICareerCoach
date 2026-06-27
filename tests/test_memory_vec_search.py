"""
tests/test_memory_vec_search.py
================================
Offline tests for the sqlite-vec-backed search_memories function.

Uses a real in-memory SQLite DB (TestConfig) with the sqlite-vec extension
loaded by the factory. Embeddings are 768-dim constant vectors so no model
is needed — we monkeypatch _get_embeddings_for_memory to return controlled
vectors.

What each test covers
---------------------
test_search_memories_returns_nearest_chunk
    Given two chunks with known embeddings, the one closest to the query
    embedding is returned first.

test_search_memories_isolates_by_user
    A chunk belonging to user_id=2 must never appear in user_id=1 results.

test_search_memories_empty
    Returns the "no memories found" string when the user has no chunks.

test_search_memories_falls_back_on_non_sqlite
    When _USE_VEC is forced to False, search_memories uses the legacy cosine
    path and still returns a correct result.
"""
import struct
import pytest
from factory import create_app
from models import db, UserMemoryChunk
from datetime import datetime

FAKE_DIM = 768


def _ser(floats):
    return struct.pack(f"{len(floats)}f", *floats)


def _vec(val: float) -> list:
    """Return a 768-dim vector where every element is `val`."""
    return [val] * FAKE_DIM


def _fake_embeddings(val: float):
    """Return a fake embeddings object whose embed_query always returns _vec(val)."""
    _v = val

    class _FakeEmb:
        def embed_query(self, q):
            return _vec(_v)

    return _FakeEmb()


@pytest.fixture
def app():
    application = create_app('test', skip_api_check=True)
    with application.app_context():
        db.create_all()
        from sqlalchemy import text
        with application.extensions['sqlalchemy'].engine.connect() as conn:
            conn.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_user_memories "
                "USING vec0(embedding float[768])"
            ))
            conn.commit()
        yield application
        db.session.remove()
        db.drop_all()


def _insert_chunk(app, user_id, content, vec_val, memory_type="fact"):
    """Insert a UserMemoryChunk + corresponding vec row."""
    from sqlalchemy import text
    emb = _vec(vec_val)
    chunk = UserMemoryChunk(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        embedding=emb,
        session_date=datetime(2026, 1, 1),
    )
    db.session.add(chunk)
    db.session.flush()
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        conn.execute(
            text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
            {"rid": chunk.id, "emb": _ser(emb)}
        )
        conn.commit()
    db.session.commit()
    return chunk


def test_search_memories_returns_nearest_chunk(app, monkeypatch):
    _insert_chunk(app, user_id=1, content="near chunk", vec_val=0.1)
    _insert_chunk(app, user_id=1, content="far chunk", vec_val=0.9)

    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: _fake_embeddings(0.1)
    )

    from chatbot.memory import search_memories
    result = search_memories(1, "test query", top_k=2)
    assert "near chunk" in result
    lines = result.strip().split("\n")
    assert lines[0].endswith("near chunk"), f"unexpected order: {result}"


def test_search_memories_isolates_by_user(app, monkeypatch):
    _insert_chunk(app, user_id=1, content="user1 chunk", vec_val=0.5)
    _insert_chunk(app, user_id=2, content="user2 chunk", vec_val=0.5)

    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: _fake_embeddings(0.5)
    )

    from chatbot.memory import search_memories
    result = search_memories(1, "anything", top_k=4)
    assert "user1 chunk" in result
    assert "user2 chunk" not in result


def test_search_memories_empty(app, monkeypatch):
    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: _fake_embeddings(0.5)
    )
    from chatbot.memory import search_memories
    result = search_memories(999, "anything", top_k=4)
    assert "No long-term memories" in result


def test_search_memories_falls_back_on_non_sqlite(app, monkeypatch):
    _insert_chunk(app, user_id=1, content="fallback chunk", vec_val=0.5)

    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: _fake_embeddings(0.5)
    )
    import chatbot.memory as mem_mod
    monkeypatch.setattr(mem_mod, "_USE_VEC", False)

    from chatbot.memory import search_memories
    result = search_memories(1, "anything", top_k=4)
    assert "fallback chunk" in result
