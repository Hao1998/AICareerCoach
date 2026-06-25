"""
Chatbot Memory

Two-tier session memory for the Career Coach agent:
  Tier 1 (Hot)  — last 10 raw messages from DB, loaded on every request
  Tier 2 (Warm) — rolling LLM-generated summary stored in AgentConfig,
                  updated when a session boundary (5-min gap) is detected

Also handles explicit preference extraction from conversation text.
"""

import json
import logging
import struct
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# _USE_VEC is True when sqlite-vec is installed. Monkeypatched to False in
# tests that exercise the cosine fallback path.
_USE_VEC = False
try:
    import sqlite_vec as _sqlite_vec  # noqa: F401
    _USE_VEC = True
except ImportError:
    pass


def _get_embeddings_for_memory():
    """Thin wrapper so tests can monkeypatch without touching jobs.utils."""
    from jobs.utils import get_embeddings
    return get_embeddings()


def _serialize_f32(floats: list) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)

from langsmith import traceable

from models import AgentConfig, ChatMessage, UserMemoryChunk, db
from services.db_lock import safe_commit

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_MINUTES = 5


def _extract_json(content: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        content = content.split("\n", 1)[-1]
        # Remove closing fence
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return content.strip()


@traceable(run_type="llm", name="preference-extractor")
def extract_explicit_preferences(messages, llm, existing: Optional[dict] = None) -> Optional[dict]:
    """Extract structured job preferences from a conversation, merging with existing ones."""
    if not messages:
        return existing

    conversation_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = f"""You are extracting job search preferences from a career coaching conversation.

Existing preferences already stored (do NOT lose these):
{json.dumps(existing or {})}

New conversation:
{conversation_text}

Extract ALL job preferences the user has mentioned (including from the existing preferences above).
Return a single valid JSON object with these keys (omit a key if not mentioned at all):
- remote_only: true/false
- avoid_sectors: list of sectors/industries to avoid (e.g. ["finance", "startups"])
- preferred_company_size: "small", "medium", "large", or "any"
- preferred_locations: list of locations or ["Remote"]
- avoid_job_types: list of job types to avoid (e.g. ["contract", "part-time"])
- summary: one sentence describing the user's preferences in plain English

If no job preferences were mentioned at all, return exactly: null

Return ONLY the JSON object or null, no explanation."""

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, 'content') else str(result)
        content = _extract_json(content)
        if not content or content.lower() == 'null':
            return existing
        prefs = json.loads(content)
        return prefs if isinstance(prefs, dict) else existing
    except Exception as e:
        logger.error("Preference extraction error: %s", e)
        return existing


@traceable(run_type="llm", name="session-summarizer")
def summarize_session(messages, llm):
    """Summarize a list of ChatMessage rows into a rolling session summary string."""
    if not messages:
        return None

    conversation_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = f"""Summarize the key career coaching insights from this conversation in 3-5 sentences.
Focus on: job preferences mentioned, skills discussed, career goals expressed, actions taken, and feedback given.

Conversation:
{conversation_text}

Summary:"""

    try:
        result = llm.invoke(prompt)
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        logger.error("Session summarization error: %s", e)
        return None


def detect_session_boundary(app, user_id: int) -> bool:
    """Return True if the last message was more than SESSION_TIMEOUT_MINUTES ago."""
    with app.app_context():
        last_msg = (ChatMessage.query
                    .filter_by(user_id=user_id)
                    .order_by(ChatMessage.timestamp.desc())
                    .first())
        if not last_msg or not last_msg.timestamp:
            return False
        return datetime.utcnow() - last_msg.timestamp > timedelta(minutes=SESSION_TIMEOUT_MINUTES)


@traceable(run_type="chain", name="session-close-and-summarize")
def close_and_summarize_session(app, user_id: int):
    """Summarize the current session and merge it into AgentConfig.conversation_summary."""
    from services.llm_service import get_llm

    with app.app_context():
        messages = (ChatMessage.query
                    .filter_by(user_id=user_id)
                    .order_by(ChatMessage.timestamp.asc())
                    .all())
        if not messages:
            return

        llm = get_llm()
        new_summary = summarize_session(messages, llm)
        if not new_summary:
            return

        config = AgentConfig.query.filter_by(user_id=user_id).first()
        if not config:
            config = AgentConfig(user_id=user_id)
            db.session.add(config)

        if config.conversation_summary:
            merge_prompt = f"""Merge these two conversation summaries into one concise summary (3-5 sentences).
Focus on: job preferences, skills, career goals, actions taken, feedback given.

Previous summary:
{config.conversation_summary}

New session summary:
{new_summary}

Merged summary:"""
            try:
                merged = llm.invoke(merge_prompt)
                config.conversation_summary = merged.content if hasattr(merged, 'content') else str(merged)
            except Exception as e:
                logger.error("Summary merge error: %s", e)
                config.conversation_summary = new_summary
        else:
            config.conversation_summary = new_summary

        updated_prefs = extract_explicit_preferences(messages, llm, config.explicit_preferences)
        if updated_prefs:
            config.explicit_preferences = updated_prefs
            logger.info("Updated explicit preferences for user %s: %s",
                        user_id, updated_prefs.get('summary', ''))

        safe_commit()

        session_date = messages[-1].timestamp if messages else datetime.utcnow()
        index_session_memories(app, user_id, messages, llm, session_date, summary=new_summary)


@traceable(run_type="llm", name="memory-fact-extractor")
def extract_memory_facts(messages, llm) -> list[str]:
    """Extract 3-5 discrete, self-contained facts from a conversation for long-term memory."""
    if not messages:
        return []

    conversation_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = f"""You are extracting memorable facts from a career coaching conversation for long-term storage.

Conversation:
{conversation_text}

Extract 3-5 discrete, self-contained facts the user revealed about themselves.
Each fact must stand alone — someone reading it with no other context should fully understand it.
Focus on: career goals, skills mentioned, job preferences, past experience highlights, decisions made, feedback given.

Return a JSON array of strings. Each string is one fact.
If there are no notable facts, return an empty array [].

Example output:
["User has 5 years of Python backend experience.", "User prefers remote-only roles.", "User is targeting senior software engineer positions at mid-size companies."]

Return ONLY the JSON array, no explanation."""

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, 'content') else str(result)
        content = _extract_json(content)
        if not content:
            return []
        facts = json.loads(content)
        return facts if isinstance(facts, list) else []
    except Exception as e:
        logger.error("Memory fact extraction error: %s", e)
        return []


def index_session_memories(app, user_id: int, messages: list, llm, session_date: datetime,
                           summary: str | None = None):
    """Embed and store session summary + discrete facts as UserMemoryChunk rows."""
    from jobs.utils import get_embeddings as _get_embeddings

    if summary is None:
        summary = summarize_session(messages, llm)
    facts = extract_memory_facts(messages, llm)

    chunks_to_index = []
    if summary:
        chunks_to_index.append(('session_summary', summary))
    for fact in facts:
        if fact and fact.strip():
            chunks_to_index.append(('fact', fact.strip()))

    if not chunks_to_index:
        return

    with app.app_context():
        for memory_type, content in chunks_to_index:
            try:
                vec = _get_embeddings().embed_query(content)
                embedding = vec if isinstance(vec, list) else list(vec)
            except Exception as e:
                logger.error("Failed to embed memory chunk: %s", e)
                embedding = None

            chunk = UserMemoryChunk(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                embedding=embedding,
                session_date=session_date,
            )
            db.session.add(chunk)

        try:
            safe_commit()
            logger.info("Indexed %d memory chunks for user %s", len(chunks_to_index), user_id)
        except Exception as e:
            logger.error("Failed to commit memory chunks: %s", e)
            return  # nothing to vec-index if commit failed

        if _USE_VEC:
            _sync_chunks_to_vec(app)


def _sync_chunks_to_vec(app):
    """Insert any UserMemoryChunk rows missing from vec_user_memories."""
    from sqlalchemy import text

    engine = app.extensions['sqlalchemy'].engine
    try:
        with engine.connect() as conn:
            missing = conn.execute(text(
                "SELECT c.id, c.embedding "
                "FROM user_memory_chunks c "
                "LEFT JOIN vec_user_memories v ON v.rowid = c.id "
                "WHERE c.embedding IS NOT NULL AND v.rowid IS NULL"
            )).fetchall()

            for row_id, embedding_json in missing:
                if not isinstance(embedding_json, list):
                    continue
                conn.execute(
                    text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
                    {"rid": row_id, "emb": _serialize_f32(embedding_json)}
                )
            conn.commit()
            if missing:
                logger.info("Synced %d new memory chunks to vec table", len(missing))
    except Exception as e:
        logger.error("Failed to sync memory chunks to vec table: %s", e)


def search_memories(user_id: int, query: str, top_k: int = 4) -> str:
    """Retrieve the most semantically relevant past memories for a given query.

    Uses sqlite-vec ANN when available; falls back to O(n) cosine on Postgres
    or when sqlite-vec is not installed.
    """
    try:
        query_vec = _get_embeddings_for_memory().embed_query(query)
    except Exception as e:
        logger.error("Failed to embed memory search query: %s", e)
        return "Memory search temporarily unavailable."

    if _USE_VEC:
        return _search_memories_vec(user_id, query_vec, top_k)
    return _search_memories_cosine(user_id, query_vec, top_k)


def _search_memories_vec(user_id: int, query_vec, top_k: int) -> str:
    """ANN search via sqlite-vec virtual table."""
    from sqlalchemy import text
    from flask import current_app

    engine = current_app.extensions['sqlalchemy'].engine
    query_bytes = _serialize_f32(list(query_vec))

    try:
        with engine.connect() as conn:
            # Fetch top_k*3 candidates globally (vec0 has no user-filter support
            # in v0.1.x), then filter to this user in Python.
            rows = conn.execute(
                text(
                    "SELECT rowid, distance "
                    "FROM vec_user_memories "
                    "WHERE embedding MATCH :q "
                    "ORDER BY distance "
                    "LIMIT :lim"
                ),
                {"q": query_bytes, "lim": top_k * 3}
            ).fetchall()
    except Exception as e:
        logger.error("sqlite-vec query failed, falling back to cosine: %s", e)
        return _search_memories_cosine(user_id, query_vec, top_k)

    if not rows:
        return "No long-term memories found for this user yet."

    candidate_ids = [r[0] for r in rows]
    distance_by_id = {r[0]: r[1] for r in rows}

    chunks = (UserMemoryChunk.query
              .filter(UserMemoryChunk.id.in_(candidate_ids))
              .filter_by(user_id=user_id)
              .all())

    if not chunks:
        return "No long-term memories found for this user yet."

    chunks.sort(key=lambda c: distance_by_id.get(c.id, float("inf")))
    chunks = chunks[:top_k]

    lines = []
    for chunk in chunks:
        date_str = chunk.session_date.strftime("%Y-%m-%d") if chunk.session_date else "unknown date"
        lines.append(f"[{date_str} | {chunk.memory_type}] {chunk.content}")
    return "\n".join(lines)


def _search_memories_cosine(user_id: int, query_vec, top_k: int) -> str:
    """Legacy O(n) cosine scan — used on Postgres or when sqlite-vec unavailable."""
    chunks = (UserMemoryChunk.query
              .filter_by(user_id=user_id)
              .filter(UserMemoryChunk.embedding.isnot(None))
              .order_by(UserMemoryChunk.session_date.desc())
              .all())

    if not chunks:
        return "No long-term memories found for this user yet."

    query_np = np.array(query_vec)
    scored = []
    for chunk in chunks:
        try:
            chunk_vec = np.array(chunk.embedding)
            sim = float(np.dot(query_np, chunk_vec) / (
                np.linalg.norm(query_np) * np.linalg.norm(chunk_vec) + 1e-9
            ))
            scored.append((sim, chunk))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return "No relevant memories found."

    lines = []
    for _score, chunk in top:
        date_str = chunk.session_date.strftime("%Y-%m-%d") if chunk.session_date else "unknown date"
        lines.append(f"[{date_str} | {chunk.memory_type}] {chunk.content}")
    return "\n".join(lines)


def get_conversation_history(user_id: int, limit: int = 10) -> list:
    """Load last N messages from DB as LangChain HumanMessage / AIMessage objects."""
    from langchain_core.messages import AIMessage, HumanMessage

    messages = (ChatMessage.query
                .filter_by(user_id=user_id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(limit).all())
    messages = list(reversed(messages))

    return [
        HumanMessage(content=m.content) if m.role == 'user' else AIMessage(content=m.content)
        for m in messages
    ]
