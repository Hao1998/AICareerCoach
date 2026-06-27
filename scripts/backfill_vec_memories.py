"""
One-time backfill: copy embeddings from user_memory_chunks.embedding (JSON)
into the vec_user_memories virtual table.

Run once after the migration:
    python scripts/backfill_vec_memories.py

Safe to re-run: uses delete-then-insert so it is idempotent.
Only runs on SQLite — exits cleanly on Postgres (no vec table there).
"""
import os
import struct
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from factory import _is_sqlite, create_app

EMBEDDING_DIM = 768


def ser(floats: list) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def _decode_embedding(raw) -> list | None:
    """Decode an embedding from either JSON list or legacy PickleType bytes."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, (bytes, bytearray)):
        # Try JSON first (some SQLite drivers return bytes for JSON columns)
        import json
        try:
            return json.loads(raw.decode('utf-8'))
        except Exception:
            pass
        # Fall back to pickle (legacy PickleType storage)
        import pickle
        try:
            val = pickle.loads(raw)  # noqa: S301 — reading our own DB data
            return list(val) if hasattr(val, '__iter__') else None
        except Exception:
            return None
    return None


def backfill():
    # Use absolute path so the script works regardless of cwd.
    if not os.environ.get('DATABASE_URL'):
        db_path = os.path.join(PROJECT_ROOT, 'instance', 'career_coach.db')
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    app = create_app('development', skip_api_check=True)

    if not _is_sqlite(app):
        print("Database is not SQLite — vec_user_memories table not used here. Exiting.")
        return

    with app.app_context():
        from sqlalchemy import text
        engine = app.extensions['sqlalchemy'].engine

        with engine.connect() as conn:
            # Read raw bytes — bypass ORM JSON deserializer to handle both
            # JSON-list and legacy PickleType formats in the same column.
            rows = conn.execute(
                text("SELECT id, embedding FROM user_memory_chunks WHERE embedding IS NOT NULL")
            ).fetchall()

        print(f"Found {len(rows)} chunks with embeddings to backfill.")

        inserted = 0
        skipped = 0
        with engine.connect() as conn:
            for row_id, raw_emb in rows:
                emb = _decode_embedding(raw_emb)
                if emb is None or len(emb) != EMBEDDING_DIM:
                    skipped += 1
                    continue
                # Upsert: delete then insert so re-runs are idempotent.
                conn.execute(
                    text("DELETE FROM vec_user_memories WHERE rowid = :rid"),
                    {"rid": row_id}
                )
                conn.execute(
                    text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
                    {"rid": row_id, "emb": ser(emb)}
                )
                inserted += 1
            conn.commit()

        print(f"Backfilled {inserted} rows. Skipped {skipped} (bad shape or wrong dim).")


if __name__ == "__main__":
    backfill()
