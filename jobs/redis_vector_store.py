"""
Redis job vector store (VECTOR_BACKEND=redis experiment).

One Redis hash per active job, indexed by a RediSearch HNSW vector index:

    job:{id}  embedding   768 x float32, raw bytes
              source      TAG      (e.g. FT.SEARCH ... "@source:{adzuna}=>[KNN ...]")
              updated_at  NUMERIC  (epoch seconds of embedding_updated_at)

    FT.CREATE idx:jobs ON HASH PREFIX 1 job:
      SCHEMA source TAG updated_at NUMERIC
             embedding VECTOR HNSW 10 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE
                                      M 16 EF_CONSTRUCTION 200

Inactive jobs are not stored (DEL job:{id}), the same idea as the partial
pgvector index, so a search needs no is_active filter. The DB stays the source
of truth: rebuild() can always repopulate this from job_postings.embedding.

`{index}:built_at` marks an index that rebuild() fully loaded. An index that
exists without it (the relay recreated it after a reset, or a rebuild is
still streaming) holds only some jobs, so dense search won't trust it.
"""
import logging
import time
from typing import Iterable

import numpy as np
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

DIM = 768
DEFAULT_INDEX = "idx:jobs"
DEFAULT_PREFIX = "job:"


def encode_vector(vec) -> bytes:
    """Vector → raw float32 bytes, the format RediSearch VECTOR fields expect."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def distance_to_similarity(dist: float) -> float:
    """RediSearch COSINE distance is 1 - cosine; return cosine clamped to 0..1."""
    return max(0.0, min(1.0, 1.0 - float(dist)))


class RedisJobVectorStore:
    def __init__(self, client, index_name: str = DEFAULT_INDEX, prefix: str = DEFAULT_PREFIX):
        self.client = client
        self.index_name = index_name
        self.prefix = prefix

    def key(self, job_id: int) -> str:
        return f"{self.prefix}{job_id}"

    @property
    def built_marker(self) -> str:
        return f"{self.index_name}:built_at"

    def is_built(self) -> bool:
        """True once rebuild() has fully loaded this index."""
        return bool(self.client.exists(self.built_marker))

    def _ft(self):
        return self.client.ft(self.index_name)

    def ensure_index(self) -> None:
        """Create the index if it doesn't exist yet. Safe to call repeatedly."""
        schema = (
            TagField("source"),
            NumericField("updated_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": DIM,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        try:
            self._ft().create_index(schema, definition=definition)
            logger.info("Created Redis vector index %s", self.index_name)
        except ResponseError as e:
            if "already exists" not in str(e).lower():
                raise

    def _mapping(self, job) -> dict:
        updated = job.embedding_updated_at
        return {
            "embedding": encode_vector(job.embedding),
            "source": job.source or "",
            "updated_at": int(updated.timestamp()) if updated else 0,
        }

    def upsert_job(self, job) -> None:
        """Write the job's hash. HSET with the full mapping, so repeating it is harmless."""
        self.client.hset(self.key(job.id), mapping=self._mapping(job))

    def remove_job(self, job_id: int) -> None:
        self.client.delete(self.key(job_id))

    def search(self, query_vec, k: int) -> list[tuple[int, float]]:
        """KNN search → [(job_id, similarity)], best first."""
        if k <= 0:
            return []
        query = (
            Query(f"*=>[KNN {int(k)} @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("dist")
            .paging(0, int(k))
            .dialect(2)
        )
        result = self._ft().search(query, query_params={"vec": encode_vector(query_vec)})
        hits = []
        for doc in result.docs:
            doc_id = doc.id.decode() if isinstance(doc.id, bytes) else doc.id
            dist = doc.dist.decode() if isinstance(doc.dist, bytes) else doc.dist
            hits.append((int(doc_id[len(self.prefix):]), distance_to_similarity(dist)))
        return hits

    def count(self) -> int:
        info = self._ft().info()
        return int(info["num_docs"])

    def drop(self) -> None:
        """Drop the index and every hash it covers. No-op if the index is missing."""
        self.client.delete(self.built_marker)
        try:
            self._ft().dropindex(delete_documents=True)
        except ResponseError as e:
            msg = str(e).lower()
            if "unknown index" not in msg and "no such index" not in msg:
                raise

    def rebuild(self, jobs: Iterable) -> int:
        """Drop everything and reload from `jobs`. Jobs without an embedding are skipped."""
        self.drop()
        self.ensure_index()
        pipe = self.client.pipeline(transaction=False)
        count = 0
        for job in jobs:
            if job.embedding is None:
                continue
            pipe.hset(self.key(job.id), mapping=self._mapping(job))
            count += 1
            if count % 500 == 0:
                pipe.execute()
        pipe.execute()
        self.client.set(self.built_marker, int(time.time()))
        logger.info("Rebuilt Redis vector index %s with %d jobs", self.index_name, count)
        return count


_store: RedisJobVectorStore | None = None


def get_store() -> RedisJobVectorStore:
    """The default store (idx:jobs / job:) on REDIS_VECTOR_URL."""
    global _store
    if _store is None:
        from services.redis_client import get_redis_vectors
        _store = RedisJobVectorStore(get_redis_vectors())
    return _store
