import json
import time
import threading
from typing import Any
import numpy as np
from langchain_core.caches import BaseCache


class SemanticCache(BaseCache):
    def __init__(self, embedding_model=None, score_threshold: float = 0.90,
                 max_size: int = 500, ttl_seconds: int = 3600,
                 bypass_prefixes: list[Any] | None = None):
        self._embed = embedding_model  # can be None; resolved lazily via _get_embed()
        self._threshold = score_threshold
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: list[tuple] = []  # list of (vector, llm_string, response, expires_at)
        self._lock = threading.Lock()
        self._bypass_prefixes = tuple(bypass_prefixes or [])

    def _get_embed(self):
        if self._embed is None:
            from job_utils import get_embeddings
            self._embed = get_embeddings()
        return self._embed

    @staticmethod
    def _extract_content(prompt: str) -> str:
        """LangChain serializes messages as JSON before passing to the cache.
        Extract the actual text content so prefix checks work correctly."""
        try:
            messages = json.loads(prompt)
            if isinstance(messages, list) and messages:
                return messages[0].get("kwargs", {}).get("content", prompt)
        except (json.JSONDecodeError, AttributeError, IndexError):
            pass
        return prompt

    def _should_bypass(self, prompt: str) -> bool:
        if not self._bypass_prefixes:
            return False
        content = self._extract_content(prompt)
        return content.lstrip().startswith(self._bypass_prefixes)

    @staticmethod
    def _cosine(a, b) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _purge_expired(self):
        """Remove expired entries. Must be called while holding self._lock."""
        now = time.time()
        self._store = [e for e in self._store if e[3] > now]

    def lookup(self, prompt: str, llm_string: str):
        if self._should_bypass(prompt):
            return None
        query_vec = self._get_embed().embed_query(prompt)
        with self._lock:
            self._purge_expired()
            for stored_vec, stored_llm, response, _ in self._store:
                if stored_llm != llm_string:
                    continue
                if self._cosine(query_vec, stored_vec) >= self._threshold:
                    return response
        return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        if self._should_bypass(prompt):
            return
        vec = self._get_embed().embed_query(prompt)
        expires_at = time.time() + self._ttl
        with self._lock:
            self._purge_expired()
            if len(self._store) >= self._max_size:
                evict_count = max(1, self._max_size // 10)
                self._store = self._store[evict_count:]
            self._store.append((vec, llm_string, return_val, expires_at))

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._store = []
