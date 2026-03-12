"""Embedding service — converts text to 384-dim vectors using sentence-transformers."""
import asyncio
import hashlib
import logging
import threading

import numpy as np

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.metrics import embedding_cache_hits_total, embedding_cache_misses_total

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """
    Return the global SentenceTransformer, loading it on first call.

    Thread-safe via double-checked locking:
      - First check (no lock): fast path when model is already loaded.
      - Second check (inside lock): prevents two threads that both saw None
        from each creating a separate model instance.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # second check — another thread may have loaded it
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _model = SentenceTransformer(settings.embedding_model)
                logger.info(
                    "Embedding model loaded (dim=%d)", settings.embedding_dim
                )
    return _model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous encode — runs inside asyncio.to_thread. Returns list[list[float]]."""
    model = _get_model()
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,  # suppress tqdm in production logs
        convert_to_numpy=True,   # explicit — ensure numpy output before .tolist()
    ).tolist()


class EmbeddingService:
    """Converts lists of text to 384-float vectors with Redis caching."""

    _CACHE_TTL_SECONDS = 86400  # 24 hours

    def __init__(self, redis=None) -> None:
        self._redis = redis  # redis.asyncio.Redis instance or None; None disables caching

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns one vector per input, in original order.

        Cache-aware batch flow:
          1. Check Redis for each text (SHA-256 hash as key).
          2. Collect indices of cache misses.
          3. Send ALL misses to the model in ONE call (batch embedding).
          4. Cache the new embeddings.
          5. Reconstruct results in original order.
        """
        if not texts:
            return []

        cached: dict[int, list[float]] = {}
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            cached_vec = await self._cache_get(text)
            if cached_vec is not None:
                cached[i] = cached_vec
            else:
                miss_indices.append(i)

        if not miss_indices:
            return [cached[i] for i in range(len(texts))]

        miss_texts = [texts[i] for i in miss_indices]
        new_vectors = await asyncio.to_thread(_encode_sync, miss_texts)

        for idx, vector in zip(miss_indices, new_vectors):
            await self._cache_set(texts[idx], vector)
            cached[idx] = vector

        return [cached[i] for i in range(len(texts))]

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_texts([text])
        return results[0]

    def _cache_key(self, text: str) -> str:
        """SHA-256 of text with 'embed:' prefix to avoid collisions with other Redis keys."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:{digest}"

    async def _cache_get(self, text: str) -> list[float] | None:
        """
        Return cached embedding vector or None.

        Returns None when:
        - Redis is not configured (self._redis is None)
        - Cache miss (key does not exist)
        - Redis connection error (graceful degradation)
        """
        if self._redis is None:
            # Skip counting — only increment hit/miss counters when Redis is active
            # so the hit-rate metric reflects real cache outcomes, not no-op calls.
            return None
        try:
            raw = await self._redis.get(self._cache_key(text))
            if raw is None:
                embedding_cache_misses_total.inc()
                return None
            embedding_cache_hits_total.inc()
            return np.frombuffer(raw, dtype=np.float32).tolist()
        except Exception:
            logger.warning("Redis cache read failed — proceeding without cache")
            return None

    async def _cache_set(self, text: str, vector: list[float]) -> None:
        """Store embedding in Redis with TTL. Silently ignores errors."""
        if self._redis is None:
            return
        try:
            await self._redis.set(
                self._cache_key(text),
                np.array(vector, dtype=np.float32).tobytes(),
                ex=self._CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning("Redis cache write failed — embedding will not be cached")
