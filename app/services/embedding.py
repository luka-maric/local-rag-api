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
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _model = SentenceTransformer(settings.embedding_model)
                logger.info("Embedding model loaded (dim=%d)", settings.embedding_dim)
    return _model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True).tolist()


class EmbeddingService:
    _CACHE_TTL_SECONDS = 86400

    def __init__(self, redis=None) -> None:
        self._redis = redis

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
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

        new_vectors = await asyncio.to_thread(_encode_sync, [texts[i] for i in miss_indices])

        for idx, vector in zip(miss_indices, new_vectors):
            await self._cache_set(texts[idx], vector)
            cached[idx] = vector

        return [cached[i] for i in range(len(texts))]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    def _cache_key(self, text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    async def _cache_get(self, text: str) -> list[float] | None:
        if self._redis is None:
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
