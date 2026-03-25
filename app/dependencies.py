import logging
import uuid
from functools import lru_cache

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from redis.asyncio import Redis

from app.config import settings
from app.services.auth import decode_access_token
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.extraction import ExtractionService
from app.services.ner import NERService
from app.services.ollama import OllamaService

logger = logging.getLogger(__name__)
security = HTTPBearer()


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=False)


async def _is_token_revoked(jti: str | None, redis: Redis) -> bool:
    """Fails open — Redis down means treat token as not revoked."""
    if not jti:
        return False
    try:
        return bool(await redis.get(f"blocklist:{jti}"))
    except Exception:
        logger.warning("Redis blocklist unavailable — skipping revocation check")
        return False


async def get_current_tenant_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    redis: Redis = Depends(get_redis),
) -> uuid.UUID:
    try:
        payload = decode_access_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    if await _is_token_revoked(payload.get("jti"), redis):
        raise HTTPException(status_code=401, detail="Token has been revoked.")

    return uuid.UUID(payload["sub"])


@lru_cache
def require_scope(required: str):
    async def _check(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        redis: Redis = Depends(get_redis),
    ) -> None:
        try:
            payload = decode_access_token(credentials.credentials)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        if await _is_token_revoked(payload.get("jti"), redis):
            raise HTTPException(status_code=401, detail="Token has been revoked.")

        if payload.get("scope") != required:
            raise HTTPException(status_code=403, detail=f"Scope '{required}' required.")

    return _check


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(redis=get_redis())


@lru_cache
def get_extraction_service() -> ExtractionService:
    return ExtractionService()


@lru_cache
def get_chunking_service() -> ChunkingService:
    return ChunkingService()


@lru_cache
def get_ner_service() -> NERService:
    return NERService()


@lru_cache
def get_ollama_service() -> OllamaService:
    return OllamaService(base_url=settings.ollama_base_url, model=settings.ollama_model)
