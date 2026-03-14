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

security = HTTPBearer()


async def get_current_tenant_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> uuid.UUID:
    try:
        return decode_access_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=False)


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
