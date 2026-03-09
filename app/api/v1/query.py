"""Vector similarity search endpoint — POST /api/v1/query."""
import uuid

import structlog
from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import DocumentChunk
from app.db.session import get_db
from app.dependencies import get_current_tenant_id
from app.schemas.query import ChunkResult, QueryRequest, QueryResponse
from app.services.embedding import EmbeddingService

logger = structlog.get_logger()

router = APIRouter(prefix="/query", tags=["query"])

_redis = Redis.from_url(settings.redis_url, decode_responses=False)
_embedding_service = EmbeddingService(redis=_redis)


@router.post(
    "",
    response_model=QueryResponse,
    status_code=200,
    summary="Search documents by semantic similarity",
    description=(
        "Embeds the query text and returns the top-k most semantically similar "
        "chunks from this tenant's documents, ordered by cosine similarity."
    ),
)
async def search_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
) -> QueryResponse:
    # Embed the query using the same model and Redis cache as ingestion
    vectors = await _embedding_service.embed_texts([request.query])
    query_vector = vectors[0]

    # Cosine distance search via HNSW index; ORDER BY distance ASC = most similar first.
    # WHERE embedding IS NOT NULL skips chunks still awaiting background processing.
    distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)

    result = await db.execute(
        select(
            DocumentChunk,
            distance_expr.label("distance"),
        )
        .where(DocumentChunk.tenant_id == tenant_id)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance_expr)
        .limit(request.top_k)
    )

    rows = result.all()

    chunk_results = [
        ChunkResult(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
            similarity=round(1.0 - float(distance), 4),
        )
        for chunk, distance in rows
    ]

    logger.info(
        "query_executed",
        tenant_id=str(tenant_id),
        top_k=request.top_k,
        results_returned=len(chunk_results),
    )

    return QueryResponse(
        query=request.query,
        total=len(chunk_results),
        results=chunk_results,
    )
