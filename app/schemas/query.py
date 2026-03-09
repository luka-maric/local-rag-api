import uuid

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return (1-50)")


class ChunkResult(BaseModel):
    document_id: uuid.UUID
    chunk_index: int
    chunk_text: str
    similarity: float  # cosine similarity in [0.0, 1.0]; scores below ~0.3 are typically not meaningful


class QueryResponse(BaseModel):
    """results ordered by similarity descending — best match first."""

    query: str
    total: int
    results: list[ChunkResult]
