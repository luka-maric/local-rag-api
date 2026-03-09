import uuid
from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str   # "processing" | "already_exists"
    message: str


class DocumentResponse(BaseModel):
    """status is derived from chunk_count: "processing" (0) or "ready" (> 0)."""

    document_id: uuid.UUID
    filename: str
    content_type: str
    chunk_count: int
    status: str       # "processing" | "ready"
    created_at: datetime
    entities: dict[str, list[str]] | None = None  # None for documents without NER


class DocumentListResponse(BaseModel):
    """total is the cross-page count; use ceil(total / page_size) for total pages."""

    total: int
    page: int
    page_size: int
    results: list[DocumentResponse]
