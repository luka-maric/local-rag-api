import asyncio
import hashlib
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.db.session import AsyncSessionLocal, get_db
from app.metrics import document_processing_seconds, documents_processed_total, documents_uploaded_total
from app.dependencies import get_current_tenant_id
from app.schemas.documents import DocumentListResponse, DocumentResponse, UploadResponse
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.extraction import ExtractionService
from app.services.ner import NERService

logger = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])

_redis = Redis.from_url(settings.redis_url, decode_responses=False)
_extraction_service = ExtractionService()
_chunking_service = ChunkingService()
_embedding_service = EmbeddingService(redis=_redis)
_ner_service = NERService()

# File extension → magic bytes. Extensions without an entry are validated as UTF-8.
_MAGIC_BYTES: dict[str, bytes] = {
    ".pdf":  b"%PDF",
    ".png":  b"\x89PNG",
    ".jpg":  b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
}


def _validate_file_magic(file_bytes: bytes, filename: str) -> None:
    ext = Path(filename).suffix.lower()
    expected_magic = _MAGIC_BYTES.get(ext)

    if expected_magic is None:
        # .txt — validate UTF-8 decodability instead of magic bytes
        try:
            file_bytes[:1024].decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File content is not valid UTF-8 text.",
            )
        return

    if file_bytes[:len(expected_magic)] != expected_magic:
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match extension '{ext}'. Possible file type mismatch.",
        )


async def process_document(
    document_id: uuid.UUID,
    file_bytes: bytes,
    filename: str,
    tenant_id: uuid.UUID,
) -> None:
    async with AsyncSessionLocal() as db:
        try:
            with document_processing_seconds.time():
                text: str = await asyncio.to_thread(
                    _extraction_service.extract, file_bytes, filename
                )

                entities = await _ner_service.extract_entities(text)
                chunks = _chunking_service.chunk(text)
                vectors = await _embedding_service.embed_texts([c.text for c in chunks])

                doc = await db.get(Document, document_id)

                db_chunks = [
                    DocumentChunk(
                        tenant_id=tenant_id,
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        chunk_text=chunk.text,
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]
                db.add_all(db_chunks)

                doc.chunk_count = len(chunks)
                doc.entities = entities or None

                await db.commit()

                documents_processed_total.labels(status="success").inc()
                logger.info(
                    "document_processed",
                    document_id=str(document_id),
                    chunks=len(chunks),
                )

        except Exception as exc:
            await db.rollback()
            documents_processed_total.labels(status="failed").inc()
            logger.error(
                "document_processing_failed",
                document_id=str(document_id),
                error=str(exc),
            )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a document for ingestion",
    description=(
        "Accepts a file upload, hashes the content for deduplication, "
        "creates a document record, and queues extraction + chunking + embedding "
        "as a background task. Returns 202 immediately."
    ),
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, TXT, PNG, JPG, or JPEG file"),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
) -> UploadResponse:
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_mb} MB.",
        )

    _validate_file_magic(file_bytes, file.filename or "")

    content_hash = hashlib.sha256(file_bytes).hexdigest()

    result = await db.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.content_hash == content_hash,
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        return JSONResponse(
            status_code=200,
            content=UploadResponse(
                document_id=existing.id,
                filename=existing.filename,
                status="already_exists",
                message="This file has already been uploaded and processed for your tenant.",
            ).model_dump(mode="json"),
        )

    doc = Document(
        tenant_id=tenant_id,
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        content_hash=content_hash,
    )
    db.add(doc)

    await db.commit()

    background_tasks.add_task(
        process_document,
        doc.id,
        file_bytes,
        file.filename or "unknown",
        tenant_id,
    )

    logger.info(
        "document_upload_accepted",
        document_id=str(doc.id),
        filename=file.filename,
        tenant_id=str(tenant_id),
        size_bytes=len(file_bytes),
    )

    documents_uploaded_total.inc()

    return UploadResponse(
        document_id=doc.id,
        filename=file.filename or "unknown",
        status="processing",
        message="Document accepted and queued for processing.",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all documents for this tenant",
    description=(
        "Returns a paginated list of documents ordered by creation date descending. "
        "Use chunk_count / status to check whether background processing has completed."
    ),
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=10, ge=1, le=100, description="Results per page (1-100)"),
) -> DocumentListResponse:
    count_result = await db.execute(
        select(func.count()).select_from(Document).where(Document.tenant_id == tenant_id)
    )
    total = count_result.scalar_one()

    result = await db.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    docs = result.scalars().all()

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=[
            DocumentResponse(
                document_id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                chunk_count=doc.chunk_count,
                status="processing" if doc.chunk_count == 0 else "ready",
                created_at=doc.created_at,
                entities=doc.entities,
            )
            for doc in docs
        ],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document metadata and processing status",
    description=(
        "Fetches a single document by ID. Poll this endpoint after uploading "
        "to check when status transitions from 'processing' to 'ready'."
    ),
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
) -> DocumentResponse:
    doc = await db.get(Document, document_id)

    # 404 (not 403) for wrong-tenant access — avoids leaking document existence
    # to tenants that don't own it.
    if doc is None or doc.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentResponse(
        document_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        chunk_count=doc.chunk_count,
        status="processing" if doc.chunk_count == 0 else "ready",
        created_at=doc.created_at,
        entities=doc.entities,
    )


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document and all its chunks",
    description=(
        "Deletes the document and cascades to all associated document_chunks. "
        "Returns 204 No Content on success."
    ),
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
) -> None:
    doc = await db.get(Document, document_id)

    if doc is None or doc.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    await db.delete(doc)
    await db.commit()

    logger.info(
        "document_deleted",
        document_id=str(document_id),
        tenant_id=str(tenant_id),
    )
