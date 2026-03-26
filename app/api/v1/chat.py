import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ChatMessage, ChatSession, Document, DocumentChunk
from app.db.session import AsyncSessionLocal, get_db
from app.dependencies import get_current_tenant_id, get_embedding_service, get_ollama_service
from app.schemas.chat import ChatRequest
from app.services.embedding import EmbeddingService
from app.services.ollama import OllamaService, OllamaServiceError

logger = structlog.get_logger()

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_HISTORY_MESSAGES = 10


def _sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n"


def _build_system_prompt(chunks_with_filenames: list[tuple]) -> str:
    if not chunks_with_filenames:
        return (
            "You are a helpful assistant. No relevant documents were found for "
            "this query. Let the user know and offer to help in another way."
        )

    context_blocks = "\n\n".join(
        f"[{filename}]\n{chunk.chunk_text}"
        for chunk, filename in chunks_with_filenames
    )

    return (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the information provided in the context below. If the answer cannot "
        "be found in the context, say: \"I don't have information about that "
        "in the uploaded documents.\"\n\n"
        f"Context:\n{context_blocks}"
    )


@router.post(
    "",
    summary="Send a chat message (RAG pipeline with streaming response)",
    description=(
        "Embeds the user message, retrieves relevant document chunks, builds a "
        "RAG prompt, and streams the LLM's response as Server-Sent Events. "
        "Pass the session_id from the first response on subsequent requests to "
        "maintain conversation history."
    ),
    response_class=StreamingResponse,
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> StreamingResponse:
    # SSE event types: session | sources | token | error | [DONE]
    if request.session_id is not None:
        session = await db.get(ChatSession, request.session_id)
        if session is None or session.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = ChatSession(tenant_id=tenant_id)
        db.add(session)
        await db.flush()

    session_id = session.id

    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    # DESC fetch → reverse to chronological order for the LLM
    history: list[ChatMessage] = list(reversed(history_result.scalars().all()))

    query_vector = await embedding_service.embed_one(request.message)

    await db.execute(text(f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}"))

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding.is_not(None),
        )
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(request.top_k)
    )
    chunks = chunks_result.scalars().all()

    sources_info: list[dict] = []
    chunks_with_filenames: list[tuple] = []

    if chunks:
        doc_ids = list({chunk.document_id for chunk in chunks})
        docs_result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
        doc_map = {
            doc.id: {"filename": doc.filename, "entities": doc.entities or {}}
            for doc in docs_result.scalars().all()
        }
        sources_info = [
            {
                "filename": doc_map.get(chunk.document_id, {}).get("filename", "unknown"),
                "chunk_index": chunk.chunk_index,
                "entities": doc_map.get(chunk.document_id, {}).get("entities", {}),
            }
            for chunk in chunks
        ]
        chunks_with_filenames = [
            (chunk, doc_map.get(chunk.document_id, {}).get("filename", "unknown"))
            for chunk in chunks
        ]

    system_prompt = _build_system_prompt(chunks_with_filenames)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    # Commit user message before streaming — HTTPException can't be raised after bytes are sent
    db.add(ChatMessage(session_id=session_id, role="user", content=request.message))
    await db.commit()

    logger.info(
        "chat_request",
        session_id=str(session_id),
        tenant_id=str(tenant_id),
        history_messages=len(history),
        chunks_retrieved=len(chunks),
    )

    async def event_generator() -> AsyncIterator[str]:
        yield _sse({"type": "session", "session_id": str(session_id)})
        yield _sse({"type": "sources", "sources": sources_info})

        tokens: list[str] = []
        try:
            async for token in ollama_service.stream(messages, model=request.model):
                tokens.append(token)
                yield _sse({"type": "token", "token": token})
        except OllamaServiceError as exc:
            logger.error("chat_ollama_error", session_id=str(session_id), error=str(exc))
            yield _sse({"type": "error", "detail": str(exc)})
            yield _sse("[DONE]")
            return

        full_reply = "".join(tokens)

        # Save assistant reply in a fresh session — request db may be in undefined state here
        async with AsyncSessionLocal() as save_db:
            save_db.add(ChatMessage(session_id=session_id, role="assistant", content=full_reply))
            await save_db.commit()

        logger.info("chat_complete", session_id=str(session_id), reply_chars=len(full_reply))
        yield _sse("[DONE]")

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
