"""Chat endpoint — POST /api/v1/chat (RAG pipeline with SSE streaming)."""
import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ChatMessage, ChatSession, Document, DocumentChunk
from app.db.session import AsyncSessionLocal, get_db
from app.dependencies import get_current_tenant_id
from app.schemas.chat import ChatRequest
from app.services.embedding import EmbeddingService
from app.services.ollama import OllamaService, OllamaServiceError

logger = structlog.get_logger()

router = APIRouter(prefix="/chat", tags=["chat"])

_redis = Redis.from_url(settings.redis_url, decode_responses=False)
_embedding_service = EmbeddingService(redis=_redis)
_ollama_service = OllamaService(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
)

MAX_HISTORY_MESSAGES = 10


def _sse(data: dict | str) -> str:
    """Format a single SSE event."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n"


def _build_system_prompt(chunks_with_filenames: list[tuple]) -> str:
    """Build the RAG system prompt using actual source filenames as context labels."""
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
) -> StreamingResponse:
    """
    RAG chat with SSE streaming. Event types:
      {"type": "session",  "session_id": "<uuid>"}                    — first event
      {"type": "sources",  "sources":    [{filename, chunk_index, entities}]}  — retrieved chunks
      {"type": "token",    "token":      "<str>"}                     — one per generated token
      {"type": "error",    "detail":     "<str>"}                     — if Ollama fails
      [DONE]                                                           — last event
    """
    # 1. Session: get or create
    if request.session_id is not None:
        session = await db.get(ChatSession, request.session_id)
        if session is None or session.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = ChatSession(tenant_id=tenant_id)
        db.add(session)
        await db.flush()  # get session.id before commit

    session_id = session.id

    # 2. History: last MAX_HISTORY_MESSAGES in chronological order
    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    # DESC query → newest first; reverse to chronological order for the LLM
    history: list[ChatMessage] = list(reversed(history_result.scalars().all()))

    # 3. Retrieval: embed query, find similar chunks
    query_vector = await _embedding_service.embed_one(request.message)

    await db.execute(text("SET hnsw.ef_search = :val"), {"val": settings.hnsw_ef_search})

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

    # 4. Look up source document filenames and NER entities
    sources_info: list[dict] = []
    chunks_with_filenames: list[tuple] = []

    if chunks:
        doc_ids = list({chunk.document_id for chunk in chunks})
        docs_result = await db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
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

    # 5. Build system prompt with filename-labelled context
    system_prompt = _build_system_prompt(chunks_with_filenames)

    # 6. Assemble message list: [system, ...history, current user message]
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    # 7. Persist user message and commit BEFORE streaming starts.
    # Once StreamingResponse begins sending bytes the status code is committed —
    # HTTPException can't be raised after that point.
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    await db.commit()

    logger.info(
        "chat_request",
        session_id=str(session_id),
        tenant_id=str(tenant_id),
        history_messages=len(history),
        chunks_retrieved=len(chunks),
    )

    # 8 & 9. Stream tokens, then persist assistant reply in a fresh session.
    async def event_generator() -> AsyncIterator[str]:
        yield _sse({"type": "session", "session_id": str(session_id)})
        yield _sse({"type": "sources", "sources": sources_info})

        tokens: list[str] = []
        try:
            async for token in _ollama_service.stream(messages):
                tokens.append(token)
                yield _sse({"type": "token", "token": token})
        except OllamaServiceError as exc:
            logger.error(
                "chat_ollama_error",
                session_id=str(session_id),
                error=str(exc),
            )
            yield _sse({"type": "error", "detail": str(exc)})
            yield _sse("[DONE]")
            return

        # Save assistant reply in its own session — request db is committed and
        # may be in an undefined state by the time the generator runs.
        full_reply = "".join(tokens)
        async with AsyncSessionLocal() as save_db:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_reply,
            )
            save_db.add(assistant_msg)
            await save_db.commit()

        logger.info(
            "chat_complete",
            session_id=str(session_id),
            reply_chars=len(full_reply),
        )

        yield _sse("[DONE]")

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer SSE
        },
    )
