import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """session_id is None for new sessions; pass the returned session_id on subsequent messages."""

    session_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
