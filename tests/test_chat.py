import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies import get_current_tenant_id, get_embedding_service, get_ollama_service
from app.db.session import get_db
from app.main import create_app
from app.services.ollama import OllamaServiceError

TENANT_ID = str(uuid.uuid4())
FAKE_DIM = 384
FAKE_QUERY_VECTOR = [0.1] * FAKE_DIM
SESSION_ID = uuid.uuid4()


def _parse_sse_events(text: str) -> list:
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data: "):
            continue
        payload = block[6:]  # strip "data: " prefix
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def _make_fake_session(session_id: uuid.UUID | None = None, tenant_id: str = TENANT_ID) -> MagicMock:
    session = MagicMock()
    session.id = session_id or SESSION_ID
    session.tenant_id = uuid.UUID(tenant_id)
    return session


def _make_fake_message(role: str, content: str) -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


FAKE_DOC_ID = uuid.uuid4()


def _make_fake_chunk(text: str = "Relevant document text.", doc_id: uuid.UUID = FAKE_DOC_ID) -> MagicMock:
    chunk = MagicMock()
    chunk.chunk_text = text
    chunk.chunk_index = 0
    chunk.document_id = doc_id
    return chunk


def _make_fake_doc(doc_id: uuid.UUID = FAKE_DOC_ID, filename: str = "test.txt") -> MagicMock:
    doc = MagicMock()
    doc.id = doc_id
    doc.filename = filename
    doc.entities = {}
    return doc


def _make_execute_result(items: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = items
    result.scalars.return_value = scalars
    return result


def _make_save_db_patch(save_db: AsyncMock) -> AsyncMock:
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=save_db)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture
def mock_db():
    db = AsyncMock()
    # Simulate SQLAlchemy's column default=uuid.uuid4 firing at flush time:
    # When add(ChatSession(...)) is called, set session.id to SESSION_ID.
    def _auto_set_id(obj):
        if type(obj).__name__ == "ChatSession":
            obj.id = SESSION_ID
    db.add = MagicMock(side_effect=_auto_set_id)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def save_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


@pytest.fixture
def app_with_mock_db(mock_db):
    app = create_app()
    mock_embed = MagicMock()
    mock_ollama = MagicMock()

    async def _mock_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = _mock_get_db
    app.dependency_overrides[get_current_tenant_id] = lambda: uuid.UUID(TENANT_ID)
    app.dependency_overrides[get_embedding_service] = lambda: mock_embed
    app.dependency_overrides[get_ollama_service] = lambda: mock_ollama
    yield app, mock_db, mock_embed, mock_ollama
    app.dependency_overrides.clear()


@pytest.fixture
async def client_and_db(app_with_mock_db):
    app, mock_db, mock_embed, mock_ollama = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, mock_db, mock_embed, mock_ollama


def _chat_request(
    message: str = "What is RAG?",
    session_id: uuid.UUID | None = None,
    top_k: int = 5,
) -> dict:
    body: dict = {"message": message, "top_k": top_k}
    if session_id is not None:
        body["session_id"] = str(session_id)
    return {"json": body}


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_returns_200(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _no_tokens(_, model=None):
        return
        yield

    mock_ollama.stream = _no_tokens

    response = await client.post("/api/v1/chat", **_chat_request())

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_first_event_is_session(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _no_tokens(_, model=None):
        return
        yield

    mock_ollama.stream = _no_tokens

    response = await client.post("/api/v1/chat", **_chat_request())
    events = _parse_sse_events(response.text)

    assert len(events) >= 1
    first = events[0]
    assert isinstance(first, dict)
    assert first["type"] == "session"
    assert uuid.UUID(first["session_id"]) == SESSION_ID  # set by fixture's add() side_effect


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_last_event_is_done(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _two_tokens(_, model=None):
        yield "Hello"
        yield " world"

    mock_ollama.stream = _two_tokens

    response = await client.post("/api/v1/chat", **_chat_request())
    events = _parse_sse_events(response.text)

    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_streams_token_events(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _tokens(_, model=None):
        yield "RAG"
        yield " stands"
        yield " for"

    mock_ollama.stream = _tokens

    response = await client.post("/api/v1/chat", **_chat_request())
    events = _parse_sse_events(response.text)

    token_events = [e for e in events if isinstance(e, dict) and e.get("type") == "token"]
    assert len(token_events) == 3
    assert token_events[0]["token"] == "RAG"
    assert token_events[1]["token"] == " stands"
    assert token_events[2]["token"] == " for"


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_loads_existing_session(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _no_tokens(_, model=None):
        return
        yield

    mock_ollama.stream = _no_tokens

    response = await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))
    events = _parse_sse_events(response.text)

    session_event = events[0]
    assert session_event["type"] == "session"
    assert uuid.UUID(session_event["session_id"]) == SESSION_ID


@pytest.mark.asyncio
async def test_chat_wrong_tenant_session_returns_404(client_and_db):
    """
    If session_id belongs to a different tenant, return 404.
    Same reasoning as document management: 404 leaks nothing; 403 confirms existence.
    """
    client, mock_db, mock_embed, mock_ollama = client_and_db
    other_tenant = str(uuid.uuid4())
    fake_session = _make_fake_session(session_id=SESSION_ID, tenant_id=other_tenant)
    mock_db.get = AsyncMock(return_value=fake_session)

    response = await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_nonexistent_session_returns_404(client_and_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_db.get = AsyncMock(return_value=None)

    response = await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))

    assert response.status_code == 404


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_includes_history_in_ollama_call(mock_asl, client_and_db, save_db):
    """
    Previous conversation turns must be in the messages list passed to Ollama.
    Order: [system, ...history_chronological..., current_user_message].

    Mock returns messages in DESC order (newest first) — what real Postgres gives.
    The endpoint then reverses this to get chronological order for the LLM.
    """
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    history = [
        _make_fake_message("assistant", "Previous answer"),  # newest — returned first by DESC
        _make_fake_message("user", "Previous question"),     # oldest — returned second by DESC
    ]
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result(history),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    captured_messages = []

    async def _capture(messages, model=None):
        captured_messages.extend(messages)
        yield "answer"

    mock_ollama.stream = _capture

    await client.post(
        "/api/v1/chat",
        **_chat_request(session_id=SESSION_ID, message="Follow-up question"),
    )

    # messages = [system, user_prev, assistant_prev, user_current]
    roles = [m["role"] for m in captured_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured_messages[-1]["content"] == "Follow-up question"
    assert captured_messages[1]["content"] == "Previous question"
    assert captured_messages[2]["content"] == "Previous answer"


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_injects_chunks_into_system_prompt(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    chunk = _make_fake_chunk(text="The invoice is due in 30 days.", doc_id=FAKE_DOC_ID)
    fake_doc = _make_fake_doc(doc_id=FAKE_DOC_ID, filename="invoice.txt")
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([chunk]),
        _make_execute_result([fake_doc]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    captured_messages = []

    async def _capture(messages, model=None):
        captured_messages.extend(messages)
        yield "answer"

    mock_ollama.stream = _capture

    await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))

    system_content = captured_messages[0]["content"]
    assert "The invoice is due in 30 days." in system_content


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_no_chunks_uses_fallback_system_prompt(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    captured_messages = []

    async def _capture(messages, model=None):
        captured_messages.extend(messages)
        yield "answer"

    mock_ollama.stream = _capture

    await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))

    system_content = captured_messages[0]["content"]
    assert "No relevant documents" in system_content


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_stores_user_message_before_streaming(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _no_tokens(_, model=None):
        return
        yield

    mock_ollama.stream = _no_tokens

    await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID, message="Test question"))

    mock_db.add.assert_called()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_stores_assistant_message_after_streaming(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _tokens(_, model=None):
        yield "Hello"
        yield " world"

    mock_ollama.stream = _tokens

    await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))

    save_db.add.assert_called_once()
    save_db.commit.assert_called_once()

    saved_msg = save_db.add.call_args[0][0]
    assert saved_msg.role == "assistant"
    assert saved_msg.content == "Hello world"


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_chat_ollama_error_emits_error_event(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    fake_session = _make_fake_session(session_id=SESSION_ID)
    mock_db.get = AsyncMock(return_value=fake_session)

    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _error_stream(_, model=None):
        raise OllamaServiceError("Cannot connect to Ollama")
        yield  # makes this an async generator

    mock_ollama.stream = _error_stream

    response = await client.post("/api/v1/chat", **_chat_request(session_id=SESSION_ID))
    events = _parse_sse_events(response.text)

    error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "error"]
    assert len(error_events) == 1
    assert "Cannot connect" in error_events[0]["detail"]
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_chat_empty_message_returns_422(client_and_db):
    client, *_ = client_and_db
    response = await client.post(
        "/api/v1/chat",
        json={"message": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_top_k_zero_returns_422(client_and_db):
    client, *_ = client_and_db
    response = await client.post(
        "/api/v1/chat",
        json={"message": "Hello", "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@patch("app.api.v1.chat.AsyncSessionLocal")
async def test_ef_search_set_before_vector_query(mock_asl, client_and_db, save_db):
    client, mock_db, mock_embed, mock_ollama = client_and_db
    mock_embed.embed_one = AsyncMock(return_value=FAKE_QUERY_VECTOR)
    mock_db.execute = AsyncMock(side_effect=[
        _make_execute_result([]),
        MagicMock(),
        _make_execute_result([]),
    ])
    mock_asl.return_value = _make_save_db_patch(save_db)

    async def _no_tokens(_, model=None):
        return
        yield

    mock_ollama.stream = _no_tokens

    await client.post("/api/v1/chat", **_chat_request())

    sql_calls = [str(c.args[0]) for c in mock_db.execute.call_args_list if c.args]
    assert any("ef_search" in sql for sql in sql_calls)
