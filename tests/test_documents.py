import hashlib
import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.documents import process_document
from app.dependencies import get_current_tenant_id
from app.db.session import get_db
from app.main import create_app
from app.services.chunking import TextChunk


TENANT_ID = str(uuid.uuid4())
FAKE_FILE_BYTES = b"Hello, this is a test document."
FAKE_FILENAME = "test.txt"
FAKE_CONTENT_HASH = hashlib.sha256(FAKE_FILE_BYTES).hexdigest()


def _make_file_upload(content: bytes = FAKE_FILE_BYTES, filename: str = FAKE_FILENAME):
    return {
        "files": {"file": (filename, io.BytesIO(content), "text/plain")},
    }


@pytest.fixture
def mock_db_session():
    session = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)

    def _assign_id(obj):
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()

    session.add = MagicMock(side_effect=_assign_id)
    session.flush = AsyncMock()

    return session


@pytest.fixture
def app_with_mock_db(mock_db_session):
    app = create_app()

    async def _mock_get_db():
        yield mock_db_session

    app.dependency_overrides[get_db] = _mock_get_db
    app.dependency_overrides[get_current_tenant_id] = lambda: uuid.UUID(TENANT_ID)
    yield app, mock_db_session
    app.dependency_overrides.clear()


@pytest.fixture
async def client(app_with_mock_db):
    app, _ = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client_and_db(app_with_mock_db):
    app, mock_db = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, mock_db


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_new_document_returns_202(mock_process, client):
    mock_process.return_value = None

    response = await client.post("/api/v1/documents/upload", **_make_file_upload())

    assert response.status_code == 202


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_new_document_response_body(mock_process, client):
    mock_process.return_value = None

    response = await client.post("/api/v1/documents/upload", **_make_file_upload())

    body = response.json()
    assert body["status"] == "processing"
    assert body["filename"] == FAKE_FILENAME
    # document_id is a UUID string — just check it parses
    uuid.UUID(body["document_id"])


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_new_document_commits_before_returning(mock_process, client_and_db):
    mock_process.return_value = None
    client, mock_db = client_and_db

    await client.post("/api/v1/documents/upload", **_make_file_upload())

    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_background_task_registered_for_new_document(mock_process, client):
    mock_process.return_value = None

    await client.post("/api/v1/documents/upload", **_make_file_upload())

    mock_process.assert_called_once()
    call_kwargs = mock_process.call_args
    assert call_kwargs.args[1] == FAKE_FILE_BYTES
    assert call_kwargs.args[2] == FAKE_FILENAME


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_duplicate_document_returns_200(mock_process, client_and_db):
    mock_process.return_value = None
    client, mock_db = client_and_db

    existing = MagicMock()
    existing.id = uuid.uuid4()
    existing.filename = FAKE_FILENAME
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    mock_db.execute = AsyncMock(return_value=result)

    response = await client.post("/api/v1/documents/upload", **_make_file_upload())

    assert response.status_code == 200


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_duplicate_response_body(mock_process, client_and_db):
    mock_process.return_value = None
    client, mock_db = client_and_db

    existing_id = uuid.uuid4()
    existing = MagicMock()
    existing.id = existing_id
    existing.filename = FAKE_FILENAME
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    mock_db.execute = AsyncMock(return_value=result)

    response = await client.post("/api/v1/documents/upload", **_make_file_upload())

    body = response.json()
    assert body["status"] == "already_exists"
    assert uuid.UUID(body["document_id"]) == existing_id


@pytest.mark.asyncio
@patch("app.api.v1.documents.process_document")
async def test_duplicate_does_not_queue_background_task(mock_process, client_and_db):
    mock_process.return_value = None
    client, mock_db = client_and_db

    existing = MagicMock()
    existing.id = uuid.uuid4()
    existing.filename = FAKE_FILENAME
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    mock_db.execute = AsyncMock(return_value=result)

    await client.post("/api/v1/documents/upload", **_make_file_upload())

    mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_empty_file_returns_400(client):
    response = await client.post(
        "/api/v1/documents/upload",
        **_make_file_upload(content=b""),
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_oversized_file_returns_413(client):
    from unittest.mock import patch
    oversized = b"x" * (51 * 1024 * 1024)  # 51 MB — over the 50 MB default
    with patch("app.api.v1.documents.settings") as mock_settings:
        mock_settings.max_upload_bytes = 50 * 1024 * 1024
        response = await client.post(
            "/api/v1/documents/upload",
            **_make_file_upload(content=oversized),
        )
    assert response.status_code == 413
    assert "50 MB" in response.json()["detail"]


from app.api.v1.documents import _validate_file_magic


def test_validate_file_magic_accepts_valid_pdf():
    valid_pdf = b"%PDF-1.4 fake content"
    _validate_file_magic(valid_pdf, "document.pdf")  # must not raise


def test_validate_file_magic_accepts_valid_png():
    valid_png = b"\x89PNG\r\n\x1a\n fake png content"
    _validate_file_magic(valid_png, "image.png")  # must not raise


def test_validate_file_magic_rejects_pdf_with_wrong_magic():
    from fastapi import HTTPException
    png_bytes = b"\x89PNG\r\n\x1a\n fake png content"
    with pytest.raises(HTTPException) as exc_info:
        _validate_file_magic(png_bytes, "not_really.pdf")
    assert exc_info.value.status_code == 400
    assert ".pdf" in exc_info.value.detail


def test_validate_file_magic_rejects_non_utf8_txt():
    from fastapi import HTTPException
    binary_bytes = b"\xff\xfe\x00invalid binary data"
    with pytest.raises(HTTPException) as exc_info:
        _validate_file_magic(binary_bytes, "notes.txt")
    assert exc_info.value.status_code == 400


def _make_fake_chunks(n: int = 2) -> list[TextChunk]:
    return [
        TextChunk(chunk_index=i, char_start=i * 10, char_end=(i + 1) * 10, text=f"chunk {i}")
        for i in range(n)
    ]


FAKE_VECTORS = [[0.1] * 384, [0.2] * 384]


@pytest.fixture
def mock_async_session_local():
    mock_session = AsyncMock()

    fake_doc = MagicMock()
    mock_session.get = AsyncMock(return_value=fake_doc)
    mock_session.add_all = MagicMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    return ctx, mock_session, fake_doc


@pytest.mark.asyncio
async def test_process_document_happy_path(mock_async_session_local):
    ctx, mock_session, fake_doc = mock_async_session_local
    fake_chunks = _make_fake_chunks(2)

    mock_ext = MagicMock()
    mock_ext.extract.return_value = "extracted text"
    mock_ner = MagicMock()
    mock_ner.extract_entities = AsyncMock(return_value={"PERSON": ["Test Person"]})
    mock_chunk = MagicMock()
    mock_chunk.chunk.return_value = fake_chunks
    mock_embed = MagicMock()
    mock_embed.embed_texts = AsyncMock(return_value=FAKE_VECTORS)

    with patch("app.api.v1.documents.AsyncSessionLocal", return_value=ctx):
        await process_document(
            uuid.uuid4(), FAKE_FILE_BYTES, FAKE_FILENAME, uuid.uuid4(),
            mock_ext, mock_chunk, mock_embed, mock_ner,
        )

    mock_session.add_all.assert_called_once()
    stored_chunks = mock_session.add_all.call_args.args[0]
    assert len(stored_chunks) == 2
    assert fake_doc.chunk_count == 2
    assert fake_doc.entities is not None
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_rollback_on_error(mock_async_session_local):
    ctx, mock_session, _ = mock_async_session_local

    mock_ext = MagicMock()
    mock_ext.extract.side_effect = RuntimeError("extraction failed")

    with patch("app.api.v1.documents.AsyncSessionLocal", return_value=ctx):
        await process_document(
            uuid.uuid4(), FAKE_FILE_BYTES, FAKE_FILENAME, uuid.uuid4(),
            mock_ext, MagicMock(), MagicMock(), MagicMock(),
        )

    mock_session.rollback.assert_called_once()
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_does_not_raise_on_any_exception(mock_async_session_local):
    ctx, mock_session, _ = mock_async_session_local

    mock_ext = MagicMock()
    mock_ext.extract.side_effect = MemoryError("out of memory")

    with patch("app.api.v1.documents.AsyncSessionLocal", return_value=ctx):
        await process_document(
            uuid.uuid4(), FAKE_FILE_BYTES, FAKE_FILENAME, uuid.uuid4(),
            mock_ext, MagicMock(), MagicMock(), MagicMock(),
        )


FAKE_CREATED_AT = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)


def _make_fake_doc(
    document_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
    chunk_count: int = 5,
) -> MagicMock:
    doc = MagicMock()
    doc.id = document_id or uuid.uuid4()
    doc.tenant_id = uuid.UUID(tenant_id or TENANT_ID)
    doc.filename = "test.txt"
    doc.content_type = "text/plain"
    doc.chunk_count = chunk_count
    doc.created_at = FAKE_CREATED_AT
    doc.entities = {"PERSON": ["John Smith"], "ORG": ["Acme Corp"]}
    return doc


def _make_count_result(n: int) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = n
    return result


def _make_list_result(docs: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = docs
    result.scalars.return_value = scalars
    return result


@pytest.mark.asyncio
async def test_get_document_returns_200(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc()
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.get(
        f"/api/v1/documents/{doc.id}",
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_document_response_shape(client_and_db):
    client, mock_db = client_and_db
    doc_id = uuid.uuid4()
    doc = _make_fake_doc(document_id=doc_id, chunk_count=3)
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.get(
        f"/api/v1/documents/{doc_id}",
    )
    body = response.json()

    assert uuid.UUID(body["document_id"]) == doc_id
    assert body["filename"] == "test.txt"
    assert body["content_type"] == "text/plain"
    assert body["chunk_count"] == 3
    assert body["status"] == "ready"
    assert "created_at" in body
    assert "entities" in body


@pytest.mark.asyncio
async def test_get_document_status_processing(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc(chunk_count=0)
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.get(
        f"/api/v1/documents/{doc.id}",
    )
    assert response.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_get_document_status_ready(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc(chunk_count=7)
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.get(
        f"/api/v1/documents/{doc.id}",
    )
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_get_document_not_found_returns_404(client_and_db):
    client, mock_db = client_and_db
    mock_db.get = AsyncMock(return_value=None)

    response = await client.get(
        f"/api/v1/documents/{uuid.uuid4()}",
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_document_wrong_tenant_returns_404(client_and_db):
    """Document exists but belongs to a different tenant → 404, not 403."""
    client, mock_db = client_and_db
    other_tenant_id = str(uuid.uuid4())
    doc = _make_fake_doc(tenant_id=other_tenant_id)
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.get(
        f"/api/v1/documents/{doc.id}",
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_returns_204(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc()
    mock_db.get = AsyncMock(return_value=doc)

    response = await client.delete(
        f"/api/v1/documents/{doc.id}",
    )
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_document_not_found_returns_404(client_and_db):
    client, mock_db = client_and_db
    mock_db.get = AsyncMock(return_value=None)

    response = await client.delete(
        f"/api/v1/documents/{uuid.uuid4()}",
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_calls_db_delete_and_commit(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc()
    mock_db.get = AsyncMock(return_value=doc)

    await client.delete(
        f"/api/v1/documents/{doc.id}",
    )

    mock_db.delete.assert_called_once_with(doc)
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_list_documents_returns_200(client_and_db):
    client, mock_db = client_and_db
    mock_db.execute = AsyncMock(side_effect=[
        _make_count_result(0),
        _make_list_result([]),
    ])

    response = await client.get(
        "/api/v1/documents/",
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_documents_empty(client_and_db):
    client, mock_db = client_and_db
    mock_db.execute = AsyncMock(side_effect=[
        _make_count_result(0),
        _make_list_result([]),
    ])

    response = await client.get(
        "/api/v1/documents/",
    )
    body = response.json()
    assert body["total"] == 0
    assert body["results"] == []


@pytest.mark.asyncio
async def test_list_documents_response_shape(client_and_db):
    client, mock_db = client_and_db
    doc = _make_fake_doc(chunk_count=3)
    mock_db.execute = AsyncMock(side_effect=[
        _make_count_result(1),
        _make_list_result([doc]),
    ])

    response = await client.get(
        "/api/v1/documents/?page=1&page_size=10",
    )
    body = response.json()

    assert body["total"] == 1
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert len(body["results"]) == 1
    assert body["results"][0]["status"] == "ready"


@pytest.mark.asyncio
async def test_list_documents_pagination_defaults(client_and_db):
    client, mock_db = client_and_db
    mock_db.execute = AsyncMock(side_effect=[
        _make_count_result(0),
        _make_list_result([]),
    ])

    response = await client.get(
        "/api/v1/documents/",
    )
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 10
