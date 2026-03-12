import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies import get_current_tenant_id
from app.db.session import get_db
from app.main import create_app


TENANT_ID = str(uuid.uuid4())
FAKE_QUERY = "What is the invoice total?"
FAKE_DIM = 384
FAKE_QUERY_VECTOR = [0.1] * FAKE_DIM  # 384-dim vector, all 0.1


def _make_fake_chunk(document_id: uuid.UUID | None = None, chunk_index: int = 0) -> MagicMock:
    chunk = MagicMock()
    chunk.document_id = document_id or uuid.uuid4()
    chunk.chunk_index = chunk_index
    chunk.chunk_text = f"This is chunk {chunk_index} text."
    return chunk


def _make_db_result(rows: list[tuple]) -> MagicMock:
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
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
async def client_and_db(app_with_mock_db):
    app, mock_db = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, mock_db


def _query_payload(query: str = FAKE_QUERY, top_k: int = 5) -> dict:
    return {"json": {"query": query, "top_k": top_k}}


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_query_returns_200(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])

    chunk = _make_fake_chunk()
    mock_db.execute = AsyncMock(return_value=_make_db_result([(chunk, 0.2)]))

    response = await client.post("/api/v1/query", **_query_payload())

    assert response.status_code == 200


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_query_response_shape(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])

    chunk = _make_fake_chunk(chunk_index=0)
    mock_db.execute = AsyncMock(return_value=_make_db_result([(chunk, 0.1)]))

    response = await client.post("/api/v1/query", **_query_payload())
    body = response.json()

    assert body["query"] == FAKE_QUERY
    assert body["total"] == 1
    assert len(body["results"]) == 1


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_similarity_score_is_one_minus_distance(mock_embed, client_and_db):
    """
    similarity = 1 - cosine_distance.

    pgvector returns a cosine distance (0=identical, 2=opposite).
    We convert it so callers get an intuitive similarity score:
      distance 0.0 → similarity 1.0  (perfect match)
      distance 0.3 → similarity 0.7
      distance 1.0 → similarity 0.0
    """
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])

    chunk = _make_fake_chunk()
    mock_db.execute = AsyncMock(return_value=_make_db_result([(chunk, 0.3)]))

    response = await client.post("/api/v1/query", **_query_payload())
    result = response.json()["results"][0]

    assert result["similarity"] == pytest.approx(0.7, abs=0.0001)


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_multiple_results_ordered_by_similarity(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])

    chunk_a = _make_fake_chunk(chunk_index=0)
    chunk_b = _make_fake_chunk(chunk_index=1)
    chunk_c = _make_fake_chunk(chunk_index=2)
    # Sorted by distance ascending (most similar first)
    mock_db.execute = AsyncMock(
        return_value=_make_db_result([(chunk_a, 0.05), (chunk_b, 0.3), (chunk_c, 0.7)])
    )

    response = await client.post("/api/v1/query", **_query_payload(top_k=3))
    results = response.json()["results"]

    assert len(results) == 3
    assert results[0]["similarity"] == pytest.approx(0.95, abs=0.0001)
    assert results[1]["similarity"] == pytest.approx(0.70, abs=0.0001)
    assert results[2]["similarity"] == pytest.approx(0.30, abs=0.0001)


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_chunk_text_and_document_id_in_result(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])

    doc_id = uuid.uuid4()
    chunk = _make_fake_chunk(document_id=doc_id, chunk_index=3)
    mock_db.execute = AsyncMock(return_value=_make_db_result([(chunk, 0.1)]))

    response = await client.post("/api/v1/query", **_query_payload())
    result = response.json()["results"][0]

    assert uuid.UUID(result["document_id"]) == doc_id
    assert result["chunk_index"] == 3
    assert result["chunk_text"] == "This is chunk 3 text."


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_no_results_returns_empty_list(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])
    mock_db.execute = AsyncMock(return_value=_make_db_result([]))

    response = await client.post("/api/v1/query", **_query_payload())
    body = response.json()

    assert response.status_code == 200
    assert body["total"] == 0
    assert body["results"] == []


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_embed_texts_called_with_query_string(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])
    mock_db.execute = AsyncMock(return_value=_make_db_result([]))

    await client.post("/api/v1/query", **_query_payload(query="specific question"))

    mock_embed.embed_texts.assert_called_once_with(["specific question"])


@pytest.mark.asyncio
async def test_empty_query_returns_422(client_and_db):
    client, _ = client_and_db
    response = await client.post(
        "/api/v1/query",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_top_k_above_max_returns_422(client_and_db):
    client, _ = client_and_db
    response = await client.post(
        "/api/v1/query",
        json={"query": FAKE_QUERY, "top_k": 100},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_top_k_zero_returns_422(client_and_db):
    client, _ = client_and_db
    response = await client.post(
        "/api/v1/query",
        json={"query": FAKE_QUERY, "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@patch("app.api.v1.query._embedding_service")
async def test_ef_search_set_before_vector_query(mock_embed, client_and_db):
    client, mock_db = client_and_db
    mock_embed.embed_texts = AsyncMock(return_value=[FAKE_QUERY_VECTOR])
    mock_db.execute = AsyncMock(return_value=_make_db_result([]))

    await client.post("/api/v1/query", **_query_payload())

    first_call_sql = str(mock_db.execute.call_args_list[0].args[0])
    assert "ef_search" in first_call_sql
