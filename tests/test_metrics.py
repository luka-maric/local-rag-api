"""Prometheus /metrics endpoint tests."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app  # module-level singleton — NOT create_app()


@pytest.fixture
async def metrics_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200(metrics_client):
    response = await metrics_client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_text_plain(metrics_client):
    response = await metrics_client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_http_requests_total_metric_present(metrics_client):
    await metrics_client.get("/health")  # triggers the middleware counter
    response = await metrics_client.get("/metrics")
    assert "http_requests_total" in response.text


@pytest.mark.asyncio
async def test_document_upload_counter_present(metrics_client):
    response = await metrics_client.get("/metrics")
    assert "rag_documents_uploaded_total" in response.text


@pytest.mark.asyncio
async def test_embedding_cache_counters_present(metrics_client):
    response = await metrics_client.get("/metrics")
    assert "rag_embedding_cache_hits_total" in response.text
    assert "rag_embedding_cache_misses_total" in response.text
