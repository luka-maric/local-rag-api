import time

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(client):
    response = await client.get("/health")
    body = response.json()

    assert "status" in body
    assert "env" in body
    assert "version" in body
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_health_is_fast(client):
    """Health check must respond in < 200ms — fails if someone adds a DB call here."""
    start = time.monotonic()
    response = await client.get("/health")
    elapsed_ms = (time.monotonic() - start) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 200, f"Health check took {elapsed_ms:.1f}ms - too slow"
