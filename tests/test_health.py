from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_db_mock(raises: Exception | None = None):
    """Return a mock AsyncSessionLocal context manager."""
    session = AsyncMock()
    if raises:
        session.execute = AsyncMock(side_effect=raises)
    else:
        session.execute = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def _make_redis_mock(raises: Exception | None = None):
    """Return a mock Redis.from_url result."""
    redis = AsyncMock()
    if raises:
        redis.ping = AsyncMock(side_effect=raises)
    else:
        redis.ping = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return MagicMock(return_value=redis)


@pytest.mark.asyncio
async def test_health_returns_200_when_all_ok(client):
    with patch("app.main.AsyncSessionLocal", _make_db_mock()), \
         patch("app.main.Redis.from_url", _make_redis_mock()):
        response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(client):
    with patch("app.main.AsyncSessionLocal", _make_db_mock()), \
         patch("app.main.Redis.from_url", _make_redis_mock()):
        response = await client.get("/health")
    body = response.json()

    assert body["status"] == "ok"
    assert body["env"] is not None
    assert body["version"] is not None
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_503_when_db_down(client):
    with patch("app.main.AsyncSessionLocal", _make_db_mock(raises=OSError("connection refused"))), \
         patch("app.main.Redis.from_url", _make_redis_mock()):
        response = await client.get("/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert "error" in body["checks"]["database"]
    assert body["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_503_when_redis_down(client):
    with patch("app.main.AsyncSessionLocal", _make_db_mock()), \
         patch("app.main.Redis.from_url", _make_redis_mock(raises=OSError("connection refused"))):
        response = await client.get("/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert body["checks"]["database"] == "ok"
    assert "error" in body["checks"]["redis"]


@pytest.mark.asyncio
async def test_health_returns_503_when_both_down(client):
    with patch("app.main.AsyncSessionLocal", _make_db_mock(raises=OSError("db down"))), \
         patch("app.main.Redis.from_url", _make_redis_mock(raises=OSError("redis down"))):
        response = await client.get("/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert "error" in body["checks"]["database"]
    assert "error" in body["checks"]["redis"]
