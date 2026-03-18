import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt

from app.api.v1.auth import _register_limiter, _token_limiter
from app.config import settings
from app.db.session import get_db
from app.dependencies import get_redis
from app.main import create_app
from app.services.auth import create_access_token, generate_refresh_token, hash_password, hash_refresh_token


def _make_app_with_db(mock_db):
    app = create_app()

    async def _mock_get_db():
        yield mock_db

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)  # no tokens blocklisted by default

    app.dependency_overrides[get_db] = _mock_get_db
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[_register_limiter] = lambda: None
    app.dependency_overrides[_token_limiter] = lambda: None
    return app


def _no_tenant_db():
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    def _auto_set_id(obj):
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()

    db.add = MagicMock(side_effect=_auto_set_id)
    db.flush = AsyncMock()
    return db


def _existing_tenant_db(name: str, password: str):
    tenant = MagicMock()
    tenant.id = uuid.uuid4()
    tenant.name = name
    tenant.password_hash = hash_password(password)

    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = tenant
    db.execute = AsyncMock(return_value=result)
    return db, tenant


def _documents_list_db():
    """
    Mock DB for GET /api/v1/documents/ — two execute() calls:
      1. COUNT(*) → scalar_one() = 0
      2. SELECT OFFSET/LIMIT → scalars().all() = []
    """
    db = AsyncMock()

    count_result = MagicMock()
    count_result.scalar_one.return_value = 0

    list_result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = []
    list_result.scalars.return_value = scalars

    db.execute = AsyncMock(side_effect=[count_result, list_result])
    return db


@pytest.mark.asyncio
async def test_register_returns_201():
    app = _make_app_with_db(_no_tenant_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={"name": "acme-corp", "password": "supersecret"},
        )

    assert response.status_code == 201
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert "tenant_id" in body
    assert "expires_in" in body


@pytest.mark.asyncio
async def test_register_token_decodes_to_correct_tenant_id():
    app = _make_app_with_db(_no_tenant_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={"name": "acme-corp", "password": "supersecret"},
        )

    body = response.json()
    payload = jwt.decode(
        body["access_token"],
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    assert payload["sub"] == body["tenant_id"]


@pytest.mark.asyncio
async def test_register_duplicate_name_returns_409():
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = MagicMock()  # existing tenant found
    db.execute = AsyncMock(return_value=result)

    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={"name": "existing-corp", "password": "supersecret"},
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_short_password_returns_422():
    app = _make_app_with_db(_no_tenant_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={"name": "acme-corp", "password": "short"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_empty_name_returns_422():
    app = _make_app_with_db(_no_tenant_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={"name": "", "password": "supersecret"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_token_returns_200_with_valid_credentials():
    db, _ = _existing_tenant_db("acme-corp", "supersecret")
    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/token",
            json={"name": "acme-corp", "password": "supersecret"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_token_wrong_password_returns_401():
    db, _ = _existing_tenant_db("acme-corp", "supersecret")
    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/token",
            json={"name": "acme-corp", "password": "wrongpassword"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials."


@pytest.mark.asyncio
async def test_token_unknown_name_returns_401():
    """
    Unknown tenant name → 401.

    The error message is identical to wrong password — this prevents tenant
    name enumeration attacks (attacker can't tell if the name exists).
    """
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None  # tenant not found
    db.execute = AsyncMock(return_value=result)

    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/token",
            json={"name": "ghost-corp", "password": "supersecret"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials."


@pytest.mark.asyncio
async def test_token_missing_fields_returns_422():
    app = _make_app_with_db(_no_tenant_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/auth/token", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_valid_jwt_grants_access():
    tenant_id = uuid.uuid4()
    token = create_access_token(tenant_id)

    app = _make_app_with_db(_documents_list_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/documents/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_jwt_returns_401():
    app = _make_app_with_db(_documents_list_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/documents/",
            headers={"Authorization": "Bearer not.a.valid.jwt"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token."


@pytest.mark.asyncio
async def test_expired_jwt_returns_401():
    tenant_id = uuid.uuid4()
    expired_payload = {
        "sub": str(tenant_id),
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    expired_token = jwt.encode(
        expired_payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    app = _make_app_with_db(_documents_list_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/documents/",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_no_authorization_header_returns_401():
    """
    No Authorization header → 401 Unauthorized.

    FastAPI ≥0.115 changed HTTPBearer to return 401 (not 403) when no
    Authorization header is present.
    """
    app = _make_app_with_db(_documents_list_db())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/documents/")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logout_returns_204():
    token = create_access_token(uuid.uuid4())

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    app = _make_app_with_db(_no_tenant_db())
    app.dependency_overrides[get_redis] = lambda: mock_redis

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 204
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_returns_new_access_token():
    tenant_id = uuid.uuid4()
    refresh_token = generate_refresh_token()

    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.refresh_token_hash = hash_refresh_token(refresh_token)

    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = tenant
    db.execute = AsyncMock(return_value=result)

    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"tenant_id": str(tenant_id), "refresh_token": refresh_token},
        )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["refresh_token"] != refresh_token  # rotated


@pytest.mark.asyncio
async def test_refresh_invalid_token_returns_401():
    tenant_id = uuid.uuid4()

    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.refresh_token_hash = hash_refresh_token("correct-token")

    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = tenant
    db.execute = AsyncMock(return_value=result)

    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"tenant_id": str(tenant_id), "refresh_token": "wrong-token"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_unknown_tenant_returns_401():
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    app = _make_app_with_db(db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"tenant_id": str(uuid.uuid4()), "refresh_token": "any-token"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_revoked_token_returns_401():
    token = create_access_token(uuid.uuid4())

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"1")  # every jti is "revoked"

    app = _make_app_with_db(_documents_list_db())
    app.dependency_overrides[get_redis] = lambda: mock_redis

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/documents/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Token has been revoked."
