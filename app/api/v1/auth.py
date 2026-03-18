import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import JWTError
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Tenant
from app.db.session import get_db
from app.dependencies import get_redis, security
from app.schemas.auth import RefreshRequest, RegisterRequest, TokenRequest, TokenResponse
from app.services.auth import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
    verify_refresh_token,
)
from app.services.rate_limit import RateLimiter

logger = structlog.get_logger()

router = APIRouter(prefix="/auth", tags=["auth"])

_register_limiter = RateLimiter(
    max_requests=settings.auth_register_rate_limit_requests,
    window_seconds=settings.auth_register_rate_limit_window,
    key_prefix="register",
)
_token_limiter = RateLimiter(
    max_requests=settings.auth_token_rate_limit_requests,
    window_seconds=settings.auth_token_rate_limit_window,
    key_prefix="token",
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=201,
    summary="Register a new tenant",
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_register_limiter),
) -> TokenResponse:
    result = await db.execute(select(Tenant).where(Tenant.name == request.name))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"A tenant named '{request.name}' already exists.")

    refresh_token = generate_refresh_token()
    tenant = Tenant(
        name=request.name,
        password_hash=hash_password(request.password),
        refresh_token_hash=hash_refresh_token(refresh_token),
    )
    db.add(tenant)
    await db.flush()

    logger.info("tenant_registered", tenant_id=str(tenant.id), name=request.name)

    return TokenResponse(
        access_token=create_access_token(tenant.id),
        tenant_id=tenant.id,
        expires_in=settings.jwt_expiry_minutes * 60,
        refresh_token=refresh_token,
    )


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=200,
    summary="Obtain a JWT for an existing tenant",
)
async def token(
    request: TokenRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_token_limiter),
) -> TokenResponse:
    result = await db.execute(select(Tenant).where(Tenant.name == request.name))
    existing = result.scalar_one_or_none()

    # Same 401 for unknown name and wrong password — prevents tenant name enumeration.
    if existing is None or not verify_password(request.password, existing.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    refresh_token = generate_refresh_token()
    existing.refresh_token_hash = hash_refresh_token(refresh_token)

    logger.info("tenant_authenticated", tenant_id=str(existing.id))

    return TokenResponse(
        access_token=create_access_token(existing.id),
        tenant_id=existing.id,
        expires_in=settings.jwt_expiry_minutes * 60,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse, status_code=200, summary="Refresh access token")
async def refresh(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(Tenant).where(Tenant.id == request.tenant_id))
    tenant = result.scalar_one_or_none()

    if tenant is None or tenant.refresh_token_hash is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    if not verify_refresh_token(request.refresh_token, tenant.refresh_token_hash):
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    new_refresh_token = generate_refresh_token()
    tenant.refresh_token_hash = hash_refresh_token(new_refresh_token)

    logger.info("token_refreshed", tenant_id=str(tenant.id))

    return TokenResponse(
        access_token=create_access_token(tenant.id),
        tenant_id=tenant.id,
        expires_in=settings.jwt_expiry_minutes * 60,
        refresh_token=new_refresh_token,
    )


@router.post("/logout", status_code=204, summary="Revoke the current access token")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    redis: Redis = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
) -> None:
    try:
        payload = decode_access_token(credentials.credentials)
        jti = payload["jti"]
        tenant_id = uuid.UUID(payload["sub"])
        ttl = max(1, payload["exp"] - int(datetime.now(timezone.utc).timestamp()))
        await redis.setex(f"blocklist:{jti}", ttl, "1")

        result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        tenant = result.scalar_one_or_none()
        if tenant:
            tenant.refresh_token_hash = None

        logger.info("token_revoked", jti=jti, tenant_id=str(tenant_id))
    except (JWTError, KeyError):
        pass  # expired or old-format token — nothing to revoke
