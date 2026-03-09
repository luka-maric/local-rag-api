"""
Authentication endpoints.

POST /api/v1/auth/register — create a new tenant and return a JWT
POST /api/v1/auth/token    — verify credentials and return a JWT
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Tenant
from app.db.session import get_db
from app.schemas.auth import RegisterRequest, TokenRequest, TokenResponse
from app.services.auth import create_access_token, hash_password, verify_password
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
    description=(
        "Create a new tenant with a unique name and password. "
        "Returns a JWT Bearer token valid for all protected endpoints."
    ),
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_register_limiter),
) -> TokenResponse:
    result = await db.execute(select(Tenant).where(Tenant.name == request.name))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A tenant named '{request.name}' already exists.",
        )

    tenant = Tenant(
        name=request.name,
        password_hash=hash_password(request.password),
    )
    db.add(tenant)
    await db.flush()  # get tenant.id without committing yet

    logger.info("tenant_registered", tenant_id=str(tenant.id), name=request.name)

    return TokenResponse(
        access_token=create_access_token(tenant.id),
        tenant_id=tenant.id,
        expires_in=settings.jwt_expiry_minutes * 60,
    )


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=200,
    summary="Obtain a JWT for an existing tenant",
    description=(
        "Exchange tenant name and password for a JWT Bearer token. "
        "Pass it as 'Authorization: Bearer <token>' on protected endpoints."
    ),
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

    logger.info("tenant_authenticated", tenant_id=str(existing.id))

    return TokenResponse(
        access_token=create_access_token(existing.id),
        tenant_id=existing.id,
        expires_in=settings.jwt_expiry_minutes * 60,
    )
