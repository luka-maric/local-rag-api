"""JWT Bearer token authentication dependency."""
import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.services.auth import decode_access_token

security = HTTPBearer()


async def get_current_tenant_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> uuid.UUID:
    """
    Validate the JWT and return the tenant UUID.

    Raises:
      HTTP 401 — missing or malformed Authorization header (raised by HTTPBearer)
      HTTP 401 — token present but invalid, expired, or malformed
    """
    try:
        return decode_access_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )
