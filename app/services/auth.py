"""JWT and password helpers."""
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
from jose import jwt

from app.config import settings


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt. Returns the hash string."""
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the stored bcrypt hash."""
    return _bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(tenant_id: uuid.UUID) -> str:
    """Create a signed JWT containing sub=str(tenant_id) and exp=now+jwt_expiry_minutes."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    payload = {
        "sub": str(tenant_id),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> uuid.UUID:
    """
    Decode and verify a JWT. Returns the tenant UUID from the 'sub' claim.

    Raises jose.JWTError if the token is expired, has an invalid signature,
    or is malformed. The caller converts JWTError into HTTP 401.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    return uuid.UUID(payload["sub"])
