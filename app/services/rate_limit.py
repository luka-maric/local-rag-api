"""Redis fixed-window rate limiter for auth endpoints."""
import structlog
from fastapi import HTTPException, Request
from redis.asyncio import Redis

from app.config import settings

logger = structlog.get_logger()

# decode_responses=True: Redis returns str, not bytes.
_redis = Redis.from_url(settings.redis_url, decode_responses=True)


def _get_client_ip(request: Request) -> str:
    """Return the real client IP, checking X-Forwarded-For for proxied requests."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Fixed-window rate limit by IP. Use as a FastAPI Depends() callable."""

    def __init__(self, max_requests: int, window_seconds: int, key_prefix: str):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def __call__(self, request: Request) -> None:
        client_ip = _get_client_ip(request)
        key = f"rate_limit:{self.key_prefix}:{client_ip}"

        try:
            count = await _redis.incr(key)

            # Set expiry only on the first increment — setting it on every call
            # would reset the window on every request, defeating the purpose.
            if count == 1:
                await _redis.expire(key, self.window_seconds)

            if count > self.max_requests:
                logger.warning(
                    "rate_limit_exceeded",
                    key_prefix=self.key_prefix,
                    client_ip=client_ip,
                    count=count,
                    max_requests=self.max_requests,
                )
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later.",
                )

        except HTTPException:
            raise  # don't swallow our own 429 inside the broad except below

        except Exception as exc:
            # fail-open: Redis unavailable → request passes through
            logger.warning(
                "rate_limiter_redis_unavailable",
                key_prefix=self.key_prefix,
                error=str(exc),
            )
