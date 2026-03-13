import structlog
from fastapi import HTTPException, Request
from redis.asyncio import Redis

from app.config import settings

logger = structlog.get_logger()

_redis = Redis.from_url(settings.redis_url, decode_responses=True)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int, key_prefix: str):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def __call__(self, request: Request) -> None:
        client_ip = _get_client_ip(request)
        key = f"rate_limit:{self.key_prefix}:{client_ip}"

        try:
            count = await _redis.incr(key)

            # EXPIRE only on first increment — resetting on every call would extend the window
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
                raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

        except HTTPException:
            raise

        except Exception as exc:
            # fail-open: Redis unavailable → let the request through
            logger.warning("rate_limiter_redis_unavailable", key_prefix=self.key_prefix, error=str(exc))
