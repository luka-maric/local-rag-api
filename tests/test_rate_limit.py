import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request

from app.services.rate_limit import RateLimiter, _get_client_ip


def _make_request(
    client_host: str = "127.0.0.1",
    forwarded_for: str | None = None,
) -> MagicMock:
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = client_host
    headers = {}
    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for
    request.headers = headers
    return request


def _make_redis(incr_return: int = 1, expire_return: int = 1) -> AsyncMock:
    redis = AsyncMock()
    redis.incr = AsyncMock(return_value=incr_return)
    redis.expire = AsyncMock(return_value=expire_return)
    return redis


@pytest.mark.asyncio
async def test_rate_limiter_allows_first_request():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="test")
    mock_redis = _make_redis(incr_return=1)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request())  # must not raise


@pytest.mark.asyncio
async def test_rate_limiter_allows_request_at_exactly_max():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="test")
    mock_redis = _make_redis(incr_return=5)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request())  # must not raise


@pytest.mark.asyncio
async def test_rate_limiter_blocks_when_over_limit():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="test")
    mock_redis = _make_redis(incr_return=6)

    with patch("app.services.rate_limit._redis", mock_redis):
        with pytest.raises(HTTPException) as exc_info:
            await limiter(_make_request())

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_sets_expire_on_first_request():
    limiter = RateLimiter(max_requests=5, window_seconds=300, key_prefix="test")
    mock_redis = _make_redis(incr_return=1)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request())

    mock_redis.expire.assert_called_once()
    key_arg, window_arg = mock_redis.expire.call_args.args
    assert window_arg == 300


@pytest.mark.asyncio
async def test_rate_limiter_does_not_set_expire_on_subsequent_requests():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="test")
    mock_redis = _make_redis(incr_return=2)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request())

    mock_redis.expire.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limiter_fails_open_on_redis_error():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="test")
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock(side_effect=ConnectionError("Redis down"))

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request())  # must not raise — fail open


@pytest.mark.asyncio
async def test_rate_limiter_key_includes_prefix_and_ip():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="token")
    mock_redis = _make_redis(incr_return=1)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request(client_host="1.2.3.4"))

    incr_key = mock_redis.incr.call_args.args[0]
    assert incr_key == "rate_limit:token:1.2.3.4"


@pytest.mark.asyncio
async def test_rate_limiter_different_ips_get_separate_keys():
    limiter = RateLimiter(max_requests=5, window_seconds=60, key_prefix="token")
    mock_redis = _make_redis(incr_return=1)

    with patch("app.services.rate_limit._redis", mock_redis):
        await limiter(_make_request(client_host="1.1.1.1"))
        await limiter(_make_request(client_host="2.2.2.2"))

    keys_used = [call.args[0] for call in mock_redis.incr.call_args_list]
    assert "rate_limit:token:1.1.1.1" in keys_used
    assert "rate_limit:token:2.2.2.2" in keys_used
    assert keys_used[0] != keys_used[1]


def test_get_client_ip_uses_x_forwarded_for_when_present():
    request = _make_request(client_host="10.0.0.1", forwarded_for="1.2.3.4")
    assert _get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_takes_first_entry_from_forwarded_for():
    request = _make_request(forwarded_for="1.2.3.4, 5.6.7.8, 9.10.11.12")
    assert _get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_strips_whitespace_from_forwarded_for():
    request = _make_request(forwarded_for="  1.2.3.4  , 5.6.7.8")
    assert _get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_falls_back_to_client_host():
    request = _make_request(client_host="192.168.1.50")
    assert _get_client_ip(request) == "192.168.1.50"
