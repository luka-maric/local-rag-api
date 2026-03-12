from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np

import pytest

from app.services.embedding import EmbeddingService


FAKE_DIM = 384  # must match settings.embedding_dim


def fake_vectors(n: int, value: float = 0.5) -> list[list[float]]:
    return [[value] * FAKE_DIM for _ in range(n)]


@pytest.fixture
def no_cache_service():
    return EmbeddingService(redis=None)


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None   # default: cache miss
    redis.set.return_value = True
    return redis


@pytest.fixture
def cached_service(mock_redis):
    return EmbeddingService(redis=mock_redis)


@pytest.mark.asyncio
async def test_embed_texts_returns_correct_count():
    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(3)):
        service = EmbeddingService(redis=None)
        result = await service.embed_texts(["text one", "text two", "text three"])

    assert len(result) == 3


@pytest.mark.asyncio
async def test_embed_texts_returns_correct_dimensions():
    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(2)):
        service = EmbeddingService(redis=None)
        result = await service.embed_texts(["hello", "world"])

    for vector in result:
        assert len(vector) == FAKE_DIM, f"Expected {FAKE_DIM} dims, got {len(vector)}"


@pytest.mark.asyncio
async def test_embed_texts_returns_list_of_list_of_float():
    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(1)):
        service = EmbeddingService(redis=None)
        result = await service.embed_texts(["test"])

    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)


@pytest.mark.asyncio
async def test_embed_texts_empty_input_returns_empty():
    with patch("app.services.embedding._encode_sync") as mock_encode:
        service = EmbeddingService(redis=None)
        result = await service.embed_texts([])

    assert result == []
    mock_encode.assert_not_called()


@pytest.mark.asyncio
async def test_embed_one_returns_single_vector():
    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(1, 0.7)):
        service = EmbeddingService(redis=None)
        result = await service.embed_one("a single sentence")

    assert isinstance(result, list)
    assert len(result) == FAKE_DIM
    assert result[0] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_model_called_once_for_entire_batch():
    """Batch efficiency: model.encode is called once for all texts, not N times."""
    expected_texts = ["text_a", "text_b", "text_c", "text_d"]

    with patch("app.services.embedding._encode_sync") as mock_encode:
        mock_encode.return_value = fake_vectors(4)
        service = EmbeddingService(redis=None)
        await service.embed_texts(expected_texts)

    mock_encode.assert_called_once()
    actual_texts_passed = mock_encode.call_args[0][0]
    assert actual_texts_passed == expected_texts


@pytest.mark.asyncio
async def test_cache_miss_calls_model(mock_redis):
    mock_redis.get.return_value = None  # explicit miss

    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(2)) as mock_encode:
        service = EmbeddingService(redis=mock_redis)
        await service.embed_texts(["text_a", "text_b"])

    mock_encode.assert_called_once()


@pytest.mark.asyncio
async def test_cache_miss_stores_result_in_redis(mock_redis):
    vector = fake_vectors(1)[0]
    mock_redis.get.return_value = None

    with patch("app.services.embedding._encode_sync", return_value=[vector]):
        service = EmbeddingService(redis=mock_redis)
        await service.embed_texts(["cacheable text"])

    mock_redis.set.assert_called_once()
    args, kwargs = mock_redis.set.call_args
    cache_key = args[0]
    stored_value = args[1]

    assert cache_key.startswith("embed:")
    assert stored_value == np.array(vector, dtype=np.float32).tobytes()
    assert "ex" in kwargs  # TTL must be set


@pytest.mark.asyncio
async def test_cache_hit_skips_model(mock_redis):
    cached_vector = fake_vectors(1, 0.42)[0]
    mock_redis.get.return_value = np.array(cached_vector, dtype=np.float32).tobytes()

    with patch("app.services.embedding._encode_sync") as mock_encode:
        service = EmbeddingService(redis=mock_redis)
        await service.embed_texts(["previously embedded text"])

    mock_encode.assert_not_called()  # model must NOT be called


@pytest.mark.asyncio
async def test_cache_hit_returns_correct_vector(mock_redis):
    cached_vector = [round(i * 0.001, 6) for i in range(FAKE_DIM)]
    mock_redis.get.return_value = np.array(cached_vector, dtype=np.float32).tobytes()

    with patch("app.services.embedding._encode_sync"):
        service = EmbeddingService(redis=mock_redis)
        result = await service.embed_one("some text")

    assert result == pytest.approx(cached_vector)


@pytest.mark.asyncio
async def test_mixed_cache_hits_and_misses_preserves_order(mock_redis):
    """
    Given texts [A, B, C] where A and C are cached but B is not:
    - A and C come from Redis (no model call for them)
    - B goes to the model
    - The returned list must be [A_vector, B_vector, C_vector] in that order

    This is the hardest edge case: the service must collect misses, batch
    them, then reconstruct the full result in the original order.
    """
    vector_a = fake_vectors(1, 0.1)[0]
    vector_b = fake_vectors(1, 0.2)[0]
    vector_c = fake_vectors(1, 0.3)[0]

    # A: hit, B: miss, C: hit
    mock_redis.get.side_effect = [
        np.array(vector_a, dtype=np.float32).tobytes(),  # A → cache hit
        None,                                             # B → cache miss
        np.array(vector_c, dtype=np.float32).tobytes(),  # C → cache hit
    ]

    with patch("app.services.embedding._encode_sync") as mock_encode:
        mock_encode.return_value = [vector_b]  # only B goes to model
        service = EmbeddingService(redis=mock_redis)
        result = await service.embed_texts(["text_a", "text_b", "text_c"])

    assert result[0] == pytest.approx(vector_a)
    assert result[1] == pytest.approx(vector_b)
    assert result[2] == pytest.approx(vector_c)

    # Model was only called for the one miss (text_b)
    mock_encode.assert_called_once()
    assert mock_encode.call_args[0][0] == ["text_b"]


@pytest.mark.asyncio
async def test_redis_get_error_falls_back_to_model(mock_redis):
    mock_redis.get.side_effect = ConnectionError("Redis connection refused")

    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(1)) as mock_encode:
        service = EmbeddingService(redis=mock_redis)
        result = await service.embed_texts(["text despite redis error"])

    # Must have called the model (cache was unavailable)
    mock_encode.assert_called_once()
    assert len(result) == 1
    assert len(result[0]) == FAKE_DIM


@pytest.mark.asyncio
async def test_redis_set_error_does_not_crash(mock_redis):
    mock_redis.get.return_value = None  # cache miss
    mock_redis.set.side_effect = ConnectionError("Redis write failed")

    with patch("app.services.embedding._encode_sync", return_value=fake_vectors(1)):
        service = EmbeddingService(redis=mock_redis)
        result = await service.embed_texts(["text with failed cache write"])

    assert len(result) == 1  # embedding returned despite cache write failure
    assert len(result[0]) == FAKE_DIM


def test_cache_key_is_deterministic():
    service = EmbeddingService()
    key1 = service._cache_key("hello world")
    key2 = service._cache_key("hello world")
    assert key1 == key2


def test_cache_key_different_texts_different_keys():
    service = EmbeddingService()
    key_a = service._cache_key("text one")
    key_b = service._cache_key("text two")
    assert key_a != key_b


def test_cache_key_has_embed_prefix():
    """Key must start with "embed:" to avoid collisions with other Redis namespaces."""
    service = EmbeddingService()
    key = service._cache_key("any text")
    assert key.startswith("embed:")
