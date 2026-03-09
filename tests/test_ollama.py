import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.ollama import OllamaService, OllamaServiceError

BASE_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG?"},
]


def _make_generate_mocks(status_code: int = 200, content: str = "RAG stands for Retrieval-Augmented Generation."):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    mock_response.text = "Internal server error"  # used by non-200 path

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_client_cm, mock_client, mock_response


def _make_stream_mocks(status_code: int = 200, ndjson_lines: list[str] | None = None):
    if ndjson_lines is None:
        ndjson_lines = [
            json.dumps({"message": {"content": "RAG"}, "done": False}),
            json.dumps({"message": {"content": " stands"}, "done": False}),
            json.dumps({"message": {"content": " for Retrieval-Augmented Generation."}, "done": True}),
        ]

    async def _aiter_lines():
        for line in ndjson_lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aiter_lines = _aiter_lines
    mock_response.aread = AsyncMock(return_value=b"Service error")

    # client.stream() is NOT awaited — plain MagicMock, returns async CM
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_cm

    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_client_cm, mock_client, mock_response


@pytest.mark.asyncio
async def test_generate_returns_assistant_content():
    expected = "RAG stands for Retrieval-Augmented Generation."
    mock_client_cm, _, _ = _make_generate_mocks(content=expected)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        result = await service.generate(MESSAGES)

    assert result == expected


@pytest.mark.asyncio
async def test_generate_posts_to_correct_url():
    mock_client_cm, mock_client, _ = _make_generate_mocks()

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        await service.generate(MESSAGES)

    call_args = mock_client.post.call_args
    url = call_args[0][0]  # first positional argument
    assert url == f"{BASE_URL}/api/chat"


@pytest.mark.asyncio
async def test_generate_sends_correct_model_and_messages():
    mock_client_cm, mock_client, _ = _make_generate_mocks()

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        await service.generate(MESSAGES)

    payload = mock_client.post.call_args[1]["json"]
    assert payload["model"] == MODEL
    assert payload["messages"] == MESSAGES
    assert payload["stream"] is False  # generate() must NOT stream


@pytest.mark.asyncio
async def test_generate_non_200_raises_ollama_error():
    mock_client_cm, _, _ = _make_generate_mocks(status_code=503)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="503"):
            await service.generate(MESSAGES)


@pytest.mark.asyncio
async def test_generate_connect_error_raises_ollama_error():
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__.side_effect = httpx.ConnectError("Connection refused")
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="Cannot connect"):
            await service.generate(MESSAGES)


@pytest.mark.asyncio
async def test_generate_timeout_raises_ollama_error():
    mock_client_cm, mock_client, _ = _make_generate_mocks()
    mock_client.post.side_effect = httpx.TimeoutException("Read timeout")

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="timed out"):
            await service.generate(MESSAGES)


@pytest.mark.asyncio
async def test_stream_yields_tokens_in_order():
    mock_client_cm, _, _ = _make_stream_mocks()

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        tokens = [t async for t in service.stream(MESSAGES)]

    assert tokens == ["RAG", " stands", " for Retrieval-Augmented Generation."]
    full_text = "".join(tokens)
    assert full_text == "RAG stands for Retrieval-Augmented Generation."


@pytest.mark.asyncio
async def test_stream_skips_empty_lines():
    lines_with_blanks = [
        "",  # blank
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        "",  # blank
        json.dumps({"message": {"content": " world"}, "done": True}),
    ]
    mock_client_cm, _, _ = _make_stream_mocks(ndjson_lines=lines_with_blanks)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        tokens = [t async for t in service.stream(MESSAGES)]

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_stops_at_done_true():
    lines = [
        json.dumps({"message": {"content": "First"}, "done": False}),
        json.dumps({"message": {"content": " token"}, "done": True}),
        json.dumps({"message": {"content": " SHOULD_NOT_APPEAR"}, "done": False}),
    ]
    mock_client_cm, _, _ = _make_stream_mocks(ndjson_lines=lines)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        tokens = [t async for t in service.stream(MESSAGES)]

    assert "SHOULD_NOT_APPEAR" not in " ".join(tokens)
    assert tokens == ["First", " token"]


@pytest.mark.asyncio
async def test_stream_non_200_raises_ollama_error():
    mock_client_cm, _, _ = _make_stream_mocks(status_code=404)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="404"):
            async for _ in service.stream(MESSAGES):
                pass


@pytest.mark.asyncio
async def test_stream_connect_error_raises_ollama_error():
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__.side_effect = httpx.ConnectError("Connection refused")
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="Cannot connect"):
            async for _ in service.stream(MESSAGES):
                pass


@pytest.mark.asyncio
async def test_stream_timeout_raises_ollama_error():
    mock_client_cm, mock_client, _ = _make_stream_mocks()
    # Raise during the stream context manager entry
    mock_client.stream.return_value.__aenter__.side_effect = httpx.TimeoutException("Read timeout")

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="timed out"):
            async for _ in service.stream(MESSAGES):
                pass


@pytest.mark.asyncio
async def test_stream_malformed_json_raises_ollama_error():
    bad_lines = [
        json.dumps({"message": {"content": "Good line"}, "done": False}),
        "NOT { valid JSON }}}",
    ]
    mock_client_cm, _, _ = _make_stream_mocks(ndjson_lines=bad_lines)

    service = OllamaService(base_url=BASE_URL, model=MODEL)
    with patch("app.services.ollama.httpx.AsyncClient", return_value=mock_client_cm):
        with pytest.raises(OllamaServiceError, match="malformed NDJSON"):
            async for _ in service.stream(MESSAGES):
                pass
