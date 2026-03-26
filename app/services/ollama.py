import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class OllamaServiceError(Exception):
    pass


class OllamaService:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return [self.model]
            return [m["name"] for m in response.json().get("models", [])] or [self.model]
        except Exception:
            return [self.model]

    async def generate(self, messages: list[dict[str, Any]], model: str | None = None) -> str:
        payload = {"model": model or self.model, "messages": messages, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaServiceError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Try: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaServiceError(f"Ollama request timed out after {self.timeout}s.") from exc

        if response.status_code != 200:
            raise OllamaServiceError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        resolved_model = model or self.model
        text: str = response.json()["message"]["content"]
        logger.info("ollama_generate_complete", model=resolved_model, output_chars=len(text))
        return text

    async def stream(self, messages: list[dict[str, Any]], model: str | None = None) -> AsyncIterator[str]:
        resolved_model = model or self.model
        payload = {"model": resolved_model, "messages": messages, "stream": True}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise OllamaServiceError(
                            f"Ollama returned HTTP {response.status_code}: {body.decode()}"
                        )

                    token_count = 0
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk["message"]["content"]
                        if token:
                            yield token
                            token_count += 1
                        if chunk.get("done"):
                            break

                    logger.info("ollama_stream_complete", model=resolved_model, tokens_yielded=token_count)

        except httpx.ConnectError as exc:
            raise OllamaServiceError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Try: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaServiceError(f"Ollama request timed out after {self.timeout}s.") from exc
        except json.JSONDecodeError as exc:
            raise OllamaServiceError(f"Ollama sent malformed NDJSON: {exc}") from exc
