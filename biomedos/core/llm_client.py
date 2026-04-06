"""Async Ollama client used across all agents."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from biomedos.config import Settings, get_settings


class ChatMessage(BaseModel):
    """A chat message for Ollama chat completion calls."""

    role: str
    content: str


class OllamaClient:
    """Async client for the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Ollama base URL.
            timeout: Request timeout in seconds.
            settings: Optional application settings.
            client: Optional preconfigured HTTP client.
        """

        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a JSON request with retries."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadTimeout)
            ),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(
                    method=method,
                    url=path,
                    json=json_body,
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    msg = "Expected a JSON object from Ollama."
                    raise ValueError(msg)
                return payload

        msg = "Retry loop exited unexpectedly."
        raise RuntimeError(msg)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion from a prompt."""

        payload: dict[str, Any] = {
            "model": model or self.settings.MODEL_REASONER,
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system
        if options is not None:
            payload["options"] = options

        response = await self._request_json("POST", "/api/generate", json_body=payload)
        return str(response.get("response", "")).strip()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Run a non-streaming chat completion."""

        payload: dict[str, Any] = {
            "model": model or self.settings.MODEL_REASONER,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
        }
        if options is not None:
            payload["options"] = options

        response = await self._request_json("POST", "/api/chat", json_body=payload)
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()
        return ""

    async def stream_generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a text generation response from Ollama."""

        payload: dict[str, Any] = {
            "model": model or self.settings.MODEL_REASONER,
            "prompt": prompt,
            "stream": True,
        }
        if system is not None:
            payload["system"] = system
        if options is not None:
            payload["options"] = options

        async with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = self._decode_stream_chunk(line)
                if chunk:
                    yield chunk

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from Ollama."""

        payload: dict[str, Any] = {
            "model": model or self.settings.MODEL_REASONER,
            "messages": [message.model_dump() for message in messages],
            "stream": True,
        }
        if options is not None:
            payload["options"] = options

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = self._decode_stream_chunk(line)
                if chunk:
                    yield chunk

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed one or more texts using an Ollama embedding model."""

        payload = {
            "model": model or self.settings.MODEL_EMBEDDING,
            "input": list(texts),
        }
        response = await self._request_json("POST", "/api/embed", json_body=payload)
        embeddings = response.get("embeddings", [])
        if not isinstance(embeddings, list):
            msg = "Ollama returned an invalid embedding payload."
            raise ValueError(msg)

        normalized: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                msg = "Embedding vector must be a list."
                raise ValueError(msg)
            normalized.append([float(value) for value in embedding])
        return normalized

    async def health_check(self) -> bool:
        """Return whether Ollama is reachable."""

        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def list_models(self) -> list[str]:
        """List installed Ollama model names."""

        payload = await self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []

        names: list[str] = []
        for model in models:
            if isinstance(model, dict) and "name" in model:
                names.append(str(model["name"]))
        return names

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model into the local Ollama cache."""

        payload = {"name": model_name, "stream": False}
        await self._request_json("POST", "/api/pull", json_body=payload)
        return True

    async def delete_model(self, model_name: str) -> bool:
        """Delete a local Ollama model."""

        payload = {"name": model_name}
        await self._request_json("POST", "/api/delete", json_body=payload)
        return True

    async def ensure_model(self, model_name: str) -> bool:
        """Ensure a model exists locally, pulling it if necessary."""

        models = await self.list_models()
        if model_name in models:
            return True
        return await self.pull_model(model_name)

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        """Enter an async context."""

        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit an async context."""

        await self.close()

    @staticmethod
    def _decode_stream_chunk(line: str) -> str:
        """Decode a streaming JSON line from Ollama."""

        data = json.loads(line)
        if not isinstance(data, dict):
            return ""
        if "response" in data:
            return str(data["response"])
        message = data.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return ""
