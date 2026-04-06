"""Tests for the async Ollama client."""

from __future__ import annotations

import httpx
import pytest

from biomedos.core.llm_client import ChatMessage, OllamaClient


@pytest.mark.asyncio
async def test_generate_retries_on_failure() -> None:
    """The client retries transient HTTP failures."""

    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if request.url.path == "/api/generate" and attempts["count"] == 1:
            return httpx.Response(500, json={"error": "temporary"})
        return httpx.Response(200, json={"response": "ready"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
        ollama = OllamaClient(base_url="http://test", client=client)
        result = await ollama.generate("hello")

    assert result == "ready"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_stream_chat_yields_chunks() -> None:
    """The client yields individual streaming chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        content = b'{"message":{"content":"hello"}}\n{"message":{"content":" world"}}\n'
        return httpx.Response(200, content=content)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
        ollama = OllamaClient(base_url="http://test", client=client)
        chunks = [
            chunk async for chunk in ollama.stream_chat([ChatMessage(role="user", content="hello")])
        ]

    assert "".join(chunks) == "hello world"


@pytest.mark.asyncio
async def test_health_check_returns_false_on_error() -> None:
    """Health checks fail gracefully."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
        ollama = OllamaClient(base_url="http://test", client=client)
        healthy = await ollama.health_check()

    assert healthy is False
