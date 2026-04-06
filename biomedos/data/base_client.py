"""Reusable async HTTP client foundation for biomedical APIs."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class AsyncAPIClient:
    """Base class for rate-limited async API clients."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 5,
        requests_per_second: float = 5.0,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the base API client."""

        default_headers = {"User-Agent": "BioMedOS/0.1.0"}
        if headers:
            default_headers.update(headers)

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.requests_per_second = requests_per_second
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=default_headers,
        )

    async def _throttle(self) -> None:
        """Enforce a simple requests-per-second rate limit."""

        if self.requests_per_second <= 0:
            return

        minimum_interval = 1.0 / self.requests_per_second
        async with self._rate_lock:
            now = time.perf_counter()
            elapsed = now - self._last_request_time
            remaining = minimum_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request_time = time.perf_counter()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry, throttling, and concurrency control."""

        async with self._semaphore:
            await self._throttle()
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
                        params=params,
                        json=json_body,
                    )
                    response.raise_for_status()
                    return response

        msg = "Retry loop exited unexpectedly."
        raise RuntimeError(msg)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a request and return a JSON object."""

        response = await self._request(method, path, params=params, json_body=json_body)
        payload = response.json()
        if not isinstance(payload, dict):
            msg = "Expected a JSON object response."
            raise ValueError(msg)
        return payload

    async def _request_text(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Execute a request and return the raw response text."""

        response = await self._request(method, path, params=params)
        return str(response.text)

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncAPIClient:
        """Enter an async context manager."""

        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit an async context manager."""

        await self.close()
