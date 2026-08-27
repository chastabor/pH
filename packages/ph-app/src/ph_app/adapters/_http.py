"""What every HTTP-streaming adapter shares: the client, the credential, the failure.

The two wires pH speaks differ in message shape and in how usage is reported.
They do **not** differ in how a request is sent, how a secret reaches a header,
or what an HTTP status means for retry — and when those were written twice the
copies drifted (one overflow heuristic matched `max_tokens` anywhere in a body,
turning a bad-request 400 into a compaction trigger). So they live here once.

One `httpx.AsyncClient` per adapter, not per request: creating a client inside
`stream()` pays a fresh TCP connect and TLS handshake on every model call and
never reaches keep-alive or HTTP/2 multiplexing. The adapter's row disposes the
client with its scope.

@module ph_app.adapters._http
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from ph.cordis import Context
from ph.llm.adapter import LlmError
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, LlmFailure

from .sse import iter_sse

__all__ = ["HttpClient", "failure_from_status", "resolve_secret"]

TIMEOUT = httpx.Timeout(600.0, connect=15.0)

_STATUS_CODES = {429: "RATE_LIMIT", 401: "AUTHENTICATION", 403: "AUTHENTICATION", 529: "OVERLOADED"}


def resolve_secret(ctx: Context, env_name: str, provider: str) -> str:
    """Turn a credential *name* into its value — here, at the edge, and nowhere above (I-3).

    The value goes into a local that goes out of scope with the request. Nothing
    that travelled to get here held it.
    """
    credentials = ctx.get("credentials")
    if credentials is None:
        raise LlmError("ctx.credentials is not mounted", "NO_CREDENTIALS")
    secret = credentials.resolve(credentials.reference(env_name))
    if secret is None:
        raise LlmError(
            f'{env_name} is not set, so provider "{provider}" cannot be called',
            "MISSING_CREDENTIAL",
        )
    value: str = secret.reveal()
    return value


def failure_from_status(status: int, body: str, *, is_overflow: Callable[[str], bool]) -> LlmError:
    """Classify an HTTP error into the codes the retry policy routes on.

    `is_overflow` is the one wire-specific judgement: each provider phrases
    "your prompt is too long" differently, and getting it wrong in either
    direction is expensive — a missed overflow retries forever, a false one
    compacts a conversation that fit.
    """
    code = _STATUS_CODES.get(status, "SERVER_ERROR" if status >= 500 else "REQUEST_FAILED")
    if is_overflow(body):
        code = CONTEXT_WINDOW_EXCEEDED
    detail = body[:400] or f"HTTP {status}"
    return LlmError(
        f"provider returned {status}: {detail}",
        code,
        LlmFailure(message=detail, code=code, status=status),
    )


class HttpClient:
    """A lazily-created, long-lived `httpx.AsyncClient` with one streaming shape."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=TIMEOUT)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream_sse(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        is_overflow: Callable[[str], bool],
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """POST and yield `(event, payload)` for every JSON SSE payload.

        A non-2xx response is raised as a classified `LlmError` before any
        payload is yielded, so a consumer never sees a half-stream.
        """
        async with self._get().stream("POST", url, headers=headers, json=json) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise failure_from_status(response.status_code, body, is_overflow=is_overflow)
            async for event, payload in iter_sse(response):
                if isinstance(payload, dict):
                    yield event, payload
