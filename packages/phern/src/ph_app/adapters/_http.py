"""What every HTTP-streaming adapter shares: the client, the credential, the failure.

The three wires pH speaks differ in message shape, in how usage is reported, and
— since P7-03 — in how a file API is shaped. They do **not** differ in how a
request is sent, how a secret reaches a header, or what an HTTP status means for
retry, and when those were written twice the copies drifted (one overflow
heuristic matched `max_tokens` anywhere in a body, turning a bad-request 400 into
a compaction trigger). So they live here once.

One `httpx.AsyncClient` per adapter, not per request: creating a client inside
`stream()` pays a fresh TCP connect and TLS handshake on every model call and
never reaches keep-alive or HTTP/2 multiplexing. The adapter's row disposes the
client with its scope.

@module ph_app.adapters._http
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx

from ph.cordis import Context
from ph.json import as_str
from ph.keys import CREDENTIALS
from ph.llm.adapter import LlmError
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    FILE_EXPIRED,
    Finish,
    FinishReason,
    LlmFailure,
)

from .sse import iter_sse

__all__ = ["HttpClient", "failure_from_status", "resolve_secret", "wire_error_finish"]

TIMEOUT = httpx.Timeout(600.0, connect=15.0)

_STATUS_CODES = {429: "RATE_LIMIT", 401: "AUTHENTICATION", 403: "AUTHENTICATION", 529: "OVERLOADED"}


def resolve_secret(ctx: Context, env_name: str, provider: str) -> str:
    """Turn a credential *name* into its value — here, at the edge, and nowhere above (I-3).

    The value goes into a local that goes out of scope with the request. Nothing
    that traveled to get here held it.
    """
    credentials = ctx.get(CREDENTIALS)
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


def _refined_code(
    code: str,
    body: str,
    *,
    is_overflow: Callable[[str], bool],
    is_missing_file: Callable[[str], bool] | None,
) -> str:
    """The wire-specific judgements, applied to a code the shape already gave.

    **One ladder, because two of them is the bug this module is named for.**
    Both entry points start from a different table — a status from
    `_STATUS_CODES`, an error frame's `type` from `_WIRE_ERROR_CODES` — and then
    ask the same two questions of the same body in the same order. Those four
    lines were written twice, and the order is load-bearing: overflow is applied
    last because it is the one a caller can act on, and a body that reads as
    both is an overflow. Kept apart, that ordering lived in two docstrings and
    nothing checked that they agreed.
    """
    if is_missing_file is not None and is_missing_file(body):
        code = FILE_EXPIRED
    if is_overflow(body):
        code = CONTEXT_WINDOW_EXCEEDED
    return code


def failure_from_status(
    status: int,
    body: str,
    *,
    is_overflow: Callable[[str], bool],
    is_missing_file: Callable[[str], bool] | None = None,
) -> LlmError:
    """Classify an HTTP error into the codes the retry policy routes on.

    The wire-specific judgements are callbacks because each provider phrases them
    differently, and both are expensive to get wrong in either direction: a missed
    overflow retries forever, a false one compacts a conversation that fit.

    `is_missing_file` is the second of them (P7-03), here rather than in an
    adapter because a body classified once must not be re-read into a different
    code further up — and because both wires have a file API, so the next one
    inherits this instead of writing its own parser. Whether the missing file was
    *ours* is a separate question only the caller can answer, and it answers it
    against the code rather than the prose.
    """
    code = _refined_code(
        _STATUS_CODES.get(status, "SERVER_ERROR" if status >= 500 else "REQUEST_FAILED"),
        body,
        is_overflow=is_overflow,
        is_missing_file=is_missing_file,
    )
    detail = body[:400] or f"HTTP {status}"
    return LlmError(
        f"provider returned {status}: {detail}",
        code,
        LlmFailure(message=detail, code=code, status=status),
    )


@contextmanager
def _classified_transport() -> Iterator[None]:
    """Turn httpx's transport failures into the codes the retry policy routes on.

    **The half of "the request failed" that never reaches a status.** A dropped
    connection, a refused socket or a read that runs out of time raises out of
    httpx rather than returning a response, so `failure_from_status` never sees
    it and the error arrives at the retry policy as a bare exception with code
    `UNKNOWN` — not transient, so the turn fails on the first hiccup. Both
    `CONNECTION_ERROR` and `TIMEOUT` are in `TRANSIENT_CODES`, and before this
    nothing in the codebase produced either: the two entries were dead and the
    failure they were written for was the one being reported as fatal.

    One `except` and a visible `isinstance`, rather than a clause per code:
    `TimeoutException` is a *subclass* of `TransportError`, so two clauses would
    make the classification depend on the order they are written in — a
    correctness hazard with nothing but a comment holding it in place.
    """
    try:
        yield
    except httpx.TransportError as error:
        code = "TIMEOUT" if isinstance(error, httpx.TimeoutException) else "CONNECTION_ERROR"
        # The class name carries the diagnosis — `ConnectError`, `ReadTimeout`,
        # `RemoteProtocolError` — and httpx's message for several of them is
        # empty, so a bare `str(error)` would log a failure with no text at all.
        detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        raise LlmError(detail, code) from error


_WIRE_ERROR_CODES = {
    # Anthropic's `error.type` vocabulary.
    "overloaded_error": "OVERLOADED",
    "rate_limit_error": "RATE_LIMIT",
    "api_error": "SERVER_ERROR",
    "authentication_error": "AUTHENTICATION",
    "permission_error": "AUTHENTICATION",
    # The OpenAI-compatible wire's.
    "server_error": "SERVER_ERROR",
    "rate_limit_exceeded": "RATE_LIMIT",
    "insufficient_quota": "RATE_LIMIT",
    "service_unavailable": "SERVICE_UNAVAILABLE",
    # Google's wire spells the same field `error.status`, in gRPC's vocabulary.
    "RESOURCE_EXHAUSTED": "RATE_LIMIT",
    "UNAVAILABLE": "SERVICE_UNAVAILABLE",
    "INTERNAL": "SERVER_ERROR",
    "DEADLINE_EXCEEDED": "TIMEOUT",
    "UNAUTHENTICATED": "AUTHENTICATION",
    "PERMISSION_DENIED": "AUTHENTICATION",
}
"""A mid-stream `error.type` onto the shared failure vocabulary (L3).

**The same overload has to answer the same way however it arrives.** A provider
under load reports it as a 529 on one request and an `{"error": {"type":
"overloaded_error"}}` frame inside a 200 on the next — `failure_from_status`
mapped the first to `OVERLOADED` and retried it, while the second became a flat
`PROVIDER_ERROR`, which is not in `TRANSIENT_CODES`, and failed the turn. One
fact, two codes, decided by which shape the provider happened to use.

Only the types whose meaning is unambiguous are here, from all three wires;
the three vocabularies are disjoint, so one exact-match lookup serves them and
no adapter has to know about another's. An unrecognized type stays
`PROVIDER_ERROR`, which is the honest answer for a
vocabulary neither this module nor `TRANSIENT_CODES` owns: retrying something
this does not understand is the failure mode the narrow map exists to avoid."""


def wire_error_finish(
    error: Mapping[str, Any],
    *,
    kind: object,
    is_overflow: Callable[[str], bool],
    is_missing_file: Callable[[str], bool],
) -> Finish:
    """A provider's mid-stream error frame, as the chunk that ends the turn.

    The other half of `failure_from_status`, and here for the same reason. All
    three wires report a failed request *inside* a 200 — Anthropic as an `error`
    event, the other two as a top-level `error` object — so the status
    classifier never sees it and each adapter built the `Finish` itself.
    The two copies had drifted on the first commit they both existed: one read
    the message through `str()`, which renders a non-string as `"None"` or
    `"3"`, the other through `as_str`, which falls back to the default. That is
    the drift this module's docstring gives as the reason it exists.

    **The same three judgements as the status path, in the same order** (O1).
    L3 gave this half a code vocabulary and stopped there, so the two halves
    answered differently for the one failure with a remedy: a mid-stream
    `{"code": "context_length_exceeded"}` — which both the Anthropic and
    OpenAI-compatible wires send — became a flat `PROVIDER_ERROR`, and
    `compaction.on_request_error` branches on `CONTEXT_WINDOW_EXCEEDED`, so the
    conversation that would have fitted after a compaction never got one. That
    is the "two answers for one fact" L3 set out to close, surviving on the
    other axis. Overflow is applied last because it is the one a caller can act
    on.

    **The whole frame is offered to the callbacks, not just its message**, which
    is what makes the two halves classify the same text: the status path hands
    them the raw body, `code` and all, so a callback written against one cannot
    quietly mean something else against the other.

    `kind` is the frame's own `error.type`, mapped through `_WIRE_ERROR_CODES`
    so a retryable failure is retryable whichever shape it arrived in. Required
    and keyword-only, because an adapter that forgot it would silently get
    `PROVIDER_ERROR` back — which is the shape this exists to prevent. A type
    nobody has mapped still answers `PROVIDER_ERROR`; that is the honest case.

    **Both predicates are required here**, unlike on the status path, for that
    same argument: every wire that streams has a file API, so an omitted
    `is_missing_file` is an oversight rather than a choice, and the silence it
    buys is an expired file reported as `PROVIDER_ERROR` — a code with no
    remedy standing in for one that has. `failure_from_status` keeps the
    default because `post_json` and `get_json` genuinely call it from paths
    with no file in hand.
    """
    code = _refined_code(
        _WIRE_ERROR_CODES.get(as_str(kind), "PROVIDER_ERROR"),
        json.dumps(dict(error), default=str),
        is_overflow=is_overflow,
        is_missing_file=is_missing_file,
    )
    failure = LlmFailure(message=as_str(error.get("message"), "provider error"), code=code)
    return Finish(reason=FinishReason(kind="error", failure=failure))


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

    async def post_multipart(
        self,
        url: str,
        *,
        headers: dict[str, str],
        field: str,
        filename: str,
        content: bytes,
        mime: str,
        is_overflow: Callable[[str], bool],
        data: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST one file as multipart form data and return the parsed reply.

        Here rather than in an adapter for `stream_sse`'s reason: a file API is
        one more thing both wires have, and the status→code classification is
        the part that must not be written twice. `Content-Type` is left to
        `httpx`, which has to compute the multipart boundary anyway.

        `data` is the form fields that ride beside the file. Anthropic's Files API
        takes none; OpenAI's requires `purpose`, and without a way to send it the
        second wire's uploader would have had to build its own request and inherit
        none of the classification above (P7-03).
        """
        sending = {name: value for name, value in headers.items() if name != "Content-Type"}
        with _classified_transport():
            response = await self._get().post(
                url, headers=sending, files={field: (filename, content, mime)}, data=data or {}
            )
        if response.status_code >= 400:
            raise failure_from_status(
                response.status_code,
                response.text,
                is_overflow=is_overflow,
            )
        parsed: dict[str, Any] = response.json()
        return parsed

    async def post_raw(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        content: bytes | None = None,
        is_overflow: Callable[[str], bool],
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """POST a JSON body or raw bytes; return `(parsed body, response headers)`.

        The **headers** are why this exists and why it is not `post_multipart`
        (P7-03). Google's Files API is a two-step resumable upload whose first
        step answers with an empty body and the destination in
        `X-Goog-Upload-URL`, and whose second step is the file's bytes with no
        form encoding around them at all. Neither shape fits a multipart helper,
        and both want the same status→code classification, which is the whole
        reason this module exists.

        A body that is not JSON comes back as `{}` rather than raising: the first
        step of that upload legitimately returns nothing, and a helper that
        insisted on JSON would make the caller catch a decode error to discover
        success.
        """
        with _classified_transport():
            response = await self._get().post(url, headers=headers, json=json, content=content)
        if response.status_code >= 400:
            raise failure_from_status(response.status_code, response.text, is_overflow=is_overflow)
        try:
            parsed: dict[str, Any] = response.json()
        except ValueError:
            parsed = {}
        return parsed, response.headers

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        is_overflow: Callable[[str], bool],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """GET and parse — the poll half of an upload that finishes asynchronously.

        `timeout` overrides `TIMEOUT` for one call, and exists for the caller
        that is not waiting on a model: a *mount-time* probe inherits a ten
        minute read budget otherwise, so a host that accepts the socket and then
        stalls holds up plugin mount — and with it a TUI start and `phern doctor`.
        A request whose answer nobody is blocked on keeps the generous default.
        """
        with _classified_transport():
            response = await self._get().get(
                url,
                headers=headers,
                # httpx's own sentinel rather than a splat: "say nothing and take
                # the client's" is a value here, and spreading a conditional dict
                # past a keyword-typed signature is a hole the checker cannot see
                # through.
                timeout=httpx.Timeout(timeout) if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            )
        if response.status_code >= 400:
            raise failure_from_status(response.status_code, response.text, is_overflow=is_overflow)
        parsed: dict[str, Any] = response.json()
        return parsed

    async def stream_sse(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        is_overflow: Callable[[str], bool],
        is_missing_file: Callable[[str], bool] | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """POST and yield `(event, payload)` for every JSON SSE payload.

        A non-2xx response is raised as a classified `LlmError` before any
        payload is yielded, so a consumer never sees a half-stream.
        """
        with _classified_transport():
            async with self._get().stream("POST", url, headers=headers, json=json) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise failure_from_status(
                        response.status_code,
                        body,
                        is_overflow=is_overflow,
                        is_missing_file=is_missing_file,
                    )
                # Inside the guard, not only around the connect: a stream that
                # dies halfway through an answer is the common shape of this
                # failure, and it raises here rather than at the `stream` call.
                async for event, payload in iter_sse(response):
                    if isinstance(payload, dict):
                        yield event, payload
