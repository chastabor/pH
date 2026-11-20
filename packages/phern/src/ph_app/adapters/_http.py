"""What every HTTP-streaming adapter shares: the client, the credential, the failure.

The three wires that pH speaks differ in message shape and in how usage is
reported. Since P7-03 they also differ in the shape of a file API. They do **not**
differ in how a request is sent, how a secret reaches a header, or what an HTTP
status means for retry. When those parts were written twice, the copies drifted:
one overflow heuristic matched `max_tokens` anywhere in a body, which turned a
bad-request 400 into a compaction trigger. So the shared parts are in this
module, one time.

Each adapter has one `httpx.AsyncClient`, not one for each request. A client that
is created inside `stream()` pays a new TCP connect and TLS handshake on every
model call, and never reaches keep-alive or HTTP/2 multiplexing. The row of the
adapter disposes the client with its scope.

@module ph_app.adapters._http
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx

from ph.characters import head
from ph.cordis import Context
from ph.json import as_str
from ph.keys import CREDENTIALS
from ph.llm.adapter import LlmError, failure_of
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    FILE_EXPIRED,
    MISSING_CREDENTIAL,
    TIMEOUT,
    Finish,
    FinishReason,
)

from .sse import iter_sse

__all__ = ["HttpClient", "failure_from_status", "resolve_secret", "wire_error_finish"]

CLIENT_TIMEOUT = httpx.Timeout(600.0, connect=15.0)

_STATUS_CODES = {429: "RATE_LIMIT", 401: "AUTHENTICATION", 403: "AUTHENTICATION", 529: "OVERLOADED"}


def resolve_secret(ctx: Context, env_name: str, provider: str) -> str:
    """Turn a credential *name* into its value, at the edge and in no layer above (I-3).

    The value goes into a local variable that goes out of scope with the request.
    Nothing on the path to this function held the value.
    """
    credentials = ctx.get(CREDENTIALS)
    if credentials is None:
        raise LlmError("ctx.credentials is not mounted", "NO_CREDENTIALS")
    secret = credentials.resolve(credentials.reference(env_name))
    if secret is None:
        raise LlmError(
            f'{env_name} is not set, so provider "{provider}" cannot be called',
            MISSING_CREDENTIAL,
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
    """Apply the wire-specific judgements to a code that the shape already gave.

    **There is one ladder, because two ladders are the bug that this module
    exists to prevent.** The two entry points start from different tables: a
    status goes through `_STATUS_CODES`, and the `type` of an error frame goes
    through `_WIRE_ERROR_CODES`. Both then ask the same two questions of the same
    body in the same order. Those four lines were written twice. The order
    matters: overflow is applied last because it is the one that a caller can act
    on, and a body that reads as both is an overflow. While the two copies were
    apart, that order was stated in two docstrings, and nothing checked that they
    agreed.
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
    """Classify an HTTP error into the codes that the retry policy routes on.

    The wire-specific judgements are callbacks, because each provider phrases them
    differently. Both are expensive to get wrong in either direction: a missed
    overflow retries forever, and a false overflow compacts a conversation that
    fit.

    `is_missing_file` is the second judgement (P7-03). It is here and not in an
    adapter for two reasons. A body that is classified one time must not be read
    again into a different code further up. And all three wires have a file API,
    so a new wire inherits this judgement and does not write its own parser.
    Whether the missing file was *ours* is a separate question that only the
    caller can answer. The caller answers it against the code and not against the
    prose.
    """
    code = _refined_code(
        _STATUS_CODES.get(status, "SERVER_ERROR" if status >= 500 else "REQUEST_FAILED"),
        body,
        is_overflow=is_overflow,
        is_missing_file=is_missing_file,
    )
    detail = head(body, 400) or f"HTTP {status}"
    return LlmError(
        f"provider returned {status}: {detail}",
        code,
        failure_of(detail, code, said=body, status=status),
    )


@contextmanager
def _classified_transport() -> Iterator[None]:
    """Turn httpx transport failures into the codes that the retry policy routes on.

    **This is the part of "the request failed" that never reaches a status.** A
    dropped connection, a refused socket, or a read that runs out of time raises
    out of httpx and returns no response. `failure_from_status` therefore never
    sees it. Without this function, the error arrives at the retry policy as a
    bare exception with code `UNKNOWN`. That code is not transient, so the turn
    fails on the first interruption. Both `CONNECTION_ERROR` and `TIMEOUT` are in
    `TRANSIENT_CODES`. Before this function, nothing in the codebase produced
    either code. The two entries were dead, and the failure that they were
    written for was reported as fatal.

    There is one `except` and a visible `isinstance`, not one clause for each
    code. `TimeoutException` is a *subclass* of `TransportError`. With two
    clauses, the classification would depend on the order in which they are
    written. That is a correctness hazard that only a comment would hold in
    place.
    """
    try:
        yield
    except httpx.TransportError as error:
        code = TIMEOUT if isinstance(error, httpx.TimeoutException) else "CONNECTION_ERROR"
        # The class name carries the diagnosis: `ConnectError`, `ReadTimeout`,
        # `RemoteProtocolError`. The httpx message for several of them is empty,
        # so a bare `str(error)` would log a failure with no text at all.
        detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        raise LlmError(detail, code) from error


_WIRE_ERROR_CODES = {
    # Anthropic's `error.type` vocabulary.
    "overloaded_error": "OVERLOADED",
    "rate_limit_error": "RATE_LIMIT",
    "api_error": "SERVER_ERROR",
    "authentication_error": "AUTHENTICATION",
    "permission_error": "AUTHENTICATION",
    # The vocabulary of the OpenAI-compatible wire.
    "server_error": "SERVER_ERROR",
    "rate_limit_exceeded": "RATE_LIMIT",
    "insufficient_quota": "RATE_LIMIT",
    "service_unavailable": "SERVICE_UNAVAILABLE",
    # Google's wire spells the same field `error.status`, in gRPC's vocabulary.
    "RESOURCE_EXHAUSTED": "RATE_LIMIT",
    "UNAVAILABLE": "SERVICE_UNAVAILABLE",
    "INTERNAL": "SERVER_ERROR",
    "DEADLINE_EXCEEDED": TIMEOUT,
    "UNAUTHENTICATED": "AUTHENTICATION",
    "PERMISSION_DENIED": "AUTHENTICATION",
}
"""Maps a mid-stream `error.type` onto the shared failure vocabulary (L3).

**The same overload must get the same answer in whatever shape it arrives.** A
provider under load reports the overload as a 529 on one request, and as an
`{"error": {"type": "overloaded_error"}}` frame inside a 200 on the next.
`failure_from_status` mapped the first to `OVERLOADED` and retried it. The second
became a flat `PROVIDER_ERROR`, which is not in `TRANSIENT_CODES`, and failed the
turn. One fact had two codes, and the shape that the provider used decided
between them.

Only the types whose meaning is unambiguous are here, from all three wires. The
three vocabularies are disjoint, so one exact-match lookup serves them, and no
adapter has to know about the vocabulary of another. An unrecognized type stays
`PROVIDER_ERROR`. That is the correct answer for a vocabulary that neither this
module nor `TRANSIENT_CODES` owns. The narrow map exists so that pH does not
retry a failure that it does not understand."""


def wire_error_finish(
    error: Mapping[str, Any],
    *,
    kind: object,
    is_overflow: Callable[[str], bool],
    is_missing_file: Callable[[str], bool],
) -> Finish:
    """A mid-stream error frame of a provider, as the chunk that ends the turn.

    This is the other half of `failure_from_status`, and it is here for the same
    reason. All three wires report a failed request *inside* a 200: Anthropic as
    an `error` event, the other two as a top-level `error` object. The status
    classifier never sees such a failure, so each adapter built the `Finish`
    itself. The two copies had drifted on the first commit in which they both
    existed. One read the message through `str()`, which renders a non-string as
    `"None"` or `"3"`. The other read it through `as_str`, which falls back to
    the default. That is the drift that the docstring of this module gives as the
    reason why the module exists.

    **The same three judgements as the status path, in the same order** (O1). L3
    gave this half a code vocabulary and stopped there. The two halves therefore
    answered differently for the one failure that has a remedy. Both the
    Anthropic wire and the OpenAI-compatible wire send a mid-stream
    `{"code": "context_length_exceeded"}`, and it became a flat `PROVIDER_ERROR`.
    `compaction.on_request_error` branches on `CONTEXT_WINDOW_EXCEEDED`, so the
    conversation that would have fitted after a compaction never got one. That is
    the "two answers for one fact" that L3 set out to close, and it survived on
    the other axis. Overflow is applied last because it is the one that a caller
    can act on.

    **The callbacks get the whole frame, not only its message.** This is what
    makes the two halves classify the same text. The status path gives the
    callbacks the raw body, including `code`. A callback that is written against
    one half therefore cannot silently mean something else against the other.

    `kind` is the `error.type` of the frame itself. It is mapped through
    `_WIRE_ERROR_CODES`, so that a retryable failure is retryable in whichever
    shape it arrived. It is required and keyword-only, because an adapter that
    forgot it would silently get `PROVIDER_ERROR` back, and that is the outcome
    that this function exists to prevent. A type that nobody has mapped still
    answers `PROVIDER_ERROR`, which is the correct answer for that case.

    **Both predicates are required here**, unlike on the status path, for the
    same argument. Every wire that streams has a file API, so an omitted
    `is_missing_file` is an oversight and not a choice. The result of the
    omission would be an expired file reported as `PROVIDER_ERROR`: a code with
    no remedy in place of a code that has one. `failure_from_status` keeps the
    default because `post_multipart`, `post_raw` and `get_json` call it from
    paths that have no file to refer to.
    """
    code = _refined_code(
        _WIRE_ERROR_CODES.get(as_str(kind), "PROVIDER_ERROR"),
        json.dumps(dict(error), default=str),
        is_overflow=is_overflow,
        is_missing_file=is_missing_file,
    )
    failure = failure_of(as_str(error.get("message"), "provider error"), code)
    return Finish(reason=FinishReason(kind="error", failure=failure))


class HttpClient:
    """A lazily-created, long-lived `httpx.AsyncClient` with one streaming shape."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=CLIENT_TIMEOUT)
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

        It is here and not in an adapter for the same reason as `stream_sse`. A
        file API is one more thing that all three wires have, and the status→code
        classification is the part that must not be written twice. `Content-Type`
        is left to `httpx`, which must compute the multipart boundary in any case.

        `data` is the form fields that are sent beside the file. The Files API of
        Anthropic takes none. The Files API of OpenAI requires `purpose`. Without
        a way to send it, the uploader of the second wire would have had to build
        its own request, and would have inherited none of the classification
        above (P7-03).
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
        """POST a JSON body or raw bytes, and return `(parsed body, response headers)`.

        The **headers** are the reason why this method exists and why it is not
        `post_multipart` (P7-03). The Files API of Google is a resumable upload in
        two steps. The first step answers with an empty body and the destination
        in `X-Goog-Upload-URL`. The second step is the bytes of the file with no
        form encoding around them at all. Neither shape fits a multipart helper.
        Both need the same status→code classification, which is the reason why
        this module exists.

        A body that is not JSON comes back as `{}` and does not raise. The first
        step of that upload legitimately returns nothing. A helper that insisted
        on JSON would make the caller catch a decode error to learn that the
        request succeeded.
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
        """GET and parse: the poll half of an upload that finishes asynchronously.

        `timeout` overrides `CLIENT_TIMEOUT` for one call. It exists for the
        caller that is not waiting on a model. Without it, a *mount-time* probe
        inherits a read budget of ten minutes. A host that accepts the socket and
        then stalls would hold up the plugin mount, and with it a TUI start and
        `phern doctor`. A request whose answer nobody is blocked on keeps the
        generous default.
        """
        with _classified_transport():
            response = await self._get().get(
                url,
                headers=headers,
                # This is the sentinel of httpx itself and not a splat. "Say
                # nothing and take the timeout of the client" is a value here. A
                # conditional dict spread past a keyword-typed signature is a hole
                # that the type checker cannot see through.
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

        A response with status 400 or above is raised as a classified `LlmError`
        before any payload is yielded, so a consumer never sees a half-stream.
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
                # The loop is inside the guard, and the guard is not only around
                # the connect. A stream that dies in the middle of an answer is
                # the common shape of this failure, and it raises here and not at
                # the `stream` call.
                async for event, payload in iter_sse(response):
                    if isinstance(payload, dict):
                        yield event, payload
