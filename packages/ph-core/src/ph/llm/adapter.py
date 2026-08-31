"""`ctx.llm` — the model adapter seam.

Definition, Provider, Consumer (invariant I5). The *definition* is
`LlmAdapter`; a *provider* is any plugin calling `ctx.llm.register_adapter`;
the *consumer* is the loop, which never learns which adapter answered.

Every call goes through the `llm/stream` waterfall, which is where retry,
replay, checkpoint policy and session-title all attach in dsh. An adapter that
raises is normalized into a terminal `finish{error}` chunk before any consumer
sees it, so a consumer never handles two shapes for the same failure.

@module ph.llm.adapter
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cordis import Context, Running, events, plugin, running
from .types import Finish, FinishReason, GenerateOptions, LlmFailure

__all__ = ["AdapterHandle", "LlmAdapter", "LlmError", "LlmRuntime", "ResolvedModel", "apply"]

log = logging.getLogger("ph.llm")

events.declare(
    "llm/stream",
    "waterfall",
    GenerateOptions,
    owner="ph.llm",
    doc="Wraps every model call. Retry, replay and recording attach here.",
)
events.declare(
    "llm/adapters-updated",
    "emit",
    owner="ph.llm",
    doc="The provider topology changed; consumers re-read list_providers().",
)


class LlmError(Exception):
    """A structured model-call failure.

    Carries the provider's own facts rather than a flattened string, because
    `turn/end{error}` records them verbatim and a later reader needs the code.
    """

    def __init__(self, message: str, code: str, failure: LlmFailure | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.failure = failure or LlmFailure(message=message, code=code)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Exact-route metadata an adapter can answer for one provider/model."""

    context_window: int | None = None
    default_max_tokens: int | None = None
    reasoning: tuple[str, ...] = ()
    accepts: frozenset[str] = frozenset()
    """MIME types this route takes as message *content* (P7-01).

    Empty means text only, which is every route until an adapter says otherwise —
    the safe default, since the failure it prevents is a media block reaching a
    wire that silently drops it. A caller that finds a block outside this set
    degrades it to a text pointer and logs a notice; it never sends it and hopes,
    and never fails the turn, because a session begun on a vision model and
    resumed on a text one must still open."""
    max_attachment_bytes: int | None = None
    """Per-attachment ceiling, when the route publishes one.

    A limit and not just `accepts`, because it is what lets an over-sized block
    degrade to a pointer here rather than be rejected at the provider. A pixel
    ceiling belongs beside it and lands with `media-transform` (P7-02), the row
    that would resize to it — declaring one now would be a knob wired to
    nothing, which is the shape this codebase declines to ship."""


class LlmAdapter(Protocol):
    """The one required method is `stream`; `resolve_model` is optional."""

    def stream(self, options: GenerateOptions) -> AsyncIterator[Any]: ...


@dataclass(slots=True)
class AdapterHandle:
    """One adapter registration, and the routes it claims."""

    adapter: LlmAdapter
    by: Running
    """Who registered it (P6-29). An adapter's `stream` is row code this registry
    invokes, and it ran unbound — the same category as a tool's `execute`.

    Missed by P6-29 because both P6-30 walks look for a `Callable`: an adapter is
    an *object satisfying a Protocol*, so `AdapterHandle.adapter: LlmAdapter`
    names none, and this registers without a `claim_*` helper at all. Resolved
    from the ambient binding rather than a `scope=`, because `register_adapter`
    has never taken one — a row calling it from its own `apply` is the only
    caller shape there is, and that is exactly what `running_for(None)` reads."""
    providers: tuple[str, ...]
    _runtime: LlmRuntime
    _disposed: bool = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._runtime._registrations.remove(self)
        self._runtime._reindex()


@dataclass(slots=True)
class LlmRuntime:
    """The service published as `ctx.llm`."""

    ctx: Context
    _registrations: list[AdapterHandle] = field(default_factory=list)
    _routes: dict[str, AdapterHandle] = field(default_factory=dict)

    def register_adapter(self, providers: Sequence[str], adapter: LlmAdapter) -> AdapterHandle:
        """Claim one or more provider routes for `adapter`.

        The caller's scope owns the handle's `dispose`, so unloading the plugin
        unregisters the routes.
        """
        handle = AdapterHandle(
            adapter=adapter,
            by=self.ctx.running_for(),
            providers=tuple(providers),
            _runtime=self,
        )
        self._registrations.append(handle)
        self._reindex()
        return handle

    def _reindex(self) -> None:
        self._routes = {
            provider: handle for handle in self._registrations for provider in handle.providers
        }
        self.ctx.emit("llm/adapters-updated")

    def list_providers(self) -> list[str]:
        return sorted(self._routes)

    def adapter_for(self, provider: str) -> LlmAdapter:
        return self._route(provider).adapter

    def _route(self, provider: str) -> AdapterHandle:
        """The whole registration, for the two paths that call into the adapter.

        `adapter_for` stays the public spelling — one out-of-package caller reads
        it — and the binding needs what it discards, which is who registered the
        thing it returns.
        """
        handle = self._routes.get(provider)
        if handle is None:
            raise LlmError(f'no adapter is registered for provider "{provider}"', "NO_ADAPTER")
        return handle

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        """Ask the owning adapter what it knows about one exact route."""
        try:
            handle = self._route(provider)
        except LlmError:
            return ResolvedModel()
        resolver: Callable[..., Any] | None = getattr(handle.adapter, "resolve_model", None)
        if resolver is None:
            return ResolvedModel()
        with running(handle.by):
            resolved = resolver(provider, model)
        return resolved if isinstance(resolved, ResolvedModel) else ResolvedModel()

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        """Dispatch one request through the `llm/stream` waterfall."""

        async def inner(request: GenerateOptions) -> AsyncIterator[Any]:
            handle = self._route(request.provider)
            with running(handle.by):
                return _normalized(handle.adapter.stream(request), request)

        result = await self.ctx.waterfall("llm/stream", options, inner=inner)
        return result  # type: ignore[no-any-return]


async def _normalized(source: AsyncIterator[Any], request: GenerateOptions) -> AsyncIterator[Any]:
    """Turn an adapter raise into a terminal `finish{error}`.

    The loop's contract is that a stream always ends with a finish. Without this
    the loop would need a second failure path, and `agent/request-error` would
    not see provider failures uniformly.
    """
    try:
        async for chunk in source:
            yield chunk
    except Exception as error:
        failure = (
            error.failure
            if isinstance(error, LlmError)
            else LlmFailure(message=str(error) or type(error).__name__, code="UNKNOWN")
        )
        log.debug("ph.llm: adapter for %s failed: %s", request.provider, failure.message)
        yield Finish(reason=FinishReason(kind="error", failure=failure))


@plugin("llm")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the model adapter seam."""
    ctx.provide("llm", LlmRuntime(ctx=ctx))
