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
from ..keys import LLM
from .types import Finish, FinishReason, GenerateOptions, LlmFailure, StreamChunk

__all__ = [
    "AdapterHandle",
    "LlmAdapter",
    "LlmError",
    "LlmRuntime",
    "MediaRoute",
    "ResolvedModel",
    "apply",
    "resolved",
]

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
    degrade to a pointer here rather than be rejected at the provider."""
    max_image_edge: int | None = None
    """Longest edge in pixels the route will *accept*, when it publishes one.

    This docstring's neighbor used to say a pixel ceiling had to wait for
    `media-transform` (P7-02) because "declaring one now would be a knob wired to
    nothing". Two things wired it (P7-03): `ph.llm.dimensions` reads an image's
    size from its header with no dependency, and a limit does not need a resizer
    to be worth declaring — refusing with a sentence the model can read is an
    action, and it is the same one `max_attachment_bytes` already takes."""
    structured_output: bool = False
    """Whether this route enforces `GenerateOptions.response_schema` on the wire.

    Declared rather than assumed, and defaulting to *no*, for the reason `accepts`
    does: the failure it prevents is a caller believing a shape was guaranteed
    when the provider ignored the field. OpenAI-compatible endpoints take
    `response_format: {"type": "json_schema", …}`; Anthropic has no equivalent
    today, so a caller there gets the instruction and the validation and not the
    guarantee — which is a real difference and one `structured` says out loud
    rather than papering over."""
    usable_image_edge: int | None = None
    """Longest edge the route can actually *use*, when it is smaller than it accepts.

    Providers scale a large image down before the model sees it — Anthropic at
    1568 px, and it accepts up to 8000 — so pixels above this are uploaded, paid
    for, and discarded at the far end. Separate from `max_image_edge` because the
    outcomes differ in kind: over that, nothing is sent; over this, everything
    works and a person is quietly overpaying, which is the failure that lasts for
    a whole session because nothing announces it."""
    credential: str | None = None
    """The credential this route resolves at the adapter edge, **by name** (I-3) —
    the `apiKeyEnv` its row names — or `None` for a route that needs none.

    What lets anything above the edge ask "can this route run here, now?" without
    holding a value: the resume check (T5) reads it and asks `ctx.credentials.has`,
    so a session whose key is missing waits for it by name rather than starting and
    failing at its first request. Checked against the mounted profile's adapters,
    never a name copied into the log, because a profile that renamed the variable
    must be asked about the new name."""


class MediaRoute(Protocol):
    """The route facts a config declares and `ResolvedModel` carries.

    Read-only members on purpose: the three configs disagree about the *types* —
    Anthropic states a pixel edge as `int` where the other two allow `None` — and
    a property is covariant where a mutable attribute would not be, so each may
    narrow its own field and still satisfy this.

    Declared rather than duck-typed, which is what makes `resolved` below more
    than a shortcut: a config that grows out of step with `ResolvedModel` fails at
    the call site instead of quietly reporting a default.

    **Beside `ResolvedModel` rather than beside the adapters that satisfy it.**
    Its member list *is* that dataclass's field list, so the two have to move
    together — and living in an application package's private module would have
    meant a third-party adapter importing `ph_app.adapters._media` or writing the
    seventh copy of the projection, which is the drift this exists to stop. It is
    also the only side of the boundary where the invariant can be checked at all:
    a rename here and a matching edit there cannot pass one suite and fail the
    other.
    """

    @property
    def context_window(self) -> int | None: ...
    @property
    def default_max_tokens(self) -> int | None: ...
    @property
    def accepts(self) -> tuple[str, ...]: ...
    @property
    def max_attachment_bytes(self) -> int | None: ...
    @property
    def max_image_edge(self) -> int | None: ...
    @property
    def usable_image_edge(self) -> int | None: ...


def resolved(
    route: MediaRoute, *, structured_output: bool, credential: str | None
) -> ResolvedModel:
    """One route's config as the `ResolvedModel` every layer above reads.

    **The projection, not the values.** What a route accepts, how large a file it
    takes and what it does with pixels are per-provider facts that stay in each
    config with the paragraph arguing them; what was copied three times is the
    *field list* — six lines mapping one name onto the same name — and that list
    is `ResolvedModel`'s vocabulary rather than any adapter's.

    The cost of the copies was a seventh capability being three edits, where an
    adapter that missed one reports the default instead of the route's number and
    `media-degrade` then applies the wrong rule for that provider alone. Now it is
    one edit here, and a config that has not caught up fails to type-check.

    `structured_output` is passed rather than read, because it is the one field
    that is not a config value: it is a claim about the *wire*, argued at each
    call site, and a route that declared it in config could promise a guarantee
    its adapter does not implement. `credential` is passed for the plainer reason
    that it is not a media fact: it is the name the adapter resolves at its edge,
    and **required** so an adapter cannot forget to say (T5).
    """
    return ResolvedModel(
        context_window=route.context_window,
        default_max_tokens=route.default_max_tokens,
        accepts=frozenset(route.accepts),
        max_attachment_bytes=route.max_attachment_bytes,
        max_image_edge=route.max_image_edge,
        usable_image_edge=route.usable_image_edge,
        structured_output=structured_output,
        credential=credential,
    )


class LlmAdapter(Protocol):
    """The one required method is `stream`; `resolve_model` is optional."""

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]: ...


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

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        """Dispatch one request through the `llm/stream` waterfall."""

        async def inner(request: GenerateOptions) -> AsyncIterator[StreamChunk]:
            handle = self._route(request.provider)
            # The binding goes *inside* the generator (C10). Both
            # `adapter.stream(...)` and `_normalized(...)` are async-generator
            # calls, so this line only constructs them — nothing of the adapter
            # ran here, and the whole stream used to be pulled later, outside any
            # `running`. Every registration the adapter made mid-stream landed on
            # the seam instead of the row, and outlived it.
            return _normalized(handle.adapter.stream(request), request, handle.by)

        return await self.ctx.waterfall("llm/stream", options, inner=inner)


async def _normalized(
    source: AsyncIterator[StreamChunk], request: GenerateOptions, by: Running
) -> AsyncIterator[StreamChunk]:
    """Turn an adapter raise into a terminal `finish{error}`.

    The loop's contract is that a stream always ends with a finish. Without this
    the loop would need a second failure path, and `agent/request-error` would
    not see provider failures uniformly.

    **`by` is entered around each pull rather than around construction** (C10).
    An adapter's body is row code this seam drives, so it runs as an effect of
    the row that registered it (P6-25) — and a generator's body runs at
    `__anext__`, on whatever task is consuming it. The binding therefore wraps
    the *await*, not the `__anext__()` call that merely builds the coroutine,
    which is the same mistake one level down. Entered and left per chunk rather
    than held across the `yield`, because a binding held across a suspension is
    released on whichever task resumes it — `_hooks`' reason for returning a
    list rather than a generator.

    **`running` and not the inline `_ACTIVATING.set/reset` `_invoke` uses**, and
    the difference between the two is a layer rather than a preference. That one
    is inside cordis and may touch its own ContextVar; reaching for it from here
    would be a private import across two packages to save a measured 291 ns per
    chunk — 0.3 ms on a thousand-chunk stream, against a stream that takes
    seconds. `_invoke`'s note about the context manager's own frame is about a
    path with no such boundary to cross.
    """
    pull = source.__anext__
    try:
        while True:
            with running(by):
                try:
                    chunk = await pull()
                except StopAsyncIteration:
                    return
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
async def apply(ctx: Context, config: None) -> None:
    """Mount the model adapter seam."""
    ctx.provide(LLM, LlmRuntime(ctx=ctx))
