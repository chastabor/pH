"""`ctx.system_prompt` — four registration kinds, one assembly.

The split that matters is **section vs context** (A12):

* a `section` is static and part of the cached prefix;
* a `context()` is dynamic but cache-safe — it is materialized as a durable
  user-role `snapshot` message placed *after* retained history, and only when
  its text changed.

That is how time, workspace state and goal context reach the model without
busting the prefix on every turn. A plugin that puts changing text in a
`section` silently doubles the bill; the two names exist so it does not have to
find that out from an invoice.

@module ph.system_prompt.assembly
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from ..cordis import Context, Disposer, Running, events, maybe_await, plugin, running
from ..llm.types import ContextSnapshotSection, ToolSchema
from ..seams._registry import claim_entry

__all__ = [
    "AssembleContext",
    "PromptAssembly",
    "PromptContext",
    "PromptSection",
    "PromptText",
    "SystemPromptService",
    "ToolsProvider",
    "apply",
    "join_context_sections",
    "render_context_sections",
    "render_prompt",
]


_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")

PromptText: TypeAlias = "str | Callable[[AssembleContext], str | Awaitable[str]]"
"""A section's body: literal text, or a provider given the whole request.

The **request**, not just the scope: a section that needs the agent — the RLM
child doctrine, which applies only below depth 0 — would otherwise have to
recover it from `scope.get("agent")`, which means a bundle in another package
knowing how `ph.agent.registry` provisions it, and silently rendering nothing
whenever an assembly runs outside an agent scope. `AssembleContext` already
carries both; handing it over costs one parameter.

The provider may be async. `assemble` is already a coroutine, and a section that
has to ask a seam a question — Code Mode listing the namespaces a program can
actually reach, a workspace section naming the tier it actually got — would
otherwise have to answer it somewhere that cannot await, which in practice means
answering it from a stale copy.

One consequence to know: `assemble` now yields to the event loop between
sections, so the render is no longer atomic — a registration landing mid-await
can produce a prompt mixing two registry generations. Providers are pure
projections of current state, so this is benign today; a provider that is not
pure is wrong for a deeper reason than atomicity."""

ORDER_HARNESS_IDENTITY = -100
ORDER_DEPLOYMENT_PERSONA = 0
ORDER_PLAN_POLICY = 50
ORDER_TOOL_GUIDANCE = 100
"""The ordering convention, stated once so rows do not invent their own."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """A static contribution to the cached system-prompt prefix."""

    name: str
    text: PromptText
    order: int = ORDER_DEPLOYMENT_PERSONA
    complete: bool = False
    """When true this section is the *sole* prompt — everything else is dropped."""


@dataclass(frozen=True, slots=True)
class PromptContext:
    """A dynamic contribution materialized after retained history."""

    name: str
    text: PromptText
    order: int = 0


@dataclass(frozen=True, slots=True)
class AssembleContext:
    """What assembly is being run for."""

    scope: Context | None = None
    agent: Any = None


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    """The ordered result of one assembly."""

    sections: tuple[tuple[str, str], ...] = ()
    contexts: tuple[ContextSnapshotSection, ...] = ()
    tools: tuple[ToolSchema, ...] = ()
    variables: dict[str, str] = field(default_factory=dict)


events.declare(
    "system-prompt/assemble",
    "waterfall",
    PromptAssembly,
    owner="ph.system_prompt",
    doc="Wraps prompt assembly; a listener may rewrite or replace the assembly.",
)


def render_prompt(assembly: PromptAssembly) -> str:
    """The `system` string logged in `request/header`."""
    return "\n\n".join(text for _, text in assembly.sections if text.strip())


def render_context_sections(assembly: PromptAssembly) -> list[ContextSnapshotSection]:
    return [section for section in assembly.contexts if section.text.strip()]


def join_context_sections(sections: list[ContextSnapshotSection]) -> str:
    return "\n\n".join(section.text for section in sections)


ToolsProvider: TypeAlias = "Callable[[Context], list[ToolSchema]]"
"""What `SystemPromptService.tools` contributes. Named so the bucket holding it
can say so — an unnamed `Callable[...]` inside a `list[_Registration[...]]` is
the point at which a reader stops reading the type."""

_Variable: TypeAlias = "tuple[str, Callable[[], str]]"
"""What `SystemPromptService.variable` contributes: the name and its provider,
paired at registration so the bucket holds one thing per registration.

Underscored, unlike `ToolsProvider` beside it, because nothing outside this
module names it — `variable(name, provider)` takes the two apart and only the
bucket ever sees them paired. A `PromptSection | PromptContext` union got no
alias at all for the same reason one step further: it had a single use site and
reads in place."""


@dataclass(frozen=True, slots=True)
class _Registration[T]:
    value: T
    """The contribution itself.

    **Parameterised rather than `Any`** — the buckets below are four different
    kinds and this is the one type standing in for all of them, so `Any` meant
    `assemble`'s `entry.value.order`, `.name`, `.text` and `.complete` all
    type-checked as nothing. It cost nothing while `_visible` returned the bare
    values and the loops read `section.name`, because that was `Any` too; it
    started costing when P6-29 made the *registration* the unit every consumer
    handles, which is exactly when the field became load-bearing at four call
    sites instead of zero.

    Frozen, so the parameter infers covariant and `_Registration[PromptSection]`
    satisfies the `_Registration[PromptSection | PromptContext]` that `resolve`
    takes."""
    by: Running
    """Both answers, kept together (P6-29).

    P6-12 found the two questions fused here and in four sibling registries, and
    split them: `by.layer` is what `_visible`'s `reaches` filters on, `by.owner`
    is what the disposer hangs from. This field held only the first, under the
    name `visible_to`, which was honest but incomplete — the owner was resolved
    and spent in the same expression and never kept, so when P6-29 came to bind
    a section's body as the row that wrote it, the answer was gone. Keeping the
    pair is what lets a contribution be *invoked* correctly rather than only
    filtered and released correctly."""


@dataclass(slots=True)
class SystemPromptService:
    """The service published as `ctx.system_prompt`."""

    ctx: Context
    _sections: list[_Registration[PromptSection]] = field(default_factory=list)
    _contexts: list[_Registration[PromptContext]] = field(default_factory=list)
    _tools: list[_Registration[ToolsProvider]] = field(default_factory=list)
    _variables: list[_Registration[_Variable]] = field(default_factory=list)

    def _register[T](
        self, bucket: list[_Registration[T]], scope: Context | None, value: T
    ) -> Disposer:
        """Contribute to a bucket, and hand back the disposer that withdraws it.

        **Two contexts, because the owner was answering two questions** (P6-12).
        `_Registration.owner` feeds `_visible`'s `reaches` — that is *who sees
        this section* — while `add_disposer` decides *when it goes away*. Both
        were `scope or self.ctx`, so a row's section outlived the row: I2 held
        only where a caller remembered `scope=`.

        **Both are kept now, not just spent** (P6-29). Four of this file's
        buckets hold a *body* — a section's `text`, a context's, a variable's
        provider, a tool provider — which `assemble` invokes later, and it ran
        them unbound: anything they registered landed on the seam and outlived
        the row. Binding needs the owner at invoke time, which means recording it
        here rather than resolving it and dropping it.

        The visibility target is unchanged and the lifetime is now the
        activating row's. For a globally mounted row the two agree anyway — an
        activation scope inherits its parent's isolation, so a root row's
        section reaches everything either way — and keeping them separate is
        what makes that true by construction rather than by inspection.

        Through `claim_entry` rather than a hand-rolled `bucket.remove`, and it
        was **required rather than tidier**: `@dataclass(slots=True)` defaults to
        `eq=True`, so `_Registration` compares by *value*, and two rows
        contributing an equal section would have had one disposal take the
        other's. An earlier draft of this docstring claimed the opposite — that
        the old code compared by identity and was accidentally right — which is
        exactly the mistake `_registry`'s own docstring was written to stop
        someone making.
        """
        by = self.ctx.running_for(scope)
        entry = _Registration(value=value, by=by)
        return claim_entry(by.owner, bucket, entry, label="system-prompt")

    def section(self, section: PromptSection, *, scope: Context | None = None) -> Disposer:
        return self._register(self._sections, scope, section)

    def context(self, context: PromptContext, *, scope: Context | None = None) -> Disposer:
        return self._register(self._contexts, scope, context)

    def tools(self, provider: ToolsProvider, *, scope: Context | None = None) -> Disposer:
        """Contribute tool schemas.

        The provider receives the *target* scope, because what a tool set
        contains is a per-agent question: a restriction or a scoped
        registration changes the answer (B7).
        """
        return self._register(self._tools, scope, provider)

    def variable(
        self, name: str, provider: Callable[[], str], *, scope: Context | None = None
    ) -> Disposer:
        return self._register(self._variables, scope, (name, provider))

    def _visible[T](
        self, bucket: list[_Registration[T]], target: Context
    ) -> list[_Registration[T]]:
        # One visibility rule, shared with event dispatch: a global
        # registration reaches every agent, an agent-scoped one reaches that
        # agent alone. Ordering within a bucket stays registration order, which
        # the `order` field then sorts.
        #
        # Returns the *registrations*, not their values: every consumer below
        # invokes the value as a body, and needs `by.owner` to bind it (P6-29).
        return [entry for entry in bucket if entry.by.layer.reaches(target)]

    async def assemble(self, request: AssembleContext | None = None) -> PromptAssembly:
        """Collect, order, interpolate, then run the assemble waterfall."""
        request = request or AssembleContext()
        # `request.scope or self.ctx` is P6-32's staged remainder, stated here
        # per §5 rule 6: an unstated boundary still assembles the mount's
        # prompt. The row names this conversion; it is deferred, not exempt.
        target = request.scope or self.ctx
        # The request a provider is handed always names the scope being
        # assembled, even when the caller left it implicit — so no provider has
        # to repeat the `request.scope or ctx` fallback.
        scoped = request if request.scope is target else replace(request, scope=target)

        # **Every body below runs as the row that contributed it, for the scope
        # being assembled** (P6-29). Four buckets, four bodies, and all four ran
        # unbound: a variable provider, a section's `text`, a context's, a tool
        # provider. They are the same category as a tool's `execute` — a row's
        # code a registry invokes — and they are the four P6-26 could not reach
        # while the binding held one `Context`, because `target` here is the
        # *layer* and the owner is still whoever registered.
        variables: dict[str, str] = {}
        for variable in self._visible(self._variables, target):
            name, read = variable.value
            with running(variable.by, target):
                variables[name] = read()

        async def resolve(entry: _Registration[PromptSection | PromptContext]) -> str:
            # The registration, not two projections of it: `resolve(a.value.text,
            # b.by.owner)` was silently valid, which is the mis-pairing the pair
            # exists to prevent, reintroduced one function down (P6-29).
            text: PromptText = entry.value.text
            with running(entry.by, target):
                raw = await maybe_await(text(scoped)) if callable(text) else text
            # Interpolation is this seam's own work, not the row's, so it is
            # deliberately outside the binding.
            return _VARIABLE.sub(lambda m: variables.get(m.group(1), m.group(0)), raw)

        sections = sorted(
            self._visible(self._sections, target),
            key=lambda one: (one.value.order, one.value.name),
        )
        complete = next((one for one in sections if one.value.complete), None)
        # Empty means absent, decided here rather than in each renderer: a
        # section opts out per-assembly by returning "" — the only mechanism that
        # can answer a per-agent question, since a row registers once — and a
        # consumer enumerating section names should not see the ones that did.
        rendered: tuple[tuple[str, str], ...]
        if complete is not None:
            rendered = ((complete.value.name, await resolve(complete)),)
        else:
            rendered = tuple(
                [
                    (section.value.name, body)
                    for section in sections
                    if (body := await resolve(section)).strip()
                ]
            )

        contexts = sorted(
            self._visible(self._contexts, target),
            key=lambda one: (one.value.order, one.value.name),
        )
        materialized = tuple(
            [
                ContextSnapshotSection(name=context.value.name, text=body)
                for context in contexts
                if (body := await resolve(context)).strip()
            ]
        )

        schemas: list[ToolSchema] = []
        for provider in self._visible(self._tools, target):
            with running(provider.by, target):
                schemas.extend(provider.value(target))

        assembly = PromptAssembly(
            sections=rendered,
            contexts=materialized,
            tools=tuple(schemas),
            variables=variables,
        )

        async def inner(candidate: PromptAssembly, _request: AssembleContext) -> PromptAssembly:
            return candidate

        result = await self.ctx.waterfall("system-prompt/assemble", assembly, request, inner=inner)
        return result if isinstance(result, PromptAssembly) else assembly


@plugin("system-prompt")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the system-prompt assembly seam."""
    ctx.provide("system_prompt", SystemPromptService(ctx=ctx))
