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

from ..cordis import Context, Disposer, events, maybe_await, plugin
from ..llm.types import ContextSnapshotSection, ToolSchema

__all__ = [
    "AssembleContext",
    "PromptAssembly",
    "PromptContext",
    "PromptSection",
    "PromptText",
    "SystemPromptService",
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


@dataclass(slots=True)
class _Registration:
    owner: Context
    value: Any


@dataclass(slots=True)
class SystemPromptService:
    """The service published as `ctx.system_prompt`."""

    ctx: Context
    _sections: list[_Registration] = field(default_factory=list)
    _contexts: list[_Registration] = field(default_factory=list)
    _tools: list[_Registration] = field(default_factory=list)
    _variables: list[_Registration] = field(default_factory=list)

    def _register(self, bucket: list[_Registration], owner: Context, value: Any) -> Disposer:
        entry = _Registration(owner=owner, value=value)
        bucket.append(entry)

        def off() -> None:
            if entry in bucket:
                bucket.remove(entry)

        return owner.add_disposer(off, label="system-prompt")

    def section(self, section: PromptSection, *, scope: Context | None = None) -> Disposer:
        return self._register(self._sections, scope or self.ctx, section)

    def context(self, context: PromptContext, *, scope: Context | None = None) -> Disposer:
        return self._register(self._contexts, scope or self.ctx, context)

    def tools(
        self, provider: Callable[[Context], list[ToolSchema]], *, scope: Context | None = None
    ) -> Disposer:
        """Contribute tool schemas.

        The provider receives the *target* scope, because what a tool set
        contains is a per-agent question: a restriction or a scoped
        registration changes the answer (B7).
        """
        return self._register(self._tools, scope or self.ctx, provider)

    def variable(
        self, name: str, provider: Callable[[], str], *, scope: Context | None = None
    ) -> Disposer:
        return self._register(self._variables, scope or self.ctx, (name, provider))

    def _visible(self, bucket: list[_Registration], target: Context) -> list[Any]:
        # One visibility rule, shared with event dispatch: a global
        # registration reaches every agent, an agent-scoped one reaches that
        # agent alone. Ordering within a bucket stays registration order, which
        # the `order` field then sorts.
        return [entry.value for entry in bucket if entry.owner.reaches(target)]

    async def assemble(self, request: AssembleContext | None = None) -> PromptAssembly:
        """Collect, order, interpolate, then run the assemble waterfall."""
        request = request or AssembleContext()
        target = request.scope or self.ctx
        # The request a provider is handed always names the scope being
        # assembled, even when the caller left it implicit — so no provider has
        # to repeat the `request.scope or ctx` fallback.
        scoped = request if request.scope is target else replace(request, scope=target)

        variables: dict[str, str] = {}
        for name, provider in self._visible(self._variables, target):
            variables[name] = provider()

        async def resolve(text: PromptText) -> str:
            raw = await maybe_await(text(scoped)) if callable(text) else text
            return _VARIABLE.sub(lambda m: variables.get(m.group(1), m.group(0)), raw)

        sections = sorted(self._visible(self._sections, target), key=lambda s: (s.order, s.name))
        complete = next((s for s in sections if s.complete), None)
        # Empty means absent, decided here rather than in each renderer: a
        # section opts out per-assembly by returning "" — the only mechanism that
        # can answer a per-agent question, since a row registers once — and a
        # consumer enumerating section names should not see the ones that did.
        rendered: tuple[tuple[str, str], ...]
        if complete is not None:
            rendered = ((complete.name, await resolve(complete.text)),)
        else:
            rendered = tuple(
                [
                    (section.name, body)
                    for section in sections
                    if (body := await resolve(section.text)).strip()
                ]
            )

        contexts = sorted(self._visible(self._contexts, target), key=lambda c: (c.order, c.name))
        materialized = tuple(
            [
                ContextSnapshotSection(name=context.name, text=body)
                for context in contexts
                if (body := await resolve(context.text)).strip()
            ]
        )

        schemas: list[ToolSchema] = []
        for provider in self._visible(self._tools, target):
            schemas.extend(provider(target))

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
