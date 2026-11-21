"""`ctx.system_prompt`: four kinds of registration, one assembly.

The split that matters is **section and context** (A12):

* A `section` is static and is part of the cached prefix.
* A `context()` is dynamic but cache-safe. It is materialized as a durable
  user-role `snapshot` message that is placed *after* retained history, and only
  when its text changed.

That is how the time, the workspace state and the goal context reach the model
without a change to the prefix on every turn. A plugin that puts changing text
in a `section` silently doubles the bill. The two names exist so that the author
of a plugin does not have to learn that from an invoice.

@module ph.system_prompt.assembly
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from ..agent.types import AgentHandle
from ..cordis import (
    Boundary,
    Context,
    Disposer,
    MaybeAwaitable,
    Running,
    boundary_of,
    events,
    maybe_await,
    plugin,
    running,
    settled,
)
from ..keys import SYSTEM_PROMPT
from ..llm.types import (
    ContextSnapshotSection,
    Message,
    PluginSource,
    ToolSchema,
    create_user_message,
    text_of,
)
from ..seams._registry import claim_entry
from ..session import Session

__all__ = [
    "CONTEXT_PLUGIN",
    "AssembleContext",
    "PromptAssembly",
    "PromptContext",
    "PromptSection",
    "PromptText",
    "SystemPromptService",
    "ToolsProvider",
    "apply",
    "context_message",
    "is_context_snapshot",
    "join_context_sections",
    "render_context_sections",
    "render_prompt",
]


_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")

PromptText: TypeAlias = "str | Callable[[AssembleContext], MaybeAwaitable[str]]"
"""The body of a section: literal text, or a provider that is given the whole
request.

The provider gets the **request**, and not only the scope. A section may need
the agent: the RLM child doctrine applies only below depth 0. With only the
scope, such a section would have to recover the agent from `scope.get(AGENT)`.
A bundle in another package would then have to know how `ph.agent.registry`
provisions the agent, and the section would silently render nothing whenever an
assembly runs outside an agent scope. `AssembleContext` already carries both,
and to pass it costs one parameter.

The provider may be async. `assemble` is already a coroutine. A section may have
to ask a seam a question: Code Mode lists the namespaces that a program can
actually reach, and a workspace section names the tier that it actually got.
Such a section would otherwise have to answer somewhere that cannot await, which
in practice means an answer from a stale copy.

One consequence: `assemble` now yields to the event loop between sections, so
the render is no longer atomic. A registration that lands during an await can
produce a prompt that mixes two generations of the registry. Providers are pure
projections of current state, so this is harmless today. A provider that is not
pure is wrong for a deeper reason than atomicity."""

ORDER_HARNESS_IDENTITY = -100
ORDER_DEPLOYMENT_PERSONA = 0
ORDER_PLAN_POLICY = 50
ORDER_TOOL_GUIDANCE = 100
"""The ordering convention, stated in one place so that rows do not invent their
own."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """A static contribution to the cached system-prompt prefix."""

    name: str
    text: PromptText
    order: int = ORDER_DEPLOYMENT_PERSONA
    complete: bool = False
    """When true, this section is the *sole* prompt and everything else is
    dropped."""


@dataclass(frozen=True, slots=True)
class PromptContext:
    """A dynamic contribution materialized after retained history."""

    name: str
    text: PromptText
    order: int = 0


@dataclass(frozen=True, slots=True)
class AssembleContext:
    """What assembly is being run for."""

    scope: Context
    """The **resolved** boundary that this prompt is assembled for.

    The type is `Context`, never `Boundary`. This payload is what a *provider*
    receives, not what a caller states. `assemble` takes the `Boundary` and
    narrows it one time. A provider therefore gets a real scope, and the
    `scope.get(AGENT)` shape that `PromptText` documents type-checks. When the
    field was typed `Boundary`, it did not: mypy reported
    `Item "Deployment" … has no attribute "get"` against the pattern that the
    docstring of that alias gives as what a provider writes.

    The tools pipeline makes the same split for the same reason.
    `ToolExecutionInput.scope` is a `Boundary` because a caller builds it.
    `ToolExecution.scope` is a `Context` because the pipeline passes it on. The
    input states the boundary, and everything downstream carries the scope."""
    agent: AgentHandle | None = None

    @property
    def session(self) -> Session | None:
        """The session behind the agent, when there is an agent.

        Two optionals answer one question. Five readers across three packages
        each wrote `request.agent.session if request.agent is not None else None`.
        That repeats the derivation and not the fact. `CommandContext` pairs the
        two in the same way.
        """
        return self.agent.session if self.agent is not None else None


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


CONTEXT_PLUGIN = "ph.system-prompt"
"""The name that a context snapshot carries. It is written and read in this
module and nowhere else."""


def context_message(assembly: PromptAssembly, shown: Sequence[Message]) -> Message | None:
    """The `context()` providers as a message, or `None` when `shown` already has
    them.

    This is the reason that `context()` exists apart from `section`. To send
    unchanged context again on every step would invalidate the cached prefix on
    each turn (A12).

    **"Already has" is read from `shown`, the conversation that the model is
    sent. It is not held by the caller** (C12). A snapshot that the caller kept
    was a promise that the model had been told, and every route around delivery
    broke the promise: a rejected step (C3), a cancel before the append, a
    resume that starts with nothing, and a compaction that shadowed the message.
    Here, a step that never logged its message sends it again, and a summary
    that shadowed the message brings it back.

    The function is beside `render_context_sections` because the tag belongs to
    this module. The module that writes a snapshot is the module that knows how
    to find it again.
    """
    sections = render_context_sections(assembly)
    if not sections:
        return None
    text = join_context_sections(sections)
    if text == _shown_context(shown):
        return None
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source=PluginSource(plugin=CONTEXT_PLUGIN, form="snapshot", sections=list(sections)),
    )


def is_context_snapshot(message: Message) -> bool:
    """Whether `message` is a context snapshot that this module wrote. This is
    the one question that another package may ask about `CONTEXT_PLUGIN`. It is
    answered here so that the name stays written and read in this module alone."""
    source = message.source
    return (
        isinstance(source, PluginSource)
        and source.plugin == CONTEXT_PLUGIN
        and source.form == "snapshot"
    )


def _shown_context(shown: Sequence[Message]) -> str | None:
    """The text of the newest snapshot in `shown`, or `None`. A later snapshot
    supersedes an earlier one."""
    for message in reversed(shown):
        if is_context_snapshot(message):
            return text_of(message.content)
    return None


ToolsProvider: TypeAlias = "Callable[[Context], list[ToolSchema]]"
"""What `SystemPromptService.tools` contributes. It has a name so that the
bucket that holds it can say so. An unnamed `Callable[...]` inside a
`list[_Registration[...]]` is the point at which a reader stops reading the
type."""

_Variable: TypeAlias = "tuple[str, Callable[[], str]]"
"""What `SystemPromptService.variable` contributes: the name and its provider.
They are paired at registration, so that the bucket holds one thing for each
registration.

The name has an underscore, unlike `ToolsProvider` beside it, because nothing
outside this module names it. `variable(name, provider)` takes the two apart,
and only the bucket ever sees them paired. A `PromptSection | PromptContext`
union got no alias for the same reason, one step further: it had a single use
site and is readable in place."""


@dataclass(frozen=True, slots=True)
class _Registration[T]:
    value: T
    """The contribution itself.

    **The type is a parameter and not `Any`.** The buckets below hold four
    different kinds, and this one type stands in for all of them. With `Any`,
    the `entry.value.order`, `.name`, `.text` and `.complete` of `assemble` were
    not type-checked at all. That cost nothing while `_visible` returned the
    bare values and the loops read `section.name`, because that was `Any` too.
    It started to cost when P6-29 made the *registration* the unit that every
    consumer handles. At that point the field became load-bearing at four call
    sites, where before it was at none.

    The class is frozen, so the parameter is inferred as covariant, and
    `_Registration[PromptSection]` satisfies the
    `_Registration[PromptSection | PromptContext]` that `resolve` takes."""
    by: Running
    """Both answers, kept together (P6-29).

    P6-12 found the two questions fused here and in four sibling registries,
    and split them. `by.layer` is what the `reaches` test of `_visible` filters
    on, and `by.owner` is what the disposer hangs from. This field used to hold
    only the first, under the name `visible_to`. That name was accurate but
    incomplete. The owner was resolved and used in the same expression and
    never kept. So when P6-29 came to bind the body of a section as the row that
    wrote it, the answer was gone. Because the pair is kept, a contribution can
    be *invoked* correctly, and not only filtered and released correctly."""


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
        """Contribute to a bucket, and return the disposer that withdraws the
        contribution.

        **Two contexts, because the owner answers two questions** (P6-12).
        `_Registration.by.layer` feeds the `reaches` test of `_visible`: *who
        sees this section*. `by.owner` takes the disposer (`add_disposer`),
        which decides *when the section goes away*. When both were written as
        `scope or self.ctx`, a section of a row outlived the row.

        **Both are kept, and not only used** (P6-29). All four buckets of this
        file hold a *body*: the `text` of a section, the `text` of a context,
        the provider of a variable, and a tool provider. `assemble` invokes the
        body later, and the binding needs the owner at that time. So the owner
        is recorded here, and not resolved and dropped.

        The visibility target is unchanged, and the lifetime is the lifetime of
        the activating row. For a globally mounted row the two agree anyway.
        Because they are kept separate, that is true by construction and not by
        inspection.

        The entry goes through `claim_entry` and not through a hand-written
        `bucket.remove`, and that is **required and not only tidier**.
        `@dataclass(slots=True)` defaults to `eq=True`, so `_Registration`
        compares by *value*. With `bucket.remove`, when two rows contribute an
        equal section, one disposal would take the entry of the other.
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

        The provider receives the *target* scope, because the content of a tool
        set is a question for each agent: a restriction or a scoped registration
        changes the answer (B7).
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
        # registration reaches every agent, and an agent-scoped registration
        # reaches that agent alone. The order within a bucket stays the
        # registration order, and the `order` field then sorts it.
        #
        # The method returns the *registrations*, not their values. Every
        # consumer below invokes the value as a body, and needs `by.owner` to
        # bind it (P6-29).
        return [entry for entry in bucket if entry.by.layer.reaches(target)]

    async def assemble(
        self, scope: Boundary, *, agent: AgentHandle | None = None
    ) -> PromptAssembly:
        """Collect, order, interpolate, then run the assemble waterfall.

        **The boundary is a parameter, and the payload is built here.** This is
        the one narrowing point, as `ToolRuntime._execution` is for a tool call.
        A caller states a `Boundary`, possibly `DEPLOYMENT`, and a provider
        reads a real `Context`. One object has one role, and it is the narrowed
        one. A `system-prompt/assemble` listener and a `PromptSection.text`
        therefore cannot see two different scopes for one assembly.
        """
        request = AssembleContext(scope=boundary_of(scope, self.ctx), agent=agent)
        target = request.scope

        # **Every body below runs as the row that contributed it, for the scope
        # that is being assembled** (P6-29). There are four buckets and four
        # bodies: a variable provider, the `text` of a section, the `text` of a
        # context, and a tool provider. All four ran unbound. They are the same
        # category as the `execute` of a tool: code of a row that a registry
        # invokes. P6-26 could not reach these four while the binding held one
        # `Context`, because `target` here is the *layer*, and the owner is
        # still whoever registered.
        variables: dict[str, str] = {}
        for variable in self._visible(self._variables, target):
            name, read = variable.value
            with running(variable.by, target):
                variables[name] = read()

        async def resolve(entry: _Registration[PromptSection | PromptContext]) -> str:
            # The registration is passed, not two projections of it.
            # `resolve(a.value.text, b.by.owner)` was silently valid. That is
            # the wrong pairing that the pair exists to prevent, and it came
            # back one function down (P6-29).
            text: PromptText = entry.value.text
            with running(entry.by, target):
                raw = await maybe_await(text(request)) if callable(text) else text
            # Interpolation is the work of this seam and not of the row, so it
            # is deliberately outside the binding.
            return _VARIABLE.sub(lambda m: variables.get(m.group(1), m.group(0)), raw)

        sections = sorted(
            self._visible(self._sections, target),
            key=lambda one: (one.value.order, one.value.name),
        )
        complete = next((one for one in sections if one.value.complete), None)
        # Empty means absent, and that is decided here and not in each renderer.
        # A section opts out of one assembly by returning "". That is the only
        # mechanism that can answer a question for each agent, because a row
        # registers one time. A consumer that enumerates section names must not
        # see the sections that opted out.
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
        return settled("system-prompt/assemble", result, PromptAssembly)


@plugin("system-prompt", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the system-prompt assembly seam."""
    ctx.provide(SYSTEM_PROMPT, SystemPromptService(ctx=ctx))
