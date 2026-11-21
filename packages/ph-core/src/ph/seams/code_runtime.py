"""`ctx.code_runtime`: the definition of the seam only (P1-06, C1).

This module ships the *contract*. One assertion in the contract is easy to miss
and expensive to omit:

> a provider that declares `persistence: "namespace"` **must** emit
> `kernel/snapshot` events. The seam checks that when the provider registers,
> and not the first time that someone forks a session and finds that the state
> was never durable.

This is D17, enforced by the seam and not left to convention. dsh withheld a
persistent Python REPL because "cross-call state would be invisible to the
log". pH admits one only from a provider that has promised, at registration, to
keep that state visible. A promise that was checked at run time would be found
by the user who lost work.

**Binding names are validated here too.** One `bindings` list must be valid
against every backend, whatever its `language`. So a name must be a portable
identifier and must avoid the reserved words of every shipped language. A name
that is legal in Python and reserved in TypeScript would make a set of bindings
silently specific to one backend.

@module ph.seams.code_runtime
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from ..cancel import CancelToken
from ..cordis import Context, Disposer, Running, plugin, running
from ..keys import CODE_RUNTIME
from ._registry import claim_key, claim_slot

__all__ = [
    "PORTABLE_NAME",
    "RESERVED_BINDING_NAMES",
    "CodeBinding",
    "CodeBindingNamespace",
    "CodeRunRequest",
    "CodeRunResult",
    "CodeRuntime",
    "CodeRuntimeSeam",
    "PersistenceObligationError",
    "apply",
    "validate_binding_name",
]

Isolation: TypeAlias = Literal["in-process", "thread", "process", "sandbox", "remote"]
Persistence: TypeAlias = Literal["none", "namespace"]

PORTABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TYPESCRIPT_RESERVED = frozenset(
    [
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "interface",
        "let",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    ]
)

RESERVED_BINDING_NAMES: frozenset[str] = frozenset(keyword.kwlist) | _TYPESCRIPT_RESERVED
"""Names that no binding may take, in any language that pH renders an SDK for.

This is the union and not the set of one language. One binding list must be
valid against every backend, so a name that is reserved anywhere is reserved
everywhere."""


class PersistenceObligationError(RuntimeError):
    """A `persistence: "namespace"` provider did not promise to snapshot."""


def validate_binding_name(name: str) -> None:
    """Assert that a binding name is a portable identifier and is not reserved."""
    if not PORTABLE_NAME.match(name):
        raise ValueError(
            f'binding name "{name}" is not a portable identifier ([A-Za-z_][A-Za-z0-9_]*)'
        )
    if name in RESERVED_BINDING_NAMES:
        raise ValueError(f'binding name "{name}" is reserved in a language pH targets')


@dataclass(frozen=True, slots=True)
class CodeBinding:
    """One governed callable that a program may await.

    `dispatch` is `None` when the namespace is being *described* (to render the
    SDK prompt) and not *bound* to a run. The description needs no closure over
    a live bridge.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    dispatch: Callable[..., Any] | None = None
    """Re-enters the tool pipeline as a sub-call (C1)."""
    counts_as_spawn: bool = False
    """The author of the namespace declares this when a call starts a child
    agent. The bridge can then hold the call to the spawn budget (C4) by a
    property, and does not guess from the name."""
    presents: str | None = None
    """The governed tool that this binding is the namespaced face of.

    It is set when a namespace presents a registered tool again under its own
    name (`rlm.run` for `rlm_run`). The tool stays dispatchable, and a policy
    row can still address it. It is only dropped from the `tools` listing, so
    the SDK offers one route to a capability and not two. The second route
    would have no prompt text that explains it. The suppression list is read
    from the bindings themselves, so it *is* the binding list and the two
    cannot drift."""


@dataclass(frozen=True, slots=True)
class CodeBindingNamespace:
    """A named group of bindings, as the program sees it (`tools.read(...)`)."""

    name: str
    bindings: tuple[CodeBinding, ...]
    description: str = ""

    def __post_init__(self) -> None:
        validate_binding_name(self.name)
        for binding in self.bindings:
            validate_binding_name(binding.name)


@dataclass(frozen=True, slots=True)
class CodeRunRequest:
    """One program to run, as one agent."""

    program: str
    agent: str
    """Whose boundary the run gets: the agent to whose workspace, confinement and
    lifetime a runtime binds the run (S21).

    The field is required, so nothing runs code as nobody. The H1 probe did
    that at one time. It ran from a namespace that no agent owned, and it got
    no workspace, no confinement, and the directory of the daemon itself for a
    module that the model had named."""
    bindings: tuple[CodeBindingNamespace, ...] = ()
    namespace: str | None = None
    """`None` keeps the fresh-per-run contract of dsh. A key selects a persistent
    namespace. A namespace other than the agent's own still runs within the
    boundary of the agent."""
    token: CancelToken | None = None
    """The caller's view of cancellation for this run.

    It is named `token` because a held `CancelToken` has that name everywhere
    else. `bridge.token` writes this field and the `token` of `Kernel.run`
    reads it, so the same word carries it the whole way. `tasks.cancel_scope`,
    in contrast, is a real `anyio.CancelScope` in several files here. See
    `ph.cancel` for why this is a token and not a scope.
    """


@dataclass(frozen=True, slots=True)
class CodeRunResult:
    """What one program produced."""

    logs: str = ""
    value: Any = None
    error: str | None = None
    truncated: bool = False
    dropped: int = 0
    """The bytes of output that the caps of the runtime threw away, counted to
    the end of the run. The marker in `logs` where a stream was cut carries no
    number, because a stream that is still running cannot know what will
    follow. `logs` ends with the total, and this field is the same total for a
    front end. It is `0` when nothing was dropped: a value that is too large to
    return sets `truncated` and drops no output. A run that was killed sends no
    `done`, so the share of the guest is not counted."""
    reset: bool = False
    """The runtime died since the last run, so this run got a fresh, empty
    namespace. The reset notice in `logs` tells the *model*, and this flag
    tells the card. A fact that was recovered from the prose of the notice
    could be forged by any program whose first output is the marker text."""
    displays: tuple[dict[str, Any], ...] = ()
    """Rich payloads that the program emitted: a plot, a table, a diff.

    They are carried on the result and not folded into `logs`, because they
    are for a front end and `logs` is for the model. A base64 PNG in the text
    that the model reads is expensive and tells the model nothing."""


@runtime_checkable
class CodeRuntime(Protocol):
    """A code-execution backend.

    `language`, `isolation` and `persistence` are read-only descriptors that a
    consumer branches on. `declares_kernel_snapshots` is the promise that the
    seam checks. It is declared here and not probed for, so a typed runtime
    that forgets it is caught at `register`, and not at the first persistent
    cell.

    **They are properties, because the type must also say "read-only
    descriptor".** When they were declared as plain attributes they were
    *settable*, which a checker reads as invariant and instance-level. So the
    real runtime did not satisfy the Protocol that it is registered against:
    its `language` and `isolation` are `ClassVar`s, and its `persistence` is
    the narrower `Literal["namespace"]`. The stub did not satisfy it either.
    `ctx.code_runtime` was `Any`, so `register(provider)` was never checked and
    nobody knew (plan P8-05).
    """

    @property
    def language(self) -> str: ...
    @property
    def isolation(self) -> Isolation: ...
    @property
    def persistence(self) -> Persistence: ...
    @property
    def declares_kernel_snapshots(self) -> bool: ...

    async def run(self, request: CodeRunRequest) -> CodeRunResult: ...


@dataclass(slots=True)
class CodeRuntimeSeam:
    """The service published as `ctx.code_runtime`.

    It holds at most one provider. Two answers to "what runs this program"
    would contradict each other, and a profile picks its tier (D16).
    """

    ctx: Context
    provider: CodeRuntime | None = None
    provider_by: Running | None = None
    """Who registered the runtime (P6-29). See `CompactionSeam.engine_by`."""
    _sdk_renderers: dict[str, Callable[[Sequence[CodeBindingNamespace]], str]] = field(
        default_factory=dict
    )

    def register(self, provider: CodeRuntime, *, scope: Context | None = None) -> Disposer:
        """Claim the runtime. This enforces the persistence obligation (D17)."""
        persistence = provider.persistence
        if (
            persistence == "namespace"
            # This is still a probe, deliberately. The Protocol is structural,
            # so an untyped provider reaches here without the attribute, and
            # `False` is what the error below is written to report about it.
            and getattr(provider, "declares_kernel_snapshots", False) is not True
        ):
            raise PersistenceObligationError(
                f"code runtime {type(provider).__name__} declares "
                'persistence="namespace" but does not declare that it emits '
                "kernel/snapshot events; cross-call state would be invisible to the "
                "log, which is the reason a persistent runtime was withheld (D17). "
                "Set declares_kernel_snapshots = True once the provider emits them."
            )
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "provider",
            provider,
            label="code_runtime",
        )

    def register_sdk_renderer(
        self,
        language: str,
        renderer: Callable[[Sequence[CodeBindingNamespace]], str],
        *,
        scope: Context | None = None,
    ) -> Disposer:
        """Claim the `tools:sdk` prompt renderer for one language (P1-04).

        The seam is the one place where a renderer lives. So a provider that
        registers a richer renderer and is then disposed leaves an *absence*,
        which is a loud failure at prompt assembly. It does not silently revert
        to a default, whose different text would invalidate the cached prefix
        (A12).
        """
        return claim_key(
            self.ctx.owner_for(scope),
            self._sdk_renderers,
            language,
            renderer,
            label="sdk-renderer",
        )

    def sdk_renderer(self, language: str) -> Callable[[Sequence[CodeBindingNamespace]], str] | None:
        return self._sdk_renderers.get(language)

    def require(self) -> CodeRuntime:
        if self.provider is None:
            raise RuntimeError("no ctx.code_runtime provider is registered; Code Mode requires one")
        return self.provider

    async def run(self, request: CodeRunRequest) -> CodeRunResult:
        provider = self.require()
        # Run as the row that registered the runtime (P6-29). The layer is that
        # of the registration, for the reason that `CompactionSeam.engine_by`
        # states.
        with running(self.provider_by):
            return await provider.run(request)


@plugin("code-runtime", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the definition of the code-runtime seam. No provider ships in `ph-base`."""
    ctx.provide(CODE_RUNTIME, CodeRuntimeSeam(ctx=ctx))
