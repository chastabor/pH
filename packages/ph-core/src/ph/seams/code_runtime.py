"""`ctx.code_runtime` — the seam definition only (P1-06, C1).

No provider ships in Phase 1. What ships is the *contract*, and one assertion
inside it that is easy to miss and expensive to omit:

> a provider declaring `persistence: "namespace"` **must** emit
> `kernel/snapshot` events, and that is checked when it registers — not the
> first time someone forks a session and discovers the state was never durable.

This is D17 enforced by the seam rather than left to convention. dsh withheld a
persistent Python REPL precisely because "cross-call state would be invisible to
the log"; pH admits one only from a provider that has promised, at registration,
to keep it visible. A promise checked at runtime would be discovered by the
person who lost work.

Binding names are also validated here: one `bindings` list has to be valid
against every backend regardless of `language`, so names must be portable
identifiers and must clear the reserved sets of every shipped language. A name
that is legal in Python and reserved in TypeScript would make a binding set
silently backend-specific.

@module ph.seams.code_runtime
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from ..cordis import Context, Disposer, plugin
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
"""Names no binding may take, in any language pH renders an SDK for.

The union rather than the per-language set: one binding list must be valid
against every backend, so a name reserved anywhere is reserved everywhere."""


class PersistenceObligationError(RuntimeError):
    """A `persistence: "namespace"` provider did not promise to snapshot."""


def validate_binding_name(name: str) -> None:
    """Assert a binding name is a portable identifier."""
    if not PORTABLE_NAME.match(name):
        raise ValueError(
            f'binding name "{name}" is not a portable identifier ([A-Za-z_][A-Za-z0-9_]*)'
        )
    if name in RESERVED_BINDING_NAMES:
        raise ValueError(f'binding name "{name}" is reserved in a language pH targets')


@dataclass(frozen=True, slots=True)
class CodeBinding:
    """One governed callable a program may await.

    `dispatch` is `None` when the namespace is being *described* (rendering the
    SDK prompt) rather than *bound* to a run — the description needs no closure
    over a live bridge.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    dispatch: Callable[..., Any] | None = None
    """Re-enters the tool pipeline as a sub-call (C1)."""
    counts_as_spawn: bool = False
    """Declared by the namespace author when a call starts a child agent, so the
    bridge can hold it to the spawn budget (C4) by property rather than by
    guessing from the name."""


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
    """One program to run."""

    program: str
    bindings: tuple[CodeBindingNamespace, ...] = ()
    namespace: str | None = None
    """`None` keeps dsh's fresh-per-run contract; a key selects a persistent one."""
    cancel_scope: Any = None


@dataclass(frozen=True, slots=True)
class CodeRunResult:
    """What one program produced."""

    logs: str = ""
    value: Any = None
    error: str | None = None
    truncated: bool = False
    reset: bool = False
    """The runtime died since the last run, so this one got a fresh, empty
    namespace. The reset notice in `logs` tells the *model*; this flag tells the
    card — a fact recovered from the notice's prose would be forged by any
    program whose first output is the marker text."""
    displays: tuple[dict[str, Any], ...] = ()
    """Rich payloads the program emitted — a plot, a table, a diff.

    Carried on the result rather than folded into `logs` because they are for a
    front-end and `logs` is for the model: a base64 PNG in the model's text costs
    a fortune and says nothing."""


@runtime_checkable
class CodeRuntime(Protocol):
    """A code-execution backend.

    `language`, `isolation` and `persistence` are read-only descriptors a
    consumer branches on; `declares_kernel_snapshots` is the promise the seam
    checks.
    """

    language: str
    isolation: Isolation
    persistence: Persistence

    async def run(self, request: CodeRunRequest) -> CodeRunResult: ...


@dataclass(slots=True)
class CodeRuntimeSeam:
    """The service published as `ctx.code_runtime`.

    Holds at most one provider: two answers to "what runs this program" is a
    contradiction, and a profile picks its tier (D16).
    """

    ctx: Context
    provider: CodeRuntime | None = None
    _sdk_renderers: dict[str, Callable[[Sequence[CodeBindingNamespace]], str]] = field(
        default_factory=dict
    )

    def register(self, provider: CodeRuntime, *, scope: Context | None = None) -> Disposer:
        """Claim the runtime. Enforces the persistence obligation (D6)."""
        persistence = getattr(provider, "persistence", "none")
        if (
            persistence == "namespace"
            and getattr(provider, "declares_kernel_snapshots", False) is not True
        ):
            raise PersistenceObligationError(
                f"code runtime {type(provider).__name__} declares "
                'persistence="namespace" but does not declare that it emits '
                "kernel/snapshot events; cross-call state would be invisible to the "
                "log, which is the reason a persistent runtime was withheld (D17). "
                "Set declares_kernel_snapshots = True once the provider emits them."
            )
        return claim_slot(scope or self.ctx, self, "provider", provider, label="code_runtime")

    def register_sdk_renderer(
        self,
        language: str,
        renderer: Callable[[Sequence[CodeBindingNamespace]], str],
        *,
        scope: Context | None = None,
    ) -> Disposer:
        """Claim the `tools:sdk` prompt renderer for one language (P1-04).

        The seam is the one place a renderer lives, so a provider that registers
        a richer one and is then disposed leaves an *absence* — a loud failure at
        prompt assembly — rather than silently reverting to a default whose
        different text would invalidate the cached prefix (A12).
        """
        return claim_key(
            scope or self.ctx, self._sdk_renderers, language, renderer, label="sdk-renderer"
        )

    def sdk_renderer(self, language: str) -> Callable[[Sequence[CodeBindingNamespace]], str] | None:
        return self._sdk_renderers.get(language)

    def require(self) -> CodeRuntime:
        if self.provider is None:
            raise RuntimeError("no ctx.code_runtime provider is registered; Code Mode requires one")
        return self.provider

    async def run(self, request: CodeRunRequest) -> CodeRunResult:
        return await self.require().run(request)


@plugin("code-runtime")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the code-runtime seam definition. No provider ships in Phase 1."""
    ctx.provide("code_runtime", CodeRuntimeSeam(ctx=ctx))
