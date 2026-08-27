"""`code-runtime-stub` — a `ctx.code_runtime` provider for testing the bridge.

Phase 1 verifies the *governance* half of Code Mode without owning a runtime
yet (P1-05's gate: "against `code-runtime-worker-thread` semantics with a stub
runtime"). This executes a program by calling a Python callable with the
namespaces bound, which exercises the exact thing that matters here — bindings
re-entering the pipeline, per-dispatch records, denial settling the run — while
the real out-of-process CPython runtime lands in Phase 3 (D19).

It declares `persistence: "none"` and therefore owes no `kernel/snapshot`
events, which is the honest answer for something with no namespace to keep.

@module ph.testing.stub_runtime
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, plugin
from ..seams.code_runtime import CodeRunRequest, CodeRunResult

__all__ = ["StubCodeRuntime", "apply"]


@dataclass(slots=True)
class StubCodeRuntime:
    """Runs a registered Python callable as if it were a model-authored program."""

    language: str = "python"
    isolation: str = "in-process"
    persistence: str = "none"
    programs: dict[str, Callable[..., Any]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def register_program(self, name: str, program: Callable[..., Any]) -> None:
        """Bind a name a test can use as the `program` string."""
        self.programs[name] = program

    async def run(self, request: CodeRunRequest) -> CodeRunResult:
        namespaces = {namespace.name: _Namespace(namespace) for namespace in request.bindings}
        program = self.programs.get(request.program)
        if program is None:
            return CodeRunResult(error=f"no stub program named {request.program!r}")
        emitted: list[str] = []

        def emit(line: str) -> None:
            emitted.append(line)
            self.logs.append(line)

        try:
            outcome = program(namespaces, emit)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as error:
            # A raise inside the program is the program's outcome, not the
            # runtime's failure — except for the ones the bridge raises to end
            # the run, which the transport re-raises.
            from ..tools.code_mode import CodeRunFailure

            if isinstance(error, CodeRunFailure):
                raise
            return CodeRunResult(logs="\n".join(emitted), error=str(error))
        return CodeRunResult(logs="\n".join(emitted), value=outcome)


@dataclass(slots=True)
class _Namespace:
    """Attribute access over one binding namespace, as a program sees it."""

    namespace: Any

    def __getattr__(self, name: str) -> Any:
        for binding in self.namespace.bindings:
            if binding.name == name:
                return binding.dispatch
        raise AttributeError(f"{self.namespace.name}.{name} is not a binding")


@plugin("code-runtime-stub", inject=["code_runtime"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the stub runtime and expose it for a test to script."""
    runtime = StubCodeRuntime()
    ctx.code_runtime.register(runtime)
    ctx.provide("code_runtime_stub", runtime)
