"""A `ctx.sandbox` backend for testing, enforcing whatever a test says it does.

Two spellings of this existed within one changeset — a local class here and a
bare `object()` registered as a provider one package over, which stopped
satisfying `SandboxProvider` the moment that protocol grew `enforcement` and
kept passing only because the reader it met happened not to ask. That is the
state `stub_workspace` was written about, arriving again.

It confines nothing: every test that has wanted one so far is asking what the
*deployment* enforces, not what a command becomes.

@module ph.testing.stub_sandbox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..seams.sandbox import ConfinedArgv, Enforcement

__all__ = ["StubSandboxProvider"]


@dataclass(slots=True)
class StubSandboxProvider:
    """A backend that reports `enforcement` and wraps nothing."""

    enforcement: Enforcement = "full"

    def confine(self, argv: tuple[str, ...], policy: Any) -> ConfinedArgv:
        return ConfinedArgv(argv=argv, enforcement=self.enforcement, backend="stub")
