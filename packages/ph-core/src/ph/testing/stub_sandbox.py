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
    """A backend that reports `enforcement` and wraps nothing.

    Not a `DenialReader`: it confines nothing, so there are no refusals of its own
    to recognise. A test that wants the reading half drives the backend whose
    kernel's words it means to read — `Bubblewrap` or `Seatbelt`, each of which
    owns its own signature table.
    """

    enforcement: Enforcement = "full"
    backend: str = "stub"

    def confine(self, argv: tuple[str, ...], policy: Any) -> ConfinedArgv:  # noqa: ANN401
        return ConfinedArgv(argv=argv, enforcement=self.enforcement, backend=self.backend)
