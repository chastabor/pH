"""A `ctx.sandbox` backend for tests, which reports the enforcement that a test gives it.

Two versions of this existed within one changeset: a local class in this
package, and a bare `object()` registered as a provider in another package. The
bare object stopped satisfying `SandboxProvider` when that protocol gained
`enforcement`. It kept passing only because the reader that it met did not ask.
`stub_workspace` was written about the same state.

The backend confines nothing. Every test that has wanted one so far asks what
the *deployment* enforces, and not what a command becomes.

@module ph.testing.stub_sandbox
"""

from __future__ import annotations

from dataclasses import dataclass

from ..seams.sandbox import ConfinedArgv, Enforcement

__all__ = ["StubSandboxProvider"]


@dataclass(slots=True)
class StubSandboxProvider:
    """A backend that reports `enforcement` and wraps nothing.

    It is not a `DenialReader`. It confines nothing, so it has no refusals of
    its own to recognize. A test that wants the reading half drives the backend
    whose kernel messages it means to read: `Bubblewrap` or `Seatbelt`. Each of
    them owns its own signature table.
    """

    enforcement: Enforcement = "full"
    backend: str = "stub"

    def confine(self, argv: tuple[str, ...], policy: object) -> ConfinedArgv:
        return ConfinedArgv(argv=argv, enforcement=self.enforcement, backend=self.backend)
