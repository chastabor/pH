"""The event declaration registry.

dsh checks every dispatch site at build time against a catalog tagged with
`@mode`. Python has no such tag, so pH declares events at import time. An event
name carries exactly one dispatch mode, and dispatching it through a different
method raises. The same registry backs `phern events`, the producer/consumer
matrix (P0-04).

@module ph.cordis.events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import EventModeError, UndeclaredEventError

__all__ = ["DispatchMode", "EventDeclaration", "EventRegistry", "events"]

DispatchMode = Literal["emit", "parallel", "serial", "waterfall"]

_MODES: frozenset[str] = frozenset({"emit", "parallel", "serial", "waterfall"})


@dataclass(frozen=True, slots=True)
class EventDeclaration:
    """The public contract of one event name."""

    name: str
    mode: DispatchMode
    payload: type[Any] | None = None
    """The payload type, when the event carries one structured argument."""
    owner: str = ""
    """The declaring module, which is the producer of record of the event."""
    doc: str = ""
    consumers: set[str] = field(default_factory=set, compare=False)
    """The modules that have registered a listener at least one time, for the matrix."""


class EventRegistry:
    """Every declared event in the process, keyed by name."""

    def __init__(self) -> None:
        self._declarations: dict[str, EventDeclaration] = {}

    def declare(
        self,
        name: str,
        mode: DispatchMode,
        payload: type[Any] | None = None,
        *,
        owner: str = "",
        doc: str = "",
    ) -> EventDeclaration:
        """Declare one event name and its dispatch mode.

        Declaring the same name again with the same mode is a no-op, so a module
        imported twice under different paths does not fail. Declaring it again
        with a different mode raises, because the mode is part of the contract
        of the event.

        **Declaring it again with a different owner also raises** (A13). The
        no-op exists for the *same* declaration that arrives twice. Two modules
        that declare one name are two owners for one contract. Returning the
        first declaration silently dropped the `owner` and `doc` of the second.
        `phern events` then named one module for an event that two modules
        dispatch, and the documentation of the second was lost. The problem is
        latent today, because no name is declared twice. It is the kind of
        problem that arrives with a plugin that somebody else wrote.
        """
        if mode not in _MODES:
            raise EventModeError(f'unknown dispatch mode "{mode}" for event "{name}"')
        existing = self._declarations.get(name)
        if existing is not None:
            if existing.mode != mode:
                raise EventModeError(
                    f'event "{name}" is already declared as "{existing.mode}"; '
                    f'cannot re-declare as "{mode}"'
                )
            if owner and existing.owner and owner != existing.owner:
                raise EventModeError(
                    f'event "{name}" is already declared by "{existing.owner}"; '
                    f'"{owner}" cannot declare it as well'
                )
            return existing
        declaration = EventDeclaration(name=name, mode=mode, payload=payload, owner=owner, doc=doc)
        self._declarations[name] = declaration
        return declaration

    def require(self, name: str) -> EventDeclaration:
        declaration = self._declarations.get(name)
        if declaration is None:
            raise UndeclaredEventError(
                f'event "{name}" is not declared; call events.declare(name, mode) '
                "at import time in the module that owns it"
            )
        return declaration

    def check(self, name: str, mode: DispatchMode) -> EventDeclaration:
        """Assert that `name` may be dispatched through `mode`."""
        declaration = self.require(name)
        if declaration.mode != mode:
            raise EventModeError(
                f'event "{name}" is declared "{declaration.mode}" but was dispatched as "{mode}"'
            )
        return declaration

    def note_consumer(self, name: str, module: str) -> None:
        declaration = self._declarations.get(name)
        if declaration is not None and module:
            declaration.consumers.add(module)

    def names(self) -> list[str]:
        return sorted(self._declarations)

    def matrix(self) -> list[dict[str, Any]]:
        """The producer/consumer matrix, as data.

        dsh generates `event-producer-consumer.md` from its catalog. `phern events`
        renders this list instead.
        """
        return [
            {
                "name": declaration.name,
                "mode": declaration.mode,
                "producer": declaration.owner,
                "payload": None if declaration.payload is None else declaration.payload.__name__,
                "consumers": sorted(declaration.consumers),
                "doc": declaration.doc,
            }
            for declaration in (self._declarations[name] for name in self.names())
        ]


events = EventRegistry()
"""The process-wide event registry."""
