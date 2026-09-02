"""The event declaration registry.

dsh checks every dispatch site against an `@mode`-tagged catalog at build time.
Python has no such tag, so pH declares events at import time instead: an event
name carries exactly one dispatch mode, and dispatching it through a different
method raises. The same registry backs `ph events`, the producer/consumer
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
    """One event name's public contract."""

    name: str
    mode: DispatchMode
    payload: type[Any] | None = None
    """The payload type, when the event carries one structured argument."""
    owner: str = ""
    """The declaring module — the event's producer of record."""
    doc: str = ""
    consumers: set[str] = field(default_factory=set, compare=False)
    """Modules that have registered a listener at least once, for the matrix."""


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

        Re-declaring the same name with the same mode is a no-op, so a module
        imported twice under different paths does not fail. Re-declaring it with
        a different mode raises: the mode is part of the event's contract.
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

        dsh generates `event-producer-consumer.md` from its catalog; `ph events`
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
