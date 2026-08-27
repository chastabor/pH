"""P0-04 — the event declaration registry.

Gate: *dispatching an event under the wrong mode raises.* dsh gets this from an
`@mode`-tagged catalog checked at build time; Python has no such tag, so the
declaration is the contract and the check is at the dispatch site.
"""

from __future__ import annotations

import pytest

from ph.cordis import Context, EventModeError, UndeclaredEventError, import_plugin_modules
from ph.cordis.events import EventRegistry, events

pytestmark = pytest.mark.anyio

events.declare("test/registry-emit", "emit", owner="tests", doc="A test event.")


async def test_wrong_dispatch_mode_raises() -> None:
    root = Context()
    with pytest.raises(EventModeError, match='declared "emit"'):
        await root.parallel("test/registry-emit")
    with pytest.raises(EventModeError):
        await root.waterfall("test/registry-emit", inner=lambda: None)


async def test_undeclared_event_raises_on_dispatch_and_listen() -> None:
    root = Context()
    with pytest.raises(UndeclaredEventError):
        root.emit("test/never-declared")
    with pytest.raises(UndeclaredEventError):
        root.on("test/never-declared", lambda: None)


def test_redeclaring_with_a_different_mode_is_refused() -> None:
    registry = EventRegistry()
    registry.declare("x/y", "emit")
    # Idempotent for the same mode, so a module imported twice is fine.
    registry.declare("x/y", "emit")
    with pytest.raises(EventModeError, match="already declared"):
        registry.declare("x/y", "serial")


def test_matrix_records_producers_and_consumers() -> None:
    registry = EventRegistry()
    registry.declare("a/b", "waterfall", owner="ph.thing", doc="Does a thing.")
    registry.note_consumer("a/b", "ph.other")
    registry.note_consumer("a/b", "ph.other")
    (row,) = registry.matrix()
    assert row == {
        "name": "a/b",
        "mode": "waterfall",
        "producer": "ph.thing",
        "payload": None,
        "consumers": ["ph.other"],
        "doc": "Does a thing.",
    }


def test_core_declares_every_event_the_loop_dispatches() -> None:
    """A dispatch site with no declaration would fail at runtime, not at import.

    Declarations live in the plugin modules that own them, so the complete
    registry is reached the way `ph events` reaches it: by importing every
    registered plugin.
    """
    import_plugin_modules()
    declared = set(events.names())
    required = {
        "agent/pre-step",
        "agent/request",
        "agent/request-error",
        "agent/turn-stopping",
        "llm/stream",
        "session/created",
        "session/disposed",
        "session/event",
        "session/flush",
        "system-prompt/assemble",
    }
    assert required <= declared


def test_waterfalls_declare_their_payload_types() -> None:
    """The matrix names the payload beside the event, so listeners can be typed."""
    from ph.agent.types import PreStepRequest, RequestFailure, RequestProposal
    from ph.llm.types import GenerateOptions

    import_plugin_modules()
    assert events.require("agent/pre-step").payload is PreStepRequest
    assert events.require("agent/request").payload is RequestProposal
    assert events.require("agent/request-error").payload is RequestFailure
    assert events.require("llm/stream").payload is GenerateOptions
