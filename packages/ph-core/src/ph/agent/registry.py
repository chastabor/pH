"""`ctx.agents` — agent creation, and the events every agent publishes.

An agent is a handle: an id, a session, a **scoped context**, an inbox, a
status and a cancel. The registry owns the scope — it creates it, provides the
handle into it as `agent`, and disposes it — and the driver runs inside it.
That split is what keeps the loop a replaceable row (invariant I1): a second
driver inherits the scope tree, the `agent` provision and the lifecycle events
without reproducing them.

@module ph.agent.registry
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, events, plugin
from ..session import Session
from .types import AgentOptions, PreStepRequest, RequestFailure, RequestProposal

__all__ = ["AgentRegistry", "apply"]

events.declare("agent/created", "emit", owner="ph.agent", doc="An agent handle was created.")
events.declare("agent/disposed", "emit", owner="ph.agent", doc="An agent handle was disposed.")
events.declare(
    "agent/status", "emit", owner="ph.agent", doc="An agent moved between idle and running."
)
events.declare(
    "agent/error", "emit", owner="ph.agent", doc="A failure was reported at its live boundary."
)
events.declare("agent/session-start", "emit", owner="ph.agent", doc="An agent bound a session.")
events.declare(
    "agent/inbox/inserted", "emit", owner="ph.agent", doc="A message entered a pending list."
)
events.declare(
    "agent/inbox/claimed", "emit", owner="ph.agent", doc="A pending message was claimed for a turn."
)
events.declare(
    "agent/inbox/discarded", "emit", owner="ph.agent", doc="A pending message was cancelled."
)
events.declare(
    "agent/pre-step",
    "waterfall",
    PreStepRequest,
    owner="ph.agent",
    doc="The authoritative reject-or-enter decision for one step.",
)
events.declare(
    "agent/request",
    "waterfall",
    RequestProposal,
    owner="ph.agent",
    doc="Proposes the call config for one request.",
)
events.declare(
    "agent/request-error",
    "waterfall",
    RequestFailure,
    owner="ph.agent",
    doc="A failed request; a listener may answer with a retry.",
)
events.declare(
    "agent/turn-stopping",
    "serial",
    owner="ph.agent",
    doc="Last chance to keep a turn alive; a listener objects by steering.",
)

DriverFactory = Callable[[Context, Session, AgentOptions], Any]
"""Builds a driver inside an already-created agent scope."""


@dataclass(slots=True)
class AgentRegistry:
    """The service published as `ctx.agents`."""

    ctx: Context
    driver_factory: DriverFactory | None = None
    _agents: dict[str, Any] = field(default_factory=dict)

    def register_driver(self, factory: DriverFactory) -> Callable[[], None]:
        """Claim the driver used by `create()`.

        The loop is a plugin like any other (invariant I1): swapping it is a row
        change, not a fork.
        """
        self.driver_factory = factory

        def release() -> None:
            if self.driver_factory is factory:
                self.driver_factory = None

        return release

    def create(
        self, session: Session, options: AgentOptions | None = None, *, parent: Any = None
    ) -> Any:
        """Build an agent and the scope it owns.

        **`parent` puts the child's scope inside its parent's** (P6-27), which is
        what makes containment structural instead of materialised. Every agent
        used to hang off the *registry*, so a parent and its child were siblings
        and `parent.ctx.reaches(child.ctx)` was `False` — the relationship that
        `SessionHeader.parent_session` records, and that B7 is entirely about,
        had no representation in the tree that answers questions about it. So
        `ctx.subagents` rebuilt the ceiling by hand on every spawn, and
        `grant_for`'s docstring had to explain that writing `None` out as an
        explicit list "is not a nicety".

        Nested, three things stop being remembered. Visibility inherits, because
        `isolation_chain` now reaches the parent's layers and the existing
        masking already walks it. Disposal cascades, because `Context.dispose`
        already unwinds `_children` — the subagent provider's
        `parent.ctx.effect(...)` was doing by hand what the tree does by shape.
        And a parent can *reach* its child, which is what makes a supervisor able
        to diagnose one.

        Optional and defaulting to today's behaviour: a root agent has no parent,
        and five of the seven call sites in the tree create one.
        """
        if self.driver_factory is None:
            raise RuntimeError("no agent driver is registered; mount an agent-loop row")
        owner = getattr(parent, "ctx", None)
        base = owner if isinstance(owner, Context) else self.ctx
        scope = base.scope(f"agent:{session.id}")
        agent = self.driver_factory(scope, session, options or AgentOptions())
        scope.provide("agent", agent)
        self._agents[agent.id] = agent
        # **The roster entry is an effect of the scope it describes**, which it
        # had to become the moment P6-27 nested agents: `dispose(agent_id)` used
        # to be the only thing that popped `_agents`, and a parent's
        # `Context.dispose` now tears a child's scope down without going through
        # it. That left `agents.get(child)` handing back a live-looking handle
        # over a dead context, `agents.list()` — "every live agent, for a sweep
        # that must visit all of them" — including it, and `agent/disposed`
        # never firing. Structural for the same reason the lifetime is: a second
        # provider that nests without registering a teardown effect of its own
        # cannot leak a handle it never had to remember.
        scope.add_disposer(lambda: self._forget(agent), label=f"agents.entry({agent.id})")
        scope.emit("agent/created", agent)
        scope.emit("agent/session-start", agent, session)
        return agent

    def get(self, agent_id: str) -> Any | None:
        """The live agent by id, or `None`. The symmetric read to `ctx.sessions.get`.

        A plugin that has an agent id — a runtime keyed by it, a policy folding
        its session — reaches the agent's scope and session through here, instead
        of shadowing the registry with its own `agent/created` side table.
        """
        return self._agents.get(agent_id)

    def list(self) -> list[Any]:
        """Every live agent, for a sweep that must visit all of them."""
        return list(self._agents.values())

    def _forget(self, agent: Any) -> None:
        """Drop the roster entry and announce it, once, whoever unwound the scope.

        Both doors reach here: `dispose(agent_id)` below, and a parent scope
        cascading into this one. `emit` before the pop is deliberate — a listener
        asking `agents.get` about the agent it is being told about should still
        find it — and the pop is idempotent, so the explicit path calling
        `scope.dispose()` afterwards runs this a second time and does nothing.
        """
        if self._agents.pop(agent.id, None) is None:
            return
        agent.ctx.emit("agent/disposed", agent)

    async def dispose(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        await agent.dispose()
        # `_forget` rides the scope's own teardown, so the announcement and the
        # roster drop happen exactly once whichever way the scope goes.
        await agent.ctx.dispose()


@plugin("agent", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the agent registry."""
    ctx.provide("agents", AgentRegistry(ctx=ctx))
