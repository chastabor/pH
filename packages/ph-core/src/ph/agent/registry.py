"""`ctx.agents`: agent creation, and the events that every agent publishes.

An agent is a handle: an id, a session, a **scoped context**, an inbox, a
status and a cancel. The registry owns the scope. It creates the scope, provides
the handle into it as `agent`, and disposes it. The driver runs inside the
scope. That split keeps the loop a replaceable row (invariant I1): a second
driver inherits the scope tree, the `agent` provision and the lifecycle events,
and does not reproduce them.

@module ph.agent.registry
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..cordis import Context, events, plugin
from ..keys import AGENT, AGENTS, SESSIONS
from ..session import Session, session_written
from .types import (
    AgentDriver,
    AgentHandle,
    AgentOptions,
    PreStepRequest,
    RequestFailure,
    RequestProposal,
)

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
    "agent/inbox/discarded", "emit", owner="ph.agent", doc="A pending message was canceled."
)
events.declare(
    "agent/woken",
    "emit",
    owner="ph.agent",
    doc="A waking message entered the inbox; a host that drives the agent on demand runs it.",
)
events.declare(
    "agent/pre-step",
    "waterfall",
    PreStepRequest,
    owner="ph.agent",
    doc="The authoritative reject-or-enter decision for one step.",
)
events.declare(
    "agent/step-rejected",
    "serial",
    PreStepRequest,
    owner="ph.agent",
    doc="A step `agent/pre-step` rejected, once the decision is final and its batch taken.",
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

DriverFactory = Callable[[Context, Session, AgentOptions], AgentDriver]
"""Builds a driver inside an already-created agent scope."""


@dataclass(slots=True)
class AgentRegistry:
    """The service published as `ctx.agents`."""

    ctx: Context
    driver_factory: DriverFactory | None = None
    _agents: dict[str, AgentDriver] = field(default_factory=dict)

    def register_driver(self, factory: DriverFactory) -> Callable[[], None]:
        """Claim the driver that `create()` uses.

        The loop is a plugin like any other (invariant I1). Swapping it is a row
        change and not a fork.
        """
        self.driver_factory = factory

        def release() -> None:
            if self.driver_factory is factory:
                self.driver_factory = None

        return release

    def create(
        self,
        session: Session,
        options: AgentOptions | None = None,
        *,
        parent: AgentHandle | None = None,
    ) -> AgentDriver:
        """Build an agent and the scope that it owns.

        **`parent` puts the scope of the child inside the scope of its parent**
        (P6-27). Containment is then structural and not materialized. Three
        things no longer have to be remembered:

        * **Visibility inherits**, because `isolation_chain` reaches the layers
          of the parent and the existing masking already walks it.
        * **Disposal cascades**, because `Context.dispose` already unwinds
          `_children`.
        * **A parent can reach its child**, so a supervisor is able to diagnose
          one.

        `parent` is optional. Without it, the agent is a root agent.
        """
        if self.driver_factory is None:
            raise RuntimeError("no agent driver is registered; mount an agent-loop row")
        base = parent.ctx if parent is not None else self.ctx
        scope = base.scope(f"agent:{session.id}")
        # **The log of the agent is written as the last thing that its scope
        # unwinds**, whoever unwinds it. The disposer is registered first so
        # that it runs last. The teardown ahead of it appends: a workspace
        # records `workspace/disposed` as it lets go. And a settled child is
        # never reopened for anything to write those records later. The
        # disposer is on the scope and not after `dispose(agent_id)`, because
        # the `Context.dispose` of a parent that cascades into a nested child
        # never calls `dispose(agent_id)` (S12). The unwind of a mount itself
        # belongs to `write_on_unwind` (F2). By then this disposer finds
        # nothing owed.
        written = scope.add_disposer(
            lambda: session_written(self.ctx, session), label=f"agents.written({session.id})"
        )
        try:
            agent = self.driver_factory(scope, session, options or AgentOptions())
        except BaseException:
            # **There is no agent, so nothing of one is left.** A driver refuses
            # a log that it cannot replay, for example an inbox record that a
            # newer build wrote. The scope used to stay under `base` with the
            # write above still on it. When `base` went, that disposer wrote a
            # session that its opener had released long before: after the
            # ending the seam had written into the same log from a second copy
            # of it. So the write is dropped, because the session belongs to
            # its opener again. The scope is unwound for whatever the driver
            # registered before it raised.
            written.drop()
            base.detach(scope.dispose(), label=f"agents.unmade({session.id})", at_dispose="finish")
            raise
        scope.provide(AGENT, agent)
        self._agents[agent.id] = agent
        # **The roster entry is an effect of the scope that it describes.** It
        # had to become one when P6-27 nested agents. `dispose(agent_id)` used
        # to be the only thing that popped `_agents`, and the `Context.dispose`
        # of a parent now tears down the scope of a child without going through
        # it. That left three faults. `agents.get(child)` returned a handle
        # that looked live over a dead context. `agents.list()`, which is
        # "every live agent, for a sweep that must visit all of them", included
        # it. And `agent/disposed` never fired. The entry is structural for the
        # same reason that the lifetime is: a second provider that nests
        # without registering a teardown effect of its own cannot leak a handle
        # that it never had to remember.
        scope.add_disposer(lambda: self._forget(agent), label=f"agents.entry({agent.id})")
        scope.emit("agent/created", agent)
        scope.emit("agent/session-start", agent, session)
        return agent

    def get(self, agent_id: str) -> AgentDriver | None:
        """The live agent by id, or `None`. The symmetric read to `ctx.sessions.get`.

        A plugin that has an agent id (a runtime keyed by it, a policy that
        folds its session) reaches the scope and the session of the agent
        through here. It does not shadow the registry with its own side table
        built from `agent/created`.

        **Live means that its scope is up** (S3), which is also the answer of
        `scope_of`. The teardown of the scope takes the roster entry back, near
        its end. For the whole of that teardown the entry therefore named an
        agent that was going. A message sent to it was acknowledged and steered
        into an inbox that its dispose had cleared. Such an agent is declined
        here. A sender then asks the seam to wake it (`ensure_addressable`). The
        provider of that seam knows that the agent it still holds is ending,
        and refuses. The sender is told that the message did not land.
        """
        agent = self._agents.get(agent_id)
        return agent if agent is not None and agent.ctx.active else None

    def scope_of(self, agent_id: str) -> Context | None:
        """The scope of the live agent, or `None` when nobody knows the id or its
        scope is down.

        This is for a caller that acquires something *for* an agent of which it
        has only the id (a workspace, a kernel) and makes it an effect of that
        agent. A scope that is down is declined, because nothing unwinds a
        registration on a dead lifetime.
        """
        agent = self.get(agent_id)
        return agent.ctx if agent is not None else None

    def list(self) -> list[AgentDriver]:
        """Every live agent, for a sweep that must visit all of them.

        Live has the meaning that `get` gives it. An agent whose scope is going
        is left out.
        """
        return [agent for agent in self._agents.values() if agent.ctx.active]

    def _forget(self, agent: AgentDriver) -> None:
        """Drop the roster entry and announce it, one time, whoever unwound the scope.

        Both paths reach here: `dispose(agent_id)` below, and a parent scope
        that cascades into this one. Both go through the teardown of the scope
        of the agent, which runs this as a disposer. **The pop comes first, and
        it is the guard.** The agent is therefore announced one time, and a
        listener that disposes again cannot announce it twice. A listener is handed the agent,
        so it has no need to look the agent up in the roster that it has left.
        An async listener runs after this returns in any case, because `emit`
        schedules it and does not await it.
        """
        # **Only the entry of this agent** (Holdings rule 4). An agent that was
        # made again for the same session while this one was still ending holds
        # the id now. When this teardown took that entry out, the new agent was
        # listed nowhere.
        if self._agents.get(agent.id) is not agent:
            return
        del self._agents[agent.id]
        agent.ctx.emit("agent/disposed", agent)

    async def dispose(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        await agent.dispose()
        # `_forget` runs in the teardown of the scope. The announcement and the
        # roster drop therefore happen exactly one time, whichever way the
        # scope goes. The same holds for writing the log that the teardown
        # appended to (`create`).
        await agent.ctx.dispose()


@plugin("agent", affects="environment", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the agent registry."""
    ctx.provide(AGENTS, AgentRegistry(ctx=ctx))
