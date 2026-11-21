"""A `ctx.subagents` provider that answers without running an agent.

The real provider is `rlm-child`, which drives a whole child agent through
`ctx.jobs`. A row in ph-core that *uses* delegation (`subagent-task`) must be
testable without it. That row is responsible for the request that it builds,
for the wait, and for how it reports an outcome. None of these needs a model.

The stub deliberately keeps the parts of the shape that a caller can get wrong:

- `result` can be left as `None` (`waitable=False`). "The children of this
  provider cannot be waited on" is a real answer that a blocking caller must
  handle.
- `granted` may be narrower than `requested`. That downgrade is the one fact
  that a parent must be told, and the one that a stub is tempted to skip.

@module ph.testing.stub_subagent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anyio

from ..keys import SESSIONS
from ..seams.subagents import (
    Access,
    DowngradeReason,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentSpawnError,
    SubagentStatus,
    open_child_log,
)

__all__ = ["HoldingProvider", "RefusingProvider", "StubSubagentProvider"]


@dataclass(slots=True)
class StubSubagentProvider:
    """Admits a child, then hands back a canned outcome."""

    answer: str = "the child's answer"
    status: SubagentStatus = "done"
    error: str | None = None
    value: dict[str, Any] | None = None
    """The value that a child with an `output` returned, as the drive of a
    provider settles it."""
    grants: Access | None = None
    """What the child actually gets. `None` grants what was asked."""
    downgrade_reason: DowngradeReason | None = None
    waitable: bool = True
    """`False` leaves `result` unset. That is a provider whose children reply
    only by message. A blocking caller must refuse such a provider and must not
    hang on it."""
    root: Any = None
    """A context, set to exercise narrowing. The stub reads only whether it is
    set: the child gets a scope only when it is.

    **The scope is made under `request.parent`**, because
    `AgentRegistry.create` has done that since P6-27. The scope of a child
    nests inside the scope of its parent, and the ceiling is inherited through
    the isolation chain and not applied. Until then this field said the
    opposite, and argued for it: *"a real child's scope is its parent's
    sibling; a stub that nested one would pass a ceiling test that production
    fails."* That was true when it was written, and the row that nested the
    scopes inverted it exactly. So the stub now takes its shape from the
    production path, and does not restate that path in prose that nobody reads
    again."""
    requests: list[SubagentRequest] = field(default_factory=list)
    """Every request that this provider was given, in order."""

    async def start(self, request: SubagentRequest) -> SubagentRun:
        self.requests.append(request)
        index = len(self.requests)
        granted: Access = self.grants or request.access
        run_id = f"run-{index}"
        # The child gets a log of its own, opened through `open_child_log` as a
        # real provider opens one (Phase 11). The seam writes the admission of
        # the child into that log and refuses a child without one. So a stubbed
        # spawn meets the same check as a real one.
        parent_log = request.parent.session
        session_id = f"session-{index}"
        if parent_log is not None and request.parent.ctx.get(SESSIONS) is not None:
            session_id = (await open_child_log(request.parent.ctx, parent_log, run_id)).id
        run = SubagentRun(
            id=run_id,
            name=request.name or f"child-{index}",
            session_id=session_id,
            parent_id=request.parent.id,
            model_provider="fake",
            model="fake-1",
            requested_access=request.access,
            granted_access=granted,
            downgrade_reason=self.downgrade_reason,
        )
        if self.root is not None:
            # `root` still decides *whether* there is a scope. A provider that
            # hands back none is the fail-closed case that `_enforce` refuses,
            # and a test must be able to be that provider. What changed is
            # *where* the scope is made: under the parent of the request, as
            # `AgentRegistry.create` has done since P6-27. So a stubbed child
            # inherits the ceiling in the same way as a real one.
            owner = request.parent.ctx
            run.scope = owner.scope(f"agent:{run.id}")
        if self.waitable:
            run.result = self._settle
        return run

    async def _settle(self) -> SubagentResult:
        return SubagentResult(
            status=self.status, answer=self.answer, error=self.error, value=self.value
        )

    def last(self) -> SubagentRequest:
        """The most recent request, for a test that asserts what was delegated."""
        return self.requests[-1]


@dataclass(slots=True)
class HoldingProvider:
    """Holds its first spawn mid-build until `go` is set, and passes the rest
    straight to `inner`.

    The spawns after the first are then judged while the first is past its
    guards and not yet admitted. That is the state in which the tool calls of
    one step, which run side by side, leave each other. A test sets `go` when
    it has spawned what it needs."""

    inner: StubSubagentProvider
    first: SubagentRequest | None = None
    """The first spawn, as the seam passed it."""
    building: anyio.Event = field(default_factory=anyio.Event)
    """Set when the first spawn is being held."""
    go: anyio.Event = field(default_factory=anyio.Event)

    async def start(self, request: SubagentRequest) -> SubagentRun:
        if self.first is None:
            self.first = request
            self.building.set()
            await self.go.wait()
        return await self.inner.start(request)


@dataclass(frozen=True, slots=True)
class RefusingProvider:
    """Refuses every child that it is asked to build, after the guards of the seam pass."""

    reason: str = "no room for this child"

    async def start(self, request: SubagentRequest) -> SubagentRun:
        raise SubagentSpawnError(self.reason)
