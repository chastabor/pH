"""A `ctx.subagents` provider that answers without running an agent.

The real one is `rlm-child`, which drives a whole child agent through
`ctx.jobs`. A row in ph-core that *uses* delegation — `subagent-task` — must be
testable without that: what it is responsible for is the request it builds, the
wait, and how it reports an outcome, none of which needs a model.

What this deliberately keeps is the shape a caller can be wrong about: `result`
is `None` unless asked for, because "this provider's children cannot be waited
on" is a real answer a blocking caller has to handle, and `granted` may be
narrower than `requested`, because that downgrade is the one fact a parent is
owed and the one a stub is tempted to skip.

@module ph.testing.stub_subagent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..seams.subagents import (
    Access,
    DowngradeReason,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentStatus,
)

__all__ = ["StubSubagentProvider"]


@dataclass(slots=True)
class StubSubagentProvider:
    """Admits a child, then hands back a canned outcome."""

    answer: str = "the child's answer"
    status: SubagentStatus = "done"
    error: str | None = None
    grants: Access | None = None
    """What the child actually gets. `None` grants what was asked."""
    downgrade_reason: DowngradeReason | None = None
    waitable: bool = True
    """`False` leaves `result` unset — a provider whose children only reply by
    message, which a blocking caller must refuse rather than hang on."""
    root: Any = None
    """A context to fall back to when a request names no parent. Set it to
    exercise narrowing.

    **The scope is made under `request.parent` when there is one**, because that
    is what `AgentRegistry.create` does since P6-27 — a child's scope nests
    inside its parent's, and the ceiling is inherited through the isolation
    chain rather than applied. This field said the opposite until then, and
    argued for it: *"a real child's scope is its parent's sibling; a stub that
    nested one would pass a ceiling test that production fails."* True when it
    was written, and exactly inverted by the row that nested them — which is why
    the stub takes its shape from the production path now instead of restating
    it in prose that nobody re-reads."""
    requests: list[SubagentRequest] = field(default_factory=list)
    """Every request this provider was handed, in order."""

    async def start(self, request: SubagentRequest) -> SubagentRun:
        self.requests.append(request)
        index = len(self.requests)
        granted: Access = self.grants or request.access
        run = SubagentRun(
            id=f"run-{index}",
            name=request.name or f"child-{index}",
            session_id=f"session-{index}",
            parent_id=getattr(request.parent, "id", "parent"),
            model_provider="fake",
            model="fake-1",
            requested_access=request.access,
            granted_access=granted,
            downgrade_reason=self.downgrade_reason,
        )
        if self.root is not None:
            # `root` still decides *whether* there is a scope — a provider that
            # hands back none is the fail-closed case `_enforce` refuses, and a
            # test needs to be able to be one. What changed is *where*: under the
            # request's parent when it has one, which is what
            # `AgentRegistry.create` does since P6-27, so a stubbed child
            # inherits the ceiling the same way a real one does.
            owner = getattr(request.parent, "ctx", None) or self.root
            run.scope = owner.scope(f"agent:{run.id}")
        if self.waitable:
            run.result = self._settle
        return run

    async def _settle(self) -> SubagentResult:
        return SubagentResult(status=self.status, answer=self.answer, error=self.error)

    def last(self) -> SubagentRequest:
        """The most recent request, for a test asserting what was delegated."""
        return self.requests[-1]
