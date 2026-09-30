"""`/revert <seq>` — put the worktree back, and say what that does not reach (E7, N3).

A denial settles the whole run (Q9a), so partial state is bounded to about one
cell — and this is what makes that cell recoverable. `workspace-checkpoint` took
a tree before the run; this restores it.

**The word "revert" is the hazard, and the listing is the answer.** A run that
called `tools.bash` to publish a package, send mail or drop a table before being
denied is *not* undone by restoring the tree, and a person who trusts the word
would believe the run had no effect (N3). So the command prints what it restored
**and** lists the run's dispatches that a tree restore does not cover — read from
each tool's own `effects_confined_to_workspace` declaration rather than a name list
here, and defaulting to *not covered*, so a capability nobody thought about is
over-reported instead of silently trusted.

**Replay is not undo.** The `tool/code-dispatch` records carry names and
arguments, so a run's governed actions can be explained or re-attempted — but
raw `pathlib`/`subprocess` writes are bounded by the worktree and never recorded
(§4.8), so replay-forward cannot reproduce them. The checkpoint holds the actual
tree, is complete regardless of what was logged, and depends on no tool being
idempotent. That is why it, and not the record, is the recovery mechanism.

@module ph.commands.revert
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from itertools import islice
from typing import Any

from ..agent.types import AgentHandle
from ..cordis import Context, plugin
from ..json import JsonValue, as_str
from ..keys import COMMANDS, SUBPROCESS, TOOLS, WORKSPACE
from ..seams.commands import CommandContext, CommandDefinition, CommandVerb, Verbs
from ..seams.workspace import checkpoints, workspace_of
from ..session import IntentNotDurable, Session
from ..text import brief_value

__all__ = ["apply"]

log = logging.getLogger("ph.commands.revert")

USAGE = "usage: /revert <seq>   (/revert with no argument lists the restore points)"


@plugin("workspace-revert", affects="environment", inject=[COMMANDS, WORKSPACE, SUBPROCESS])
async def apply(ctx: Context, config: None) -> None:
    """Register `/revert`."""

    async def revert(argument: str, invocation: CommandContext) -> str:
        session = invocation.session
        if session is None:
            return "refusing: /revert needs a session to read restore points from"
        workspace = workspace_of(ctx, invocation.agent)
        if workspace is not None and not ctx.require(WORKSPACE).can_checkpoint(workspace):
            # **A refusal, not "no restore points in this session"** (P6-20's own
            # gate). That sentence is true of a kind that cannot checkpoint and
            # useless: it reads as "not yet", so a person waits for one to appear.
            # Naming the kind says the mechanism is absent rather than the points.
            # **The tier, not the kind**, because that is what decides it now: an
            # overlay's delta is a perfectly good restore point and nothing has
            # taught that tier to use it. Naming the workspace as well keeps the
            # sentence about the thing in front of the person.
            article = "an" if workspace.kind[0] in "aeiou" else "a"
            return (
                f"refusing: the mounted tier has no restore mechanism for {article} "
                f"{workspace.kind} workspace, so it has no restore points and will not "
                "grow any"
            )
        points = checkpoints(session)
        raw = argument.strip()
        if not raw:
            return _listing(points)
        if not raw.isdigit() or (seq := int(raw)) not in points:
            known = ", ".join(str(seq) for seq in sorted(points)) or "none"
            return f"refusing: no restore point at {raw!r} (known: {known})\n{USAGE}"

        point = points[seq]
        call_id = as_str(point.get("callId"))
        # By id, not by comparing roots: a restore point belongs to the agent
        # that took it, and asking the seam a second time for that agent's root
        # was both a second spelling of one question and *less* safe — a disposed
        # agent whose directory got reused would have compared equal.
        agent_id = invocation.agent.id if invocation.agent is not None else ""
        if workspace is None or agent_id != as_str(point["agentId"]):
            return (
                f"refusing: restore point {raw} belongs to agent "
                f"{point['agentId']!r}, which does not hold a workspace here"
            )
        try:
            # With the session, so the seam records the restore around the act
            # (`WORKSPACE_RESTORE`): a crash inside it reads as "may be partly
            # restored", not as a command whose outcome is unknown.
            removed = await ctx.require(WORKSPACE).restore(
                agent_id, as_str(point["tree"]), session=session
            )
        except IntentNotDurable:
            return (
                f"refusing: restore point {raw} was not restored, because the log could "
                "not record it"
            )
        except FileNotFoundError as gone:
            # Not a crash window any more: a restore point is pinned before it is
            # recorded (`WorkspaceSeam.checkpoint`), so a recorded one named state
            # that was kept alive. Gone means something removed that pin since.
            return f"restore point {raw} is no longer available: {gone}"

        # Not a file count. `restored 1,900 file(s)` for a cell that changed one
        # is the wrong sentence in the one place a person is checking whether
        # "revert" meant it.
        lines = [f"restored the workspace to the state before call {call_id or '?'}"]
        if removed:
            lines.append(f"removed {len(removed)} file(s) the run created")
        lines.extend(
            _not_undone(ctx, session, call_id, scope=invocation.scope, agent=invocation.agent)
        )
        return "\n".join(lines)

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="revert",
            summary="Restore this agent's workspace to a per-run checkpoint.",
            argument_hint="<seq>",
            # Bare, it lists — a question; with a seq it restores. One body for both,
            # because the checks ahead of either are the same.
            run=Verbs({"": CommandVerb(revert, reads=True)}, otherwise=revert),
        ),
        scope=ctx,
    )


def _listing(points: dict[int, dict[str, Any]]) -> str:
    if not points:
        return "no restore points in this session"
    rows = [
        f"{seq:<6} {point['agentId']:<16} call {point.get('callId', '?')}"
        for seq, point in sorted(points.items())
    ]
    return "\n".join(["seq    agent            run", *rows])


def _not_undone(
    ctx: Context, session: Session, call_id: str, *, scope: Context, agent: AgentHandle | None
) -> list[str]:
    """The run's dispatches a tree restore does not cover, in the order they ran.

    Asked of each tool's own declaration, so a deployment that renamed `bash` or
    an MCP server that added a publish tool is covered without this module
    knowing either name — and an *unknown* tool counts as not covered, which is
    the direction a person checking whether "revert" meant it needs. Asked of each
    call's own arguments (S4), so a `write` whose path left the tree is listed.
    """
    if not call_id:
        return []
    # Stated by the dispatch, not derived from the approval-routing target
    # (P6-24). This is a policy read — which tools a revert covers — so the
    # boundary has to be the one the caller named.
    tools = ctx.require(TOOLS)
    outside: list[tuple[str, JsonValue]] = []
    for event in session.events:
        if event.type != "tool/code-dispatch-start":
            continue
        if as_str(event.data.get("parentCallId")) != call_id:
            continue
        name, arguments = as_str(event.data.get("name"), "?"), event.data.get("arguments")
        if not tools.restore_covers(name, arguments, scope=scope, agent=agent):
            outside.append((name, arguments))
    if not outside:
        return []
    return [
        "",
        "a restore puts the tree back, not the world. This run also did the following, "
        "and restoring the workspace did NOT undo it:",
        *(f"  - {name}({_brief(arguments)})" for name, arguments in outside),
    ]


def _brief(arguments: JsonValue) -> str:
    """Enough of the arguments to recognize the call, never the whole payload.

    `Mapping`, not `dict`: the log freezes payloads into `MappingProxyType`,
    which is a `Mapping` and is *not* a `dict` instance — a `dict` check here
    silently rendered every call as `bash()` with the one detail a person needs
    to recognize it stripped out. The same trap P4-05 hit reading a frozen
    argument tree.
    """
    if not isinstance(arguments, Mapping):
        return ""
    return ", ".join(f"{key}={_clip(value)}" for key, value in islice(arguments.items(), 2))


def _clip(value: JsonValue) -> str:
    """One argument, short enough for a listing — see `brief_value` for why it
    is not `str()`: a nested argument object rendered as a Python literal in the
    one report a person reads while deciding whether a restore was enough."""
    text = brief_value(value)
    return text if len(text) <= 40 else f"{text[:37]}..."
