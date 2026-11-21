"""`/revert <seq>`: put the worktree back, and say what that does not reach (E7, N3).

A denial settles the whole run (Q9a), so partial state is limited to about one
cell. This command makes that cell recoverable. `workspace-checkpoint` took a
tree before the run, and this command restores it.

**The word "revert" is the hazard, and the listing is the answer.** A run may
call `tools.bash` to publish a package, send mail or drop a table before it is
denied. A restore of the tree does *not* undo that, and a user who trusts the
word would believe that the run had no effect (N3). So the command prints what
it restored **and** lists the dispatches of the run that a tree restore does not
cover. The list is read from the `effects_confined_to_workspace` declaration of
each tool and not from a list of names here. The default is *not covered*, so a
capability that nobody thought about is reported too often and not silently
trusted.

**Replay is not undo.** The `tool/code-dispatch-start` records carry names and
arguments, so the governed actions of a run can be explained or attempted again.
But raw `pathlib` and `subprocess` writes are bounded by the worktree and never
recorded (§4.8), so a replay forward cannot reproduce them. The checkpoint holds
the actual tree. It is complete whatever was logged, and it depends on no tool
being idempotent. That is why the checkpoint, and not the record, is the
recovery mechanism.

@module ph.commands.revert
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from typing import Any

from ..cordis import Context, plugin
from ..json import JsonValue, as_str
from ..keys import COMMANDS, SUBPROCESS, WORKSPACE
from ..seams.commands import CommandContext, CommandDefinition, CommandVerb, Verbs
from ..seams.workspace import checkpoints, workspace_of
from ..session import IntentNotDurable, Session
from ..text import brief_value, one_line

__all__ = ["apply"]


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
            # **A refusal, not "no restore points in this session"** (the gate
            # of P6-20). That sentence is true of a kind that cannot checkpoint,
            # and it is useless. It reads as "not yet", so a user waits for a
            # restore point to appear. A sentence that names the kind says that
            # the mechanism is absent, and not only the points. **The refusal is
            # about the tier, not the kind**, because the tier is what decides
            # now. The delta of an overlay is a good restore point, and nothing
            # has taught that tier to use it. The sentence also names the
            # workspace, so that it stays about the thing that the user is
            # looking at.
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
        # The match is by id, not by a comparison of roots. A restore point
        # belongs to the agent that took it. A second request to the seam for
        # the root of that agent was a second form of one question, and it was
        # also *less* safe: a disposed agent whose directory was reused would
        # have compared equal.
        agent_id = invocation.agent.id if invocation.agent is not None else ""
        if workspace is None or agent_id != as_str(point["agentId"]):
            return (
                f"refusing: restore point {raw} belongs to agent "
                f"{point['agentId']!r}, which does not hold a workspace here"
            )
        try:
            # The session is passed so that the seam records the restore around
            # the act (`WORKSPACE_RESTORE`). A crash inside the restore then
            # reads as "may be partly restored", and not as a command whose
            # outcome is unknown.
            removed = await ctx.require(WORKSPACE).restore(
                agent_id, as_str(point["tree"]), session=session
            )
        except IntentNotDurable:
            return (
                f"refusing: restore point {raw} was not restored, because the log could "
                "not record it"
            )
        except FileNotFoundError as gone:
            # This is no longer a crash window. A restore point is pinned before
            # it is recorded (`WorkspaceSeam.checkpoint`), so a recorded restore
            # point named state that was kept alive. A missing one means that
            # something removed the pin since.
            return f"restore point {raw} is no longer available: {gone}"

        # The reply gives no file count. `restored 1,900 file(s)` for a cell
        # that changed one file is the wrong sentence in the one place where a
        # user checks whether "revert" did what the word says.
        lines = [f"restored the workspace to the state before call {call_id or '?'}"]
        if removed:
            lines.append(f"removed {len(removed)} file(s) the run created")
        lines.extend(_not_undone(session, call_id))
        return "\n".join(lines)

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="revert",
            summary="Restore this agent's workspace to a per-run checkpoint.",
            argument_hint="<seq>",
            # With no argument the command lists, which is a question. With a
            # seq it restores. One body serves both, because the checks before
            # either are the same. The record of the restore is written with the
            # record of its act: the seam opens `WORKSPACE_RESTORE` before the
            # tree is touched, and that one barrier writes `command/run` with
            # it.
            run=Verbs(
                {"": CommandVerb(revert, record="after")},
                otherwise=CommandVerb(revert, record="with-act"),
            ),
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


def _not_undone(session: Session, call_id: str) -> list[str]:
    """The dispatches of the run that a tree restore does not cover, in the order
    they ran.

    **The answer is the one that each dispatch gave when it ran**
    (`restoreCovered`, S4). It is not asked again after the restore, when a link
    that the run made and the restore removed would hide a write that left the
    tree. The answer is the declaration of the tool itself for that call
    (`ToolRuntime.restore_covers`). A deployment that renamed `bash`, or an MCP
    server that added a publish tool, is therefore covered, and this module
    knows neither name. A record without an answer counts as not covered. That
    is the direction that a user needs who checks whether "revert" did what the
    word says.
    """
    if not call_id:
        return []
    outside: list[tuple[str, JsonValue]] = []
    for event in session.events:
        if event.type != "tool/code-dispatch-start":
            continue
        if as_str(event.data.get("parentCallId")) != call_id:
            continue
        if event.data.get("restoreCovered") is not True:
            outside.append((as_str(event.data.get("name"), "?"), event.data.get("arguments")))
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

    The check is for `Mapping`, not `dict`. The log freezes payloads into
    `MappingProxyType`, which is a `Mapping` and is *not* a `dict` instance. A
    `dict` check here silently rendered every call as `bash()`, without the one
    detail that a user needs to recognize it. P4-05 hit the same trap when it
    read a frozen argument tree.
    """
    if not isinstance(arguments, Mapping):
        return ""
    return ", ".join(
        f"{key}={one_line(brief_value(value), 40)}" for key, value in islice(arguments.items(), 2)
    )
