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

from ..cordis import Context, plugin
from ..seams.commands import CommandDefinition
from ..seams.workspace import workspace_of
from ..seams.workspace_git import checkpoints, restore
from ..session import Session

__all__ = ["apply"]

log = logging.getLogger("ph.commands.revert")

USAGE = "usage: /revert <seq>   (/revert with no argument lists the restore points)"


@plugin("workspace-revert", inject=["commands", "workspace", "subprocess"])
async def apply(ctx: Context, _config: Any) -> None:
    """Register `/revert`."""

    async def revert(argument: str, invocation: Any) -> str:
        session: Session | None = invocation.session
        if session is None:
            return "refusing: /revert needs a session to read restore points from"
        points = checkpoints(session)
        raw = argument.strip()
        if not raw:
            return _listing(points)
        if not raw.isdigit() or (seq := int(raw)) not in points:
            known = ", ".join(str(seq) for seq in sorted(points)) or "none"
            return f"refusing: no restore point at {raw!r} (known: {known})\n{USAGE}"

        point = points[seq]
        call_id = str(point.get("callId", ""))
        workspace = workspace_of(ctx, invocation.agent)
        # By id, not by comparing roots: a restore point belongs to the agent
        # that took it, and asking the seam a second time for that agent's root
        # was both a second spelling of one question and *less* safe — a disposed
        # agent whose directory got reused would have compared equal.
        if workspace is None or getattr(invocation.agent, "id", "") != str(point["agentId"]):
            return (
                f"refusing: restore point {raw} belongs to agent "
                f"{point['agentId']!r}, which does not hold a workspace here"
            )
        try:
            removed = await restore(ctx, workspace, str(point["tree"]))
        except FileNotFoundError as gone:
            # The write-ahead window (A10): the event was appended before the ref
            # that keeps the tree alive, so a crash in between leaves a restore
            # point that names a tree git has since collected.
            return f"restore point {raw} is no longer available: {gone}"

        # Not a file count. `restored 1,900 file(s)` for a cell that changed one
        # is the wrong sentence in the one place a person is checking whether
        # "revert" meant it.
        lines = [f"restored the workspace to the state before call {call_id or '?'}"]
        if removed:
            lines.append(f"removed {len(removed)} file(s) the run created")
        lines.extend(_not_undone(ctx, invocation.scope, session, call_id))
        return "\n".join(lines)

    ctx.commands.register(
        CommandDefinition(
            name="revert",
            summary="Restore this agent's workspace to a per-run checkpoint.",
            argument_hint="<seq>",
            run=revert,
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


def _not_undone(ctx: Context, scope: Context, session: Session, call_id: str) -> list[str]:
    """The run's dispatches a tree restore does not cover, in the order they ran.

    Asked of each tool's own declaration, so a deployment that renamed `bash` or
    an MCP server that added a publish tool is covered without this module
    knowing either name — and an *unknown* tool counts as not covered, which is
    the direction a person checking whether "revert" meant it needs.
    """
    if not call_id:
        return []
    # Stated by the dispatch, not derived from the approval-routing target
    # (P6-24). This is a policy read — which tools a revert covers — so the
    # boundary has to be the one the caller named.
    outside = [
        (str(event.data.get("name", "?")), event.data.get("arguments"))
        for event in session.events
        if event.type == "tool/code-dispatch-start"
        and str(event.data.get("parentCallId", "")) == call_id
        and not _covered(ctx, str(event.data.get("name", "?")), scope)
    ]
    if not outside:
        return []
    return [
        "",
        "git restores the tree, not the world. This run also did the following, "
        "and restoring the workspace did NOT undo it:",
        *(f"  - {name}({_brief(arguments)})" for name, arguments in outside),
    ]


def _covered(ctx: Context, name: str, scope: Any) -> bool:
    """Scope-aware, because a shadowed registration is a different tool.

    `offload`'s reader of `self_limits` makes the same point: an agent-scoped or
    MCP-registered tool is invisible at root scope. Here the unscoped lookup ran
    in the *unsafe* direction — a row shadowing `write` would have resolved to
    the global builtin's `True` and a dispatch that reached past the tree would
    never have been listed.
    """
    definition = ctx.tools.get(name, scope=scope)
    return bool(definition is not None and definition.effects_confined_to_workspace)


def _brief(arguments: Any) -> str:
    """Enough of the arguments to recognise the call, never the whole payload.

    `Mapping`, not `dict`: the log freezes payloads into `MappingProxyType`,
    which is a `Mapping` and is *not* a `dict` instance — a `dict` check here
    silently rendered every call as `bash()` with the one detail a person needs
    to recognise it stripped out. The same trap P4-05 hit reading a frozen
    argument tree.
    """
    if not isinstance(arguments, Mapping):
        return ""
    return ", ".join(f"{key}={_clip(value)}" for key, value in islice(arguments.items(), 2))


def _clip(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 40 else f"{text[:37]}..."
