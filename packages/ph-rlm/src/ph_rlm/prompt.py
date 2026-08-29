"""`rlm-prompt` — the RLM doctrine, and what it deliberately does not say (P3-14).

Ported from prime-agent's `prompts/rlm.ts`. Three things about the port are
decisions rather than transcription:

**Nothing here re-describes the generated surface.** Prime Agent's doctrine
carried an "RLM-native call contract" paragraph, and a first draft of this file
carried a bullet list of the five delegation calls. Both existed because
prime-agent had no generated listing; `tools:sdk` *is* that listing, built from
the registry, so prose beside it is a second description of one surface with the
hand-written copy going stale first. What survives is what the listing cannot
say: the rules.

**The non-blocking rule is stated as a rule, not a hint.** `rlm.run` returns an
admission handle, so a model that waits for an answer waits forever.

**Volatile facts are a `context()`, not a `section`.** Depth, working directory
and the family change between turns; in a cached `section` each change would
re-bill the whole prefix (A12). They are materialized as a snapshot after
retained history, and only when the text changed.

@module ph_rlm.prompt
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.seams.subagents import reachable_family
from ph.seams.workspace import workspace_of
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, PromptContext, PromptSection
from ph_runtime.cell import MAGIC_PREFIXES

from .bindings import RUN_TOOL
from .presentation import IPYTHON
from .subagents import RLM_MAX_DEPTH, TASK_PREFIX, delegation_depth

__all__ = ["CHILD_DOCTRINE", "DELEGATION", "DOCTRINE", "MAGICS", "WORKSPACE_LINE", "apply"]

MAGICS = ", ".join(f"`{prefix}`" for prefix in MAGIC_PREFIXES)
"""The prefixes the guest actually refuses, rendered for the doctrine.

Derived rather than restated: a fourth prefix added to the runtime would
otherwise leave the prompt describing a different runtime from the one the model
is talking to — which is exactly the drift that had the doctrine promising
`%%bash` shell cells for a phase (P3-22)."""

ORDER_RLM_DOCTRINE = ORDER_TOOL_GUIDANCE + 10
ORDER_RLM_DELEGATION = ORDER_TOOL_GUIDANCE + 20
ORDER_RLM_CHILD = ORDER_TOOL_GUIDANCE + 30
"""After `tools:sdk` (order 100), because the doctrine refers to the surface the
SDK block has just described. Prime Agent's order: RLM → subagent guidance."""

DOCTRINE = f"""\
# Writing code to solve tasks

You are a general purpose agent that uses code to solve tasks. Your one callable
is `{IPYTHON}`: you write Python, it runs in a kernel that persists across calls,
and you read what it printed and returned.

The kernel is *yours* and it keeps its state. A variable you set in one cell is
there in the next, so build up a working set rather than recomputing it. Import
from the target project's own environment for project imports, tests, scripts and
CLIs — that is what the kernel is for — rather than reimplementing what the
project already does.

There are no IPython magics — no {MAGICS}, so no `%%bash` cell. A cell that
starts with one is a syntax error. Shell commands go through
`await tools.bash(...)`, which is governed and recorded like every other
binding; the magic was the one shell that nothing could see.

Every capability outside plain Python is reached as an `await` on a binding — the
SDK block above lists them. Those calls are governed and recorded individually: a
cell that edits forty files produces forty reviewable records, not one. Prefer
them over reaching for `pathlib` or `subprocess` directly, because a binding is
the form the harness can check, report and, where a policy says so, refuse.

If a binding call is refused, the cell stops there. That is deliberate: re-plan
with the refusal in mind rather than looking for another route to the same effect.
"""

DELEGATION = """\
# Delegating to child agents

`rlm.run` starts a child agent and **returns immediately** with an admission
handle — never the answer. The child works while you do.

This is the part that trips up a control loop: there is nothing to await. Do not
`time.sleep()`, poll, or loop waiting for a child. A child's reply arrives as an
agent message on one of your later turns; until then, do other useful work or
finish your turn.

Two things the call signatures above cannot tell you:

- `name` is how you address a child later, so name it for its task rather than
  by number.
- `access` defaults to `"read"`. Ask for `"write"` only when the child must
  change files — a child that only reads should not be able to.

You may message your parent, your siblings and your own children, and nobody
else. Delegate work that is genuinely separable and worth another agent's
context: two children that have to agree with each other will spend more effort
coordinating than one agent would have spent doing the work.
"""

CHILD_DOCTRINE = f"""\
# You are a child agent

You were spawned by another agent to do one task. A task from your parent arrives
labelled `{TASK_PREFIX}`.

When you have an answer, **send it**:

    await agent_message.send(message="<your answer>", receiver_role="parent")

Finishing without sending a reply is the one failure mode that looks like success
from your side: your parent is told you finished silently and has to guess whether
you did the work. Send the answer, then finish.
"""

WORKSPACE_LINE = (
    "Workspace: none acquired, so file access is whatever this process "
    'already has and a child\'s `access="write"` request is recorded but not granted'
)
"""What the workspace line says when this agent holds no workspace (D21).

Stated rather than omitted, because the reason the plan wants this line is that
an agent handed a read-only repo *without notice* attempts writes and reads the
failures as its own bug — and "nothing is bounding you" is the same warning.

It is reached by *asking* — `facts()` emits it only when the seam has no
workspace for this agent — rather than by asserting an absence. An assertion
would keep telling every agent that nothing is mounted after something was, and
the test pinning it would defend the falsehood.

**"none acquired", not "no tier is mounted".** The first draft said the latter,
which stopped being true the moment P4-07 put `workspace-shared` in `ph-base`:
the seam is mounted in every profile, and what varies is whether the agent
lifecycle has taken a workspace and which tier answered. The practical fact for
the model is the same either way — nothing is bounding it — and that is what the
sentence has to carry."""


@plugin("rlm-prompt", inject=["system_prompt", "tools", "sessions", "subagents"])
async def apply(ctx: Context, _config: Any) -> None:
    """Contribute the doctrine sections and the volatile-facts snapshot."""

    def delegation(request: Any) -> str:
        """The delegation rules, and only when the agent can actually delegate.

        Keyed on `rlm_run`'s *visibility* — the same question the SDK block asks
        per tool — rather than on the namespace being registered. A deployment
        that denied `rlm_run` alone would otherwise drop it from the listing
        while this section kept teaching it.
        """
        return DELEGATION if ctx.tools.view(request.scope).visible.get(RUN_TOOL) else ""

    def child_doctrine(request: Any) -> str:
        session = getattr(request.agent, "session", None)
        return CHILD_DOCTRINE if session is not None and delegation_depth(session) else ""

    def facts(request: Any) -> str:
        """The turn-to-turn state, as a cache-safe snapshot (A12).

        One roster fold, reused: the family names and the children line are both
        answers from it, and folding once per name turned prompt assembly into
        N+1 scans of the parent's whole log per model step.
        """
        session = getattr(request.agent, "session", None)
        if session is None:
            return ""
        depth = delegation_depth(session)
        limit = _depth_limit(ctx)
        gate = " (you may not delegate further)" if depth >= limit else ""
        lines = ["# Session", f"Recursive agent depth: {depth} of {limit}{gate}"]
        if session.header.cwd:
            lines.append(f"Working directory: {session.header.cwd}")
        lines.append(f"Conversation log: {session.id}")
        lines.extend(_workspace(ctx, getattr(request.agent, "id", "")))

        sessions = ctx.sessions.list()
        family = [
            f"{role} {ctx.subagents.name_of(sessions, agent_id)}"
            for agent_id, role in sorted(reachable_family(sessions, session.id).items())
            if role != "self"
        ]
        if family:
            lines.append(f"Reachable agents: {', '.join(family)}")
        children = ctx.subagents.roster(session)
        if children:
            lines.append(
                "Your children: "
                + ", ".join(
                    f"{row.get('name')} "
                    f"({'deleted' if row.get('deleted') else row.get('status', 'queued')})"
                    for row in children.values()
                )
            )
        return "\n".join(lines)

    ctx.system_prompt.section(
        PromptSection(name="rlm:doctrine", order=ORDER_RLM_DOCTRINE, text=DOCTRINE)
    )
    ctx.system_prompt.section(
        PromptSection(name="rlm:delegation", order=ORDER_RLM_DELEGATION, text=delegation)
    )
    ctx.system_prompt.section(
        PromptSection(name="rlm:child", order=ORDER_RLM_CHILD, text=child_doctrine)
    )
    ctx.system_prompt.context(PromptContext(name="rlm:session", order=10, text=facts))


def _workspace(ctx: Context, agent_id: str) -> list[str]:
    """What is true about *this agent's* workspace, asked of the seam.

    Asked of the seam for a specific agent, not read off `ctx.workspace` as if
    the provision were a workspace: `ctx.<name>` is a service everywhere in this
    codebase, and a workspace is per-agent state a service hands out. The first
    draft of this line duck-typed the provision itself, which would have started
    describing the seam object the day P4-07 mounted it.

    `repo_writable` is reported as the seam states it and never inferred from the
    kind: `worktree-ephemeral` is writable and reaches nobody, and a line that
    called it read-only would be telling the model the opposite of what happens.

    Lines rather than one string with newlines in it, because `facts()` already
    owns the joining — a second assembly mechanism for one block of text is one
    that can disagree with the first about spacing.
    """
    workspace = workspace_of(ctx, agent_id)
    if workspace is None:
        return [WORKSPACE_LINE]
    writable = "writable" if workspace.repo_writable else "read-only"
    lines = [
        f"Workspace: {workspace.root} ({writable}, {workspace.kind})",
        f"Writable scratch: {workspace.scratch}",
    ]
    if workspace.ref:
        lines.append(f"Branch: {workspace.ref}")
    return lines


def _depth_limit(ctx: Context) -> int:
    """The delegation limit in force, asked of the provider that enforces it.

    Read from the provider rather than from this row's own config, because two
    copies of one limit is how a prompt tells a child it has a level left that
    the provider refuses. The `None` branch is load-bearing: the doctrine row can
    mount in a deployment with no delegation provider at all.
    """
    provider = ctx.get("rlm_children")
    return RLM_MAX_DEPTH if provider is None else int(provider.depth_limit)
