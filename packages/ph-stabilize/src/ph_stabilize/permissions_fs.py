"""`permissions-fs` — path rules over filesystem access (P4-06, G7, E9).

`FilesystemPermission {operations, paths, mode}` evaluated **first-match-wins**
with a default of `allow`, attached to the `ctx.fs` seam rather than to the tools
that happen to call it. That placement is the row: the seam fires
`fs/read-intent`, `fs/write-intent` and `fs/edit-intent` before anything reaches
disk, so anything that goes through `ctx.fs` is covered without this module
knowing a single tool name — the failure `compaction`'s truncation pass and
`offload` both argue against out loud, and that P6-16 files against `hitl`.

**Be honest about how much that buys today.** `tool-fs` is currently the *only*
caller of `FsService.read`/`write`/`edit` in the tree, so the coverage this shape
earns is mostly future: a Code Mode binding is the namespaced face of a governed
tool and does route through here, but `agent-instructions` and `ph-rlm`'s harness
service both read with `Path.read_text` and are not covered, and an MCP server is
a remote client that can never reach `ctx.fs` at all. The placement is still
right — it is where the next writer lands for free — but the argument for it is
"one gate rather than one per tool", not a list of things already behind it.

**First-match-wins, and the order is the operator's.** Rules are evaluated top to
bottom and the first one whose operation set and path pattern both match decides;
nothing further is consulted. That makes a narrow `allow` above a broad `deny`
the way an exception is written, which is how every path ACL anyone has used
behaves. An empty rule list therefore allows everything, and layering this row
changes nothing until a profile writes rules — the same posture `hitl` takes.

**Enumeration is filtered rather than asked about.** `glob` and `grep` go through
`FsService.hide`, a synchronous predicate consulted during the walk, because a
walk visits thousands of candidates: a per-path prompt is not something a person
would sit through, and post-filtering `grep`'s matches would hide the rows while
having already read every byte of the protected file. So for enumeration, `deny`
and `interrupt` both conceal — you cannot approve nine hundred paths, and
concealing is the answer that does not leak the thing being protected.

**Recursive delete fails closed.** Deleting is a `write` — the port plan's own
two-operation vocabulary, and the reason it is not a third: with a separate
`delete` operation the obvious rule
`{operations: ["write"], paths: ["secrets/**"], mode: "deny"}` would still have
permitted deleting `secrets/`. What differs is the *blast radius*, not the
permission: a rule set is a statement about paths, and a recursive delete is a
statement about a subtree, so "may I delete this tree" cannot be answered by
matching the tree's own path — the paths the rules describe need not exist yet,
and the tree is enumerated by the operating system at delete time.
`deletion_reason` therefore refuses whenever a non-allow rule *could* match any
descendant, judged from the pattern's literal head, which is conservative in the
only direction a delete may be wrong in. pH registers no delete tool and this row
does not add one — a permission row that grows a capability is the wrong row — so
this has no caller yet and is tested directly.

**The row says what it does not cover (E9).** These rules bound *seam-mediated*
access. A model-authored `open(path, "w")` inside a code cell, or a `subprocess`
that shells out, never fires an intent and is not touched — N1, and no wording
here should suggest otherwise. When no confining provider is mounted the row says
so at mount and carries the sentence on `ctx.fs_permissions.reach`, so the
statement toggles with the sandbox rather than being a paragraph in a README that
is wrong half the time.

@module ph_stabilize.permissions_fs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ph.cordis import Context, plugin
from ph.seams.approval import denial_reason
from ph.seams.fs import EditIntent, FsService, ReadIntent, WriteIntent, matches_glob
from ph.wire import WireModel

__all__ = [
    "Config",
    "Decision",
    "FsPermissions",
    "Operation",
    "Rule",
    "apply",
]

log = logging.getLogger("ph_stabilize.permissions_fs")

Operation: TypeAlias = Literal["read", "write"]
"""What a rule may speak about — two questions, as the port plan defines them.

`write` is every mutation: create, overwrite, in-place edit, **and delete**. A
rule set that split them would be one where "do not write here" left `edit`
open, and — the trap that decided it — where the obvious
`{operations: ["write"], paths: ["secrets/**"], mode: "deny"}` still permitted
deleting `secrets/`. Deletion has a different *blast radius*, which is why
`deletion_reason` asks a different question; it is not a different permission.
"""

Decision: TypeAlias = Literal["allow", "deny", "interrupt"]
"""What a rule says to do. `interrupt` asks a human through `ctx.approval`, which
is the same route `hitl` uses — one approval channel, so a deployment configures
its answerer once."""

UNBOUNDED_REACH = (
    "permissions-fs applies to tool calls through ctx.fs; raw open()/subprocess "
    "from inside a code cell is not covered. Mount a sandbox provider to bound it."
)
"""E9's sentence, said when no confining provider is mounted."""

BOUNDED_REACH = (
    "permissions-fs applies to tool calls through ctx.fs; a sandbox provider "
    "bounds what a code cell can reach directly."
)
"""And the sentence that replaces it once one is."""

DENIAL = "{operation} denied by permissions-fs: {path}"
INTERRUPT_REASON = "{operation} {path} — permissions-fs asks about this path"
RECURSIVE_DENIAL = (
    "recursive delete of {path} refused: permissions-fs has a rule that could match "
    "something inside it, and a subtree cannot be checked one path at a time"
)


class Rule(WireModel):
    """One row of the ACL."""

    operations: tuple[Operation, ...] = ("read", "write")
    """Which operations this rule speaks about. Both by default, so a rule that
    names only paths reads as "this path, entirely" rather than silently covering
    whichever operation the author happened to think of first."""
    paths: tuple[str, ...] = ()
    """Glob patterns, matched by `ph.seams.fs.matches_glob` — the *same* matcher
    the `glob` tool uses, so `**/*.env` cannot mean two things in one harness.

    Each candidate is offered twice: as its absolute posix path, and as its path
    relative to the workspace root when it lies under one. That is what lets
    `.env` and `/etc/**` both be written the obvious way. Empty matches nothing,
    because a rule with no paths is far more likely to be unfinished config than
    a deliberate match-everything."""
    mode: Decision = "deny"
    """What to do on a match. `deny` by default: a rule someone wrote and left
    half-configured should be the restrictive one."""
    description: str = ""
    """Words for the prompt an `interrupt` raises, when the path is not enough."""


class Config(WireModel):
    """Row config."""

    rules: tuple[Rule, ...] = ()
    """Evaluated in order, first match wins, default allow. Empty by default:
    layering the bundle must not start refusing file access."""


@dataclass(frozen=True, slots=True)
class FsPermissions:
    """The row's resolved policy, published as `ctx.fs_permissions`.

    Provided rather than kept in a closure because the reach sentence is only
    worth anything if something can read it back, and `ph-app` cannot import this
    package. That makes this the fourth row wanting to hand `ph doctor` a
    reading — after the sandbox tier, the workspace kind and the worker model —
    which is `ctx.tui_status`' problem again and wants the same answer: a
    registry of named readings, contributed with `scope=`, iterated once by
    whoever is printing. P4-12 owns that seam; until it exists this is a bespoke
    name, and saying so is cheaper than pretending it was the plan.
    """

    rules: tuple[Rule, ...]
    root: Path
    ctx: Context | None = None
    """The activation scope, held so `reach` stays *live*.

    A sentence computed at mount would be the wrong one for a profile that layers
    its sandbox after this row, and E9 asks for a statement that toggles rather
    than a paragraph that was true once. Asked of `ctx` rather than of a seam
    object captured at mount, because that captures `None` forever when the
    sandbox seam itself is layered later. `None` — no scope, as in the policy
    tests — reports unconfined, which is the fail-closed direction and true.
    """
    _prefix: str = field(default="", init=False, repr=False, compare=False)
    """`root` as a posix path with a trailing slash, computed once.

    `_spellings` runs per gated read *and per file a walk visits*, and
    `Path.relative_to` allocates a `Path` per root segment on every call — 26 µs
    at a five-segment root, which measured as 92% of the concealment check and
    nearly doubled the wall time of a repository-wide `grep`. A string prefix
    strip is the same answer for 0.16 µs.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "_prefix", f"{self.root.as_posix().rstrip('/')}/")

    @property
    def confined(self) -> bool:
        """Whether a backend is mounted that bounds what this row cannot."""
        sandbox = None if self.ctx is None else self.ctx.get("sandbox")
        return sandbox is not None and bool(sandbox.available)

    @property
    def reach(self) -> str:
        """What these rules do and do not cover, as it stands (E9)."""
        return BOUNDED_REACH if self.confined else UNBOUNDED_REACH

    def decide(self, operation: Operation, path: Path) -> Rule | None:
        """The first rule that matches, or `None` for the default allow."""
        candidates = self._spellings(path)
        for rule in self.rules:
            if operation not in rule.operations:
                continue
            if any(
                matches_glob(candidate, pattern)
                for pattern in rule.paths
                for candidate in candidates
            ):
                return rule
        return None

    def objection(self, operation: Operation, path: Path) -> Rule | None:
        """The first rule that would *stop* this, or `None`.

        `decide` answers "which rule spoke"; this answers "did it say no", which
        is the question all three consumers actually ask. One spelling, because
        three hand-written `rule is not None and rule.mode != "allow"` checks are
        three edit sites the day `Decision` grows a fourth member.
        """
        rule = self.decide(operation, path)
        return rule if rule is not None and rule.mode != "allow" else None

    def conceals(self, path: Path) -> bool:
        """Whether a listing must not reveal this path.

        `interrupt` conceals as well as `deny`: enumeration cannot ask, and the
        alternative — showing a path the rules said to check with a human — is
        the leak the rule was written to prevent.
        """
        return self.objection("read", path) is not None

    def deletion_reason(self, path: Path, *, recursive: bool) -> str | None:
        """Why this delete must not happen, or `None`.

        A non-recursive delete is an ordinary first-match question. A recursive
        one is refused whenever a non-allow rule *could* match a descendant,
        because the rules describe paths that do not exist yet — the tree is
        walked by the operating system, not by this module, and a rule matching
        one file inside it is a rule the delete would violate without ever
        asking about that file.
        """
        if self.objection("write", path) is not None:
            return DENIAL.format(operation="delete", path=path)
        if not recursive:
            return None
        # Both spellings, for `decide`'s reason: a rule written `build/**` and a
        # rule written `/w/build/**` are the same rule, and a check that knew
        # only one of them would let the other kind of config through.
        spellings = self._spellings(path)
        for candidate in self.rules:
            if candidate.mode == "allow" or "write" not in candidate.operations:
                continue
            if any(
                _could_match_under(pattern, directory)
                for pattern in candidate.paths
                for directory in spellings
            ):
                return RECURSIVE_DENIAL.format(path=path)
        return None

    def _spellings(self, path: Path) -> tuple[str, ...]:
        """Both ways to name this path: absolute, and relative to the workspace.

        A prefix strip rather than `Path.relative_to` — see `_prefix`. A path
        outside the workspace has no relative spelling, and an absolute rule is
        the only honest way to name one anyway.
        """
        absolute = path.as_posix()
        if absolute.startswith(self._prefix):
            return (absolute, absolute[len(self._prefix) :])
        return (absolute,)


def _could_match_under(pattern: str, directory: str) -> bool:
    """Whether `pattern` could match anything inside `directory`.

    Judged from the pattern's literal head — everything before the first
    wildcard — which is the most a glob will tell you without enumerating the
    tree. Two ways to be inside: the head already points into the tree, or the
    head is an ancestor of it and the wildcard is free to descend. Both round
    towards refusal, which is the only direction a delete may be wrong in.
    """
    head = pattern.split("*", 1)[0].rstrip("/")
    if not head:
        return True  # a leading wildcard reaches everywhere
    return _under(head, directory) or _under(directory, head)


def _under(inner: str, outer: str) -> bool:
    """Path containment on separators, so `/home/xyz` is not inside `/home/x`."""
    return inner == outer or inner.startswith(f"{outer}/")


@plugin("permissions-fs", inject=["fs"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Attach the rules to `ctx.fs`."""
    fs: FsService = ctx.fs
    permissions = FsPermissions(rules=config.rules, root=fs.root, ctx=ctx)
    ctx.provide("fs_permissions", permissions)
    if not config.rules:
        # Nothing to enforce, so nothing is attached — not even a predicate that
        # would return False. `hide` is consulted once per file a walk visits,
        # and a `grep` over a repository should not pay a Python call per
        # candidate for a rule set nobody wrote. The service is still published,
        # because "no rules" is an answer `ph doctor` wants to be able to give.
        return
    if not permissions.confined:
        # Said at mount, and only when there are rules to be wrong about: an
        # operator who wrote a deny list deserves to be told what it does not
        # reach before they rely on it rather than after. `ph doctor` (P4-12)
        # asks `ctx.fs_permissions.reach` for the same sentence on demand.
        log.warning("ph_stabilize.permissions_fs: %s", permissions.reach)

    ctx.on("fs/read-intent", _gate(ctx, permissions, "read"))
    # `write` twice: a rule saying "do not write here" that left `edit` open
    # would be a rule nobody meant to write.
    ctx.on("fs/write-intent", _gate(ctx, permissions, "write"))
    ctx.on("fs/edit-intent", _gate(ctx, permissions, "write"))
    fs.hide(permissions.conceals, scope=ctx)


def _gate(ctx: Context, permissions: FsPermissions, operation: Operation) -> Any:
    """One intent listener, holding only what it reads.

    A closure over three small values rather than a class, and `ctx` rather than
    a captured `ctx.approval`: the seam may be provided by a row layered after
    this one, and a lookup at mount would have captured `None` and refused every
    `interrupt` for the life of the process.
    """

    async def gate(intent: ReadIntent | WriteIntent | EditIntent, next_: Any) -> Any:
        rule = permissions.objection(operation, intent.path)
        if rule is None:
            return await next_()
        if rule.mode == "interrupt":
            granted = await _ask(ctx, intent, rule, operation)
            if granted is None:
                return await next_()
            return granted
        return DENIAL.format(operation=operation, path=intent.path)

    return gate


async def _ask(ctx: Context, intent: Any, rule: Rule, operation: Operation) -> str | None:
    """Route an `interrupt` to whoever answers approvals. `None` means granted.

    Fails closed on both halves of "there is nobody to ask", the way
    `ToolRuntime._service_ask` and the RLM harness's own gate do — **a missing
    agent denies as surely as a missing service**. That second one is not
    pedantry: `ApprovalService.request` derives the session from the agent, so an
    agentless ask would prompt a human with nothing written to the log and would
    walk straight past the `approval_policy: "never"` short-circuit that a
    deployment set deliberately.

    The refusal says *which* refusal it was, through the seam's own sentences: a
    model that cannot tell a human's "no" from a missing channel cannot tell
    which one is worth re-planning around.
    """
    approval = ctx.get("approval")
    subject = f"{operation} of {intent.path}"
    if approval is None or intent.agent is None:
        return denial_reason("unavailable", subject)
    outcome = await approval.request(
        agent=intent.agent,
        tool_name=f"fs.{operation}",
        reason=rule.description or INTERRUPT_REASON.format(operation=operation, path=intent.path),
        cancel=getattr(intent.agent, "signal", None),
        allowed_decisions=("approve", "reject"),
    )
    return None if outcome == "allowed-once" else denial_reason(outcome, subject)
