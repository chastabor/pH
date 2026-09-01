"""`permissions-fs` — path rules over filesystem access (P4-06, G7, E9).

`FilesystemPermission {operations, paths, mode}` evaluated **first-match-wins**
with a default of `allow`, attached to the `ctx.fs` seam rather than to the tools
that happen to call it. That placement is the row: the seam fires
`fs/read-intent`, `fs/write-intent` and `fs/edit-intent` before anything reaches
disk, so anything going through `ctx.fs` is covered without this module knowing a
single tool name.

**First-match-wins, and the order is the operator's.** Rules are evaluated top to
bottom and the first whose operation set and path pattern both match decides. A
narrow `allow` above a broad `deny` is how an exception is written. An empty rule
list allows everything, so layering this row changes nothing until a profile
writes rules.

**Enumeration is filtered rather than asked about**, through `FsService.screen` —
a synchronous decision consulted during the walk. For enumeration `deny` and
`interrupt` both conceal: you cannot approve nine hundred paths, and concealing is
the answer that does not leak the thing being protected.

**Recursive delete fails closed.** Deleting is a `write`, not a third operation:
what differs is the *blast radius*, not the permission. "May I delete this tree"
cannot be answered by matching the tree's own path, because the paths the rules
describe need not exist yet and the tree is enumerated by the operating system at
delete time. So `deletion_reason` refuses whenever a non-allow rule *could* match
any descendant, judged from the pattern's literal head — conservative in the only
direction a delete may be wrong in. pH registers no delete tool and this row does
not add one, so this has no caller yet and is tested directly.

**What this does not cover (E9).** These rules bound *seam-mediated* access. A
model-authored `open(path, "w")` inside a code cell, or a `subprocess` that shells
out, never fires an intent and is not touched — N1, and no wording here should
suggest otherwise. When no confining provider is mounted the row says so at mount
and carries the sentence on `ctx.fs_permissions.reach`, so the statement toggles
with the sandbox rather than being a paragraph in a README that is wrong half the
time.

How much that buys today, and which readers in the tree are *not* behind this
gate: `tests/test_permissions_fs.py`.

@module ph_stabilize.permissions_fs
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ph.cordis import Context, plugin
from ph.paths import is_under
from ph.seams.approval import denial_reason
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import (
    EditIntent,
    FsService,
    ReadIntent,
    WalkDecision,
    WriteIntent,
    matches_glob,
)
from ph.seams.sandbox import enforcement_of
from ph.seams.workspace import workspace_of, writable_roots
from ph.wire import WireModel

__all__ = [
    "Config",
    "Decision",
    "FsPermissions",
    "Operation",
    "Rule",
    "Scope",
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

Scope: TypeAlias = Literal["anywhere", "outside-workspace"]
"""Whether a rule speaks about every path or only about ones leaving the agent's
own workspace. A closed vocabulary, so a `Literal` — the rule this package holds
`Decision`, `ApprovalMode` and `ContainmentTier` to."""

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
OUTSIDE_REASON = "{operation} {path} — outside this agent's workspace ({root}) and its scratch"
"""What a `scope: outside-workspace` rule asks.

The path *and* the boundary it is leaving, because "approve this write?" with no
frame is a question nobody can answer well — and the whole point of prompting
rarely is that the rare prompt carries enough to decide on."""
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
    scope: Scope = "anywhere"
    """Which paths this rule is *eligible* for, before its globs are consulted.

    `outside-workspace` is what makes E6's default write scope a rule rather
    than a second gate: the agent's own tree is per-agent and unnameable in a
    static `paths:` list, but "is this path outside the tree the seam gave this
    agent" is a question `decide` can now ask, because it has the agent and the
    per-agent root. One first-match-wins list, one prompt, and precedence that
    comes from the rule's position rather than from which row a bundle happened
    to layer first — two gates on one waterfall queue rather than compose, and a
    write refused by one and asked about by the other prompts *twice*.
    """
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
    package. Being the fourth row to want that is what bought `ctx.diagnostics`
    (P4-12) — `ctx.tui_status`' shape, minus the `Session` — so `ph doctor`
    prints `reach` without knowing this module exists. The name stays published
    as well: `describe` is for a person reading a report, and `objection` is for
    the three callers that need a verdict.
    """

    rules: tuple[Rule, ...]
    roots: Callable[[Any], Path]
    """`ctx.fs.root_for` — the acting agent's root, asked rather than captured.

    The one member of `ctx.fs` this needs, so it is that member and not the
    service: a `frozen=True` value holding a `slots=True` service is unhashable
    and compares by a live mutable object, and typing it `Any` was what made the
    tests build a stub class to stand in for one call.

    A root read once was right until D21: under the `worktree` tier an agent's
    paths sit *outside* `fs.root`, so `_spellings` offered only the absolute
    form and every anchored rule — `secrets/*`, a bare `.env` — silently stopped
    applying inside a worktree, which is the one place a rule is most needed. The
    seam already answers this per agent (`root_for`), and the intents already
    carry the agent."""
    ctx: Context | None = None
    """The activation scope, held so `reach` stays *live*.

    A sentence computed at mount would be the wrong one for a profile that layers
    its sandbox after this row, and E9 asks for a statement that toggles rather
    than a paragraph that was true once. Asked of `ctx` rather than of a seam
    object captured at mount, because that captures `None` forever when the
    sandbox seam itself is layered later. `None` — no scope, as in the policy
    tests — reports unconfined, which is the fail-closed direction and true.
    """

    def _prefix(self, agent: Any) -> str:
        return _prefix_of(self.roots(agent))

    @property
    def confined(self) -> bool:
        """Whether something actually bounds what this row cannot.

        `full`, not merely *mounted*: E9's sentence promises a person that "a
        sandbox provider bounds what a code cell can reach directly", and a
        backend enforcing `partial` does not — saying so would be the tier-name
        overstatement E1 exists to prevent, and it would contradict `strict`,
        which refuses a partial backend outright.
        """
        return self.ctx is not None and enforcement_of(self.ctx) == "full"

    @property
    def reach(self) -> str:
        """What these rules do and do not cover, as it stands (E9)."""
        return BOUNDED_REACH if self.confined else UNBOUNDED_REACH

    def describe(self) -> list[tuple[str, str]]:
        """What `ph doctor` prints about file permissions (E9).

        **The reach sentence even when there are no rules**, which is the
        counter-intuitive half: a deployment that wrote nothing has a *wider*
        reach than one that wrote a deny list, and a report that stayed silent
        about the empty case would only tell people what they cannot do — never
        that nothing was stopping them.

        Read live, not at mount, for the reason `confined` is a property: a
        profile may layer its sandbox after this row.
        """
        rows = [
            ("rules", str(len(self.rules)) if self.rules else "none — nothing is refused here"),
            ("reach", self.reach),
        ]
        for rule in self.rules:
            # The paths are the label and the verdict is the value, because
            # "what happens to this path" is the question a person is reading
            # the report to answer — and it keeps every contributed row a plain
            # pair rather than a nesting convention the printer has to learn.
            operations = "/".join(rule.operations)
            where = " outside the workspace" if rule.scope == "outside-workspace" else ""
            rows.append((", ".join(rule.paths), f"{rule.mode} on {operations}{where}"))
        return rows

    def decide(self, operation: Operation, path: Path, agent: Any = None) -> Rule | None:
        """The first rule that matches, or `None` for the default allow."""
        posix = path.as_posix()
        return self._decide_from(self._spellings(posix, agent), operation, posix, agent)

    def _decide_from(
        self, candidates: tuple[str, ...], operation: Operation, path: str, agent: Any
    ) -> Rule | None:
        """`decide` over spellings already computed, for a caller that has them.

        The split exists for the walk. `screen` asks two questions about one
        path — "does a rule name this" and "could a rule match inside it" — and
        both need the spellings, so deriving them twice paid `_spellings` per
        candidate on the hot path P4-06 measured and P6-17 rewrote. `decide`
        keeps its `Path` signature, because the gate has a `Path` and one
        `as_posix()` per gated read is not worth a second public shape.
        """
        outside = None
        for rule in self.rules:
            if operation not in rule.operations:
                continue
            if rule.scope == "outside-workspace":
                # Computed at most once per decision, and only if a scoped rule
                # is actually reached: the common list has none.
                if outside is None:
                    outside = self._outside_workspace(Path(path), agent)
                if not outside:
                    continue
            if any(
                matches_glob(candidate, pattern)
                for pattern in rule.paths
                for candidate in candidates
            ):
                return rule
        return None

    def _objection_to(
        self, candidates: tuple[str, ...], operation: Operation, path: str, agent: Any
    ) -> Rule | None:
        """`objection` over spellings already computed. See `_decide_from`."""
        rule = self._decide_from(candidates, operation, path, agent)
        return rule if rule is not None and rule.mode != "allow" else None

    def objection(self, operation: Operation, path: Path, agent: Any = None) -> Rule | None:
        """The first rule that would *stop* this, or `None`.

        `decide` answers "which rule spoke"; this answers "did it say no", which
        is the question all three consumers actually ask. One spelling, because
        three hand-written `rule is not None and rule.mode != "allow"` checks are
        three edit sites the day `Decision` grows a fourth member.
        """
        posix = path.as_posix()
        return self._objection_to(self._spellings(posix, agent), operation, posix, agent)

    def conceals(self, path: Path, agent: Any = None) -> bool:
        """Whether a listing must not reveal this path.

        `interrupt` conceals as well as `deny`: enumeration cannot ask, and the
        alternative — showing a path the rules said to check with a human — is
        the leak the rule was written to prevent.
        """
        return self.objection("read", path, agent) is not None

    def screen(self, path: str, name: str, agent: Any, is_dir: bool) -> WalkDecision:
        """What the walk may do with this path — the `ctx.fs.screen` contract (P6-19).

        Two questions, not one. For a file it is `conceals`. For a **directory** it is
        the question `deletion_reason` asks about a recursive delete, and for the same
        reason: the rules describe paths that do not exist yet, so "is anything in here
        concealed" cannot be answered by matching the directory's own path. `deny
        secrets/**` does not match `secrets` — `**` names what is *under* it — so a
        concealment check on the directory says no, the walk descends, and the rule is
        then enforced once per file inside a tree it should never have entered.

        `_refuses_under` is that judgement. It rounds towards refusal, which for
        enumeration is the same direction it rounds for deletes: a directory pruned
        because a rule *might* conceal something inside costs a listing that omits paths
        a person could have seen, while entering one costs the leak the rule was written
        to prevent.
        """
        spellings = self._spellings(path, agent)
        named = self._objection_to(spellings, "read", path, agent) is not None
        if not is_dir:
            return "skip" if named else "yield"
        # The directory's *own* path first — a rule may name it outright — then
        # the question only a directory raises. Two passes because they are two
        # questions: matching the directory catches a mid-segment wildcard like
        # `sec*ts`, whose literal head (`sec`) is not a path ancestor of
        # `secrets` and which `_could_match_under` therefore misses.
        if named:
            return "prune"
        return (
            "prune"
            if self._refuses_under(
                spellings, "read", path, agent, honour_scope=True, require_head=True
            )
            else "yield"
        )

    def _refuses_under(
        self,
        spellings: tuple[str, ...],
        operation: Operation,
        path: str,
        agent: Any,
        *,
        honour_scope: bool,
        require_head: bool,
    ) -> bool:
        """Whether a non-allow rule could match *something inside* this directory.

        The rules describe paths that do not exist yet — the tree is walked by the
        operating system, not by this module — so a rule matching one file inside a
        directory applies to it without ever naming it.

        **`honour_scope` is the deliberate difference between the two callers**, a
        parameter rather than a comment in each copy so the asymmetry is visible from
        both. Enumeration honours `outside-workspace`; a recursive delete does not. A
        delete may only ever be wrong towards refusal, while enumeration over-refusing
        *hides files a person may see*.

        **`require_head` is the second difference, and the same asymmetry.**
        `_could_match_under` answers `True` for a pattern with no literal head, because
        for a delete "could match anywhere" must round to refusal. For a walk that same
        rounding refuses *everywhere*: `deny read **/.env` — the idiomatic spelling — has
        an empty head, so every directory "could" hold a match and the whole tree is
        pruned. The file such a rule names is still concealed on the way past; what a
        leading wildcard cannot justify is refusing to *look*.
        """
        outside = None
        for rule in self.rules:
            if rule.mode == "allow" or operation not in rule.operations:
                continue
            if honour_scope and rule.scope == "outside-workspace":
                # `decide`'s own guard. Load-bearing here in a way it is not for
                # a file: a scoped rule written `paths: ["**"]` is precisely the
                # rule written to *exempt* the workspace, and without this it
                # would prune every directory in it.
                if outside is None:
                    # The one branch that still wants a `Path`, and it is rare
                    # by construction: only a scoped rule reaches it.
                    outside = self._outside_workspace(Path(path), agent)
                if not outside:
                    continue
            if any(
                (_has_head(pattern) or not require_head) and _could_match_under(pattern, directory)
                for pattern in rule.paths
                for directory in spellings
            ):
                return True
        return False

    def deletion_reason(self, path: Path, *, recursive: bool, agent: Any = None) -> str | None:
        """Why this delete must not happen, or `None`.

        A non-recursive delete is an ordinary first-match question. A recursive
        one is refused whenever a non-allow rule *could* match a descendant —
        `_refuses_under`, which `screen` shares and which records why the two
        callers differ.

        `honour_scope=False`: a rule scoped `outside-workspace` still refuses a
        recursive delete of a directory inside it, because the delete would
        reach paths outside as soon as the tree contains a symlink or the
        workspace boundary moves. The enumeration side, which can be wrong in
        the direction of hiding a person's own files, honours the scope.
        """
        if self.objection("write", path, agent) is not None:
            return DENIAL.format(operation="delete", path=path)
        if not recursive:
            return None
        if self._refuses_under(
            self._spellings(path.as_posix(), agent),
            "write",
            path.as_posix(),
            agent,
            honour_scope=False,
            require_head=False,
        ):
            return RECURSIVE_DENIAL.format(path=path)
        return None

    def prompt(self, rule: Rule, operation: Operation, path: Path, agent: Any) -> str:
        """The sentence an `interrupt` puts in front of a person.

        A rule's own `description` wins where it has one. Otherwise a scoped rule
        names the boundary rather than only the path: the default write scope
        prompts *because* a write is leaving the agent's tree, and a prompt that
        did not say which tree would be the rare interruption arriving without
        the one fact it exists to convey.
        """
        if rule.description:
            return rule.description
        if rule.scope == "outside-workspace":
            workspace = None if self.ctx is None else workspace_of(self.ctx, agent)
            if workspace is not None:
                return OUTSIDE_REASON.format(operation=operation, path=path, root=workspace.root)
        return INTERRUPT_REASON.format(operation=operation, path=path)

    def _outside_workspace(self, path: Path, agent: Any) -> bool:
        """Whether this write is leaving the workspace the seam gave this agent.

        `None` — no workspace — means there is no scope to be outside of, so a
        scoped rule simply does not apply: a profile layering the rule without a
        containment tier gets today's behaviour rather than a boundary drawn
        around a directory nobody chose.
        """
        workspace = None if self.ctx is None else workspace_of(self.ctx, agent)
        if workspace is None:
            return False
        return not any(is_under(path, root) for root in writable_roots(workspace))

    def _spellings(self, absolute: str, agent: Any = None) -> tuple[str, ...]:
        """Both ways to name this path: absolute, and relative to the workspace.

        A prefix strip rather than `Path.relative_to` — see `_prefix`. A path
        outside the workspace has no relative spelling, and an absolute rule is
        the only honest way to name one anyway.

        Takes the posix **string**, not a `Path`, because that is what it
        immediately made of one and what both callers already hold: `decide` has
        a `Path` and spends one `as_posix()`, while `screen` is handed the string
        by the walk (P6-17's finding, applied to the seam's own contract).
        """
        prefix = self._prefix(agent)
        if absolute.startswith(prefix):
            return (absolute, absolute[len(prefix) :])
        return (absolute,)


@lru_cache(maxsize=256)
def _prefix_of(root: Path) -> str:
    """`root` as a posix path with a trailing slash.

    Cached because `_spellings` runs per gated read *and per file a walk visits*
    — P4-06 measured `Path.relative_to` at 26 µs there, 92% of the check — and
    `lru_cache` rather than a field on the value because the bound is then the
    cache size rather than "every root ever seen", and because a mutable cache
    on a frozen dataclass is a public constructor argument nobody meant to add.
    """
    return f"{root.as_posix().rstrip('/')}/"


def _has_head(pattern: str) -> bool:
    """Whether `pattern` names anything literal before its first wildcard.

    Separated from `_could_match_under` rather than folded into it because the
    two callers want opposite answers for a headless pattern: a delete must
    treat "could match anywhere" as a refusal, and a walk must not treat it as a
    reason to enter nothing. See `_refuses_under`.
    """
    return bool(pattern.split("*", 1)[0].rstrip("/"))


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
    permissions = FsPermissions(rules=config.rules, roots=fs.root_for, ctx=ctx)
    ctx.provide("fs_permissions", permissions)

    # Offered before the no-rules return below: "nothing is refused here, and
    # here is how far that goes" is the reading a person most needs, and it is
    # the one an early return would have silently dropped.
    contribute(
        ctx,
        Diagnostic(
            id="permissions-fs", title="File permissions", read=permissions.describe, order=30
        ),
    )
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
    # `screen`, not the retired `hide`: this row can now refuse a *tree* rather
    # than the same tree once per file in it (P6-19), and the `scope=` it has
    # always passed finally does something on the enumeration side too — an
    # agent-scoped mount screens its own agent's walks rather than everybody's
    # (P6-18).
    fs.screen(permissions.screen, scope=ctx)


def _gate(ctx: Context, permissions: FsPermissions, operation: Operation) -> Any:
    """One intent listener, holding only what it reads.

    A closure over three small values rather than a class, and `ctx` rather than
    a captured `ctx.approval`: the seam may be provided by a row layered after
    this one, and a lookup at mount would have captured `None` and refused every
    `interrupt` for the life of the process.
    """

    async def gate(intent: ReadIntent | WriteIntent | EditIntent, next_: Any) -> Any:
        rule = permissions.objection(operation, intent.path, intent.agent)
        if rule is None:
            return await next_()
        if rule.mode == "interrupt":
            granted = await _ask(ctx, permissions, intent, rule, operation)
            if granted is None:
                return await next_()
            return granted
        return DENIAL.format(operation=operation, path=intent.path)

    return gate


async def _ask(
    ctx: Context, permissions: FsPermissions, intent: Any, rule: Rule, operation: Operation
) -> str | None:
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
        reason=permissions.prompt(rule, operation, intent.path, intent.agent),
        cancel=getattr(intent.agent, "signal", None),
        allowed_decisions=("approve", "reject"),
    )
    return None if outcome == "allowed-once" else denial_reason(outcome, subject)
