"""`permissions-fs`: path rules over filesystem access (P4-06, G7, E9).

A rule is a `FilesystemPermission {operations, paths, mode}`. The rules are
evaluated **first-match-wins** with a default of `allow`. They are attached to
the `ctx.fs` seam and not to the tools that call it. That placement is the row.
The seam fires `fs/read-intent`, `fs/write-intent` and `fs/edit-intent` before
anything reaches the disk. Everything that goes through `ctx.fs` is therefore
covered, and this module does not know a single tool name.

**The first match wins, and the operator sets the order.** The rules are
evaluated from top to bottom. The first rule whose operation set and path
pattern both match decides. An exception is written as a narrow `allow` above a
broad `deny`. An empty rule list allows everything, so this row changes nothing
when it is layered, until a profile writes rules.

**Enumeration is filtered, not asked about.** The filter is `FsService.screen`,
a synchronous decision that the walk consults. For enumeration, `deny` and
`interrupt` both conceal. Nobody can approve nine hundred paths, and concealing
is the answer that does not leak the protected thing.

**A recursive delete fails closed.** A delete is a `write` and not a third
operation. The permission is the same. The *blast radius* is different. The
question "may I delete this tree" cannot be answered by a match on the path of
the tree itself. The paths that the rules describe need not exist yet, and the
operating system enumerates the tree at delete time. `deletion_reason` therefore
refuses whenever a non-allow rule *could* match any descendant. It judges that
from the pattern, one path segment at a time. It is conservative in the only
direction in which a delete may be wrong. pH registers no delete tool and this
row does not add one, so `deletion_reason` has no caller yet and is tested
directly.

**What this does not cover (E9).** These rules bound access *through the seam*.
A model-authored `open(path, "w")` inside a code cell never fires an intent. A
`subprocess` that runs a shell command does not fire one either. *These rules* do
not touch either of them (N1), and no wording here may suggest otherwise. The
rung below the rules bounds them. `code-runtime-python` confines the kernel
itself, so the OS refuses a raw `open()` in a cell, and no rule ever saw that
call. `BOUNDED_REACH` has always promised this. Nothing delivered it until the
kernel was wrapped.

**The two do not turn on together, and this text must not imply that they do.**
`confined` below is `enforcement_of(ctx) == "full"`. The kernel is confined when
a backend is *available* and the acting agent has a workspace. So with a
`partial` backend, or with an agent that has no workspace, one is true and the
other is not. The Code runtime section of `phern doctor` reports what is in
force for each kernel. This row speaks only for these rules.

When no confining provider is mounted, the row says so at mount. It also keeps
the sentence on `ctx.fs_permissions.reach`. The statement therefore changes with
the sandbox. A paragraph in a README would be wrong half the time.

@module ph_stabilize.permissions_fs
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import field_validator

from ph.agent.types import AgentHandle
from ph.cordis import Context, Next, ServiceKey, plugin
from ph.keys import APPROVAL, FS
from ph.paths import canonical, is_under
from ph.seams.approval import denial_reason
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import (
    FsIntent,
    FsService,
    WalkDecision,
    covers_whole,
    glob_error,
    matches_glob,
)
from ph.seams.sandbox import allowed_paths_of, enforcement_of
from ph.seams.spill import handed_paths_of
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
"""What a rule can apply to. There are two operations, as the port plan defines
them.

`write` is every mutation: create, overwrite, in-place edit, **and delete**. If
a rule set split them, "do not write here" would leave `edit` open. One case
decided it: the obvious rule
`{operations: ["write"], paths: ["secrets/**"], mode: "deny"}` would still
permit a delete of `secrets/`. A deletion has a different *blast radius*, and
that is why `deletion_reason` asks a different question. It is not a different
permission.
"""

Scope: TypeAlias = Literal["anywhere", "outside-workspace"]
"""Whether a rule applies to every path, or only to paths that leave the
workspace of the agent. The vocabulary is closed, so the type is a `Literal`.
`Decision`, `ApprovalMode` and `ContainmentTier` follow the same rule."""

Decision: TypeAlias = Literal["allow", "deny", "interrupt"]
"""What a rule says to do. `interrupt` asks a human through `ctx.approval`, and
`hitl` uses the same route. There is one approval channel, so a deployment
configures its answerer one time."""

UNBOUNDED_REACH = (
    "permissions-fs applies to tool calls through ctx.fs; raw open()/subprocess "
    "from inside a code cell is not covered. Mount a sandbox provider to bound it."
)
"""The sentence of E9. The row reports it when no confining provider is mounted."""

BOUNDED_REACH = (
    "permissions-fs applies to tool calls through ctx.fs; a sandbox provider "
    "bounds what a code cell can reach directly."
)
"""The sentence that replaces it when a confining provider is mounted."""

DENIAL = "{operation} denied by permissions-fs: {path}"
INTERRUPT_REASON = "{operation} {path} — permissions-fs asks about this path"
OUTSIDE_REASON = "{operation} {path} — outside this agent's workspace ({root}) and its scratch"
"""What a `scope: outside-workspace` rule asks.

The text names the path *and* the boundary that the path leaves. Nobody can
answer "approve this write?" well without that frame. The purpose of prompting
rarely is that the rare prompt carries enough to decide on."""
RECURSIVE_DENIAL = (
    "recursive delete of {path} refused: permissions-fs has a rule that could match "
    "something inside it, and a subtree cannot be checked one path at a time"
)


class Rule(WireModel):
    """One row of the ACL."""

    operations: tuple[Operation, ...] = ("read", "write")
    """The operations that this rule applies to. The default is both. A rule that
    names only paths then means "this path, entirely". It does not silently cover
    only the operation that the author thought of first."""
    paths: tuple[str, ...] = ()
    """Glob patterns, matched by `ph.seams.fs.matches_glob`. The `glob` tool uses
    the *same* matcher, so `**/*.env` cannot mean two things in one harness.

    Each candidate is offered two times: as its absolute posix path, and as its
    path relative to the workspace root when it lies under one. Because of that,
    `.env` and `/etc/**` can both be written in the obvious way. An empty tuple
    matches nothing. A rule with no paths is far more likely to be unfinished
    config than a deliberate match on everything."""

    @field_validator("paths")
    @classmethod
    def _usable_globs(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a pattern that `matches_glob` could only match literally (N6).

        **The fallback of the matcher is right for a model and wrong for a
        rule.** A glob that does not compile matches itself and nothing else. The
        reason is that a pattern that the model typed must not raise out of the
        policy check that it passes through. Here the same fallback turns a
        `deny` into a rule that guards nothing. `[z-a]/**` is a bad character
        range. The rule stops matching `secrets/key.pem` and starts matching the
        eight literal characters, which nobody will ever use as a file name.

        The pattern is refused at config time, and not warned about at mount.
        The two populations of patterns meet here, and this is the population
        that an operator can still fix before anything relies on it.
        """
        for pattern in paths:
            because = glob_error(pattern)
            if because is not None:
                raise ValueError(f"{pattern!r} is not a usable glob pattern: {because}")
        return paths

    scope: Scope = "anywhere"
    """The paths that this rule is *eligible* for, before its globs are consulted.

    `outside-workspace` makes the default write scope of E6 a rule and not a
    second gate. The tree of an agent is per-agent, so a static `paths:` list
    cannot name it. But `decide` can now ask "is this path outside the tree that
    the seam gave this agent", because it has the agent and the per-agent root.
    The result is one first-match-wins list and one prompt. Precedence comes
    from the position of the rule, and not from which row a bundle layered
    first. Two gates on one waterfall run one after the other and do not
    compose, and a write that one gate refuses and the other gate asks about
    prompts *two times*.
    """
    mode: Decision = "deny"
    """What to do on a match. The default is `deny`: a rule that someone wrote and
    left half-configured must be the restrictive one."""
    description: str = ""
    """The text for the prompt that an `interrupt` raises, when the path is not
    enough."""


class Config(WireModel):
    """Row config."""

    rules: tuple[Rule, ...] = ()
    """Evaluated in order. The first match wins, and the default is allow. The
    list is empty by default: layering the bundle must not start to refuse file
    access."""


FS_PERMISSIONS: ServiceKey[FsPermissions] = ServiceKey("fs_permissions")
"""The permission table, for the rows and tools that consult it."""


@dataclass(frozen=True, slots=True)
class FsPermissions:
    """The resolved policy of the row, published as `ctx.fs_permissions`.

    The policy is provided as a service and not kept in a closure. The reach
    sentence is only useful if something can read it back, and `ph_app` cannot
    import this package. This was the fourth row to need that, and that need is
    why `ctx.diagnostics` (P4-12) exists. It has the shape of `ctx.tui_status`
    without the `Session`. So `phern doctor` prints `reach` and does not know
    that this module exists. The service stays published by name as well:
    `describe` is for a person who reads a report, and `objection` is for the
    callers that need a verdict.
    """

    rules: tuple[Rule, ...]
    roots: Callable[[Any], Path]
    """`ctx.fs.root_for`: the root of the acting agent, asked each time and not
    captured.

    This is the one member of `ctx.fs` that the policy needs, so the field holds
    that member and not the service. A `frozen=True` value that holds a
    `slots=True` service is unhashable and compares by a live mutable object.
    When the field was typed `Any`, the tests had to build a stub class to stand
    in for one call.

    A root that was read one time was correct until D21. Under the `worktree`
    tier, the paths of an agent are *outside* `fs.root`. `_spellings` then
    offered only the absolute form, and every anchored rule (`secrets/*`, a bare
    `.env`) silently stopped applying inside a worktree. That is the one place
    where a rule is most needed. The seam already answers this for each agent
    (`root_for`), and the intents already carry the agent."""
    ctx: Context | None = None
    """The activation scope, held so that `reach` stays *live*.

    A sentence computed at mount would be wrong for a profile that layers its
    sandbox after this row. E9 asks for a statement that changes with the
    deployment, and not for a paragraph that was true at one moment. The policy
    asks `ctx` and not a seam object captured at mount: when the sandbox seam
    itself is layered later, such a capture holds `None` forever. `None` here
    means no scope, as in the policy tests. The policy then reports unconfined,
    which is the fail-closed direction and is true.
    """

    def _prefix(self, agent: AgentHandle | None) -> str:
        return _prefix_of(self.roots(agent))

    @property
    def confined(self) -> bool:
        """Whether something really bounds what this row cannot bound.

        The test is `full`, and not merely *mounted*. The sentence of E9 promises
        that "a sandbox provider bounds what a code cell can reach directly". A
        backend that enforces `partial` does not do that. To say that it does
        would be the tier-name overstatement that E1 exists to prevent. It would
        also contradict `strict`, which refuses a partial backend outright.
        """
        return self.ctx is not None and enforcement_of(self.ctx) == "full"

    @property
    def reach(self) -> str:
        """What these rules cover and do not cover, at this moment (E9)."""
        return BOUNDED_REACH if self.confined else UNBOUNDED_REACH

    def describe(self) -> list[tuple[str, str]]:
        """What `phern doctor` prints about file permissions (E9).

        **The reach sentence is printed even when there are no rules.** This is
        the counter-intuitive half. A deployment that wrote nothing has a
        *wider* reach than one that wrote a deny list. A report that stayed
        silent about the empty case would only tell people what they cannot do.
        It would never tell them that nothing stopped them.

        The values are read live and not at mount, for the same reason that
        `confined` is a property: a profile may layer its sandbox after this row.
        """
        rows = [
            ("rules", str(len(self.rules)) if self.rules else "none — nothing is refused here"),
            ("reach", self.reach),
        ]
        for rule in self.rules:
            # The paths are the label and the verdict is the value. A person reads
            # the report to answer "what happens to this path". This also keeps
            # every contributed row a plain pair, so the printer does not have to
            # learn a nesting convention.
            operations = "/".join(rule.operations)
            where = " outside the workspace" if rule.scope == "outside-workspace" else ""
            rows.append((", ".join(rule.paths), f"{rule.mode} on {operations}{where}"))
        return rows

    def decide(
        self, operation: Operation, path: Path, agent: AgentHandle | None = None
    ) -> Rule | None:
        """The first rule that matches, or `None` for the default allow.

        The path is normalized here, where the `Path` of a caller arrives. See
        `_spellings`.
        """
        posix = posixpath.normpath(path.as_posix())
        return self._decide_from(self._spellings(posix, agent), operation, posix, agent)

    def _decide_from(
        self,
        candidates: tuple[str, ...],
        operation: Operation,
        path: str,
        agent: AgentHandle | None,
    ) -> Rule | None:
        """`decide` over spellings that are already computed, for a caller that
        has them.

        The split exists for the walk. `screen` asks two questions about one
        path: "does a rule name this" and "does a rule cover everything inside
        it". Both need the spellings. When each question derived them, every
        candidate paid for `_spellings` a second time, on the hot path that P4-06
        measured and P6-17 rewrote. `decide` keeps its `Path` signature, because
        the gate has a `Path`, and one `as_posix()` per gated read is not worth a
        second public shape.
        """
        outside = None
        for rule in self.rules:
            if operation not in rule.operations:
                continue
            if rule.scope == "outside-workspace":
                # Computed at most one time per decision, and only when a scoped
                # rule is reached. The common list has none.
                if outside is None:
                    outside = self._outside_workspace(Path(path), agent, operation)
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
        self,
        candidates: tuple[str, ...],
        operation: Operation,
        path: str,
        agent: AgentHandle | None,
    ) -> Rule | None:
        """`objection` over spellings already computed. See `_decide_from`."""
        rule = self._decide_from(candidates, operation, path, agent)
        return rule if rule is not None and rule.mode != "allow" else None

    def objection(
        self, operation: Operation, path: Path, agent: AgentHandle | None = None
    ) -> Rule | None:
        """The first rule that would *stop* this operation, or `None`.

        `decide` answers "which rule matched". This method answers "did the rule
        say no", which is the question that all three consumers ask. There is one
        implementation, because three hand-written
        `rule is not None and rule.mode != "allow"` checks are three places to
        edit on the day that `Decision` gets a fourth member.

        The path is normalized here, where the `Path` of a caller arrives. See
        `_spellings`.
        """
        posix = posixpath.normpath(path.as_posix())
        return self._objection_to(self._spellings(posix, agent), operation, posix, agent)

    def screen(self, path: str, name: str, agent: AgentHandle, is_dir: bool) -> WalkDecision:
        """What the walk may do with this path: the `ctx.fs.screen` contract (P6-19).

        There are two questions, not one. For a file, the question is whether a
        rule objects to a read of it. `interrupt` hides a file as well as `deny`,
        because enumeration cannot ask. The alternative is to show a path that
        the rules said to check with a human. That is the leak that the rule was
        written to prevent.

        For a **directory**, a match on the path of the directory itself is not
        enough, for the same reason as in `deletion_reason`: the rules describe
        paths that do not exist yet. `deny secrets/**` does not match `secrets`,
        because `**` names what is *under* it. A concealment check on the
        directory therefore says no and the walk descends. The rule is then
        enforced one time per file, inside a tree that the walk should never
        have entered.

        `_refuses_under` makes the judgment for a directory. `deletion_reason`
        uses the same function with a different question (D11). A delete refuses
        when a rule *could* match something inside. The walk prunes only when a
        rule covers the whole directory (`covers_whole`), because only then does
        the listing lose nothing that a person could have seen. A rule that
        covers part of a directory is enforced file by file as the walk passes.
        """
        spellings = self._spellings(path, agent)
        named = self._objection_to(spellings, "read", path, agent) is not None
        if not is_dir:
            return "skip" if named else "yield"
        # The path of the directory itself comes first, because a rule may name
        # it outright. Then the check is whether a rule covers everything inside
        # it. Pruning is lossless only in the second case. A rule that covers
        # *some* of a directory is enforced file by file as the walk passes. That
        # is `named` above, one level down.
        if named:
            return "prune"
        covered = self._refuses_under(
            spellings, "read", path, agent, honor_scope=True, ask=covers_whole
        )
        return "prune" if covered else "yield"

    def _refuses_under(
        self,
        spellings: tuple[str, ...],
        operation: Operation,
        path: str,
        agent: AgentHandle | None,
        *,
        honor_scope: bool,
        ask: Callable[[str, str], bool],
    ) -> bool:
        """Whether a non-allow rule answers `ask` about this directory.

        The rules describe paths that do not exist yet, because the operating
        system walks the tree and this module does not. A rule that matches one
        file inside a directory therefore applies to the directory, although the
        rule never names it.

        **`honor_scope` is the deliberate difference between the two callers.**
        It is a parameter, and not a comment in each copy, so that the asymmetry
        is visible from both. Enumeration honors `outside-workspace`. A recursive
        delete does not. A delete may only ever be wrong towards refusal, but
        enumeration that refuses too much *hides files that a person may see*.

        **`ask` is the second difference. It is a different question, and not a
        different rounding** (D11). A delete asks `_could_match_under`: could the
        rule name anything that this delete removes. It must refuse on "could".
        A walk asks `covers_whole`: does the rule name everything in here.
        Pruning is lossless only when it does. One function used to answer both,
        and a `require_head` flag adjusted it for the walk. No setting of the
        flag was right for both `deny read src/*.pem` and
        `deny read **/node_modules/**`.
        """
        outside = None
        for rule in self.rules:
            if rule.mode == "allow" or operation not in rule.operations:
                continue
            if honor_scope and rule.scope == "outside-workspace":
                # The same guard as in `decide`. It is essential here in a way
                # that it is not for a file. A scoped rule written
                # `paths: ["**"]` is precisely the rule that is written to
                # *exempt* the workspace. Without this guard, that rule would
                # prune every directory in the workspace.
                if outside is None:
                    # This is the one branch that still needs a `Path`. It is rare
                    # by construction: only a scoped rule reaches it.
                    outside = self._outside_workspace(Path(path), agent, operation)
                if not outside:
                    continue
            if any(ask(pattern, directory) for pattern in rule.paths for directory in spellings):
                return True
        return False

    def deletion_reason(
        self, path: Path, *, recursive: bool, agent: AgentHandle | None = None
    ) -> str | None:
        """Why this delete must not happen, or `None`.

        A non-recursive delete is an ordinary first-match question. A recursive
        delete is refused whenever a non-allow rule *could* match a descendant.
        `_refuses_under` makes that check. `screen` shares it, and its docstring
        records why the two callers differ.

        `honor_scope=False`: a rule scoped `outside-workspace` still refuses a
        recursive delete of a directory inside the workspace. The delete would
        reach paths outside as soon as the tree contains a symlink or the
        workspace boundary moves. The enumeration side can be wrong in the
        direction of hiding the files of the user, so it honors the scope.
        """
        if self.objection("write", path, agent) is not None:
            return DENIAL.format(operation="delete", path=path)
        if not recursive:
            return None
        # Normalize the path, as `objection` does (D6). A `Path` keeps a `..`
        # segment, and the spellings of `/w/x/../build` match no rule that names
        # `build`.
        posix = posixpath.normpath(path.as_posix())
        if self._refuses_under(
            self._spellings(posix, agent),
            "write",
            posix,
            agent,
            honor_scope=False,
            ask=lambda pattern, directory: _could_match_under(
                pattern, directory, at_root=directory == ""
            ),
        ):
            return RECURSIVE_DENIAL.format(path=path)
        return None

    def prompt(self, rule: Rule, operation: Operation, path: Path, agent: AgentHandle) -> str:
        """The sentence that an `interrupt` shows to a person.

        The `description` of a rule wins when the rule has one. Otherwise a
        scoped rule names the boundary and not only the path. The default write
        scope prompts *because* a write leaves the tree of the agent. A prompt
        that did not say which tree would be the rare interruption that arrives
        without the one fact that it exists to convey.
        """
        if rule.description:
            return rule.description
        if rule.scope == "outside-workspace":
            workspace = None if self.ctx is None else workspace_of(self.ctx, agent)
            if workspace is not None:
                return OUTSIDE_REASON.format(operation=operation, path=path, root=workspace.root)
        return INTERRUPT_REASON.format(operation=operation, path=path)

    def _handed_paths(self, agent: AgentHandle | None, operation: Operation) -> tuple[Path, ...]:
        """Paths that the harness gave to *this* agent, which a read may follow.

        The answer comes from `ph.seams.spill`, in the same way that
        `_outside_workspace` asks `ph.seams.sandbox` for the writable roots of
        the deployment. They are two named sets and not one tuple called `roots`,
        because they are two different claims: "may write here" and "was handed
        this". Folding them together is what made `_outside_workspace` stop
        meaning what its name says.

        Reads only. Nothing invites an agent to *write* into the store of the
        harness. An agent that could write there could forge a spilled blob
        under a locator that the log already names.

        With the shipped rule set, a read of a handed path works anyway, because
        that rule set restricts writes and nothing else. So this exemption
        changes no shipped behavior and is not meant to. What it changes is that
        the reach is *stated*. Without it, a deployment that adds
        `deny read outside-workspace` would break every retrieval hint in every
        session, silently, and would have no way to see why.

        **Attachments are deliberately absent.** This is structural and not an
        oversight. `AttachmentStore` never hands a path to the model: the bytes
        are carried in a `MediaBlock`, and `attach` returns the *workspace*
        path. An exemption for it would not restate the rule that its seam
        states (a model reaches a file through `ctx.fs`). It would break that
        rule.
        """
        if operation != "read" or self.ctx is None:
            return ()
        return handed_paths_of(self.ctx, session=None if agent is None else agent.session)

    def _outside_workspace(
        self, path: Path, agent: AgentHandle | None, operation: Operation
    ) -> bool:
        """Whether this operation leaves the workspace that the seam gave this agent.

        `None` (no workspace) means that there is no scope to be outside of, so a
        scoped rule does not apply. A profile that layers the rule without a
        containment tier keeps the behavior that it has today. It does not get a
        boundary drawn around a directory that nobody chose.
        """
        workspace = None if self.ctx is None else workspace_of(self.ctx, agent)
        if workspace is None:
            return False
        inside = writable_roots(workspace)
        if any(is_under(path, root) for root in inside):
            return False
        # The allowances of the deployment itself are inside too. The backend
        # binds them writable, and a prompt about a write that the kernel would
        # permit describes a boundary that is not there (E6). They are asked only
        # after the workspace has said no. Nearly every screened path is inside
        # the workspace, and this is a stat per configured directory and, for a
        # narrowed agent, a walk of its scope.
        #
        # **The two sets are not quite one set, and the difference is worth
        # stating.** `effective` drops these allowances under `read-only`, where
        # the session has said that nothing is writable. So in that posture the
        # prompt boundary is *wider* than the enforced one, and the kernel
        # refuses a write here that nobody was asked about. It is wider in the
        # direction that fails closed, and this rule has no session to read the
        # mode from. A seam that answered "may this path be written, in this
        # mode" would merge the two. It is worth building when something needs
        # the mode for another reason.
        #
        # The answer is per agent. A child that an assigned profile narrowed is
        # asked about the directories that it lost, because its sandbox no longer
        # binds them.
        who = None if agent is None else agent.id
        allowed = () if self.ctx is None else allowed_paths_of(self.ctx, who)
        if any(is_under(path, root) for root in allowed):
            return False
        roots = (*inside, *allowed)
        # Asked only after the workspace itself has said no, which keeps it free.
        # Nearly every screened path is inside, and this is a service lookup on
        # a walk that reaches five figures of candidates.
        handed = self._handed_paths(agent, operation)
        if any(is_under(path, root) for root in handed):
            return False
        # **Only now is a syscall worth its cost.** Every root here is canonical,
        # and the candidate is not. `FsService.resolve` passes an absolute path
        # through untouched, deliberately (`docs/seams/fs.md`). So on macOS a
        # model-authored `/var/folders/…/repo/x` misses a workspace root spelled
        # `/private/var/folders/…/repo`, and the user would be prompted about a
        # write that the kernel permits. The two boundaries would disagree about
        # one file, which is the failure of E6 in the fail-safe direction.
        #
        # The path is resolved *after* the cheap compare and not before it, and
        # that is why the fix costs nothing. A path that is already inside
        # answers with no `realpath` at all, and this code runs on every gated
        # write. The remaining case was about to interrupt a person, and one
        # `realpath` is nothing beside that. The contract of `is_under` is kept:
        # neither side is resolved implicitly. This caller resolves, as the
        # `is_under` docstring says a caller must.
        #
        # **What this deliberately does not do**, so that nobody reads it as more
        # than it is. A path that is textually *inside* the tree is accepted
        # without resolving. So a symlink that is planted inside the workspace
        # and points out of it is not asked about. That is the hole that the
        # `is_under` docstring names, and it predates the canonical roots. To
        # close it means resolving every candidate. That is a syscall on every
        # gated write. It is also a change to what this boundary claims, and not
        # to how it spells things. `sandbox` is the tier that bounds an escaping
        # write (N2, E13). This tier bounds tool calls and says so.
        resolved = canonical(path)
        if resolved == path:
            return True
        return not any(is_under(resolved, root) for root in (*roots, *handed))

    def _spellings(self, absolute: str, agent: AgentHandle | None = None) -> tuple[str, ...]:
        """Both ways to name this path: absolute, and relative to the workspace.

        The relative form is a prefix strip and not `Path.relative_to`. See
        `_prefix_of`. A path outside the workspace has no relative spelling, and
        an absolute rule is the only correct way to name such a path anyway.

        The argument is the posix **string**, not a `Path`. The function made a
        string of a `Path` immediately, and the callers already hold one. A
        caller with a `Path`, such as `decide`, spends one `as_posix()`. The walk
        passes the string to `screen` (the finding of P6-17, applied to the
        contract of the seam itself).

        **The string is not normalized here** (D6). A rule matches the string as
        given. Nothing collapsed `..`, so `public/../secrets/key` bypassed
        `deny read secrets/**`: its relative spelling never matches the pattern.
        The collapse belongs at the entry points that take the `Path` of a
        caller (`decide`, `objection`, and `deletion_reason`), and not in this
        function. `screen` also
        calls this function. It runs *per walked file*, on strings that `os.walk`
        built from an already resolved base, and those strings can never carry a
        `..`. To normalize here was guaranteed dead work on the one path that is
        hot enough for `_prefix_of` to be cached.
        """
        prefix = self._prefix(agent)
        if absolute.startswith(prefix):
            return (absolute, absolute[len(prefix) :])
        if f"{absolute}/" == prefix:
            # The root itself, whose relative spelling is `""`. It is the one
            # path that the prefix test cannot match, because the prefix has the
            # trailing slash and the root does not (D11).
            return (absolute, "")
        return (absolute,)


@lru_cache(maxsize=256)
def _prefix_of(root: Path) -> str:
    """`root` as a posix path with a trailing slash.

    Cached because `_spellings` runs per gated read *and per file that a walk
    visits*. It is an `lru_cache` and not a field on the value, so the bound is
    the cache size and not "every root ever seen". Also, a mutable cache on a
    frozen dataclass is a public constructor argument that nobody meant to add.
    """
    return f"{root.as_posix().rstrip('/')}/"


def _could_match_under(pattern: str, directory: str, *, at_root: bool) -> bool:
    """Whether `pattern` could match `directory` or anything that a delete of it
    removes.

    **This is the existential question, and only a delete asks it** (D11). A
    recursive delete removes the directory and everything under it. It must be
    refused if the rule could name *any* of that. "Could" rounds to refusal,
    because a delete is the one direction in which a permission gate must never
    fail open.

    **The pattern is walked segment by segment, in the dialect of the glob
    itself.** The function used to compare the literal head of the pattern as a
    *path*, which is exact only when the head stops at a separator. `sec?ets/**`
    has the head `sec`. That is an ancestor of nothing that the rule covers, and
    it is exactly the name of a directory that the rule can never touch. So the
    rule allowed `rm -rf secrets` and refused `rm -rf sec`. Each segment is
    matched with `matches_glob`, so `?`, `*` and bracket classes mean here
    exactly what they mean to the matcher. There is no second copy of the
    alphabet that can drift.

    **`**` crosses separators wherever it appears, not only as a whole segment.**
    `matches_glob` compiles `data**/key` to cross from `data` into any depth. A
    segment that *contains* `**` is therefore free to descend after the part
    before it matches. A first version of this segment walk recognized `**` only
    as a whole segment, and so it *allowed* a delete of `data/v1` under
    `deny data**/key`. That was a permissive regression on the case that this
    function exists for. A review caught it, and a test now checks it.

    A pattern that ends *above* the directory names an ancestor. A delete of a
    descendant removes nothing that the pattern names: a delete of `secrets/old`
    does not delete `secrets`.

    **`at_root` is asked, not encoded.** The relative spelling of the workspace
    root is `""`, and every relative path is inside it. The caller states which
    question it asks.
    """
    if at_root:
        return True
    names, parts = directory.split("/"), pattern.split("/")
    # Not `strict`: the lengths differ by design, and the return says why.
    for name, segment in zip(names, parts, strict=False):
        if "**" in segment:
            head, _, _ = segment.partition("**")
            return matches_glob(name, f"{head}*")
        if not matches_glob(name, segment):
            return False
    return len(names) <= len(parts)


@plugin("permissions-fs", affects="environment", inject=[FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Attach the rules to `ctx.fs`."""
    fs: FsService = ctx.require(FS)
    permissions = FsPermissions(rules=config.rules, roots=fs.root_for, ctx=ctx)
    ctx.provide(FS_PERMISSIONS, permissions)

    # The diagnostic is offered before the no-rules return below. "Nothing is
    # refused here, and here is how far that goes" is the reading that a person
    # most needs, and an early return would have silently dropped it.
    contribute(
        ctx,
        Diagnostic(
            id="permissions-fs", title="File permissions", read=permissions.describe, order=30
        ),
    )
    if not config.rules:
        # There is nothing to enforce, so nothing is attached, not even a
        # predicate that would return False. A screen is consulted for each file
        # that a walk visits, and a `grep` over a repository must not pay a
        # Python call per candidate for a rule set that nobody wrote. The service
        # is still published, because "no rules" is an answer that `phern doctor`
        # must be able to give.
        return
    if not permissions.confined:
        # The warning is logged at mount, and only when there are rules to be
        # wrong about. An operator who wrote a deny list must be told what the
        # list does not reach before they rely on it, and not after. `phern doctor`
        # (P4-12) asks `ctx.fs_permissions.reach` for the same sentence on demand.
        log.warning("ph_stabilize.permissions_fs: %s", permissions.reach)

    ctx.on("fs/read-intent", _gate(ctx, permissions, "read"))
    # `write` is registered two times. A rule that says "do not write here" and
    # leaves `edit` open would be a rule that nobody meant to write.
    ctx.on("fs/write-intent", _gate(ctx, permissions, "write"))
    ctx.on("fs/edit-intent", _gate(ctx, permissions, "write"))
    # `screen`, not the retired `hide`. This row can now refuse a *tree*, and not
    # the same tree one time per file in it (P6-19). The `scope=` that the row
    # has always passed now has an effect on the enumeration side too: an
    # agent-scoped mount screens the walks of its own agent, and not the walks
    # of every agent (P6-18).
    fs.screen(permissions.screen, scope=ctx)


def _gate(
    ctx: Context, permissions: FsPermissions, operation: Operation
) -> Callable[[FsIntent, Callable[[], Awaitable[str | None]]], Awaitable[str | None]]:
    """One intent listener, which holds only what it reads.

    It is a closure over three small values and not a class. It holds `ctx` and
    not a captured `ctx.approval`. A row layered after this one may provide the
    seam, and a lookup at mount would have captured `None` and refused every
    `interrupt` for the life of the process.
    """

    async def gate(
        intent: FsIntent,
        next_: Next[str | None],
    ) -> str | None:
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
    ctx: Context,
    permissions: FsPermissions,
    intent: FsIntent,
    rule: Rule,
    operation: Operation,
) -> str | None:
    """Route an `interrupt` to whoever answers approvals. `None` means granted.

    The function fails closed on both halves of "there is nobody to ask", as
    `ToolRuntime._service_ask` and the gate of the RLM harness do. **A missing
    agent denies as surely as a missing service.** The second half matters.
    `ApprovalService.request` derives the session from the agent. An ask with no
    agent would prompt a human with nothing written to the log. It would also
    skip the `approval_policy: "never"` short-circuit that a deployment set
    deliberately.

    The refusal says *which* refusal it was, through the sentences of the seam
    itself. A model that cannot tell the "no" of a human from a missing channel
    cannot tell which one is worth re-planning around.
    """
    approval = ctx.get(APPROVAL)
    subject = f"{operation} of {intent.path}"
    if approval is None or intent.agent is None:
        return denial_reason("unavailable", subject)
    outcome = await approval.request(
        agent=intent.agent,
        tool_name=f"fs.{operation}",
        reason=permissions.prompt(rule, operation, intent.path, intent.agent),
        allowed_decisions=("approve", "reject"),
    )
    return None if outcome == "allowed-once" else denial_reason(outcome, subject)
