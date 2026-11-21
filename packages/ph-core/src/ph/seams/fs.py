"""`ctx.fs`: filesystem access with an interception point before every access.

Every read, write and edit passes through a waterfall (`fs/read-intent`,
`fs/write-intent`, `fs/edit-intent`) *before* it touches the disk. That order is
the value of the seam. A policy plugin that ran after the write would be a
reporter and not a gate.

**Enumeration is filtered and not gated** (`screen`). `glob` and `grep` visit
thousands of candidates. So a policy row registers a synchronous screen, and the
walk never yields, or never enters, what the screen refuses. The screen runs
*during* the walk and not over the results, because `grep` reads the files that
it visits. A filter over the matches of `grep` would return no rows, but `grep`
would already have read every byte of the file that the rule protects.

**Filtered is not refused. These are two separate questions on purpose.** A
screen answers "may I be told that this exists". The intent waterfall answers
"may I open it". A path that a screen hides is still readable through `read`,
unless a rule also vetoes `fs/read-intent`. So `permissions-fs` registers *both*
and does not derive one from the other. A deployment that reads `ignore:` as
access control has misread it.

`fs/observed` records reads. Because of that record, read-before-edit can be a
policy row and not a hard-coded rule.

**The scope of this seam: it bounds *tool-mediated* access.** A code cell that
the model writes can call `open(path, "w")`, and this seam cannot reach that
call by construction (N1): a deny-list needs a registered name. The containment
ladder exists for that case, and no wording in this module should suggest
otherwise.

@module ph.seams.fs
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

import anyio
from pydantic import Field

from ..agent.types import AgentHandle
from ..cordis import (
    Boundary,
    Context,
    Disposer,
    Next,
    Running,
    boundary_of,
    events,
    plugin,
    running,
    settled_or_none,
)
from ..keys import FS, PROJECT_ROOT
from ..paths import write_atomic
from ..session import Session
from ..session.writers import log_writer
from ..text import around, first_lines
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel
from ._registry import claim_entry, claim_slot

_LOG = log_writer(__name__)

__all__ = [
    "EditIntent",
    "FileSlice",
    "FileTooLarge",
    "FsIntent",
    "FsRefused",
    "FsService",
    "GrepMatch",
    "IndexScope",
    "ReadIntent",
    "WalkDecision",
    "WalkScreen",
    "WriteIntent",
    "Written",
    "apply",
    "covers_whole",
    "glob_error",
    "matches_glob",
    "read_before_edit",
    "skipped_lines",
]

log = logging.getLogger("ph.seams.fs")

events.declare(
    "fs/read-intent",
    "waterfall",
    owner="ph.seams.fs",
    doc="Before a file is read. A listener that vetoes prevents it.",
)
events.declare(
    "fs/write-intent",
    "waterfall",
    owner="ph.seams.fs",
    doc="Before a whole-file write reaches disk. A listener that vetoes prevents it.",
)
events.declare(
    "fs/edit-intent",
    "waterfall",
    owner="ph.seams.fs",
    doc="Before an in-place edit reaches disk. A listener that vetoes prevents it.",
)
events.declare(
    "fs/changed",
    "emit",
    owner="ph.seams.fs",
    doc="A path was written or edited; consumers refresh their view.",
)


class FileSlice(WireModel):
    """A window onto a file, with enough context to ask for the next one."""

    path: str
    text: str
    offset: int = 0
    """First line index included (0-based)."""
    lines: int = 0
    total_lines: int = 0
    truncated: bool = False


class GrepMatch(WireModel):
    path: str
    line: int
    text: str
    """The matching line, or as much of it as fits around the match
    (`ph.text.around`). A minified line is shown where it matched and not from
    its start, because the start may hold no match at all."""


WalkDecision = Literal["yield", "skip", "prune"]
"""What a screen says about one path that `glob` or `grep` considered.

`yield` means that the screen has nothing to say. `skip` drops this path from
the results. `prune` also refuses to enter the path, which has a meaning only
for a directory. The type is a closed `Literal` and not two booleans. The
consumers then `match` exhaustively, and a fourth answer fails to type-check at
every site that must handle it."""

WalkScreen = Callable[[str, str, Any, bool], WalkDecision]
"""`(path, name, agent, is_dir) -> WalkDecision`. See `FsService.screen`.

**Strings, and not a `Path`.** A walk pays for every `Path` that it builds
(P6-17), and a screen that took a `Path` would bring that cost back. `_walk`
already holds the joined string. It would build one `Path` per candidate only to
pass it to the screens, and every screen would then convert it back. The ignore
list of `fs-local` needs the bare name, and `permissions-fs` needs the posix
string (it called `as_posix()` on its first line).

`path` is absolute and **posix-separated on every platform**, because rule
patterns and `matches_glob` are written in that form. `name` is the bare final
component, so the common screen is a set lookup and not a parse. A screen that
really needs a `Path` builds one on the branch that needs it.
"""

WalkDecider = Callable[[str, str, bool], WalkDecision]
"""A walk's screens, with its agent already bound. See `FsService._decider`."""


@dataclass(frozen=True, slots=True)
class _Screen:
    """One registered screen and the scope that owns it.

    The owner is the whole enumeration half of P6-18. The earlier `hide`
    registration kept bare callables. `Context.reaches` then had nothing to
    filter on, and an agent-scoped row concealed paths from every agent."""

    decide: WalkScreen
    owner: Context


def _scope_of(agent: AgentHandle) -> Context:
    """The scope of an agent, for the workspace root, or `None` for no agent.

    The scope is read off `AgentHandle`, which `ph.agent.types` declares and
    this module imports. It was a `getattr` probe before. The reasoning was that
    `ph.seams.fs` sits below `ph.agent_loop` and must not import its consumer.
    But the Protocol is *not* the driver, and that is the reason the Protocol
    exists. So naming it inverts no layering. `tests/test_agent_vocabulary.py`
    now checks that.

    **There is one caller, and `None` is benign there** (P6-32). There were two
    callers, and they read `None` in opposite ways: the policy boundary had to
    *refuse* it, and the workspace root absorbs it. The requirement of a stated
    `Boundary` deleted the first caller. What is left is the *physical*
    question of D21: which worktree belongs to this agent. There, `None` means
    that no layer overrides the layer of the resolver itself. That is a
    fallback with nothing wider to widen to, and so it is silent.
    """
    return agent.ctx


@dataclass(frozen=True, slots=True)
class ReadIntent:
    """A read awaiting policy.

    It carries the same `agent` as the other two intents, so a row that wants to
    *ask* about a read has somewhere to route the prompt.

    **`scope` is the boundary in which this intent was judged** (P6-24). This
    holds for all three intents. The field is required, and it comes before
    `agent` so that it cannot be defaulted. When an agent is involved, there is
    no correct fallback for "which boundary is this". The public method
    resolves the boundary one time, from what the caller stated (P6-32).

    The boundary is on the payload and is not passed to `_gate`. The frozen
    record then says which boundary it was judged in, and the gate cannot be
    handed a different boundary from the one that the screen used. No listener
    reads the field yet.
    """

    path: Path
    scope: Context
    agent: AgentHandle | None = None


@dataclass(frozen=True, slots=True)
class Written:
    """What one write did, for a caller that has to report it."""

    path: Path
    created: bool
    bytes: int


@dataclass(frozen=True, slots=True)
class WriteIntent:
    """A whole-file write awaiting policy."""

    path: Path
    content: str
    creating: bool
    scope: Context
    agent: AgentHandle | None = None


@dataclass(frozen=True, slots=True)
class EditIntent:
    """An in-place replacement awaiting policy."""

    path: Path
    old_text: str
    new_text: str
    replace_all: bool
    scope: Context
    agent: AgentHandle | None = None


FsIntent: TypeAlias = ReadIntent | WriteIntent | EditIntent
"""The three things that a policy row is asked about.

The union is spelled one time because it is the argument of the gate in two
packages: `FsService._gate` here, and the row of `ph_stabilize.permissions_fs`.
A fourth intent should be a one-line edit and not a search."""


class FsDenied(HarnessError, PermissionError):
    """Policy refused a read, a write or an edit.

    This is a `HarnessError` as well as a `PermissionError`, so the refusal
    reaches a consumer as a *denial*. Without that, it reached
    `ToolFailure.kind` as an ordinary `failed`. That is the same hole that C3
    closes for `tools/pre-execute`: a Code Mode program could `except` a policy
    veto from this seam and route around it, and the durable record said that
    the tool merely failed. `SandboxError` in the sandbox seam already had
    this. This seam did not, although its whole argument is that it is a gate
    and not a report.

    It is still a `PermissionError`, so every existing catcher keeps working.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FS_DENIED")


class FileTooLarge(HarnessError):
    """A whole-file read was refused because the file is over the cap of the
    caller (P7-01).

    This is a **failure and not a denial**, and `FailureKind` exists to carry
    that distinction. No policy refused anything here: the caller named a
    bound, and the file is bigger than it. A model that reads `denied` looks
    for a permission to ask for. A model that reads this error should pick a
    smaller file. Under Code Mode (C3), the difference decides whether its
    whole program ends.
    """

    failure_kind: FailureKind = "failed"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FILE_TOO_LARGE")


class FsRefused(HarnessError, ValueError):
    """The model described something that this seam cannot use (L11).

    This is a **failure and not a denial**, for the reason that `FileTooLarge`
    gives: no policy refused anything. There are three cases. The `old_text`
    that an edit named appears nowhere, or more than one time. The file is not
    text that an edit can be written back into. A `grep` pattern is not a
    regex. In every case the answer is to look again and describe it
    differently.

    The error is coded, because the four refusals that it replaces were bare
    `ValueError`s in a module whose other two errors carry a `failure_kind` and
    a code. The benefit is exactly this. `error_info` is reflective, so the
    code reaches `ToolFailure.info` and from there the durable `tool/result`
    event. A replay and the TUI can then name the refusal, and in-process
    callers get a class to catch. The code does **not** currently reach a Code
    Mode program: `DispatchBridge.call` branches on `kind == "denied"` and
    folds everything else into `ToolCallError`, and the RLM reply frame carries
    no code. To say otherwise would promise a routing that does not exist.

    The class is named for the seam and not for `edit`, because the invalid
    regex of `grep` is the same kind of refusal one method over. Three coded
    refusals and one bare one would be a worse invariant than four bare ones,
    because a reader could no longer tell "not yet" from "deliberately".

    It is still a `ValueError`, in the way that `FsDenied` is still a
    `PermissionError`, so every existing catcher keeps working.
    """

    failure_kind: FailureKind = "failed"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FS_REFUSED")


def _decode(raw: bytes) -> tuple[str, bool]:
    """The text of a file, and whether the decode lost anything.

    `read` and `edit` share this one decode policy. They had two: `read`
    replaced and `edit` was strict. So a file that the model could read raised a
    bare `UnicodeDecodeError` out of the edit. That failure named an encoding,
    for a file that the harness had just shown to the model (J7).

    The decode is strict first and lenient only on failure. It is not
    `errors="replace"` from the start. The common file decodes one time either
    way, and the flag lets `edit` refuse a file whose bytes it would otherwise
    quietly rewrite. A binary file belongs in `read_bytes`, which never comes
    through here.
    """
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


def _read_decoded(target: Path) -> tuple[str, bool]:
    """`_decode` of the file's bytes, as one hop off the event loop."""
    return _decode(target.read_bytes())


READ_MAX_BYTES = 8 * 1024 * 1024
"""The default cap on a `read`, in bytes.

The cap is generous on purpose. It is not a policy about what an agent may look
at. It is the line past which a file is not source, and a whole read of it was
an accident: a checked-in database, a minified bundle, a log. The window that a
caller asks for bounds what comes *back*. This cap bounds what the read costs,
which `limit` never did.

This cap raises, and the bound of `skip_reason` skips. The docstring of
`skip_reason` explains that difference.
"""


class IndexScope(WireModel):
    """What a bulk reader looks at when a call does not say: the trees, the
    files in them, and the files to leave out. These are the arguments of
    `collect`, as the setting of a row.

    Both indexing rows take one. A profile that points a row at a package has
    then pointed every bare `…_index` call at it, and the two rows cannot drift
    on how a call narrows the scope (`narrowed`). Each row subclasses it, and
    the `text-index` subclass sets its own `include`.
    """

    paths: list[str] = Field(default_factory=lambda: ["."])
    """Directories or files under the workspace root. **This is how a row is
    pointed at a package**: a monorepo names the packages that it wants and
    stays under the file bound of the row. It does not walk everything and then
    discard most of it."""
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    """Which files to take from a directory. A file that matches any pattern is
    taken one time. The patterns are read against the directory that is
    walked."""
    exclude: list[str] = Field(default_factory=list)
    """Which files to leave out, whatever `include` says. The patterns are read
    against the **workspace root**, so `**/tests/**` means the same thing under
    every entry of `paths`. A directory that a pattern covers is never
    entered."""

    def narrowed(
        self, *, paths: Sequence[str] | None, glob: str | None, exclude: Sequence[str] | None
    ) -> Self:
        """This scope, as the arguments of one call narrow it.

        The `paths` and `glob` of the call replace those of the row. The
        `exclude` of the call is added to that of the row. A call may always
        narrow. A call that could lift the exclusion of a deployment would make
        the exclusion a suggestion.
        """
        return self.model_copy(
            update={
                "paths": list(paths) if paths is not None else self.paths,
                "include": [glob] if glob else self.include,
                "exclude": [*self.exclude, *(exclude or ())],
            }
        )


@dataclass(slots=True)
class FsService:
    """The service published as `ctx.fs`.

    **`agent` and `scope` are two values, and nothing checks them against each
    other** (P6-24, P6-32). `scope` is the policy boundary: which screens apply,
    and which intent listeners the gate reaches. It is required, so a caller
    states it or does not call. `agent` is the *physical* key. `root_for` asks
    the D21 rebase resolver which worktree belongs to this agent, and the
    approval seam needs somewhere to route a prompt. `agent=child,
    scope=parent.ctx` gets the worktree of the child under the screens of the
    parent. That is legal by construction and invisible at the call site.

    **The two enforcement layers that P6-32 depends on are not symmetric for
    this seam** (§5 rule 6). mypy names every unstated call site that holds a
    typed reference, and the runtime `TypeError` names the rest. But the five
    methods that the built-in file tools call (`read`, `write`, `edit`, `glob`
    and `grep`) run inside a tool body. An unconverted `ctx.fs.read(path)` that
    is reached through `ctx.<seam>` (which is `Any`) fails at *model* time and
    is normalized to an uncoded `failed`. The mypy layer is the only one that
    fires early here.
    """

    ctx: Context
    root: Path
    """Where relative paths resolve for an agent that holds no workspace. This
    is the directory of the process itself, and it is the answer that
    `phern doctor` and a bare CLI probe get."""
    _observed: dict[Path, float] = field(default_factory=dict)
    """The mtime at the last read, per path. Read-before-edit consults this state."""
    _screens: list[_Screen] = field(default_factory=list)
    """What the walk may show and where it may go. See `screen`."""
    _rebase: Callable[[Any], Path | None] | None = None
    """Where *this agent's* relative paths resolve. See `rebase`."""
    _rebase_by: Running | None = None
    """Who registered the resolver (P6-29). Set and cleared with it by `claim_slot`."""

    def rebase(
        self, resolver: Callable[[Any], Path | None], *, scope: Context | None = None
    ) -> Disposer:
        """Resolve relative paths per agent and not per process (D21).

        This is a *slot* and not a list like the one behind `screen`. Two
        answers to "where does this agent write" are a contradiction, and
        nobody could configure the first-match-wins reading that a list would
        need. `None` from the resolver means that the agent has no workspace,
        and `root` stands.

        This method does not consult the workspace seam directly, and that is
        the layering. Otherwise `ctx.fs` would have to know which seam owns
        agent state, when it needs only one path. The row that knows about both
        seams connects them (`workspace-lifecycle`). A deployment that mounts
        no such row keeps the per-process behavior: every relative path
        resolves against `root`.
        """
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "_rebase",
            resolver,
            label="fs.rebase",
        )

    def root_for(self, agent: AgentHandle | None = None) -> Path:
        """The root of this agent: its cwd, and what `bash` and `glob` run against.

        A resolver that raises falls back to `root` and does not fail the call.
        An agent whose workspace lookup broke must still be able to read a
        file, and a directory that is wrong but works is a better failure than
        a traceback out of `read`.
        """
        if self._rebase is None or agent is None:
            return self.root
        try:
            # The resolver runs as the row that registered it, **for the agent
            # that is being resolved** (P6-29). This is the one provider in the
            # tree whose target is already in hand and already derived, because
            # `_scope_of` belongs to this module. So this is the shape that the
            # other four would take if they were handed a scope and not an
            # agent. P6-24 did not reach them. It fixed the *policy* boundary,
            # and a provider slot is a lifetime question. So
            # `CompactionSeam.engine_by` and its three siblings still bind the
            # layer of the registration itself, and say so there.
            with running(self._rebase_by, _scope_of(agent)):
                resolved = self._rebase(agent)
        except Exception:
            log.warning("ph.seams.fs: the root resolver failed; using %s", self.root, exc_info=True)
            return self.root
        return resolved or self.root

    def named(
        self, path: str | Path, *, agent: AgentHandle | None = None, root: Path | None = None
    ) -> str:
        """How a path should be *written down*: relative to the root of this agent.

        This is the inverse of `resolve`. It is the form for a path that
        reaches the model or the log. An absolute path inside the workspace
        names the same file as its relative form, but it also puts the machine
        and the run into the conversation. A read that echoes
        `/tmp/ph-w-7/src/x.py` makes that string part of the transcript. A
        replay of the session against a fresh workspace (a retried job, a
        re-provisioned worktree, the same repo checked out elsewhere) then
        changes every such string. That moves the cached prefix of the provider
        (A11/A12) for a difference that the conversation cannot see. The agent
        has no use for the outside of its workspace, so the outside does not
        appear.

        A path *outside* the workspace keeps its absolute form. The workspace
        cannot express it as a name, and the approval that let it through is
        what made it legible in the first place. To hide it would be the one
        case where a relative path would mislead.

        `glob` and `grep` do not answer in this form: they return absolute
        paths, and the `glob` and `grep` tools pass each one through this
        method before the model reads it. `collect` answers in this form
        itself, by a slice of the prefix that the walk already has. This method
        gives the same answer for the paths that arrive one at a time.
        """
        resolved = self.resolve(path, agent=agent, root=root)
        try:
            return resolved.relative_to(root or self.root_for(agent)).as_posix()
        except ValueError:
            return str(resolved)

    def resolve(
        self, path: str | Path, *, agent: AgentHandle | None = None, root: Path | None = None
    ) -> Path:
        """Resolve against the workspace root of the agent, **with `..` collapsed** (J1).

        A relative path is the business of the agent. An absolute path is
        passed through, because a refusal here would be a confinement claim
        that this layer cannot make (N2). That is also why the `worktree` tier
        bounds a relative write and not an absolute one.

        **The bound is lexical, so the path must be lexical too.** `is_under`
        is a separator-aware prefix comparison, and its docstring says so.
        Joining an unnormalized relative path onto the root produced
        `<root>/../../../etc/cron.d/x`. That path *starts with* the root, so it
        passed every check that the `worktree` tier makes: the
        `outside-workspace` rule of `permissions-fs` did not prompt, the write
        landed outside the tree, and the log recorded the relative spelling
        that hid it. Collapsing here
        fixes it one time for every caller, because every caller comes through
        this method.

        The method uses `normpath` and **not** `resolve()`. `normpath` is a
        pure string operation, so the symlink question stays where the
        docstring of `is_under` puts it: with the caller that needs symlink
        safety. A syscall here would half answer that question. It would also
        make the meaning of a relative path depend on what is on disk.

        `root` resolves against a root that the caller already knows, and does
        not ask for the root of the agent. It is for a caller with no agent to
        ask, such as a tool that reconciles a call on resume from the root that
        its log recorded.
        """
        candidate = Path(path).expanduser()
        base = root if root is not None else self.root_for(agent)
        joined = candidate if candidate.is_absolute() else (base / candidate)
        return Path(os.path.normpath(joined))

    # ------------------------------------------------------------------ read --

    def screen(self, decide: WalkScreen, *, scope: Context) -> Disposer:
        """Decide what `glob` and `grep` may show, and where they may go (P6-19).

        This is one contribution point over **any** path that the walk
        considers, directories included. A screen answers `"yield" | "skip" |
        "prune"`. It takes directories so that it can prune: `deny secrets/**`
        should never enter the tree, and should not refuse the same tree one
        time per file inside it. `fs-local` registers the built-in ignore list
        through this seam like any other screen. So there is one way to drop a
        path, and a deployment can configure it.

        This is deliberately *not* a waterfall. A walk visits thousands of
        candidates. Awaiting a listener per file would make `grep` over a
        repository a different kind of operation, and nobody would sit through
        a question to a human about each file.

        The screen is asked about a path, **the agent that walks**, and whether
        the path is a directory. It gets the agent because the root that a rule
        is written against is per-agent when a containment tier is in force
        (D21): `secrets/**` names one directory in the checkout of the user and
        a different one in the worktree of each agent.

        **`Context` and not `Boundary`** (P6-32). This is a *registration*, so
        its `scope` becomes `_Screen.owner`. It needs a real scope to be owned
        by and to unwind with.

        **`scope` is required**, where every sibling registration in this
        package defaults it to the context of the service itself. P6-18 made
        this same field the input to `Context.reaches`. So to omit it would not
        mean "clean up with the mount". It would mean **"screen every agent"**:
        the widest policy in the seam, chosen by forgetting.

        The entry is registered through `claim_entry`, which removes it **by
        identity**. `_Screen` and `FsPermissions` are frozen dataclasses and
        compare by value. With two mounts that have equal rules, `list.remove`
        would have taken the wrong one.
        """
        entry = _Screen(decide=decide, owner=scope)
        return claim_entry(scope, self._screens, entry, label="fs.screen")

    def _decider(self, agent: AgentHandle | None, *, scope: Boundary) -> WalkDecider | None:
        """The screens of this walk, filtered and with its agent bound, or `None`.

        The screens are resolved **one time per walk and not one time per
        path**. The visibility filter is fixed for the whole walk, because the
        agent does not change halfway. Otherwise `reaches` would be answered
        again for every screen, for every one of eleven thousand candidates.

        The result is `None` when no screen reaches this agent. `_walk` can then
        skip the question and not call a predicate that always agrees. This is
        **not** the ordinary case for a mounted profile: `fs-local` registers
        the ignore screen on every mount, unless its `ignore` list is empty. So
        the branch belongs mostly to a hand-built service: that of a test, or
        of a consumer that wants a raw walk.

        `agent` is required and not defaulted at the call sites. A default
        would silently answer for the *process* root, and per-agent roots were
        introduced to fix exactly that bug.
        """
        target = boundary_of(scope, self.ctx)
        screens = [entry for entry in self._screens if entry.owner.reaches(target)]
        if not screens:
            return None

        def decide(path: str, name: str, is_dir: bool) -> WalkDecision:
            strictest: WalkDecision = "yield"
            for entry in screens:
                try:
                    said = entry.decide(path, name, agent, is_dir)
                except Exception:
                    log.warning("ph.seams.fs: a walk screen failed; refusing", exc_info=True)
                    return "prune" if is_dir else "skip"
                if said == "prune":
                    return "prune"
                if said == "skip":
                    strictest = "skip"
            return strictest

        return decide

    async def read(
        self,
        path: str | Path,
        *,
        scope: Boundary,
        offset: int = 0,
        limit: int | None = 2_000,
        max_bytes: int | None = READ_MAX_BYTES,
        agent: AgentHandle | None = None,
        session: Session | None = None,
    ) -> FileSlice:
        """Read a line window, after `fs/read-intent` allows it.

        **The window keeps its line endings** (J7). Splitting on line
        boundaries and joining again with `"\n"` rewrote every CRLF file into LF
        on the way out. The model could then not edit with the text that it got
        back. `edit` matches against what is on disk, so an `old_text` copied
        out of a read of a CRLF file matched nothing, and the model was told "no
        occurrence of the target text" about a line that it had just been
        shown. `keepends` costs nothing, and `lines` and `total_lines` still
        count lines.

        **Bounded, like `read_bytes`.** The decode below is of the *whole*
        file, however narrow the window is. So `limit=2_000` bounded what came
        back and not what the read cost. `max_bytes=None` is for a caller that
        means it, and the indexers are such callers. Each has already asked
        `skip_reason` with its own, smaller bound, because a bulk walk should
        report what it passed over and not stop.

        **A whole-file window is handed back as it was read.** With `keepends`,
        `"".join(text.splitlines(keepends=True))` *is* `text`. So the slice and
        the join rebuild a string that is already in hand, at **26.9 µs per
        file against the 10.0 µs read that they wrap**. That is 2.7x the useful
        work, and ~539 ms over a 20 000-file tree. The old `"\n".join` was not
        the identity, so the round trip had a job. `keepends` removed the job
        and left the work. The line list is still built, because `total_lines`
        is a count of lines.
        """
        target = self.resolve(path, agent=agent)
        await self._gate(
            "fs/read-intent",
            ReadIntent(path=target, agent=agent, scope=boundary_of(scope, self.ctx)),
        )
        self._refuse_oversize(target, max_bytes, agent)
        # The file is read *and* decoded in the thread. The bytes then die there
        # and do not stay alive beside the `str` for the life of this frame,
        # which would be one extra copy of the file at the 8 MiB ceiling. Also,
        # a 913 µs decode does not land on the event loop for a file of that
        # size.
        text, _lossy = await anyio.to_thread.run_sync(_read_decoded, target)
        all_lines = text.splitlines(keepends=True)
        whole = offset == 0 and (limit is None or limit >= len(all_lines))
        window = (
            all_lines if whole else all_lines[offset : None if limit is None else offset + limit]
        )
        self._observe(target, session, agent)
        return FileSlice(
            path=self.named(target, agent=agent),
            text=text if whole else "".join(window),
            offset=offset,
            lines=len(window),
            total_lines=len(all_lines),
            truncated=limit is not None and offset + len(window) < len(all_lines),
        )

    async def read_bytes(
        self,
        path: str | Path,
        *,
        scope: Boundary,
        max_bytes: int | None = None,
        agent: AgentHandle | None = None,
        session: Session | None = None,
    ) -> bytes:
        """Read a whole file as bytes, after `fs/read-intent` allows it (P7-01).

        **A model-initiated attach must go through this method** (I-9). The
        user may attach anything that they can already open, and
        `AttachmentStore.save_path` is the path for that. A *model* may attach
        only what this seam lets it read. So `permissions-fs`, the workspace
        tier and every other `fs/read-intent` listener bound this method
        exactly as they bound `read`. That takes one line here, and a policy
        row has nothing to opt into. Without this method the tool producer had
        two bad options. One was to reach around the seam with
        `Path.read_bytes`, which is the exfiltration primitive that I-9 exists
        to prevent. The other was to push a PNG through the `errors="replace"`
        decode of `read`, which returns something that is not the file.

        This is a method and not a flag on `read`, because almost nothing that
        the two share survives the change of unit. `offset`, `limit`,
        `total_lines` and `truncated` are all statements about *lines*,
        `FileSlice.text` is `str`, and a `bytes | FileSlice` return would move
        that branch into every caller. What the two really share is the gate,
        and that is one call here.

        `max_bytes` is the bound of `_refuse_oversize`, whose docstring says
        why the check is a `stat`.

        The read is recorded through `_observe` like every other read. The
        read-before-edit rule and `fs/observed` are about the fact that *this
        file was seen*, and a binary file was seen just as much.
        """
        target = self.resolve(path, agent=agent)
        await self._gate(
            "fs/read-intent",
            ReadIntent(path=target, agent=agent, scope=boundary_of(scope, self.ctx)),
        )
        self._refuse_oversize(target, max_bytes, agent)
        content = await anyio.to_thread.run_sync(target.read_bytes)
        self._observe(target, session, agent)
        return content

    def _refuse_oversize(
        self, target: Path, max_bytes: int | None, agent: AgentHandle | None
    ) -> None:
        """Raise `FileTooLarge` if the file is over the cap of the caller.

        The answer comes from the size of the file **before it is opened**, so
        refusing a 2 GB video costs a `stat` and not 2 GB of resident memory. A
        `stat` that fails is not treated as a refusal. The read that follows is
        about to raise the real `OSError`, and a size limit invented for a file
        that does not exist would report the wrong thing.

        The check is synchronous and inline, for the reason that `skip_reason`
        gives: a `to_thread` hop is 60.6 µs against a ~2 µs syscall.
        """
        if max_bytes is None:
            return
        try:
            size = target.stat().st_size
        except OSError:
            return
        if size > max_bytes:
            raise FileTooLarge(
                f"{self.named(target, agent=agent)} is {size} bytes, over the "
                f"{max_bytes}-byte limit for this call"
            )

    def _observe(
        self, target: Path, session: Session | None, agent: AgentHandle | None = None
    ) -> None:
        """Remember the mtime of this file, and record the read.

        The dict is keyed by the resolved `Path`, because it answers a question
        about *this process*: has this file changed since we read it. The log
        gets the workspace-relative name, for the reason that `named` gives: a
        record whose path is `/tmp/ph-w-7/...` describes a directory that will
        not exist the next time this session runs.
        """
        try:
            self._observed[target] = target.stat().st_mtime
        except OSError:  # pragma: no cover - raced deletion
            return
        if session is not None:
            _LOG.append(session, "fs/observed", {"path": self.named(target, agent=agent)})

    def observed_mtime(self, path: str | Path) -> float | None:
        return self._observed.get(self.resolve(path))

    # ----------------------------------------------------------------- write --

    async def write(
        self,
        path: str | Path,
        content: str,
        *,
        scope: Boundary,
        agent: AgentHandle | None = None,
        session: Session | None = None,
    ) -> Written:
        """Write a whole file, after `fs/write-intent` allows it.

        It reports what it did, in the way that `edit` reports its replacement
        count. The caller wanted `created` and a byte count, and was deriving
        both again: a second `exists()` syscall on the event loop for a fact
        that `creating` already holds, and a second encode of the content for a
        length.
        """
        target = self.resolve(path, agent=agent)
        intent = WriteIntent(
            path=target,
            content=content,
            creating=not target.exists(),
            agent=agent,
            scope=boundary_of(scope, self.ctx),
        )
        await self._gate("fs/write-intent", intent)
        written = await anyio.to_thread.run_sync(_write_text, target, content)
        self._observe(target, session, agent)
        self.ctx.emit("fs/changed", target)
        return Written(path=target, created=intent.creating, bytes=written)

    async def edit(
        self,
        path: str | Path,
        old_text: str,
        new_text: str,
        *,
        scope: Boundary,
        replace_all: bool = False,
        agent: AgentHandle | None = None,
        session: Session | None = None,
    ) -> int:
        """Replace text in place, after `fs/edit-intent` allows it.

        Returns the replacement count. This method enforces the unique-match
        requirement: it refuses an `old_text` that does not occur, and one that
        occurs more than one time unless `replace_all` is set.
        """
        target = self.resolve(path, agent=agent)
        intent = EditIntent(
            path=target,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            agent=agent,
            scope=boundary_of(scope, self.ctx),
        )
        await self._gate("fs/edit-intent", intent)
        original, lossy = await anyio.to_thread.run_sync(_read_decoded, target)
        if lossy:
            # `read` hands this file back with U+FFFD in it, so the model can
            # hold text that looks editable. Writing the edit would encode those
            # replacements over the bytes that they stood in for: a
            # single-character change would silently rewrite every undecodable
            # byte in the file. The edit is refused here and the problem is not
            # hidden, because an `old_text` copied out of a read means something
            # only when the two decodes agree (J7).
            raise FsRefused(
                f"{self.named(target, agent=agent)} is not valid UTF-8; editing it "
                "would rewrite the bytes that could not be decoded"
            )
        # `named` and not `target`. These are the two messages that a model
        # reads most often, and they echoed the absolute worktree path. That is
        # the same non-replayable spelling that J8 took out of `fs/observed`
        # (see `_observe`).
        named = self.named(target, agent=agent)
        count = original.count(old_text)
        if count == 0:
            raise FsRefused(f"no occurrence of the target text in {named}")
        if count > 1 and not replace_all:
            raise FsRefused(
                f"{count} occurrences of the target text in {named}; pass replace_all "
                "or include more surrounding context to make it unique"
            )
        updated = original.replace(old_text, new_text, -1 if replace_all else 1)
        await anyio.to_thread.run_sync(_write_text, target, updated)
        self._observe(target, session, agent)
        self.ctx.emit("fs/changed", target)
        return count if replace_all else 1

    async def _gate(self, event: str, intent: FsIntent) -> None:
        """Ask the policy rows about one intent, in the scope that owns it (P6-18).

        `scope=` is the fix, and it is one argument. The waterfall defaulted to
        the context of this *service*, which is the mount. So the hook lookup of
        the waterfall (`Context._hooks`) asked whether each listener reached the
        root. An agent-scoped listener on `fs/read-intent` reaches only its own
        agent, so it never fired at all. The sibling gate that `ToolRuntime`
        built has passed `scope=execution.scope` since it was written. This
        gate is the same gate one seam over, and it did not. That is why
        `permissions-fs` could be mounted per agent and asked per process.

        Nothing changes for a globally mounted row. A global registration
        reaches everything, so the same listeners run in the same order.
        """

        async def inner(_intent: object) -> str | None:
            return None

        # The boundary that the intent was *judged in* is carried on the intent
        # itself and is not derived again here (P6-24). The public method
        # resolved it one time, from what the caller stated. So the payload
        # that a listener receives says which boundary it is asked about, and
        # this gate cannot answer a different question from the one that the
        # screen answered.
        answered = await self.ctx.waterfall(event, intent, inner=inner, scope=intent.scope)
        reason = settled_or_none(event, answered, str, refusal=FsDenied)
        if reason is not None:
            raise FsDenied(reason)

    # ------------------------------------------------------------------ find --

    async def glob(
        self,
        pattern: str | Sequence[str],
        *,
        scope: Boundary,
        root: str | Path | None = None,
        limit: int = 1_000,
        agent: AgentHandle | None = None,
        exclude: Sequence[str] = (),
    ) -> list[str]:
        """Files under `root` that match `pattern`, or any of several patterns, as
        absolute paths, in one walk.

        `exclude` leaves out what matches any of its patterns. The patterns are
        **read against the workspace root** and not against `root`.
        `**/tests/**` then means the same thing whichever directory the walk
        started in. The setting of a row needs that when its `paths` name
        several directories. A directory that an exclusion covers is never
        entered, for the reason that `_walk` gives: a `limit` spent on walking
        `node_modules` to discard it is the budget that the exclusion was
        written to save.
        """
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = _excluding(self.root_for(agent), exclude, self._decider(agent, scope=scope))
        patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
        return await anyio.to_thread.run_sync(lambda: list(_walk(base, patterns, limit, decide)))

    async def collect(
        self,
        paths: Sequence[str],
        pattern: str | Sequence[str],
        *,
        scope: Boundary,
        limit: int = 1_000,
        agent: AgentHandle | None = None,
        exclude: Sequence[str] = (),
    ) -> list[str]:
        """The files that `paths` names, with a directory expanded by `pattern`, in
        the form that `named` gives.

        This is the shape that every **bulk** reader needs and that `glob` does
        not quite give. `glob` takes one root and answers with **absolute**
        paths (it matches the pattern against the walk-relative slice but
        yields the full path). A caller with a list of arguments, each of which
        may be a file or a directory, needs one de-duplicated list in the
        spelling that it will store and report.

        The method is here and not in each caller because two callers arrived
        within a week: the code-graph indexer and the text-index indexer. They
        had the same loop, the same `is_dir()` branch, and the same paragraph
        that explained that `glob` returns absolute paths. They had already
        drifted on the `limit`. A third caller would have copied whichever it
        read first.

        **The names are in the `named` form and not absolute.** That is the
        part that is worth centralizing. An absolute path in a stored record or
        a tool result puts the machine and the run into the conversation. A
        replay of a session against a fresh workspace then changes every such
        path, and moves the cached prefix of the provider for a difference that
        the conversation cannot see (`named`).

        The result is sorted and de-duplicated, so two overlapping arguments
        name a file one time. An indexer that saw a document twice would either
        do the work twice or rely on a replace being idempotent.

        **`limit` is the budget of the whole call, not of each argument.** Both
        callers pass the `max_files` of a row, which is documented as the
        number of files that one call considers. When it was forwarded to
        `glob` per argument, `paths: [a, b, c]` considered three times that
        number. So the one setting that a deployment has to bound an indexing
        call did not bound it. `glob` keeps its own per-walk cap, and this
        method spends against the remainder.

        The spelling is a string slice and not `named` per result, for the
        reason that `ph.paths.is_under` gives. `glob` yields absolute paths
        under a root that this method already knows. The `resolve` and
        `relative_to` of `named` cost 44.8 µs each. That is 0.9 s of event-loop
        time for a 20 000-file walk, to compute again a prefix that the walk
        had. `named` stays the fallback for anything that somehow lands
        outside, which is the case that a slice cannot answer.

        `exclude` is the `exclude` of `glob`: patterns against the workspace
        root, applied inside the walk. A **file** that `paths` names outright
        is taken whatever the patterns say, because the caller asked for it by
        name.

        `pattern` may be several patterns. One walk tries them all, and a file
        that matches any of them is taken one time.
        """
        base = f"{self.root_for(agent)}/"
        found: set[str] = set()
        for one in paths:
            target = self.resolve(one, agent=agent)
            if not target.is_dir():
                found.add(self.named(target, agent=agent))
                continue
            remaining = limit - len(found)
            if remaining <= 0:
                # The budget is spent, but the loop continues over the
                # arguments. A file that is named outright is not a walk, and
                # must not be dropped because a directory ahead of it filled
                # the quota.
                continue
            for path in await self.glob(
                pattern, root=one, scope=scope, limit=remaining, agent=agent, exclude=exclude
            ):
                spelled = str(path)
                found.add(
                    spelled[len(base) :]
                    if spelled.startswith(base)
                    else self.named(path, agent=agent)
                )
        return sorted(found)

    def skip_reason(
        self, path: str | Path, *, max_bytes: int, agent: AgentHandle | None = None
    ) -> str:
        """Why a bulk reader should pass `path` over. `""` when it should read it.

        This is the companion of `collect`, and it is here for the same reason.
        Both indexers wrote this preamble out: resolve, `stat`, compare, skip
        with a sentence. The two copies had already drifted in three ways, in
        code that was written a week apart. They disagreed on the limit
        sentence, on the `OSError` sentence (one of them printed `None`
        whenever `strerror` was unset), and on how the `stat` was threaded. A
        third caller would have copied whichever it read.

        **This is a skip and not a raise.** That makes it different from the
        `max_bytes` of `read_bytes`. An indexing call over a directory must not
        fail because it found a minified bundle. It must report what it passed
        over. And a caller who never learns what was skipped cannot tell a
        quiet corpus from a quiet failure.

        The method is synchronous, and the `stat` is inline exactly as
        `read_bytes` does it (`_refuse_oversize`). A `to_thread` hop is 60.6 µs
        against a ~2 µs syscall. So a threaded size guard cost more than the
        ~38 µs read that it exists to avoid: 1.2 s of it across a 20 000-file
        tree, paid to save nothing.
        """
        target = self.resolve(path, agent=agent)
        try:
            size = target.stat().st_size
        except OSError as error:
            return f"cannot be read ({error.strerror or error})"
        if size > max_bytes:
            return f"{size} bytes, past the {max_bytes}-byte limit"
        return ""

    async def grep(
        self,
        pattern: str,
        *,
        scope: Boundary,
        root: str | Path | None = None,
        glob: str = "**/*",
        limit: int = 200,
        agent: AgentHandle | None = None,
    ) -> list[GrepMatch]:
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = self._decider(agent, scope=scope)
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise FsRefused(f"invalid regular expression: {error}") from error

        def scan() -> list[GrepMatch]:
            matches: list[GrepMatch] = []
            for found in _walk(base, (glob,), None, decide):
                candidate = Path(found)
                if not _greppable(candidate):
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    hit = expression.search(line)
                    if hit is not None:
                        shown = around(line, hit.start(), 500, stop=hit.end())
                        matches.append(GrepMatch(path=found, line=number, text=shown))
                        if len(matches) >= limit:
                            return matches
            return matches

        return await anyio.to_thread.run_sync(scan)


GREP_MAX_BYTES = 2 * 1024 * 1024
"""`grep` skips files above this size. A build artifact that happens to match
`**/*` must not be read whole into memory line by line."""


def _excluding(
    root: Path, exclude: Sequence[str], decide: WalkDecider | None
) -> WalkDecider | None:
    """`decide`, which also leaves out what `exclude` matches under `root`.

    A directory is pruned when a pattern covers it whole (`covers_whole`).
    `**/tests/**` names the contents of every `tests` directory. The pattern
    exists to skip the walk that descends into one and refuses each file in
    turn.

    Each question is one compiled alternation and not one pattern at a time.
    The screen runs per candidate, and on a 20 000-file walk the loop over
    patterns cost more than the walk that it screened.
    """
    if not exclude:
        return decide
    files = _any_glob(tuple(exclude))
    directories = _any_glob(tuple(whole for one in exclude for whole in _wholesale(one)))
    cut = len(os.fspath(root)) + 1

    def decided(path: str, name: str, is_dir: bool) -> WalkDecision:
        if (directories if is_dir else files).match(path[cut:]) is not None:
            return "prune" if is_dir else "skip"
        return "yield" if decide is None else decide(path, name, is_dir)

    return decided


def _walk(
    base: Path, patterns: tuple[str, ...], limit: int | None, decide: WalkDecider | None
) -> Iterator[str]:
    """Files under `base` that match any of `patterns`. Refused directories are
    pruned *before* descent.

    `Path.glob` would materialize `node_modules` in full and then discard it.
    Pruning `dirs` in place means that a refused tree is never entered.
    Directories and files are visited in sorted order, so results are stable.

    `decide` is consulted here and not over the results, because `grep` reads
    every file that this yields. It is asked about directories too (P6-19), so
    a policy row can say "never go in there".

    **`None` means that no screen reaches this walk.** It is spelled as an
    absence and not as a predicate that always agrees, so that the loop can
    skip the call per candidate for a policy question that nobody asked. This
    is not the ordinary case for a mounted profile (see `FsService._decider`).

    Three choices here matter on a walk of thousands of files: the relative
    path is a slice and not `Path.relative_to`, the join happens one time, and
    `glob` returns `str` and not `Path`. `os.sep` is translated and not
    assumed, because `matches_glob` uses posix separators. The branch is
    hoisted out of the loop, because the answer is fixed for the process.
    """
    root = os.fspath(base)
    cut = len(root) + (0 if root.endswith(os.sep) else 1)
    native = os.sep != "/"
    wanted = _any_glob(patterns)

    def posix(path: str) -> str:
        return path.replace(os.sep, "/") if native else path

    yielded = 0
    for current, dirs, files in os.walk(base):
        dirs.sort()
        if decide is not None:
            dirs[:] = [
                name
                for name in dirs
                if decide(posix(os.path.join(current, name)), name, True) != "prune"
            ]
        for name in sorted(files):
            full = os.path.join(current, name)
            if wanted.match(posix(full[cut:])) is None:
                continue
            if decide is not None and decide(posix(full), name, False) != "yield":
                continue
            yield full
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def matches_glob(candidate: str, pattern: str) -> bool:
    """`Path.glob` semantics: `*` stays inside one segment, `**` crosses them.

    The function is public because the path patterns of a permission row and
    the `glob` of this seam must mean the same thing by `**/*.env`. Two glob
    dialects in one harness would be a rule that someone writes one time and is
    then wrong about forever.

    **`*` must not cross a separator, and an error here has a direction.** The
    `*` of `fnmatch` matches `/`, so `docs/*.md` would also match
    `docs/private/keys.md`. For a search box that is a quirk. For an ACL that
    is evaluated first-match-wins it is a hole, because the rules are written
    as a narrow `allow` above a broad `deny`. An `allow` that is silently wider
    than written permits what the `deny` under it was there to stop.
    """
    return _compiled(pattern).match(candidate) is not None


def _wholesale(pattern: str) -> tuple[str, ...]:
    """The patterns that a directory must match for `pattern` to cover it whole:
    the pattern itself, and for `Q/**` also `Q`. `covers_whole` and the
    exclusions of the walk are both this question, asked one time and asked of
    a set."""
    return (pattern, pattern[:-3]) if pattern.endswith("/**") else (pattern,)


def covers_whole(pattern: str, directory: str) -> bool:
    """Whether `pattern` names `directory` and everything inside it.

    **This is the universal question, and it belongs to the walk** (D11).
    Pruning a directory is lossless only when the rule would conceal every
    entry in it anyway. So the walk asks whether the rule covers the directory
    *wholesale*, and not whether the rule could match something inside. A rule
    that covers only some of the contents of a directory is enforced file by
    file as the walk passes. That conceals exactly what the rule names and
    hides nothing else.

    That is why this is not the question of the delete with a different
    rounding. It was before: one function answered both, with a `require_head`
    flag that bent the answer for the walk. The flag could not be made right.
    `deny read src/*.pem` pruned all of `src` and hid every allowed file in it,
    while `deny read **/node_modules/**` pruned no `node_modules` at all. The
    two callers ask different questions, and the flag went away when each got
    its own function.

    `matches_glob` gives the whole answer, so this function cannot disagree
    with the matcher: the directory itself matches, or the pattern is `Q/**`
    and the directory matches `Q`. The function is beside the matcher because
    two walks prune by it: the screen of `permissions-fs` and the exclusions
    of a bulk reader.
    """
    return any(matches_glob(directory, one) for one in _wholesale(pattern))


@lru_cache(maxsize=128)
def _any_glob(patterns: tuple[str, ...]) -> re.Pattern[str]:
    """One regex that matches what any of `patterns` matches, as `matches_glob`
    reads each pattern. It is built from the same compiled sources, so the two
    cannot disagree. Each source is anchored at its end, and `match` anchors
    the start."""
    if len(patterns) == 1:
        return _compiled(patterns[0])
    return re.compile("|".join(f"(?:{_compiled(one).pattern})" for one in patterns))


def glob_error(pattern: str) -> str | None:
    """Why `pattern` is not a usable glob, or `None` when it is (N6).

    **There are two populations of patterns and one matcher.** A model types a
    `glob` argument, and the fallback of `matches_glob` is right for it: a
    pattern that does not compile matches itself and nothing else. That is
    inert, and it is not an exception out of a policy check. An operator writes
    a `permissions-fs` rule, and there inert is exactly wrong. A `deny` whose
    glob does not compile silently guards nothing, and that is the one
    direction in which an ACL must never fail.

    So the fallback stays where the model can reach it, and config asks this
    function first. The reason is returned and not raised, so the caller
    decides whether an unusable pattern is a config error or a warning.
    """
    try:
        re.compile(_glob_source(pattern))
    except re.error as error:
        return str(error)
    return None


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern[str]:
    """One glob, as a regex. Cached, because a walk asks per candidate file.

    The cache is bounded. Patterns come from the `glob` calls of a model as
    well as from config, so the LRU bound keeps a session that greps a thousand
    different patterns from holding a thousand compiled regexes forever.
    """
    source = _glob_source(pattern)
    try:
        return re.compile(source)
    except re.error:
        # **This keeps the promise, whatever the scanner missed.** `_glob_source`
        # handles every shape that somebody thought of, and this branch is here
        # for the others. A pattern is a string that a model typed, and
        # `permissions-fs` evaluates it first-match-wins on the read path. A
        # pattern that does not compile matches itself and nothing else. That
        # is inert and not wrong, where a raise would take down the call that
        # was being checked. Config does not rely on this. See `glob_error`.
        log.warning("ph.seams.fs: %r is not a usable glob; matching it literally", pattern)
        return re.compile(re.escape(pattern) + r"\Z")


def _glob_source(pattern: str) -> str:
    """One glob, as regex source. Separate so `glob_error` can try it too."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            # Any number of leading segments, including none. That is why
            # `**/*.py` finds `main.py` at the root as well as `a/b/main.py`.
            parts.append("(?:[^/]*/)*")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        elif pattern[index] == "[":
            # Find where the class ends, in the way that `fnmatch` reads it: a
            # `!` right after the bracket negates, and a `]` right after *that*
            # is a member and not the end. A search from `index + 1` made `[]`,
            # `[!]` and `[]abc]` produce `[]`, `[^]` and `[]abc]`. None of these
            # is a regex, so `re.compile` raised out of a policy check that
            # promises that it cannot raise (D5).
            close = index + 1
            if close < len(pattern) and pattern[close] == "!":
                close += 1
            if close < len(pattern) and pattern[close] == "]":
                close += 1
            close = pattern.find("]", close)
            if close == -1:
                # An unbalanced bracket is a literal and not a syntax error: a
                # pattern that a model typed must not raise out of a policy
                # check. `[]` and `[!]` reach this too. They are literals for
                # the same reason that `fnmatch` makes them literals: there is
                # no class in them.
                parts.append(re.escape("["))
                index += 1
            else:
                parts.append(_character_class(pattern[index + 1 : close]))
                index = close + 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return "".join(parts) + r"\Z"


def _character_class(inner: str) -> str:
    """One glob `[...]`, as a regex character class, with all four rules in one place.

    `!` negates, in the way that `fnmatch` spells it. `]` and `\\` must be
    escaped, or they end the class and start an escape. A leading `^` must be
    escaped, or it negates a class that the author did not write as negated.
    """
    negated = inner.startswith("!")
    body = (inner[1:] if negated else inner).replace("\\", "\\\\").replace("]", "\\]")
    if body.startswith("^"):
        body = f"\\{body}"
    return f"[{'^' if negated else ''}{body}]"


def _greppable(path: Path) -> bool:
    """Text, and small enough to read whole."""
    try:
        if path.stat().st_size > GREP_MAX_BYTES:
            return False
        with path.open("rb") as handle:
            return b"\0" not in handle.read(8192)
    except OSError:
        return False


def _write_text(target: Path, content: str) -> int:
    """Write the file and report its size in bytes.

    **The file is replaced and not overwritten** (S5). `write_bytes` truncated
    the file of the user and then wrote it. So a full disk, an `EIO` or a power
    cut in the middle of an `edit` left a prefix of the new text where the file
    had been. The reconcile of `write` then read "it did not happen" off a file
    whose original was gone. Through `write_atomic` the file is the old one or
    the new one, and it is `fsync`ed. `preserve` keeps what a replace would
    otherwise change (a link, the mode bits).

    The content is encoded here and not by `write_text`, because the caller
    wants the byte count and `write_text` returns *characters*. The count was
    therefore a second full encode of the same string, 30 µs on a megabyte.
    Now there is one pass, on this thread.
    """
    data = content.encode("utf-8")
    write_atomic(target, data, preserve=True)
    return len(data)


_IGNORED_PARTS: tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
)
"""The directory names that `fs-local` prunes unless a deployment says otherwise.

This is a **default** and not a rule (P6-19). It was a module-level `frozenset`
that `_walk` consulted inline. That made it the second of two ways to drop a
path, and the only one that nobody could configure. A repository with a
`vendor/` or a `target/` had no way to say so, and a `.gitignore` row had no
place to register. It is now the `ignore` config of `fs-local`, registered
through `ctx.fs.screen` like any other policy, so the walk has no built-in
knowledge of it at all."""


class Config(WireModel):
    """The row config for the local filesystem provider."""

    root: str | None = None
    ignore: list[str] | None = None
    """Directory **names** that `glob` and `grep` never enter. `None` keeps
    `_IGNORED_PARTS`, and `[]` turns pruning off entirely.

    Each entry is a bare final component, matched anywhere in the tree: write
    `"dist"` and not `"build/dist"`, which would silently never match. A path
    pattern is the job of a `permissions-fs` rule. Unlike this list, such a
    rule also refuses to *read* what it hides. This list keeps noise out of a
    listing and protects nothing.

    This is config and not a constant (P6-19), and the difference between
    `None` and `[]` is the reason that the type is `list[str] | None`. A
    repository with a `vendor/` or a `target/` had no way to add one, and a
    deployment that really wants to see inside `.git` had no way to say so. A
    plain `[]` default would have silently turned pruning off for every
    profile that did not mention the setting."""


@plugin("fs-local", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local filesystem provider, and its one built-in screen."""
    # There are three answers, most specific first. `config.root` is a
    # deployment that says "always work here", and it wins. `project_root` is
    # the working directory of *this mount*, provided before the first row by
    # whoever mounted. A daemon mounts one composition one time per session,
    # and those sessions live in different repositories, so the directory
    # cannot come from the profile (P5-14). `Path.cwd()` is the directory of
    # the process itself. That is right for a mode that is *in* the project,
    # and it was silently wrong for every root that a daemon held.
    provided = ctx.get(PROJECT_ROOT)
    if config.root:
        root = Path(config.root).expanduser()
    elif provided is not None:
        root = Path(provided).expanduser()
    else:
        root = Path.cwd()
    service = FsService(ctx=ctx, root=root)
    ctx.provide(FS, service)
    ignored = frozenset(_IGNORED_PARTS if config.ignore is None else config.ignore)
    if not ignored:
        return

    def ignore(_path: str, name: str, _agent: object, is_dir: bool) -> WalkDecision:
        """The ignore list as an ordinary screen, and not as a branch inside the walk.

        **Directories only.** The list is matched against directory names, so a
        file called `dist` stays visible. It is registered here and is not
        known to `_walk`, so there is one mechanism for what the walk skips. A
        deployment can therefore reconfigure it, and a `.gitignore` row has a
        place to register.

        The check is a set membership on the `name` that the walk already had.
        The `str` half of the `WalkScreen` contract is for this.
        """
        return "prune" if is_dir and name in ignored else "yield"

    service.screen(ignore, scope=ctx)


@plugin("fs-read-before-edit", affects="environment", inject=[FS])
async def read_before_edit(ctx: Context, config: None) -> None:
    """Refuse an edit to a file that this session has not read since it last changed.

    This is its own row and not a rule inside `edit`, for two reasons. A
    deployment can drop it. And it applies to *every* editing tool, and not
    only to the one that remembered to check.
    """
    fs: FsService = ctx.require(FS)

    async def gate(intent: EditIntent, next_: Next[str | None]) -> str | None:
        observed = fs.observed_mtime(intent.path)
        if observed is None:
            return (
                f"read {intent.path} before editing it, so the edit is based on its "
                "current contents"
            )
        try:
            current = intent.path.stat().st_mtime
        except OSError:
            return f"{intent.path} no longer exists"
        if current > observed:
            return f"{intent.path} changed on disk since it was read; read it again before editing"
        return await next_()

    ctx.on("fs/edit-intent", gate)


def skipped_lines(skipped: Sequence[Mapping[str, Any]], shown: int) -> list[str]:
    """What a bulk reader passed over (`FsService.skip_reason`): one line each up
    to `shown`, then how many more. The results of both indexers use these
    words."""
    return first_lines(
        skipped, shown, lambda one: f"skipped {one['path']}: {one['reason']}", rest="more skipped"
    )
