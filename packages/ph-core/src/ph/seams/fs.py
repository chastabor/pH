"""`ctx.fs` — filesystem access with an interception point before every access.

Every read, write and edit passes through a waterfall (`fs/read-intent`,
`fs/write-intent`, `fs/edit-intent`) *before* it touches the disk. That ordering
is the whole value: a policy plugin that ran after the write would be a reporter,
not a gate.

**Enumeration is filtered, not gated** (`screen`). `glob` and `grep` visit
thousands of candidates, so a policy row registers a synchronous screen and the
walk never yields, or never enters, what it refuses. It runs *during* the walk
rather than over the results, because `grep` reads the files it visits:
post-filtering its matches would return no rows while having read every byte of
the file the rule was protecting.

**Filtered is not refused, and the two are separate questions on purpose.** A
screen answers "may I be told this exists"; the intent waterfall answers "may I
open it". A path a screen hides is still readable through `read` unless a rule
also vetoes `fs/read-intent`. So `permissions-fs` registers *both* rather than
deriving one from the other, and a deployment reading `ignore:` as access control
has misread it.

`fs/observed` records reads, which is what lets read-before-edit be a policy row
rather than a hard-coded rule.

**The honest scope: this bounds *tool-mediated* access.** Model-authored
`open(path, "w")` inside a code cell is unreachable from here by construction
(N1) — a deny-list needs a registered name. That is what the containment ladder
is for, and no wording in this module should suggest otherwise.

@module ph.seams.fs
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import anyio

from ..cordis import Boundary, Context, Disposer, Running, boundary_of, events, plugin, running
from ..session import Session
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel
from ._registry import claim_entry, claim_slot

__all__ = [
    "EditIntent",
    "FileSlice",
    "FileTooLarge",
    "FsService",
    "GrepMatch",
    "ReadIntent",
    "WalkDecision",
    "WalkScreen",
    "WriteIntent",
    "apply",
    "matches_glob",
    "read_before_edit",
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


WalkDecision = Literal["yield", "skip", "prune"]
"""What a screen says about one path `glob`/`grep` considered.

`yield` is "nothing to say"; `skip` drops this path from the results; `prune`
additionally refuses to enter it, which only means anything for a directory. A
closed `Literal` rather than two booleans so the consumers `match` exhaustively
and a fourth answer fails to type-check at every site that has to handle it."""

WalkScreen = Callable[[str, str, Any, bool], WalkDecision]
"""`(path, name, agent, is_dir) -> WalkDecision`. See `FsService.screen`.

**Strings, not a `Path`.** The walk pays for `pathlib` (P6-17), and a screen asked
about a `Path` puts the cost straight back: `_walk` holds the joined string
already and would build one per candidate purely to hand it over, which every
screen then converts back — `fs-local`'s ignore list wants the bare name, and
`permissions-fs` calls `as_posix()` on its first line.

`path` is absolute and **posix-separated on every platform**, because that is what
rule patterns and `matches_glob` are written in; `name` is the bare final
component, so the common screen is a set lookup rather than a parse. A screen that
genuinely needs a `Path` builds one on the branch that needs it.
"""

WalkDecider = Callable[[str, str, bool], WalkDecision]
"""A walk's screens, with its agent already bound. See `FsService._decider`."""


@dataclass(frozen=True, slots=True)
class _Screen:
    """One registered screen and the scope that owns it.

    The owner is the whole of P6-18's enumeration half: `hide` kept bare
    callables, so `Context.reaches` had nothing to filter on and an agent-scoped
    row concealed from every agent."""

    decide: WalkScreen
    owner: Context


def _scope_of(agent: Any) -> Context | None:
    """An agent's own scope, for the workspace root — or `None` for no agent.

    Duck-typed rather than importing the agent: `ph.seams.fs` sits below
    `ph.agent_loop`, and a seam that imported its consumer would invert the
    layering the whole plugin model rests on. Every driver in the tree assigns
    `self.ctx` in its constructor.

    **One caller, and `None` is benign there** (P6-32). It had two that read
    `None` oppositely — the policy boundary had to *refuse* it, the workspace
    root absorbs it — and requiring a stated `Boundary` deleted the first. What
    is left is the *physical* question, D21's "which worktree is this agent's",
    where `None` means no layer to override the resolver's own with: a fallback
    with nothing wider to widen to, which is why it is silent.
    """
    scope = getattr(agent, "ctx", None)
    return scope if isinstance(scope, Context) else None


@dataclass(frozen=True, slots=True)
class ReadIntent:
    """A read awaiting policy.

    Carries the `agent` its siblings carry, so a row that wants to *ask* about a read
    has somewhere to route the prompt.

    **`scope` is the boundary this intent was judged in** (P6-24) — stated here for
    all three intents, required, and ahead of `agent` so it cannot be defaulted:
    there is no honest fallback for "which boundary is this" once an agent is in
    play. Resolved once at the public method, from what the caller stated (P6-32).

    On the payload rather than passed to `_gate`, so the frozen record says which
    boundary it was judged in and the gate cannot be handed a different boundary than
    the one the screen used. No listener reads it yet.
    """

    path: Path
    scope: Context
    agent: Any = None


@dataclass(frozen=True, slots=True)
class WriteIntent:
    """A whole-file write awaiting policy."""

    path: Path
    content: str
    creating: bool
    scope: Context
    agent: Any = None


@dataclass(frozen=True, slots=True)
class EditIntent:
    """An in-place replacement awaiting policy."""

    path: Path
    old_text: str
    new_text: str
    replace_all: bool
    scope: Context
    agent: Any = None


class FsDenied(HarnessError, PermissionError):
    """A write or edit was refused by policy.

    A `HarnessError` as well as a `PermissionError`, so the refusal survives the
    trip to a consumer as a *denial*. Without that it reached `ToolFailure.kind`
    as an ordinary `failed`, which is the same hole C3 closes for
    `tools/pre-execute`: a Code Mode program could `except` a policy veto from
    this seam and route around it, and the durable record said the tool merely
    failed. `SandboxError` next door already had this; this seam, whose whole
    argument is being a gate rather than a report, did not.

    Still a `PermissionError`, so every existing catcher keeps working.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FS_DENIED")


class FileTooLarge(HarnessError):
    """A whole-file read was refused because the file is over the caller's cap (P7-01).

    A **failure, not a denial**, and the distinction is the one `FailureKind`
    exists to carry: no policy refused anything here — the caller named a bound
    and the file is bigger than it. A model that reads `denied` looks for
    permission to ask for; a model that reads this should pick a smaller file, and
    under Code Mode (C3) the difference decides whether its whole program ends.
    """

    failure_kind: FailureKind = "failed"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FILE_TOO_LARGE")


@dataclass(slots=True)
class FsService:
    """The service published as `ctx.fs`.

    **`agent` and `scope` are two values, and nothing checks them against each
    other** (P6-24, P6-32). `scope` is the policy boundary — which screens apply,
    which intent listeners the gate reaches — and is required, so a caller states it
    or does not call. `agent` is the *physical* key: `root_for` asks the D21 rebase
    resolver which worktree is this agent's, and the approval seam needs somewhere to
    route a prompt. `agent=child, scope=parent.ctx` gets the child's worktree under
    the parent's screens, which is legal by construction and invisible at the call
    site.

    **The two enforcement layers P6-32 rests on are not symmetric for this seam**
    (§5 rule 6). mypy names every unstated call site holding a typed reference; the
    runtime `TypeError` names the rest — but these five run inside a tool body, so an
    unconverted `ctx.fs.read(path)` reached through `ctx.<seam>` (which is `Any`)
    fails at *model* time and is normalized to an uncoded `failed`. The mypy layer is
    the only one that fires early here.
    """

    ctx: Context
    root: Path
    """Where relative paths resolve for an agent that holds no workspace — the
    process's own directory, and the answer `ph doctor` and a bare CLI probe
    get."""
    _observed: dict[Path, float] = field(default_factory=dict)
    """Last-read mtime per path — the state read-before-edit consults."""
    _screens: list[_Screen] = field(default_factory=list)
    """What the walk may show and where it may go. See `screen`."""
    _rebase: Callable[[Any], Path | None] | None = None
    """Where *this agent's* relative paths resolve. See `rebase`."""
    _rebase_by: Running | None = None
    """Who registered the resolver (P6-29). Set and cleared with it by `claim_slot`."""

    def rebase(
        self, resolver: Callable[[Any], Path | None], *, scope: Context | None = None
    ) -> Disposer:
        """Resolve relative paths per agent rather than per process (D21).

        A *slot*, not a list like `hide`: two answers to "where does this agent
        write" is a contradiction, and the first-match-wins reading a list would
        need is one nobody could configure. `None` from the resolver means the
        agent has no workspace and `root` stands.

        The workspace seam is not consulted from here directly, and that is the
        layering: `ctx.fs` would otherwise have to know which seam owns agent
        state, when what it needs is one path. The row that knows about both
        wires them (`workspace-lifecycle`), and a deployment that mounts no such
        row keeps exactly today's behaviour.
        """
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "_rebase",
            resolver,
            label="fs.rebase",
        )

    def root_for(self, agent: Any = None) -> Path:
        """This agent's root — its cwd, and what `bash` and `glob` run against.

        A resolver that raises falls back to `root` rather than failing the
        call: an agent whose workspace lookup broke still has to be able to read
        a file, and the wrong-but-working directory is a better failure than a
        traceback out of `read`.
        """
        if self._rebase is None or agent is None:
            return self.root
        try:
            # As the row that registered the resolver, **for the agent being
            # resolved** (P6-29). The one provider in the tree whose target is
            # already in hand and already derived — `_scope_of` is this module's
            # own — so this is the shape the other four would take if they were
            # handed a scope instead of an agent. P6-24 did not reach them: it
            # fixed the *policy* boundary, and a provider slot is a lifetime
            # question, so `CompactionSeam.engine_by` and its three siblings
            # still bind the registration's own layer and say so there.
            with running(self._rebase_by, _scope_of(agent)):
                resolved = self._rebase(agent)
        except Exception:
            log.warning("ph.seams.fs: the root resolver failed; using %s", self.root, exc_info=True)
            return self.root
        return resolved or self.root

    def named(self, path: str | Path, *, agent: Any = None) -> str:
        """How a path should be *written down* — relative to this agent's root.

        `resolve`'s inverse, and the form every path that reaches the model or
        the log takes. An absolute path inside the workspace names the same file
        as its relative form, but it also puts the machine and the run into the
        conversation: a read echoing `/tmp/ph-w-7/src/x.py` makes that string
        part of the transcript, so replaying the session against a fresh
        workspace — a retried job, a re-provisioned worktree, the same repo
        checked out elsewhere — changes every one of them and moves the
        provider's cached prefix (A11/A12) for a difference the conversation
        cannot see. The agent has no use for the outside of its workspace, so
        the outside does not appear.

        A path *outside* the workspace keeps its absolute form. It is not a name
        the workspace can express, and the approval that let it through is what
        made it legible in the first place — hiding it would be the one case
        where a relative path would mislead.

        `glob` and `grep` already answer this way, by slicing the walk's own
        prefix off; this is the same answer for the paths that arrive one at a
        time.
        """
        resolved = self.resolve(path, agent=agent)
        try:
            return resolved.relative_to(self.root_for(agent)).as_posix()
        except ValueError:
            return str(resolved)

    def resolve(self, path: str | Path, *, agent: Any = None) -> Path:
        """Resolve against the agent's workspace root.

        A relative path is the agent's business; an absolute one is passed
        through, because refusing it here would be a confinement claim this
        layer cannot make (N2) — which is also why the `worktree` tier bounds a
        relative write and not an absolute one.
        """
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.root_for(agent) / candidate)

    # ------------------------------------------------------------------ read --

    def screen(self, decide: WalkScreen, *, scope: Context) -> Disposer:
        """Decide what `glob` and `grep` may show, and where they may go (P6-19).

        One contribution point over **any** path the walk considers, directories
        included, answering `"yield" | "skip" | "prune"`. Pruning is why it takes
        directories: `deny secrets/**` should never enter the tree rather than refuse the
        same tree once per file inside it. The built-in ignore list is registered through
        this seam by `fs-local` like any other screen, so there is one way to drop a path
        and a deployment can configure it.

        Deliberately *not* a waterfall: a walk visits thousands of candidates, so
        awaiting a listener per file would make `grep` over a repository a different kind
        of operation, and asking a human about each one is not a thing anyone would sit
        through.

        The screen is asked about a path, **the agent walking**, and whether the path is
        a directory. The agent, because the root a rule is written against is per-agent
        once a containment tier is in force (D21): `secrets/**` names one directory in
        the person's checkout and a different one in each agent's worktree.

        **`Context`, not `Boundary`** (P6-32): this is a *registration*, so its `scope`
        becomes `_Screen.owner` and needs a real scope to be owned by and unwind with.

        **`scope` is required**, where every sibling registration in this package
        defaults it to the service's own context. P6-18 made this same field the input to
        `Context.reaches`, so omitting it would not mean "clean up with the mount", it
        would mean **"screen every agent"**: the widest policy in the seam, chosen by
        forgetting.

        Through `claim_entry`, which removes the entry **by identity**: `_Screen` and
        `FsPermissions` are frozen dataclasses and compare by value, so two mounts with
        equal rules would have had `list.remove` take the wrong one.
        """
        entry = _Screen(decide=decide, owner=scope)
        return claim_entry(scope, self._screens, entry, label="fs.screen")

    def _decider(self, agent: Any, *, scope: Boundary) -> WalkDecider | None:
        """This walk's screens, filtered and with its agent bound — or `None`.

        Resolved **once per walk, not once per path**: the visibility filter is
        fixed for the whole walk — the agent does not change halfway — so
        `reaches` would otherwise be re-answered for every screen for every one
        of eleven thousand candidates.

        `None` when no screen reaches this agent, so `_walk` can skip the ask
        entirely rather than call a predicate that always agrees. Note this is
        **not** the ordinary case for a mounted profile: `fs-local` registers the
        ignore screen on every mount, so the branch belongs to a hand-built
        service — a test's, or a consumer that wants a raw walk.

        `agent` is required rather than defaulted at the call sites: a default
        would silently answer for the *process* root, which is the exact bug
        per-agent roots were introduced to fix.
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
        agent: Any = None,
        session: Session | None = None,
    ) -> FileSlice:
        """Read a line window, after `fs/read-intent` allows it."""
        target = self.resolve(path, agent=agent)
        await self._gate(
            "fs/read-intent",
            ReadIntent(path=target, agent=agent, scope=boundary_of(scope, self.ctx)),
        )
        text = await anyio.to_thread.run_sync(
            lambda: target.read_text(encoding="utf-8", errors="replace")
        )
        all_lines = text.splitlines()
        window = all_lines[offset:] if limit is None else all_lines[offset : offset + limit]
        self._observe(target, session, agent)
        return FileSlice(
            path=self.named(target, agent=agent),
            text="\n".join(window),
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
        agent: Any = None,
        session: Session | None = None,
    ) -> bytes:
        """Read a whole file as bytes, after `fs/read-intent` allows it (P7-01).

        **The door a model-initiated attach has to go through** (I-9). A person
        may attach anything they can already open, and `AttachmentStore.save_path`
        is that door; a *model* may attach only what this seam lets it read, so
        `permissions-fs`, the workspace tier and every other `fs/read-intent`
        listener bound this exactly as they bound `read` — one line, and nothing
        for a policy row to opt into. Without it the tool producer had two bad
        options: reach around the seam with `Path.read_bytes`, which is the
        exfiltration primitive I-9 exists to prevent, or push a PNG through
        `read`'s `errors="replace"` decode, which returns something that is not
        the file.

        A method rather than a flag on `read`, because almost nothing they share
        survives the change of unit: `offset`, `limit`, `total_lines` and
        `truncated` are all statements about *lines*, `FileSlice.text` is `str`,
        and a `bytes | FileSlice` return would move that branch into every caller.
        What the two genuinely share is the gate, and that is one call here.

        `max_bytes` is answered from the file's own size **before it is opened**,
        so refusing a 2 GB video costs a `stat` rather than 2 GB of resident
        memory. A `stat` that fails is not treated as a refusal — the read below
        is about to raise the real `OSError`, and inventing a size limit for a
        file that does not exist would report the wrong thing.

        Recorded through `_observe` like every other read: what the read-before-edit
        rule and `fs/observed` are about is *this file was seen*, and a binary one
        was seen just as much.
        """
        target = self.resolve(path, agent=agent)
        await self._gate(
            "fs/read-intent",
            ReadIntent(path=target, agent=agent, scope=boundary_of(scope, self.ctx)),
        )
        if max_bytes is not None:
            try:
                size: int | None = target.stat().st_size
            except OSError:
                size = None
            if size is not None and size > max_bytes:
                raise FileTooLarge(
                    f"{self.named(target, agent=agent)} is {size} bytes, over the "
                    f"{max_bytes}-byte limit for this call"
                )
        content = await anyio.to_thread.run_sync(target.read_bytes)
        self._observe(target, session, agent)
        return content

    def _observe(self, target: Path, session: Session | None, agent: Any = None) -> None:
        """Remember this file's mtime, and record the read.

        The dict is keyed by the resolved `Path` because it answers a question
        about *this process* — has this file changed since we read it — and the
        log gets the workspace-relative name for the reason `named` gives: a
        record whose path is `/tmp/ph-w-7/...` describes a directory that will
        not exist the next time this session runs.
        """
        try:
            self._observed[target] = target.stat().st_mtime
        except OSError:  # pragma: no cover - raced deletion
            return
        if session is not None:
            session.append("fs/observed", {"path": self.named(target, agent=agent)})

    def observed_mtime(self, path: str | Path) -> float | None:
        return self._observed.get(self.resolve(path))

    # ----------------------------------------------------------------- write --

    async def write(
        self,
        path: str | Path,
        content: str,
        *,
        scope: Boundary,
        agent: Any = None,
        session: Session | None = None,
    ) -> Path:
        """Write a whole file, after `fs/write-intent` allows it."""
        target = self.resolve(path, agent=agent)
        intent = WriteIntent(
            path=target,
            content=content,
            creating=not target.exists(),
            agent=agent,
            scope=boundary_of(scope, self.ctx),
        )
        await self._gate("fs/write-intent", intent)
        await anyio.to_thread.run_sync(_write_text, target, content)
        self._observe(target, session)
        self.ctx.emit("fs/changed", target, contained=True)
        return target

    async def edit(
        self,
        path: str | Path,
        old_text: str,
        new_text: str,
        *,
        scope: Boundary,
        replace_all: bool = False,
        agent: Any = None,
        session: Session | None = None,
    ) -> int:
        """Replace text in place, after `fs/edit-intent` allows it.

        Returns the replacement count. A unique-match requirement is the tool's
        to enforce; this layer reports what it did.
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
        original = await anyio.to_thread.run_sync(lambda: target.read_text(encoding="utf-8"))
        count = original.count(old_text)
        if count == 0:
            raise ValueError(f"no occurrence of the target text in {target}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"{count} occurrences of the target text in {target}; pass replace_all "
                "or include more surrounding context to make it unique"
            )
        updated = original.replace(old_text, new_text, -1 if replace_all else 1)
        await anyio.to_thread.run_sync(_write_text, target, updated)
        self._observe(target, session)
        self.ctx.emit("fs/changed", target, contained=True)
        return count if replace_all else 1

    async def _gate(self, event: str, intent: Any) -> None:
        """Ask the policy rows about one intent, in the scope that owns it (P6-18).

        `scope=` is the fix and it is one argument: the waterfall defaulted to
        this *service's* context, which is the mount, so `collect` asked whether
        each listener reached the root — and an agent-scoped listener on
        `fs/read-intent` reaches only its own agent, so it never fired at all.
        The sibling gate `ToolRuntime` built has passed `scope=execution.scope`
        since it was written; this one is the same gate one seam over and did
        not, which is why `permissions-fs` could be mounted per agent and asked
        per process.

        Nothing changes for a globally mounted row: a global registration
        reaches everything, so the same listeners run in the same order.
        """

        async def inner(_intent: Any) -> str | None:
            return None

        # The boundary the intent was *judged in*, carried on the intent itself
        # rather than re-derived here (P6-24). The public method resolved it
        # once, from what the caller stated — so the payload a
        # listener receives says which boundary it is being asked about, and this
        # gate cannot answer a different question than the screen did.
        reason = await self.ctx.waterfall(event, intent, inner=inner, scope=intent.scope)
        if reason is not None:
            raise FsDenied(str(reason))

    # ------------------------------------------------------------------ find --

    async def glob(
        self,
        pattern: str,
        *,
        scope: Boundary,
        root: str | Path | None = None,
        limit: int = 1_000,
        agent: Any = None,
    ) -> list[str]:
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = self._decider(agent, scope=scope)
        return await anyio.to_thread.run_sync(lambda: list(_walk(base, pattern, limit, decide)))

    async def grep(
        self,
        pattern: str,
        *,
        scope: Boundary,
        root: str | Path | None = None,
        glob: str = "**/*",
        limit: int = 200,
        agent: Any = None,
    ) -> list[GrepMatch]:
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = self._decider(agent, scope=scope)
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error

        def scan() -> list[GrepMatch]:
            matches: list[GrepMatch] = []
            for found in _walk(base, glob, None, decide):
                candidate = Path(found)
                if not _greppable(candidate):
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    if expression.search(line):
                        matches.append(GrepMatch(path=found, line=number, text=line[:500]))
                        if len(matches) >= limit:
                            return matches
            return matches

        return await anyio.to_thread.run_sync(scan)


GREP_MAX_BYTES = 2 * 1024 * 1024
"""Files above this are skipped by `grep`: a build artifact that happens to
match `**/*` must not be read whole into memory line by line."""


def _walk(base: Path, pattern: str, limit: int | None, decide: WalkDecider | None) -> Iterator[str]:
    """Matching files under `base`, pruning refused directories *before* descent.

    `Path.glob` would materialize `node_modules` in full and then discard it; pruning
    `dirs` in place means a refused tree is never entered. Directories and files are
    visited in sorted order so results are stable.

    `decide` is consulted here rather than over the results, because `grep` reads
    every file this yields. It is asked about directories too (P6-19), which is what
    lets a policy row say "never go in there".

    **`None` means no screen reaches this walk** — the ordinary case, and the reason
    it is spelled as an absence rather than a predicate that always agrees: it lets
    the loop skip building a `Path` per candidate for a policy question nobody asked.

    Three shapes here are load-bearing on a walk of thousands of files: the relative
    path is a slice rather than `Path.relative_to`, the join happens once, and `glob`
    returns `str` rather than `Path`. `os.sep` is translated rather than assumed,
    because `matches_glob` speaks posix, and the branch is hoisted out of the loop
    since the answer is fixed for the process.
    """
    root = os.fspath(base)
    cut = len(root) + (0 if root.endswith(os.sep) else 1)
    native = os.sep != "/"

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
            if not matches_glob(posix(full[cut:]), pattern):
                continue
            if decide is not None and decide(posix(full), name, False) != "yield":
                continue
            yield full
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def matches_glob(candidate: str, pattern: str) -> bool:
    """`Path.glob` semantics: `*` stays inside one segment, `**` crosses them.

    Public because a permission row's path patterns and this seam's own `glob` tool
    must mean the same thing by `**/*.env`. Two glob dialects in one harness is a rule
    someone writes once and is then wrong about forever.

    **`*` must not cross a separator, and getting that wrong has a direction.**
    `fnmatch`'s `*` matches `/`, so `docs/*.md` would also match
    `docs/private/keys.md`. For a search box that is a quirk; for an ACL evaluated
    first-match-wins it is a hole, because the idiom the rules are written in is a
    narrow `allow` above a broad `deny` — and an `allow` silently wider than written
    permits what the `deny` under it was there to stop.
    """
    return _compiled(pattern).match(candidate) is not None


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern[str]:
    """One glob, as a regex. Cached, because a walk asks per candidate file.

    Bounded rather than unbounded: patterns come from a model's `glob` calls as
    well as from config, so an LRU is what keeps a session that greps a thousand
    different patterns from holding a thousand compiled regexes forever.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            # Any number of leading segments, including none — which is why
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
            close = pattern.find("]", index + 1)
            if close == -1:
                # An unbalanced bracket is a literal, not a syntax error: a
                # pattern typed by a model must not raise out of a policy check.
                parts.append(re.escape("["))
                index += 1
            else:
                inner = pattern[index + 1 : close].replace("\\", "\\\\")
                parts.append(f"[{'^' + inner[1:] if inner.startswith('!') else inner}]")
                index = close + 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def _greppable(path: Path) -> bool:
    """Text, and small enough to read whole."""
    try:
        if path.stat().st_size > GREP_MAX_BYTES:
            return False
        with path.open("rb") as handle:
            return b"\0" not in handle.read(8192)
    except OSError:
        return False


def _write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


_IGNORED_PARTS: tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
)
"""Directory names `fs-local` prunes unless a deployment says otherwise.

A **default**, not a rule (P6-19). It was a module-level `frozenset` consulted
inline by `_walk`, which made it the second of two ways to drop a path and the
only one nobody could configure — a repository with a `vendor/` or a `target/`
had no way to say so, and the `.gitignore` row had nowhere to land. It is now
`fs-local`'s `ignore` config, registered through `ctx.fs.screen` like any other
policy, so the walk has no built-in knowledge of it at all."""


class Config(WireModel):
    """Row config for the local filesystem provider."""

    root: str | None = None
    ignore: list[str] | None = None
    """Directory **names** `glob` and `grep` never enter. `None` keeps
    `_IGNORED_PARTS`; `[]` turns pruning off entirely.

    A bare final component, matched anywhere in the tree — `"dist"`, not
    `"build/dist"`, which would silently never fire. A path pattern is a
    `permissions-fs` rule's job, and unlike this one it also refuses to *read*
    what it hides: this list keeps noise out of a listing and protects nothing.

    Config rather than a constant (P6-19), and the distinction between `None` and
    `[]` is the whole reason it is `list[str] | None`: a repository with a
    `vendor/` or a `target/` had no way to add one, and a deployment that
    genuinely wants to see inside `.git` had no way to say so — while a plain
    `[]` default would have silently turned pruning off for every profile that
    did not mention it."""


@plugin("fs-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local filesystem provider, and its one built-in screen."""
    # Three answers, most specific first. `config.root` is a deployment saying
    # "always work here" and wins. `project_root` is *this mount's* working
    # directory, provided before the first row by whoever mounted — a daemon
    # mounts one composition once per session, and those sessions live in
    # different repositories, so the directory cannot come from the profile
    # (P5-14). `Path.cwd()` is the process's own, which is right for a mode that
    # is *in* the project and was silently wrong for every root a daemon held.
    provided = ctx.get("project_root")
    if config.root:
        root = Path(config.root).expanduser()
    elif provided is not None:
        root = Path(provided).expanduser()
    else:
        root = Path.cwd()
    service = FsService(ctx=ctx, root=root)
    ctx.provide("fs", service)
    ignored = frozenset(_IGNORED_PARTS if config.ignore is None else config.ignore)
    if not ignored:
        return

    def ignore(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        """The ignore list as an ordinary screen, not as a branch inside the walk.

        **Directories only**: the constant is matched against directory names, so a file
        called `dist` stays visible. Registered here rather than known to `_walk` so there
        is one mechanism for what the walk skips — which is what lets a deployment
        reconfigure it and what gives the `.gitignore` row somewhere to land.

        A set membership on the `name` the walk already had, which is what the `str` half
        of the `WalkScreen` contract is for.
        """
        return "prune" if is_dir and name in ignored else "yield"

    service.screen(ignore, scope=ctx)


@plugin("fs-read-before-edit", inject=["fs"])
async def read_before_edit(ctx: Context, config: Any) -> None:
    """Refuse an edit to a file this session has not read since it last changed.

    Its own row, not a rule inside `edit`, for two reasons: a deployment can
    drop it, and it applies to *every* editing tool rather than the one that
    remembered to check.
    """
    fs: FsService = ctx.fs

    async def gate(intent: EditIntent, next_: Callable[[], Any]) -> Any:
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
