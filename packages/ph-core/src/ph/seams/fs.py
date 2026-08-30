"""`ctx.fs` — filesystem access with an interception point before every access.

Every read, every write and every edit passes through a waterfall
(`fs/read-intent`, `fs/write-intent`, `fs/edit-intent`) *before* it touches the
disk. That ordering is the whole value: a policy plugin that ran after the write
would be a reporter, not a gate — which is precisely the failure the feature map
records in prime-agent's `edit` skill, where the diff is emitted **after** the
file changed, so "there is no point at which anything can say no".

**Enumeration is filtered, not gated** (`screen`). `glob` and `grep` visit
thousands of candidates and a waterfall per candidate would be both slow and
absurd — nobody approves nine hundred prompts — so a policy row registers a
synchronous screen and the walk never yields, or never enters, what it refuses.
It runs *during* the walk rather than over the results, because `grep` reads the
files it visits: post-filtering its matches would return no rows while having
read every byte of the file the rule was protecting.

**Filtered is not refused, and the two are separate questions on purpose.** A
screen answers "may I be told this exists"; the intent waterfall answers "may I
open it". A path a screen hides is still readable through `read` unless a rule
also vetoes `fs/read-intent` — which is exactly right for `fs-local`'s ignore
list, whose whole job is to keep `node_modules` out of a listing rather than to
protect it, and which is why `permissions-fs` registers *both* a screen and a
gate rather than deriving one from the other. A deployment reading `ignore:` as
access control has misread it; a deployment writing a `deny` rule gets both.

`fs/observed` records reads, which is what lets read-before-edit be a policy
row rather than a hard-coded rule.

The honest scope: this bounds *tool-mediated* access. Model-authored
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

from ..cordis import Context, Disposer, Running, events, plugin, running
from ..session import Session
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel
from ._registry import claim_entry, claim_slot

__all__ = [
    "EditIntent",
    "FileSlice",
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

**Strings, not a `Path`, and that is a measurement rather than a preference.**
P6-17's whole finding was that the walk pays for `pathlib` — and a screen asked
about a `Path` puts the cost straight back, because `_walk` holds the joined
string already and would build one per candidate purely to hand it over. Every
screen then converts it back: `fs-local`'s ignore list wants the bare name, and
`permissions-fs` calls `as_posix()` inside `_spellings` on its first line.
Measured over this repository, `Path` against `str`: **41.1 ms → 28.8 ms** with
the ignore screen alone, and **112.6 ms → 62.8 ms** with three anchored rules
mounted — the largest single win left after P6-17, and it is entirely the type.

`path` is absolute and **posix-separated on every platform**, because that is
what rule patterns and `matches_glob` are written in; `name` is the bare final
component, so the common screen is a set lookup rather than a parse. A screen
that genuinely needs a `Path` builds one on the branch that needs it, which for
`permissions-fs` is the rare `outside-workspace` check."""

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
    """An agent's own scope, for the visibility rule — or `None` for no agent.

    Duck-typed rather than importing the agent: `ph.seams.fs` sits below
    `ph.agent_loop` and a seam that imported its consumer would invert the
    layering the whole plugin model rests on. Every driver in the tree assigns
    `self.ctx` in its constructor, and anything that does not simply falls back
    to the service's own context — which is exactly today's behaviour.

    **Deriving this at all is the wrong altitude, and P6-24 is the row.** The
    pipeline already states the boundary: `ToolExecutionInput.scope` *is* "the
    per-agent policy boundary", says in its own docstring that "the registry
    must not guess it", and sits beside an `agent` field documented only as the
    approval-routing target — two values, passed separately by `code_mode`
    today. `fs_tools` hands this seam the second and leaves `run.scope` unused,
    so `ctx.fs` and `ctx.tools` can resolve different boundaries for one call.
    The fallback is also fail-*open*: an agent whose `.ctx` is absent lands on
    the mount, which no agent-scoped screen reaches — the mirror of the bug
    P6-18 fixed. Until that row lands, this is the honest approximation and the
    caveat belongs here rather than only in the plan.
    """
    scope = getattr(agent, "ctx", None)
    return scope if isinstance(scope, Context) else None


@dataclass(frozen=True, slots=True)
class ReadIntent:
    """A read awaiting policy.

    Carries the `agent` its siblings carry, so a row that wants to *ask* about
    one has somewhere to route the prompt — a permission rule that could only
    ever refuse a read would be a narrower thing than the one that gates writes,
    for no reason anybody chose.
    """

    path: Path
    agent: Any = None


@dataclass(frozen=True, slots=True)
class WriteIntent:
    """A whole-file write awaiting policy."""

    path: Path
    content: str
    creating: bool
    agent: Any = None


@dataclass(frozen=True, slots=True)
class EditIntent:
    """An in-place replacement awaiting policy."""

    path: Path
    old_text: str
    new_text: str
    replace_all: bool
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


@dataclass(slots=True)
class FsService:
    """The service published as `ctx.fs`."""

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
            # own — so this is the shape the other four take once P6-24 gives
            # them a scope to read instead of an agent to guess from.
            with running(self._rebase_by, _scope_of(agent)):
                resolved = self._rebase(agent)
        except Exception:
            log.warning("ph.seams.fs: the root resolver failed; using %s", self.root, exc_info=True)
            return self.root
        return resolved or self.root

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
        included, answering `"yield" | "skip" | "prune"`. It replaces `hide`,
        which could refuse a file and could not refuse a tree: `deny secrets/**`
        concealed every file in `secrets/` while still entering the directory and
        paying a predicate call per file, and `permissions-fs` had no way to say
        "never go in there" even though `_walk` already knew how to prune. Two
        ways to drop a path collapse into one — the built-in ignore list is
        registered through this seam by `fs-local` like any other screen, which
        is also what lets a deployment configure it and what gives the
        `.gitignore` row somewhere to land instead of arriving as mechanism
        three.

        Deliberately *not* a waterfall, for the reason `hide` gave and which
        still holds: a walk visits thousands of candidates, so asking a human
        about each one is not a thing anyone would sit through, and awaiting a
        listener per file would make `grep` over a repository a different kind of
        operation. A row that both screens and vetoes keeps the two consistent;
        this seam does not derive one from the other, because they answer
        different questions ("may I be told this exists" and "may I open it").

        The screen is asked about a path, **the agent walking**, and whether the
        path is a directory. The agent, because the root a rule is written
        against is per-agent once a containment tier is in force (D21): a rule
        written `secrets/**` names one directory in the person's checkout and a
        different one in each agent's worktree, and a screen that could not tell
        them apart would answer for the wrong tree.

        **`scope` is required**, where every sibling registration in this package
        defaults it to the service's own context. That default is harmless where
        the owner is a *lifetime* handle — `rebase` two methods up still takes
        it — but P6-18 made this same field the input to `Context.reaches`, so
        omitting it would no longer mean "clean up with the mount", it would mean
        **"screen every agent"**: the widest policy in the seam, chosen by
        forgetting. `_decider` refuses to make that inference about the agent for
        the same reason and says so; making it about the owner would be the same
        mistake one field over, and it is the one P6-18 was written to fix.

        Through `claim_entry`, which removes the entry it appended **by
        identity**. `_Screen` is a frozen dataclass and so compares by value, and
        `FsPermissions` is one too — so two mounts with equal rules registering
        against one service would have had `list.remove` take the wrong one.
        That is the defect `_registry`'s own docstring exists to name.
        """
        entry = _Screen(decide=decide, owner=scope)
        return claim_entry(scope, self._screens, entry, label="fs.screen")

    def _decider(self, agent: Any) -> WalkDecider | None:
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
        target = _scope_of(agent) or self.ctx
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
        offset: int = 0,
        limit: int | None = 2_000,
        agent: Any = None,
        session: Session | None = None,
    ) -> FileSlice:
        """Read a line window, after `fs/read-intent` allows it."""
        target = self.resolve(path, agent=agent)
        await self._gate("fs/read-intent", ReadIntent(path=target, agent=agent))
        text = await anyio.to_thread.run_sync(
            lambda: target.read_text(encoding="utf-8", errors="replace")
        )
        all_lines = text.splitlines()
        window = all_lines[offset:] if limit is None else all_lines[offset : offset + limit]
        self._observe(target, session)
        return FileSlice(
            path=str(target),
            text="\n".join(window),
            offset=offset,
            lines=len(window),
            total_lines=len(all_lines),
            truncated=limit is not None and offset + len(window) < len(all_lines),
        )

    def _observe(self, target: Path, session: Session | None) -> None:
        try:
            self._observed[target] = target.stat().st_mtime
        except OSError:  # pragma: no cover - raced deletion
            return
        if session is not None:
            session.append("fs/observed", {"path": str(target)})

    def observed_mtime(self, path: str | Path) -> float | None:
        return self._observed.get(self.resolve(path))

    # ----------------------------------------------------------------- write --

    async def write(
        self, path: str | Path, content: str, *, agent: Any = None, session: Session | None = None
    ) -> Path:
        """Write a whole file, after `fs/write-intent` allows it."""
        target = self.resolve(path, agent=agent)
        intent = WriteIntent(
            path=target, content=content, creating=not target.exists(), agent=agent
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

        reason = await self.ctx.waterfall(
            event, intent, inner=inner, scope=_scope_of(intent.agent) or self.ctx
        )
        if reason is not None:
            raise FsDenied(str(reason))

    # ------------------------------------------------------------------ find --

    async def glob(
        self,
        pattern: str,
        *,
        root: str | Path | None = None,
        limit: int = 1_000,
        agent: Any = None,
    ) -> list[str]:
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = self._decider(agent)
        return await anyio.to_thread.run_sync(lambda: list(_walk(base, pattern, limit, decide)))

    async def grep(
        self,
        pattern: str,
        *,
        root: str | Path | None = None,
        glob: str = "**/*",
        limit: int = 200,
        agent: Any = None,
    ) -> list[GrepMatch]:
        base = self.resolve(root, agent=agent) if root is not None else self.root_for(agent)
        decide = self._decider(agent)
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

    `Path.glob` would materialize `node_modules` in full and then discard it;
    pruning `dirs` in place means a refused tree is never entered. Directories
    and files are visited in sorted order so results are stable.

    `decide` is consulted here rather than over the results, because `grep` reads
    every file this yields: filtering afterwards would hide the matches and still
    have opened the file. It is asked about directories too (P6-19), which is
    what lets a policy row say "never go in there" rather than refusing the same
    tree once per file inside it. **`None` means no screen reaches this walk** —
    the ordinary case, and the whole reason it is spelled as an absence rather
    than as a predicate that always agrees: it is what lets the loop below skip
    building a `Path` per candidate for a policy question nobody asked.

    Three findings, all P6-17, and together with P6-19's string screen contract
    they are **414 ms → 32 ms** over this repository's 11 489 files:

    * **The relative path is a slice, not a `Path.relative_to`.** That allocates
      a `Path` per root segment and measured **23.5 µs per file, 62% of the
      walk**, for a string that is already a prefix of what `os.walk` handed us.
      The slice is 0.32 µs. `os.sep` is translated rather than assumed, because
      `matches_glob` speaks posix and Windows does not, and the branch is hoisted
      out of the loop since the answer is fixed for the process.
    * **The join happens once.** `os.path.join(current, name)` was computed for
      the slice and then thrown away, and `Path(current, name)` re-joined it.
    * **`str` out, not `Path`.** `glob` returns `list[str]` and was paying
      `Path.__str__` — which re-parses — for every one of them, having just
      had the string. `grep` builds the `Path` for the files it actually opens,
      where a construction is noise beside the read.
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

    Public because a permission row's path patterns and this seam's own `glob`
    tool must mean the same thing by `**/*.env`. Two glob dialects in one
    harness is a rule someone writes once and is then wrong about forever.

    **`*` does not cross a separator, and getting that wrong has a direction.**
    The first version of this was `fnmatch`, whose `*` matches `/` — so
    `docs/*.md` also matched `docs/private/keys.md`. For a search box that is a
    quirk; for an ACL evaluated first-match-wins it is a hole, because the idiom
    the rules are written in is a narrow `allow` above a broad `deny`, and an
    `allow` that is silently wider than written permits what the `deny` under it
    was there to stop.
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
    root = Path(config.root).expanduser() if config.root else Path.cwd()
    service = FsService(ctx=ctx, root=root)
    ctx.provide("fs", service)
    ignored = frozenset(_IGNORED_PARTS if config.ignore is None else config.ignore)
    if not ignored:
        return

    def ignore(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        """The ignore list as an ordinary screen, not as a branch inside the walk.

        Directories only: the constant was matched against directory names and
        nothing else, and a file called `dist` was always visible. Registered
        here rather than known to `_walk` so there is one mechanism for what the
        walk skips — which is what lets a deployment reconfigure it above and
        what gives the `.gitignore` row somewhere to land.

        A set membership on the `name` the walk already had, which is what the
        `str` half of the `WalkScreen` contract is for: reading `.name` off a
        `Path` built for the purpose measured 70x this.
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
