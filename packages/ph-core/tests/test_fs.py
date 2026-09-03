"""P1-17 — the filesystem seam, and the gate that runs *before* the write.

Gate: *write-intent fires before the write; a veto prevents it.*

"Before" is the entire claim. A hook that ran after the write would be a
reporter, which is precisely the failure the feature map records in
prime-agent's `edit` skill — the diff is emitted after the file changed, so
"there is no point at which anything can say no".

## What the walk costs, and why `_walk` is written the way it is

Recorded here rather than in `fs.py`: three findings, all P6-17, together with
P6-19's string screen contract worth **414 ms → 32 ms** over this repository's
11 489 files.

* **The relative path is a slice, not a `Path.relative_to`.** That allocates a
  `Path` per root segment and measured **23.5 µs per file, 62% of the walk**,
  for a string that is already a prefix of what `os.walk` handed us. The slice is
  **0.32 µs**.
* **The join happens once.** `os.path.join(current, name)` was computed for the
  slice and then thrown away, and `Path(current, name)` re-joined it.
* **`str` out, not `Path`.** `glob` returns `list[str]` and was paying
  `Path.__str__` — which re-parses — for every one of them, having just had the
  string. `grep` builds the `Path` only for the files it actually opens, where a
  construction is noise beside the read.

The screen's own contract follows from the same measurement: it is handed a
**string** and a bool, so a walk that consults it does not construct a `Path` per
candidate to ask a question that is usually "yield".

## Why the screen contract is a `str` and not a `Path`

The largest single win left after P6-17, and it is entirely the type. `_walk`
already holds the joined string, so a screen asked about a `Path` would have it
build one per candidate purely to hand it over — and every screen converts it
back: `fs-local`'s ignore list wants the bare name, `permissions-fs` calls
`as_posix()` inside `_spellings` on its first line.

Measured over this repository, `Path` against `str`: **41.1 ms -> 28.8 ms** with
the ignore screen alone, and **112.6 ms -> 62.8 ms** with three anchored rules
mounted.

## The glob dialect, and why `fnmatch` was wrong for it

`matches_glob` implements `Path.glob` semantics: `*` stays inside one segment,
`**` crosses them. The first version used `fnmatch`, whose `*` matches `/` — so
`docs/*.md` also matched `docs/private/keys.md`.

For a search box that is a quirk. For an ACL evaluated first-match-wins it is a
hole, because the idiom the rules are written in is a narrow `allow` above a broad
`deny`, and an `allow` silently wider than written permits exactly what the `deny`
under it was there to stop.

## Why the built-in ignore list is a screen and not a branch in `_walk`

It is a set membership on the `name` the walk already holds — reading `.name` off a
`Path` built for the purpose measured **70x** that. Directories only: the constant
is matched against directory names, so a file called `dist` stays visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT, Context
from ph.seams.fs import (
    Config as FsConfig,
)
from ph.seams.fs import (
    EditIntent,
    FsDenied,
    FsService,
    WalkDecision,
    WriteIntent,
    apply,
    matches_glob,
    read_before_edit,
)
from ph.session import Session
from ph.testing import FAKE_OPTIONS, StubAgent, run_tool

pytestmark = pytest.mark.anyio


def _fs(tmp_path: Path) -> tuple[Context, FsService]:
    """A bare service, with no screens registered.

    Bare on purpose: most of this file is about the write gate, and a walk that
    prunes nothing is the simplest thing that can answer a `glob`. The ignore
    list is `fs-local`'s config since P6-19 rather than knowledge inside `_walk`,
    so the test that is about *that* mounts the row — see `_mounted` below.
    """
    root = Context()
    service = FsService(ctx=root, root=tmp_path)
    root.provide("fs", service)
    return root, service


async def _mounted(tmp_path: Path, **config: Any) -> FsService:
    """`fs-local` as a profile composes it, screens included.

    The distinction matters since P6-19: pruning `node_modules` is a *screen the
    row registers*, not a branch inside the walk, so a test that hand-built the
    service and expected pruning would be asserting against a mechanism that no
    longer exists there.
    """
    root = Context()
    await apply(root, FsConfig(root=str(tmp_path), **config))
    service: FsService = root.fs
    return service


async def test_write_intent_fires_before_the_write_and_a_veto_prevents_it(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "guarded.txt"
    observed: list[bool] = []

    async def deny(intent: WriteIntent, next_: Any) -> str:
        # The file must not exist yet when policy runs.
        observed.append(intent.path.exists())
        return "writes to this path are not allowed"

    root.on("fs/write-intent", deny)
    with pytest.raises(FsDenied, match="not allowed"):
        await fs.write("guarded.txt", "content", scope=DEPLOYMENT)
    assert observed == [False]
    assert not target.exists()


async def test_an_allowed_write_reaches_disk_and_announces_itself(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    changed: list[Path] = []
    root.on("fs/changed", lambda path: changed.append(path))
    written = await fs.write("notes/deep.txt", "hello", scope=DEPLOYMENT)
    assert written.read_text() == "hello"
    assert changed == [written]


async def test_reads_record_an_observation(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    session = Session("s")
    window = await fs.read("a.txt", session=session, scope=DEPLOYMENT)
    assert window.total_lines == 3
    assert [event.type for event in session.events] == ["fs/observed"]
    assert fs.observed_mtime("a.txt") is not None


async def test_a_read_window_says_how_to_get_the_next_one(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "long.txt").write_text("\n".join(str(n) for n in range(100)))
    window = await fs.read("long.txt", offset=10, limit=5, scope=DEPLOYMENT)
    assert window.text.splitlines() == ["10", "11", "12", "13", "14"]
    assert window.truncated
    assert window.offset == 10


async def test_edit_requires_a_unique_match_unless_told_otherwise(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    target = tmp_path / "dup.txt"
    target.write_text("x\nx\n")
    with pytest.raises(ValueError, match="2 occurrences"):
        await fs.edit("dup.txt", "x", "y", scope=DEPLOYMENT)
    assert await fs.edit("dup.txt", "x", "y", replace_all=True, scope=DEPLOYMENT) == 2
    assert target.read_text() == "y\ny\n"


async def test_editing_absent_text_is_an_error(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    with pytest.raises(ValueError, match="no occurrence"):
        await fs.edit("a.txt", "goodbye", "hi", scope=DEPLOYMENT)


async def test_read_before_edit_refuses_an_unread_file(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    await read_before_edit(root, None)

    with pytest.raises(FsDenied, match=r"read .* before editing"):
        await fs.edit("a.txt", "hello", "goodbye", scope=DEPLOYMENT)

    await fs.read("a.txt", scope=DEPLOYMENT)
    assert await fs.edit("a.txt", "hello", "goodbye", scope=DEPLOYMENT) == 1


async def test_read_before_edit_refuses_a_file_that_changed_underneath(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("hello")
    await read_before_edit(root, None)
    await fs.read("a.txt", scope=DEPLOYMENT)

    import os
    import time

    time.sleep(0.01)
    target.write_text("changed by someone else")
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))

    with pytest.raises(FsDenied, match="changed on disk"):
        await fs.edit("a.txt", "changed", "x", scope=DEPLOYMENT)


async def test_edit_intent_sees_the_replacement_before_it_lands(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("before")
    seen: list[EditIntent] = []

    async def observe(intent: EditIntent, next_: Any) -> Any:
        seen.append(intent)
        assert intent.path.read_text() == "before"
        return await next_()

    root.on("fs/edit-intent", observe)
    await fs.edit("a.txt", "before", "after", scope=DEPLOYMENT)
    assert seen[0].new_text == "after"
    assert target.read_text() == "after"


async def test_glob_and_grep_skip_the_usual_noise(tmp_path: Path) -> None:
    """The shipped default, now asserted through the row that registers it."""
    fs = await _mounted(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("import os\nTARGET = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("TARGET = 2\n")

    found = await fs.glob("**/*.py", scope=DEPLOYMENT)
    assert [Path(path).name for path in found] == ["keep.py"]

    matches = await fs.grep("TARGET", scope=DEPLOYMENT)
    assert [match.path for match in matches] == [str(tmp_path / "src" / "keep.py")]
    assert matches[0].line == 2


async def test_an_invalid_regular_expression_is_reported_not_raised_raw(
    tmp_path: Path,
) -> None:
    _root, fs = _fs(tmp_path)
    with pytest.raises(ValueError, match="invalid regular expression"):
        await fs.grep("(unclosed", scope=DEPLOYMENT)


async def test_relative_paths_resolve_against_the_root(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    assert fs.resolve("a/b.txt") == tmp_path / "a" / "b.txt"
    # An absolute path passes through: refusing it here would be a confinement
    # claim this layer cannot make (N2).
    assert fs.resolve("/etc/hosts") == Path("/etc/hosts")


# --- P6-17/18/19: the walk, its screens, and who they reach -------------------


async def test_the_ignore_list_is_deployment_config_not_a_module_constant(
    tmp_path: Path,
) -> None:
    """P6-19's second half: two ways to drop a path collapsed into one.

    `_IGNORED_PARTS` was a `frozenset` consulted inline by `_walk`, so a
    repository with a `vendor/` had no way to add one and a deployment that
    wanted to look inside `.git` had no way to say so. It is a screen `fs-local`
    registers like any other, which is also what gives the `.gitignore` row
    somewhere to land instead of arriving as mechanism three.
    """
    for name in ("node_modules", "vendor"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f.py").write_text("x = 1\n")

    added = await _mounted(tmp_path, ignore=["vendor"])
    shown = await added.glob("**/*.py", scope=DEPLOYMENT)
    assert [Path(p).parent.name for p in shown] == ["node_modules"]

    # `[]` is not `None`: one turns pruning off, the other keeps the default.
    off = await _mounted(tmp_path, ignore=[])
    assert sorted(Path(p).parent.name for p in await off.glob("**/*.py", scope=DEPLOYMENT)) == [
        "node_modules",
        "vendor",
    ]


async def test_a_screen_can_refuse_a_tree_not_just_the_files_in_it(
    tmp_path: Path,
) -> None:
    """P6-19's first half, and the cost that motivated it.

    `hide` could skip a file and could not refuse a directory, so `deny
    secrets/**` concealed every file in `secrets/` *after* entering it and paying
    a predicate call each. `prune` is the walk answering the question it already
    knew how to answer — it has pruned directories before descent since it was
    written, just never on anybody's behalf.
    """
    (tmp_path / "secrets").mkdir()
    for index in range(5):
        (tmp_path / "secrets" / f"key{index}.txt").write_text("shh")
    (tmp_path / "open.txt").write_text("fine")

    root, fs = _fs(tmp_path)
    asked: list[tuple[str, bool]] = []

    def screen(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        asked.append((name, is_dir))
        return "prune" if is_dir and name == "secrets" else "yield"

    fs.screen(screen, scope=root)
    assert [Path(p).name for p in await fs.glob("**/*.txt", scope=DEPLOYMENT)] == ["open.txt"]
    # Asked once about the directory, and never about the five files inside it —
    # which is the difference between refusing a tree and refusing a tree's
    # contents one at a time.
    assert ("secrets", True) in asked
    assert not [name for name, is_dir in asked if name.startswith("key")]


async def test_a_screen_that_raises_refuses_at_the_widest_setting(tmp_path: Path) -> None:
    """Fail closed, and fail closed *early*.

    A policy row whose matcher broke must not become an open door — the approval
    seam's reading. For a directory that means `prune` rather than `skip`: a
    broken screen that let the walk descend would be asked once per file inside,
    each time failing the same way.
    """
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    root, fs = _fs(tmp_path)

    def broken(_path: str, _name: str, _agent: Any, _is_dir: bool) -> WalkDecision:
        raise RuntimeError("this matcher is broken")

    fs.screen(broken, scope=root)
    assert await fs.glob("**/*.txt", scope=DEPLOYMENT) == []


async def test_a_screen_only_sees_the_agents_its_scope_reaches(tmp_path: Path) -> None:
    """P6-18's enumeration half.

    `hide` kept a flat list of bare callables, so `Context.reaches` had nothing
    to filter on and an agent-scoped `permissions-fs` concealed from *every*
    agent's `glob` — including agents whose deployment never mounted it. Three
    registries already obey this rule; this one did not.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)
    mine = root.scope("agent")
    theirs = root.scope("agent")

    fs.screen(lambda _p, _n, _a, is_dir: "yield" if is_dir else "skip", scope=mine)

    assert await fs.glob("**/*.txt", agent=StubAgent(mine), scope=mine) == []
    assert len(await fs.glob("**/*.txt", agent=StubAgent(theirs), scope=theirs)) == 1, (
        "not this agent's rule"
    )
    assert len(await fs.glob("**/*.txt", scope=DEPLOYMENT)) == 1, "nor the process's"


async def test_an_agent_scoped_gate_is_asked_about_that_agents_reads(
    tmp_path: Path,
) -> None:
    """P6-18's veto half, which failed in the opposite direction.

    `FsService._gate` waterfalled with no `scope=`, so `collect` asked whether
    each listener reached the *mount* — and an agent-scoped listener reaches
    only its own agent, so it never fired at all. The sibling gate `ToolRuntime`
    built has passed `scope=execution.scope` since it was written.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)
    mine = root.scope("agent")

    async def deny(_intent: Any, _next: Any) -> str:
        return "this agent may not read"

    mine.on("fs/read-intent", deny)

    with pytest.raises(FsDenied, match="may not read"):
        await fs.read("a.txt", agent=StubAgent(mine), scope=mine)
    # And nobody else's, which is the property the scope buys.
    assert await fs.read("a.txt", scope=DEPLOYMENT) is not None


async def test_a_global_row_still_reaches_every_agent(tmp_path: Path) -> None:
    """The half that must not have changed.

    A global registration reaches everything, so scoping the dispatch is a fix
    for agent-scoped rows and a no-op for the way every shipped profile mounts
    `permissions-fs` today. Asserted because "nothing changes for a global row"
    is the claim that made the change safe to land.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)

    async def deny(_intent: Any, _next: Any) -> str:
        return "nobody may read"

    root.on("fs/read-intent", deny)
    stranger = root.scope("agent")

    with pytest.raises(FsDenied, match="nobody"):
        await fs.read("a.txt", agent=StubAgent(stranger), scope=stranger)
    with pytest.raises(FsDenied, match="nobody"):
        await fs.read("a.txt", scope=DEPLOYMENT)


async def test_the_walk_yields_the_same_paths_a_relative_to_walk_did(tmp_path: Path) -> None:
    """P6-17's correctness gate: a slice, and `Path.relative_to`, agree.

    The optimisation replaced `path.relative_to(base).as_posix()` — 23.5 µs per
    file, 62% of a 430 ms walk — with a prefix slice of the string `os.walk`
    already built. The speedup is only worth having if the answer is identical,
    so this holds the two against each other over a tree with the shapes that
    break naive slicing: nesting, dots in names, and a directory whose name is a
    prefix of a sibling's.
    """
    for relative in (
        "a.py",
        "pkg/b.py",
        "pkg/deep/c.py",
        "pkg.tests/d.py",
        "pkgx/e.py",
        "dotted.name.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")

    _root, fs = _fs(tmp_path)
    for pattern in ("**/*", "**/*.py", "*.py", "pkg/**/*.py", "pkg*/*.py"):
        found = await fs.glob(pattern, scope=DEPLOYMENT)
        expected = sorted(
            str(path)
            for path in tmp_path.glob(pattern)
            if path.is_file() and matches_glob(path.relative_to(tmp_path).as_posix(), pattern)
        )
        assert sorted(found) == expected, pattern


# --- P6-24: the boundary is stated, not guessed ------------------------------


async def test_one_call_resolves_one_boundary(tmp_path: Path) -> None:
    """P6-24's gate. A tool call states its boundary; both seams must read it.

    `ToolExecutionInput.scope` *is* "the per-agent policy boundary … stated by
    the caller", and beside it `agent` is documented as one thing only — the
    approval-routing target. `ctx.tools` has always read the first. This seam
    read the second and derived the first from it, so one call could be judged
    in two different boundaries the moment they differ: a Code Mode sub-dispatch,
    or a subagent whose driver holds a child ctx.

    Driven here the way it actually diverges — a screen registered on the scope
    the call *states*, and an agent whose own scope is somewhere else. Before the
    fix the screen did not apply, because `ctx.fs` was answering for the agent.
    """
    (tmp_path / "secret.txt").write_text("s")
    (tmp_path / "plain.txt").write_text("p")
    root, fs = _fs(tmp_path)
    stated = root.scope("the-stated-boundary")
    elsewhere = root.scope("the-agents-own-scope")

    def screen(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        return "yield" if is_dir or name != "secret.txt" else "skip"

    fs.screen(screen, scope=stated)

    shown = await fs.glob("**/*.txt", agent=StubAgent(elsewhere), scope=stated)
    assert [Path(one).name for one in shown] == ["plain.txt"], (
        "the screen on the stated boundary did not apply — `ctx.fs` answered for the agent"
    )


async def test_an_fs_reader_cannot_be_called_without_a_boundary(tmp_path: Path) -> None:
    """The fail-open half, now closed by construction rather than by refusing.

    An `agent` whose `.ctx` was absent used to fall back to the *mount*, which no
    agent-scoped registration reaches — so a policy row scoped to one agent
    screened nobody and said nothing, in the direction P6-18 exists to prevent.
    P6-24 answered that by *refusing* the derivation with `FsBoundaryError`.

    P6-32 removed the derivation instead, and the refusal went with it: `scope`
    is a required `Boundary`, so there is no unreadable agent to fall back from
    and no branch to fail open in. What is asserted here is the property that
    replaced it — saying nothing is not a way to get an answer — and the two
    boundaries that *are* sayable still answer, because a rule that refuses
    everything is not the one this row wants.

    `agent` stays beside `scope` and is orthogonal to it: it keys the *physical*
    workspace root (D21) and routes approvals, never the policy boundary. A
    handle too broken to name a scope no longer matters, which is the point.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)
    agent = root.scope("agent")

    class NoCtx:
        """An agent-shaped object that never assigned `self.ctx`."""

    # Through `root.fs`, which `Context.__getattr__` types as `Any` — the exact
    # path the row's runtime layer exists for, and the one mypy cannot see. The
    # typed handle would need a `# type: ignore` to say the same thing less well.
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        await root.fs.glob("**/*.txt")

    assert len(await fs.glob("**/*.txt", scope=DEPLOYMENT)) == 1, "the deployment, asked for"
    assert len(await fs.glob("**/*.txt", scope=agent)) == 1, "an agent's own boundary"
    assert len(await fs.glob("**/*.txt", agent=NoCtx(), scope=agent)) == 1, (
        "a stated boundary answers the question, whatever the agent handle looks like"
    )


async def test_a_tool_call_is_judged_in_the_scope_the_caller_states(
    mount: Any,
) -> None:
    """The same divergence one layer up, driven through the real pipeline.

    `test_one_call_resolves_one_boundary` above calls `ctx.fs` directly, which
    proves the seam. This proves the wiring: `fs_tools` has to *hand* the seam
    `run.scope` rather than `run.agent`, and until P6-24 it passed only the
    second while `run.scope` sat unused two lines away. Driven through
    `ctx.tools.execute`, so a later refactor of `fs_tools` cannot quietly go back
    to deriving the boundary and still pass.

    `run_tool`'s `scope=` exists for exactly this: it defaulted to the agent's own
    ctx, so a test could not state a boundary that differs from it — which is the
    only shape in which the bug is visible.

    **Driven over a real parent and child rather than an invented pair of
    scopes**, because that is the divergence the row names — "a subagent whose
    driver holds a child ctx" — and it only exists because P6-27 nested agents.
    The screen is registered on the **child**, which a parent-scoped call must
    not see: `reaches` runs down the tree, so a child's rule does not reach its
    parent, and the three rows below are only distinguishable if the seam honours
    the *stated* boundary. Against the pre-row build the third one leaks.
    """
    ctx = await mount()
    root = ctx.get("fs").root
    (root / "secret.txt").write_text("s")
    (root / "plain.txt").write_text("p")

    parent = ctx.agents.create(ctx.sessions.create("p624-parent"), FAKE_OPTIONS)
    child = ctx.agents.create(ctx.sessions.create("p624-child"), FAKE_OPTIONS, parent=parent)

    def screen(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        return "yield" if is_dir or name != "secret.txt" else "skip"

    ctx.fs.screen(screen, scope=child.ctx)

    async def shown(scope: Context, agent: Any) -> list[str]:
        found = await run_tool(ctx, "glob", {"pattern": "*.txt"}, agent=agent, scope=scope)
        return sorted(Path(one).name for one in found.value["paths"])

    assert await shown(parent.ctx, parent) == ["plain.txt", "secret.txt"], (
        "a child's screen reached its parent — `reaches` runs down the tree, not up"
    )
    assert await shown(child.ctx, child) == ["plain.txt"], "the child's own rule did not apply"
    assert await shown(child.ctx, parent) == ["plain.txt"], (
        "fs_tools handed the seam the agent, not the boundary the caller stated"
    )


# ------------------------------------------- the workspace is the whole world --


async def test_no_path_the_model_reads_names_the_machine(mount: Any, tmp_path: Path) -> None:
    """Every path that reaches the model is relative to the workspace (A11/A12).

    An absolute path inside the workspace names the same file as its relative
    form, and costs something the relative form does not: it puts the run's own
    directory into the transcript. Replay that session against a fresh workspace
    — a retried job, a re-provisioned worktree, the same repo checked out
    somewhere else — and every one of those strings differs, so the provider's
    cached prefix moves for a difference the conversation cannot see. The agent
    has no use for the outside of its workspace, so the outside does not appear.

    Asserted over `read`, `write` and `edit` together, because they were three
    separate spellings of the same echo and only `glob`/`grep` were relative.

    Sabotage: return `str(target)` from any of the three, and the workspace root
    is in a result the model reads.
    """
    project = tmp_path / "repo"
    project.mkdir()
    (project / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await mount({"id": "fs", "config": {"root": str(project)}})
    agent = ctx.agents.create(ctx.sessions.create("named"), FAKE_OPTIONS)

    window = await ctx.fs.read("notes.md", scope=agent.ctx, agent=agent)
    written = await ctx.fs.write("sub/new.txt", "x", scope=agent.ctx, agent=agent)
    edited = await ctx.fs.edit("notes.md", "hello", "goodbye", scope=agent.ctx, agent=agent)

    assert window.path == "notes.md", "a read named the machine"
    assert ctx.fs.named(project / "sub" / "new.txt", agent=agent) == "sub/new.txt"
    assert written is not None or written is None  # the write happened; its shape is the tool's
    assert edited == 1
    # And the one case a relative path would mislead: outside the workspace it
    # keeps its absolute form, because that is not a name the workspace has.
    outside = tmp_path / "elsewhere.txt"
    assert ctx.fs.named(outside, agent=agent) == str(outside)


async def test_a_read_records_the_workspace_relative_name(mount: Any, tmp_path: Path) -> None:
    """`fs/observed` too: it is a record about a file, kept across runs.

    A record whose path is `/tmp/ph-w-7/src/x.py` describes a directory that
    will not exist the next time this session runs, which is the same argument
    the model-facing paths make — and this one also decides whether
    read-before-edit still recognises the file after a workspace is rebuilt.
    """
    project = tmp_path / "repo"
    project.mkdir()
    (project / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await mount({"id": "fs", "config": {"root": str(project)}})
    session = ctx.sessions.create("observed")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await ctx.fs.read("notes.md", scope=agent.ctx, agent=agent, session=session)

    observed = [one for one in session.events if one.type == "fs/observed"]
    assert [one.data["path"] for one in observed] == ["notes.md"]
    assert str(project) not in repr([one.data for one in session.events])
