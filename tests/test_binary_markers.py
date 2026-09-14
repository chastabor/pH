"""Every test that drives a real `git` or `jj` says so, in a marker.

`conftest.NEEDS_BINARY` skips the tests carrying `needs_git` and `needs_jj` when
the host has no such binary, and its docstring makes a claim this file exists to
make true:

    A marker cannot be forgotten the same way — it is spelled at the test, and
    the hook finds every test that carries it.

The second half is true. The first was not: the hook finds every test that
*carries* the marker, and nothing found a test that drove the binary without
one. `packages/ph-text-index` had such a test and it took the Linux CI job down
with `FileNotFoundError: [Errno 2] No such file or directory: 'jj'`, which is
not reproducible on any machine that has jj installed.

**Two things the first version of this walk got wrong, both of which made it a
proxy for the rule rather than the rule.**

It read `node.decorator_list` alone, so it could not see
`pytestmark = [pytest.mark.anyio, pytest.mark.needs_jj]` — which is how
`test_workspace_jj.py`, `test_workspace_git.py` and `test_workspace_checkpoint.py`
all declare it. Every test in those three files looked like an offender, and the
five "fixes" that produced were redundant decorators on tests the module had
already marked.

And it matched the fixture name in the test's own body, so one indirection hid a
call completely. `_tiered` and `_repo_with_materials` are module-local helpers
that build repositories, and `ph.testing`'s own `jj_agent` / `worktree_agent`
wrap `jj_repo` / `git_repo` and are the *preferred* entry points — 37 call sites
between them. The original bug, one call deep, would have walked straight past.

So the walk now resolves what a name transitively reaches, and reads a marker
wherever a marker can be written.
"""

from __future__ import annotations

import ast
from pathlib import Path

from workspace_layout import workspace_tests

DRIVES_BINARY = {
    "jj_repo": "needs_jj",
    "jj_agent": "needs_jj",
    "git_repo": "needs_git",
    "worktree_agent": "needs_git",
}
"""The `ph.testing` helpers that shell out, and the marker each one obliges.

Keyed on the *helper* rather than the binary because that is what a test says
out loud. The two `*_agent` wrappers are here in their own right: they are what
most tests actually call, and a walk that knew only `jj_repo` would have been
blind to the majority of the tree.

A test that reaches for `subprocess.run(["jj", ...])` by hand still slips past —
and should, because it would also be doing something these helpers exist to stop
it doing.
"""

MARKERS = frozenset(DRIVES_BINARY.values())


def called_names(node: ast.AST) -> set[str]:
    """Every plain function name called anywhere inside `node`."""
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def markers_on(node: ast.FunctionDef | ast.AsyncFunctionDef, module: set[str]) -> set[str]:
    """The binary markers in force for one test — its own, plus the module's.

    `pytestmark` is not decoration a reader can skim past: it is how three of the
    four files this rule guards declare the thing being checked.
    """
    decorators = " ".join(ast.unparse(decorator) for decorator in node.decorator_list)
    return module | {marker for marker in MARKERS if marker in decorators}


def module_markers(tree: ast.Module) -> set[str]:
    """The markers a module-level `pytestmark` puts on every test in the file."""
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]
    declared = " ".join(ast.unparse(node.value) for node in assignments)
    return {marker for marker in MARKERS if marker in declared}


def reaching(tree: ast.Module) -> dict[str, str]:
    """Name -> marker, for every name in this module that reaches a real binary.

    Seeded with the `ph.testing` helpers and then closed over the module's own
    functions, so a helper that wraps one — `_tiered`, `_repo_with_materials` —
    obliges the marker exactly as the helper it wraps does. Iterated to a fixed
    point because a helper may wrap a helper.
    """
    reaches = dict(DRIVES_BINARY)
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("test_")
    ]
    while True:
        found = False
        for helper in helpers:
            if helper.name in reaches:
                continue
            for name in called_names(helper) & reaches.keys():
                reaches[helper.name] = reaches[name]
                found = True
                break
        if not found:
            return reaches


def test_every_test_driving_a_real_binary_declares_it() -> None:
    offenders: list[str] = []
    for name, path in workspace_tests():
        source = path.read_text(encoding="utf-8")
        # A file that never spells a helper cannot call one, and this skips the
        # parse for all but a handful of the tree's test modules.
        if not any(f"{helper}(" in source for helper in DRIVES_BINARY):
            continue

        tree = ast.parse(source)
        reaches = reaching(tree)
        on_module = module_markers(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            declared = markers_on(node, on_module)
            for called in sorted(called_names(node) & reaches.keys()):
                marker = reaches[called]
                if marker not in declared:
                    offenders.append(
                        f"{name}:{node.lineno} {node.name} reaches {called}() "
                        f"without @pytest.mark.{marker}"
                    )

    assert not offenders, (
        "these drive a real binary but do not declare it, so on a host without "
        "that binary they error instead of skipping — which is a CI failure "
        f"nobody can reproduce locally: {offenders}"
    )


def test_the_walk_reads_a_marker_wherever_one_can_be_written() -> None:
    """The two spellings, against a module shaped like the ones being guarded.

    A walk that silently stopped seeing one of these would report every test in
    three real files as an offender — which is exactly what the first version
    did, and what nothing caught until the redundant markers it produced were
    noticed by hand.
    """
    tree = ast.parse(
        "import pytest\n"
        "pytestmark = [pytest.mark.anyio, pytest.mark.needs_jj]\n"
        "async def _helper(ctx, path):\n"
        "    return await jj_repo(ctx, path)\n"
        "async def test_by_module(ctx, path):\n"
        "    await _helper(ctx, path)\n"
        "@pytest.mark.needs_git\n"
        "async def test_by_decorator(ctx, path):\n"
        "    await git_repo(ctx, path)\n"
    )
    tests = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if node.name.startswith("test_")
    }
    on_module = module_markers(tree)

    assert on_module == {"needs_jj"}, "a module-level `pytestmark` list was not read"
    assert reaching(tree)["_helper"] == "needs_jj", "one indirection was not followed"
    assert "needs_jj" in markers_on(tests["test_by_module"], on_module)
    assert "needs_git" in markers_on(tests["test_by_decorator"], on_module)


def test_the_walk_finds_the_suites_it_is_supposed_to() -> None:
    """A walk over nothing asserts nothing, and would pass forever."""
    modules = [path for _, path in workspace_tests()]

    assert modules, "the walk found no test modules at all"
    assert any("ph-text-index" in str(path) for path in modules), "a known suite is missing"
    assert all(isinstance(path, Path) for path in modules)
