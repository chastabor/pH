"""Every test that drives a real `git` or `jj` says so, in the marker.

`conftest.NEEDS_BINARY` skips the tests carrying `needs_git` and `needs_jj` when
the host has no such binary, and its docstring makes a claim this file exists to
make true:

    A marker cannot be forgotten the same way — it is spelled at the test, and
    the hook finds every test that carries it.

The second half holds. The first does not, and the difference is the whole bug:
the hook finds every test that *carries* the marker, and nothing at all finds a
test that drives the binary without one. Seven had accumulated — six of them
`needs_git`, invisible on any developer machine and on both CI runners because
git is everywhere, and one `needs_jj` that took down the Linux CI job with
`FileNotFoundError: [Errno 2] No such file or directory: 'jj'` from inside
`packages/ph-text-index`.

**Why it survives being obvious.** The fixtures are `jj_repo` and `git_repo`
from `ph.testing`, and calling one is the entire tell — but the failure only
appears on a host that lacks the binary, so the author never sees it, the
reviewer never sees it, and the suite is green everywhere it is run by anyone
who would fix it. It surfaces on somebody else's machine, as an error rather
than a skip, in a package that has nothing to do with version control.

So the rule is checked the way the tests it guards are: by walking the tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

from workspace_layout import REPO, workspace_tests

NEEDS_MARKER = {"jj_repo": "needs_jj", "git_repo": "needs_git"}
"""The fixture that drives a binary, and the marker that declares it.

Keyed on the *fixture* rather than the binary because that is what a test says
out loud. A test that reaches for `subprocess.run(["jj", ...])` directly would
slip past this — and should, because it would also be doing something these two
fixtures exist to stop it doing.
"""


def _declared(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The binary markers on one test, however they are spelled."""
    return {
        marker
        for decorator in node.decorator_list
        for marker in NEEDS_MARKER.values()
        if marker in ast.unparse(decorator)
    }


def _suite_files() -> list[Path]:
    """Every `test_*.py` under `packages/*/tests` and the root `tests/`.

    Through `workspace_layout` rather than a glob of its own, for the reason
    that module states: a suite added tomorrow is covered without anybody
    remembering this file.
    """
    return sorted({REPO / name for name, _ in workspace_tests() if name.endswith(".py")})


def test_every_test_driving_a_real_binary_declares_it() -> None:
    offenders: list[str] = []
    for path in _suite_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            body = ast.unparse(node)
            declared = _declared(node)
            for fixture, marker in NEEDS_MARKER.items():
                if f"{fixture}(" in body and marker not in declared:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno} {node.name} "
                        f"calls {fixture}() without @pytest.mark.{marker}"
                    )

    assert not offenders, (
        "these drive a real binary but do not declare it, so on a host without "
        "that binary they error instead of skipping — which is a CI failure "
        f"nobody can reproduce locally: {offenders}"
    )


def test_the_walk_finds_the_suites_it_is_supposed_to() -> None:
    """A walk over nothing asserts nothing, and would pass forever.

    The failure this prevents is `workspace_tests()` changing shape — returning
    directories, say, or paths relative to something else — after which the loop
    above parses an empty list and reports every suite clean.
    """
    files = _suite_files()

    assert len(files) > 50, f"only {len(files)} test modules found; the walk is broken"
    assert all(path.exists() for path in files), "workspace_tests() named a path that is not there"
    assert any("ph-text-index" in str(path) for path in files), "a known suite is missing"
