"""What the `ph-code-graph` suites share: a grammar cache, a tree, an agent.

A `<package>_helpers.py` on `pythonpath`, as `clm_helpers` is, rather than a
`conftest.py`: `ph-rlm`'s tests import their conftest as a top-level module, and a
second file of that name anywhere in `testpaths` shadows it on `sys.path` —
adding one here broke ten of their modules at collection.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.keys import AGENTS, SESSIONS
from ph.testing import FAKE_OPTIONS
from ph_code_graph._config import GRAMMAR_CACHE_ENV

__all__ = ["SHARED", "grammar_cache", "new_agent", "shared_tree"]

SHARED = '''\
def shared(value):
    """The thing everyone calls."""
    return value * 2
'''
"""One definition every caller in `shared_tree` reaches."""


@pytest.fixture(scope="module", autouse=True)
def grammar_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One writable grammar cache for the whole run, outside every test's `$PH_CACHE`.

    `tree-sitter-language-pack` materializes even its *bundled* grammars into a
    writable directory on first use — the wheel ships them as an archive, not as
    loadable libraries — and fails hard when it cannot create one. So every test
    that parses needs it, including the ones that call `extract` directly and
    mount no row.

    **One directory, not the row's own path or one per module.** The suite's
    `_isolated_home` pins `$PH_CACHE` per test, so leaving the row to choose would
    re-materialize the grammars for each — and a directory per module did it once
    per module, four seconds each. Grammars are read-only content keyed by the
    pack's version, so sharing them shares nothing a test could disturb.

    Module-scoped, and autouse in each module that imports it: the variable is set
    only while that module runs. Set through the environment rather than
    `use_cache`, which is that function's operator-override path — so the run
    exercises the branch where an operator's own spelling wins. It reaches the
    command's processes too, because the manual's templates name it among the
    variables to inherit, and a confined command because the row grants it.
    """
    directory = tmp_path_factory.getbasetemp() / "tree-sitter-grammars"
    directory.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(GRAMMAR_CACHE_ENV, str(directory))
        yield directory


def shared_tree(root: Path, callers: int = 2) -> Path:
    """`root/pkg`: `helpers.py` defining `shared`, and `callers` modules calling it."""
    package = root / "pkg"
    package.mkdir(parents=True, exist_ok=True)
    (package / "helpers.py").write_text(SHARED, encoding="utf-8")
    for n in range(callers):
        (package / f"mod{n}.py").write_text(
            f"from helpers import shared\n\n\ndef caller{n}():\n    return shared({n})\n",
            encoding="utf-8",
        )
    return root


_SEQ = itertools.count()


def new_agent(ctx: Context) -> Any:  # noqa: ANN401
    """A fresh agent, on a session id nothing else has taken.

    Counted rather than fixed: a test that needs two agents (an index call and a
    query call) would otherwise collide on the session id, which fails as
    `SESSION_ALREADY_EXISTS` several frames from the cause.
    """
    return ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create(f"cg-{next(_SEQ)}"), FAKE_OPTIONS
    )
