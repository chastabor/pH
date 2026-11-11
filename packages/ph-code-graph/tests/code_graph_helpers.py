"""What the `ph-code-graph` suites share: a grammar cache, a tree, an agent.

A `<package>_helpers.py` on `pythonpath`, as `clm_helpers` is, rather than a
`conftest.py`: `ph-rlm`'s tests import their conftest as a top-level module, and a
second file of that name anywhere in `testpaths` shadows it on `sys.path` —
adding one here broke ten of their modules at collection.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from ph.testing import new_agent
from ph_code_graph._config import GRAMMAR_CACHE_ENV

__all__ = ["SHARED", "cache_beside_the_tree", "grammar_cache", "new_agent", "shared_tree"]

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


@pytest.fixture(autouse=True)
def cache_beside_the_tree(tmp_path: Path) -> Iterator[Path]:
    """`$PH_CACHE` outside the test's workspace, where a deployment's is.

    The suite's `_isolated_home` puts it at `tmp_path/cache`, and `tmp_path` is
    also the workspace a mounted profile gets — so every directory the session's
    row grants under the cache would overlap the tree, and a grant that overlaps
    the workspace is refused (`ph.seams.sandbox.grants_apart`). That is the rule
    working, on a layout no deployment has. Autouse in each module that imports
    it, after `_isolated_home`, so this spelling wins.

    And a `$PH_RUNTIME` of the test's own, and short: a host listens on a socket
    under it, the suite otherwise shares the developer's real runtime directory,
    and a socket's address is at most 108 bytes.
    """
    directory = tmp_path.parent / f"{tmp_path.name}-cache"
    runtime = Path(tempfile.mkdtemp(prefix="ph-run-", dir="/tmp"))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("PH_CACHE", str(directory))
            patch.setenv("PH_RUNTIME", str(runtime))
            yield directory
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


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
