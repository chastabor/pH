"""What the `ph-code-graph` suites share: a grammar cache, a tree, and an agent.

This is a `<package>_helpers.py` on `pythonpath`, like `clm_helpers`, and not a
`conftest.py`. The tests of `ph-rlm` import their conftest as a top-level module.
A second file of that name anywhere in `testpaths` shadows it on `sys.path`.
When one was added here, ten of their modules failed at collection.
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
"""The one definition that every caller in `shared_tree` reaches."""


@pytest.fixture(scope="module", autouse=True)
def grammar_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One writable grammar cache for the whole run, outside the `$PH_CACHE` of
    every test.

    `tree-sitter-language-pack` materializes even its *bundled* grammars into a
    writable directory on first use. The wheel ships them as an archive, not as
    loadable libraries, and the pack fails hard when it cannot create the
    directory. Every test that parses therefore needs the cache, including the
    tests that call `extract` directly and mount no row.

    **One directory, not the path of the row and not one per module.** The
    `_isolated_home` fixture of the suite sets `$PH_CACHE` for each test. If the
    row chose the path, the grammars would be materialized again for each test.
    A directory per module did it one time per module, at four seconds each.
    Grammars are read-only content, keyed by the version of the pack, so sharing
    them shares nothing that a test could disturb.

    The fixture is module-scoped, and autouse in each module that imports it. The
    variable is set only while that module runs. It is set through the
    environment and not through `use_cache`. The environment is the
    operator-override path of that function, so the run exercises the branch in
    which the value of an operator wins. The variable also reaches the processes
    of the command, because the templates of the manual name it among the
    variables to inherit. It reaches a confined command because the row grants
    the directory.
    """
    directory = tmp_path_factory.getbasetemp() / "tree-sitter-grammars"
    directory.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(GRAMMAR_CACHE_ENV, str(directory))
        yield directory


@pytest.fixture(autouse=True)
def cache_beside_the_tree(tmp_path: Path) -> Iterator[Path]:
    """`$PH_CACHE` outside the workspace of the test, where a deployment has it.

    The `_isolated_home` fixture of the suite puts it at `tmp_path/cache`, and
    `tmp_path` is also the workspace that a mounted profile gets. Every directory
    that the session row grants under the cache would then overlap the tree, and
    a grant that overlaps the workspace is refused
    (`ph.seams.sandbox.grants_apart`). That refusal is the rule working
    correctly, on a layout that no deployment has. The fixture is autouse in each
    module that imports it, after `_isolated_home`, so this value wins.

    The fixture also sets a `$PH_RUNTIME` for the test, with a short path. A host
    listens on a socket under it. Without this, the suite shares the real runtime
    directory of the developer. The address of a socket is at most 108 bytes.
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
