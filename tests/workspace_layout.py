"""Where this workspace keeps its code, answered once.

`packages/<dist>/src/<package>` is the layout every distribution here uses, and
three root gates need to walk it: `test_wire.py` (every model uses the shared
wire base), `test_json_narrowing.py` (no reader fabricates a JSON value), and
`test_packaging.py` (the console script resolves). A fourth,
`test_fixture_types.py`, walks both halves — those packages *and* the suites,
through `workspace_tests`. Each had written the glob out itself, which is the
shape `workspace_packages`' own argument is against — *discovered, not listed*,
so that a package added tomorrow is covered without anybody remembering. Two
implementations of "discovered" go stale the same way a list does, just more
slowly.

Importable by name because `pyproject.toml`'s `pythonpath` puts `tests` on the
path, which is the same slot `daemon_helpers`, `tui_helpers` and `app_fixtures`
already occupy.

@module tests.workspace_layout
"""

from __future__ import annotations

import pathlib

__all__ = [
    "REPO",
    "workspace_modules",
    "workspace_packages",
    "workspace_tests",
]

REPO = pathlib.Path(__file__).resolve().parent.parent
"""The checkout root — the directory `packages/` sits in."""


def workspace_packages() -> list[pathlib.Path]:
    """Every top-level package this workspace ships, as a directory.

    A package that is present but does not import fails at its caller rather
    than being skipped here, because a skip is how 113 models once went
    unchecked.
    """
    return sorted(
        path for path in (REPO / "packages").glob("*/src/*") if (path / "__init__.py").exists()
    )


def workspace_modules() -> list[tuple[str, pathlib.Path]]:
    """Every shipped `.py`, with the package-relative name a gate reports."""
    return [
        (f"{package.name}/{path.relative_to(package)}", path)
        for package in workspace_packages()
        for path in sorted(package.rglob("*.py"))
    ]


def workspace_tests() -> list[tuple[str, pathlib.Path]]:
    """Every test module in the workspace, with the name a gate reports.

    The suites rather than the shipped packages, which is the other half of this
    file: `packages/<dist>/tests` beside the root `tests/`, both of which mypy
    already checks (`pyproject.toml`'s `files`). Discovered for the same reason
    `workspace_packages` is — a suite added tomorrow is covered without anybody
    remembering — and `conftest.py` is included because a fixture defined there
    is the one every suite receives.
    """
    roots = [*sorted((REPO / "packages").glob("*/tests")), REPO / "tests"]
    # `glob` on the root rather than a test for the file: it yields the conftest
    # when there is one and nothing when there is not, so its being a single
    # known file does not have to become a branch.
    found = [path for root in roots for path in sorted(root.rglob("*.py"))]
    return [(str(path.relative_to(REPO)), path) for path in [*found, *REPO.glob("conftest.py")]]
