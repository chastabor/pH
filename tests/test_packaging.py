"""The two declarations of the `ph` console script, held against each other.

`ph` is declared twice on purpose. `ph-app` owns it because that is the
distribution the command lives in — `pip install ph-app` gives you a working
`ph` — and the workspace root restates it because `uv tool install` links the
executables of the package it was *asked for* and not of its dependencies, so a
root with no `[project.scripts]` installs the whole harness and then reports
that it provides no commands. There is no uv option that lends a dependency's
console script, so the restatement is unavoidable; what is avoidable is the two
copies drifting.

**They drift silently and late.** Nothing imports either string: `uv sync` never
builds the root's launcher, so a rename of `ph_app.cli:main` leaves the root
pointing at a dead target and the first person to find out is whoever runs
`uv tool install .` — after which `ph` fails at startup with an `ImportError`
from a module they did not name. This is `ph_app.web.DEFAULT_HOST`'s lesson in
the other direction: two spellings of one value that "agreed by accident" until
one moved.

So both are read here, and the target is resolved rather than only compared —
matching strings that both name a function nobody exports would pass a test and
fail an install.
"""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from workspace_layout import REPO, workspace_tests


def scripts_of(pyproject: Path) -> dict[str, str]:
    """The `[project.scripts]` table of one manifest."""
    return dict(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get("scripts", {}))


def test_the_root_restates_ph_apps_console_script_exactly() -> None:
    root = scripts_of(REPO / "pyproject.toml")
    app = scripts_of(REPO / "packages" / "ph-app" / "pyproject.toml")
    assert app == {"ph": "ph_app.cli:main"}, "ph-app is the owner; update this test with it"
    assert root == app, (
        "the workspace root and `ph-app` declare different console scripts — "
        "`uv tool install .` would install a launcher that disagrees with the one "
        f"`pip install ph-app` gives you: root={root}, ph-app={app}"
    )


def test_every_declared_console_script_resolves() -> None:
    """A string that names nothing is the failure this pair produces at install time."""
    for source in (REPO / "pyproject.toml", REPO / "packages" / "ph-app" / "pyproject.toml"):
        for command, target in scripts_of(source).items():
            module_name, _, attribute = target.partition(":")
            entry = getattr(import_module(module_name), attribute, None)
            assert callable(entry), f"{source.name}: `{command}` points at nothing callable"


def test_every_suite_is_type_checked_and_collected() -> None:
    """A `packages/*/tests` this workspace has, against the two lists that must name it.

    `workspace_layout` discovers those directories and `test_fixture_types.py`
    walks them; `pyproject.toml` writes them out by hand. A new package whose
    suite is missing from these two is the quiet failure — mypy checks nobody's
    annotations there and pytest collects none of its tests, while every gate
    that globs keeps reporting green.

    **Only these two.** `pythonpath` and `mypy_path` name the suites that ship an
    *importable helper* beside their tests, which is a smaller set on purpose:
    `ph-code-graph` and `ph-text-index` hold nothing but `test_*.py`, so their
    absence there is correct and asserting all four lists together would fail on
    a truth.
    """
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    suites = {name.rsplit("/", 1)[0] for name, _ in workspace_tests() if "/" in name}
    checked = set(config["tool"]["mypy"]["files"])
    collected = set(config["tool"]["pytest"]["ini_options"]["testpaths"])

    assert suites <= checked, (
        f"`mypy.files` does not name {sorted(suites - checked)} — nothing type-checks it"
    )
    assert suites <= collected, (
        f"`pytest.testpaths` does not name {sorted(suites - collected)} — nothing runs it"
    )
