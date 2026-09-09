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

REPO = Path(__file__).resolve().parent.parent


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
