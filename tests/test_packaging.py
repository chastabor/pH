"""What the seven manifests promise about each other, held against the tree.

**Every fact here is declared in a file nothing imports.** A version, a pin, a
`packages` entry, a classifier — read by the build backend at upload time and by
nobody before then, which is what makes this whole class of mistake silent and
late: the gate that catches it is a person running `pip install` after the
release, and by then the release is immutable.

The one this file was originally written for is gone, and the deletion is worth
recording. `phern` used to be an *empty* distribution at the workspace root
whose only content was a dependency list, because `uv tool install` installs a
package and links only the executables of the package it was asked for — so the
root had to restate the app's `[project.scripts]` verbatim, and two tests here
existed to keep the copies from drifting. `phern` is now `packages/phern`
itself: one package, one declaration, nothing to hold together.
"""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from workspace_layout import REPO, workspace_tests

WORKSPACE_MEMBERS = tuple(sorted(p.name for p in (REPO / "packages").iterdir() if p.is_dir()))
"""The directories under `packages/`, discovered rather than listed.

They are *not* the distribution names and must not be read as them:
`packages/ph-core` builds `ph-core`, but `packages/phern` builds `phern`, and
nothing requires a directory and the thing it builds to agree. So this addresses
the filesystem, and every assertion below reads the name out of the manifest it
finds there.
"""

DEPLOYMENT = REPO / "packages" / "phern" / "pyproject.toml"
"""The distribution a person installs, and the only one that names the others."""


def project_of(pyproject: Path) -> dict[str, object]:
    """The `[project]` table of one manifest."""
    table = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert isinstance(table, dict)
    return table


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


# ---------------------------------------------------------------------------
# The seven versions, held against each other and against the pins between them.
# `phern` pins the other six with `==`, so a member bumped without it is not a
# skew a resolver papers over but an install that cannot resolve at all —
# discovered by whoever runs `pip install phern` after the upload.


def test_every_distribution_is_released_in_lockstep() -> None:
    versions: dict[str, str] = {}
    for directory in WORKSPACE_MEMBERS:
        member = project_of(REPO / "packages" / directory / "pyproject.toml")
        versions[str(member["name"])] = str(member["version"])

    assert len(versions) == len(WORKSPACE_MEMBERS), (
        f"two packages declare the same distribution name: {sorted(versions)}"
    )
    assert len(set(versions.values())) == 1, (
        f"the distributions disagree about the release: {versions}"
    )


def test_the_workspace_root_is_not_a_distribution() -> None:
    """It builds nothing, so it must not claim it can.

    A `[project]` table here is how the empty-umbrella arrangement comes back by
    accident: uv would treat the root as a member again, `uv build
    --all-packages` would emit a wheel with no modules in it, and whoever
    published the directory would ship it beside the real one.
    """
    root = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))

    assert "project" not in root, "the workspace root declared itself a distribution again"
    assert "build-system" not in root, "the workspace root acquired a build backend"


def test_the_deployment_pins_every_other_member_at_the_version_it_is() -> None:
    """`phern` is what "pH" means to an installer, so its list is what can be wrong."""
    deployment = project_of(DEPLOYMENT)
    release = str(deployment["version"])
    dependencies = deployment["dependencies"]
    assert isinstance(dependencies, list)
    pins = {str(entry) for entry in dependencies if str(entry).startswith("ph-")}

    expected = {
        f"{project_of(REPO / 'packages' / directory / 'pyproject.toml')['name']}=={release}"
        for directory in WORKSPACE_MEMBERS
        if directory != "phern"
    }
    assert pins == expected, (
        "`phern` does not pin the members it is made of at this release — "
        f"declared {sorted(pins)}, workspace holds {sorted(expected)}"
    )


def test_the_deployment_owns_the_only_console_script() -> None:
    """One command, declared once, in the package whose module it points into.

    Two declarations is what the empty umbrella needed and what this file used to
    police. A second one appearing means either the root is a distribution again
    (the test above catches that) or a member grew a command of its own — which
    is allowed, but should be a decision rather than a surprise, and `phern` has
    to stay the one a person gets.
    """
    scripts: dict[str, dict[str, str]] = {}
    for directory in WORKSPACE_MEMBERS:
        declared = project_of(REPO / "packages" / directory / "pyproject.toml").get("scripts")
        if declared:
            assert isinstance(declared, dict)
            scripts[directory] = declared

    assert scripts == {"phern": {"phern": "ph_app.cli:main"}}, (
        f"the console scripts this workspace installs are not the one expected: {scripts}"
    )


def test_the_declared_console_script_resolves() -> None:
    """A string that names nothing is the failure this produces at install time."""
    scripts = project_of(DEPLOYMENT)["scripts"]
    assert isinstance(scripts, dict)

    for command, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        entry = getattr(import_module(module_name), attribute, None)
        assert callable(entry), f"`{command}` points at nothing callable"


def test_the_core_module_reports_the_version_its_manifest_declares() -> None:
    """`ph.__version__` is hand-written, so it is the copy that drifts.

    Nothing in the build reads it: hatchling takes the version from the manifest,
    which means a release can ship a wheel whose metadata says 0.2.0 and whose
    `ph.__version__` still says 0.1.0 — and the first reader of the wrong one is
    a bug report quoting a version that was never released.
    """
    from ph import __version__

    assert __version__ == project_of(REPO / "packages" / "ph-core" / "pyproject.toml")["version"]


def test_every_distribution_ships_the_marker_that_makes_its_types_visible() -> None:
    """A strict-mypy tree whose wheels are untyped is a private joke (PEP 561).

    Every manifest claims `Typing :: Typed`, and the classifier is only true if
    `py.typed` is beside the package's `__init__.py` — `packages` in the wheel
    config ships whatever is in that directory, so the marker's presence in the
    checkout is what puts it in the wheel.
    """
    for directory in WORKSPACE_MEMBERS:
        manifest = REPO / "packages" / directory / "pyproject.toml"
        config = tomllib.loads(manifest.read_text(encoding="utf-8"))
        included = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        for relative in included:
            marker = manifest.parent / relative / "py.typed"
            assert marker.exists(), f"{manifest.parent.name}: {marker} is missing"
