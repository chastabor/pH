"""Python skills (P3-18): discovered from `SKILL.md`, callable in a cell.

The row's gate is one sentence — *a skill's `run()` is callable in a cell* — and
it spans three layers that were built at different times: discovery here, the
venv's editable install (P3-07), and the guest's `wrap_skill_module` (P3-08),
which until now had no producer. The last test drives the whole path against a
real kernel; the rest pin the validation, because a skill that half-registers is
a name the catalog offers and the runtime cannot import.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
from conftest import HOST_RUNTIME_ROW
from runtime_helpers import run_ipython_cell

from ph.system_prompt import render_prompt
from ph.system_prompt.assembly import AssembleContext
from ph.testing import FAKE_OPTIONS
from ph_rlm.skills import MAX_SKILL_BYTES, discover, render_catalog

pytestmark = pytest.mark.anyio


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "search the web for a query",
    front_name: str | None = None,
    package: bool = False,
    body: str = "",
) -> Path:
    """One skill directory. `package=True` makes it importable."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {front_name if front_name is not None else name}\n"
        f"description: {description}\n---\n\n# {name}\n\nInstructions here.\n"
    )
    if package:
        module = name.replace("-", "_")
        (directory / "pyproject.toml").write_text(
            textwrap.dedent(f"""
                [build-system]
                requires = ["hatchling"]
                build-backend = "hatchling.build"

                [project]
                name = "{name}"
                version = "0.0.1"
                requires-python = ">=3.12"

                [tool.hatch.build.targets.wheel]
                packages = ["{module}"]
                """).strip()
            + "\n"
        )
        package_dir = directory / module
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text(body or "def run(**kwargs):\n    return kwargs\n")
    return directory


def row(paths: list[Path]) -> dict[str, Any]:
    return {
        "id": "rlm-skills-python",
        "name": "rlm-skills-python",
        "config": {"paths": [str(path) for path in paths]},
    }


# ------------------------------------------------------------- discovery --


def test_a_documentation_skill_is_discovered_without_a_package(tmp_path: Path) -> None:
    write_skill(tmp_path, "note-taking")
    (skill,) = discover([str(tmp_path)])

    assert (skill.name, skill.description) == ("note-taking", "search the web for a query")
    assert skill.module is None
    assert skill.requirement is None, "nothing to install for a docs-only skill"


def test_a_package_skill_carries_its_import_name(tmp_path: Path) -> None:
    """Read from `pyproject.toml`, not guessed: `acme-websearch` imports as
    `acme_websearch`, which is the normal case rather than the odd one."""
    write_skill(tmp_path, "acme-websearch", package=True)
    (skill,) = discover([str(tmp_path)])

    assert skill.module == "acme_websearch"
    assert skill.requirement == str(tmp_path / "acme-websearch")


def test_a_name_that_disagrees_with_its_directory_is_refused(tmp_path: Path) -> None:
    """Addressed by one string in the catalog and found under another on disk is
    a bug report from a confused model later."""
    write_skill(tmp_path, "websearch", front_name="web-search")
    assert discover([str(tmp_path)]) == []


@pytest.mark.parametrize(
    ("name", "description"),
    [("Web-Search", "fine"), ("web search", "fine"), ("websearch", "")],
)
def test_an_invalid_frontmatter_field_is_refused(
    tmp_path: Path, name: str, description: str
) -> None:
    directory = tmp_path / name.lower().replace(" ", "-")
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n")
    assert discover([str(tmp_path)]) == []


def test_a_file_without_frontmatter_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# plain\n\nno frontmatter at all\n")
    assert discover([str(tmp_path)]) == []


def test_a_later_source_shadows_an_earlier_one(tmp_path: Path) -> None:
    """The layering `skills-progressive` adopts: base → user → project."""
    first, second = tmp_path / "user", tmp_path / "project"
    write_skill(first, "websearch", description="the user's version")
    write_skill(second, "websearch", description="the project's version")

    (skill,) = discover([str(first), str(second)])
    assert skill.description == "the project's version"


def test_an_oversized_skill_is_refused(tmp_path: Path) -> None:
    """Refused on `stat` alone — the size gate runs before any read, so the test
    needs a sparse file, not ten real megabytes."""
    directory = tmp_path / "huge"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: huge\ndescription: big\n---\n")
    os.truncate(directory / "SKILL.md", MAX_SKILL_BYTES + 1)
    assert discover([str(tmp_path)]) == []


# --------------------------------------------------------------- the row --


async def test_the_row_registers_the_catalog_and_feeds_the_runtime(
    mount: Any, tmp_path: Path
) -> None:
    """The two lists are two: a requirement spec to install, a module to import."""
    write_skill(tmp_path, "acme-websearch", package=True)
    write_skill(tmp_path, "note-taking")
    ctx = await mount(HOST_RUNTIME_ROW, row([tmp_path]))

    assert [skill.name for skill in ctx.skills.list()] == ["acme-websearch", "note-taking"]
    assert ctx.skills.get("acme-websearch").source == "rlm-skills-python"
    # The body stays on disk until something asks for it (G9).
    assert "Instructions here." in ctx.skills.body("acme-websearch")

    runtime = ctx.python_runtime
    assert runtime.skills == (str(tmp_path / "acme-websearch"),)
    assert runtime.skill_modules == ("acme_websearch",)
    assert "note-taking" not in " ".join(runtime.skill_modules), "a docs-only skill was installed"


async def test_the_catalog_reaches_the_prompt_without_the_bodies(
    mount: Any, tmp_path: Path
) -> None:
    write_skill(tmp_path, "acme-websearch", package=True)
    ctx = await mount(HOST_RUNTIME_ROW, row([tmp_path]))
    session = ctx.sessions.create("skills")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    prompt = render_prompt(assembly)

    assert "**acme-websearch** — search the web for a query" in prompt
    assert "await acme_websearch(...)" in prompt
    assert "Instructions here." not in prompt, "a skill body reached the prompt"


async def test_no_paths_means_no_row(mount: Any) -> None:
    ctx = await mount(HOST_RUNTIME_ROW, {"id": "rlm-skills-python", "name": "rlm-skills-python"})
    assert ctx.skills.list() == []
    assert ctx.python_runtime.skill_modules == ()


def test_a_catalog_of_nothing_renders_nothing() -> None:
    assert render_catalog([]) == ""


# ------------------------------------------------------------- the gate --


async def test_a_skills_run_is_callable_in_a_cell(
    mounted_runtime: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row's gate, end to end on a real kernel.

    The host interpreter can already import the package once it is on `sys.path`,
    so this drives discovery → boot frame → `wrap_skill_module` without building
    a venv (which would shell out to `uv` and reach the network). What it proves
    is the wiring the venv path shares: the *names* reach the guest and come back
    callable, and `run()` is reachable by calling the module itself — the §6.8
    convention prime-agent established.
    """
    write_skill(
        tmp_path,
        "greeter",
        package=True,
        body="async def run(name):\n    return f'hello {name}'\n\nVERSION = 3\n",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "greeter"))

    ctx, session, agent = await mounted_runtime(
        session_id="skills-gate", presentation=True, extra_rows=[row([tmp_path])]
    )
    assert ctx.python_runtime.skill_modules == ("greeter",)

    result = await run_ipython_cell(
        ctx,
        "(await greeter(name='world'), greeter.VERSION)",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == ["hello world", 3]


async def test_a_skill_that_does_not_import_says_so(mounted_runtime: Any, tmp_path: Path) -> None:
    """A model that finds a name undefined reads it as its own mistake and spends
    a turn working around it. A stub that explains itself costs one line."""
    write_skill(tmp_path, "broken", package=True)
    ctx, session, agent = await mounted_runtime(
        session_id="skills-broken", presentation=True, extra_rows=[row([tmp_path])]
    )
    # Nothing put `broken` on the child's path, so the import fails.
    result = await run_ipython_cell(ctx, "repr(broken)", agent=agent, session=session)

    assert result.is_error is False
    assert "unavailable skill broken" in result.value["value"]

    # Calling it raises *inside the cell*, so the traceback is the cell's result
    # rather than a tool failure — a skill that did not import is the
    # deployment's problem to fix, not a refusal the model should route around.
    called = await run_ipython_cell(ctx, "broken()", agent=agent, session=session, call_id="c2")
    rendered = "".join(block.text for block in called.content if hasattr(block, "text"))
    assert "is unavailable" in rendered
    assert "ask for it to be" in rendered
