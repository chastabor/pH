"""Python skills (P3-18): the half of a skill that only a cell can use.

The gate of the row is one sentence: *the `run()` of a skill is callable in a
cell*. It spans three layers that were built at different times:

* The format (now `ph.seams.skills`, tested there).
* The editable install of the venv (P3-07).
* The `wrap_skill_module` of the guest (P3-08), which had no producer until
  P3-18.

The last test drives the whole path against a real kernel.

What remains here is what is true only in Code Mode. P4-13 moved the reader, the
catalog and the `skill` tool into the seam, so this file no longer tests the
frontmatter rules again. It tests that a skill directory that is *also* an
installable package reaches the kernel, and that the catalog says so.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rlm_fixtures import HOST_RUNTIME_ROW, MountedRuntime
from runtime_helpers import run_python_cell

from ph.cordis import DEPLOYMENT, Context
from ph.keys import AGENTS, SESSIONS, SKILLS, SYSTEM_PROMPT
from ph.seams.skills import discover_skills
from ph.system_prompt import render_prompt
from ph.testing import FAKE_OPTIONS, MountProfile, not_none
from ph.testing import write_skill as write_skill_md
from ph_rlm.keys import PYTHON_RUNTIME
from ph_rlm.skills import give_to_runtime, python_half

pytestmark = pytest.mark.anyio


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "search the web for a query",
    package: bool = False,
    body: str = "",
) -> Path:
    """One skill directory. `package=True` makes it importable.

    The `SKILL.md` half comes from `ph.testing`, so the two packages that test
    against this format cannot disagree about what a valid file looks like. The
    package half is the whole subject of this row and stays here.
    """
    directory = write_skill_md(root, name, description=description)
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


def test_a_documentation_skill_has_nothing_to_install(tmp_path: Path) -> None:
    write_skill(tmp_path, "note-taking")

    (skill,) = discover_skills([str(tmp_path)])
    half = python_half(skill)

    assert half.module is None
    assert half.requirement is None, "nothing to install for a docs-only skill"
    assert half.hint == "", "a docs-only skill has no second way to be used"


def test_a_package_skill_carries_its_import_name(tmp_path: Path) -> None:
    """The name is read from `pyproject.toml`, not guessed. `acme-websearch`
    imports as `acme_websearch`, which is the normal case and not the unusual
    one."""
    write_skill(tmp_path, "acme-websearch", package=True)

    (skill,) = discover_skills([str(tmp_path)])
    half = python_half(skill)

    assert half.module == "acme_websearch"
    assert half.requirement == str(tmp_path / "acme-websearch")
    assert half.hint == "Callable in a cell as `await acme_websearch(...)`."


# --------------------------------------------------------------- the row --


async def test_the_row_registers_the_catalog_and_feeds_the_runtime(
    mount: MountProfile, tmp_path: Path
) -> None:
    """There are two lists: a requirement spec to install, and a module to import."""
    write_skill(tmp_path, "acme-websearch", package=True)
    write_skill(tmp_path, "note-taking")
    ctx = await mount(HOST_RUNTIME_ROW, row([tmp_path]))

    assert [skill.name for skill in ctx.require(SKILLS).list(DEPLOYMENT)] == [
        "acme-websearch",
        "note-taking",
    ]
    assert (
        not_none(ctx.require(SKILLS).get("acme-websearch", DEPLOYMENT)).source
        == "rlm-skills-python"
    )
    # The body stays on disk until something asks for it (G9).
    body = ctx.require(SKILLS).body("acme-websearch", DEPLOYMENT)
    assert body is not None and "Instructions here." in body

    runtime = ctx.require(PYTHON_RUNTIME)
    assert runtime.skills == (str(tmp_path / "acme-websearch"),)
    assert runtime.skill_modules == ("acme_websearch",)
    assert "note-taking" not in " ".join(runtime.skill_modules), "a docs-only skill was installed"


async def test_what_a_row_gave_the_runtime_goes_with_the_row() -> None:
    """A row takes back the skills that it added to the runtime when it unmounts.

    The row used to append its requirements and modules to tuples that the
    runtime keeps, and it took nothing back. A row that was mounted again added
    them a second time, and a row that was let go left its packages for every
    later kernel to install and import.

    Sabotage: drop the take-back from `give_to_runtime`, and the first row's skills
    stay after it goes.
    """
    root = Context()
    runtime = SimpleNamespace(skills=(), skill_modules=())
    first, second = root.scope("first"), root.scope("second")
    give_to_runtime(first, runtime, ("/skills/a",), ("a",))
    give_to_runtime(second, runtime, ("/skills/b",), ("b",))

    await first.dispose()

    assert runtime.skills == ("/skills/b",)
    assert runtime.skill_modules == ("b",)


async def test_the_catalog_reaches_the_prompt_without_the_bodies(
    mount: MountProfile, tmp_path: Path
) -> None:
    write_skill(tmp_path, "acme-websearch", package=True)
    ctx = await mount(HOST_RUNTIME_ROW, row([tmp_path]))
    session = ctx.require(SESSIONS).create("skills")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    prompt = render_prompt(assembly)

    # There is one catalog, the catalog of `skills-progressive`. The hint of this
    # row is attached to the skill, and the row has no second section of its own.
    assert "**acme-websearch** — search the web for a query" in prompt
    assert "await acme_websearch(...)" in prompt
    assert "Instructions here." not in prompt, "a skill body reached the prompt"


async def test_no_paths_means_no_row(mount: MountProfile) -> None:
    ctx = await mount(HOST_RUNTIME_ROW, {"id": "rlm-skills-python", "name": "rlm-skills-python"})
    assert ctx.require(SKILLS).list(DEPLOYMENT) == []
    assert ctx.require(PYTHON_RUNTIME).skill_modules == ()


# ------------------------------------------------------------- the gate --


async def test_a_skills_run_is_callable_in_a_cell(
    mounted_runtime: MountedRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate of the row, end to end on a real kernel.

    The host interpreter can already import the package when it is on
    `sys.path`. The test therefore drives discovery → boot frame →
    `wrap_skill_module` without building a venv, which would run `uv` and reach
    the network. The test proves the wiring that the venv path shares: the
    *names* reach the guest and come back callable, and `run()` is reachable by a
    call of the module itself. That is the §6.8 convention that prime-agent
    established.
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
    assert ctx.require(PYTHON_RUNTIME).skill_modules == ("greeter",)

    result = await run_python_cell(
        ctx,
        "(await greeter(name='world'), greeter.VERSION)",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == ["hello world", 3]


async def test_a_skill_that_does_not_import_says_so(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """A model that finds a name undefined reads it as its own mistake and spends
    a turn working around it. A stub that explains itself costs one line."""
    write_skill(tmp_path, "broken", package=True)
    ctx, session, agent = await mounted_runtime(
        session_id="skills-broken", presentation=True, extra_rows=[row([tmp_path])]
    )
    # Nothing put `broken` on the child's path, so the import fails.
    result = await run_python_cell(ctx, "repr(broken)", agent=agent, session=session)

    assert result.is_error is False
    assert "unavailable skill broken" in result.value["value"]

    # A call raises *inside the cell*, so the traceback is the result of the cell
    # and not a tool failure. A skill that did not import is a problem for the
    # deployment to fix. It is not a refusal that the model should work around.
    called = await run_python_cell(ctx, "broken()", agent=agent, session=session, call_id="c2")
    rendered = "".join(block.text for block in called.content if hasattr(block, "text"))
    assert "is unavailable" in rendered
    assert "ask for it to be" in rendered
