"""P4-13 — `skills-progressive`: the catalog in the prompt, the body on demand (G9).

The gate is one sentence — *skill body absent until requested* — and it is worth
saying why that is a design rather than an optimisation. A deployment with
twenty skills has twenty bodies of up to 10 MiB each; putting them in the prompt
would spend the context window on instructions for the nineteen the model is not
about to use. So the prompt gets names and descriptions, and a `skill` call gets
one body.

The validation tests below are the seam's, not this row's: a skill that
half-registers is a name the catalog offers and nothing can serve. They moved
here from `ph-rlm` when the reader did, because the format belongs to whoever
defines a skill and both rows now read it through this one.

## Why skill restrictions are keyed by scope

A flat list scanned per read made **every agent pay for every other agent's
narrowing**: a fan-out of sixteen children put sixteen filters in front of the
parent's own catalog, none of which could change what the parent sees, and the
parent's prompt assembly measured **3.9x slower** for it. Keyed, a read walks
`isolation_chain()` — the filters that can possibly apply and no others.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT
from ph.llm.types import text_of
from ph.seams.skills import (
    MAX_SKILL_BYTES,
    Skill,
    SkillRestriction,
    discover_skills,
    render_catalog,
)
from ph.system_prompt import render_prompt
from ph.testing import FAKE_OPTIONS, run_tool, skill, write_skill

pytestmark = pytest.mark.anyio


def row(*paths: Path) -> dict[str, Any]:
    return {
        "id": "skills-progressive",
        "config": {"paths": [str(path) for path in paths]},
    }


# ------------------------------------------------------------------ the format --


def test_a_skill_is_discovered_from_its_frontmatter(tmp_path: Path) -> None:
    write_skill(tmp_path, "note-taking")

    (skill,) = discover_skills([str(tmp_path)])

    assert (skill.name, skill.description) == ("note-taking", "search the web for a query")
    assert skill.path == str(tmp_path / "note-taking" / "SKILL.md")


def test_a_name_that_disagrees_with_its_directory_is_refused(tmp_path: Path) -> None:
    """Addressed by one string in the catalog and found under another on disk is
    a bug report from a confused model later."""
    write_skill(tmp_path, "websearch", front_name="web-search")

    assert discover_skills([str(tmp_path)]) == []


@pytest.mark.parametrize(
    ("name", "description"),
    [("Web-Search", "fine"), ("web search", "fine"), ("websearch", "")],
)
def test_an_invalid_frontmatter_field_is_refused(
    tmp_path: Path, name: str, description: str
) -> None:
    write_skill(tmp_path, name.lower().replace(" ", "-"), front_name=name, description=description)

    assert discover_skills([str(tmp_path)]) == []


def test_a_file_without_frontmatter_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# plain\n\nno frontmatter at all\n", encoding="utf-8")

    assert discover_skills([str(tmp_path)]) == []


def test_a_later_source_shadows_an_earlier_one(tmp_path: Path) -> None:
    """The list is a precedence order — base, user, project — so a user's own
    `websearch` wins over the one their distribution ships. A collision that
    refused instead would make overriding a skill impossible."""
    first, second = tmp_path / "user", tmp_path / "project"
    write_skill(first, "websearch", description="the user's version")
    write_skill(second, "websearch", description="the project's version")

    (skill,) = discover_skills([str(first), str(second)])

    assert skill.description == "the project's version"


def test_an_oversized_skill_is_refused(tmp_path: Path) -> None:
    """G9's third limit. Refused on `stat` alone — the size gate runs before any
    read, so the test needs a sparse file, not ten real megabytes."""
    directory = write_skill(tmp_path, "huge", description="big")
    os.truncate(directory / "SKILL.md", MAX_SKILL_BYTES + 1)

    assert discover_skills([str(tmp_path)]) == []


def test_a_catalog_of_nothing_renders_nothing() -> None:
    assert render_catalog([]) == ""


# --------------------------------------------------------------------- the row --


async def test_the_catalog_reaches_the_prompt_without_the_bodies(
    mount: Any, tmp_path: Path
) -> None:
    """The gate. Names and descriptions are cheap and always useful; a body is
    neither, so it waits to be asked for."""
    write_skill(tmp_path, "note-taking", body="Step one: open a file. Step two: write in it.")
    ctx = await mount(row(tmp_path))

    prompt = render_prompt(await ctx.system_prompt.assemble(DEPLOYMENT))

    assert "**note-taking** — search the web for a query" in prompt
    assert "Step one" not in prompt, "a skill body reached the prompt"


async def test_the_tool_hands_over_the_body_when_asked(mount: Any, tmp_path: Path) -> None:
    write_skill(tmp_path, "note-taking", body="Step one: open a file.")
    ctx = await mount(row(tmp_path))
    agent = ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)

    result = await run_tool(ctx, "skill", {"name": "note-taking"}, agent=agent)

    assert "Step one: open a file." in text_of(result.content)


async def test_an_unknown_skill_names_what_is_installed(mount: Any, tmp_path: Path) -> None:
    """A typo is the likely cause and the model can fix it from the list — but
    only if the refusal carries the list rather than just saying no."""
    write_skill(tmp_path, "note-taking")
    ctx = await mount(row(tmp_path))
    agent = ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)

    result = await run_tool(ctx, "skill", {"name": "note-takin"}, agent=agent)

    assert result.is_error
    assert "note-taking" in text_of(result.content)


async def test_the_catalog_covers_skills_another_row_registered(mount: Any, tmp_path: Path) -> None:
    """The reason the catalog is a provider rather than a fixed string: a second
    row that installs skills — `rlm-skills-python` is the one that exists —
    appears in *this* catalog instead of adding a second one, and neither row
    has to be mounted before the other."""
    ctx = await mount(row())

    ctx.skills.register(
        Skill(name="deploy", description="ship the thing", hint="Callable as `await deploy(...)`.")
    )

    prompt = render_prompt(await ctx.system_prompt.assemble(DEPLOYMENT))

    assert "**deploy** — ship the thing Callable as `await deploy(...)`." in prompt


async def test_no_skills_means_no_tool(mount: Any) -> None:
    """The same rule `subagent-task` follows: a tool in every prompt that can
    only fail teaches the model a capability the deployment does not have. The
    check is at `profile/mounted`, so a skill installed by *any* row counts."""
    ctx = await mount(row())

    assert ctx.tools.get("skill", scope=DEPLOYMENT) is None


async def test_no_paths_scans_nothing(mount: Any) -> None:
    """`$PH_HOME/skills` is deliberately not a default: scanning a well-known
    directory at every start would make "install a skill" mean "drop a file
    there", and a skill is something a distribution or a user installs on
    purpose (I7)."""
    ctx = await mount()

    assert ctx.skills.list(DEPLOYMENT) == []


async def test_the_deployment_wide_answer_has_to_be_asked_for(mount: Any) -> None:
    """P6-32's behavioural half, on the seam P6-31 named and left.

    `reach` resolved `scope or self.ctx`, and `self.ctx` is the mount — the
    *unrestricted* set. So an unstated boundary was not "no skills", it was all
    of them, and P6-31 could not simply make it refuse: the prompt catalog and
    the "did this deployment install anything" check at mount both legitimately
    mean the deployment, and with one spelling they had no way to say so.

    Both answers still exist. What changed is that each is now spelled, and the
    one that widens is the one you have to type.
    """
    ctx = await mount()
    ctx.skills.register(skill("wide"))
    ctx.skills.register(skill("narrow"))
    agent = ctx.agents.create(ctx.sessions.create("p632"), FAKE_OPTIONS)
    ctx.skills.restrict(SkillRestriction(deny=("wide",)), scope=agent.ctx)

    assert [one.name for one in ctx.skills.list(DEPLOYMENT)] == ["narrow", "wide"]
    assert [one.name for one in ctx.skills.list(agent.ctx)] == ["narrow"], "the narrowing was lost"

    # And saying nothing is no longer a way to get the wide answer by accident.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        ctx.skills.list()
