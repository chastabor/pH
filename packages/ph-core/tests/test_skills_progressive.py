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
    ARGUMENT_HINT_MAX,
    MAX_ALLOWED_TOOLS,
    MAX_PARAMETERS,
    MAX_SKILL_BYTES,
    Skill,
    SkillRestriction,
    discover_skills,
    read_skill,
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


def _front(root: Path, name: str, extra: str) -> Path:
    """A `SKILL.md` with extra frontmatter keys, for the optional half (P7-17)."""
    return write_skill(root, name, description="does a thing", extra=extra) / "SKILL.md"


def test_the_optional_frontmatter_is_read_and_each_field_has_a_reader(tmp_path: Path) -> None:
    """P7-17's skills half: a `SKILL.md` may say more, and pH must use it.

    Three fields, three readers, which is the bar `Skill` already set when it
    refused `activeForm` for having none: `argument-hint` goes in the catalog so
    the model can tell how to *start* a skill rather than guessing; `version` and
    `allowed-tools` ride the `skill` tool's result, told at the one moment either
    is worth saying.
    """
    found = read_skill(
        _front(
            tmp_path,
            "release",
            "version: 2.1.0\nargument-hint: '[--dry-run]'\nallowed-tools:\n  - bash\n  - read",
        )
    )

    assert found is not None
    assert found.version == "2.1.0"
    assert found.argument_hint == "[--dry-run]"
    assert found.allowed_tools == ["bash", "read"]
    assert "Usage: `release [--dry-run]`" in render_catalog([found])


def test_a_skill_that_says_nothing_extra_is_unchanged(tmp_path: Path) -> None:
    """Optional means optional — every field defaults to saying nothing, and the
    catalog line for a plain skill is the line it always was."""
    found = read_skill(_front(tmp_path, "notes", "# nothing else"))

    assert found is not None
    assert (found.version, found.argument_hint, found.allowed_tools) == ("", "", [])
    assert render_catalog([found]).endswith("- **notes** — does a thing")


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ("allowed-tools: bash", "a string is not a list of names"),
        ("version: 'a version with spaces'", "an unusable version"),
        (f"argument-hint: '{'x' * (ARGUMENT_HINT_MAX + 1)}'", "a hint longer than the cap"),
        (
            "allowed-tools:\n" + "".join(f"  - t{n}\n" for n in range(MAX_ALLOWED_TOOLS + 1)),
            "more tools than the cap",
        ),
    ],
)
def test_a_malformed_optional_field_refuses_the_skill(tmp_path: Path, extra: str, why: str) -> None:
    """Refused, not dropped — the same rule the required fields follow.

    A skill installed with `allowed-tools` silently empty gives its author no way
    to tell "pH ignored my list" from "pH does not support lists", and the hint
    and tool caps are bounded for the reason everything model-adjacent is: they
    reach a prompt.
    """
    assert read_skill(_front(tmp_path, "risky", extra)) is None, why


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


async def test_reading_a_skill_resolves_the_tools_it_declared(mount: Any, tmp_path: Path) -> None:
    """`allowed-tools` is answered, not echoed (P7-17).

    An echoed list leaves the model to diff it against its own catalog. Resolved
    against the same scope the body was fetched with, it answers the question the
    declaration exists for: can I actually follow these instructions from here?

    Sabotage: return `allowed_tools` alone and a skill that needs a tool this
    deployment does not mount reads as perfectly followable, until the model is
    halfway through it.
    """
    write_skill(
        tmp_path,
        "release",
        description="cut a release",
        extra="allowed-tools:\n  - read\n  - deploy",
    )
    ctx = await mount(row(tmp_path))
    agent = ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)

    result = await run_tool(ctx, "skill", {"name": "release"}, agent=agent)

    assert result.value["allowed_tools"] == ["read", "deploy"]
    assert result.value["missing_tools"] == ["deploy"], "`read` is mounted; `deploy` is not"


# ------------------------------------------------------------- parameters --

RELEASE = """parameters:
  version-type:
    type: string
    required: true
    enum: [major, minor, patch]
    hint: which part of the version to bump
  tag-prefix:
    type: string
    default: v"""


async def _release(mount: Any, root: Path, body: str = "Bump {{parameters.version-type}}.") -> Any:
    write_skill(root, "release", description="cut a release", extra=RELEASE, body=body)
    ctx = await mount(row(root))
    return ctx, ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)


async def test_a_declared_input_is_filled_into_the_instructions(mount: Any, tmp_path: Path) -> None:
    """**P7-18's first half.** A skill takes inputs, and the body says so.

    Instructions with the value already in them are what makes a parameter worth
    declaring: the alternative is generic prose plus a separate sentence about
    what the model should mentally substitute, which is a step it can get wrong
    silently. A declared default fills in too, so an author writing
    `{{parameters.tag-prefix}}` never has to handle the caller who passed none.
    """
    ctx, agent = await _release(
        mount, tmp_path, body="Bump {{parameters.version-type}}, tag {{parameters.tag-prefix}}1.0."
    )

    result = await run_tool(
        ctx, "skill", {"name": "release", "arguments": {"version-type": "minor"}}, agent=agent
    )

    assert "Bump minor, tag v1.0." in text_of(result.content)
    assert result.value["parameters"] == ["tag-prefix", "version-type"]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, "version-type"),
        ({"version-type": "mayor"}, "must be one of"),
        ({"version-type": "minor", "dry-run": True}, "dry-run"),
        ({"version-type": 3}, "string"),
    ],
    ids=["missing-required", "outside-the-enum", "unknown-name", "wrong-type"],
)
async def test_arguments_that_do_not_satisfy_the_declaration_are_refused(
    mount: Any, tmp_path: Path, arguments: dict[str, Any], expected: str
) -> None:
    """Refused before the model starts following the instructions.

    The unknown-name case is why declaring is worth anything: an argument the
    skill never heard of, silently ignored, is how a `--dry-run` misspelled
    `dry-run` runs for real. `additionalProperties: False` is what makes that a
    refusal rather than a shrug.

    The refusal names what was wanted, so the second call can be right — which is
    the trade for keeping parameters out of the catalog, where they would be paid
    for on every request instead of only by the call that got it wrong.
    """
    ctx, agent = await _release(mount, tmp_path)

    result = await run_tool(ctx, "skill", {"name": "release", "arguments": arguments}, agent=agent)

    assert result.is_error
    assert expected in text_of(result.content)


async def test_a_body_naming_an_undeclared_input_is_refused(mount: Any, tmp_path: Path) -> None:
    """The author's typo, surfaced where both halves are finally in hand.

    G9 keeps the body on disk until something asks for it, so there is no earlier
    moment that has the frontmatter *and* the prose — a load-time check would
    have to read the 10 MiB this row exists to defer.

    Sabotage: substitute silently and `{{parameters.verison}}` reaches the model
    verbatim, as an instruction to fill in something it has no value for.
    """
    ctx, agent = await _release(mount, tmp_path, body="Bump {{parameters.verison}}.")

    result = await run_tool(
        ctx, "skill", {"name": "release", "arguments": {"version-type": "minor"}}, agent=agent
    )

    assert result.is_error
    assert "undeclared parameter(s) verison" in text_of(result.content)


async def test_a_skill_that_declares_nothing_is_untouched(mount: Any, tmp_path: Path) -> None:
    """Optional, and the body of a plain skill is returned byte for byte.

    A `SKILL.md` is prose *and* code samples, and code is full of braces — a
    general template pass would make a skill unable to contain an example of
    itself."""
    write_skill(tmp_path, "notes", body="Use `{{mustache}}` and {braces} freely.")
    ctx = await mount(row(tmp_path))
    agent = ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)

    result = await run_tool(ctx, "skill", {"name": "notes"}, agent=agent)

    assert "Use `{{mustache}}` and {braces} freely." in text_of(result.content)
    assert result.value["parameters"] == []


@pytest.mark.parametrize(
    ("block", "why"),
    [
        ("parameters: two", "not a mapping"),
        ("parameters:\n  ok:\n    type: list", "an unsupported type"),
        ("parameters:\n  ok: 3", "an entry that is not a mapping"),
        ("parameters:\n  ok:\n    type: string\n    enum: []", "an empty enum"),
        (
            "parameters:\n"
            + "".join(f"  p{n}:\n    type: string\n" for n in range(MAX_PARAMETERS + 1)),
            "more parameters than the cap",
        ),
    ],
    ids=["not-a-mapping", "bad-type", "entry-not-a-mapping", "empty-enum", "over-the-cap"],
)
async def test_a_malformed_declaration_refuses_the_skill(
    tmp_path: Path, block: str, why: str
) -> None:
    """Refused rather than installed with the block dropped — the rule every
    other field in this frontmatter follows."""
    write_skill(tmp_path, "risky", extra=block)

    assert read_skill(tmp_path / "risky" / "SKILL.md") is None, why


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
