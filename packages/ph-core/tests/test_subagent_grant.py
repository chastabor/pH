"""P4-13b — what a child is given, and the ceiling it cannot pass.

One rule carries this file: **a child holds a subset of its parent, always.**
There is no elevation, no preset that grants beyond the selector, no flag. To
give a child a skill or a tool, grant it to the parent first. Applied down the
tree that means the root agent's grant bounds every descendant and capability
narrows monotonically, which is the property that makes a fan-out auditable
from one place instead of N.

The mechanism is chosen to make the rule structural rather than remembered.
Narrowing is by **restriction**, which can only intersect; nothing in the spawn
path registers on a child's scope, because `ctx.tools` treats a scope's own
registration as unmaskable and that would be a way to hand a child something its
parent cannot see. So the widening case is not guarded against — it is absent.

The second claim is smaller and about usefulness rather than safety: a skill
named at spawn is **direction**. Its body goes in the child's prompt, because a
child created to follow a procedure should not have to spend a turn fetching it,
and might not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams.skills import SkillRestriction
from ph.seams.subagents import SubagentRequest, SubagentSpawnError
from ph.system_prompt import render_prompt
from ph.system_prompt.assembly import AssembleContext
from ph.testing import FAKE_OPTIONS, StubSubagentProvider, run_tool, skill, write_skill

pytestmark = pytest.mark.anyio


def _agent(ctx: Any) -> Any:
    return ctx.agents.create(ctx.sessions.create("parent"), FAKE_OPTIONS)


async def _spawn(ctx: Any, parent: Any, **request: Any) -> Any:
    """One delegation through the seam, which is where the ceiling lives.

    Through `start` rather than by applying a grant by hand: the refusal, the
    materialization and the application are three steps on one path, and a test
    that called the last of them directly would pass while the path was broken.
    """
    if ctx.subagents.provider_names() == []:
        ctx.subagents.register_provider("stub", StubSubagentProvider(root=ctx))
    return await ctx.subagents.start("stub", SubagentRequest(prompt="go", parent=parent, **request))


async def _granted(mount: Any, *names: str, rows: Any = None) -> tuple[Any, Any]:
    """`(ctx, parent)` with these skills installed and a provider mounted."""
    ctx = await mount(*(rows or []))
    for name in names:
        ctx.skills.register(skill(name))
    ctx.subagents.register_provider("stub", StubSubagentProvider(root=ctx))
    return ctx, _agent(ctx)


# ------------------------------------------------------------------- the seam --


async def test_a_child_may_be_narrowed_to_a_subset(mount: Any) -> None:
    ctx, parent = await _granted(mount, "review", "deploy")

    run = await _spawn(ctx, parent, skills=("review",))

    assert [one.name for one in ctx.skills.list(run.scope)] == ["review"]
    # And the parent is untouched, which is what makes this narrowing rather
    # than a deployment-wide policy change.
    assert [one.name for one in ctx.skills.list(parent.ctx)] == ["deploy", "review"]


async def test_a_spawn_cannot_name_a_skill_the_parent_does_not_hold(mount: Any) -> None:
    """The whole security content of the row. A spawn that could widen would
    make delegation the privilege escalation I7 exists to prevent."""
    ctx, parent = await _granted(mount, "review")

    with pytest.raises(SubagentSpawnError) as refused:
        await _spawn(ctx, parent, skills=("review", "deploy"))

    assert "deploy" in str(refused.value)
    # The refusal says what to do about it, because the answer is a real
    # workflow and not a dead end.
    assert "Grant it to the parent first" in str(refused.value)


async def test_a_narrowed_parent_cannot_re_grant_what_it_lost(mount: Any) -> None:
    """Transitivity, which is what makes the root's grant bound the whole tree.

    A child narrowed to `review` is a parent in turn, and the skill it no longer
    holds is one it can no longer hand on — otherwise the ceiling would only
    bind the first generation.
    """
    ctx, parent = await _granted(mount, "review", "deploy")
    ctx.skills.restrict(SkillRestriction(allow=frozenset({"review"})), scope=parent.ctx)

    with pytest.raises(SubagentSpawnError):
        await _spawn(ctx, parent, skills=("deploy",))


async def test_a_sibling_scope_does_not_inherit_the_parents_narrowing(mount: Any) -> None:
    """The defect this row nearly shipped, pinned as its own test.

    `AgentRegistry.create` scopes every agent under the *registry*, so a child
    agent's scope is the parent's **sibling**, not its descendant — a filter
    owned by the parent reaches the parent alone. A spawn that named nothing and
    therefore applied nothing would have handed the child the deployment-wide
    set, which is wider than its narrowed parent: the one thing the ceiling
    forbids. Inheritance is written out explicitly for this reason.
    """
    ctx, parent = await _granted(mount, "review", "deploy")
    ctx.skills.restrict(SkillRestriction(allow=frozenset({"review"})), scope=parent.ctx)

    run = await _spawn(ctx, parent)

    assert run.scope.isolation is not parent.ctx.isolation, "the stub nested the child"
    assert [one.name for one in ctx.skills.list(run.scope)] == ["review"]


async def test_an_empty_selection_is_a_real_answer(mount: Any) -> None:
    """`()` says "no skills", which a caller may legitimately mean — and which
    `None` cannot express."""
    ctx, parent = await _granted(mount, "review")

    run = await _spawn(ctx, parent, skills=())

    assert ctx.skills.list(run.scope) == []


async def test_restrictions_intersect_rather_than_replace(mount: Any) -> None:
    """Two narrowings compose to the narrower of them. If the second replaced
    the first, a nested spawn would be a way back out."""
    ctx, parent = await _granted(mount, "one", "two", "three")
    ctx.skills.restrict(SkillRestriction(allow=frozenset({"one", "two"})), scope=parent.ctx)
    inner = parent.ctx.scope("inner")

    ctx.skills.restrict(SkillRestriction(deny=frozenset({"one"})), scope=inner)

    assert [one.name for one in ctx.skills.list(inner)] == ["two"]


# ------------------------------------------------------------------ the tools --


async def test_a_child_may_be_narrowed_to_a_subset_of_tools(mount: Any) -> None:
    ctx, parent = await _granted(mount)
    before = set(ctx.tools.view(parent.ctx).visible)
    assert {"read", "write"} <= before

    run = await _spawn(ctx, parent, tools=("read",))

    assert "read" in ctx.tools.view(run.scope).visible
    assert "write" not in ctx.tools.view(run.scope).visible
    assert set(ctx.tools.view(parent.ctx).visible) == before


async def test_a_spawn_cannot_name_a_tool_the_parent_does_not_hold(mount: Any) -> None:
    ctx, parent = await _granted(mount)

    with pytest.raises(SubagentSpawnError) as refused:
        await _spawn(ctx, parent, tools=("read", "nonesuch"))

    assert "nonesuch" in str(refused.value)


async def test_a_provider_with_no_child_scope_cannot_be_narrowed(mount: Any) -> None:
    """Fail-closed, and narrowly. A provider that hands back no scope cannot be
    bounded, so it is refused the moment a spawn means to narrow — and left
    alone in a deployment where nothing is restricted, which is every provider
    that existed before this row."""
    # Two installed, one named — otherwise "narrowed to the only skill there is"
    # is not a narrowing and the check correctly says nothing.
    ctx, parent = await _granted(mount, "review", "deploy")
    ctx.subagents.register_provider("blind", StubSubagentProvider())

    unnarrowed = await ctx.subagents.start("blind", SubagentRequest(prompt="go", parent=parent))
    assert unnarrowed.scope is None

    with pytest.raises(SubagentSpawnError) as refused:
        await ctx.subagents.start(
            "blind", SubagentRequest(prompt="go", parent=parent, skills=("review",))
        )

    assert "cannot be bounded" in str(refused.value)


# ----------------------------------------------------------------- direction --


async def test_a_named_skill_is_in_the_childs_prompt(mount: Any, tmp_path: Path) -> None:
    """G9 inverted for the one case where the question it defers is already
    answered: a child spawned *for* this skill will need it, certainly."""
    write_skill(tmp_path, "review", body="Read the diff. Say what is wrong. Do not fix it.")
    ctx, parent = await _granted(
        mount, rows=[{"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}}]
    )

    run = await _spawn(ctx, parent, skills=("review",))
    prompt = render_prompt(await ctx.system_prompt.assemble(AssembleContext(scope=run.scope)))

    assert "Read the diff. Say what is wrong." in prompt
    assert "not background reading" in prompt


async def test_an_unnamed_skill_stays_out_of_the_prompt(mount: Any, tmp_path: Path) -> None:
    """The other half of G9: a skill the child merely *may* use is a catalog
    line, not a body. Otherwise narrowing a child to five skills would put five
    bodies in every one of its requests."""
    write_skill(tmp_path, "review", body="Read the diff carefully.")
    write_skill(tmp_path, "deploy", body="Push the button twice.")
    ctx, parent = await _granted(
        mount, rows=[{"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}}]
    )

    run = await _spawn(ctx, parent, skills=("review",))
    prompt = render_prompt(await ctx.system_prompt.assemble(AssembleContext(scope=run.scope)))

    assert "Read the diff carefully." in prompt
    assert "Push the button twice." not in prompt
    assert "**deploy**" not in prompt, "a skill it cannot reach was still advertised"


async def test_the_brief_is_read_once_not_per_assembly(mount: Any, tmp_path: Path) -> None:
    """A `PromptSection` is the *cached prefix*; one that hits the filesystem on
    every model step is neither static nor free — and a skill body may be 10 MiB."""
    write_skill(tmp_path, "review", body="Read the diff carefully.")
    ctx, parent = await _granted(
        mount, rows=[{"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}}]
    )
    run = await _spawn(ctx, parent, skills=("review",))

    opens = 0
    original = Path.open

    def counted(self: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        if self.name == "SKILL.md":
            opens += 1
        return original(self, *args, **kwargs)

    Path.open = counted  # type: ignore[method-assign]
    try:
        for _ in range(3):
            await ctx.system_prompt.assemble(AssembleContext(scope=run.scope))
    finally:
        Path.open = original  # type: ignore[method-assign]

    assert opens == 0, "the skill body was re-read to render a section already rendered"


# ------------------------------------------------------------------- presets --


PRESET_ROW = {
    "id": "subagent-presets",
    "config": {"presets": {"reviewer": {"skills": ["review"], "tools": ["read"]}}},
}


async def test_a_preset_fills_in_what_the_caller_did_not_name(mount: Any) -> None:
    """A deployment writes "what a reviewer is" once. The direction still comes
    from the skill, which is why a preset has no prompt of its own."""
    ctx, parent = await _granted(mount, "review", rows=[PRESET_ROW])

    resolved = ctx.subagents.resolve_preset(
        SubagentRequest(prompt="go", parent=parent, preset="reviewer")
    )

    assert resolved.skills == ("review",)
    assert resolved.tools == ("read",)


async def test_a_preset_cannot_grant_past_the_parent(mount: Any) -> None:
    """The reason a preset is a menu and not a grant: one that widened whatever
    selected it would put the escalation one indirection away, and under the
    model's control rather than a human's."""
    # The deployment configured `review`, but nobody installed it.
    ctx, parent = await _granted(mount, rows=[PRESET_ROW])

    with pytest.raises(SubagentSpawnError) as refused:
        ctx.subagents.check_grant(
            ctx.subagents.resolve_preset(
                SubagentRequest(prompt="go", parent=parent, preset="reviewer")
            )
        )

    assert "review" in str(refused.value)


async def test_an_unknown_preset_is_refused_rather_than_ignored(mount: Any) -> None:
    """A spawn that asked for a `reviewer` and silently got a generic child is
    the failure `_resolve_model` refuses one field over, for the same reason."""
    ctx, parent = await _granted(mount, rows=[PRESET_ROW])

    with pytest.raises(SubagentSpawnError) as refused:
        ctx.subagents.resolve_preset(SubagentRequest(prompt="go", parent=parent, preset="nonesuch"))

    assert "reviewer" in str(refused.value), "the refusal did not say what is on offer"


async def test_an_explicit_selection_still_narrows_a_preset(mount: Any) -> None:
    """Defaults, not a ceiling — the ceiling is the parent."""
    ctx, parent = await _granted(mount, rows=[PRESET_ROW])

    resolved = ctx.subagents.resolve_preset(
        SubagentRequest(prompt="go", parent=parent, preset="reviewer", skills=())
    )

    assert resolved.skills == ()


# ------------------------------------------------------------- through a tool --


async def test_the_task_tool_carries_the_selection(mount: Any) -> None:
    """End to end through the model-facing surface, since that is where a
    selector that never reaches the request would look like it worked."""
    provider = StubSubagentProvider()
    ctx = await mount({"id": "subagent-task"})
    ctx.skills.register(skill("review"))
    ctx.subagents.register_provider("stub", provider)
    await ctx.serial("profile/mounted")
    parent = _agent(ctx)

    await run_tool(
        ctx,
        "task",
        {"prompt": "review this", "skills": ["review"], "tools": ["read"]},
        agent=parent,
    )

    assert provider.last().skills == ("review",)
    assert provider.last().tools == ("read",)


async def test_the_task_tool_refuses_to_widen(mount: Any) -> None:
    ctx = await mount({"id": "subagent-task"})
    ctx.subagents.register_provider("stub", StubSubagentProvider())
    await ctx.serial("profile/mounted")
    parent = _agent(ctx)

    result = await run_tool(ctx, "task", {"prompt": "go", "skills": ["nonesuch"]}, agent=parent)

    assert result.is_error


async def test_a_child_without_the_tool_is_not_told_to_use_it(mount: Any, tmp_path: Path) -> None:
    """The catalog names the `skill` tool, so it must ask whether *this* agent
    has it. Telling a model to call something absent from its schema reads as
    the model's mistake rather than the profile's."""
    write_skill(tmp_path, "review", body="Read the diff.")
    ctx, parent = await _granted(
        mount, rows=[{"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}}]
    )
    run = await _spawn(ctx, parent, tools=("read",))

    prompt = render_prompt(await ctx.system_prompt.assemble(AssembleContext(scope=run.scope)))

    assert "**review**" in prompt, "the child can still reach the skill"
    assert "`skill` tool" not in prompt, "it was told to call a tool it does not have"
