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

## Why `SubagentRun.scope` is handed back to the seam

The ceiling was a documented obligation on *providers* before this field, and the
second call site had already missed it: `rehydrate` builds a fresh scope for a
settled child and narrowed nothing, so a child that outlived its own restriction
came back holding the deployment-wide set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams._restriction import NameFilter
from ph.seams.skills import SkillRestriction
from ph.seams.subagents import Grant, SubagentRequest, SubagentSpawnError
from ph.system_prompt import render_prompt
from ph.testing import FAKE_OPTIONS, StubSubagentProvider, run_tool, simple_tool, skill, write_skill

pytestmark = pytest.mark.anyio


def _agent(ctx: Any, name: str = "parent", *, parent: Any = None) -> Any:
    return ctx.agents.create(ctx.sessions.create(name), FAKE_OPTIONS, parent=parent)


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


async def test_a_child_never_holds_more_than_its_narrowed_parent(mount: Any) -> None:
    """The ceiling, through the seam path, whichever way it is enforced.

    This asserted the *opposite structure* until P6-27 — `run.scope.isolation is
    not parent.ctx.isolation, "the stub nested the child"` — because a child was
    a sibling and inheritance had to be written out explicitly, so a stub that
    nested one would have passed a ceiling test production failed. Nesting made
    that guard assert the shape the row removed.

    What it was *for* survives and is what stays: a spawn that names nothing must
    not hand the child more than its narrowed parent holds. That is now true
    twice over — the isolation chain bounds it, and the admission grant bounds it
    — and this test does not care which, because a reader of a transcript does
    not either.
    """
    ctx, parent = await _granted(mount, "review", "deploy")
    ctx.skills.restrict(SkillRestriction(allow=frozenset({"review"})), scope=parent.ctx)

    run = await _spawn(ctx, parent)

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
    prompt = render_prompt(await ctx.system_prompt.assemble(run.scope))

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
    prompt = render_prompt(await ctx.system_prompt.assemble(run.scope))

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
            await ctx.system_prompt.assemble(run.scope)
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

    prompt = render_prompt(await ctx.system_prompt.assemble(run.scope))

    assert "**review**" in prompt, "the child can still reach the skill"
    assert "`skill` tool" not in prompt, "it was told to call a tool it does not have"


# --- P6-27: containment from the tree, not from a materialised grant --------
#
# `AgentRegistry.create` used to scope every agent under the *registry*, so a
# parent and its child were siblings and `parent.ctx.reaches(child.ctx)` was
# `False`. The relationship `SessionHeader.parent_session` records, and that B7
# is entirely about, had no representation in the tree that answers questions
# about it — so `ctx.subagents` rebuilt the ceiling by hand on every spawn.
# `parent=` nests the scope; these hold what that buys.


def _agents(ctx: Any) -> tuple[Any, Any]:
    """A parent and a child agent, created the way a spawn creates them.

    Through `_agent`, so these carry `FAKE_OPTIONS` like every other agent in the
    suite rather than a bare `AgentOptions()` — an unexplained second agent shape
    in one file is a difference the next reader has to rule out.
    """
    parent = _agent(ctx, "p627-parent")
    return parent, _agent(ctx, "p627-child", parent=parent)


async def test_a_childs_scope_nests_inside_its_parents(mount: Any) -> None:
    """The structural claim, and the two questions that follow from it.

    `reaches` is the visibility rule shared by event dispatch and every scoped
    registry, so a parent that reaches its child can *see* it — which is what
    lets a supervisor diagnose one. And the child's isolation chain now carries
    the parent, which is what the tool and skill registries walk.
    """
    ctx = await mount()
    parent, child = _agents(ctx)

    assert child.ctx.path.startswith(parent.ctx.path + "/")
    assert parent.ctx.reaches(child.ctx), "a parent cannot reach the child it supervises"
    assert not child.ctx.reaches(parent.ctx), "containment is one-way"

    chain = list(child.ctx.isolation_chain())
    assert parent.ctx in chain, "the parent is not on the chain the registries walk"


async def test_a_child_inherits_its_parents_narrowing_with_no_grant_applied(
    mount: Any,
) -> None:
    """The point of the row: the ceiling is the tree, not a list somebody wrote.

    No `Grant`, no `check_grant`, no `apply` — the parent is narrowed and the
    child is narrowed by consequence. Before nesting this returned the
    deployment-wide set, which is what `grant_for`'s docstring means by "applying
    nothing would hand the child of a narrowed parent the deployment-wide set".
    """
    ctx = await mount()
    ctx.tools.register(simple_tool("p627_tool"))
    parent, child = _agents(ctx)

    assert "p627_tool" in ctx.tools.view(child.ctx).visible
    ctx.tools.restrict(NameFilter(deny=("p627_tool",)), scope=parent.ctx)

    assert "p627_tool" not in ctx.tools.view(parent.ctx).visible
    assert "p627_tool" not in ctx.tools.view(child.ctx).visible, "the child outran its parent"


async def test_a_child_can_be_narrowed_below_its_parent(mount: Any) -> None:
    """The prerequisite this row needed, and why it was invisible before.

    `_build_view` filtered **global** names only, so a restriction could never
    take away a tool an *ancestor* had registered on its own scope. Latent while
    agents were siblings — no child had an ancestor layer — and load-bearing the
    moment they nest: a child would inherit its parent's scoped tools and be
    unable to be narrowed out of them, so a grant could widen a child but never
    bound it.
    """
    ctx = await mount()
    parent, child = _agents(ctx)
    ctx.tools.register(simple_tool("parent_scoped"), scope=parent.ctx)
    assert "parent_scoped" in ctx.tools.view(child.ctx).visible, "it should inherit first"

    ctx.tools.restrict(NameFilter(deny=("parent_scoped",)), scope=child.ctx)
    assert "parent_scoped" not in ctx.tools.view(child.ctx).visible
    assert "parent_scoped" in ctx.tools.view(parent.ctx).visible, "the parent kept its own"


async def test_a_scope_still_owns_its_own_registration(mount: Any) -> None:
    """The half of the old guard that was right, kept.

    "An agent's own registration cannot be masked out from under it" — a filter
    reaches everything *outside* the scope that wrote it and nothing inside, so
    dropping the `key is None` condition had to preserve this and not merely
    widen the filter.
    """
    ctx = await mount()
    _parent, child = _agents(ctx)
    ctx.tools.register(simple_tool("mine"), scope=child.ctx)
    ctx.tools.restrict(NameFilter(deny=("mine",)), scope=child.ctx)

    assert "mine" in ctx.tools.view(child.ctx).visible


async def test_disposing_the_parent_disposes_the_childs_scope(mount: Any) -> None:
    """Teardown follows the shape rather than a registered effect.

    `Context.dispose` unwinds `_children` before its own effects, so a child goes
    with its parent by construction. The subagent provider still registers
    `parent.ctx.effect(...)` — it writes the tombstone and updates the roster,
    which the scope teardown knows nothing about — but it is no longer what
    *stops* the child, so a provider that forgot it could not orphan one.
    """
    ctx = await mount()
    parent, child = _agents(ctx)
    assert child.ctx.active

    await ctx.agents.dispose(parent.id)
    assert not child.ctx.active, "the child's scope outlived its parent's"


async def test_a_childs_capability_is_fixed_at_admission(mount: Any) -> None:
    """The ruling P6-27 settled, and the reason `Grant` survives nesting.

    Nesting makes the *chain* bound a child — "no more than the parent holds" —
    and that alone would let a parent widen a running child by registering
    something new. The allow-list `Grant.apply` installs answers the narrower
    question instead: **no more than the parent held at admission.**

    Deliberate, not incidental. A child is spawned for a job, with a ceiling its
    prompt and its brief were written against; growing that mid-flight would make
    "what could this child do" unanswerable from the admission record, which is
    the one durable account of the delegation. A parent that needs more spawns a
    *new* child, whose admission says so.

    Held here because it is now the only thing `Grant` is for. With the chain
    doing the ceiling, a future reader could reasonably delete the allow-list as
    redundant — this is what would stop them.
    """
    ctx = await mount()
    ctx.tools.register(simple_tool("at_admission"))
    parent, child = _agents(ctx)

    # `names`, not a hand-rolled fold of `view().visible`: `held_by`'s docstring
    # is about exactly this — "computing them three times in three spellings is
    # how the three come to disagree about what 'holds' means".
    held = tuple(ctx.tools.names(scope=parent.ctx))
    Grant(skills=(), tools=held).apply(ctx, child.ctx)
    assert "at_admission" in ctx.tools.view(child.ctx).visible

    ctx.tools.register(simple_tool("after_admission"))
    assert "after_admission" in ctx.tools.view(parent.ctx).visible, "the parent did gain it"
    assert "after_admission" not in ctx.tools.view(child.ctx).visible, (
        "a running child widened when its parent did — the ceiling must be the one "
        "the admission recorded, not the one the parent happens to hold now"
    )


# --- P6-31: the ceiling is computed in the boundary the caller stated ---------


async def test_a_spawn_computes_its_ceiling_in_the_stated_boundary(mount: Any) -> None:
    """P6-31's first half, and the shape `fs_tools` had before P6-24.

    `held_by` derived the ceiling's boundary from `request.parent` — the field
    documented as the handle the *provider* needs, not as a policy boundary —
    while both tool bodies that spawn had `run.scope`, a non-optional `Context`,
    sitting two lines from `parent=run.agent`. When the two differ the ceiling is
    computed in one boundary and enforced in another.

    Driven with a narrowed *scope* whose agent is not narrowed: before the fix
    the ceiling came from the agent and kept the denied tool, so the child was
    granted something the stated boundary does not hold.
    """
    ctx = await mount()
    ctx.tools.register(simple_tool("wide_open"))
    parent = _agent(ctx, "p631-parent")
    stated = ctx.scope("the-delegating-boundary")
    ctx.tools.restrict(NameFilter(deny=("wide_open",)), scope=stated)

    _, held = ctx.subagents.held_by(SubagentRequest(prompt="go", parent=parent, scope=stated))
    assert "wide_open" not in held, (
        "the ceiling was computed from the agent, not the boundary the caller stated"
    )
    _, from_agent = ctx.subagents.held_by(SubagentRequest(prompt="go", parent=parent))
    assert "wide_open" in from_agent, "the un-narrowed agent is the control for that"


async def test_an_unreadable_parent_refuses_instead_of_granting_everything(
    mount: Any,
) -> None:
    """P6-31's second half: `None` was not "no ceiling", it was the widest one.

    `getattr(parent, "ctx", None)` returning `None` flowed into
    `SkillService.reach` and `ToolRuntime.names`, both of which resolve a missing
    scope to the **mount** — the unrestricted set. So an unreadable parent did
    not fail to narrow a child, it handed the child everything the deployment
    holds. And `_enforce` skipped its containment check on the same `None`, so
    the one assertion that would have noticed was off for the same reason.

    Fail-open on the path whose own docstring says *"a spawn that could widen
    would make delegation the privilege escalation I7 exists to prevent"*.
    Reproduced before the fix: a parent denied one tool produced a **7-tool
    ceiling where its own was 6**.
    """
    ctx = await mount()
    ctx.tools.register(simple_tool("wide_open"))
    parent = _agent(ctx, "p631-narrowed")
    ctx.tools.restrict(NameFilter(deny=("wide_open",)), scope=parent.ctx)

    class NoCtx:
        """A parent-shaped handle that never assigned `self.ctx`."""

        id = "broken"

    with pytest.raises(SubagentSpawnError, match="ceiling this child inherits is unknowable"):
        ctx.subagents.held_by(SubagentRequest(prompt="go", parent=NoCtx()))

    # The three that must keep working, because refusing is only worth it if it
    # refuses nothing else.
    _, narrowed = ctx.subagents.held_by(SubagentRequest(prompt="go", parent=parent))
    assert "wide_open" not in narrowed, "a readable parent still narrows"
    _, stated = ctx.subagents.held_by(
        SubagentRequest(prompt="go", parent=NoCtx(), scope=parent.ctx)
    )
    assert "wide_open" not in stated, "a stated boundary answers whatever the parent looks like"
    _, rootless = ctx.subagents.held_by(SubagentRequest(prompt="go", parent=None))
    assert "wide_open" in rootless, "a spawn with no parent is a root delegation"
