"""A child's model from its parent's list, and a profile its parent assigns (S7b).

The gate: a skill naming `classify` starts its child on that route; a key the list
does not hold is refused; and an assigned profile wider than the parent is refused,
naming the row. Around it: a spawn names a key itself, the admission records the key
and the route, and an assigned profile narrows what a child holds — its tools, its
model and its access — on the parent's own mount.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rlm_fixtures import PROVIDER_ROW

from ph.cordis import Context, Profile, ProfileDocument
from ph.json import JsonObject, as_obj, as_seq
from ph.keys import AGENTS, MOUNT, NAMED_PROFILES, SESSIONS, SKILLS, SUBAGENTS
from ph.seams.skills import Skill
from ph.seams.subagents import SubagentRequest, SubagentSpawnError
from ph.testing import FAKE_OPTIONS, MountProfile, write_skill
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio

MODELS: JsonObject = {
    "id": "models",
    "config": {
        "default": "main",
        "models": {
            "main": {"provider": "fake", "model": "fake-1"},
            "classify": {"provider": "fake", "model": "fake-9"},
        },
    },
}


class _Profiles:
    """`ctx.named_profiles` for a test: each name, the parent's own layers plus a patch."""

    def __init__(self, parent: Profile, **patches: list[JsonObject]) -> None:
        self.parent = parent
        self.patches = patches

    def compose(self, name: str) -> Profile:
        if name not in self.patches:
            raise ValueError(f'unknown profile "{name}"')
        documents = [*self.parent.documents, ProfileDocument(name, list(self.patches[name]))]
        return Profile.from_documents(documents, name=name)


async def _parent(mount: MountProfile, **profiles: list[JsonObject]) -> tuple[Context, Any]:
    ctx = await mount(dict(PROVIDER_ROW), dict(MODELS))
    ctx.provide(NAMED_PROFILES, _Profiles(ctx.require(MOUNT).profile, **profiles))
    session = ctx.require(SESSIONS).create("parent")
    return ctx, ctx.require(AGENTS).create(session, FAKE_OPTIONS)


async def _spawn(ctx: Context, parent: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="sort these", parent=parent, **kwargs)
    )


def _admitted(parent: Any) -> JsonObject:  # noqa: ANN401
    (event,) = parent.session.select("subagent/admitted")
    return as_obj(event.data)


async def test_a_skill_naming_a_listed_model_starts_its_child_on_that_route(
    mount: MountProfile,
) -> None:
    """The gate's first half. The skill directs the child, so it says what the child
    thinks with; the key is resolved by the parent's own list, and the admission
    records both the key and the route. Sabotage: drop `_skill_model` from
    `resolve_model`, and the child runs on its parent's `fake-1`."""
    ctx, parent = await _parent(mount)
    ctx.require(SKILLS).register(
        Skill(name="sort", description="sorts things into kinds", model="classify")
    )

    run = await _spawn(ctx, parent, skills=("sort",))

    assert (run.model_provider, run.model) == ("fake", "fake-9")
    admitted = _admitted(parent)
    assert admitted["modelKey"] == "classify" and admitted["model"] == "fake-9"
    child = ctx.require(AGENTS).get(run.session_id)
    assert child is not None and child.options.model_key == "classify"


async def test_a_spawn_names_a_listed_model_by_key(mount: MountProfile) -> None:
    ctx, parent = await _parent(mount)

    run = await _spawn(ctx, parent, model_key="classify")

    assert run.model == "fake-9"


async def test_a_key_the_list_does_not_hold_is_refused_and_admits_nothing(
    mount: MountProfile,
) -> None:
    """The models an agent can reach are the ones its profile says. Sabotage: fall
    back to the parent's route for an unknown key, and the child is admitted."""
    ctx, parent = await _parent(mount)

    with pytest.raises(SubagentSpawnError, match="it lists classify, main"):
        await _spawn(ctx, parent, model_key="summarize")

    assert not parent.session.select("subagent/admitted")


async def test_skills_that_name_different_models_are_refused(mount: MountProfile) -> None:
    ctx, parent = await _parent(mount)
    for name, key in (("sort", "classify"), ("write", "main")):
        ctx.require(SKILLS).register(Skill(name=name, description=f"{name}s", model=key))

    with pytest.raises(SubagentSpawnError, match="name different models"):
        await _spawn(ctx, parent, skills=("sort", "write"))


async def test_an_assigned_profile_wider_than_its_parent_is_refused_naming_the_row(
    mount: MountProfile,
) -> None:
    """The gate's second half. The parent runs no question tool; a profile that does
    cannot be given to its child, since there is no mount but the parent's to run it
    on. Sabotage: drop the row check in `narrowing`, and the child is admitted."""
    ctx, parent = await _parent(mount, asking=[{"id": "tool-ask-user", "disabled": False}])

    with pytest.raises(SubagentSpawnError, match='profile "asking": it runs tool-ask-user'):
        await _spawn(ctx, parent, profile="asking")


async def test_an_assigned_profile_narrows_the_child_s_tools_model_and_access(
    mount: MountProfile,
) -> None:
    """What it runs of what the parent holds: no `bash`, its own default model from
    the parent's list, and the read-only sandbox it starts in — recorded at the
    admission, so the reach is fixed there. Sabotage: skip the dropped rows in
    `_tools`, and the child keeps `bash`."""
    ctx, parent = await _parent(
        mount,
        sorter=[
            {"id": "tool-bash", "disabled": True},
            {"id": "models", "config": {**MODELS["config"], "default": "classify"}},  # type: ignore[dict-item]
        ],
    )

    run = await _spawn(ctx, parent, profile="sorter")

    admitted = _admitted(parent)
    tools = as_seq(admitted["tools"])
    assert "bash" not in tools and "read" in tools
    assert run.model == "fake-9" and admitted["profile"] == "sorter"
    assert run.requested_access == "read"


async def test_an_assigned_profile_s_model_must_be_one_its_parent_lists(
    mount: MountProfile,
) -> None:
    ctx, parent = await _parent(
        mount,
        elsewhere=[
            {
                "id": "models",
                "config": {
                    "default": "big",
                    "models": {"big": {"provider": "fake", "model": "fake-99"}},
                },
            }
        ],
    )

    with pytest.raises(SubagentSpawnError, match="runs on big, which its parent does not list"):
        await _spawn(ctx, parent, profile="elsewhere")


async def test_a_read_only_profile_cannot_give_its_child_write(mount: MountProfile) -> None:
    ctx, parent = await _parent(mount, reader=[])

    with pytest.raises(SubagentSpawnError, match="is read-only, so its child cannot write"):
        await _spawn(ctx, parent, profile="reader", access="write")


async def test_a_spawn_cannot_name_a_tool_its_assigned_profile_does_not_give(
    mount: MountProfile,
) -> None:
    """The profile is the child's ceiling as well as its defaults."""
    ctx, parent = await _parent(mount, nobash=[{"id": "tool-bash", "disabled": True}])

    with pytest.raises(SubagentSpawnError, match="does not give its child the tools bash"):
        await _spawn(ctx, parent, profile="nobash", tools=("bash",))


async def test_a_skill_given_to_a_child_is_recorded_as_read_into_its_prompt(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A spawn that names a skill puts its body in the child's prompt, which is a read
    at runtime like any other (S8): the parent's log says which text, by hash."""
    write_skill(tmp_path, "sort", body="Sort by kind.")
    ctx = await mount(
        dict(PROVIDER_ROW),
        dict(MODELS),
        {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}},
    )
    session = ctx.require(SESSIONS).create("parent")
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await _spawn(ctx, parent, skills=("sort",))

    (read,) = session.select("skill/read")
    assert (read.data["name"], read.data["via"]) == ("sort", "brief")
