"""A child's model from its parent's list, and a profile its parent assigns (S7b).

The gate: a skill naming `classify` starts its child on that route; a key the list
does not hold is refused; and an assigned profile wider than the parent is refused,
naming the row. Around it: a spawn names a key itself, the admission records the key
and the route, and an assigned profile narrows what a child holds — its tools, its
model, its access, its skills and its writable directories — on the parent's own
mount, each row by the narrower its plugin declares.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field
from rlm_fixtures import PROVIDER_ROW

from ph.cordis import ChildLimit, ChildReach, Context, Profile, ProfileDocument, plugin
from ph.json import JsonObject, as_obj, as_seq, thaw_json
from ph.keys import AGENTS, MOUNT, NAMED_PROFILES, SANDBOX, SESSIONS, SKILLS, SUBAGENTS
from ph.paths import canonical
from ph.seams.sandbox import SandboxPolicy, allowed_paths_of
from ph.seams.skills import Skill
from ph.seams.subagents import SubagentRequest, SubagentSpawnError
from ph.testing import FAKE_OPTIONS, MountProfile, not_none, write_skill
from ph.wire import WireModel
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


async def _parent(
    mount: MountProfile, *rows: JsonObject, **profiles: list[JsonObject]
) -> tuple[Context, Any]:
    ctx = await mount(dict(PROVIDER_ROW), dict(MODELS), *rows)
    ctx.provide(NAMED_PROFILES, _Profiles(ctx.require(MOUNT).profile, **profiles))
    session = ctx.require(SESSIONS).create("parent")
    return ctx, ctx.require(AGENTS).create(session, FAKE_OPTIONS)


async def _spawn(ctx: Context, parent: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="sort these", parent=parent, **kwargs)
    )


def _admitted(ctx: Context, parent: Any) -> JsonObject:  # noqa: ANN401
    """The one child's admission, from the child's own log — where a restart reads the
    narrowing back from, so where it has to be recorded."""
    (child,) = ctx.require(SUBAGENTS).children(parent.session.id).values()
    return as_obj(thaw_json(not_none(child.admission).to_wire()))


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
    admitted = _admitted(ctx, parent)
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

    assert ctx.require(SUBAGENTS).children(parent.session.id) == {}


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

    admitted = _admitted(ctx, parent)
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


async def test_a_skill_goes_with_the_row_that_gave_it(mount: MountProfile) -> None:
    """(b) A row the assigned profile does not run takes its skills with it, as it
    takes its tools — found by the row that installed each (`SkillService.
    registrants`), whatever row that is. A skill no row installed stays. Sabotage:
    skip the registrants in `_skills`, and the dropped row's skill stays."""
    ctx, parent = await _parent(mount, nobash=[{"id": "tool-bash", "disabled": True}])
    service = ctx.require(SKILLS)
    beside = not_none(ctx.require(MOUNT).forks["tool-bash"].ctx)
    service.register(Skill(name="shell-tips", description="running things"), scope=beside)
    service.register(Skill(name="review", description="reviewing things"))

    await _spawn(ctx, parent, profile="nobash")

    skills = as_seq(_admitted(ctx, parent)["skills"])
    assert "shell-tips" not in skills and "review" in skills


def _dirs(root: Path, *names: str) -> list[str]:
    """Directories that exist, spelled as the sandbox binds them."""
    made = []
    for name in names:
        (root / name).mkdir()
        made.append(str(canonical(root / name)))
    return made


def _writable(*paths: str) -> tuple[JsonObject, JsonObject]:
    """A parent whose confined commands may write `paths` beside its workspace."""
    return (
        {"id": "sandbox", "config": {"defaultMode": "workspace-write"}},
        {"id": "sandbox-allow", "config": {"paths": list(paths)}},
    )


async def test_a_child_given_fewer_writable_directories_writes_only_those(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Item 3's paths: an assigned `sandbox-allow` with fewer directories binds only
    those for the child — in the sandbox and in the prompt boundary drawn from it —
    and its parent keeps both. Sabotage: skip `restrict_paths` in `Grant.apply`, and
    the child binds its parent's store too."""
    cache, store = _dirs(tmp_path, "cache", "store")
    ctx, parent = await _parent(
        mount,
        *_writable(cache, store),
        cacheonly=[{"id": "sandbox-allow", "config": {"paths": [cache]}}],
    )

    run = await _spawn(ctx, parent, profile="cacheonly")

    sandbox = ctx.require(SANDBOX)
    assert list(as_seq(_admitted(ctx, parent)["paths"])) == [cache]
    assert sandbox.effective(SandboxPolicy(), agent=run.session_id).writable_extra == [cache]
    assert allowed_paths_of(ctx, run.session_id) == (Path(cache),)
    assert sandbox.effective(SandboxPolicy(), agent=parent.id).writable_extra == [cache, store]


async def test_a_child_cannot_be_given_a_directory_its_parent_cannot_write(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Sabotage: drop the `beyond` check in `sandbox_allow.narrows`, and the child is
    admitted."""
    cache, store = _dirs(tmp_path, "cache", "store")
    ctx, parent = await _parent(
        mount,
        *_writable(cache),
        wider=[{"id": "sandbox-allow", "config": {"paths": [cache, store]}}],
    )

    with pytest.raises(SubagentSpawnError, match=f"sandbox-allow lets a child write {store}"):
        await _spawn(ctx, parent, profile="wider")


@pytest.mark.parametrize(
    ("network", "said"),
    [
        ({"mode": "full"}, "gives a child full network where its parent has allowlist"),
        ({"hosts": ["pypi.org", "example.com"]}, r"other hosts than its parent's \(.*example\.com"),
        ({"hosts": ["pypi.org"]}, "other hosts than its parent's"),
    ],
    ids=["wider-mode", "other-host", "fewer-hosts"],
)
async def test_a_child_s_network_is_its_parent_s(
    mount: MountProfile, network: JsonObject, said: str
) -> None:
    """Wider is refused as every widening is; narrower is refused too, since one
    egress proxy serves every agent and a child cannot hold fewer hosts yet — given
    its parent's instead, the ceiling its admission states would not be the one it
    ran under. Sabotage: return the limit whatever the network says, and each is
    admitted."""
    ctx, parent = await _parent(
        mount, other=[{"id": "sandbox-allow", "config": {"network": network}}]
    )

    with pytest.raises(SubagentSpawnError, match=said):
        await _spawn(ctx, parent, profile="other")


async def test_a_child_s_sandbox_cannot_be_wider_than_its_parent_s(mount: MountProfile) -> None:
    ctx, parent = await _parent(
        mount, writer=[{"id": "sandbox", "config": {"defaultMode": "workspace-write"}}]
    )

    with pytest.raises(SubagentSpawnError, match="run workspace-write, where its parent runs"):
        await _spawn(ctx, parent, profile="writer")


async def test_a_child_holds_the_skills_found_under_the_paths_it_keeps(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Of the row's own skills — by the row that installed each, so another row's,
    found where this one does not look, stays. Sabotage: return an empty limit from
    `skills.narrows`, and the child keeps the skill under the directory its profile
    does not scan; test the skill's `source` rather than its row, and the other
    row's goes too."""
    kept, other = tmp_path / "kept", tmp_path / "other"
    write_skill(kept, "sort", body="Sort by kind.")
    write_skill(other, "audit", body="Audit it.")
    ctx, parent = await _parent(
        mount,
        {"id": "skills-progressive", "config": {"paths": [str(kept), str(other)]}},
        sorting=[{"id": "skills-progressive", "config": {"paths": [str(kept)]}}],
    )
    beside = not_none(ctx.require(MOUNT).forks["tool-bash"].ctx)
    found = Skill(
        name="shell-kit",
        description="shell things",
        path=str(other / "shell-kit" / "SKILL.md"),
        source="skills-progressive",
    )
    ctx.require(SKILLS).register(found, scope=beside)

    await _spawn(ctx, parent, profile="sorting")

    skills = as_seq(_admitted(ctx, parent)["skills"])
    assert "sort" in skills and "audit" not in skills
    assert "shell-kit" in skills, "another row's skill is not this row's to withhold"


class _Shelf(WireModel):
    skills: list[str] = Field(default_factory=list)


def _shelf_narrows(mounted: _Shelf, asked: _Shelf, _reach: ChildReach) -> ChildLimit:
    return ChildLimit(withheld_skills=frozenset(mounted.skills) - frozenset(asked.skills))


@plugin("shelf", affects="environment", inject=[SKILLS], config=_Shelf, narrows=_shelf_narrows)
async def _shelf(ctx: Context, config: _Shelf) -> None:
    for name in config.skills:
        ctx.require(SKILLS).register(Skill(name=name, description=f"{name}s"), scope=ctx)


async def test_a_row_says_how_a_child_holds_less_of_it(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) A row the narrowing module has never heard of joins by declaring
    `narrows=` beside its body. Sabotage: skip the declared narrowers in
    `narrowing`, and the child keeps the skill its profile's shelf leaves off."""
    module = types.ModuleType("ph_test_shelf")
    setattr(module, "shelf", _shelf)  # noqa: B010 - a module built at runtime
    monkeypatch.setitem(sys.modules, "ph_test_shelf", module)
    ctx, parent = await _parent(
        mount,
        {"id": "shelf", "name": "ph_test_shelf:shelf", "config": {"skills": ["sort", "audit"]}},
        sorting=[{"id": "shelf", "config": {"skills": ["sort"]}}],
    )

    await _spawn(ctx, parent, profile="sorting")

    skills = as_seq(_admitted(ctx, parent)["skills"])
    assert "sort" in skills and "audit" not in skills
