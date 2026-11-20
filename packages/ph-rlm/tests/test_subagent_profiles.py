"""The model of a child from the list of its parent, and a profile that its
parent assigns (S7b).

Gates:

- A skill that names `classify` starts its child on that route.
- A key that the spawn names and the list does not hold is refused. A key from a
  skill that this mount cannot run falls back to the route of the parent, and
  the fallback is reported.
- An assigned profile that is wider than the parent is refused, and the refusal
  names the row.

The other tests cover the behavior around the gates:

- A spawn names a key or a whole route itself.
- The admission records what the spawn named separately from what that name
  resolved to. A restart resolves the name again (`test_child_logs`).
- An assigned profile narrows what a child holds: its tools, its model, its
  access, its skills and its writable directories. The narrowing is read against
  the mount of the parent itself, and each row narrows by the narrower that its
  plugin declares.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field
from rlm_fixtures import MODELS_ROW, PROVIDER_ROW

from ph.cordis import ChildLimit, ChildReach, Context, Profile, ProfileDocument, plugin
from ph.json import JsonObject, as_obj, as_seq, thaw_json
from ph.keys import (
    AGENTS,
    LLM,
    LLM_FAKE,
    MOUNT,
    NAMED_PROFILES,
    SANDBOX,
    SESSIONS,
    SKILLS,
    SUBAGENTS,
)
from ph.llm.adapter import ResolvedModel
from ph.llm.fake import FakeAdapter
from ph.llm.types import Sampling
from ph.paths import canonical
from ph.seams.models import ModelChoice, choose
from ph.seams.sandbox import SandboxPolicy, allowed_paths_of
from ph.seams.skills import Skill
from ph.seams.subagents import SubagentRequest, SubagentSpawnError, admission_notes
from ph.testing import FAKE_OPTIONS, MountProfile, not_none, skill, write_skill
from ph.wire import WireModel
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio


class _Profiles:
    """`ctx.named_profiles` for a test. Each name is the layers of the parent plus
    a patch."""

    def __init__(self, parent: Profile, **patches: list[JsonObject]) -> None:
        self.parent = parent
        self.patches = patches

    def compose(self, name: str) -> Profile:
        if name not in self.patches:
            raise ValueError(f'unknown profile "{name}"')
        documents = [*self.parent.documents, ProfileDocument(name, list(self.patches[name]))]
        return Profile.from_documents(documents, name=name)


async def _parent(
    mount: MountProfile,
    *rows: JsonObject,
    models: JsonObject = MODELS_ROW,
    **profiles: list[JsonObject],
) -> tuple[Context, Any]:
    """A parent under `models`, with `profiles` it may assign by name."""
    ctx = await mount(dict(PROVIDER_ROW), dict(models), *rows)
    ctx.provide(NAMED_PROFILES, _Profiles(ctx.require(MOUNT).profile, **profiles))
    session = ctx.require(SESSIONS).create("parent")
    return ctx, ctx.require(AGENTS).create(session, FAKE_OPTIONS)


async def _spawn(ctx: Context, parent: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="sort these", parent=parent, **kwargs)
    )


def _admitted(ctx: Context, parent: Any) -> JsonObject:  # noqa: ANN401
    """The admission of the one child, read from the log of the child. A restart
    reads the narrowing back from there, so it must be recorded there."""
    (child,) = ctx.require(SUBAGENTS).children(parent.session.id).values()
    return as_obj(thaw_json(not_none(child.admission).to_wire()))


async def test_a_skill_naming_a_listed_model_starts_its_child_on_that_route(
    mount: MountProfile,
) -> None:
    """The first half of the gate. The skill directs the child, so the skill says
    which model the child thinks with. The list of the parent resolves the key.
    The admission records both the key and the route, but not as the choice of
    the spawn, because a restart asks the skill again. Sabotage: ask no skill's
    model in `_resolve_route`, and the child runs on its parent's `fake-1`; record
    the skill's key as the choice, and a restart keeps it after the skill names
    another."""
    ctx, parent = await _parent(mount)
    ctx.require(SKILLS).register(skill("sort", model="classify"))

    run = await _spawn(ctx, parent, skills=("sort",))

    assert (run.model_provider, run.model) == ("fake", "fake-9")
    admitted = _admitted(ctx, parent)
    assert admitted["modelKey"] == "classify" and admitted["model"] == "fake-9"
    assert "choice" not in admitted
    child = ctx.require(AGENTS).get(run.session_id)
    assert child is not None and child.options.model_key == "classify"


async def test_a_spawn_names_a_listed_model_by_key(mount: MountProfile) -> None:
    """The admission records the key as the name that the spawn gave, and a
    restart resolves that name again. Sabotage: record the resolved key only, and
    a restart has nothing to resolve."""
    ctx, parent = await _parent(mount)

    run = await _spawn(ctx, parent, choice=ModelChoice(key="classify"))

    assert run.model == "fake-9"
    assert as_obj(_admitted(ctx, parent)["choice"])["key"] == "classify"


TUNED: JsonObject = {
    "id": "models",
    "config": {
        "default": "main",
        "models": {
            "main": {
                "provider": "fake",
                "model": "fake-1",
                "sampling": {"temperature": 0.9},
                "maxTokens": 64,
                "reasoningEffort": "low",
            },
            "careful": {
                "provider": "fake",
                "model": "fake-7",
                "sampling": {"temperature": 0.2},
                "maxTokens": 777,
                "reasoningEffort": "high",
            },
            "plain": {"provider": "fake", "model": "fake-9"},
        },
    },
}
"""Routes with call settings and one route with none. A child that took the
settings of its parent, and not the settings of its route, then shows it."""


async def _tuned_parent(mount: MountProfile) -> tuple[Context, Any]:
    """A parent started on `main` in the way that a daemon starts a root: with the
    options of the entry."""
    ctx = await mount(dict(PROVIDER_ROW), dict(TUNED))
    session = ctx.require(SESSIONS).create("parent")
    return ctx, ctx.require(AGENTS).create(session, choose(ctx, ModelChoice(key="main")).options())


async def test_a_child_on_a_listed_route_runs_with_that_route_s_settings(
    mount: MountProfile,
) -> None:
    """The child has the temperature, the token budget and the effort of the
    route, as a root started on the key has them. The admission records them with
    the route for the audit. Sabotage: drop the entry from `resolve_model`, and
    the child runs with its parent's."""
    ctx, parent = await _tuned_parent(mount)

    run = await _spawn(ctx, parent, choice=ModelChoice(key="careful"))

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == choose(ctx, ModelChoice(key="careful")).options()
    route = as_obj(_admitted(ctx, parent)["route"])
    sampling = as_obj(route["sampling"])
    assert (route["model"], sampling["temperature"], route["maxTokens"]) == ("fake-7", 0.2, 777)


async def test_a_listed_route_that_sets_nothing_takes_nothing_from_the_parent(
    mount: MountProfile,
) -> None:
    """A temperature that was tuned for the model of the parent is not a
    temperature for this model. Sabotage: build the child from its parent's
    options with the route swapped in, as `ph_rlm` did, and it runs at its
    parent's 0.9 on a 64-token budget."""
    ctx, parent = await _tuned_parent(mount)

    run = await _spawn(ctx, parent, choice=ModelChoice(key="plain"))

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == choose(ctx, ModelChoice(key="plain")).options()


async def test_a_child_on_its_parent_s_route_runs_with_its_parent_s_settings(
    mount: MountProfile,
) -> None:
    ctx, parent = await _tuned_parent(mount)

    run = await _spawn(ctx, parent)

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == parent.options


async def test_a_child_runs_at_the_spawn_s_effort_else_its_skills_else_its_route_s(
    mount: MountProfile,
) -> None:
    """The order of `child_options`: the effort that the spawn names, else an
    effort that its skills agree on (`default` among them asks for the effort of
    the provider itself), else the effort of its route. Skills that name
    different efforts leave the effort to the route. Sabotage: drop `_with_effort`
    from `resolve_model`, put the skills' effort over the spawn's, or let one of
    two disagreeing skills win."""
    ctx, parent = await _tuned_parent(mount)
    skills = ctx.require(SKILLS)
    skills.register(skill("quick", effort="default"))
    skills.register(skill("deep", effort="high"))

    async def effort(**kwargs: Any) -> str | None:  # noqa: ANN401
        run = await _spawn(ctx, parent, **kwargs)
        return not_none(ctx.require(AGENTS).get(run.session_id)).options.reasoning_effort

    assert await effort() == "low", "the parent's route, at the parent's effort"
    assert await effort(skills=("quick",)) == "default"
    assert await effort(skills=("quick",), reasoning_effort="medium") == "medium"
    assert await effort(skills=("quick", "deep")) == "low"


EFFORTS: JsonObject = {
    "id": "models",
    "config": {
        "default": "main",
        "models": {
            "main": {
                "provider": "fake",
                "model": "fake-1",
                "efforts": {"low": {"think": "brief"}, "none": {"think": "off"}},
            },
        },
    },
}
"""The route of `FAKE_OPTIONS`, listed with the two efforts that its profile says
how to send."""


async def test_a_child_runs_without_an_effort_its_route_cannot_send_and_its_parent_is_told(
    mount: MountProfile,
) -> None:
    """The request of the child itself withholds an effort that the profile of its
    route does not define. The parent is therefore told. Otherwise the parent
    reads a quick answer as a careful one. The admission records the effort that
    wins, and the answer to the spawn says what the route runs at. An effort that
    the route defines, `default`, or no effort produces no note. The handle of a
    resumed parent reads the same from the log of the child. Sabotage: drop
    `_unsent_effort`, ask about the spawn's effort rather than the one that wins,
    or leave it out of `ChildState.run`."""
    ctx, parent = await _parent(mount, models=EFFORTS)
    ctx.require(SKILLS).register(skill("deep", effort="high"))

    async def told(**kwargs: Any) -> list[str]:  # noqa: ANN401
        run = await _spawn(ctx, parent, **kwargs)
        children = ctx.require(SUBAGENTS).children(parent.session.id).values()
        (child,) = (child for child in children if child.session_id == run.session_id)
        assert admission_notes(child.run()) == admission_notes(run)
        return admission_notes(run)

    [note] = await told(reasoning_effort="high")
    assert "'high'" in note and "fake/fake-1" in note and "low, none or default" in note
    [note] = await told(skills=("deep",))
    assert "'high'" in note, "the effort its skill names is the one it runs without"
    assert await told(reasoning_effort="low") == []
    assert await told(reasoning_effort="default") == []
    assert await told() == []


async def test_a_whole_route_the_list_holds_runs_as_its_entry(mount: MountProfile) -> None:
    """A route that is named as provider and model, and not by key, is still the
    listed entry. The child gets the key and the settings of the entry, as a root
    started on the route does. Sabotage: leave a whole route unresolved, and the
    child runs `fake-7` at its parent's 0.9."""
    ctx, parent = await _tuned_parent(mount)

    run = await _spawn(ctx, parent, choice=ModelChoice(provider="fake", model="fake-7"))

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == choose(ctx, ModelChoice(key="careful")).options()
    assert _admitted(ctx, parent)["modelKey"] == "careful"


async def test_a_whole_route_the_list_does_not_hold_takes_nothing_from_the_parent(
    mount: MountProfile,
) -> None:
    """A route that no entry matches runs as given, with no settings, because the
    settings of the parent were tuned for another model. Sabotage: leave a whole
    route unresolved, and the child runs at its parent's 0.9 on a 64-token
    budget."""
    ctx, parent = await _tuned_parent(mount)

    unlisted = ModelChoice(provider="fake", model="fake-unlisted")
    run = await _spawn(ctx, parent, choice=unlisted)

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == choose(ctx, unlisted).options()
    assert (child.options.sampling, child.options.model_key) == (Sampling(), "")


async def test_a_key_the_list_does_not_hold_is_refused_and_admits_nothing(
    mount: MountProfile,
) -> None:
    """An agent can reach the models that its profile lists. A key that the spawn
    names is the choice of the parent itself, so an unknown key is refused and
    the child is not run elsewhere. Sabotage: fall back to the parent's route for
    an unknown key, and the child is admitted."""
    ctx, parent = await _parent(mount)

    with pytest.raises(SubagentSpawnError, match="it lists classify, main"):
        await _spawn(ctx, parent, choice=ModelChoice(key="summarize"))

    assert ctx.require(SUBAGENTS).children(parent.session.id) == {}


UNSERVED: JsonObject = {
    "id": "models",
    "config": {
        "default": "main",
        "models": {
            "main": {"provider": "fake", "model": "fake-1"},
            "classify": {"provider": "absent", "model": "sorter-2"},
        },
    },
}
"""A list holding a route that no adapter here serves."""


@pytest.mark.parametrize(
    ("models", "asked", "reason", "said"),
    [
        (MODELS_ROW, "summarize", "unlisted", '"summarize", which this profile does not list'),
        (UNSERVED, "classify", "unserved", "lists on a provider no adapter here serves"),
    ],
)
async def test_a_skill_s_model_this_mount_cannot_run_falls_back_to_the_parent_s_route(
    mount: MountProfile,
    caplog: pytest.LogCaptureFixture,
    models: JsonObject,
    asked: str,
    reason: str,
    said: str,
) -> None:
    """The author of the skill wrote the key without knowing this profile. The
    model of the parent itself therefore does the work as well as it can until
    the user sets one up. This is graceful degradation: it is reported to the
    parent, logged, and recorded in the admission. Sabotage: refuse it as a
    spawn's own key is refused, or drop the fallback from the run, the admission
    or the log."""
    ctx, parent = await _parent(mount, models=models)
    ctx.require(SKILLS).register(skill("sort", model=asked))

    with caplog.at_level("WARNING", logger="ph.seams.subagents"):
        run = await _spawn(ctx, parent, skills=("sort",))

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert child.options == parent.options, "on the parent's route, with its settings"
    fallback = {"asked": [asked], "reason": reason}
    assert as_obj(thaw_json(run.to_wire()))["modelFallback"] == fallback
    assert _admitted(ctx, parent)["modelFallback"] == fallback
    [note] = admission_notes(run)
    assert said in note and "your own model" in note
    assert asked in caplog.text, "the person running it is not told"


async def test_skills_that_name_different_models_are_refused(mount: MountProfile) -> None:
    ctx, parent = await _parent(mount)
    for name, key in (("sort", "classify"), ("write", "main")):
        ctx.require(SKILLS).register(skill(name, model=key))

    with pytest.raises(SubagentSpawnError, match="name different models"):
        await _spawn(ctx, parent, skills=("sort", "write"))


async def test_an_assigned_profile_wider_than_its_parent_is_refused_naming_the_row(
    mount: MountProfile,
) -> None:
    """The second half of the gate. The parent runs no question tool. A profile
    that runs one cannot be given to its child, because the only mount to run it
    on is the mount of the parent. Sabotage: drop the row check in `narrowing`,
    and the child is admitted."""
    ctx, parent = await _parent(mount, asking=[{"id": "tool-ask-user", "disabled": False}])

    with pytest.raises(SubagentSpawnError, match='profile "asking": it runs tool-ask-user'):
        await _spawn(ctx, parent, profile="asking")


async def test_an_assigned_profile_narrows_the_child_s_tools_model_and_access(
    mount: MountProfile,
) -> None:
    """The child runs part of what the parent holds: no `bash`, its own default
    model from the list of the parent, and the read-only sandbox that it starts
    in. The admission records these, so the reach is fixed there. Sabotage: skip
    the dropped rows in `_given`, and the child keeps `bash`."""
    ctx, parent = await _parent(
        mount,
        sorter=[
            {"id": "tool-bash", "disabled": True},
            {"id": "models", "config": {**MODELS_ROW["config"], "default": "classify"}},
        ],
    )

    run = await _spawn(ctx, parent, profile="sorter")

    admitted = _admitted(ctx, parent)
    tools = as_seq(admitted["tools"])
    assert "bash" not in tools and "read" in tools
    assert run.model == "fake-9" and admitted["profile"] == "sorter"
    assert as_obj(admitted["choice"])["key"] == "classify", "a restart resolves it again"
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
    """A spawn that names a skill puts the body of the skill in the prompt of the
    child. That is a read at runtime like any other (S8), so the log of the
    parent says which text was read, by hash."""
    write_skill(tmp_path, "sort", body="Sort by kind.")
    ctx = await mount(
        dict(PROVIDER_ROW),
        dict(MODELS_ROW),
        {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}},
    )
    session = ctx.require(SESSIONS).create("parent")
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await _spawn(ctx, parent, skills=("sort",))

    (read,) = session.select("skill/read")
    assert (read.data["name"], read.data["via"]) == ("sort", "brief")


async def test_a_skill_goes_with_the_row_that_gave_it(mount: MountProfile) -> None:
    """(b) A row that the assigned profile does not run takes its skills with it,
    as it takes its tools. The skills are found by the row that installed each
    (`SkillService.registrants`), whatever row that is. A skill that no row
    installed stays. Sabotage: skip the registrants in `_skills`, and the dropped
    row's skill stays."""
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
    """The paths of item 3. An assigned `sandbox-allow` with fewer directories
    binds only those directories for the child, both in the sandbox and in the
    prompt boundary that is drawn from it. The parent keeps both directories.
    Sabotage: skip `restrict_paths` in `Grant.apply`, and the child binds its
    parent's store too."""
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
    """A wider network is refused, as every widening is. A narrower network is
    refused too, because one egress proxy serves every agent and a child cannot
    hold fewer hosts yet. If the child were given the network of its parent
    instead, the ceiling that its admission states would not be the ceiling that
    it ran under. Sabotage: return the limit whatever the network says, and each
    is admitted."""
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
    """This applies to the skills of the row itself, decided by the row that
    installed each skill. A skill of another row stays, even when it was found
    in a directory that the assigned row does not scan. Sabotage: return an empty
    limit from `skills.narrows`, and the child keeps the skill under the
    directory its profile does not scan; test the skill's `source` rather than
    its row, and the other row's goes too."""
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
    """(a) A row that the narrowing module does not know joins by declaring
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


MAIN_ONLY: JsonObject = {
    "id": "models",
    "config": {"default": "main", "models": {"main": {"provider": "fake", "model": "fake-1"}}},
}
"""A list that serves no shared kind (`MODEL_KINDS`)."""


async def test_the_shipped_categorize_skill_runs_on_the_classify_kind_where_one_is_listed(
    mount: MountProfile,
) -> None:
    """The gate of A2. The skill that ph-core ships for sorting asks for the
    shared `classify` kind and owes a checked value. It therefore runs on the
    route that a profile lists under that name. A profile that lists none runs
    the child on the model of the parent and tells the parent so. Sabotage: name
    a model rather than the kind, or stop offering the skill with the provider."""
    listed, parent = await _parent(mount)
    shipped = not_none(listed.require(SKILLS).get("categorize", parent.ctx))
    assert shipped.model == "classify" and shipped.output is not None
    assert list(as_seq(shipped.output["required"])) == ["category", "reason"]

    run = await _spawn(listed, parent, skills=("categorize",))
    child = not_none(listed.require(AGENTS).get(run.session_id))
    assert (child.options.model_key, child.options.model) == ("classify", "fake-9")

    unlisted, other = await _parent(mount, models=MAIN_ONLY)
    run = await _spawn(unlisted, other, skills=("categorize",))
    assert not_none(unlisted.require(AGENTS).get(run.session_id)).options == other.options
    [note] = admission_notes(run)
    assert '"classify"' in note and "your own model" in note


BLIND: JsonObject = {
    "id": "models",
    "config": {
        "default": "main",
        "models": {
            "main": {"provider": "fake", "model": "fake-1"},
            "classify": {"provider": "blind", "model": "sorter-1"},
        },
    },
}
"""`classify` on a provider whose routes take text alone."""


async def _seeing(mount: MountProfile, *, parent_sees: bool) -> tuple[Context, Any]:
    """A parent under `BLIND`, on a route that takes images or not, and a sorting
    skill that hands its child images."""
    ctx, parent = await _parent(mount, models=BLIND)
    ctx.require(LLM).register_adapter(["blind"], FakeAdapter())
    if parent_sees:
        ctx.require(LLM_FAKE).route = ResolvedModel(accepts=frozenset({"image/png"}))
    ctx.require(SKILLS).register(skill("sort-picture", model="classify", accepts=("image/*",)))
    return ctx, parent


async def test_a_child_whose_kind_cannot_see_its_media_runs_on_a_parent_that_can(
    mount: MountProfile,
) -> None:
    """The first half of the gate of A3. The route that the kind resolves to takes
    no images, and the route of the parent does. The child therefore runs on the
    route of the parent, and the parent and the admission say why. Sabotage: keep
    the blind route, or drop the reason."""
    ctx, parent = await _seeing(mount, parent_sees=True)

    run = await _spawn(ctx, parent, skills=("sort-picture",))

    assert not_none(ctx.require(AGENTS).get(run.session_id)).options == parent.options
    fallback = {"asked": ["classify"], "reason": "cannot-accept", "media": ["image/*"]}
    assert _admitted(ctx, parent)["modelFallback"] == fallback
    [note] = admission_notes(run)
    assert "cannot take image/*" in note and "your own model" in note


async def test_a_child_no_route_here_lets_see_its_media_runs_and_its_parent_is_told(
    mount: MountProfile,
) -> None:
    """The second half of the gate of A3. Neither route takes images. The child
    still runs on the route of its kind and sees placeholders, and the parent is
    told what to set up. Sabotage: refuse the spawn, or say nothing."""
    ctx, parent = await _seeing(mount, parent_sees=False)

    run = await _spawn(ctx, parent, skills=("sort-picture",))

    child = not_none(ctx.require(AGENTS).get(run.session_id))
    assert (child.options.provider, child.options.model) == ("blind", "sorter-1")
    shortfall = {"media": ["image/*"], "route": "blind/sorter-1"}
    assert _admitted(ctx, parent)["mediaShortfall"] == shortfall
    [note] = admission_notes(run)
    assert "placeholder" in note and "image/*" in note


async def test_a_spawn_that_names_a_route_its_media_cannot_reach_is_refused(
    mount: MountProfile,
) -> None:
    """The parent chose the route itself and can choose again at once, so the
    spawn is refused and not moved. Sabotage: degrade it as a skill's own key is."""
    ctx, parent = await _seeing(mount, parent_sees=True)

    with pytest.raises(SubagentSpawnError, match="cannot take image/\\*"):
        await _spawn(ctx, parent, skills=("sort-picture",), choice=ModelChoice(key="classify"))
