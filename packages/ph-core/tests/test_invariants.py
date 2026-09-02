"""P6-01 — the invariants registry, and the three core checks it collects (A11, I6, I2).

**What the registry is for is not the checking; it is the *saying*.** Every
invariant in pH is enforced by a row, and a row is optional — so "pH enforces I3"
was never true of pH, only of a profile that happened to mount
`agent-loop-invariant`, and nothing anywhere said which profiles those were. A
person could read `DESIGN.md`, run a deployment that promised strictly less, and
have no way to find out. The registry closes that by making the enforced set
enumerable; the checks are what make the claim testable rather than decorative.

**The two kinds are asserted separately on purpose.** An inline invariant refuses
on the path it governs and has nothing to poll between requests; a pollable one
answers about state right now. Collapsing them would let the report say "holds"
about an invariant whose listener had never been reached — the reassuring answer,
given for the deployment where it is least earned.

Each check below is exercised by *breaking the thing it watches*, not by
substituting a check that returns a violation. A test that only proves the
registry can carry a string proves nothing about whether the invariant would ever
fire, which is the failure mode an invariant suite has.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.cordis import DEPLOYMENT, Context
from ph.seams.invariants import Invariant, InvariantRegistry, Violation
from ph.seams.scope_invariant import violations as scope_violations
from ph.seams.skills import SkillRestriction, SkillService
from ph.seams.skills_invariant import violations as skill_violations
from ph.session import SurfaceIntent
from ph.session.invariant import violations as session_violations
from ph.testing import (
    report_section,
    simple_tool,
    skill,
    skill_service,
    tool_runtime,
    user_payload,
)
from ph.tools.invariant import violations as tool_violations

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------ registry --


async def test_invariants_are_ordered_then_named() -> None:
    registry = InvariantRegistry(ctx=Context())
    registry.register(Invariant(id="second", statement="b", check=list, order=10))
    registry.register(Invariant(id="first", statement="a", check=list, order=1))

    assert [one.id for one in registry.enforced()] == ["first", "second"]


async def test_an_inline_invariant_is_enforced_but_never_reported_as_holding() -> None:
    """The distinction the `check=None` field exists to keep.

    I3 refuses on the path it governs; there is no state between requests to ask
    about. A `check` returning `[]` here would read as "verified" while verifying
    nothing — and it would read that way most confidently in a deployment where
    the enforcing listener had never once been reached.
    """
    registry = InvariantRegistry(ctx=Context())
    registry.register(Invariant(id="inline", statement="every request is checked"))

    assert registry.verify() == [], "an unpollable invariant must not invent an answer"
    assert [one.id for one in registry.enforced()] == ["inline"]
    ((_, answer),) = registry.describe()
    assert answer.startswith("enforced inline"), "an inline invariant was reported as checked"


async def test_a_check_that_raises_becomes_a_violation_rather_than_a_dropped_row() -> None:
    """Where this seam parts company with `ctx.diagnostics`, and why.

    A diagnostic that cannot read is dropped so the report survives. An invariant
    that cannot be evaluated is not evidence of health — swallowing it would make
    the quietest possible report the one where the most is broken, which is the
    exact inversion a health check must not have.
    """

    def explode() -> list[str]:
        raise RuntimeError("the fold is unreadable")

    registry = InvariantRegistry(ctx=Context())
    registry.register(Invariant(id="broken", statement="something", check=explode))

    (violation,) = registry.verify()
    assert violation.invariant == "broken"
    assert "the check itself failed" in violation.detail
    assert "unreadable" in violation.detail, "the reason was lost on the way to the report"
    ((_, answer),) = registry.describe()
    assert answer.startswith("VIOLATED")


async def test_an_invariant_stops_being_claimed_when_its_row_unwinds() -> None:
    """The promise and the enforcer have one lifetime.

    A row that is unloaded stops enforcing, and a registry that kept advertising
    its invariant would be making exactly the claim this seam was built to stop
    anyone from having to guess at.
    """
    root = Context()
    registry = InvariantRegistry(ctx=root)
    disposer = registry.register(Invariant(id="temporary", statement="held", check=list))

    assert [one.id for one in registry.enforced()] == ["temporary"]
    disposer()
    assert registry.enforced() == []


async def test_verify_reports_every_violation_not_the_first() -> None:
    """A report that stopped at the first would make the second fix a surprise."""
    registry = InvariantRegistry(ctx=Context())
    registry.register(
        Invariant(id="two-ways", statement="s", check=lambda: ["one way", "another way"])
    )

    assert registry.verify() == [
        Violation("two-ways", "one way"),
        Violation("two-ways", "another way"),
    ]


# ------------------------------------------------------------------- session --


async def test_a_log_written_behind_append_trips_the_events_snapshot(mount: Any) -> None:
    """The first of `Session.stale`'s three projections, broken the only way it can be.

    `events` is a snapshot invalidated on append, so it agrees with the log for as
    long as `append` is the only writer — which is precisely why the check has to
    reach past it to have any content. Reaching into `_log` here *is* the
    scenario: the caches exist to be invalidated by `append`, and only a writer
    that bypassed it can leave one stale.

    This was labelled A1 for one draft. It is not: `seq` *is* `len(_log)`
    (`Session.seq` is a property over it), so A1 holds by construction and a
    check for it would have no content. What can drift is the snapshot, and the
    message now says that rather than asserting a cause it cannot see.
    """
    ctx = await mount()
    session = ctx.sessions.create("s1")
    session.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))

    assert session_violations(ctx) == [], "a session built by appending disagreed with itself"

    session._log.pop()

    found = session_violations(ctx)
    assert any("events snapshot holds 1" in one for one in found), found


async def test_a_surface_that_outran_its_log_trips_the_session_invariant(mount: Any) -> None:
    """I6's surface half, isolated from the snapshot.

    `SurfaceManager` folds incrementally and `fold_surface` replays the whole log
    through the same rules; the manager's own docstring says an external
    reconstructor "must reach the same nodes", and nothing asserted it. Dropping
    an event *and* the snapshot leaves only the projection wrong — which is the
    shape a drifted projection actually has.

    What it would cost is the log ceasing to be the single source of truth for
    what a model saw: a live session, a resumed one and an offloaded one would
    disagree about the conversation, and only one of the three is the log.
    """
    ctx = await mount()
    session = ctx.sessions.create("s1")
    session.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))
    assert session.surface.nodes, "the manager had not folded, so there is nothing to outrun"

    session._log.pop()
    session._events_snapshot = None

    found = session_violations(ctx)
    assert any("surface projects 1 node(s)" in one for one in found), found


async def test_a_stale_derivation_trips_the_session_invariant(mount: Any) -> None:
    """The third projection, and the one I3 stands on.

    `derive_messages` is memoized per surface node, and it is what
    `agent-loop-invariant` compares every request against. A memo that drifted
    from the surface would let that invariant pass a request built from a stale
    history — the reassuring answer, given for exactly the case it exists to
    refuse. Emptying the memo while leaving its node count is what a forgotten
    invalidation looks like from the inside.
    """
    ctx = await mount()
    session = ctx.sessions.create("s1")
    session.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))
    assert session.derive_messages(), "nothing was derived, so there is nothing to go stale"

    session._derived = ()

    found = session_violations(ctx)
    assert any("derive_messages holds 0" in one for one in found), found


# --------------------------------------------------------------------- tools --


async def test_a_layer_changed_without_a_generation_bump_trips_the_tool_invariant() -> None:
    """I6 applied to a cache, and the reason it is worth applying there.

    `ToolRegistry.view` memoizes per isolation chain and invalidates on a
    counter. What a stale entry serves is not a stale *listing* — it is the set
    of tools an agent may call, so a registration path that forgot `_changed()`
    keeps a tool callable after the row owning it unloaded. That is a capability
    outliving its scope, which is I2's failure wearing a cache's clothes.

    Mutating the global layer without bumping the counter is what a forgetful
    path does. A bare runtime and a tool this test names, rather than a whole
    profile and whichever tool `base.yaml` happened to register first.
    """
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    tools.view(root)

    assert tool_violations(root) == []

    tools._layers[None].tools.pop("read")

    (found,) = tool_violations(root)
    assert "differs from a rebuild on ['read']" in found


async def test_a_registry_change_empties_the_cache_rather_than_ageing_it() -> None:
    """Why `stale_views` compares every cached entry, with no generation filter.

    A draft skipped entries below the current generation as "invalidated, never
    served" — and had a test that passed vacuously, because `_changed()` clears
    the table when it bumps the counter and there was nothing left to skip. The
    generation in each entry is therefore always the current one, and the
    filter guarded a state the cache cannot hold. This pins the mechanism the
    check now relies on.
    """
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    tools.view(root)
    assert tools._views, "nothing was cached, so there is nothing to invalidate"

    tools._changed()

    assert tools._views == {}, "a change aged the cache instead of emptying it"
    assert tool_violations(root) == []


# -------------------------------------------------------------------- skills --


async def test_a_reach_cache_that_outlived_its_generation_trips_the_skill_invariant() -> None:
    """The same I6 the tool view gets, on the registry where drift costs the ceiling.

    `SkillService.reach` is `ToolRegistry.view`'s arrangement — chain-keyed, memoized
    against a counter — and has its failure mode. But `reach` is the one place skill
    filters compose, so a stale entry does not serve a stale *listing*: it keeps a
    skill reachable after the row owning it unloaded.

    Mutating `_skills` without `_changed()` is what a forgetful registration path
    does; every shipped one calls it, which is why the check has no content until
    somebody writes the one that does not.
    """
    root, skills = skill_service()
    skills.register(skill("read-code"))
    assert skills.reach(DEPLOYMENT) == {"read-code"}, "nothing was cached to go stale"

    assert skill_violations(root) == []

    skills._skills.pop("read-code")

    (found,) = skill_violations(root)
    assert "['read-code']" in found and "rebuild gives []" in found


async def test_a_narrowing_clears_the_cache_rather_than_ageing_it() -> None:
    """The ordinary path stays silent, including the one that changes the answer.

    `restrict` mutates the table *and* calls `_changed()`, which clears it — so the
    next `reach` rebuilds and there is nothing stale to find. A check that reported
    every narrowing would fire on the deployments that use narrowing most, which is
    the same as not firing at all.

    The clearing is asserted directly. A first draft cached a set, restricted, and
    then only checked the *answer* — but `_changed()` empties `_reach`, so the entry
    was gone before either assertion ran and the mechanism the docstring argued for
    was the one thing nothing observed.
    """
    root, skills = skill_service()
    skills.register(skill("read-code"))
    skills.register(skill("write-code"))
    skills.reach(DEPLOYMENT)
    assert skills._reach, "nothing was cached to invalidate"

    skills.restrict(SkillRestriction(deny=("write-code",)))

    assert skills._reach == {}, "a narrowing aged the cache instead of emptying it"
    assert skills.reach(DEPLOYMENT) == {"read-code"}, "the narrowing did not take"
    assert skill_violations(root) == []


async def test_a_child_reaching_past_its_ancestor_trips_the_skill_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling itself — the failure comparing a fold against itself cannot see.

    Both sides of the staleness check go through `_resolve`, so a fold that composed
    restrictions *wrongly* would be confirmed rather than caught. And the wrong
    composition is the one this registry exists to prevent: P4-13b's whole security
    content is that a child holds a subset of its parent and never more. The tool
    registry shipped exactly this bug — "a child inherited its parent's scoped tools
    and could not be narrowed out of them, so a grant could widen a child but never
    bound it" — and `stale_views` could not have caught it either.

    Checkable here and not next door because two things hold for skills alone:
    `isolation_chain` is suffix-ordered, so an ancestor's chain is a suffix of its
    descendant's, and a restriction only ever intersects. A deeper chain therefore
    applies a superset of the filters and must reach a subset. A tool view has no
    such rule — a grandchild may legitimately hold what its granting parent cannot
    see.

    The fold is replaced with the broken one rather than a cache entry being planted,
    because that is the shape of the real defect: with the bug in `_resolve` itself
    both sides of the equality agree, the staleness clause stays silent, and only the
    ceiling clause has anything to say. Planting a widened entry would have tripped
    the staleness clause instead and proved nothing about this one.
    """
    root, skills = skill_service()
    skills.register(skill("read-code"))
    skills.register(skill("write-code"))
    parent = root.scope("parent", module="test")
    child = parent.scope("child", module="test")
    skills.restrict(SkillRestriction(deny=("write-code",)), scope=parent)

    assert skills.reach(child) == {"read-code"}, "the parent's narrowing did not reach the child"
    assert skill_violations(root) == []

    # The fold as the tool registry once had it: this layer's filters only, so an
    # ancestor's narrowing never reaches the child.
    monkeypatch.setattr(
        SkillService,
        "_resolve",
        lambda self, chain: frozenset(
            name
            for name in self._skills
            if all(one.admits(name) for one in self._restrictions.get(chain[0], ()))
        ),
    )
    skills._changed()
    assert skills.reach(parent) == {"read-code"}
    assert skills.reach(child) == {"read-code", "write-code"}, "the bug did not take"

    found = skill_violations(root)
    assert [one for one in found if "where a rebuild gives" in one] == [], (
        "the staleness clause fired, so this is not testing the ceiling clause"
    )
    assert any("a narrowing widened" in one and "write-code" in one for one in found), found


# ------------------------------------------------------ dead cache keys (I2) --


@pytest.mark.parametrize("registry", ["tools", "skills"])
async def test_a_disposed_scope_is_not_retained_as_a_cache_key(registry: str) -> None:
    """The leak the two chain-keyed caches had, and the one nothing could observe.

    Both memos key on `tuple(target.isolation_chain())`, which holds its scopes
    **strongly** — so a disposed agent's `Context` was retained by its own cache
    key until something invalidated the registry, which a deployment that has
    stopped registering never does. Measured before the fix: 500 sequential
    agents left 500 entries in each table, ~2 KiB apiece, unbounded; 2000 agents
    retained 4.0 MiB. After: 9 KiB.

    **`scope-unwind` cannot see this.** That poll walks live children from the
    root, and these scopes are gone from the tree — properly, since P6-02 made
    leaving it a guarantee. The retention is on the registry side, which is why
    the fix is a sweep on the miss path rather than another row in the report.

    Both registries are asserted through one test because the property belongs to
    the *key*, which is the only thing they share — their values, folds and
    invalidation side effects all differ.
    """
    if registry == "tools":
        root, service = tool_runtime()
        service.register(simple_tool("read"))
        fill, table = service.view, service._views
    else:
        root, service = skill_service()
        service.register(skill("read-code"))
        fill, table = service.reach, service._reach

    kept = root.scope("kept", module="test")
    fill(kept)
    for index in range(20):
        agent = root.scope(f"agent:{index}", module="test")
        fill(agent)
        await agent.dispose()
    fill(root.scope("fresh", module="test"))

    assert all(key is None or key.active for chain in table for key in chain), (
        f"a disposed scope is still a {registry} cache key"
    )
    # The live scope's entry survives: this is a sweep of the dead, not a flush.
    assert tuple(kept.isolation_chain()) in table, "sweeping dropped a live scope's entry"


# --------------------------------------------------------------------- scope --


async def test_a_disposed_scope_a_parent_still_holds_trips_the_scope_invariant() -> None:
    """I2's structural half: the leak that does not announce itself.

    Nothing errors and nothing is served twice — the scope simply stays
    reachable, with its services and everything they close over, for as long as
    its parent lives. For a deployment scope that is the process. It is also
    invisible to a test that disposes a root and asserts on what ran, because
    that part works.
    """
    root = Context()
    child = root.scope("child")

    assert scope_violations(root) == []

    await child.dispose()
    assert scope_violations(root) == [], "a disposed scope was not released by its parent"

    # What a `dispose` that unwound but did not deregister leaves behind.
    root._children.append(child)

    (found,) = scope_violations(root)
    assert "disposed scope" in found and "child" in found


async def test_the_scope_walk_terminates_on_a_cycle() -> None:
    """A health check that hangs is worse than one that is absent.

    A parent listing itself is not a shape the tree should ever reach, which is
    exactly why the walk must survive meeting one: the run where the structure is
    broken is the run this is being asked to report on. The `seen` set is what
    turns a hang into a finding — and the finding is the broken back-link, since
    a context whose `parent` is not the scope listing it is unreachable for
    disposal from there.
    """
    root = Context()
    root._children.append(root)

    (found,) = scope_violations(root)
    assert "does not point back" in found


# ------------------------------------------------------------------ mounting --


async def test_the_rows_reach_the_report_through_the_real_mount(mount: Any) -> None:
    """The half no unit test can reach: `contribute` waiting on the key.

    Every one of these rows calls `contribute`, which registers through
    `ctx.inject` rather than a `ctx.get` at `apply` time — precisely so a row
    mounted *above* the seam still lands. A `ctx.get` would work in a unit test
    that provided the registry first and fail in exactly the profile where the
    ordering happened to differ, which is the failure this asserts is absent.

    `ph-base` and nothing else, so a typo in `pyproject.toml` or a row left out
    of `base.yaml` fails here rather than in somebody's deployment.
    """
    ctx = await mount()

    rows = report_section(ctx, "Invariants")

    pollable = ("session-log", "tool-view-cache", "skill-reach-cache", "scope-unwind")
    assert set(rows) == {"model-visible-logged", *pollable}
    assert rows["model-visible-logged"].startswith("enforced inline ·")
    assert all(rows[name].startswith("holds ·") for name in pollable), rows


@pytest.mark.parametrize(
    ("row", "service", "invariant"),
    [
        ("skills-invariant", "skills", "skill-reach-cache"),
        ("tools-invariant", "tools", "tool-view-cache"),
    ],
)
async def test_a_row_whose_service_is_gone_reports_nothing_rather_than_holds(
    mount: Any, row: str, service: str, invariant: str
) -> None:
    """`inject` is what makes "absent rather than assumed" true of a *service*.

    Each of these rows checks a property *of* a registry, so dropping the
    registry leaves the row with nothing to check. Guarding on
    `ctx.get(...) is None` and returning `[]` — which all three did — printed
    `holds` about a service that is not there: the lie shaped like reassurance
    that this seam's two-kinds distinction exists to prevent, given in the
    deployment where it is least earned. An unmet `inject` key means the row
    never activates, so the invariant leaves the report along with its subject.

    `session-invariant` gained the same `inject` and is not parametrised here:
    removing the `session` row makes the profile itself incoherent — three other
    rows require `sessions` — so its overstatement was unreachable rather than
    latent. The guard is consistent, not load-bearing.
    """
    ctx = await mount({"id": service, "remove": True})

    assert invariant not in report_section(ctx, "Invariants")


async def test_an_unmounted_invariant_is_absent_rather_than_assumed(mount: Any) -> None:
    """The whole reason the registry exists, stated as a test.

    Dropping `agent-loop-invariant` means I3 is *not* enforced in this profile,
    and the report says so by not listing it. That was previously
    indistinguishable from a deployment enforcing everything `DESIGN.md`
    promises: a reader had no way to tell which of the two they were running, and
    the quieter answer was the one that looked identical to the safest.
    """
    ctx = await mount({"id": "agent-loop-invariant", "remove": True})

    assert "model-visible-logged" not in report_section(ctx, "Invariants")
