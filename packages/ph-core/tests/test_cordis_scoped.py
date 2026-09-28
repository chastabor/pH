"""Registrations bucketed by isolation scope (`ph.cordis.scoped`).

The one shape the tool registry's layers, the skill registry's restrictions and the
sandbox's path limits share: keyed by the registration's layer, released by the
disposer the scope holds, the bucket gone with its last entry, read along a chain.
"""

from __future__ import annotations

import pytest

from ph.cordis import Context, ScopedEntries, ScopedTable
from ph.seams.skills import SkillRestriction
from ph.testing import skill, skill_service

pytestmark = pytest.mark.anyio


async def test_a_bucket_goes_with_its_last_entry_and_the_release_says_so() -> None:
    """Sabotage: keep an emptied bucket in `claim`'s release, and the settled scope
    stays keyed; run `then` outside the disposer the scope holds, and an unwinding
    scope announces nothing."""
    root = Context()
    child = root.scope("child")
    table: ScopedEntries[str] = ScopedEntries()
    told: list[str] = []

    table.add(root.running_for(child), "a", label="t", then=lambda: told.append("a"))
    table.add(root.running_for(child), "b", label="t", then=lambda: told.append("b"))
    assert table.gathered([child, None]) == ["a", "b"]
    await child.dispose()

    assert not table and len(table) == 0
    assert sorted(told) == ["a", "b"], "each release said so as the scope unwound"


def test_a_value_is_removed_by_identity_not_equality() -> None:
    """Two rows contributing an equal value: each release takes its own. Sabotage:
    compare with `==`, and releasing the second takes the first."""
    root = Context()
    table: ScopedEntries[list[int]] = ScopedEntries()
    first, second = [1], [1]

    table.add(root.running_for(None), first, label="t")
    release = table.add(root.running_for(None), second, label="t")
    release()

    (left,) = table.gathered([None])
    assert left is first


def test_a_refused_registration_leaves_no_bucket_behind() -> None:
    """A duplicate name refused inside `mutate` — the tool registry's case. Sabotage:
    drop the `except` in `claim`, and an empty bucket stays keyed by the scope."""
    root = Context()
    child = root.scope("child")
    table: ScopedTable[dict[str, int]] = ScopedTable(dict, lambda bucket: not bucket)

    def refuse(_bucket: dict[str, int]) -> None:
        raise ValueError("already registered")

    with pytest.raises(ValueError, match="already registered"):
        table.claim(root.running_for(child), refuse, lambda _bucket: None, label="t")

    assert table.get(child) is None


def test_a_chain_reads_its_buckets_in_its_own_order() -> None:
    root = Context()
    parent = root.scope("parent")
    child = parent.scope("child")
    table: ScopedEntries[str] = ScopedEntries()
    table.add(root.running_for(None), "global", label="t")
    table.add(root.running_for(parent), "parent", label="t")
    table.add(root.running_for(child), "child", label="t")

    chain = child.isolation_chain()
    assert table.gathered(chain) == ["child", "parent", "global"]
    assert table.gathered(reversed(chain)) == ["global", "parent", "child"]


async def test_a_child_s_skill_filter_lifted_by_its_scope_refreshes_the_catalog() -> None:
    """The skill registry's restriction released by the child's scope, not by hand,
    tells the registry, so no cached reach outlives the filter. Sabotage: pass no
    `then` to `ScopedEntries.add` in `SkillService.restrict`, and nothing moves."""
    root, skills = skill_service()
    skills.register(skill("read-code"))
    child = root.scope("child")
    skills.restrict(SkillRestriction(deny=frozenset({"read-code"})), scope=child)
    before = skills._generation

    await child.dispose()

    assert skills._generation > before
