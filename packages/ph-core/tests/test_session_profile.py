"""`ph.session_profile` — whose log a base is recorded in (session profiles, S3), and
what a log's profile records fold to once a base can change (S6).

A root's. A child runs on its root's mount and a fork continues its root's log, so
both are answered by the root's base, and neither records one of its own: a
second base in a family would be a second answer to "what did this run in".

A base switch keeps the session's overrides (decision 5) and clears, in the same
batch, the ones the new base already says (decision 3); `fold_environment` is the
one reading of all of it, and `listing` is what a person is shown before deciding.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ph.json import JsonObject, JsonValue, as_obj
from ph.keys import SESSIONS
from ph.session_profile import (
    ADOPTED,
    BASE,
    CLEARED,
    DECLINED,
    OVERRIDE,
    Difference,
    Override,
    ProfileBase,
    ProfileChange,
    fold_environment,
    listing,
    logged_environment,
    record_base,
    saved_base,
    switch_base,
)
from ph.testing import MountProfile, log_event, not_none

pytestmark = pytest.mark.anyio


async def test_a_root_records_its_base_and_a_child_or_fork_does_not(mount: MountProfile) -> None:
    """A child is a sub-agent's log (`origin`); a fork names its parent too, and is a
    session of its own that already holds its root's base. Sabotage: drop the
    `is_subagent` check, and the child records a base of its own."""
    ctx = await mount()
    sessions = ctx.require(SESSIONS)
    root = sessions.create("root")
    child = sessions.create("child", meta={"parent_session": "root", "origin": "subagent"})

    recorded = await record_base(ctx, root)
    fork = sessions.fork(root, child_session_id="fork")

    assert recorded is not None and saved_base(root) == recorded
    assert await record_base(ctx, child) is None
    assert not [event for event in child.events if event.type == BASE]
    assert await record_base(ctx, fork) is None
    assert saved_base(fork) == recorded, "a fork inherits its root's base with the prefix"
    assert [event.type for event in fork.events].count(BASE) == 1


def test_every_reconfigure_goes_through_the_door() -> None:
    """S4's gate, made structural: `Mount.reconfigure` is called from
    `ph.session_profile` and nowhere else in shipped code, so no command can change a
    row without the session's log saying so first. The shape `test_log_writers.py`
    holds the log's one door to. Sabotage: call `mount.reconfigure` from `/sandbox`
    directly, and this names the file."""
    packages = Path(__file__).resolve().parents[2]
    callers = sorted(
        str(path.relative_to(packages))
        for path in packages.glob("*/src/**/*.py")
        if ".reconfigure(" in path.read_text(encoding="utf-8")
    )

    assert callers == ["ph-core/src/ph/session_profile.py"], callers


def _base(name: str) -> JsonObject:
    return {"name": name, "rows": [], "sources": [], "phVersion": "0"}


def _override(row: str, command: str) -> JsonObject:
    return Override(row, {"id": row, "config": {}}, "command", command).to_wire()


def test_an_override_outlives_a_base_switch_until_its_row_is_cleared() -> None:
    """Swapping base profiles keeps the overrides (decision 5), so the fold reads every
    override since the first base, less the ones a clear names. A new base settles
    what was adopted or declined against the old one. Sabotage: reset the overrides
    at each base, as S4's fold did, and `/one` is lost at the switch."""
    records: list[tuple[str, JsonObject]] = [
        (BASE, _base("first")),
        (OVERRIDE, _override("x", "/one")),
        (DECLINED, _base("offered")),
        (ADOPTED, _base("second")),
        (BASE, _base("second")),
        (OVERRIDE, _override("y", "/two")),
        (OVERRIDE, _override("z", "/three")),
        (CLEARED, {"row": "z", "command": "adopt second"}),
    ]

    env = fold_environment(records)

    assert not_none(env.base).name == "second"
    assert [one.command for one in env.overrides] == ["/one", "/two"]
    assert env.adopted is None and env.declined is None
    pending = fold_environment([*records, (ADOPTED, _base("third"))])
    assert not_none(pending.starts_on).name == "third", "an adopted version is what starts next"


def _with_config(base: ProfileBase, row_id: str, config: JsonValue) -> ProfileBase:
    rows = tuple({**row, "config": config} if row.get("id") == row_id else row for row in base.rows)
    return replace(base, rows=rows)


async def test_a_base_switch_clears_the_overrides_the_new_base_already_says(
    mount: MountProfile,
) -> None:
    """Decision 3's own rule at a switch: an override the new base already says is no
    deviation, so it is cleared — compared as settings, through the row's model,
    though the override was written sparse and the base states every default. One
    that still deviates is kept. Sabotage: compare the raw entries in `_said_by`,
    and the sparse override is never cleared."""
    ctx = await mount()
    session = ctx.require(SESSIONS).create("switched")
    await record_base(ctx, session)
    saved = not_none(saved_base(session))
    log_event(
        session,
        OVERRIDE,
        Override(
            "llm-retry", {"id": "llm-retry", "config": {"maxAttempts": 5}}, "command", "/retry 5"
        ).to_wire(),
    )
    log_event(
        session,
        OVERRIDE,
        Override("tool-bash", {"id": "tool-bash", "disabled": True}, "cli", "--patch").to_wire(),
    )
    retry = next(as_obj(row) for row in saved.rows if row.get("id") == "llm-retry")
    adopted = _with_config(saved, "llm-retry", {**as_obj(retry["config"]), "maxAttempts": 5})

    cleared = await switch_base(ctx, session, adopted, command="adopt headless")

    assert cleared == ["llm-retry"]
    env = logged_environment(session)
    assert env.base == adopted
    assert [one.row for one in env.overrides] == ["tool-bash"]
    switch = [event for event in session.events if event.type in (BASE, CLEARED)][-2:]
    assert [event.type for event in switch] == [BASE, CLEARED]
    assert switch[0].batch is not None and switch[0].batch == switch[1].batch, "one batch"


def test_the_listing_says_each_setting_s_old_and_new_value_and_whose_it_is() -> None:
    """Decision 13: what a person reads before deciding. A list by what was added and
    removed, the rest old → new; the person's edits as theirs and pH's by version;
    then the overrides that still apply over a changed row."""
    was = ProfileBase("work", (), (), "0.4.0")
    now = ProfileBase("work", (), (), "0.5.0")
    change = ProfileChange(
        was=was,
        now=now,
        differences=(
            Difference("sandbox-allow", "config.network.hosts", ["a", "b"], ["a", "c"], "person"),
            Difference("llm-retry", "config.maxAttempts", 3, 5, "pH"),
            Difference("tool-x", "", None, {"id": "tool-x"}, "person"),
        ),
        shadowed=(
            Override("sandbox-allow", {"id": "sandbox-allow"}, "command", "/sandbox allow host d"),
        ),
        declined=False,
    )

    lines = listing(change)

    assert lines[0] == "work has changed since this session started:"
    assert "+ c, - b" in lines[1] and lines[1].endswith("(your profile)")
    assert "3 → 5" in lines[2] and lines[2].endswith("(pH 0.4.0 → 0.5.0)")
    assert "(the row)" in lines[3] and "added" in lines[3]
    assert lines[4] == "Still applied over it, from this session:"
    assert lines[5].split() == ["sandbox-allow", "/sandbox", "allow", "host", "d"]
