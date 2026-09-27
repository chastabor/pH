"""`ph.session_profile` — whose log a base is recorded in (session profiles, S3).

A root's. A child runs on its root's mount and a fork continues its root's log, so
both are answered by the root's base, and neither records one of its own: a
second base in a family would be a second answer to "what did this run in".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ph.keys import SESSIONS
from ph.session_profile import BASE, record_base, saved_base
from ph.testing import MountProfile

pytestmark = pytest.mark.anyio


async def test_a_root_records_its_base_and_a_child_or_fork_does_not(mount: MountProfile) -> None:
    """Sabotage: drop the `parent_session` check, and the child records a base of
    its own — the fork too, over the one it inherited."""
    ctx = await mount()
    sessions = ctx.require(SESSIONS)
    root = sessions.create("root")
    child = sessions.create("child", meta={"parent_session": "root"})

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
