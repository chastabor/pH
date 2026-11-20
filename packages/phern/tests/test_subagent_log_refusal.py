"""P11-08, past the daemon: no host opens a sub-agent's log as a root.

The daemon refuses `session/attach` on a child's id (`NotARoot`, in `test_daemon`).
`phern -p --session <child>` and an rpc peer open sessions through `mount_session`
instead, and a child's log is its only record since Phase 11 — so a root of its own
on the same file would be a second writer there too.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from ph.paths import resolve_roots
from ph.session import Session, SessionForkError, SessionHeader
from ph.testing import log_event, stored_log, write_stored_log
from ph_app.profiles import compose_profile
from ph_app.runtime import mount_session, read_start
from ph_app.sessions import Owner, not_a_root, recorded_start

pytestmark = pytest.mark.anyio


def _stored(session_id: str, *, parent: str | None, orphan: bool = False) -> bytes:
    """One log on disk, in the lead's family: a child of `parent` when one is named.
    `orphan` makes it the log of a child whose header names no spawner."""
    family = SessionHeader(id="lead", created_at=1).family
    child = parent is not None or orphan
    session = Session(
        session_id,
        header=SessionHeader(
            id=session_id,
            created_at=1,
            family=family,
            parent_session=parent,
            origin="subagent" if child else None,
            delegation_depth=1 if child else None,
        ),
    )
    log_event(session, "workspace/checkpoint", {"agentId": session_id, "tree": "t1"})
    sessions_dir = resolve_roots().sessions_dir()
    return write_stored_log(sessions_dir, session.header, session.events).read_bytes()


async def test_a_one_shot_run_on_a_subagents_log_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before anything is mounted or claimed, naming the root to open instead,
    with a code an rpc peer can branch on — and the child's file left as it was.

    Sabotage: drop the check in `mount_session`, and the child mounts as a root.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    _stored("lead", parent=None)
    before = _stored("lead-child-1", parent="lead")

    start = await read_start("lead-child-1")
    async with AsyncExitStack() as exits:
        with pytest.raises(SessionForkError, match=r"sub-agent's log.*root lead;") as refused:
            await mount_session(exits, start, compose_profile("headless"))

    assert refused.value.code == "SESSION_IS_SUBAGENT"
    child = stored_log(resolve_roots().sessions_dir(), "lead-child-1")
    assert child.read_bytes() == before, "a refused open wrote to the child's log"


async def test_a_child_whose_header_names_no_spawner_is_refused_without_a_root_to_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three states of a stored log have three shapes, and no empty string.

    A session of its own has no owner. A child has an owner that names its root.
    A child whose header names no spawner has an owner with no root: it is still
    the log of a sub-agent, so it is still refused, and the sentence names no
    root because there is none to name.

    Sabotage: answer `None` for the owner in `recorded_start` when `_owning_root`
    finds no spawner, and the orphan mounts as a root.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    _stored("lead", parent=None)
    _stored("lead-child-1", parent="lead")
    before = _stored("lead-orphan", parent=None, orphan=True)
    sessions_dir = resolve_roots().sessions_dir()

    assert recorded_start(sessions_dir, "lead").owner is None
    assert recorded_start(sessions_dir, "lead-child-1").owner == Owner(root="lead")
    assert recorded_start(sessions_dir, "lead-orphan").owner == Owner(root=None)
    assert "root lead;" in not_a_root("lead-child-1", "lead")
    assert "the root that spawned it;" in not_a_root("lead-orphan", None)

    start = await read_start("lead-orphan")
    async with AsyncExitStack() as exits:
        with pytest.raises(SessionForkError, match="the root that spawned it") as refused:
            await mount_session(exits, start, compose_profile("headless"))

    assert refused.value.code == "SESSION_IS_SUBAGENT"
    orphan = stored_log(sessions_dir, "lead-orphan")
    assert orphan.read_bytes() == before, "a refused open wrote to the orphan's log"
