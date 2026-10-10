"""P11-08, past the daemon: no host opens a sub-agent's log as a root.

The daemon refuses `session/attach` on a child's id (`NotARoot`, in `test_daemon`).
`phern -p --session <child>` and an rpc peer open sessions through `mount_session`
instead, and a child's log is its only record since Phase 11 — so a root of its own
on the same file would be a second writer there too.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from ph.json import JsonObject
from ph.paths import resolve_roots
from ph.persistence.jsonl import session_path
from ph.session import Session, SessionForkError, SessionHeader
from ph.testing import log_event
from ph_app.profiles import compose_profile
from ph_app.runtime import mount_session, read_start

pytestmark = pytest.mark.anyio


def _stored(session_id: str, *, parent: str | None) -> bytes:
    """One log on disk, in the lead's family: a child of `parent` when one is named."""
    family = SessionHeader(id="lead", created_at=1).family
    session = Session(
        session_id,
        header=SessionHeader(
            id=session_id,
            created_at=1,
            family=family,
            parent_session=parent,
            origin="subagent" if parent else None,
            delegation_depth=1 if parent else None,
        ),
    )
    log_event(session, "workspace/checkpoint", {"agentId": session_id, "tree": "t1"})
    lines: list[JsonObject] = [{"type": "session/header", "header": session.header.to_wire()}]
    lines += [event.to_wire() for event in session.events]
    path = session_path(resolve_roots().sessions_dir(), session_id, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines), encoding="utf-8")
    return path.read_bytes()


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
    family = SessionHeader(id="lead", created_at=1).family
    child = session_path(resolve_roots().sessions_dir(), "lead-child-1", family)
    assert child.read_bytes() == before, "a refused open wrote to the child's log"
