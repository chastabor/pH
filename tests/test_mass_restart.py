"""T1 — the durability goal, as a gate: everything stops at once, and everything comes back.

Several sub-agents are working, each with its own session log. The process holding
them is killed — `SIGKILL`, so no teardown of ours runs at all — and a fresh process
opens the same `$PH_HOME` and does what the daemon does when it starts a root:
`resume_session`, then `resume_children` (`Supervisor._start`). Every sub-agent must
pick up where it left off, and every log must read back whole.

`mass_restart_host.py` is the process that is killed. It leaves one of each kind of
in-flight work on disk: a turn whose `write` landed with no result recorded, a `!!`
command, an approval nobody answered, a keyed effect mid-body, a sub-agent at the
model and one queued behind it. What this file asserts is what a restart makes of
them — the property `plans/Contained_Log_Writes_Todo.md` is weighed against:

    in whichever process resumes them, every intent in every log is settled or
    reconciled, every consumer reads what happened the same way, and reopening
    again changes nothing.

Written first so the rows after it (T2 to T6) each tighten it; where today falls short
the test says so with a strict `xfail`, which starts failing the day it is fixed.

T5's half runs the same crash with the sub-agents on a route whose credential the
restarting process lacks: they are held — not failed, no rung of their ladder spent —
while everything else resumes, supplying the key releases them, and a second restart
before it arrives holds them again and appends nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.agent.types import AgentDriver, AgentOptions
from ph.cordis import Context
from ph.json import as_obj, as_seq
from ph.keys import AGENTS, CREDENTIALS, LLM, SESSION_PERSISTENCE, SESSIONS, SUBAGENTS
from ph.llm.adapter import ResolvedModel
from ph.llm.fake import FakeAdapter, text_script
from ph.llm.types import (
    content_from_wire,
    text_of,
)
from ph.persistence import interrupted_turn_closers, resume_session
from ph.persistence.lease import SessionBusy, claim_session
from ph.seams.credentials import waiting_for
from ph.seams.subagents import subagent_roster
from ph.session import Session, SessionEvent, declared_intents, open_intents, outcome_of
from ph.session.kinds import APPROVAL_ASK, SHELL_COMMAND, TOOL_EFFECT
from ph.testing import MountProfile, not_none, stored_types

pytestmark = pytest.mark.anyio

HOST = Path(__file__).with_name("mass_restart_host.py")
PROVIDER_ROW: dict[str, Any] = {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"}
SCRIPTED = AgentOptions(provider="scripted", model="s1")
RETRIES = 3
KEY = "PH_T5_CHILD_KEY"
"""What the `keyed` route's adapter resolves at its edge in the restarted process."""


def _killed_mid_work(project: Path, *mode: str) -> dict[str, Any]:
    """Run the host until it says everything is in flight, then `SIGKILL` it."""
    host = subprocess.Popen(
        [sys.executable, str(HOST), str(project), *mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    try:
        assert host.stdout is not None and host.stderr is not None
        line = host.stdout.readline()
        if not line:
            host.wait(timeout=10)
            raise AssertionError(f"the host never got its work in flight:\n{host.stderr.read()}")
        ids: dict[str, Any] = json.loads(line)
        host.send_signal(signal.SIGKILL)
        host.wait(timeout=10)
        return ids
    finally:
        if host.poll() is None:  # pragma: no cover
            host.kill()
            host.wait()


async def _restart(mount: MountProfile) -> tuple[Context, Session]:
    """What `Supervisor._start` does for a root: resume it, then owe its children."""
    ctx, root, _parent = await _restarted(mount)
    return ctx, root


async def _restarted(mount: MountProfile) -> tuple[Context, Session, AgentDriver]:
    """`_restart`, with the parent agent a credential's arrival re-asks about."""
    ctx = await mount(dict(PROVIDER_ROW, config={"maxConcurrent": 2}))
    # The restarted deployment's model answers every request, briefly — on the
    # `keyed` route, one whose adapter names `KEY` at its edge.
    ctx.require(LLM).register_adapter(("scripted",), FakeAdapter(respond=text_script("done")))
    ctx.require(LLM).register_adapter(
        ("keyed",),
        FakeAdapter(respond=text_script("done"), route=ResolvedModel(credential=KEY)),
    )
    root = await resume_session(ctx, "root")
    parent = ctx.require(AGENTS).create(root, SCRIPTED)
    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=RETRIES)
    return ctx, root, parent


async def _until_settled(root: Session, *runs: str) -> None:
    with anyio.fail_after(20):
        while any(subagent_roster(root)[run].get("status") != "done" for run in runs):
            await anyio.sleep(0.02)


def _open(events: Any) -> list[str]:  # noqa: ANN401
    """Every intent a log holds open, of any declared kind, by its opening type."""
    return [
        f"{kind.opened} {intent.key}"
        for kind in declared_intents()
        for intent in open_intents(events, kind)
    ]


def _latest(session: Session, event_type: str) -> SessionEvent:
    return not_none(session.latest(event_type), f"no {event_type} in {session.id}")


async def _flushed(ctx: Context) -> None:
    sessions = ctx.require(SESSIONS)
    for session in sessions.list():
        await sessions.flush(session)


@pytest.fixture
def crashed(tmp_path: Path) -> dict[str, Any]:
    return _killed_mid_work(tmp_path)


@pytest.fixture
def crashed_keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The same crash, with the sub-agents on the `keyed` route — and no `KEY` here."""
    monkeypatch.delenv(KEY, raising=False)
    return _killed_mid_work(tmp_path, "keyed")


async def test_every_sub_agent_picks_up_where_it_left_off(
    crashed: dict[str, Any], mount: MountProfile
) -> None:
    """The sub-agents: the one at the model is put back on its ladder with its task
    re-presented, the queued one is driven for the first time, and both finish."""
    ctx, root = await _restart(mount)
    running, queued = crashed["running"]["run"], crashed["queued"]["run"]

    await _until_settled(root, running, queued)

    roster = subagent_roster(root)
    assert roster[running]["starts"] == 2, "one first run, one after the restart"
    assert "resumes" not in roster[queued], "a child that never ran is a first attempt"
    assert len(roster) == 2, "no child invented, none lost"
    child = ctx.require(SESSIONS).get(crashed["running"]["session"])
    assert child is not None and child.latest("session/resumed") is not None, (
        "the child's own log came off disk rather than being made again"
    )


async def test_every_intent_in_the_root_is_settled_or_reconciled(
    crashed: dict[str, Any], mount: MountProfile
) -> None:
    """The root's in-flight work, each answered in the way its kind says."""
    _ctx, root = await _restart(mount)

    assert outcome_of(SHELL_COMMAND, _latest(root, "shell/result")) == "outcome-unknown"
    approval = _latest(root, "approval/decided")
    assert (approval.data["outcome"], approval.data["automatic"]) == ("interrupted", True)
    assert outcome_of(TOOL_EFFECT, _latest(root, "tool/effect-settled")) == "outcome-unknown"
    assert outcome_of(APPROVAL_ASK, approval) == "outcome-unknown"
    write = _latest(root, "tool/result")
    block = as_obj(as_seq(as_obj(write.data["message"])["content"])[0])
    assert block["isError"] is False, "the file says the write landed, so it did"
    assert text_of(content_from_wire(block["content"])).startswith("Wrote notes.md")
    assert as_obj(write.data["meta"])["reconciled"] is True
    assert as_obj(_latest(root, "turn/end").data["reason"])["kind"] == "interrupted"
    assert _latest(root, "session/resumed").data["interrupted"] is True
    assert _open(root.events) == []


async def test_every_log_reads_back_whole_and_a_second_restart_changes_nothing(
    crashed: dict[str, Any], mount: MountProfile
) -> None:
    """No log is refused, none holds an open intent, and reopening them all again —
    what a second crash before anybody noticed would do — closes nothing."""
    ctx, root = await _restart(mount)
    await _until_settled(root, crashed["running"]["run"], crashed["queued"]["run"])
    await _flushed(ctx)

    store = ctx.require(SESSION_PERSISTENCE)
    for session_id in (
        "root",
        crashed["running"]["session"],
        crashed["queued"]["session"],
    ):
        _header, events = store.read(session_id)
        assert _open(events) == [], session_id
        assert interrupted_turn_closers(events) == [], f"{session_id} would be repaired again"

    again = await mount(dict(PROVIDER_ROW, config={"maxConcurrent": 2}))
    revived = await resume_session(again, "root")
    assert revived.events[-1].data["closed"] == 0, "the second restart closed something"


# ---------------------------------------------------------------- T5 --


async def test_a_sub_agent_whose_key_a_restart_lost_is_held_then_released(
    crashed_keyed: dict[str, Any], mount: MountProfile
) -> None:
    """Held by name, not failed, and no retry spent; the rest resumes; supplying the
    key puts both back to work, the interrupted one on its second start."""
    ctx, root, parent = await _restarted(mount)
    running, queued = crashed_keyed["running"]["run"], crashed_keyed["queued"]["run"]

    roster = subagent_roster(root)
    assert (roster[running]["status"], roster[queued]["status"]) == ("queued", "queued")
    assert roster[running]["starts"] == 1, "held, so no rung of its ladder spent"
    assert "resumes" not in roster[queued]
    assert waiting_for(ctx, root) == {running: KEY, queued: KEY}
    assert _latest(root, "session/resumed").data["interrupted"] is True, "the root resumed"
    assert [one.split()[0] for one in _open(root.events)] == ["credential/needed"] * 2, (
        "everything else the crash left open was settled as before"
    )

    ctx.require(CREDENTIALS).provide_value(KEY, "supplied")
    revived = await ctx.require(SUBAGENTS).readmit_waiting(parent, retry_limit=RETRIES)
    await _until_settled(root, running, queued)

    assert sorted(revived) == sorted([running, queued])
    assert waiting_for(ctx, root) == {}
    assert subagent_roster(root)[running]["starts"] == 2, "its one restart, counted once"


async def test_a_second_restart_before_the_key_arrives_holds_again_and_grows_nothing(
    crashed_keyed: dict[str, Any], mount: MountProfile
) -> None:
    """What a second power cut before anybody noticed does: the holds are still in
    the log, the check finds them, and nothing but the resume's own record is added."""
    first, _root = await _restart(mount)
    await _flushed(first)
    before = stored_types(first, "root")

    again, root = await _restart(mount)
    await _flushed(again)
    grown = stored_types(again, "root")[len(before) :]

    assert grown == ["session/end-seed", "session/resumed"], grown
    running, queued = crashed_keyed["running"]["run"], crashed_keyed["queued"]["run"]
    assert waiting_for(again, root) == {running: KEY, queued: KEY}
    assert subagent_roster(root)[running]["starts"] == 1


async def test_a_readmitted_childs_log_is_leased_by_the_process_that_resumed_it(
    crashed: dict[str, Any], mount: MountProfile
) -> None:
    """L2, closed. A child's log is a session of its own, openable by its id, so the
    process that resumed it must hold its lease as a root's is held (I-5) — or a
    second process opening the child while this one drives it makes two writers on
    one log. T1 recorded this as a strict `xfail`; `_child_session` now opens the
    child through `open_session`, which claims it.

    Sabotage: open the child with `resume_session` directly again, and the second
    claim below is granted.
    """
    ctx, root = await _restart(mount)
    await _until_settled(root, crashed["running"]["run"], crashed["queued"]["run"])
    store = ctx.require(SESSION_PERSISTENCE)

    with pytest.raises(SessionBusy):
        await claim_session(Context(), store.root, crashed["running"]["session"])  # type: ignore[attr-defined]
