"""`utcp` — a tool whose work is long, run in the background and read from stderr.

Gates: *the footer moves and the agent is told, from the command's stderr alone;
a run that fails, and one that dies with no last word, are each said; a run
nobody asked for that changed nothing is not news; a call during a run answers
what the command says while another run holds its work, and starts no second
run.*

## The provider

A script that speaks what `ph.toolcli`'s `call` speaks to a caller that runs it
directly: `call <tool> --progress --args-json <object>` reports JSONL on stderr
and prints one envelope, and `--wait 0` answers at once when another run holds
the work. The "lock" is a marker file, and a run started with `mode: "held"`
waits for a second file before it goes on — so a test decides when a run ends
instead of racing one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from ph.agent.types import PreStepDecision, PreStepRequest
from ph.keys import AGENTS
from ph.llm.types import text_of
from ph.testing import MountProfile, new_agent, run_tool, session_of, settled
from ph_utcp import UTCP

pytestmark = pytest.mark.anyio

PROVIDER = r"""
import json, os, sys, time

VALUE = {
    "type": "object",
    "properties": {"state": {"type": "string"}, "built": {"type": "integer"}},
    "required": ["state"],
}
MANUAL = {
    "utcp_version": "1.2.0",
    "manual_version": "1.0.0",
    "tools": [
        {
            "name": name,
            "description": description,
            "inputs": {"type": "object", "properties": {"mode": {"type": "string"}}},
            "outputs": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "value": VALUE,
                    "text": {"type": "string"},
                },
            },
            "tool_call_template": {
                "call_template_type": "cli",
                "commands": [{
                    "command": f"{sys.executable} {os.path.abspath(__file__)} call {name}"
                    " --mode=UTCP_ARG_mode_UTCP_END"
                }],
                "inherit_env_vars": ["PATH", "FX_DIR"],
            },
        }
        for name, description in (("build", "Build the thing."), ("ask", "Ask the thing."))
    ],
}

if sys.argv[1] == "manual":
    print(json.dumps(MANUAL))
    sys.exit(0)

tool, rest = sys.argv[2], sys.argv[3:]
here = os.environ["FX_DIR"]
held, go, runs = (os.path.join(here, name) for name in ("held", "go", "runs"))
arguments = json.loads(rest[rest.index("--args-json") + 1]) if "--args-json" in rest else {}
mode = arguments.get("mode")
for one in rest:
    if one.startswith("--mode=") and not one.split("=", 1)[1].startswith("MISSING_ARG_"):
        mode = one.split("=", 1)[1]


def answer(value, text):
    print(json.dumps({"ok": True, "value": value, "text": text}))


def say(**line):
    if "--progress" in rest:
        print(json.dumps(line), file=sys.stderr, flush=True)


if tool == "ask":
    answer({"state": "indexing" if os.path.exists(held) else "done"}, "asked")
    sys.exit(0)
if os.path.exists(held):
    # Another run holds the work. A run of its own waits out its `wait_s`
    # first — here, until told to go — and ends `busy`; a call that will not
    # wait says so at once.
    if mode == "late" and "--progress" in rest:
        while not os.path.exists(go):
            time.sleep(0.02)
    say(state="busy", done=0, total=0, what="things")
    answer({"state": "indexing"}, "Another run is building the thing.")
    sys.exit(0)
open(held, "w").close()
with open(runs, "a") as log:
    log.write(f"{mode}\n")
try:
    say(state="working", done=0, total=2, what="things")
    if mode == "held":
        say(state="working", done=1, total=2, what="things")
        while not os.path.exists(go):
            time.sleep(0.02)
    if mode == "die":
        os.remove(held)
        print("the parser fell over", file=sys.stderr, flush=True)
        os._exit(7)
    if mode == "fail":
        say(state="failed", done=1, total=2, what="things", code="FX_FAILED",
            failure="the thing would not build")
        print(json.dumps({"ok": False, "error": {"code": "FX_FAILED",
                                                 "message": "the thing would not build"}}))
        sys.exit(1)
    say(state="done", done=2, total=2, what="things", changed=mode != "same",
        note="Built 2 things, 0 skipped.", spent={"requests": 2, "tokens": 1500})
    answer({"state": "done", "built": 2}, "Built 2 things, 0 skipped.")
finally:
    if os.path.exists(held):
        os.remove(held)
"""


def _row(
    tmp_path: Path,
    *,
    eager: bool = False,
    wait_s: float = 30.0,
    program: str | None = None,
) -> dict[str, Any]:
    """The `utcp` row with the provider's manual, its `build` in the background
    — run by `program` when one is named, as a manual that names the wrong one
    would be."""
    home = _home(tmp_path)
    home.mkdir(exist_ok=True)
    script = home / "provider.py"
    script.write_text(PROVIDER, encoding="utf-8")
    command = [sys.executable, str(script)]
    manual = {
        "template": {
            "name": "fx",
            "call_template_type": "cli",
            "commands": [{"command": f"{command[0]} {command[1]} manual"}],
            "inherit_env_vars": ["PATH"],
        },
        "envelope": True,
        "env": {"FX_DIR": str(home)},
        "background": {
            "command": [program, "call"] if program is not None else [*command, "call"],
            "label": "thing",
            "tools": ["build"],
            "eager": eager,
            "wait_s": wait_s,
        },
    }
    return {"id": "utcp", "name": "utcp", "config": {"manuals": [manual]}}


def _home(tmp_path: Path) -> Path:
    return tmp_path.parent / f"{tmp_path.name}-provider"


def _started(tmp_path: Path) -> list[str]:
    """The runs that did the work, by the mode each was asked with."""
    runs = _home(tmp_path) / "runs"
    return runs.read_text(encoding="utf-8").split() if runs.exists() else []


async def _first_step(ctx: Any, agent: Any) -> None:  # noqa: ANN401
    async def entered() -> PreStepDecision:
        return PreStepDecision(kind="enter")

    request = PreStepRequest(agent=agent, session=session_of(agent), messages=(), turn=1, step=1)
    await ctx.require(UTCP).eager(request, entered)


def _notes(agent: Any) -> list[str]:  # noqa: ANN401
    return [text_of(note.content) for note in agent.inbox.next_step]


async def test_a_call_that_its_run_outlasts_is_answered_and_the_agent_told_when_it_ends(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** The call starts the run and waits `wait_s` for it. Past
    that it answers what the command says while a run holds its work, with how
    far the run has got — which only this row knows, having read it from the
    run's stderr — and the footer says the same. When the run ends the agent is
    told, with the run's own words, by a note that waits for its next step."""
    ctx = await mount(_row(tmp_path, wait_s=0.5))
    agent = new_agent(ctx)
    utcp = ctx.require(UTCP)

    asked = await run_tool(ctx, "fx_build", {"mode": "held"}, agent=agent)

    assert asked.value == {"state": "indexing"}
    said = text_of(asked.content)
    assert "Another run is building the thing." in said
    assert "[The thing is still working: 1/2 things.]" in said
    reading = utcp.reading(session_of(agent))
    assert reading is not None and reading.text == "thing 1/2 things"
    assert _notes(agent) == []

    (_home(tmp_path) / "go").touch()
    await settled(lambda: bool(agent.inbox.next_step), "the completion note", seconds=30)

    [note] = _notes(agent)
    assert "The thing is finished: 2 things" in note
    assert "Built 2 things, 0 skipped." in note
    assert not utcp.runs and utcp.reading(session_of(agent)) is None
    assert _started(tmp_path) == ["held"], "the call that was answered started no second run"


async def test_a_run_whose_job_is_canceled_before_it_starts_is_not_left_to_join(
    mount: MountProfile, tmp_path: Path
) -> None:
    """S6. A run its job never reached is ended, and taken off the table.

    The run was registered before its job started, so a later call would join it
    rather than start a second, and it was ended only inside the job's body: a job
    canceled before that body ran left a run whose `ended` never set, which every
    later call joined, waited out `wait_s` on, and no run was started again.

    Sabotage: end the runs in the body alone, and the run is still on the table.
    """
    ctx = await mount(_row(tmp_path))
    agent = new_agent(ctx)
    utcp = ctx.require(UTCP)

    run = await utcp._start("fx", utcp.manuals["fx"], agent, [("build", {"mode": "held"})])
    assert run is not None and run.job is not None
    run.job.cancel()

    await settled(lambda: run.ended.is_set(), "the run that never ran to end", seconds=10)
    assert not utcp.runs
    assert _started(tmp_path) == []


async def test_a_run_that_ends_within_the_wait_answers_the_call_and_nobody_is_told_twice(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The ordinary case: the run is short, the call answers with the run's own
    result as any call would, and there is no note — its caller has the counts."""
    ctx = await mount(_row(tmp_path))
    agent = new_agent(ctx)

    built = await run_tool(ctx, "fx_build", {}, agent=agent)

    assert built.value == {"state": "done", "built": 2}
    assert text_of(built.content) == "Built 2 things, 0 skipped."
    assert _notes(agent) == []
    assert _started(tmp_path) == ["None"]


async def test_a_run_that_fails_is_said_in_the_tools_own_words(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A caller that waited gets the failure as the tool raised it. An agent
    whose run nobody waited for is told the same sentence, from the last line."""
    ctx = await mount(_row(tmp_path, eager=True))
    agent = new_agent(ctx)

    failed = await run_tool(ctx, "fx_build", {"mode": "fail"}, agent=agent)

    assert failed.is_error and failed.error is not None
    assert (failed.error.info or {}).get("code") == "FX_FAILED"
    assert _notes(agent) == []

    utcp, unwaited = ctx.require(UTCP), new_agent(ctx)
    [manual] = utcp.manuals.values()
    await utcp._start("fx", manual, unwaited, [("build", {"mode": "fail"})])
    await settled(lambda: bool(unwaited.inbox.next_step), "the note", seconds=30)

    [note] = _notes(unwaited)
    assert "The thing stopped: the thing would not build" in note
    assert "It got through 1 of 2 things." in note


async def test_a_run_that_dies_without_a_last_line_is_said_as_that(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** A command killed, out of memory or crashed below Python
    writes no last line and prints no envelope. The stream ending is the
    signal: the agent is told the run stopped, with the exit code, how far it
    had got and the last thing it wrote."""
    ctx = await mount(_row(tmp_path, wait_s=0.2))
    agent = new_agent(ctx)
    utcp = ctx.require(UTCP)
    [manual] = utcp.manuals.values()

    await utcp._start("fx", manual, agent, [("build", {"mode": "die"})])
    await settled(lambda: bool(agent.inbox.next_step), "the note", seconds=30)

    [note] = _notes(agent)
    assert "The thing stopped: its command ended (exit 7) without finishing" in note
    assert "the parser fell over" in note
    assert "It got through 0 of 2 things." in note
    assert not utcp.runs


async def test_a_run_that_found_its_work_held_by_another_tells_a_waiting_agent_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A run that waited behind another run — one of another root, a person's
    command — and did nothing is not "finished". An agent answered "still
    going" is told the run did not happen and to call again; the other run's
    end is not this row's to report."""
    ctx = await mount(_row(tmp_path, wait_s=0.3))
    agent = new_agent(ctx)
    (_home(tmp_path) / "held").touch()

    asked = await run_tool(ctx, "fx_build", {"mode": "late"}, agent=agent)
    (_home(tmp_path) / "go").touch()
    await settled(lambda: bool(agent.inbox.next_step), "the note", seconds=30)

    assert asked.value == {"state": "indexing"}
    [note] = _notes(agent)
    assert "The thing did not run: another run holds what it builds" in note
    assert "Call its tool again once that run has finished." in note
    assert "finished: " not in note


async def test_a_run_whose_command_cannot_be_started_is_said_and_not_lost(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** A command that never runs — a program that is not there, a
    sandbox that refuses it — writes no stderr and leaves no result, so nothing
    the run reports can say so. Started unasked before the first step, it is
    said to the agent with why; asked for by a call, the call fails with it."""
    missing = str(tmp_path.parent / "no-such-program")
    ctx = await mount(_row(tmp_path, eager=True, program=missing))
    agent, caller = new_agent(ctx), new_agent(ctx)

    await _first_step(ctx, agent)
    await settled(lambda: bool(agent.inbox.next_step), "the note", seconds=30)
    called = await run_tool(ctx, "fx_build", {}, agent=caller)

    [note] = _notes(agent)
    assert "The thing could not be started:" in note and "no-such-program" in note
    assert "calling its tool tries again" in note
    assert called.is_error and called.error is not None
    assert "fx.build could not be started:" in called.error.message
    assert _notes(caller) == [], "the call that failed with it is not told twice"
    assert not ctx.require(UTCP).runs


async def test_a_root_agents_first_step_starts_the_run_and_only_a_change_is_news(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Unasked, before the first step, and once: an index is being built before
    anything asks for it. The agent is told when that changed something. A run
    that found everything current — every restart of a session over a tree that
    has not moved — says nothing."""
    ctx = await mount(_row(tmp_path, eager=True))
    agent, utcp = new_agent(ctx), ctx.require(UTCP)

    await _first_step(ctx, agent)
    await settled(lambda: bool(agent.inbox.next_step), "the completion note", seconds=30)
    await _first_step(ctx, agent)

    assert ["The thing is finished: 2 things" in note for note in _notes(agent)] == [True]
    assert _started(tmp_path) == ["None"], "started once, not at every step"

    quiet = new_agent(ctx)
    [manual] = utcp.manuals.values()
    run = await utcp._start("fx", manual, quiet, [("build", {"mode": "same"})])
    assert run is not None
    await run.ended.wait()

    assert _notes(quiet) == []


async def test_a_warm_mark_goes_with_its_agent(mount: MountProfile, tmp_path: Path) -> None:
    """The mark that a tree was warmed is its owner's: kept past the agent, the set
    held one per root agent the process ever stepped.

    Sabotage: drop the disposer in `_warm`, and the mark outlives the agent.
    """
    ctx = await mount(_row(tmp_path, eager=True))
    agent, utcp = new_agent(ctx), ctx.require(UTCP)
    await _first_step(ctx, agent)
    await settled(lambda: bool(agent.inbox.next_step), "the completion note", seconds=30)
    assert set(utcp.warmed) == {("fx", agent.id)}

    await ctx.require(AGENTS).dispose(agent.id)

    assert utcp.warmed == {}


async def test_a_manual_with_eager_off_starts_nothing_until_its_tool_is_called(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_row(tmp_path, eager=False))
    agent = new_agent(ctx)

    await _first_step(ctx, agent)

    assert not ctx.require(UTCP).runs and _started(tmp_path) == []


async def test_another_tool_of_the_manual_is_an_ordinary_call_during_a_run(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A search during an indexing run: called as any tool is, answered by the
    command — which knows a run holds its work — with the count added here."""
    ctx = await mount(_row(tmp_path, wait_s=0.3))
    agent = new_agent(ctx)
    await run_tool(ctx, "fx_build", {"mode": "held"}, agent=agent)

    asked = await run_tool(ctx, "fx_ask", {}, agent=agent)
    (_home(tmp_path) / "go").touch()

    assert asked.value == {"state": "indexing"}
    assert text_of(asked.content) == "asked\n[The thing is still working: 1/2 things.]"
