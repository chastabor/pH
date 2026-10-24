"""The first half of `test_daemon_restart.py`: a daemon caught mid-work, then killed.

`mass_restart_host.py` mounts a profile and leaves work in flight beneath it, and
its test restarts with the core's own resume. This host runs the daemon itself —
`serve`, its supervisor, its scheduler and the live-roots index it keeps — so what
is killed is what an orchestrator kills, and what the test brings up after it is a
second daemon on the same `$PH_HOME` (Phase 13's gates).

Run as its own process — `python tests/daemon_restart_host.py <mode>`, with the
test's `$PH_HOME` and `$PH_RUNTIME` — so the test can `SIGKILL` it. Every model
request waits to be killed. Each mode puts one shape of work in flight through the
daemon's own doors, waits until the log and the index the next daemon reads hold
it, prints the ids the test needs as one JSON line, and waits:

* `goal`: a root mid-turn on an open goal, with a sub-agent at the model;
* `wake`: an idle root whose sub-agent is at the model;
* `turn`: a root mid-turn on a person's prompt, and nothing else;
* `tick-claim`: a schedule's tick claimed and its prompt not yet delivered;
* `tick-write`: the tick and its prompt written, the index not yet told, and the
  turn they woke not yet begun;
* `resume`: no work of its own — the `--resume-owed` pass over what the last
  daemon left, and the kill lands the moment it ends.

Nothing here asserts anything; the test does, against what the next daemon makes
of what this one leaves behind.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Literal, TypeAlias

import anyio
from mass_restart_host import PROVIDER_ROW

from ph.bundles import BASE, HEADLESS
from ph.cordis import Profile, ProfileDocument, load_profile_documents
from ph.json import JsonValue
from ph.keys import GOALS, SUBAGENTS
from ph.llm.fake import FakeAdapter
from ph.llm.types import Finish, FinishReason, GenerateOptions, StreamChunk
from ph.paths import resolve_roots
from ph.seams.goals import Goal
from ph.seams.live_index import LiveIndex
from ph.seams.schedule import TICK, Schedule
from ph.seams.subagents import SubagentRequest
from ph.session import Session, now_ms
from ph.testing import settled, stored_types
from ph.wire import literal_lookup
from ph_app import verbs
from ph_app.daemon.client import connected
from ph_app.daemon.server import DaemonServer, serve
from ph_app.daemon.supervisor import Root, Supervisor
from ph_app.params import CreateScheduleParams
from ph_rlm.subagents import PROVIDER_NAME

Mode: TypeAlias = Literal["goal", "wake", "turn", "tick-claim", "tick-write", "resume"]
MODE_NAMES: Mapping[str, Mode] = literal_lookup(Mode)

PROFILE = Profile.from_documents(
    [*load_profile_documents([BASE, HEADLESS]), ProfileDocument("host-overlay", [PROVIDER_ROW])]
)
"""Base and headless with the sub-agent provider, which neither carries: the
deployment both daemons run, so the test imports it rather than spelling it again."""
ROOT = "root"
GOAL = Goal(id="g1", objective="write the quarterly report", gates=[])
DUE_IN_MS = 300
"""How far ahead a mode's schedule is due, so the claim is the scheduler's own pass
and not the host's. The pass lands a second after the boot pass whatever this says
(`recovery.PASS_FLOOR`)."""
WAIT_SECONDS = 30.0


async def _held(self: FakeAdapter, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
    """Every model request, the root's and its children's: waits to be killed."""
    await anyio.sleep_forever()
    yield Finish(reason=FinishReason(kind="stop"))  # pragma: no cover


def _say(**ids: JsonValue) -> None:
    """The one line the test waits for: everything is in flight and on disk."""
    print(json.dumps(ids), flush=True)


async def _owing(*holds: str) -> None:
    """Wait until the live-roots index on disk says the root owes `holds`."""
    index = LiveIndex(resolve_roots().home)

    def owed() -> bool:
        entry = index.read().get(ROOT)
        return entry is not None and set(entry.holds) >= set(holds)

    await settled(owed, f"the index to hold {', '.join(holds)}", seconds=WAIT_SECONDS)


async def _child(root: Root) -> dict[str, JsonValue]:
    """One sub-agent of the root's, at the model when the kill comes."""
    run = await root.ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="gather the quarter's numbers", parent=root.agent)
    )
    await settled(
        lambda: "turn/start" in stored_types(root.ctx, run.session_id),
        "the child's first request to reach its disk",
        seconds=WAIT_SECONDS,
    )
    return {"run": run.id, "session": run.session_id}


async def _prompted(server: DaemonServer, text: str) -> None:
    """A person's prompt to the root over the daemon's own socket, acked the way a
    client's is, and its turn at the model with the turn's first barrier on disk."""
    await connected(server.path, lambda client: client.prompt(ROOT, text))
    root = server.supervisor.roots[ROOT]
    await settled(
        lambda: "turn/start" in stored_types(root.ctx, ROOT),
        "the root's turn to reach its disk",
        seconds=WAIT_SECONDS,
    )


async def goal(server: DaemonServer) -> None:
    root = await server.supervisor.start(ROOT)
    root.ctx.require(GOALS).set(root.session, GOAL)
    child = await _child(root)
    await _prompted(server, "work toward the goal")
    await _owing("turn", "goal", "children")
    _say(child=child)


async def wake(server: DaemonServer) -> None:
    root = await server.supervisor.start(ROOT)
    child = await _child(root)
    await _owing("children")
    _say(child=child)


async def turn(server: DaemonServer) -> None:
    await _prompted(server, "summarize the notes")
    await _owing("turn")
    _say()


async def _schedule(server: DaemonServer) -> None:
    """A `once` schedule on the root, due in `DUE_IN_MS`, through `schedule/create`."""
    params = CreateScheduleParams(
        session_id=ROOT,
        schedule_id="report",
        kind="once",
        spec=str(now_ms() + DUE_IN_MS),
        prompt="run the nightly report",
    )
    await connected(server.path, lambda client: client.call(verbs.SCHEDULE_CREATE, params))


async def tick_claim(server: DaemonServer) -> None:
    """The kill lands inside `fire_due`, after `claim` and before `deliver` — the
    door's own seam, since `_fire` is the delivery `tick` hands it."""

    async def paused(self: Supervisor, root: Root, entry: Schedule) -> None:
        _say()
        await anyio.sleep_forever()

    Supervisor._fire = paused  # type: ignore[method-assign]
    await _schedule(server)


async def tick_write(server: DaemonServer) -> None:
    """The kill lands inside `fire_due`, after the write and before the index is told:
    the moment the log says the tick is durable (`session/durable`, emitted inline
    after the flush), the process stops outright, the loop with it.

    The root's ring is held first: the write yields to the loop, and a turn the
    delivery woke would otherwise start or not in that gap by the scheduler's luck,
    so its `turn/start` would be on disk in one run and absent in the next. Held, the
    kill always lands with the turn not begun; a turn cut before it took its prompt
    is `test_daemon_resume`'s.
    """
    Root.ring = lambda self: None  # type: ignore[method-assign]
    root = await server.supervisor.start(ROOT)

    def written(session: Session, through: int) -> None:
        tick = session.latest(TICK)
        if tick is not None and tick.seq < through:
            _say()
            time.sleep(3600)

    root.ctx.on("session/durable", written)
    await _schedule(server)


async def resume(server: DaemonServer) -> None:
    await settled(lambda: server.resumed, "the resume pass to end", seconds=WAIT_SECONDS)
    _say(resumed=sorted(server.supervisor.roots))


MODES: Mapping[Mode, Callable[[DaemonServer], Awaitable[None]]] = {
    "goal": goal,
    "wake": wake,
    "turn": turn,
    "tick-claim": tick_claim,
    "tick-write": tick_write,
    "resume": resume,
}


async def main(mode: Mode) -> None:
    FakeAdapter.stream = _held  # type: ignore[method-assign]
    ready = anyio.Event()
    started: list[DaemonServer] = []
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            lambda: serve(
                PROFILE, ready=ready, started=started.append, resume_owed=mode == "resume"
            )
        )
        await ready.wait()
        await MODES[mode](started[0])
        await anyio.sleep_forever()


if __name__ == "__main__":
    chosen = MODE_NAMES.get(sys.argv[1])
    if chosen is None:
        raise SystemExit(f"a mode is one of {', '.join(MODES)}, not {sys.argv[1]!r}")
    anyio.run(main, chosen)
