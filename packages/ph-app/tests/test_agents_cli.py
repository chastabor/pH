"""P5-10 — `ph agents`, the client side of the daemon.

Gate: *each round-trips.* So every test here starts a real supervisor on a real
unix socket and drives the real command through `CliRunner` — there is no
in-process shortcut and no faked client, because what this row delivers is
precisely that a person can reach a run they are not attached to, and a fake
transport would agree with whatever the code happened to do.

The commands run in a worker thread (`to_thread.run_sync`) because each one
calls `anyio.run` of its own, which cannot start inside the loop the daemon is
serving on. That is not a test artefact: it is the shape a person's shell has.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import Daemon, daemon_socket, running
from typer.testing import CliRunner

from ph_app.cli import app
from ph_app.wire import obj

pytestmark = pytest.mark.anyio

runner = CliRunner()

ReapedHost = Callable[..., Path]
"""The repo-root `reaped_host` fixture, spelled where it is read — structurally
rather than by `from conftest import …`, which resolves to this package's own
conftest rather than to the root one the fixture lives in."""


@asynccontextmanager
async def _daemon(tmp_path: Path, monkeypatch: Any, **options: Any) -> AsyncIterator[Daemon]:
    """A supervisor listening where `ph agents` will look for it.

    Through the shared `running`, which is where `serve()`'s startup contract
    lives — a second copy of it here would be the one that gets missed when that
    contract changes, and a missed one fails as a hang rather than as a diff.
    The socket is the one `$PH_RUNTIME` derives, and it is handed to `serve` so
    the two halves are pinned to *the same* derivation rather than to two.
    """
    _runtime(tmp_path, monkeypatch)
    async with running(tmp_path, path=daemon_socket(), **options) as daemon:
        yield daemon


def _runtime(tmp_path: Path, monkeypatch: Any) -> Path:
    """Point `$PH_RUNTIME` somewhere private.

    Explicit in every test, including the ones with no daemon: the fallback is
    `$XDG_RUNTIME_DIR`, which on a developer's machine is where their *real*
    daemon listens — a "no daemon is running" test that connected to it would
    pass for the wrong reason, or worse, shut it down.
    """
    runtime = tmp_path / "run"
    runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PH_RUNTIME", str(runtime))
    return runtime


async def _watchers(client: Any, session_id: str) -> int:
    """How many clients the daemon says are attached to this root."""
    listed = await client.call("sessions/list")
    row = next((one for one in listed["sessions"] if one["sessionId"] == session_id), None)
    return 0 if row is None else int(row["watchers"])


async def _ph(*args: str) -> Any:
    """One `ph …` invocation, off the loop the daemon is serving on."""

    def invoke() -> Any:
        # Wide and unstyled, because these assertions are about content. Rich
        # wraps to 80 columns off a terminal, and a wrapped session id is a
        # substring that is present and unfindable; `FORCE_COLOR` — which CI
        # images and plenty of shells set — puts escape sequences *inside* the
        # words, so `"interval 3600000" in output` is false for output that
        # reads as exactly that.
        return runner.invoke(
            app,
            list(args),
            env={"COLUMNS": "200", "FORCE_COLOR": None, "NO_COLOR": "1", "TERM": "dumb"},
        )

    return await anyio.to_thread.run_sync(invoke)


# ------------------------------------------------------------------ the seven --


async def test_send_queues_a_turn_and_attach_shows_the_answer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The pair a person actually types, and the round trip that matters.

    `send` returns as soon as the prompt is logged — the protocol's own contract
    — so the answer is something to *watch*, which is what `attach --until-idle`
    is for. Whether the turn finishes before the attach or during it, the same
    assertion holds: catch-up and the live stream are one rendering of one log.
    """
    async with _daemon(tmp_path, monkeypatch):
        sent = await _ph("agents", "send", "alpha", "what is the answer")
        assert sent.exit_code == 0, sent.output
        assert "queued on alpha" in sent.output

        followed = await _ph("agents", "attach", "alpha", "--until-idle")
        assert followed.exit_code == 0, followed.output
        assert "user/message" in followed.output
        assert "what is the answer" in followed.output
        assert "assistant/message" in followed.output
        assert "ok" in followed.output
        # Once, not twice. Attach subscribes *before* the history is read, so a
        # frame can arrive live and again in a snapshot page; the follower
        # discards anything at or below the last sequence the pages showed.
        assert followed.output.count("user/message") == 1


async def test_since_skips_the_history_a_client_already_has(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A reattach is not a replay.

    The cursor a client resumes from is `{generation, sequence}`, and the
    generation is what makes a bare sequence mean anything — so `--since` is
    paged against the generation the *attach reply* just named rather than
    against one this side invented, which the server would read as "you have
    seen nothing" and answer with the whole log.
    """
    async with _daemon(tmp_path, monkeypatch) as daemon:
        client = await daemon.client()
        await _ph("agents", "send", "resumed", "the first thing")
        whole = await _ph("agents", "attach", "resumed", "--until-idle")
        assert "the first thing" in whole.output
        # Asked over the wire rather than read off the supervisor or scraped
        # out of the table `status` draws: this file's whole claim is that a
        # person reaches a run through the protocol, and a test that reached
        # around it would be proving something else.
        head = obj((await client.call("session/status", sessionId="resumed"))["cursor"])["sequence"]

        await _ph("agents", "send", "resumed", "the second thing")
        rest = await _ph("agents", "attach", "resumed", "--since", str(head), "--until-idle")
        assert rest.exit_code == 0, rest.output
        assert "the second thing" in rest.output
        assert "the first thing" not in rest.output


async def test_agents_lists_every_root_the_daemon_is_running(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The bare command is the listing — the question a person asks first."""
    async with _daemon(tmp_path, monkeypatch):
        empty = await _ph("agents")
        assert empty.exit_code == 0, empty.output
        assert "no roots running" in empty.output

        await _ph("agents", "send", "beta", "hello")
        await _ph("agents", "attach", "beta", "--until-idle")

        listed = await _ph("agents")
        assert listed.exit_code == 0, listed.output
        assert "beta" in listed.output
        assert "idle" in listed.output


async def test_status_reports_one_root_in_detail(tmp_path: Path, monkeypatch: Any) -> None:
    """What the listing has no room for: the ladder, and what is still to fire."""
    async with _daemon(tmp_path, monkeypatch):
        await _ph("agents", "send", "gamma", "hello")
        await _ph("agents", "attach", "gamma", "--until-idle")

        detail = await _ph("agents", "status", "gamma")
        assert detail.exit_code == 0, detail.output
        assert "root gamma" in detail.output
        assert "retry attempts" in detail.output
        assert "given up" in detail.output

        missing = await _ph("agents", "status", "nobody")
        assert missing.exit_code == 1
        assert "no session" in missing.output


async def test_schedule_creates_lists_and_cancels(tmp_path: Path, monkeypatch: Any) -> None:
    """All three verbs of one command, against the seam that owns them.

    The timing flag picks the kind, so nothing here spells `--kind interval`:
    that pair is the wire's shape, and putting it in front of a person is how a
    CLI comes to be a transcription of a protocol.
    """
    async with _daemon(tmp_path, monkeypatch, tick_every=0.0):
        empty = await _ph("agents", "schedule", "delta")
        # Listing refuses on a root nobody has started — the honest answer, and
        # the same one `session/snapshot` gives.
        assert empty.exit_code == 1
        assert "no session" in empty.output

        made = await _ph(
            "agents", "schedule", "delta", "--every", "3600000", "--prompt", "check the build"
        )
        assert made.exit_code == 0, made.output
        assert "interval 3600000" in made.output
        schedule_id = made.output.split("scheduled ")[1].split(" ")[0]

        listed = await _ph("agents", "schedule", "delta")
        assert listed.exit_code == 0, listed.output
        assert schedule_id in listed.output
        assert "check the build" in listed.output
        # A next fire time, not a dash: an hourly schedule created just now is
        # due in an hour, and a listing that could not say so would be a table
        # of names.
        assert "—" not in listed.output

        gone = await _ph("agents", "schedule", "delta", "--cancel", schedule_id)
        assert gone.exit_code == 0, gone.output
        assert (await _ph("agents", "schedule", "delta")).output.count("no schedules") == 1

        again = await _ph("agents", "schedule", "delta", "--cancel", schedule_id)
        assert again.exit_code == 1
        assert "no schedule" in again.output


async def test_a_schedule_needs_one_timing_and_something_to_say(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Two refusals that would otherwise be silent, permanent mistakes.

    A schedule with no prompt is claimed, recorded, counted as fired and
    delivers nothing — forever, because at-most-once never retries a claim. Two
    timing flags is the other half: whichever one lost would be a schedule
    firing on a rule its author did not write.
    """
    async with _daemon(tmp_path, monkeypatch, tick_every=0.0):
        mute = await _ph("agents", "schedule", "eps", "--every", "60000")
        assert mute.exit_code == 2
        assert "--prompt" in mute.output

        both = await _ph(
            "agents", "schedule", "eps", "--every", "60000", "--cron", "* * * * *", "-p", "x"
        )
        assert both.exit_code == 2


async def test_doctor_reports_the_socket_the_daemon_actually_bound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Read back over the wire, never re-derived on this side.

    A doctor that printed what *this* invocation's environment would have chosen
    would agree with a daemon started from a different one and say nothing at
    all. The passivation policy is the sharpest case: it is a flag on `ph
    daemon`, and the client has no way to guess it.
    """
    async with _daemon(tmp_path, monkeypatch, passivate_after=600.0) as daemon:
        reported = await _ph("agents", "doctor")
        assert reported.exit_code == 0, reported.output
        assert str(daemon.path) in reported.output
        # Ten minutes, which the *default* (ninety) renders no part of: an
        # assertion on "30m" would have passed against `1h 30m` and proved
        # nothing about where the number came from.
        assert "10m" in reported.output, "the daemon's own passivation policy"
        assert "attach" in reported.output, "the capability block it answers with"


async def test_shutdown_waits_for_the_daemon_to_actually_be_gone(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """ "I asked" and "it stopped" are different claims, and this makes the second.

    `shutdown` takes no id by contract, so there is no reply to wait on; the
    confirmation is the connection the daemon closes on its way out — which is
    also when roots are flushed and leases released.
    """
    async with _daemon(tmp_path, monkeypatch) as daemon:
        stopped = await _ph("agents", "shutdown")
        assert stopped.exit_code == 0, stopped.output
        assert "daemon stopped" in stopped.output
        with anyio.fail_after(5):
            while daemon.path.exists():
                await anyio.sleep(0.02)


async def test_a_follow_ends_when_the_daemon_goes_away(tmp_path: Path, monkeypatch: Any) -> None:
    """The other way a follow ends, and the one nothing else would notice.

    `attach` without `--until-idle` waits for a root that may never go idle, so
    the daemon shutting down has to end it — and every reply the client was
    waiting on has to fail rather than park, because the event a `call` waits on
    is set by a pump that is no longer reading. Both halves are exercised here:
    the wait, and the `session/detach` in the teardown behind it.
    """
    async with _daemon(tmp_path, monkeypatch) as daemon:
        client = await daemon.client()
        await _ph("agents", "send", "watched", "hello")
        followed: list[Any] = []
        async with anyio.create_task_group() as tasks:

            async def follow() -> None:
                followed.append(await _ph("agents", "attach", "watched"))

            tasks.start_soon(follow)
            # Wait for the subscription itself rather than sleeping: a fixed
            # pause is a flake on a loaded machine and a slow test on an idle
            # one. `watchers` is what `sessions/list` calls it, so this waits on
            # the fact through the protocol rather than on a supervisor field.
            with anyio.fail_after(10):
                while not await _watchers(client, "watched"):
                    await anyio.sleep(0.01)
            # Stopped through a second client rather than a second `ph agents
            # shutdown`: `CliRunner.invoke` swaps `sys.stdout` process-wide, so
            # two invocations at once have one of them writing into the other's
            # closed buffer. This is the frame the real command sends.
            await client.notify("shutdown")

        assert followed[0].exit_code == 1, followed[0].output
        assert "closed the connection" in followed[0].output
        # Reported as a disconnection, not as a refusal: nobody said no, and
        # "the daemon refused: the daemon closed the connection" is exactly the
        # confusion the absent-socket / stale-socket split exists to prevent.
        assert "refused" not in followed[0].output


# ---------------------------------------------------------------- when it is not --


async def test_no_daemon_names_the_socket_and_how_to_start_one(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _runtime(tmp_path, monkeypatch)
    for command in (
        ["agents"],
        ["agents", "send", "x", "hi"],
        ["agents", "attach", "x"],
        ["agents", "schedule", "x"],
        ["agents", "status", "x"],
        ["agents", "doctor"],
        ["agents", "shutdown"],
    ):
        result = await _ph(*command)
        assert result.exit_code == 1, f"{command}: {result.output}"
        assert "no daemon socket" in result.output, command
        assert "ph daemon" in result.output, command


async def test_a_socket_nobody_answers_is_told_apart_from_no_socket(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Two failures, two next steps.

    A path left behind by a crashed daemon is the ordinary aftermath of one, and
    it is *not* the same situation as never having started one: `ph daemon`
    clears a stale socket on its way up, so saying so is what stops a reader
    deleting a file by hand.
    """
    runtime = _runtime(tmp_path, monkeypatch)
    (runtime / "daemon.sock").write_text("not a socket")
    result = await _ph("agents", "doctor")
    assert result.exit_code == 1
    assert "nothing is listening" in result.output
    assert "ph daemon" in result.output


# ------------------------------------------------- P5-11: lingering detection --
#
# The client half of I-6. "No daemon socket" and "your login session took the
# socket with it" are the same `OSError` and the same absent path, and only one
# of them is fixed by starting a daemon — the other has one still running,
# holding every lease the new one will be refused (I-5).
#
# `reaped_host` is the repo-root fixture: `$PH_RUNTIME` inside an
# `$XDG_RUNTIME_DIR` these tests own, with the linger marker directory
# redirected — never read from the machine running the suite, which would assert
# whatever that host happened to say.


async def test_a_reaped_socket_is_not_reported_as_one_never_started(
    reaped_host: ReapedHost,
) -> None:
    """The message that stops an afternoon on `session_already_active`.

    A person who logged out and back in sees exactly what a person who never
    ran `ph daemon` sees. Telling both of them to start one sends the first to a
    refusal from a daemon that is still running and that they have been given no
    reason to look for.
    """
    reaped_host()
    result = await _ph("agents", "doctor")
    assert result.exit_code == 1
    assert "no daemon socket" in result.output
    assert "logind removes" in result.output, "why it is absent"
    assert "may still be running" in result.output, "and why not to just start another"
    assert "loginctl enable-linger someone" in result.output


async def test_an_ordinary_missing_socket_still_just_says_start_one(
    tmp_path: Path, monkeypatch: Any, reaped_host: ReapedHost
) -> None:
    """The advice is conditional, which is what keeps it worth reading.

    `$PH_RUNTIME` outside the reaped tree is the common case on a developer's
    machine and on any host with lingering on; a paragraph about logind printed
    there teaches readers to skip the paragraph.
    """
    reaped_host(linger=True)
    result = await _ph("agents", "doctor")
    assert result.exit_code == 1
    assert "no daemon socket" in result.output
    assert "ph daemon" in result.output
    assert "loginctl" not in result.output


async def test_doctor_prints_the_lifetime_the_daemon_reports(
    tmp_path: Path, monkeypatch: Any, reaped_host: ReapedHost
) -> None:
    """Read back over the wire like every other row in that table.

    The daemon asks about the socket *it* bound, so a client started from a
    different environment is told what is in force rather than what it would
    have chosen — the same reasoning the passivation policy is asserted for one
    test above.
    """
    reaped_host()
    async with _daemon(tmp_path, monkeypatch):
        reported = await _ph("agents", "doctor")
        assert reported.exit_code == 0, reported.output
        assert "socket lifetime" in reported.output
        assert "linger" in reported.output
        assert "loginctl enable-linger someone" in reported.output
        # Absent while it can be reached, which this invocation just proved by
        # arriving: a permanent "reachable: yes" row is a fact delivered by its
        # own delivery, and it would push the row that matters off the eye.
        assert "reachable" not in reported.output


# ------------------------------------------------------------------ registration --


def test_every_agents_command_is_registered() -> None:
    """The seven names, held against the app rather than against a docstring.

    `ph daemon` once stopped being a registered command because a module-level
    helper defined under `@app.command()` captured the decorator, and 1 284
    tests stayed green because nothing asserted registration. This is that
    guard for the group that just grew six of them.
    """
    from ph_app.agents import agents_app

    registered = agents_app.registered_commands
    names = {command.name or command.callback.__name__ for command in registered}
    assert names == {"send", "attach", "schedule", "status", "doctor", "shutdown"}
    groups = {group.name for group in app.registered_groups}
    assert "agents" in groups


def test_the_daemon_status_reply_is_json_and_says_what_it_is(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The reply `doctor` renders, checked as data.

    A rendering test can only assert on what a table happens to print; this
    holds the *fields*, so a rename on the daemon side fails here rather than
    quietly emptying a row in someone's terminal.
    """
    from daemon_helpers import PROFILE

    from ph_app.daemon.server import DaemonServer
    from ph_app.daemon.supervisor import Supervisor

    async def read() -> dict[str, Any]:
        async with anyio.create_task_group() as tasks:
            server = DaemonServer(
                supervisor=Supervisor(documents=PROFILE, tasks=tasks),
                stop=anyio.Event(),
                path=tmp_path / "daemon.sock",
            )
            facts = server.status()
            tasks.cancel_scope.cancel()
            return facts

    facts = anyio.run(read)
    # Round-trips as JSON, because it does: every reply goes through `dumps`.
    assert json.loads(json.dumps(facts))["socket"].endswith("daemon.sock")
    assert facts["protocolVersion"] == 1
    assert set(facts["capabilities"]) >= {"sessions", "streaming", "roots", "attach"}
    assert facts["roots"] == 0
    assert facts["uptimeMs"] >= 0


async def test_the_follower_shows_each_event_once_and_in_the_log_s_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Buffered until catch-up finishes, then replayed without the overlap.

    `session/attach` subscribes *before* the history is fetched — deliberately,
    since the other order drops whatever happens in between — so live frames
    arrive while `session/snapshot` is still paging. Held against the class
    rather than through a command, because forcing that overlap through a real
    turn is a race: the fake adapter answers in microseconds, so the end-to-end
    test reaches this code with nothing in flight and passes either way.
    """
    from ph_app.agents import _Follow

    follow = _Follow(session_id="s")
    live = {"sessionId": "s", "event": {"seq": 3, "type": "turn/end"}}
    later = {"sessionId": "s", "event": {"seq": 4, "type": "turn/start"}}
    other = {"sessionId": "elsewhere", "event": {"seq": 9, "type": "turn/end"}}

    follow("session.event", live)
    follow("session.event", other)
    assert capsys.readouterr().out == "", "nothing prints before catch-up is done"

    follow.seen = 3
    follow.start()
    follow("session.event", later)
    printed = capsys.readouterr().out
    assert "turn/end" not in printed, "seq 3 was already in a snapshot page"
    assert printed.count("turn/start") == 1
    assert "9" not in printed, "another session's frames are not this follow's"


def test_a_followed_line_says_what_the_event_says() -> None:
    """Every type gets a body, and the four that carry a conversation get theirs.

    The four were spelled out and everything else returned `""`, which left 57
    of the 61 known types as a bare word — including every `supervisor/*` record
    that says why a root stopped, which is what a person follows a remote run to
    find out. And `tool/result` was spelled with one hop missing: the text is
    `message.content[0].content`, so `text_of_wire` selected `type: "text"`
    against a `tool-result` block and every tool result rendered blank.
    """
    from ph_app.agents import _line

    result = {
        "seq": 7,
        "type": "tool/result",
        "data": {
            "message": {
                "content": [
                    {"type": "tool-result", "content": [{"type": "text", "text": "42 files"}]}
                ]
            }
        },
    }
    assert "42 files" in _line(result)

    gave_up = {"seq": 9, "type": "supervisor/failed", "data": {"attempts": 3, "reason": "boom"}}
    line = _line(gave_up)
    assert "attempts=3" in line and "reason=boom" in line

    # A payload with nothing in it still says its type, and says it once.
    assert _line({"seq": 1, "type": "turn/end", "data": {}}).count("turn/end") == 1


async def test_a_follow_leaves_out_the_keystroke_log_unless_asked(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A turn is mostly `assistant/chunk`, and its text arrives again as the
    message that closes it — so showing both is a keystroke log wrapped around
    the thing a person came to read. It was also 98 µs of rendering per frame,
    which is 196 ms a turn spent on output nobody reads.
    """
    async with _daemon(tmp_path, monkeypatch):
        await _ph("agents", "send", "quiet", "hello")
        default = await _ph("agents", "attach", "quiet", "--until-idle")
        assert default.exit_code == 0, default.output
        assert "assistant/chunk" not in default.output
        assert "assistant/message" in default.output

        everything = await _ph("agents", "attach", "quiet", "--until-idle", "--all")
        assert everything.exit_code == 0, everything.output
        assert "assistant/chunk" in everything.output


async def test_a_follow_filters_to_a_namespace_and_drills_into_one(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """P6-33 through the follower: `--type` selects, and a namespace is not a
    substring. `log:turn` must bring turn boundaries and nothing else."""
    async with _daemon(tmp_path, monkeypatch):
        await _ph("agents", "send", "picky", "hello")

        turns = await _ph("agents", "attach", "picky", "--until-idle", "--type", "turn")
        assert turns.exit_code == 0, turns.output
        assert "turn/start" in turns.output and "turn/end" in turns.output
        assert "assistant/message" not in turns.output, "a namespace is not everything"

        one = await _ph("agents", "attach", "picky", "--until-idle", "--type", "turn/end")
        assert one.exit_code == 0, one.output
        assert "turn/end" in one.output and "turn/start" not in one.output


async def test_a_named_namespace_overrides_the_per_delta_hush(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Naming a namespace is a stronger signal than the default quiet.

    `assistant/chunk` is hidden without `--all` because its text arrives again in
    the message that closes the turn. But somebody who typed
    `--type assistant/chunk` asked for exactly that, and making them add `--all`
    on top would answer a question they did not ask.
    """
    async with _daemon(tmp_path, monkeypatch):
        await _ph("agents", "send", "loud", "hello")
        result = await _ph("agents", "attach", "loud", "--until-idle", "--type", "assistant/chunk")
        assert result.exit_code == 0, result.output
        assert "assistant/chunk" in result.output, "asked for by name, and still hidden"


async def test_a_follow_refuses_the_other_vocabulary(tmp_path: Path, monkeypatch: Any) -> None:
    """`bus:tools` follows nothing — this is a session log. Refused rather than
    answered with an empty stream, which would read as a quiet session."""
    async with _daemon(tmp_path, monkeypatch):
        result = await _ph("agents", "attach", "wrong", "--type", "bus:tools")
        assert result.exit_code == 2, result.output
        assert "does not serve" in result.output


async def test_an_unnamed_failure_surfaces_as_itself_not_as_a_group(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """What a person sees when something goes wrong that nobody anticipated.

    The exchange runs inside a task group, and anyio wraps whatever comes out of
    one — even a single exception — so re-raising what was caught meant handing
    back the `ExceptionGroup` rather than the failure inside it. "unhandled
    errors in a TaskGroup (1 sub-exception)" is a sentence about anyio, not
    about what broke.
    """
    from ph_app.agents import _ask

    async def boom(client: Any) -> None:
        raise ValueError("nothing to do with the daemon")

    def invoke() -> None:
        _ask(boom)

    async with _daemon(tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="nothing to do with the daemon"):
            await anyio.to_thread.run_sync(invoke)
