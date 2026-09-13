"""P5-10 — `ph agents`, the client side of the daemon.

Gate: *each round-trips.* So every test here starts a real supervisor on a real
unix socket and drives the real command through `CliRunner` — there is no
in-process shortcut and no faked client, because what this row delivers is
precisely that a person can reach a run they are not attached to, and a fake
transport would agree with whatever the code happened to do.

The commands run in a worker thread (`to_thread.run_sync`) because each one
calls `anyio.run` of its own, which cannot start inside the loop the daemon is
serving on. That is not a test artefact: it is the shape a person's shell has.

## Why the client joins a run of events into one write

One `console.print` per event measured **98 µs — 2.3x the cost of the same events
joined, and 240x the string they carry** — so a 2 048-event snapshot page spent
**202 ms in Rich rather than 86 ms**.

**And why `assistant/chunk` is in `NOISE`.** One line per delta is a keystroke log
rather than a follow, and it is also almost all of the cost: a 2 000-chunk turn is
**196 ms of rendering against 2 ms** once the chunks are dropped.

## Why the client raises the group member and not the group

A bare `raise` re-raises the `ExceptionGroup`, so anything `_ask` does not name
reached the person as a group traceback instead of the exception inside it — and a
`typer.Exit` raised by a command's own exchange became an exit code nothing reads.
Two of these commands used to carry a `bool` back out and raise at command level to
work around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import break_the_provider, private_runtime, serving
from typer.testing import CliRunner

from ph.json import JsonObject, as_obj
from ph.testing import ReapedHost
from ph_app.cli import app
from ph_app.daemon.client import DaemonClient
from ph_app.payloads import DaemonStatusReply
from ph_app.protocol import Cursor

pytestmark = pytest.mark.anyio

runner = CliRunner()


async def _watchers(client: DaemonClient, session_id: str) -> int:
    """How many clients the daemon says are attached to this root."""
    listed = await client.call("sessions/list")
    row = next((one for one in listed["sessions"] if one["sessionId"] == session_id), None)
    return 0 if row is None else int(row["watchers"])


async def _ph(*args: str) -> Any:  # noqa: ANN401
    """One `ph …` invocation, off the loop the daemon is serving on."""

    def invoke() -> Any:  # noqa: ANN401
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair a person actually types, and the round trip that matters.

    `send` returns as soon as the prompt is logged — the protocol's own contract
    — so the answer is something to *watch*, which is what `attach --until-idle`
    is for. Whether the turn finishes before the attach or during it, the same
    assertion holds: catch-up and the live stream are one rendering of one log.
    """
    async with serving(tmp_path, monkeypatch):
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


async def test_until_idle_exits_non_zero_when_the_last_turn_errored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idle is how "answered" and "the last answer failed" both look (P5-04).

    A script chaining on `--until-idle` read the second as success, which is the
    whole reason the root's last turn is projected beside its status. The turn is
    made to fail at the provider, so the loop records `turn/end{error}` the way it
    would for a real outage rather than the test asserting on a synthetic record.
    """
    break_the_provider(monkeypatch)
    async with serving(tmp_path, monkeypatch):
        assert (await _ph("agents", "send", "sour", "answer me")).exit_code == 0

        followed = await _ph("agents", "attach", "sour", "--until-idle")

        assert followed.exit_code == 1, followed.output
        assert "the last turn ended in an error" in followed.output
        assert "last turn error" in followed.output, "the status line names it too"


async def test_until_idle_exits_zero_when_the_turn_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half, so the exit code is a signal rather than a constant."""
    async with serving(tmp_path, monkeypatch):
        assert (await _ph("agents", "send", "sweet", "answer me")).exit_code == 0

        followed = await _ph("agents", "attach", "sweet", "--until-idle")

        assert followed.exit_code == 0, followed.output
        assert "last turn" not in followed.output, "an ordinary ending is not annotated"


async def test_since_skips_the_history_a_client_already_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reattach is not a replay.

    The cursor a client resumes from is `{generation, sequence}`, and the
    generation is what makes a bare sequence mean anything — so a bare `--since`
    is stamped with the generation the *attach reply* just named. That is the
    form a person types at a session they were just watching; the verifiable
    form is the next test's.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        client = await daemon.client()
        await _ph("agents", "send", "resumed", "the first thing")
        whole = await _ph("agents", "attach", "resumed", "--until-idle")
        assert "the first thing" in whole.output
        # Asked over the wire rather than read off the supervisor or scraped
        # out of the table `status` draws: this file's whole claim is that a
        # person reaches a run through the protocol, and a test that reached
        # around it would be proving something else.
        head = as_obj((await client.call("session/status", sessionId="resumed"))["cursor"])[
            "sequence"
        ]

        await _ph("agents", "send", "resumed", "the second thing")
        rest = await _ph("agents", "attach", "resumed", "--since", str(head), "--until-idle")
        assert rest.exit_code == 0, rest.output
        assert "the second thing" in rest.output
        assert "the first thing" not in rest.output
        assert "history starts at" not in rest.output, "honoured, so nothing to report"


async def test_a_full_cursor_is_verified_and_a_stale_one_skips_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The hole.** A bare `--since` was the only form, and it was stamped with the
    *current* generation — so a sequence kept from another incarnation of the log
    was honoured against this one, skipping events this reader had never seen.
    That is precisely the case `resume_at` exists to refuse, defeated by the
    caller handing it a fresh generation with a stale sequence.

    Two things are pinned. The `GENERATION:SEQ` form goes through intact, so the
    daemon can check it. And when the generation does not match, the fallback to 0
    is *visible and lossless*: the whole log is shown and a line says so. It used
    to be silent and lossy — `seen` was pre-set to `since - 1`, so the first `since`
    events of the new incarnation were dropped as already seen.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        client = await daemon.client()
        await _ph("agents", "send", "kept", "the first thing")
        await _ph("agents", "attach", "kept", "--until-idle")
        cursor = as_obj((await client.call("session/status", sessionId="kept"))["cursor"])
        await _ph("agents", "send", "kept", "the second thing")

        right = await _ph(
            "agents",
            "attach",
            "kept",
            "--since",
            f"{cursor['generation']}:{cursor['sequence']}",
            "--until-idle",
        )
        assert right.exit_code == 0, right.output
        assert "the second thing" in right.output
        assert "the first thing" not in right.output, "a verified cursor resumes where it says"
        assert "history starts at" not in right.output

        stale = await _ph(
            "agents", "attach", "kept", "--since", f"1:{cursor['sequence']}", "--until-idle"
        )
        assert stale.exit_code == 0, stale.output
        assert "the first thing" in stale.output, "another incarnation: nothing is skipped"
        assert "the second thing" in stale.output
        assert "history starts at 0" in stale.output, "and the fallback is said, not hidden"


async def test_status_prints_the_cursor_attach_can_take_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves of a cursor were printed as two rows a script would have to
    reassemble; the `resume with` row is the one form `--since` can verify."""
    async with serving(tmp_path, monkeypatch):
        await _ph("agents", "send", "shown", "hello")
        await _ph("agents", "attach", "shown", "--until-idle")

        shown = await _ph("agents", "status", "shown")

        assert shown.exit_code == 0, shown.output
        assert "resume with" in shown.output
        assert "--since " in shown.output
        # Generation and sequence, joined the way the parser splits them.
        import re

        assert re.search(r"--since \d+:\d+", shown.output), shown.output


_CURSOR = Cursor(generation="1700000000000", sequence=0)
"""A position in a log nobody is reading — these tests are about what `seed`
does with a *status*, and the cursor is only there because a reply carries one."""


def test_the_since_parser_stamps_a_bare_sequence_and_keeps_a_full_cursor() -> None:
    """The parse is the protocol's, beside `cursor_of` and `resume_at` which define
    what a cursor is; `rpartition(":")` is unambiguous because the generation is an
    integer timestamp. `None` for anything that is not two integers — what to do
    about that is the caller's, and for the CLI it is exit 2."""
    from ph_app.protocol import Cursor, cursor_text, parse_cursor

    current = {"generation": "1700000000000", "sequence": 40}
    assert parse_cursor("7", current) == Cursor(generation="1700000000000", sequence=7)
    assert parse_cursor("42:7", current) == Cursor(generation="42", sequence=7)
    bare = parse_cursor("0", current)
    assert bare is not None and bare.sequence == 0, "a real position, not a default"
    for bad in ("seven", "a:7", "7:b", ":", "1:2:3x"):
        assert parse_cursor(bad, current) is None, bad

    # And the printed form round-trips through it, which is why the two live together.
    assert cursor_text(Cursor(generation="42", sequence=7)) == "42:7"
    assert parse_cursor(cursor_text(Cursor.model_validate(current)), {}) == Cursor(
        generation="1700000000000", sequence=40
    )


def test_the_cli_refuses_an_unparseable_since_with_exit_two() -> None:
    """The refusal is the command's, not the parser's — the same split
    `selectors_or_exit` makes one module over."""
    import typer

    from ph_app.agents import _since_cursor

    # Cursors in and out: `--since` is parsed against the position the attach
    # reply named, and what comes back is what the typed send carries.
    current = Cursor(generation="9", sequence=40)
    assert _since_cursor("7", current) == Cursor(generation="9", sequence=7)
    with pytest.raises(typer.Exit) as refused:
        _since_cursor("seven", current)
    assert refused.value.exit_code == 2


async def test_agents_lists_every_root_the_daemon_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bare command is the listing — the question a person asks first."""
    async with serving(tmp_path, monkeypatch):
        empty = await _ph("agents")
        assert empty.exit_code == 0, empty.output
        assert "no roots running" in empty.output

        await _ph("agents", "send", "beta", "hello")
        await _ph("agents", "attach", "beta", "--until-idle")

        listed = await _ph("agents")
        assert listed.exit_code == 0, listed.output
        assert "beta" in listed.output
        assert "idle" in listed.output


async def test_status_reports_one_root_in_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the listing has no room for: the ladder, and what is still to fire."""
    async with serving(tmp_path, monkeypatch):
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


async def test_schedule_creates_lists_and_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three verbs of one command, against the seam that owns them.

    The timing flag picks the kind, so nothing here spells `--kind interval`:
    that pair is the wire's shape, and putting it in front of a person is how a
    CLI comes to be a transcription of a protocol.
    """
    async with serving(tmp_path, monkeypatch, tick_every=0.0):
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two refusals that would otherwise be silent, permanent mistakes.

    A schedule with no prompt is claimed, recorded, counted as fired and
    delivers nothing — forever, because at-most-once never retries a claim. Two
    timing flags is the other half: whichever one lost would be a schedule
    firing on a rule its author did not write.
    """
    async with serving(tmp_path, monkeypatch, tick_every=0.0):
        mute = await _ph("agents", "schedule", "eps", "--every", "60000")
        assert mute.exit_code == 2
        assert "--prompt" in mute.output

        both = await _ph(
            "agents", "schedule", "eps", "--every", "60000", "--cron", "* * * * *", "-p", "x"
        )
        assert both.exit_code == 2


async def test_doctor_reports_the_socket_the_daemon_actually_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read back over the wire, never re-derived on this side.

    A doctor that printed what *this* invocation's environment would have chosen
    would agree with a daemon started from a different one and say nothing at
    all. The passivation policy is the sharpest case: it is a flag on `ph
    daemon`, and the client has no way to guess it.
    """
    async with serving(tmp_path, monkeypatch, passivate_after=600.0) as daemon:
        reported = await _ph("agents", "doctor")
        assert reported.exit_code == 0, reported.output
        assert str(daemon.path) in reported.output
        # Ten minutes, which the *default* (ninety) renders no part of: an
        # assertion on "30m" would have passed against `1h 30m` and proved
        # nothing about where the number came from.
        assert "10m" in reported.output, "the daemon's own passivation policy"
        assert "attach" in reported.output, "the capability block it answers with"


async def test_shutdown_waits_for_the_daemon_to_actually_be_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "I asked" and "it stopped" are different claims, and this makes the second.

    `shutdown` takes no id by contract, so there is no reply to wait on; the
    confirmation is the connection the daemon closes on its way out — which is
    also when roots are flushed and leases released.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        stopped = await _ph("agents", "shutdown")
        assert stopped.exit_code == 0, stopped.output
        assert "daemon stopped" in stopped.output
        with anyio.fail_after(5):
            while daemon.path.exists():
                await anyio.sleep(0.02)


async def test_a_follow_ends_when_the_daemon_goes_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other way a follow ends, and the one nothing else would notice.

    `attach` without `--until-idle` waits for a root that may never go idle, so
    the daemon shutting down has to end it — and every reply the client was
    waiting on has to fail rather than park, because the event a `call` waits on
    is set by a pump that is no longer reading. Both halves are exercised here:
    the wait, and the `session/detach` in the teardown behind it.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_runtime(tmp_path, monkeypatch)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two failures, two next steps.

    A path left behind by a crashed daemon is the ordinary aftermath of one, and
    it is *not* the same situation as never having started one: `ph daemon`
    clears a stale socket on its way up, so saying so is what stops a reader
    deleting a file by hand.
    """
    runtime = private_runtime(tmp_path, monkeypatch)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped_host: ReapedHost
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped_host: ReapedHost
) -> None:
    """Read back over the wire like every other row in that table.

    The daemon asks about the socket *it* bound, so a client started from a
    different environment is told what is in force rather than what it would
    have chosen — the same reasoning the passivation policy is asserted for one
    test above.
    """
    reaped_host()
    async with serving(tmp_path, monkeypatch):
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
    names = {
        command.name or (command.callback.__name__ if command.callback else "<no callback>")
        for command in registered
    }
    assert names == {"send", "attach", "schedule", "status", "doctor", "shutdown"}
    groups = {group.name for group in app.registered_groups}
    assert "agents" in groups


def test_the_daemon_status_reply_is_json_and_says_what_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reply `doctor` renders, checked as data.

    A rendering test can only assert on what a table happens to print; this
    holds the *fields*, so a rename on the daemon side fails here rather than
    quietly emptying a row in someone's terminal.
    """
    from daemon_helpers import PROFILE

    from ph_app.daemon.server import DaemonServer
    from ph_app.daemon.supervisor import Supervisor

    async def read() -> DaemonStatusReply:
        facts: DaemonStatusReply
        async with anyio.create_task_group() as tasks:
            server = DaemonServer(
                supervisor=Supervisor(profile=PROFILE, tasks=tasks),
                stop=anyio.Event(),
                path=tmp_path / "daemon.sock",
            )
            facts = server.status()
            tasks.cancel_scope.cancel()
        # Assigned inside the group and returned outside it: a task group's
        # `__aexit__` is typed as one that may suppress, so a `return` in the
        # block leaves a path that falls off the end. `connected()` in
        # `daemon/client.py` carries the same note for the same reason.
        return facts

    facts = anyio.run(read)
    # Round-trips as JSON, because it does: every reply goes through `dumps`.
    # Through `to_wire()`, which is what `respond` calls on its way to the
    # frame — the reply is a `DaemonStatusReply` now (issue 74) and the thing
    # this pins is that the *frame* it becomes still round-trips.
    wire = json.loads(json.dumps(facts.to_wire()))
    assert wire["socket"].endswith("daemon.sock")
    assert wire["protocolVersion"] == 1
    assert facts.protocol_version == 1
    assert set(facts.capabilities) >= {"sessions", "streaming", "roots", "attach"}
    assert facts.roots == 0
    assert facts.uptime_ms >= 0


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

    follow.feed("session.event", live)
    follow.feed("session.event", other)
    assert capsys.readouterr().out == "", "nothing prints before catch-up is done"

    follow.feed.seen = 3
    follow.feed.live()
    follow.feed("session.event", later)
    printed = capsys.readouterr().out
    assert "turn/end" not in printed, "seq 3 was already in a snapshot page"
    assert printed.count("turn/start") == 1
    assert "9" not in printed, "another session's frames are not this follow's"


async def test_attach_reads_the_status_it_was_handed_rather_than_asking_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The round trip is gone, and that is the point of `Followed.seed`.**

    The attach reply is `root.describe()` plus the footer — the same shape
    `session.status` sends — so it already carries `status` and `lastTurn`. The
    command used to call `session/status` for them anyway, on the one path where a
    root was idle before the attach landed. Asserted by watching what the client
    actually sends, because "we no longer need it" is the kind of claim that
    quietly stops being true.
    """
    from ph_app.daemon.client import DaemonClient

    called: list[str] = []
    original = DaemonClient.call

    async def recording(
        self: Any,  # noqa: ANN401
        verb: Any,  # noqa: ANN401
        params: Any = None,  # noqa: ANN401
        /,
        **fields: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        # `call`'s two doors since issue 74: a `Verb` with its params
        # positionally, or a bare name with the keyword form for a caller with
        # no model. The spy has to mirror both, or it drops whichever half it
        # forgot — and it records the *name* either way, which is what this test
        # is about.
        called.append(getattr(verb, "name", verb))
        return await original(self, verb, params, **fields)

    monkeypatch.setattr(DaemonClient, "call", recording)
    async with serving(tmp_path, monkeypatch):
        assert (await _ph("agents", "send", "asked", "answer me")).exit_code == 0
        called.clear()

        followed = await _ph("agents", "attach", "asked", "--until-idle")

        assert followed.exit_code == 0, followed.output
        assert "session/attach" in called
        assert "session/status" not in called, f"asked for what it was handed: {called}"


def test_the_attach_reply_is_the_first_status_and_stops_an_already_idle_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A root idle *before* the attach announces nothing afterwards, so the reply is
    the only status this feed will ever see.

    Losing it is how `--until-idle` came to hang on a finished root, and reading it
    outside `_status` is how the same outcome came to print differently depending on
    which side of a race the host landed on. One entry point, one decision.

    Held against the class because the end-to-end version is the race: the fake
    adapter answers in microseconds, so a real turn is usually finished before the
    attach and the *other* branch is the one that never runs.
    """
    from ph_app.agents import _Follow
    from ph_app.payloads import AttachReply

    follow = _Follow(session_id="s", until_idle=True)
    # The reply's own type: `seed` takes an `AttachReply` since P8-08, so a test
    # that hand-built the dict was pinning a shape the daemon no longer sends.
    reply = AttachReply(
        session_id="s", status="idle", last_turn="error", watchers=0, cursor=_CURSOR
    )

    follow.feed.seed(reply)

    assert follow.done.is_set(), "an already-idle root releases the wait"
    assert follow.last_turn == "error", "and carries the reason the exit code needs"
    printed = capsys.readouterr().out
    assert "idle" in printed and "last turn error" in printed, printed


def test_a_busy_root_is_seeded_without_ending_the_follow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half, for `probe_sandbox`'s reason: a seed that always stopped the
    follow would pass the test above while making `attach` useless. The reply is
    still printed — a person is told what they attached to — and the wait stands."""
    from ph_app.agents import _Follow
    from ph_app.payloads import AttachReply

    follow = _Follow(session_id="s", until_idle=True)

    follow.feed.seed(AttachReply(session_id="s", status="busy", watchers=0, cursor=_CURSOR))

    assert not follow.done.is_set()
    assert "busy" in capsys.readouterr().out


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

    result: JsonObject = {
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

    gave_up: JsonObject = {
        "seq": 9,
        "type": "supervisor/failed",
        "data": {"attempts": 3, "reason": "boom"},
    }
    line = _line(gave_up)
    assert "attempts=3" in line and "reason=boom" in line

    # A payload with nothing in it still says its type, and says it once.
    assert _line({"seq": 1, "type": "turn/end", "data": {}}).count("turn/end") == 1


async def test_a_follow_leaves_out_the_keystroke_log_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn is mostly `assistant/chunk`, and its text arrives again as the
    message that closes it — so showing both is a keystroke log wrapped around
    the thing a person came to read. It was also 98 µs of rendering per frame,
    which is 196 ms a turn spent on output nobody reads.
    """
    async with serving(tmp_path, monkeypatch):
        await _ph("agents", "send", "quiet", "hello")
        default = await _ph("agents", "attach", "quiet", "--until-idle")
        assert default.exit_code == 0, default.output
        assert "assistant/chunk" not in default.output
        assert "assistant/message" in default.output

        everything = await _ph("agents", "attach", "quiet", "--until-idle", "--all")
        assert everything.exit_code == 0, everything.output
        assert "assistant/chunk" in everything.output


async def test_a_follow_filters_to_a_namespace_and_drills_into_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P6-33 through the follower: `--type` selects, and a namespace is not a
    substring. `log:turn` must bring turn boundaries and nothing else."""
    async with serving(tmp_path, monkeypatch):
        await _ph("agents", "send", "picky", "hello")

        turns = await _ph("agents", "attach", "picky", "--until-idle", "--type", "turn")
        assert turns.exit_code == 0, turns.output
        assert "turn/start" in turns.output and "turn/end" in turns.output
        assert "assistant/message" not in turns.output, "a namespace is not everything"

        one = await _ph("agents", "attach", "picky", "--until-idle", "--type", "turn/end")
        assert one.exit_code == 0, one.output
        assert "turn/end" in one.output and "turn/start" not in one.output


async def test_a_named_namespace_overrides_the_per_delta_hush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a namespace is a stronger signal than the default quiet.

    `assistant/chunk` is hidden without `--all` because its text arrives again in
    the message that closes the turn. But somebody who typed
    `--type assistant/chunk` asked for exactly that, and making them add `--all`
    on top would answer a question they did not ask.
    """
    async with serving(tmp_path, monkeypatch):
        await _ph("agents", "send", "loud", "hello")
        result = await _ph("agents", "attach", "loud", "--until-idle", "--type", "assistant/chunk")
        assert result.exit_code == 0, result.output
        assert "assistant/chunk" in result.output, "asked for by name, and still hidden"


async def test_a_follow_refuses_the_other_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bus:tools` follows nothing — this is a session log. Refused rather than
    answered with an empty stream, which would read as a quiet session."""
    async with serving(tmp_path, monkeypatch):
        result = await _ph("agents", "attach", "wrong", "--type", "bus:tools")
        assert result.exit_code == 2, result.output
        assert "does not serve" in result.output


async def test_an_unnamed_failure_surfaces_as_itself_not_as_a_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a person sees when something goes wrong that nobody anticipated.

    The exchange runs inside a task group, and anyio wraps whatever comes out of
    one — even a single exception — so re-raising what was caught meant handing
    back the `ExceptionGroup` rather than the failure inside it. "unhandled
    errors in a TaskGroup (1 sub-exception)" is a sentence about anyio, not
    about what broke.
    """
    from ph_app.agents import _ask

    async def boom(client: DaemonClient) -> None:
        raise ValueError("nothing to do with the daemon")

    def invoke() -> None:
        _ask(boom)

    async with serving(tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="nothing to do with the daemon"):
            await anyio.to_thread.run_sync(invoke)
