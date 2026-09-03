"""P5-14 — the terminal over a socket, against a real daemon.

`DaemonSession` is the second implementation of `FrontSession`, and the claim
being tested is *equality*: the app above it cannot tell which one it has, so a
person gets the same layout, the same fold and the same verbs at a tty or in a
browser tab. Every test here drives the front end directly with a `StubHost`,
the way `test_tui_frontend.py` drives the in-process one — no terminal, so a
failure is about the protocol rather than about Textual.

The daemon is in-process (`daemon_helpers.running`) and the socket is real. That
combination is deliberate: an in-memory double for the wire would pass for a
design whose frames never round-trip, and the defects this file exists to catch —
a snapshot page and a live frame both drawing the same event, a status word the
screen has no room for, a verb sent to the end that has never heard of it — are
all about what actually crosses.

**The gate the whole increment is named for is the last one**: a turn started
here finishes after this front end is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import running, until
from tui_helpers import StubHost

from ph.seams.user_questions import UserQuestion
from ph.testing import StubAgent
from ph_app.daemon.follow import Followed
from ph_app.tui.commands import TUI_VERBS
from ph_app.tui.remote import attach_session

pytestmark = pytest.mark.anyio


async def _front(daemon: Any, session_id: str = "remote", **options: Any) -> Any:
    """One attached `DaemonSession` and the host behind it."""
    host = StubHost()
    client = await daemon.client()
    front = await attach_session(client, session_id, host=host, **options)
    return front, host


# ------------------------------------------------------------ the transcript --


async def test_a_turn_reaches_the_transcript_over_the_socket(tmp_path: Path) -> None:
    """The whole path: prompt in, events back, one fold, one state.

    `submit` waits for the root to go idle rather than for the reply, because the
    reply lands as soon as the prompt is *in the inbox* — which is what makes it
    survive this client dying, and the reason the two are separate.
    """
    async with running(tmp_path) as daemon:
        front, host = await _front(daemon)

        await front.submit("hello")

        assert front.state.status == "idle"
        assert [item.role for item in front.state.items][:1] == ["user"]
        assert any(item.role == "assistant" for item in front.state.items)
        assert host.redraws > 0, "the screen was told to redraw"


async def test_a_front_end_attaching_to_a_finished_turn_rebuilds_it_exactly(
    tmp_path: Path,
) -> None:
    """Catch-up alone: the snapshot pages rebuild what the live stream showed.

    Two front ends on one session reach byte-identical transcripts by two
    different routes — one folded the turn as it happened, the other paged it
    afterwards. That the two routes agree is the property `session/attach`'s
    no-replay rule depends on; the *overlap* between them is the next test,
    because this one cannot create it.
    """
    async with running(tmp_path) as daemon:
        first, _ = await _front(daemon, "shared")
        await first.submit("hello")
        before = [(item.role, item.text) for item in first.state.items]
        assert before, "the first front end saw the turn live"

        second, _ = await _front(daemon, "shared")

        assert [(item.role, item.text) for item in second.state.items] == before


async def test_an_event_arriving_on_both_routes_is_folded_once() -> None:
    """The buffer's whole purpose, and the one window that can double an event.

    `session/attach` subscribes *before* the history is paged, so live frames
    arrive while `session/snapshot` is still running — and an event at the head
    of the log can come down both routes. Driven against `_Feed` directly,
    because a real turn against the fake provider finishes faster than the
    overlap it would have to be caught in: an integration test here would pass
    whether or not the rule existed, which is what it did before this was
    written.

    Sabotage: drop the `at <= self.seen` check in `Followed`, and seq 2 folds twice.
    """
    folded: list[int] = []
    feed = Followed(
        session_id="s",
        on_events=lambda pairs: folded.extend(int(one.get("seq", -1)) for one, _ in pairs),
        on_status=lambda params: None,
    )

    # A live frame arrives during catch-up and is held.
    feed("session.event", {"sessionId": "s", "event": {"seq": 2, "type": "turn/end"}})
    # The page that follows already contains it.
    feed.on_events(
        [({"seq": 1, "type": "turn/start"}, None), ({"seq": 2, "type": "turn/end"}, None)]
    )
    feed.seen = 2
    feed.live()

    assert folded == [1, 2], "seq 2 came down both routes and was folded once"


async def test_a_caught_up_page_is_folded_as_history_and_a_frame_as_live(
    tmp_path: Path,
) -> None:
    """Which phase an event arrived in, told to the fold rather than assumed.

    `TuiEventAdapter` has taken `Frame(live=…)` since P3 because a transcript
    being *rebuilt* is not one being *streamed*: a page of history holds a turn's
    `assistant/chunk` records and the `assistant/message` that superseded them,
    so a fold told they were live builds a streaming row and then replaces it
    inside a single pass — which in Textual is a widget mount inside a widget
    mount, and surfaced as `MountError` from an unrelated test.

    Asserted on the contract rather than on the symptom, because the symptom is a
    race: the burst has to be big enough and the frame boundary has to fall in
    the wrong place. Sabotage: pass `True` from `catch_up`, and this fails every
    time while the resume test passes most of the time.
    """
    async with running(tmp_path) as daemon:
        seen: list[bool] = []
        await daemon.root("phases")
        client = await daemon.client()
        feed = Followed(
            session_id="phases",
            on_events=lambda pairs, live: seen.append(live),
            on_status=lambda params: None,
        )

        await feed.catch_up(client, None)
        assert seen == [False], "a snapshot page is history"

        feed.live()
        feed("session.event", {"sessionId": "phases", "event": {"seq": 99, "type": "turn/end"}})

        assert seen == [False, True], "and a notification is not"


async def test_the_second_front_end_sees_the_first_ones_prompt(tmp_path: Path) -> None:
    """The multiplex rule: a submitted prompt is a log entry everyone sees.

    Not the composer — un-submitted text never leaves a client — but pressing
    enter is an act in the session, so it reaches every attached front end by the
    one route everything else does.
    """
    async with running(tmp_path) as daemon:
        first, _ = await _front(daemon, "both")
        second, _ = await _front(daemon, "both")

        await first.submit("from the first")

        await until(
            lambda: any("from the first" in (item.text or "") for item in second.state.items),
            what="the other front end to see the prompt",
        )


# ---------------------------------------------------------------- the words --


async def test_a_root_parked_on_a_person_is_not_shown_as_running(tmp_path: Path) -> None:
    """Five root states, one field, and a bool the widgets actually read.

    `TuiState.status` carries the root's own word — `waiting` and `retrying`
    included — and `TuiState.busy` is what drives the spinner. The alternative
    was a second status field on the remote front end kept in step by hand:
    three writers of one fact, and an in-process screen that could never show
    `retrying` because its type had no room for it.

    Sabotage: make `busy` `status == "running"`, and a retry shows as idle.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        front._status({"status": "waiting"})

        assert front.state.status == "waiting", "the daemon's word is kept, once"
        assert not front.state.busy, "and the spinner stops"

        front._status({"status": "retrying"})

        assert front.state.busy, "a retry is still work in flight"


async def test_the_footer_arrives_beside_the_status(tmp_path: Path) -> None:
    """Readings are pushed with the status, not polled on the 30 Hz tick.

    A reading is a fold of the log, so the moment worth re-reading them is the
    moment the agent moved. Rebuilt through `model_validate` rather than field by
    field, so a reading that grows a field reaches a browser tab with no edit at
    this end — the same argument `to_wire` makes in the other direction.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        front._status({"status": "idle", "readings": [{"text": "12k / 200k", "level": "warning"}]})

        assert [(one.text, one.level) for one in front.status_readings()] == [
            ("12k / 200k", "warning")
        ]


# ----------------------------------------------------------------- the verbs --


async def test_the_command_list_is_both_ends_merged(tmp_path: Path) -> None:
    """One list to a person, two owners underneath.

    `/model` and `/theme` change *this* client's display and mean nothing to a
    daemon serving three of them; `/compact` is the harness's. So the palette
    shows the union, and `run_command` routes on which side owns the name.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        front.attach_surfaces(object())

        names = {definition.name for definition in front.commands()}

        assert names >= {verb.name for verb in TUI_VERBS}, "the terminal's own verbs are offered"
        assert names - {verb.name for verb in TUI_VERBS}, "and so are the daemon's"


async def test_a_daemon_verb_is_dispatched_over_the_wire(tmp_path: Path) -> None:
    """The remote half of the merge: a daemon verb's body *is* `session/command`.

    A command registered in the daemon's `ctx.commands` cannot be executed here —
    its body closes over services in another process — so the definition this
    client holds for it has a `run` that sends the line across. That is what
    lets `run_command` dispatch every verb the same way.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        local = {verb.name for verb in TUI_VERBS}
        remote = next(one for one in front.commands() if one.name not in local)

        # The definition's own `run` is the wire call — one kind of thing in the
        # palette, dispatched one way, whichever end executes it.
        await remote.run("", None)
        await front.run_command(f"/{remote.name}")
        root = daemon.server.supervisor.roots["remote"]
        assert any(one.type == "command/run" for one in root.session.events)


async def test_a_local_verb_never_reaches_the_daemon(tmp_path: Path) -> None:
    """The other side of the routing, and the failure it prevents.

    Sabotage: send every line to `session/command`, and `/model` comes back as
    `unknown_command` from a daemon that has no display to change.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        ran: list[str] = []

        class Recording:
            async def run_action(self, action: str) -> None:
                ran.append(action)

        front.attach_surfaces(Recording())

        await front.run_command("/model")

        assert ran == ["open_models"]
        root = daemon.server.supervisor.roots["remote"]
        assert not any(one.type == "command/run" for one in root.session.events)


# --------------------------------------------------------------- projections --


async def test_the_screens_offered_are_the_ones_this_client_can_draw(
    tmp_path: Path,
) -> None:
    """`build` is the one thing that cannot travel, so the sets are intersected.

    The daemon says which screens its profile mounted; `LOCAL_SCREENS` says which
    this build knows how to draw. An id in the first and not the second is
    dropped rather than listed and then failing to open — and a screen that *is*
    in both is built here, from the log this client already holds.
    """
    async with running(tmp_path, profile=None) as daemon:
        front, _ = await _front(daemon)

        assert front.screen("nothing-like-this") is None
        for screen_id, definition in front.screens.items():
            built = definition.build(front.session)
            assert built is not None, screen_id


async def test_the_picker_reads_no_session_file(tmp_path: Path) -> None:
    """The list is folded on the daemon; this client touches no disk.

    Both halves come back from one call: the session this front end is on is
    *live*, so its row carries the status the daemon calls it and the `cwd` from
    its own header — the repo it belongs to — even though its log is still in a
    write buffer and has never been on disk.

    That is what makes the client filesystem-free. Before this, the daemon handed
    over a *directory* and the client walked it, which held only while the two
    shared a machine and let them disagree about which `$PH_HOME` they meant.

    Sabotage: fold the live roots out of `browse_of` and a person cannot find the
    session they are sitting in.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "browsed", cwd=Path("/repos/thing"), trust="once")

        rows = await front.browse_sessions()

        row = next(one for one in rows if one.session_id == "browsed")
        assert row.state == "idle", "a live root reports what the daemon calls it"
        assert row.cwd == "/repos/thing", "and which repo it belongs to"
        assert not hasattr(front, "sessions_directory"), "no path crosses the wire any more"


# ------------------------------------------------------------------- the asks --


async def test_an_approval_from_the_daemon_reaches_this_screen(tmp_path: Path) -> None:
    """The ask direction, end to end, and the decision recorded once.

    The handler runs on the read loop, so it hands off to `ModalHost` — the same
    contract the in-process answerer lives under. `reason` travels back on the
    wire rather than being steered from here: the daemon holds the agent, and a
    client steering a turn it does not own would be writing into somebody else's
    session.
    """
    async with running(tmp_path) as daemon:
        # Held, not discarded: this front end being attached is what makes the
        # desk have somebody to ask.
        _front_end, host = await _front(daemon)
        root = daemon.server.supervisor.roots["remote"]

        outcome = await root.ctx.approval.request(
            agent=StubAgent(ctx=root.ctx, session=root.session),
            tool_name="write",
            call_id="c1",
        )

        assert outcome == "allowed-once"
        assert [one.tool_name for one in host.approvals] == ["write"]
        assert [one.type for one in root.session.events].count("approval/decided") == 1


async def test_a_question_from_the_daemon_reaches_this_screen(tmp_path: Path) -> None:
    """The other ask, and the attendance rule it turns on.

    A `DaemonSession` declares `asks` at `initialize` and attaches, so the desk
    counts it as a front end — which is what makes `ctx.user_questions.attended`
    true and the question loggable at all (P7-09).
    """
    async with running(tmp_path) as daemon:
        # Held for the same reason: `attended` is true because this is attached.
        _front_end, host = await _front(daemon)
        root = daemon.server.supervisor.roots["remote"]

        answer = await root.ctx.user_questions.ask(
            UserQuestion(question="which port?", ask_id="q1"), session=root.session
        )

        assert answer == "42"
        assert [one.question for one in host.questions] == ["which port?"]
        assert [one.type for one in root.session.events].count("question/asked") == 1


# ------------------------------------------------------ the gate this is for --


async def test_a_turn_started_here_finishes_after_this_front_end_is_gone(
    tmp_path: Path,
) -> None:
    """P5-01's promise, driven through the thing that makes it visible.

    A prompt is queued, this front end closes *without* waiting, and the root
    goes on working — then a second front end attaches and finds the finished
    turn in the transcript it rebuilds. That is the whole reason the harness moved
    into the daemon, and it is the one behaviour the in-process front end cannot
    have at all.

    `close()` detaches and does not flush or shut down: this front end is
    leaving, not ending the session. Sabotage: cancel the turn in `close`, and
    the second front end finds no assistant message.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "outlives")
        root = daemon.server.supervisor.roots["outlives"]

        await front.client.prompt("outlives", "keep going")
        await front.close()

        await until(lambda: root.status == "idle", what="the turn to finish without a client")

        second, _ = await _front(daemon, "outlives")

        assert any(item.role == "assistant" for item in second.state.items)
        assert any(one.type == "assistant/message" for one in root.session.events)


async def test_closing_a_front_end_leaves_the_root_running(tmp_path: Path) -> None:
    """Detach, not shutdown — and the root is still there to attach to.

    Sabotage: send `shutdown` from `close`, and one person closing their terminal
    stops every other person's session.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "kept")

        await front.close()

        assert "kept" in daemon.server.supervisor.roots
        assert not daemon.server.stop.is_set()
