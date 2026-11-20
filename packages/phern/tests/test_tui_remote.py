"""P5-14: the terminal over a socket, against a real daemon.

`DaemonSession` is the implementation of `FrontSession` that ships. It was
written as the second implementation, and the in-process one has since been
deleted. The app above it reads only the protocol, so a user gets the same
layout, the same fold and the same verbs at a tty or in a browser tab. Every
test here drives the front end directly with a `StubHost`. There is no
terminal, so a failure is about the protocol and not about Textual.

The daemon is in-process (`daemon_helpers.running`) and the socket is real. That
combination is deliberate. An in-memory double for the wire would pass for a
design whose frames never make the round trip. The defects that this file exists
to catch are all about what crosses the wire:

- A snapshot page and a live frame both draw the same event.
- The daemon sends a status word that the screen has no room for.
- A verb is sent to the end that does not know it.

**The gate that the whole increment is named for is
`test_a_turn_started_here_finishes_after_this_front_end_is_gone`**: a turn that
starts here finishes after this front end is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from daemon_helpers import Daemon, running, until
from tui_helpers import StubApp, StubClient, StubHost, WorkingApp

from ph.bundles import BASE, HEADLESS
from ph.cordis import DEPLOYMENT, Profile, ProfileDocument, load_profile_documents, maybe_await
from ph.json import as_int
from ph.keys import APPROVAL, SKILLS, TOOLS, USER_QUESTIONS
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.commands import CommandContext, CommandSchema
from ph.seams.skills import Skill
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import SessionHeader
from ph.testing import StubAgent, raising
from ph_app import verbs
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.follow import Followed
from ph_app.params import CancelScheduleParams, CreateScheduleParams
from ph_app.payloads import AskKey, DaemonLifetime, MutationRepeated, RepeatOutcome, StatusFacts
from ph_app.protocol import Cursor
from ph_app.tui.adapter import TuiEventAdapter
from ph_app.tui.commands import TUI_VERBS, local_commands, run_line
from ph_app.tui.remote import UNKNOWN_REPEAT, DaemonSession, _remote_command, attach_session
from ph_app.tui.state import TuiState
from ph_app.tui.widgets.status import daemon_line

pytestmark = pytest.mark.anyio


async def _front(
    daemon: Daemon,
    session_id: str | None = "remote",
    *,
    host: StubHost | None = None,
    **options: Any,  # noqa: ANN401
) -> tuple[DaemonSession, StubHost]:
    """One attached `DaemonSession` and the host behind it.

    The return type is `DaemonSession` and not `Any`. That is why `StubApp` has
    to be a real `AppSurface`. While this function returned `Any`,
    `attach_surfaces(object())` type-checked. The double then implemented one of
    three members without a failure, because no test reached the other two.
    """
    host = host or StubHost()
    client = await daemon.client()
    front = await attach_session(client, session_id, host=host, **options)
    return front, host


def _detached(session_id: str, *, client: Any = None) -> DaemonSession:  # noqa: ANN401
    """A `DaemonSession` with no daemon behind it, for the tests of the folds.

    **The adapter and the front end share one `TuiState`**, as `attach_session`
    builds them. The adapter folds into the state that the screen reads. A test
    with two states asserts on an object that nothing draws.

    This is one function because the construction was written out at each call
    site, and the two variants had already diverged. `DaemonSession` also gains
    constructor arguments, as it gained `generation`, and each one then has to
    be added five times.
    """
    state = TuiState()
    return DaemonSession(
        client=client,
        session_id=session_id,
        state=state,
        adapter=TuiEventAdapter(state=state),
        host=StubHost(),
    )


# ------------------------------------------------------------ the transcript --


async def test_a_turn_reaches_the_transcript_over_the_socket(tmp_path: Path) -> None:
    """The whole path: prompt in, events back, one fold, one state.

    `submit` waits for the root to go idle and not for the reply. The reply
    arrives as soon as the prompt is *in the inbox*. That is why the prompt
    survives the death of this client, and why the two waits are separate.
    """
    async with running(tmp_path) as daemon:
        front, host = await _front(daemon)

        await front.submit("hello")

        assert front.state.status == "idle"
        assert [item.role for item in front.state.items][:1] == ["user"]
        assert any(item.role == "assistant" for item in front.state.items)
        assert host.redraws > 0, "the screen was told to redraw"


async def test_a_fresh_session_is_named_by_the_daemon_and_never_searched_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A front end that opens a new session sends no id, and the daemon names the
    session as it opens (protocol 8). So neither the read at the start nor the
    open searches the store for a log that cannot exist. The TUI used to make an
    id and send it. The daemon could not tell that id from a stored one, and it
    searched the store for it two times. Every later call uses the id that the
    reply carried.

    Sabotage: mint the id in `_new_session` and pass it on, and the store is searched.
    """
    from ph.persistence import families, jsonl

    searched = raising(AssertionError("searched the store for a session the daemon made"))
    monkeypatch.setattr(jsonl, "locate_under", searched)
    monkeypatch.setattr(families, "locate_under", searched)
    async with running(tmp_path) as daemon:
        front, _host = await _front(daemon, None)

        await front.submit("hello")

        assert front.session_id in daemon.running.supervisor.roots
        assert any(item.role == "assistant" for item in front.state.items)


async def test_a_front_end_attaching_to_a_finished_turn_rebuilds_it_exactly(
    tmp_path: Path,
) -> None:
    """Catch-up alone: the snapshot pages rebuild what the live stream showed.

    Two front ends on one session reach byte-identical transcripts by two
    different routes. One folded the turn as it happened, and the other paged it
    afterwards. The no-replay rule of `session/attach` depends on the agreement
    of the two routes. This test cannot create the *overlap* between them, so
    the overlap is the subject of
    `test_an_event_arriving_on_both_routes_is_folded_once`.
    """
    async with running(tmp_path) as daemon:
        first, _ = await _front(daemon, "shared")
        await first.submit("hello")
        before = [(item.role, item.text) for item in first.state.items]
        assert before, "the first front end saw the turn live"

        second, _ = await _front(daemon, "shared")

        assert [(item.role, item.text) for item in second.state.items] == before


async def test_the_front_ends_log_is_a_live_mirror_not_a_rebuild(tmp_path: Path) -> None:
    """**The TUI is a log reader of the same shape as the daemon (P6-44).**

    The front end used to hold a list of events and rebuild `Session(seed=…)`
    from it each time a screen opened. That validated the whole log again each
    time, and it put a `session/end-seed` marker on the end that the log of the
    daemon does not have. Now the front end keeps one `Session` and admits each
    event as it arrives. So the object that a screen builds from is the same
    object across a turn. It holds exactly what the daemon holds, and it passes
    the same `stale()` check as the log of the daemon.
    """
    from ph_app.protocol import cursor_of

    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "mirror")
        before = front.session
        await front.submit("hello")
        root = await daemon.root("mirror")

        assert front.session is before, "one session, extended — not rebuilt"
        assert front.session.seq == root.session.seq
        assert [e.type for e in front.session.events] == [e.type for e in root.session.events]
        assert front.session.stale() == [], "the client's incremental fold agrees with its replay"
        # The timestamps are those of the daemon, and not the clock of this client.
        assert [e.time for e in front.session.events] == [e.time for e in root.session.events]
        # The mirror is built on the header of the daemon, so it has the same
        # `cursor_of` as the daemon. A cursor that is made here is therefore one
        # that `resume_at` honors, and does not treat as the cursor of another log.
        assert cursor_of(front.session) == cursor_of(root.session)


def test_the_mirror_is_built_on_the_daemons_header() -> None:
    """The mirror is keyed when it is built, on the header of the root itself.

    `session/new` answers with the header *before* the front end is built, so
    the header is available at construction. It used to arrive in a `begin()`
    that replaced the `Session` afterwards and raised if anything had been
    admitted first. That was a runtime guard against an ordering that need not
    exist. The header was also one that the client made, keyed by the generation
    of the reply. So the mirror of a forked root had no `seed_length`, and every
    fold over the events that the log itself owns counted the inherited history
    of the fork as part of them.

    Sabotage: build the mirror's header from the id and a generation again, and the
    fork's boundary is gone.
    """
    header = SessionHeader(
        id="k", created_at=1_700_000_000_000, parent_session="p", seed_length=3, family="p"
    )
    front = DaemonSession(
        client=None,  # type: ignore[arg-type]
        session_id="k",
        state=TuiState(),
        adapter=TuiEventAdapter(state=TuiState()),
        host=StubHost(),
        header=header,
    )

    assert front.session.header == header
    assert front.session.header.first_own_seq == 3
    assert not hasattr(front, "begin"), "no second phase to forget"
    # Driven without a daemon, it still has a valid session of its own.
    assert _detached("k").session.seq == 0


def test_a_mirror_that_missed_a_frame_says_so_rather_than_serving_a_prefix() -> None:
    """A frame that will not rebuild and a frame that the mirror will not admit
    both desynchronize the mirror permanently, because a skipped frame makes
    every later seq non-contiguous. So `diverged` is the fact, and the path that
    opens a screen is the one that must ask about it.

    Under the rebuild that this replaced, the same skip was refused loudly when
    a screen opened. Keeping the mirror incrementally moved the refusal earlier,
    and this is what stops the refusal from becoming silent.
    """

    front = _detached("d")
    assert not front.diverged

    # seq 1 with nothing at seq 0: the hole a dropped frame leaves.
    front._apply([({"type": "turn/start", "seq": 1, "time": 1, "data": {}}, None)], True)

    assert front.diverged, "the mirror knows it is short"
    assert front.session.seq == 0, "and did not admit a log with a gap"


def test_a_short_mirror_still_draws_what_arrives_after_the_hole() -> None:
    """The mirror and the transcript are two objects with two rules (H4).

    A *log* with a gap in it is a worse artifact than a short log, so the mirror
    refuses the frame. But the refusal used to skip the adapter as well, and the
    adapter has no contiguity to keep. Every frame after the hole was then
    refused because it followed the dropped frame. So one unreadable frame froze
    the transcript for the rest of the session. No rows were drawn, no error was
    on screen, and a single log line was the whole account.

    `diverged` is how a screen that rebuilds *from the log* learns that it
    cannot, and it is still set. The change is that the user keeps seeing the
    conversation.
    """
    front = _detached("d")

    # A hole at seq 0, then an ordinary message behind it.
    front._apply(
        [
            ({"type": "turn/start", "seq": 1, "time": 1, "data": {}}, None),
            (
                {
                    "type": "user/message",
                    "seq": 2,
                    "time": 2,
                    "surfaceOp": "append",
                    "data": {
                        "id": "m1",
                        "role": "user",
                        "content": [{"type": "text", "text": "still here"}],
                        "source": {"kind": "user"},
                    },
                },
                None,
            ),
        ],
        True,
    )

    assert front.diverged, "the mirror still knows it is short"
    assert front.session.seq == 0, "and still refuses a log with a gap"
    assert [item.text for item in front.state.items] == ["still here"], (
        "the transcript froze behind the hole"
    )


async def test_an_event_arriving_on_both_routes_is_folded_once() -> None:
    """The purpose of the buffer, and the one window that can show an event twice.

    `session/attach` subscribes *before* the history is paged, so live frames
    arrive while `session/snapshot` is still running. An event at the head of
    the log can therefore arrive by both routes. The test drives `Followed`
    directly, because a real turn against the fake provider finishes faster
    than the overlap in which it must be caught. An integration test here would
    pass whether or not the rule existed, and it did so before this test was
    written.

    Sabotage: drop the `at <= self.seen` check in `Followed`, and seq 2 folds twice.
    """
    folded: list[int] = []
    feed = Followed(
        session_id="s",
        on_events=lambda pairs, _live: folded.extend(
            as_int(one.get("seq"), -1) for one, _ in pairs
        ),
        on_status=lambda _facts: None,
    )

    # A live frame arrives during catch-up and is held.
    feed("session.event", {"sessionId": "s", "event": {"seq": 2, "type": "turn/end"}})
    # The page that follows already contains it.
    feed.on_events(
        [({"seq": 1, "type": "turn/start"}, None), ({"seq": 2, "type": "turn/end"}, None)],
        False,
    )
    feed.seen = 2
    feed.live()

    assert folded == [1, 2], "seq 2 came down both routes and was folded once"


async def test_a_feed_that_has_shown_nothing_shows_the_first_event() -> None:
    """The first event of a log is seq 0, and a new feed must show it.

    `Followed.seen` is `None` until an event is shown. When it started at `0`,
    "nothing shown" and "seq 0 shown" were the same value, and the opening event
    of every new session was dropped as already seen.

    Sabotage: start `seen` at `0`, and the opening event is not shown.
    """
    shown: list[int] = []
    feed = Followed(
        session_id="s",
        on_events=lambda pairs, _live: shown.extend(as_int(one.get("seq"), -1) for one, _ in pairs),
        on_status=lambda _facts: None,
    )
    assert feed.seen is None, "a new feed has shown nothing"
    feed.live()

    opening = {"sessionId": "s", "event": {"seq": 0, "type": "session/start"}}
    feed("session.event", opening)
    feed("session.event", opening)
    # An event with no readable seq is dropped, and it does not move the mark.
    feed("session.event", {"sessionId": "s", "event": {"type": "turn/end"}})

    assert shown == [0], "seq 0 is shown one time"
    assert feed.seen == 0


async def test_a_caught_up_page_is_folded_as_history_and_a_frame_as_live(
    tmp_path: Path,
) -> None:
    """The fold is told in which phase an event arrived, and does not assume it.

    `TuiEventAdapter` has taken `Frame(live=…)` since P3, because a transcript
    that is *rebuilt* is not a transcript that is *streamed*. A page of history
    holds the `assistant/chunk` records of a turn and the `assistant/message`
    that superseded them. A fold that was told that they were live builds a
    streaming row and then replaces it inside a single pass. In Textual that is
    a widget mount inside a widget mount, and it appeared as a `MountError` from
    an unrelated test.

    The assertion is on the contract and not on the symptom, because the symptom
    is a race: the burst has to be big enough, and the frame boundary has to
    fall in the wrong place. Sabotage: pass `True` from `catch_up`, and this
    fails every time while the resume test passes most of the time.
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
    """The multiplex rule: a submitted prompt is a log entry that everyone sees.

    The composer is not shared, and text that was not submitted never leaves a
    client. But pressing enter is an act in the session, so the prompt reaches
    every attached front end by the one route that everything else uses.
    """
    async with running(tmp_path) as daemon:
        first, _ = await _front(daemon, "both")
        second, _ = await _front(daemon, "both")

        await first.submit("from the first")

        await until(
            lambda: any("from the first" in (item.text or "") for item in second.state.items),
            what="the other front end to see the prompt",
        )


async def test_a_queued_prompt_is_counted_once_and_by_the_log() -> None:
    """The footer shows the count of the inbox, and the log supplies that count (H6).

    `queue` used to increase the count itself *as well*, so the prompt of the
    user was counted two times: optimistically on the keystroke, and again when
    `agent/inbox/spliced` came back. The footer read "2 queued" for one pending
    message, and then fell to 1 when the turn claimed it.

    The test drives the seam and not a live turn, because the two halves have to
    be told apart: what the keystroke does, and what the event does. A real root
    claims the prompt of an idle session at the moment when it inserts it. That
    is a race and not a reading.
    """

    front = _detached("counted", client=StubClient())

    async with anyio.create_task_group() as tasks:
        # This app runs the work that it is given, so the prompt is really sent.
        # `StubApp` closes the work, which would leave the inner coroutine of
        # the wrapper unawaited.
        front.attach_surfaces(WorkingApp(tasks))
        front.queue("while you are at it")

    assert front.state.queued == 0, "the keystroke is not the account; the log is"

    # And the event the daemon sends back for that same prompt.
    front._apply(
        [
            (
                {
                    "type": "agent/inbox/spliced",
                    "seq": 0,
                    "time": 1,
                    "data": {"inserted": [{"target": "next-turn"}]},
                },
                None,
            )
        ],
        True,
    )

    assert front.state.queued == 1, "one pending message, counted once"


# ---------------------------------------------------------------- the words --


async def test_a_root_parked_on_a_person_is_not_shown_as_running(tmp_path: Path) -> None:
    """Every status word of the root is in one field, and the widgets read a bool.

    `TuiState.status` carries the status word of the root itself, `waiting` and
    `retrying` included. `TuiState.busy` is what drives the spinner. The
    alternative was a second status field on the remote front end, kept in step
    by hand. That made three writers of one fact, and an in-process screen that
    could never show `retrying` because its type had no room for it.

    Sabotage: make `busy` `status == "running"`, and a retry shows as idle.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        front._status(StatusFacts(status="waiting"))

        assert front.state.status == "waiting", "the daemon's word is kept, once"
        assert not front.state.busy, "and the spinner stops"

        front._status(StatusFacts(status="retrying"))

        assert front.state.busy, "a retry is still work in flight"


async def test_the_footer_arrives_beside_the_status(tmp_path: Path) -> None:
    """The readings are pushed with the status, and not polled on the 30 Hz tick.

    A reading is a fold of the log, so the moment to read the readings again is
    the moment when the agent moved. They are rebuilt through `model_validate`
    and not field by field. A reading that gains a field therefore reaches a
    browser tab with no edit at this end. `to_wire` makes the same argument in
    the other direction.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        front._status(
            StatusFacts(status="idle", readings=[StatusReading(text="12k / 200k", level="warning")])
        )

        assert [(one.text, one.level) for one in front.status_readings()] == [
            ("12k / 200k", "warning")
        ]


async def test_the_posture_is_read_from_the_attach_reply_not_guessed(tmp_path: Path) -> None:
    """A front end draws the posture that the daemon resolved, from the first frame.

    `TuiState` used to hold `"read-only"`, and it changed only on a
    `sandbox/mode` event. That event is appended when somebody *switches* the
    mode and never at the start. So a session under a row that says otherwise
    drew the wrong posture until somebody changed it. The rows now contribute
    readings, and the attach reply already carries readings. The client
    therefore has the right answer before it has folded a single event.

    **The row says the posture and the log does not.** That is the only
    arrangement that tests the seed, because a mode that somebody changed is an
    event that the client would have folded anyway.
    """
    writable = Profile.from_documents(
        [
            *load_profile_documents([BASE, HEADLESS]),
            ProfileDocument(
                "test", [{"id": "sandbox", "config": {"defaultMode": "workspace-write"}}]
            ),
        ]
    )
    async with running(tmp_path, profile=writable) as daemon:
        await daemon.root("remote")

        front, _ = await _front(daemon)

        drawn = {one.id: one.text for one in front.status_readings()}
        assert drawn["sandbox-mode"] == "sandbox workspace-write", (
            "the client must not fall back to a posture it compiled in"
        )
        assert drawn["posture"] == "read-only accepted"


async def test_the_catalogs_are_read_at_attach_and_carry_their_descriptions(
    tmp_path: Path,
) -> None:
    """What fills the context window is read one time, for two renderings.

    The panel draws the names and `/tools` draws the descriptions, so both must
    arrive. An earlier draft projected the names alone, and the command then had
    nothing to say that the sidebar had not already said. The check is against
    the answers of the seams themselves in the same mount. That is the rule of
    `test_daemon_projections`, applied at the other end of the wire.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("remote")
        root.ctx.require(SKILLS).register(
            Skill(name="code-review", description="what to look for in a diff")
        )

        front, _ = await _front(daemon)

        offered = root.ctx.require(TOOLS).schemas(scope=DEPLOYMENT)
        assert [one.name for one in front.state.tools] == [one.name for one in offered]
        assert [one.description for one in front.state.tools] == [
            one.description for one in offered
        ]
        assert ("code-review", "what to look for in a diff") in [
            (one.name, one.description) for one in front.state.skills
        ]


# ----------------------------------------------------------------- the verbs --


async def test_a_front_end_offers_the_daemon_s_verbs_and_not_the_terminal_s(
    tmp_path: Path,
) -> None:
    """The user sees one list, and two owners are behind it.

    `/model` and `/theme` change the display of *this* client, and they mean
    nothing to a daemon that serves three clients. `/compact` belongs to the
    harness. The verbs of the terminal belong to the app, so they answer with no
    front end at all (H3). A front end offers the verbs of the daemon, and the
    app puts the two lists together (`PHTuiApp.commands`).

    Sabotage: build `local_commands(app)` back into `_wire_screens`, and the
    terminal's verbs are offered twice once a front end attaches.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        front.attach_surfaces(StubApp())

        names = {definition.name for definition in front.commands()}

        local = {verb.name for verb in TUI_VERBS}

        assert names, "the daemon's verbs are offered"
        assert not names & local, "and the terminal's own are not this front end's"


async def test_a_daemon_verb_is_dispatched_over_the_wire(tmp_path: Path) -> None:
    """The remote half of the merge: the body of a daemon verb *is* `session/command`.

    A command that is registered in the `ctx.commands` of the daemon cannot be
    executed here, because its body closes over services in another process. So
    the definition that this client holds for it has a `run` that sends the line
    across. That is what lets `run_line` dispatch every verb in the same way.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        local = {verb.name for verb in TUI_VERBS}
        remote = next(one for one in front.commands() if one.name not in local)

        # The palette holds one kind of thing, and it is dispatched in one way,
        # whichever end executes it. `run_line` reaches the `run` of the remote
        # proxy exactly as it reaches the `run` of a local verb.
        await run_line(front.commands(), f"/{remote.name}")
        root = daemon.running.supervisor.roots["remote"]
        assert any(one.type == "command/run" for one in root.session.events)


async def test_a_local_verb_never_reaches_the_daemon(tmp_path: Path) -> None:
    """The other side of the routing, and the failure it prevents.

    Sabotage: send every line to `session/command`, and `/model` comes back as
    `unknown_command` from a daemon that has no display to change.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        app = StubApp()
        front.attach_surfaces(app)

        # This is the list that `PHTuiApp.commands` dispatches through: the
        # commands of the app, then the commands of the front end.
        await run_line([*local_commands(app), *front.commands()], "/model")

        assert app.ran == ["open_models"]
        root = daemon.running.supervisor.roots["remote"]
        assert not any(one.type == "command/run" for one in root.session.events)


# --------------------------------------------------------------- projections --


async def test_the_screens_offered_are_the_ones_this_client_can_draw(
    tmp_path: Path,
) -> None:
    """`build` is the one thing that cannot travel, so the two sets are intersected.

    The daemon says which screens its profile mounted. `LOCAL_SCREENS` says
    which screens this build can draw. An id that is in the first set and not in
    the second is dropped. It is not listed and then left to fail when it is
    opened. A screen that *is* in both sets is built here, from the log that
    this client already holds.
    """
    async with running(tmp_path, profile=None) as daemon:
        front, _ = await _front(daemon)

        assert front.screen("nothing-like-this") is None
        for screen_id, definition in front.screens.items():
            built = definition.build(front.session)
            assert built is not None, screen_id


async def test_the_picker_reads_no_session_file(tmp_path: Path) -> None:
    """The daemon folds the list, and this client touches no disk.

    Both halves come back from one call. The session that this front end is on
    is *live*. So its row carries the status that the daemon reports for it and
    the `cwd` from its own header, which is the repo that it belongs to. That
    holds although its log is still in a write buffer and has never been on
    disk.

    That is what makes the client independent of the filesystem. Before this,
    the daemon handed over a *directory* and the client walked it. That worked
    only while the two shared a machine, and it let them disagree about which
    `$PH_HOME` they meant.

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
    """The ask direction, end to end, with the decision recorded one time.

    The handler runs on the read loop, so it hands the ask to `ModalHost`. That
    is the same contract that the in-process answerer works under. `reason` goes
    back on the wire, and this client does not steer with it. The daemon holds
    the agent, and a client that steered a turn which it does not own would
    write into the session of somebody else.
    """
    async with running(tmp_path) as daemon:
        # The front end is held and not discarded. The desk has somebody to ask
        # because this front end is attached.
        _front_end, host = await _front(daemon)
        root = daemon.running.supervisor.roots["remote"]

        outcome = await root.ctx.require(APPROVAL).request(
            agent=StubAgent(ctx=root.ctx, session=root.session),
            tool_name="write",
            call_id="c1",
        )

        assert outcome == "allowed-once"
        assert [one.tool_name for one in host.approvals] == ["write"]
        assert [one.type for one in root.session.events].count("approval/decided") == 1


async def test_a_second_terminals_modal_comes_down_when_the_first_answers(
    tmp_path: Path,
) -> None:
    """One question, one decision, and **every** terminal is told (H1).

    The desk asks every attached front end and keeps the answer that arrives
    first. It discards the others on purpose: pH appends the decision that it
    acted on, and a second decision would make the log claim two. `ask.settled`
    exists to tell the front ends whose answers were discarded, and nothing
    consumed it. The modal of the other terminal stayed up, and the answer of
    the person who answered it was dropped silently. That person was left to
    believe that they had refused a call which had already been allowed.

    The slow host makes the order a fact and not a race. The fast host answers,
    the desk settles, and the notice reaches the slow host while its modal is
    still up. That is exactly the state that the fix is about.
    """

    class SlowHost(StubHost):
        """A front end whose user has not answered yet."""

        def __init__(self) -> None:
            super().__init__()
            self.release = anyio.Event()
            self.asked_under: list[AskKey | None] = []

        async def ask_approval(
            self, request: ApprovalRequest, *, ask: AskKey | None = None
        ) -> tuple[ApprovalAnswer, str]:
            self.approvals.append(request)
            self.asked_under.append(ask)
            await self.release.wait()
            return "rejected", "too late"

    async with running(tmp_path) as daemon:
        # The local variable keeps the narrow type. `_front` returns a
        # `StubHost`, and this test needs the `release` of the subclass.
        slow = SlowHost()
        _slow_front, _ = await _front(daemon, "remote", host=slow)
        _fast_front, fast = await _front(daemon, "remote")
        root = daemon.running.supervisor.roots["remote"]

        outcome = await root.ctx.require(APPROVAL).request(
            agent=StubAgent(ctx=root.ctx, session=root.session),
            tool_name="write",
            call_id="c1",
        )

        assert outcome == "allowed-once", "the fast terminal's answer is the decision"
        assert [one.tool_name for one in fast.approvals] == ["write"]
        await until(
            lambda: bool(slow.withdrawn),
            what="the other terminal to be told its modal is moot",
        )
        assert slow.withdrawn == slow.asked_under, "the modal it was asked under"
        # The decision is recorded one time, whatever the slow host does next.
        slow.release.set()
        assert [one.type for one in root.session.events].count("approval/decided") == 1


async def test_a_question_from_the_daemon_reaches_this_screen(tmp_path: Path) -> None:
    """The other ask, and the attendance rule that it depends on.

    A `DaemonSession` declares `asks` at `initialize` and attaches, so the desk
    counts it as a front end. That is what makes `ctx.user_questions.attended`
    true, and what lets the question be logged at all (P7-09).
    """
    async with running(tmp_path) as daemon:
        # Held for the same reason: `attended` is true because this is attached.
        _front_end, host = await _front(daemon)
        root = daemon.running.supervisor.roots["remote"]

        outcome = await root.ctx.require(USER_QUESTIONS).ask(
            UserQuestion(question="which port?", ask_id="q1"), session=root.session
        )

        assert outcome.resolution == "answered"
        assert outcome.answer == "42"
        assert [one.question for one in host.questions] == ["which port?"]
        assert [one.type for one in root.session.events].count("question/asked") == 1


# ------------------------------------------------------ the gate this is for --


async def test_a_turn_started_here_finishes_after_this_front_end_is_gone(
    tmp_path: Path,
) -> None:
    """The promise of P5-01, driven through the thing that makes it visible.

    A prompt is queued, this front end closes *without* waiting, and the root
    goes on working. Then a second front end attaches and finds the finished
    turn in the transcript that it rebuilds. That is the reason why the harness
    moved into the daemon, and it is the one behavior that the in-process front
    end could not have at all.

    `close()` detaches, and it does not flush or shut down: this front end is
    leaving, and it is not ending the session. Sabotage: cancel the turn in
    `close`, and the second front end finds no assistant message.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "outlives")
        root = daemon.running.supervisor.roots["outlives"]

        await front.client.prompt("outlives", "keep going")
        await front.close()

        await until(lambda: root.status == "idle", what="the turn to finish without a client")

        second, _ = await _front(daemon, "outlives")

        assert any(item.role == "assistant" for item in second.state.items)
        assert any(one.type == "assistant/message" for one in root.session.events)


async def test_closing_a_front_end_leaves_the_root_running(tmp_path: Path) -> None:
    """A close detaches and does not shut down. The root is still there to
    attach to.

    Sabotage: send `shutdown` from `close`, and one person closing their terminal
    stops every other person's session.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "kept")

        await front.close()

        assert "kept" in daemon.running.supervisor.roots
        assert not daemon.running.stop.is_set()


# ---------------------------------------------- the daemon's own lifetime --
# P9-07. Three claims:
# - A front end learns at attach why its daemon is still running.
# - It is *told* when that changes.
# - The sidebar renders the answer as the sentence that a person wants: what
#   happens if they close this window.


def test_the_sidebar_names_the_reason_a_daemon_is_held() -> None:
    """Four shapes. `client` is dropped unless another client is connected.

    This is a pure fold, and it is tested as one. The rendering is everything
    that a person sees of this row. To drive it through a terminal would assert
    on a picture, where the claim is about a sentence.

    `client` is always in `holds` here, because this *is* the client. A line
    that always printed it would spend a row of a 32-column panel to tell
    somebody that they have a window open. With one client connected, what
    passes the filter is exactly what remains after that person leaves, and
    that is the question that the line answers.

    Sabotage: keep `client` among the reasons, and every ephemeral daemon reads
    as `held · client` forever — the line that can never say anything else.
    """
    assert daemon_line(None) == [], "a front end with no daemon draws no row"
    assert daemon_line(DaemonLifetime(mode="service", holds=["client"])) == ["daemon  service"]
    assert daemon_line(DaemonLifetime(mode="ephemeral", holds=["client"], clients=1)) == [
        "daemon  exits on detach"
    ]
    # The same `holds`, and the opposite sentence. Somebody else is on the
    # daemon, so closing this window does not end anything. Without `clients` on
    # the frame, the two cases look the same and one of the two lines is false.
    assert daemon_line(DaemonLifetime(mode="ephemeral", holds=["client"], clients=2)) == [
        "daemon  held · client"
    ]
    assert daemon_line(
        DaemonLifetime(mode="ephemeral", holds=["client"], keep_alive_ms=300_000)
    ) == ["daemon  exits 5m after detach"]
    assert daemon_line(DaemonLifetime(mode="ephemeral", holds=["client", "task", "schedule"])) == [
        "daemon  held · task · schedule"
    ]


async def test_a_client_reads_the_lifetime_at_attach(tmp_path: Path) -> None:
    """Read, and not waited for. That is why there is a verb as well as a notice.

    `daemon.lifetime` is sent when the answer *changes*. A front end that only
    listened would therefore draw nothing until something happened. On a quiet
    daemon, the row that says whether the daemon stays would be blank in a new
    terminal, which is exactly when a person asks.

    Sabotage: drop the read from `attach_session` and this is `None`.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        life = front.state.lifetime
        assert life is not None, "the attach did not read it"
        assert life.mode == "service", "the fixture's daemon is one somebody started"
        assert "client" in life.holds, "and this front end is why it is held"


async def test_a_lifetime_notice_reaches_a_client_watching_another_session(
    tmp_path: Path,
) -> None:
    """The frame with no owner is delivered to everyone.

    A turn on *some other* root is what changes the lifetime of this daemon. The
    notice that says so carries no `sessionId`, because it is about no session.
    Every other notification is filtered on that field before it is parsed. That
    is right for a client that watches one root among several, but the filter
    would drop this notice because it belongs to nobody.

    Sabotage: route it through the `sessionId` filter — move the
    `DaemonLifetime` branch below `params.get("sessionId") != self.session_id`
    — and the sidebar never learns that anything holds the process.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "watched")
        assert front.state.lifetime is not None
        assert "task" not in front.state.lifetime.holds, "nothing is running yet"

        await daemon.busy_root("somewhere-else")
        daemon.running.check_lifetime()

        await until(
            lambda: "task" in (front.state.lifetime.holds if front.state.lifetime else []),
            what="the lifetime frame to arrive",
        )


async def test_a_second_terminal_arriving_tells_the_first_it_is_not_alone(
    tmp_path: Path,
) -> None:
    """An arrival is a lifetime transition too, and it was the unannounced one.

    `holds` cannot carry this alone. It says `client` whenever anybody is
    connected, so it reads the same with one terminal and with two. But the
    answer to the question that a person asks ("does closing this end it?")
    changes. `clients` is the count that separates the two cases, and it changes
    when a client arrives.

    Sabotage: leave the announcement to the *departure* alone, as it was, and
    the first terminal goes on promising `exits on detach` with a second one
    open on the same daemon.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        front, _ = await _front(daemon, "first")
        assert front.state.lifetime is not None
        assert front.state.lifetime.clients == 1, "nobody else is here yet"

        await daemon.client("asks")

        await until(
            lambda: (front.state.lifetime.clients if front.state.lifetime else 0) == 2,
            what="the second client to be announced",
        )
        assert daemon_line(front.state.lifetime) == ["daemon  held · client"]


async def test_setting_and_clearing_an_appointment_moves_the_line_at_once(
    tmp_path: Path,
) -> None:
    """The third term of `holds`, and the only one that a client changes by asking.

    `schedule` changes when somebody creates or cancels a schedule. Nothing
    about that arrives as an agent status or a connection. So the schedule watch
    reports it itself, from the log, whichever path wrote the record. Otherwise
    a person who had just set an appointment would see the row still claim that
    the daemon leaves when they close the window, after the command that they
    had just run told them the opposite.

    The cancel is the half with the larger consequence. It is a claim that is
    *withdrawn*. On a daemon whose only hold was that claim, the withdrawal is
    what lets the process go.

    Sabotage: drop `_recheck()` from `Supervisor._watch_schedules`, and each
    `until` here waits out its ten seconds.
    """

    def held() -> list[str]:
        return list(front.state.lifetime.holds) if front.state.lifetime else []

    async with running(tmp_path, ephemeral=True) as daemon:
        front, _ = await _front(daemon, "planner")
        assert "schedule" not in held(), "nothing is on the books yet"

        await front.client.call(
            verbs.SCHEDULE_CREATE,
            CreateScheduleParams(
                session_id="planner",
                schedule_id="nightly",
                kind="interval",
                spec="60000",
                prompt="do the thing",
            ),
        )
        await until(lambda: "schedule" in held(), what="the appointment to reach the line")

        await front.client.call(
            verbs.SCHEDULE_CANCEL,
            CancelScheduleParams(session_id="planner", schedule_id="nightly"),
        )
        await until(lambda: "schedule" not in held(), what="the withdrawal to reach the line")


@pytest.mark.parametrize(("outcome", "said"), [("unknown", True), ("settled", False)])
async def test_a_re_sent_command_whose_outcome_is_unknown_says_so(
    outcome: RepeatOutcome, said: bool
) -> None:
    """P10-10. A repeat of a command that finished says nothing, because the user
    already saw it run. A repeat of a command whose first attempt the daemon
    cannot vouch for says so, because the user is the one who can check whether
    it happened."""

    class Repeating:
        async def mutate(self, verb: object, params: object) -> MutationRepeated:
            return MutationRepeated(
                session_id="s",
                status="idle",
                watchers=0,
                cursor=Cursor(generation="1", sequence=0),
                outcome=outcome,
            )

    command = _remote_command(
        cast(DaemonClient, Repeating()), "s", CommandSchema(name="probe", summary="a probe")
    )
    shown = await maybe_await(command.run("", cast(CommandContext, None)))
    assert shown == (UNKNOWN_REPEAT if said else None)


# ----------------------------------------- the route of a root is one fact --


async def test_a_frame_that_does_not_state_the_route_leaves_the_route_alone() -> None:
    """Most status frames say where the root is and nothing about its route.

    A frame states the route when it carries `model` (`StatusFacts`). A frame
    without it must leave all three fields of the route as the client had them.

    Sabotage: assign the three route fields outside the `facts.model` test in
    `_status`, and a `running` frame blanks the footer.
    """
    front = _detached("s")
    front._status(StatusFacts(status="idle", provider="fake", model="fake-1", model_key="main"))

    front._status(StatusFacts(status="running"))
    front._status(StatusFacts(readings=[]))

    state = front.state
    assert (state.provider, state.model, state.model_key) == ("fake", "fake-1", "main")
    assert state.route == "fake/fake-1"


async def test_a_route_with_no_listed_name_clears_the_name() -> None:
    """A frame that states a route with no `model_key` gives an answer: the
    profile lists no name for that route.

    The name of the route before it must not survive. The model picker would
    otherwise mark a listed model as in force when none is.

    Sabotage: keep the old `model_key` in `_status` when the frame has none, and
    the unlisted route goes by the name of the route before it.
    """
    front = _detached("s")
    front._status(StatusFacts(provider="fake", model="fake-1", model_key="main"))

    front._status(StatusFacts(provider="fake", model="fake-9"))

    state = front.state
    assert (state.provider, state.model, state.model_key) == ("fake", "fake-9", None)


async def test_a_name_with_no_model_is_not_a_statement_of_the_route() -> None:
    """The three fields of the route are read together, and `model` is what says
    that a frame states them.

    A `model_key` alone is therefore not applied. Applied on its own, it would
    give the route in force the name of another route.

    Sabotage: read `facts.model_key` under its own `is not None` test in
    `_status`, and the name moves without the route.
    """
    front = _detached("s")
    front._status(StatusFacts(provider="fake", model="fake-1", model_key="main"))

    front._status(StatusFacts(model_key="other"))

    state = front.state
    assert (state.provider, state.model, state.model_key) == ("fake", "fake-1", "main")


async def test_a_frame_that_states_no_status_leaves_the_status_alone() -> None:
    """A footer frame arrives while a turn runs, and the spinner must keep
    turning.

    Sabotage: assign `facts.status` in `_status` without the `is not None` test,
    and a footer frame ends the spinner.
    """
    front = _detached("s")
    front._status(StatusFacts(status="running"))

    front._status(StatusFacts(readings=[]))

    assert front.state.status == "running" and front.state.busy


async def test_a_front_end_with_nothing_to_tell_has_no_arrival(tmp_path: Path) -> None:
    """The attach reply of a root whose profile has not moved carries no note.

    The app shows the note as a toast when there is one. A note of `""` would be
    an empty toast at every attach.

    Sabotage: describe a root whose profile has not moved with a `profile_note`
    of `""` in `Root.describe`, and the front end has an arrival to show.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)

        assert front.arrival is None
