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
from typing import Any, cast

import anyio
import pytest
from daemon_helpers import Daemon, running, until
from tui_helpers import StubApp, StubClient, StubHost, WorkingApp

from ph.bundles import BASE, HEADLESS
from ph.cordis import DEPLOYMENT, Profile, load_profile_documents, maybe_await
from ph.json import as_int
from ph.keys import APPROVAL, SKILLS, TOOLS, USER_QUESTIONS
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.commands import CommandContext, CommandSchema
from ph.seams.skills import Skill
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.testing import StubAgent
from ph_app import verbs
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.follow import Followed
from ph_app.params import CancelScheduleParams, CreateScheduleParams
from ph_app.payloads import DaemonLifetime, MutationRepeated, RepeatOutcome, StatusFacts
from ph_app.protocol import Cursor
from ph_app.tui.adapter import TuiEventAdapter
from ph_app.tui.commands import TUI_VERBS
from ph_app.tui.remote import UNKNOWN_REPEAT, DaemonSession, _remote_command, attach_session
from ph_app.tui.state import TuiState
from ph_app.tui.widgets.status import daemon_line

pytestmark = pytest.mark.anyio


async def _front(
    daemon: Daemon,
    session_id: str = "remote",
    *,
    host: StubHost | None = None,
    **options: Any,  # noqa: ANN401
) -> tuple[DaemonSession, StubHost]:
    """One attached `DaemonSession` and the host behind it.

    Typed rather than `Any`, which is what makes `StubApp` have to be a real
    `AppSurface`: while this returned `Any`, `attach_surfaces(object())`
    type-checked and the double got away with implementing one of three members
    because the other two happened not to be reached.
    """
    host = host or StubHost()
    client = await daemon.client()
    front = await attach_session(client, session_id, host=host, **options)
    return front, host


def _detached(session_id: str, *, client: Any = None) -> DaemonSession:  # noqa: ANN401
    """A `DaemonSession` with no daemon behind it, for the folds.

    **One `TuiState` between the adapter and the front end**, as `attach_session`
    builds it: the adapter folds into the state the screen reads, and a test with
    two of them asserts on an object nothing draws. Written out at each call site
    the two variants had already diverged, and `DaemonSession` gains constructor
    arguments (`generation` did) that then have to be added five times.
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


async def test_the_front_ends_log_is_a_live_mirror_not_a_rebuild(tmp_path: Path) -> None:
    """**The TUI is a log reader of the same shape as the daemon's (P6-44).**

    It used to hold a list of events and rebuild `Session(seed=…)` from it on every
    screen open — re-validating the whole log each time, and with a
    `session/end-seed` marker on the end that the daemon's log does not have. Now
    it keeps one `Session` and admits each event as it arrives, so the object a
    screen builds from is the same object across a turn, holds exactly what the
    daemon holds, and passes the same `stale()` check the daemon's own log does.
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
        # The daemon's own timestamps, not this client's clock.
        assert [e.time for e in front.session.events] == [e.time for e in root.session.events]
        # Keyed to the daemon's generation by `begin`, so a cursor made here is one
        # `resume_at` will honor rather than treat as another log's.
        assert cursor_of(front.session) == cursor_of(root.session)


def test_the_generation_is_a_constructor_argument_not_a_later_setter() -> None:
    """The mirror is keyed at birth, so there is no window in which it is not.

    `session/new` answers with this root's cursor *before* the front end is built,
    so the generation is in hand at construction. It arrived as a `begin()` that
    swapped the `Session` afterwards and raised if anything had been admitted
    first — a runtime guard against an ordering that need not exist, which is the
    shape `Session.durable_length`'s own docstring rejects for itself. A value the
    type cannot be built without is one nobody can get wrong.
    """
    front = DaemonSession(
        client=None,  # type: ignore[arg-type]
        session_id="k",
        state=TuiState(),
        adapter=TuiEventAdapter(state=TuiState()),
        host=StubHost(),
        generation=1_700_000_000_000,
    )

    assert front.session.header.created_at == 1_700_000_000_000
    assert not hasattr(front, "begin"), "no second phase to forget"
    # Driven without a daemon, it still has a valid session of its own.
    assert _detached("k").session.seq == 0


def test_a_mirror_that_missed_a_frame_says_so_rather_than_serving_a_prefix() -> None:
    """Either refusal desynchronizes the mirror permanently — a skipped frame makes
    every later seq non-contiguous — so `diverged` is the fact, and the screen path
    is what must ask. Under the rebuild this replaced, the same skip refused loudly
    at screen-open time; keeping the mirror incrementally moved the refusal earlier,
    and this is what keeps it from becoming silent."""

    front = _detached("d")
    assert not front.diverged

    # seq 1 with nothing at seq 0: the hole a dropped frame leaves.
    front._apply([({"type": "turn/start", "seq": 1, "time": 1, "data": {}}, None)], True)

    assert front.diverged, "the mirror knows it is short"
    assert front.session.seq == 0, "and did not admit a log with a gap"


def test_a_short_mirror_still_draws_what_arrives_after_the_hole() -> None:
    """The mirror and the transcript are two objects with two rules (H4).

    A *log* with a gap in it is a worse artifact than a short one, so the mirror
    refuses — but the refusal used to skip the adapter as well, and the adapter
    has no contiguity to keep. Every frame after the hole was then refused for
    following the one that was dropped, so one unreadable frame froze the
    transcript for the rest of the session: no rows, no error on screen, and a
    single log line as the whole account.

    `diverged` is how a screen that rebuilds *from the log* learns it cannot, and
    it is still set. What changes is that the person keeps seeing the
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


async def test_a_queued_prompt_is_counted_once_and_by_the_log() -> None:
    """The footer counts the inbox, and the inbox is counted by the log (H6).

    `queue` bumped the count itself *as well*, so a person's own prompt was
    counted twice — once optimistically on the keystroke and once when
    `agent/inbox/spliced` came back — and the footer read "2 queued" for one
    pending message, then fell to 1 when the turn claimed it.

    Driven at the seam rather than through a live turn, because the two halves
    have to be told apart: what the keystroke does, and what the event does. A
    real root claims an idle session's prompt in the same breath it inserts it,
    which is a race, not a reading.
    """

    front = _detached("counted", client=StubClient())

    async with anyio.create_task_group() as tasks:
        # An app that runs what it is given, so the prompt is actually sent —
        # `StubApp` closes it, which would leave the wrapper's inner coroutine
        # unawaited.
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

        front._status(StatusFacts(status="waiting"))

        assert front.state.status == "waiting", "the daemon's word is kept, once"
        assert not front.state.busy, "and the spinner stops"

        front._status(StatusFacts(status="retrying"))

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

        front._status(
            StatusFacts(status="idle", readings=[StatusReading(text="12k / 200k", level="warning")])
        )

        assert [(one.text, one.level) for one in front.status_readings()] == [
            ("12k / 200k", "warning")
        ]


async def test_the_posture_is_read_from_the_attach_reply_not_guessed(tmp_path: Path) -> None:
    """A front end draws the posture the daemon resolved, from the first frame.

    `TuiState` held `"read-only"` and moved only on a `sandbox/mode` event —
    which is appended when somebody *switches* and never at the start — so a
    session under a row that says otherwise drew the wrong posture until
    somebody changed it. The rows contribute readings now, and the attach reply
    already carries readings, so the client has the right answer before it has
    folded a single event.

    **The row says it and the log does not**, which is the only arrangement
    that gates the seed: a mode somebody changed is an event the client would
    have folded anyway.
    """
    writable = Profile.from_documents(
        [
            *load_profile_documents([BASE, HEADLESS]),
            ("test", [{"id": "sandbox", "config": {"defaultMode": "workspace-write"}}]),
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
    """What fills the context window, read once, for two renderings.

    The panel draws the names and `/tools` draws the descriptions, so both have
    to arrive — an earlier draft projected names alone and the command had
    nothing to say that the sidebar had not already said. Against the seams'
    own answers in the same mount, which is `test_daemon_projections`' rule
    applied at the other end of the wire.
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


async def test_the_command_list_is_both_ends_merged(tmp_path: Path) -> None:
    """One list to a person, two owners underneath.

    `/model` and `/theme` change *this* client's display and mean nothing to a
    daemon serving three of them; `/compact` is the harness's. So the palette
    shows the union, and `run_command` routes on which side owns the name.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon)
        front.attach_surfaces(StubApp())

        names = {definition.name for definition in front.commands()}

        local = {verb.name for verb in TUI_VERBS}

        assert names >= local, "the terminal's own verbs are offered"
        assert names - local, "and so are the daemon's"


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

        # One kind of thing in the palette, dispatched one way, whichever end
        # executes it: `run_command` reaches the remote proxy's `run` exactly as
        # it reaches a local verb's.
        await front.run_command(f"/{remote.name}")
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

        await front.run_command("/model")

        assert app.ran == ["open_models"]
        root = daemon.running.supervisor.roots["remote"]
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
    """One question, one decision, and **every** terminal told (H1).

    The desk asks every attached front end and keeps whichever answer arrives
    first; the rest are discarded, on purpose — pH appends the decision it acted
    on, and a second would be a log claiming two. `ask.settled` exists to tell
    the losers, and nothing consumed it: the other terminal's modal stayed up,
    and the person answering it had their answer dropped in silence. They were
    left believing they had refused a call that had already been allowed.

    The slow host is what makes the order a fact rather than a race: the fast one
    answers, the desk settles, and the notice reaches the slow one while its
    modal is still up — which is exactly the state the fix is about.
    """

    class SlowHost(StubHost):
        """A front end whose person has not answered yet."""

        def __init__(self) -> None:
            super().__init__()
            self.release = anyio.Event()

        async def ask_approval(
            self, request: ApprovalRequest, *, ask_id: str = ""
        ) -> tuple[ApprovalAnswer, str]:
            self.approvals.append(request)
            await self.release.wait()
            return "rejected", "too late"

    async with running(tmp_path) as daemon:
        # The local keeps the narrow type: `_front` answers `StubHost`, and this
        # test reaches for the subclass's own `release`.
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
            lambda: slow.withdrawn == ["c1"],
            what="the other terminal to be told its modal is moot",
        )
        # And the decision is recorded once, whatever the slow one does next.
        slow.release.set()
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
    """P5-01's promise, driven through the thing that makes it visible.

    A prompt is queued, this front end closes *without* waiting, and the root
    goes on working — then a second front end attaches and finds the finished
    turn in the transcript it rebuilds. That is the whole reason the harness moved
    into the daemon, and it is the one behavior the in-process front end cannot
    have at all.

    `close()` detaches and does not flush or shut down: this front end is
    leaving, not ending the session. Sabotage: cancel the turn in `close`, and
    the second front end finds no assistant message.
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
    """Detach, not shutdown — and the root is still there to attach to.

    Sabotage: send `shutdown` from `close`, and one person closing their terminal
    stops every other person's session.
    """
    async with running(tmp_path) as daemon:
        front, _ = await _front(daemon, "kept")

        await front.close()

        assert "kept" in daemon.running.supervisor.roots
        assert not daemon.running.stop.is_set()


# ---------------------------------------------- the daemon's own lifetime --
# P9-07. Three claims: a front end learns why its daemon is still running at
# attach, it is *told* when that changes, and the sidebar renders the answer as
# the sentence a person actually wants — what happens if they close this window.


def test_the_sidebar_names_the_reason_a_daemon_is_held() -> None:
    """Four shapes, and `client` dropped from every one of them.

    A pure fold, tested as one: the rendering is the whole of what a person sees
    of this row, and driving it through a terminal would assert on a picture
    where the claim is about a sentence.

    `client` is always in `holds` here — this *is* the client — so a line that
    printed it would spend a row of a 32-column panel telling somebody they have
    a window open. What survives the filter is exactly what survives their
    leaving, which is the question the line answers.

    Sabotage: keep `client` among the reasons, and every ephemeral daemon reads
    as `held · client` forever — the line that can never say anything else.
    """
    assert daemon_line(None) == [], "a front end with no daemon draws no row"
    assert daemon_line(DaemonLifetime(mode="service", holds=["client"])) == ["daemon  service"]
    assert daemon_line(DaemonLifetime(mode="ephemeral", holds=["client"], clients=1)) == [
        "daemon  exits on detach"
    ]
    # The same `holds`, and the opposite sentence: somebody else is on it, so
    # closing this window does not end anything. Without `clients` on the frame
    # these two are indistinguishable and one of them is a lie.
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
    """Read, not waited for — which is why there is a verb as well as a notice.

    `daemon.lifetime` is sent when the answer *moves*, so a front end that only
    listened would draw nothing at all until something happened: open a terminal
    on a quiet daemon and the row that says whether it stays would stay blank,
    which is exactly when a person is asking.

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
    """The frame with no owner, delivered to everyone.

    A turn on *some other* root is what changes this daemon's lifetime, and the
    notice that says so carries no `sessionId` — there is no session it is about.
    Every other notification is filtered on that field before it is parsed, which
    is right for a client watching one root among several and would drop this one
    as belonging to nobody.

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
    """The arrival is a lifetime transition too, and it was the unannounced one.

    `holds` cannot carry this on its own: it says `client` whenever anybody is
    connected, so it reads the same with one terminal and with two, while the
    sentence a person needs — "does closing this end it?" — flips. `clients` is
    the count that separates them, and a client arriving is when it moves.

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
    """The third term of `holds`, and the only one a client changes by asking.

    `schedule` moves when somebody creates or cancels one, and nothing about
    that arrives as an agent status or a connection — so the two handlers say so
    themselves. Left to the sweep, a person who had just set an appointment
    would watch the row go on claiming their daemon leaves when they close the
    window, for up to a minute, having been told otherwise by the command they
    just ran.

    The cancel is the half with teeth: it is a claim being *withdrawn*, and on a
    daemon whose only hold it was, the withdrawal is what lets the process go.

    Sabotage: drop either `check_lifetime()` from the schedule handlers, and the
    corresponding `until` here waits out its ten seconds.
    """

    def held() -> list[str]:
        return list(front.state.lifetime.holds) if front.state.lifetime else []

    async with running(tmp_path, ephemeral=True, sweep_every=600.0) as daemon:
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
    """P10-10. A repeat of a command that finished says nothing — the person
    already saw it run. One whose first attempt the daemon cannot vouch for says
    so, because the person is the one who can check whether it happened."""

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
