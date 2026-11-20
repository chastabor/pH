"""The terminal as a daemon client: the same screen, with the harness elsewhere (P5-14).

The terminal used to mount pH in its own process, so closing the terminal ended
the turn. This module replaced that. `DaemonSession` is a `FrontSession` in
which every member answers by asking the daemon. The app above it reads only the
protocol. For that reason one layout, one fold and one set of verbs serve both a
tty and a browser tab.

Four things here are decisions and not plumbing.

**A modal is awaited in a worker, never on the pump.** The asks of the daemon
arrive as `approval/ask`, `question/ask` and `profile/ask` on the read loop,
where `push_screen_wait` is illegal. So the handlers pass each ask to
`ModalHost`, which the app implements inside a Textual worker. That is the
reason why `ModalHost` exists and the app is not called back directly.

**Verbs come from both ends, merged into one list with one dispatch.** `/model`
and `/theme` belong to the terminal. They change the display of *this* client,
and they mean nothing to a daemon that serves three clients. `/compact` and
`/profile` belong to the harness. Each remote verb is a `CommandDefinition`
whose `run` sends `session/command`. So `run_line` finds a name and calls `run`,
and it does not know which side owns the verb.

**The `Session` is a mirror, and it is extended as events arrive.** Only one
thing wants it: `ScreenDefinition.build(session)`, when somebody opens a screen.
Even so, the mirror is kept whole and incrementally, and it is not rebuilt on
demand.

That reverses an earlier decision, and the objection that it had to answer is
worth keeping: *"`Session` has no way to admit an already-numbered event, and
inventing one would be a second append path into the type whose whole contract
is that appends are its own."* The answer is that `admit` is not a second append
path. It shares `_commit` with `_append`. The surface validation, the push and
the publish are therefore one tail, and an observer cannot tell the two paths
apart. The only difference is who stamped the event, and that is the whole
distinction between owning a log and mirroring one. The rebuild cost a full
`Session(seed=…)` for each screen open, with every event frozen and validated
again. It also added a `session/end-seed` marker that the log of the daemon
does not have. The id is on the protocol separately, so the sidebar never asks
for the whole `Session`.

**The `build` of a screen is the one thing that cannot travel**, as
`ScreenDefinition` already says. Each screen that pH ships exports a
`CLIENT_SIDE` definition from its own module. `screens/list` names what the
deployment mounted, and the user is offered the intersection. Rule 6: a screen
that a third-party row contributes does not reach a remote front end at all.
The declarative bodies of P7-07 are what will close that gap.

@module ph_app.tui.remote
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Coroutine, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio
from textual.binding import Binding

from ph.cancel import first_of
from ph.llm.types import AttachmentRef
from ph.seams.approval import answer_to_wire
from ph.seams.attachments import read_for_attach
from ph.seams.commands import (
    CommandDefinition,
    CommandSchema,
)
from ph.seams.models import ModelChoice, ModelEntry
from ph.seams.permission_presets import PresetName, PresetSchema
from ph.seams.tui_screens import ScreenDefinition, ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.session import Session, SessionEvent, SessionHeader
from ph.wire import WireModel

from .. import verbs
from ..attach import Tray, stage_bytes
from ..daemon.client import DaemonClient
from ..daemon.duplex import answering
from ..daemon.follow import EventFrame, Followed
from ..params import (
    BrowseParams,
    CommandParams,
    ModelParams,
    NewSessionParams,
    PresetParams,
    ShellParams,
    StoreCredentialParams,
    TrustAnswer,
)
from ..payloads import (
    FED,
    ApprovalAsk,
    ApprovalAskReply,
    AskSettledNotice,
    CommandShown,
    DaemonLifetime,
    MutationRepeated,
    ProfileAsk,
    ProfileAskReply,
    QuestionAsk,
    QuestionAskReply,
    RootDescription,
    SessionChildrenNotice,
    SessionCommandsNotice,
    SessionModelsReply,
    SessionScreensNotice,
    SessionSkillsReply,
    SessionStagedNotice,
    SessionToolsReply,
    StatusFacts,
    notice_of,
)
from ..protocol import DaemonGone, NoParams, SessionParams, Verb
from ..sessions import SessionSummary
from ..wire import view_of
from .adapter import Frame, TuiEventAdapter
from .commands import action_command
from .diff_screen import CLIENT_SIDE as DIFF
from .frontend import ModalHost
from .screens import AppSurface, open_screen_action
from .state import CatalogEntry, Surface, TuiState
from .trajectory_screen import CLIENT_SIDE as TRAJECTORY

__all__ = ["LOCAL_SCREENS", "UNKNOWN_REPEAT", "DaemonSession", "attach_session"]

UNKNOWN_REPEAT = (
    "This command was sent before and the daemon cannot say whether it finished — "
    "it may have stopped partway. Check its effect before running it again."
)
"""What a re-sent slash command says when the outcome of its first attempt is unknown."""

log = logging.getLogger("ph_app.tui.remote")

LOCAL_SCREENS: tuple[ScreenDefinition, ...] = (TRAJECTORY, DIFF)
"""Every screen this build can draw without a harness. See the module docstring."""


@dataclass(slots=True)
class DaemonSession:
    """One session on a daemon, presented as the harness that the terminal expects."""

    client: DaemonClient
    session_id: str
    state: TuiState
    adapter: TuiEventAdapter
    host: ModalHost
    remote_commands: list[CommandDefinition] = field(default_factory=list)
    screens: dict[str, ScreenDefinition] = field(default_factory=dict)
    listed_models: list[ModelEntry] = field(default_factory=list)
    """The reply to `models/list`, read one time at attach. The list belongs to the
    profile, and it does not change while a session is open."""
    hidden_screens: frozenset[str] = frozenset()
    """What the `tui.json` of this terminal hides. It is applied again whenever the
    list changes."""
    feed: Followed = field(init=False)
    app: AppSurface | None = None
    """The Textual app, after `attach_surfaces` has received it. A local verb then
    has something to dispatch into, and async work has an owner. It is `None`
    until then, and a headless test drives this class in that state."""
    _screen_verbs: list[CommandDefinition] = field(default_factory=list)
    _keys: list[Callable[[], None]] = field(default_factory=list)
    _unreadable: int = 0
    """How many frames this client could not rebuild **or admit**. The count keeps
    the warning to one. It is also read as a fact, and not only used to quiet the
    log: see `diverged`."""
    header: SessionHeader | None = None
    """The header of the daemon session, from the `session/new` reply. The mirror
    is built on it, so the mirror reads its log as the daemon does. The
    `seed_length` of a fork separates its inherited history from its own work.
    When the mirror holds the same `session/resumed` records, its `cursor_of`
    names the generation of the daemon.

    **It is a constructor argument and not a setter**, for the reason that the
    docstring of `Session.durable_length` gives about that field. An ordering
    constraint that is checked at run time is one that a reader has to learn. A
    value that the type cannot be built without is one that nobody can get
    wrong. This was a `begin()` that replaced `session` after construction and
    raised if anything had been admitted first. That guarded a window that need
    not exist, because `session/new` answers with the header before this object
    is built.

    It is `None` only where no daemon said otherwise: a headless test that
    drives this class directly gets a session with a header of its own."""
    arrival: str = ""
    """`FrontSession.arrival`: the `profile_note` of the attach reply. It is the
    sentence of the daemon about a session that is kept on a version which its
    named profile has moved past."""
    session: Session = field(init=False)
    """The mirror that this client keeps of the log of the daemon. **It is a
    `Session`, and it is kept incrementally.**

    The daemon rehydrates a root one time from its store and then appends. This
    class used to keep a bare list of events and rebuild `Session(seed=…)` from
    it on every read. That validated the whole log again each time, and it added
    a `session/end-seed` marker that the log of the daemon does not have. Now
    each event is admitted (`admit`) as it arrives. The mirror is therefore the
    same type as the log that it mirrors. It has the same surface fold, and the
    same `stale()` check is available to this client. On the `header` of the
    daemon, it also has the same `cursor_of` as the daemon. A screen that is
    built from the mirror reads a live log and not a copy."""
    _readings: list[StatusReading] = field(default_factory=list)
    _moved: anyio.Event = field(default_factory=anyio.Event)
    """Set and replaced whenever the status of the root changes. It wakes
    `_until_idle`. It is not a second copy of the status, which `_until_idle`
    reads again."""
    _staged: Tray = field(default_factory=Tray)
    """The view that this client has of the tray of the root. `session.staged`
    keeps it in step.

    It is a mirror and not the truth. The *root* holds the tray, because two
    people who look at one conversation must see one composer."""
    _families: int = 0
    """How many `session.children` frames have arrived. With this count, the
    answer to a `session/children` call that was made before one of those frames
    is not drawn over that frame (`refresh_children`)."""

    def __post_init__(self) -> None:
        self.session = Session(self.session_id, header=self.header)
        self.feed = Followed(
            session_id=self.session_id, on_events=self._apply, on_status=self._status
        )

    # -------------------------------------------------------------- the log --

    @property
    def diverged(self) -> bool:
        """Whether this mirror has stopped matching the log of the daemon.

        Either refusal desynchronizes the mirror permanently. A frame that will
        not rebuild is skipped. The next `admit` then meets a seq that is no
        longer the next one and refuses it, and it refuses every frame after
        that. The mirror is then a *prefix* of the log of the daemon, and nothing
        says how short it is.

        This is a fact that a consumer reads, and not a counter that only quiets
        the log. Under the code that this replaced, the same skip appeared
        loudly and late. `Session(seed=…)` refused a non-contiguous log the next
        time a screen opened, which is how a past instance of it was found.
        Because the mirror is kept incrementally, nothing refuses later. So the
        divergence has to be *asked about* at the one place that builds from the
        log.
        """
        return self._unreadable > 0

    def _unread(self, message: str, *args: object) -> None:
        """Record a frame that this client could not take. The first one is loud.

        **Loud one time, then quiet.** A frame that will not rebuild is a
        protocol mismatch between this client and its daemon. The silent version
        of this has hidden a real defect two times: an extra key that
        `_EventWire` forbids, and a dropped seq. Each time it turned "the
        transcript is wrong" into "the transcript is empty", with nothing to say
        why. But a mismatch is *systematic*. It fails for every chunk of a
        streaming turn. A traceback for each chunk, formatted on the read loop
        that also drives redraws, costs more than the diagnosis is worth. One
        traceback is the diagnosis.

        This is one method because there are two ways to fail and one policy.
        The second way arrived as a copy of the first, and a third would have
        been a third copy.
        """
        log.log(
            logging.WARNING if not self._unreadable else logging.DEBUG,
            message,
            *args,
            exc_info=not self._unreadable,
        )
        self._unreadable += 1

    def _apply(self, events: Sequence[EventFrame], live: bool) -> None:
        """Fold a run of wire events into the transcript and the log of this client.

        `live` is passed through and not assumed. The docstring of `Sink` in
        `ph_app.daemon.follow` says why.
        """
        for wire, sidecar in events:
            try:
                event = SessionEvent.from_wire(wire)
            except Exception:
                self._unread("ph_app.tui: a frame would not rebuild as an event")
                continue
            try:
                # The mirror keeps the stamps of the daemon and refuses a gap. A
                # frame that would not rebuild above has already been skipped,
                # so the next seq no longer matches. The mirror stops there and
                # does not admit a log with a gap in it.
                self.session.admit(event)
            except ValueError:
                # **This is recorded, and the loop does not `continue`** (H4).
                # The mirror and the transcript are two objects with two rules.
                # A *log* with a gap in it is a worse artifact than a short log,
                # so the refusal stands. But the *view* renders what arrived,
                # and it has no contiguity to keep. When the fold was skipped
                # too, one unreadable frame froze the transcript for the rest of
                # the session. Every later seq was refused because it followed
                # the dropped one, so nothing was drawn again, and a single log
                # line was the whole account. The screens that rebuild *from the
                # log* read `diverged`, and `_unread` has already reported it
                # one time.
                self._unread(
                    "ph_app.tui: the mirror refused seq %s; it stops at %s",
                    event.seq,
                    self.session.seq,
                )
            try:
                # `view_of` runs at the wire edge, so the fold receives a type
                # and not a mapping that it would have to distrust. `tools`
                # stays `None`, because the registry that renders a card is in
                # the daemon.
                self.adapter.apply(event, Frame(live=live, view=view_of(event.type, sidecar)))
            except Exception:
                log.exception("ph_app.tui: the adapter refused an event")
        # This reports what this batch moved, and not everything. The adapter
        # accumulated it for each event and the draw is coalesced for each
        # batch, so the two match.
        self.host.state_changed(self.adapter.take_touched())

    def _status(self, facts: StatusFacts) -> None:
        """Handle `session.status`, which carries the footer beside it.

        The status is pushed and not polled. A reading is a fold of the log, so
        the moment to read the readings again is the moment when the agent
        moved. The 30 Hz tick of the TUI stays local to the client: it exists
        for the spinner.

        `StatusFacts` is the three shapes that reach this method, as one type.
        The attach reply states the route, an `announce` carries the footer, and
        a root that announces `passivated` has neither. `None` means "this frame
        does not say", so each field is kept and not cleared. The four
        `params.get(...) or self.state.<x>` reads that this replaced wrote the
        same rule by hand.
        """
        self.state.provider = facts.provider or self.state.provider
        self.state.model = facts.model or self.state.model
        if facts.model_key is not None:
            # `""` is an answer here: it means an unlisted route. So the old
            # value is kept only when the frame does not say, and not whenever
            # the new value is falsy.
            self.state.model_key = facts.model_key
        if facts.readings is not None:
            self._readings = list(facts.readings)
        if facts.compaction is not None:
            # `None` inside the facts is an answer: nothing compacts on
            # pressure, and the gauge then has no share to warn at.
            self.state.compacts_at = facts.compaction.trigger_fraction
        if facts.status:
            self.state.status = facts.status
            self._moved.set()
            self._moved = anyio.Event()
        # Both surfaces, because the readings of this frame are placed on both.
        # The footer draws all of them except the one that the sidebar claims
        # by id.
        self.host.state_changed(Surface.FOOTER | Surface.SIDEBAR)

    def dispatch(self, method: str, params: dict[str, Any]) -> None:
        """Every notification that this front end reads, in one place.

        The feed owns the two that it buffers. The others are *snapshots* and
        not deltas: the whole tray, the whole command list. Each one is
        therefore correct in whatever order it arrives, and it needs no buffer
        of its own.
        """
        if method in FED:
            self.feed(method, params)
            return
        # **This comes before the ownership check below**, because this frame
        # has no owner. It is about the daemon that every attached session
        # shares. So it carries no `sessionId`, and the filter would drop it
        # because it belongs to nobody. It has its own type for the same
        # reason: see `DaemonNotice`.
        if method == DaemonLifetime.METHOD:
            self.state.lifetime = DaemonLifetime.model_validate(params)
            self.host.state_changed(Surface.SIDEBAR)
            return
        # This is read from the raw frame, before any model sees it. "Is this
        # mine" is the one question that must be answered *without* validation.
        # A client that watches one root receives notices for the other roots,
        # and this check skips the work of parsing them only to discard them.
        if params.get("sessionId") != self.session_id:
            return
        # The frame is validated one time, through the table that owns the
        # mapping from method to payload, and is narrowed by type below. Each
        # branch used to name its own model beside its own `METHOD`, which
        # wrote that pairing out again for each reader. The feed route above is
        # the exception, and it stays one. It is a question about *which sink*.
        # It is answered before anything is parsed, because `Followed.pending`
        # buffers frames whose owner it has not checked.
        notice = notice_of(method, params)
        if isinstance(notice, SessionCommandsNotice):
            self.remote_commands = [
                _remote_command(self.client, self.session_id, one) for one in notice.commands
            ]
            self.host.state_changed()
        elif isinstance(notice, SessionScreensNotice):
            # The screens are wired again and not merged. The routes of a
            # screen are a verb *and* a key binding, and the key is registered
            # on the app. So the old bindings have to be released before the
            # new list is built.
            self.screens = _screens_of(notice.screens, self.hidden_screens)
            if self.app is not None:
                self._wire_screens(self.app)
            self.host.state_changed()
        elif isinstance(notice, SessionChildrenNotice):
            # The whole family, from the logs of the children (P11-09). The
            # stream of a root no longer carries anything about them, so this
            # is the only source that the panel has.
            self._families += 1
            self.state.take_children(notice.children)
            self.host.state_changed(Surface.SIDEBAR)
        elif isinstance(notice, AskSettledNotice):
            # Another terminal answered first (H1). The daemon settled the ask
            # before it published this notice. So the modal here can only
            # produce an answer that would be discarded, and a person who
            # answered it would be told nothing at all.
            self.host.withdraw_ask(notice.key)
        elif isinstance(notice, SessionStagedNotice):
            self._staged = Tray()
            for ref in notice.staged:
                self._staged.stage(ref)
            self.host.state_changed()

    # ----------------------------------------------------------- projections --

    def status_readings(self) -> list[StatusReading]:
        return list(self._readings)

    def commands(self) -> list[CommandDefinition]:
        return [*self._screen_verbs, *self.remote_commands]

    def screen(self, screen_id: str) -> ScreenDefinition | None:
        return self.screens.get(screen_id)

    def models(self) -> list[ModelEntry]:
        return list(self.listed_models)

    async def choose_model(self, choice: ModelChoice) -> RootDescription:
        """Ask the daemon to run this session on `choice` from its next request.

        This is awaited and not spawned like `set_preset`, because a choice can
        be refused: a key that this profile does not list, or a provider that
        nothing serves. The person who typed the choice must get the sentence,
        and not a log line. The reply is the description of the root, and the
        reply to a repeat is one too. It goes where every status frame goes, so
        a caller that reads the state after this returns reads the new route.
        """
        reply = await self.client.mutate(
            verbs.SESSION_MODEL, ModelParams(session_id=self.session_id, choice=choice)
        )
        self._status(reply.facts())
        return reply

    async def browse_sessions(self, *, cwd: str = "") -> list[SessionSummary]:
        """The list from the daemon: stored logs and its live roots, already merged."""
        return await browse_on(self.client, cwd=cwd)

    async def presets(self) -> list[PresetSchema]:
        """The postures that this root offers, asked for when the picker opens.

        That is the only moment when anything needs them. It is also the moment
        when a fold would have been wrong, because this client may have attached
        long after the posture was last changed.
        """
        reply = await self.client.call(
            verbs.PRESETS_LIST, SessionParams(session_id=self.session_id)
        )
        return list(reply.presets)

    async def refresh_children(self) -> None:
        """Read the family one time, for a client that has just attached (P11-09).

        The call is made **after** the attach, so every change from then on also
        arrives as a `session.children` frame, which is pushed after the change
        that it announces. A frame can land while this call is in flight. The
        frames then hold the newest state, whichever of the two the daemon built
        first. The answer is drawn only if no frame arrived, because a list that
        was read before a change must not be drawn over the frame that announces
        the change.
        """
        heard = self._families
        reply = await self.client.call(
            verbs.SESSION_CHILDREN, SessionParams(session_id=self.session_id)
        )
        if self._families == heard:
            self.state.take_children(reply.children)
            self.host.state_changed(Surface.SIDEBAR)

    async def refresh_credentials(self) -> Mapping[str, bool]:
        """Ask the daemon which credentials the profile of the root names, and
        which of them it holds. Nothing is cached: the answer can change while a
        session is open, and the login screen is the only reader."""
        reply = await self.client.call(
            verbs.CREDENTIALS_HELD, SessionParams(session_id=self.session_id)
        )
        return dict(reply.held)

    # ---------------------------------------------------------------- turns --

    async def submit(self, text: str) -> None:
        """Queue a turn and wait for the root to go idle.

        There are two halves because the wire has two. `session/prompt` returns
        as soon as the prompt is *in the inbox*, and that is why the prompt
        survives the death of this client. The end of the turn arrives later as
        a `session.status`. The status is set to `running` here as well as by
        that frame, for two reasons. `_until_idle` would otherwise return before
        the frame that announces the turn had arrived. And a person who pressed
        enter should see the spinner before a socket round trip.
        """
        self.state.status = "running"
        self.host.state_changed()
        try:
            await self.client.prompt(self.session_id, text)
            await self._until_idle()
        finally:
            if self.client.closed.is_set():
                # `_until_idle` also returns when the daemon goes away, and the
                # `running` that was set above is an assumption of this client
                # and not something that the root reported. If it were left in
                # place, the spinner would never stop, on a turn that nobody is
                # running, with `state.busy` true. Every later prompt would
                # then be queued silently into a socket that is gone. The
                # watcher of the app *reports* the death of the daemon. This
                # code must only stop claiming work that is not happening, and
                # `stand_down` is that rule for both of them.
                self.state.stand_down()
                self.host.state_changed()

    async def _until_idle(self) -> None:
        """Wait for this root to stop working, or for the daemon to go away.

        It waits for both, because a wait on only the first hangs whenever the
        second happens. It is woken by `_moved` and does not poll. The status is
        still the one place that holds the answer, and the event only tells this
        method that the status changed. `waiting` ends the wait too. A root that
        is parked on a person waits for the modal of *this screen*. If that
        counted as work in flight, `submit` would await the very thing that it
        is blocking.
        """
        while not self.client.closed.is_set() and self.state.busy:
            await first_of(self._moved, self.client.closed)

    def queue(self, text: str) -> None:
        """Add to the inbox in the middle of a turn, without waiting. It is sync
        because the app calls it from a key handler, so a worker sends the frame.
        The text comes back to the person who typed it in a `session.event`, as
        it does to everybody else.

        **The count is not changed here** (H6). `agent/inbox/spliced` carries
        `inserted`, and `_on_agent_inbox_spliced` folds it. That is the
        authoritative account. It is the one that a replay produces, and it
        arrives for this prompt as for any other. An optimistic increment here
        as well counted the message of the person twice. A footer then said
        "2 queued" for one pending prompt, and went to 1 when the turn claimed
        it.
        """
        self._spawn(self.client.prompt(self.session_id, text))

    def cancel(self) -> None:
        self._spawn(
            self.client.call(verbs.SESSION_CANCEL, SessionParams(session_id=self.session_id))
        )

    def _spawn(self, work: Coroutine[Any, Any, Any]) -> None:
        """Run an awaitable from a sync caller. The worker pool of the app owns
        it, so it is canceled with the app. The sync members of `FrontSession`
        exist for key handlers, and key handlers exist only when there is an
        app."""
        if self.app is None:
            raise RuntimeError("attach_surfaces first: nothing owns background work yet")
        self.app.run_worker(_reported(work))

    async def shell(self, command: str, *, surface: bool = False) -> None:
        """Run `!` or `!!` in the workspace of the session. The shell is the shell
        of the daemon, and not of this client.

        A browser tab has no shell, and "the shell of the session" is the
        correct meaning in both cases. Nothing is returned, because the command
        and its output arrive as `shell/*` events. The person who typed the
        command reads it back from the same log as everybody else.
        """
        await self.client.mutate(
            verbs.SESSION_SHELL,
            ShellParams(session_id=self.session_id, command=command, surface=surface),
        )

    async def attach(self, paths: Sequence[str]) -> list[AttachmentRef]:
        """Read these files here and stage them on the root.

        **This client reads the bytes.** That is the human door (I-9). It goes
        through the same `read_for_attach` that `AttachmentStore.save_path`
        uses, so a file that a person attaches is classified the same way on
        both paths. The daemon learns the content and a name, and never a path.
        """
        for path in paths:
            name, mime, content = await read_for_attach(path)
            await stage_bytes(self.client, self.session_id, name, mime, content)
        return self._staged.refs

    # ------------------------------------------------------------ lifecycle --

    def attach_surfaces(self, app: AppSurface) -> list[Callable[[], None]]:
        """Take the app, and build the verb of each screen, which dispatches into it.

        The verbs are built one time here and not on each `commands()` call,
        which the completion source makes on every keystroke. The verbs of the
        daemon belong to its registry and unwind with the row that made them,
        and the verbs of the terminal belong to the app. What unwinds here is
        what this client made: the verb and the key binding of each screen.
        """
        self.app = app
        self._wire_screens(app)
        self.host.state_changed()

        def release() -> None:
            self._release_screens()
            self.app = None
            self._screen_verbs = []

        return [release]

    def _wire_screens(self, app: AppSurface) -> None:
        """Build one verb for each screen that this client can draw.

        A screen gets three routes: a verb, a palette entry and a key. The key
        is a binding on the app, so the old bindings must be released before the
        set is replaced. This is called again whenever `session.screens` says
        that the list of the deployment changed.
        """
        self._release_screens()
        self._screen_verbs = []
        for screen in self.screens.values():
            action = open_screen_action(screen.id)
            self._screen_verbs.append(action_command(app, screen.id, screen.label, action))
            if screen.key:
                binding = Binding(
                    screen.key, action, screen.label, priority=True, id=screen.id, show=False
                )
                self._keys.append(app.add_binding(binding))

    def _release_screens(self) -> None:
        for key in self._keys:
            key()
        self._keys = []

    def set_preset(self, name: PresetName) -> None:
        self._spawn(
            self.client.mutate(
                verbs.SESSION_PRESET, PresetParams(session_id=self.session_id, preset=name)
            )
        )

    def store_credential(self, name: str, value: str) -> None:
        """Give a secret to the daemon. It is never logged, on either side. The
        member is sync, so the answer of the daemon arrives after this returns,
        and this returns nothing. A refusal by the daemon is raised in the
        worker, where `_reported` logs it."""
        self._spawn(
            self.client.call(
                verbs.CREDENTIALS_STORE,
                StoreCredentialParams(session_id=self.session_id, name=name, value=value),
            )
        )

    async def close(self) -> None:
        """Detach and stop reading. **The root keeps running**: the purpose of P5-01.

        There is no flush and no shutdown. This front end is leaving, and it is
        not ending the session. What happens to the root afterwards is for the
        daemon to decide.

        **The detach is a courtesy, so a failed detach is not a failure.** The
        daemon drops the subscriptions of a connection when the socket closes in
        either case, in the `finally` of `_Connection.serve`. The detach only
        lets that happen a moment earlier, which matters to a client that keeps
        one connection across several sessions. So a connection that is already
        gone is a detach that is already done. A check of `closed` cannot close
        the race, because the pump can stop between the check and the reply.
        That is exactly what happens when Textual cancels its workers at
        shutdown.
        """
        with suppress(DaemonGone), anyio.move_on_after(2.0):
            await self.client.call(verbs.SESSION_DETACH, SessionParams(session_id=self.session_id))


async def browse_on(client: DaemonClient, *, cwd: str = "") -> list[SessionSummary]:
    """The sessions that this daemon can offer, optionally narrowed to one directory.

    This is a function on a bare client, beside `attach_session`, for the same
    reason. The startup picker asks this *before* there is a session to ask. A
    method on `FrontSession`, which means "an attached session", could not
    answer. It is named for the question and not for the situation of that
    caller, because `DaemonSession.browse_sessions` is the same call and now
    makes it through this function.
    """
    reply = await client.call(verbs.SESSIONS_BROWSE, BrowseParams(cwd=cwd))
    return list(reply.sessions)


async def attach_session(
    client: DaemonClient,
    session_id: str | None,
    *,
    host: ModalHost,
    cwd: Path | None = None,
    trust: TrustAnswer = "",
    hidden_screens: Collection[str] = (),
    choice: ModelChoice = ModelChoice(),
    profile: str = "",
) -> DaemonSession:
    """Start or resume a session on the daemon, and catch this client up on it.

    The order is the one that the docstring of `session/attach` argues for:

    1. Say what this client can answer.
    2. Start the root.
    3. Read its projections.
    4. Subscribe.
    5. Page the history from the cursor that the attach reply named.
    6. Go live.

    The projections are independent reads of a mounted root, and they are
    fetched together. The attach follows them only so that `front` can be built
    whole. The attach reply seeds the status, the route and the footer.

    Credentials are deliberately not among the projections. Whether a credential
    is held can change while a session is open, so an answer that was read at
    attach would be stale by the time a picker drew it. The login screen asks
    when it opens, which is where the answer is read. It asks for *all* of them
    at once, because the daemon composes the profile of the root and so knows
    which credentials that profile names.

    A `session_id` of `None` is a new session, which the daemon names
    (protocol 8). Every call after `session/new` uses the id that its reply
    carries.
    """
    state = TuiState()
    client.handlers[ApprovalAsk.METHOD] = answering(ApprovalAsk, _asking_approval(host))
    client.handlers[QuestionAsk.METHOD] = answering(QuestionAsk, _asking_question(host))
    client.handlers[ProfileAsk.METHOD] = answering(ProfileAsk, _asking_profile(host))
    # `asks` is declared **before** the attach. The desk joins a front end as it
    # attaches, and a client that declared nothing is never asked.
    await client.initialize("asks")
    # `trust` is the answer of the user. This client asked for it and the daemon
    # enforces it: the daemon refuses a `cwd` that nobody has vouched for
    # (P5-14).
    # The reply carries the header of this root, and the mirror below is built
    # on it. The header is read and not discarded. Because it is taken now, the
    # mirror can be built whole and need not be re-keyed afterwards.
    created = await client.call(
        verbs.SESSION_NEW,
        NewSessionParams(
            session_id=session_id,
            cwd=str(cwd) if cwd else None,
            trust=trust,
            choice=choice,
            profile=profile,
        ),
    )
    session_id = created.session_id

    # Six startup reads run at once. Each goes through the one typed call,
    # `client.call`, and each lands in a slot that has the type of its own
    # reply. The first draft kept a `dict[str, dict[str, Any]]` and decoded
    # afterwards with `verbs.X.read(replies[verbs.X.name])`. That named the verb
    # twice on each line, so
    # `verbs.COMMANDS_LIST.read(replies[verbs.SCREENS_LIST.name])` type-checked.
    # That is the shape in which two values must agree, which `Verb` exists to
    # delete. The draft reintroduced it at the three sites that had opted out
    # of `call`, and it failed as a `ValidationError` at TUI startup and not as
    # a type error. A one-slot list for each read does what a heterogeneous
    # dict could not.
    async def fetch[P: WireModel, R: WireModel](verb: Verb[P, R], params: P, into: list[R]) -> None:
        into.append(await client.call(verb, params))

    listed_commands: list[SessionCommandsNotice] = []
    listed_screens: list[SessionScreensNotice] = []
    listed_tools: list[SessionToolsReply] = []
    listed_skills: list[SessionSkillsReply] = []
    listed_models: list[SessionModelsReply] = []
    read_lifetime: list[DaemonLifetime] = []
    asked = SessionParams(session_id=session_id)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(fetch, verbs.COMMANDS_LIST, asked, listed_commands)
        tasks.start_soon(fetch, verbs.SCREENS_LIST, asked, listed_screens)
        # Two more reads are in the same group and not made on demand, because
        # both are drawn in the sidebar from the first frame. A panel that
        # filled in a moment later would be a second kind of "not yet", beside
        # the history that is still paging. Neither changes while a session is
        # open: a row that registers a tool does so at mount, and a skill is
        # installed with a restart.
        tasks.start_soon(fetch, verbs.TOOLS_LIST, asked, listed_tools)
        tasks.start_soon(fetch, verbs.SKILLS_LIST, asked, listed_skills)
        tasks.start_soon(fetch, verbs.MODELS_LIST, asked, listed_models)
        # The one read here that is not about this root: whether the daemon
        # behind it intends to stay. It is read one time and not waited for,
        # because `daemon.lifetime` is sent when the answer *changes*. A client
        # that connected between two changes would otherwise draw nothing until
        # something moved.
        tasks.start_soon(fetch, verbs.DAEMON_LIFETIME, NoParams(), read_lifetime)

    hidden = frozenset(hidden_screens)
    state.tools = _catalog(listed_tools[0].tools)
    state.skills = _catalog(listed_skills[0].skills)
    state.lifetime = read_lifetime[0]
    front = DaemonSession(
        client=client,
        session_id=session_id,
        state=state,
        # `tools=None`: the registry that renders a card is in the daemon, which
        # sends the rendered view beside each event. See `Frame.view`.
        adapter=TuiEventAdapter(state=state),
        host=host,
        remote_commands=[
            _remote_command(client, session_id, one) for one in listed_commands[0].commands
        ],
        screens=_screens_of(listed_screens[0].screens, hidden),
        listed_models=list(listed_models[0].models),
        hidden_screens=hidden,
        header=created.header,
    )
    client.peer.on_notify = front.dispatch
    # The attach reply carries the status, the route and the footer, so it is
    # the one frame that the front end starts from.
    attached = await client.call(verbs.SESSION_ATTACH, SessionParams(session_id=session_id))
    # It goes through the feed, which owns the rule that this reply is the first
    # status frame. The CLI read it separately and got a different answer.
    front.feed.seed(attached)
    front.arrival = attached.profile_note
    # This is the **cursor** of the attach reply, wound back to the start. It is
    # not a `from`. A `from` is the *index* at which the live stream begins, and
    # it is not a cursor at all. When a `from` was passed as a cursor, the
    # client lost seq 0 of every session. `session/snapshot` read a bare int as
    # "no cursor" and paged from 1. The loss was invisible until
    # `Session(seed=…)` refused a log that did not start at 0, which happens
    # only when somebody opens a screen. `phern agents attach` builds the same
    # shape for `--since`.
    await front.feed.catch_up(client, attached.cursor.model_copy(update={"sequence": 0}))
    front.feed.live()
    await front.refresh_children()
    return front


class _Described(Protocol):
    """The two fields that a catalog panel draws, whatever carries them.

    This is a Protocol and not `Any`. `ToolSchema` and `Skill` differ in
    everything that the sidebar does not show (the parameter schema of a tool,
    the path and version of a skill), and they agree on these two fields.
    Because the agreement is named, it is checked. As `Any` it was a comment,
    and a wire model that dropped `description` would have failed at the first
    draw and not at the call.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...


def _catalog(entries: Sequence[_Described]) -> tuple[CatalogEntry, ...]:
    """A projected catalog as the panels read it: a name and what it is for."""
    return tuple(CatalogEntry(name=one.name, description=one.description) for one in entries)


def _remote_command(
    client: DaemonClient, session_id: str, schema: CommandSchema
) -> CommandDefinition:
    """One of the commands of the daemon, with a `run` that runs it on the daemon.

    The body is a `session/command`. The palette, the completer and `run_line`
    therefore see one kind of thing and dispatch it one way. Which end executes
    a verb is decided by the definition, and not by the caller.
    """

    async def elsewhere(argument: str, _context: object) -> str | None:
        line = f"/{schema.name} {argument}".rstrip()
        reply = await client.mutate(
            verbs.SESSION_COMMAND, CommandParams(session_id=session_id, line=line)
        )
        # A repeat answers with a description and no `shown`, and `shown` is the
        # one thing that this caller wanted. So the union is narrowed and not
        # ignored, and a re-sent command says nothing where it would otherwise
        # say the wrong thing. The exception is a first attempt whose outcome
        # is unknown. Then the user is the one to decide, and is told so
        # (P10-10).
        if isinstance(reply, MutationRepeated) and reply.outcome == "unknown":
            return UNKNOWN_REPEAT
        return reply.shown if isinstance(reply, CommandShown) else None

    return CommandDefinition(
        name=schema.name,
        summary=schema.summary,
        run=elsewhere,
        argument_hint=schema.argument_hint,
    )


def _screens_of(
    schemas: Sequence[ScreenSchema], hidden: frozenset[str]
) -> dict[str, ScreenDefinition]:
    """The screens that the deployment has, this client can draw, and the user has not hidden.

    The wire supplies `label`, `order` and `key`, which belong to the
    deployment. The local definition supplies `build`, the one field that cannot
    travel.
    """
    local = {definition.id: definition for definition in LOCAL_SCREENS}
    found: dict[str, ScreenDefinition] = {}
    for schema in schemas:
        if schema.id in hidden:
            continue
        mine = local.get(schema.id)
        if mine is None:
            log.debug("ph_app.tui: no local builder for screen %r", schema.id)
            continue
        found[schema.id] = ScreenDefinition(build=mine.build, **schema.model_dump())
    return found


async def _reported(work: Coroutine[Any, Any, Any]) -> None:
    """Await background work and **log** what it raised, and do not let it out (H3).

    Textual ends the app on an unhandled worker exception. That is right for a
    bug and wrong here. Every caller is a fire-and-forget frame from a key
    handler, and the ordinary way for one to fail is that the daemon went away.
    The app then died on the *next* keystroke, which was the Enter of the user.
    It showed a traceback where an explanation belonged, and the death of the
    daemon was still unreported.

    The work is wrapped, and `exit_on_error=False` is not passed to Textual, for
    two reasons. `AppSurface.run_worker` deliberately declares only what is
    called on it. And a silent drop is not wanted either: the reason is worth a
    log line.
    """
    try:
        await work
    except Exception:
        log.exception("ph_app.tui: background work failed")


def _asking_approval(host: ModalHost) -> Callable[[ApprovalAsk], Awaitable[ApprovalAskReply]]:
    """`approval/ask` → the modal, in a worker.

    `reason` goes back on the wire, and this client does not steer with it. The
    daemon holds the agent, and a client that steered a turn which it does not
    own would write into the session of somebody else.
    """

    async def ask(asked: ApprovalAsk) -> ApprovalAskReply:
        outcome, reason = await host.ask_approval(asked.request, ask=asked.key)
        # The answer goes through the encoder of the seam. `Edited` and
        # `Responded` are frozen dataclasses, and an unencoded one in a frame
        # is a `TypeError` inside the task group that answers the ask. The desk
        # reads that as "this front end cannot answer" and drops the front end.
        return ApprovalAskReply(answer=answer_to_wire(outcome), reason=reason or "")

    return ask


def _asking_question(host: ModalHost) -> Callable[[QuestionAsk], Awaitable[QuestionAskReply]]:
    """`question/ask` → the ask-user modal, in a worker."""

    async def ask(asked: QuestionAsk) -> QuestionAskReply:
        answer = await host.ask_question(asked.question, ask=asked.key)
        return QuestionAskReply(answer=answer)

    return ask


def _asking_profile(host: ModalHost) -> Callable[[ProfileAsk], Awaitable[ProfileAskReply]]:
    """`profile/ask` → the profile-changed modal, in a worker (S6)."""

    async def ask(asked: ProfileAsk) -> ProfileAskReply:
        decision = await host.ask_profile(asked.name, asked.listing, ask=asked.key)
        return ProfileAskReply(decision=decision)

    return ask
