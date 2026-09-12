"""The terminal as a daemon client — the same screen, a harness elsewhere (P5-14).

The terminal used to mount pH in its own process, which is why closing it ended
the turn. This is what replaced that: a `FrontSession` whose every member answers
by asking the daemon. The app above it reads only the protocol, which is what
lets one layout, one fold and one set of verbs serve a tty and a browser tab
alike.

Four things here are decisions rather than plumbing.

**A modal is awaited in a worker, never on the pump.** The daemon's asks arrive as
`approval/ask` and `question/ask` on the read loop, where `push_screen_wait` is
illegal. So the handlers hand off to `ModalHost`, which the app implements inside a
Textual worker — the same contract `HarnessSession`'s answerers live under, and the
reason `ModalHost` exists rather than the app being called back directly.

**Verbs come from both ends, merged into one list with one dispatch.** `/model`
and `/theme` are the terminal's own — they change *this* client's display and mean
nothing to a daemon serving three of them — while `/compact` and `/schedule` are
the harness's. Each remote verb is a `CommandDefinition` whose `run` sends
`session/command`, so `run_command` finds a name and calls `run` without knowing
which side owns it.

**The `Session` is a mirror, extended as events arrive.** Only one thing wants it:
`ScreenDefinition.build(session)` when somebody opens a screen — but it is kept
whole and incrementally, not rebuilt on demand.

That is a reversal, and the objection it had to answer is worth keeping: *"`Session`
has no way to admit an already-numbered event, and inventing one would be a second
append path into the type whose whole contract is that appends are its own."* The
answer is that `admit` is not a second append path. It shares `_commit` with
`append`, so the surface validation, the push and the publish are one tail and an
observer cannot tell the two doors apart; what differs is only who stamped the
event, which is the whole distinction between owning a log and mirroring one.
Rebuilding instead cost a full `Session(seed=…)` — every event re-frozen and
re-validated — per screen open, and grew a `session/end-seed` marker the daemon's
log does not have. The id is on the protocol separately so the sidebar never asks
for the whole thing.

**A screen's `build` is the one thing that cannot travel**, which
`ScreenDefinition` already says. Each screen pH ships exports a `CLIENT_SIDE`
definition from its own module; `screens/list` names what the deployment mounted,
and the intersection is what a person is offered. Rule 6: a screen a third-party
row contributes reaches a remote front end as nothing at all — P7-07's declarative
bodies are what close that.

@module ph_app.tui.remote
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio
from textual.binding import Binding

from ph.llm.types import AttachmentRef
from ph.seams.approval import answer_to_wire
from ph.seams.attachments import read_for_attach
from ph.seams.commands import CommandDefinition, CommandSchema, parse_command_line
from ph.seams.permission_presets import PresetName, PresetSchema
from ph.seams.tui_screens import ScreenDefinition, ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.session import Session, SessionEvent, SessionHeader
from ph.wire import WireModel

from .. import verbs
from ..attach import Tray, stage_bytes
from ..daemon.client import DaemonClient
from ..daemon.duplex import answering
from ..daemon.follow import Followed, first_of
from ..params import (
    CommandParams,
    HeldCredentialsParams,
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
    CommandShown,
    DaemonConfigReply,
    QuestionAsk,
    QuestionAskReply,
    SessionCommandsNotice,
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
from .commands import action_command, local_commands
from .frontend import ModalHost
from .screens import open_screen_action
from .state import CatalogEntry, Surface, TuiState
from .trajectory_screen import CLIENT_SIDE as TRAJECTORY

__all__ = ["LOCAL_SCREENS", "DaemonSession", "attach_session"]

log = logging.getLogger("ph_app.tui.remote")

LOCAL_SCREENS: tuple[ScreenDefinition, ...] = (TRAJECTORY,)
"""Every screen this build can draw without a harness. See the module docstring."""


@dataclass(slots=True)
class DaemonSession:
    """One session on a daemon, dressed as the harness the terminal expects."""

    client: DaemonClient
    session_id: str
    state: TuiState
    adapter: TuiEventAdapter
    host: ModalHost
    config_rows: tuple[Any, ...] = ()
    remote_commands: list[CommandDefinition] = field(default_factory=list)
    screens: dict[str, ScreenDefinition] = field(default_factory=dict)
    held: dict[str, bool] = field(default_factory=dict)
    feed: Followed = field(init=False)
    app: Any = None
    """The Textual app, once `attach_surfaces` has been given it — so a local verb
    has something to dispatch into and async work has an owner. `None` until then,
    which is the state a headless test drives this in."""
    _verbs: list[CommandDefinition] = field(default_factory=list)
    _keys: list[Callable[[], Any]] = field(default_factory=list)
    _unreadable: int = 0
    """Frames this client could not rebuild **or admit**. Counted so the warning is
    one — and read as a fact, not only as a log-quietener: see `diverged`."""
    generation: int | None = None
    """The daemon session's `created_at`, from the `session/new` reply.

    **A constructor argument rather than a setter**, for the reason
    `Session.durable_length`'s own docstring gives about itself: an ordering
    constraint policed at runtime is one a reader has to learn, and a value the
    type cannot be built without is one nobody can get wrong. This was a `begin()`
    that replaced `session` after construction and raised if anything had been
    admitted first — a guard against a window that need not exist, since
    `session/new` answers with the cursor before this object is built.

    `None` only where no daemon said otherwise: a headless test driving this
    directly gets a session with a generation of its own."""
    session: Session = field(init=False)
    """This client's mirror of the daemon's log — **a `Session`, kept incrementally.**

    The daemon rehydrates a root once from its store and then appends; this used to
    keep a bare list of events and rebuild `Session(seed=…)` from it on every read,
    which re-validated the whole log each time and grew a `session/end-seed` marker
    the daemon's log does not have. Now each event is `admit`ted as it arrives, so
    the mirror is the same type as what it mirrors, with the same surface fold, the
    same `stale()` check available to this client, and — once `begin` has keyed the
    header to the attach reply's generation — the same `cursor_of` as the daemon's
    own. A screen built from it reads a live log rather than a copy.

    A placeholder until `begin`: valid and empty, with a generation of its own, so a
    front end driven without an attach reply still has a session to build on."""
    _readings: list[StatusReading] = field(default_factory=list)
    _moved: anyio.Event = field(default_factory=anyio.Event)
    """Set and replaced whenever the root's status changes — a wake-up for
    `_until_idle`, not a second copy of the status, which it re-reads."""
    _staged: Tray = field(default_factory=Tray)
    """This client's view of the root's tray, kept in step by `session.staged`.

    A mirror rather than the truth: the *root* holds the tray, because two people
    looking at one conversation must see one composer."""

    def __post_init__(self) -> None:
        header = (
            None
            if self.generation is None
            else SessionHeader(id=self.session_id, created_at=self.generation)
        )
        self.session = Session(self.session_id, header=header)
        self.feed = Followed(
            session_id=self.session_id, on_events=self._apply, on_status=self._status
        )

    # -------------------------------------------------------------- the log --

    @property
    def diverged(self) -> bool:
        """Whether this mirror has stopped matching the daemon's log.

        Either refusal desynchronises it permanently: a frame that will not rebuild
        is skipped, so the next `admit` meets a seq that is no longer next and
        refuses too, and every frame after it. The mirror is then a *prefix* of the
        daemon's log with no way to tell how short.

        A fact a consumer reads rather than a counter that only quietens logs.
        Under the code this replaced, the same skip surfaced loudly and late —
        `Session(seed=…)` refused a non-contiguous log the next time a screen
        opened, which is how a past instance of it was found. Keeping the mirror
        incrementally means nothing refuses later, so the divergence has to be
        *asked about* at the one place that builds from the log.
        """
        return self._unreadable > 0

    def _unread(self, message: str, *args: Any) -> None:
        """Record a frame this client could not take, loudly once.

        **Loud once, then quiet.** A frame that will not rebuild is a protocol
        mismatch between this client and its daemon, and twice now the silent
        version of this hid a real defect — an extra key `_EventWire` forbids, and
        a dropped seq — by turning "the transcript is wrong" into "the transcript
        is empty" with nothing to say why. But a mismatch is *systematic*: it fails
        for every chunk of a streaming turn, and formatting a traceback per chunk
        on the read loop that also drives redraws costs more than the diagnosis is
        worth. One is the diagnosis.

        One method because there are two ways to fail and one policy: the second
        arrived as a copy of the first, and a third would have been a third copy.
        """
        log.log(
            logging.WARNING if not self._unreadable else logging.DEBUG,
            message,
            *args,
            exc_info=not self._unreadable,
        )
        self._unreadable += 1

    def _apply(self, events: Sequence[tuple[Mapping[str, Any], Any]], live: bool) -> None:
        """Fold a run of wire events into the transcript and this client's log.

        `live` is passed through rather than assumed; `Followed.Sink` says why.
        """
        for wire, sidecar in events:
            try:
                event = SessionEvent.from_wire(wire)
            except Exception:
                self._unread("ph_app.tui: a frame would not rebuild as an event")
                continue
            try:
                # The mirror keeps the daemon's stamps and refuses a hole: a
                # frame that would not rebuild above has already been skipped, so
                # the next seq no longer matches and the mirror stops there rather
                # than admitting a log with a gap in it.
                self.session.admit(event)
            except ValueError:
                self._unread(
                    "ph_app.tui: the mirror refused seq %s; it stops at %s",
                    event.seq,
                    self.session.seq,
                )
                continue
            try:
                # `view_of` at the wire edge, so the fold is handed a type rather
                # than a mapping it would have to distrust; `tools` stays `None`
                # because the registry that renders a card is in the daemon.
                self.adapter.apply(event, Frame(live=live, view=view_of(event.type, sidecar)))
            except Exception:
                log.exception("ph_app.tui: the adapter refused an event")
        # What this batch moved, not everything: the adapter accumulated it per
        # event and the draw is coalesced per batch, so the two line up.
        self.host.state_changed(self.adapter.take_touched())

    def _status(self, facts: StatusFacts) -> None:
        """`session.status`, which carries the footer beside it.

        Pushed rather than polled because a reading is a fold of the log, so the
        moment worth re-reading them is the moment the agent moved. The TUI's own
        30 Hz tick stays client-local: it exists for the spinner.

        `StatusFacts` is the three shapes that reach here as one type: the attach
        reply states the route, an `announce` carries the footer, and a root
        announcing `passivated` has neither. `None` is "this frame does not say",
        so each field is kept rather than cleared — which is what the four
        `params.get(...) or self.state.<x>` reads this replaced were spelling.
        """
        self.state.provider = facts.provider or self.state.provider
        self.state.model = facts.model or self.state.model
        if facts.readings is not None:
            self._readings = list(facts.readings)
        if facts.status:
            self.state.status = facts.status
            self._moved.set()
            self._moved = anyio.Event()
        # Both, because the readings this frame carries are placed on both: the
        # footer draws all of them but the one the sidebar claims by id.
        self.host.state_changed(Surface.FOOTER | Surface.SIDEBAR)

    def dispatch(self, method: str, params: dict[str, Any]) -> None:
        """Every notification this front end reads, in one place.

        The feed owns the two it buffers; the rest are *snapshots* rather than
        deltas — the whole tray, the whole command list — so each is correct
        whatever order it arrives in and needs no buffer of its own.
        """
        if method in FED:
            self.feed(method, params)
            return
        # Read off the raw frame, before any model sees it: "is this mine"
        # is the one question that must be answered *without* validating,
        # because a client watching one root receives notices for the
        # others and parsing them to discard them is the work this skips.
        if params.get("sessionId") != self.session_id:
            return
        # Validated once, through the table that owns method → payload, and
        # narrowed by type below. Each branch used to name its own model beside
        # its own `METHOD`, which is that pairing written out per reader. The
        # feed route above is the exception and stays one: it is a question
        # about *which sink*, answered before anything is parsed, because
        # `Followed.pending` buffers frames it has not checked the owner of.
        notice = notice_of(method, params)
        if isinstance(notice, SessionCommandsNotice):
            self.remote_commands = [
                _remote_command(self.client, self.session_id, one) for one in notice.commands
            ]
            self.host.state_changed()
        elif isinstance(notice, SessionScreensNotice):
            # Re-wired rather than merged: a screen's routes are a verb *and* a
            # key binding, and the key is registered on the app — so the old
            # ones have to be released before the new list is built.
            self.screens = _screens_of(notice.screens)
            if self.app is not None:
                self._wire_screens(self.app)
            self.host.state_changed()
        elif isinstance(notice, SessionStagedNotice):
            self._staged = Tray()
            for ref in notice.staged:
                self._staged.stage(ref)
            self.host.state_changed()

    # ----------------------------------------------------------- projections --

    def status_readings(self) -> list[StatusReading]:
        return list(self._readings)

    def commands(self) -> list[CommandDefinition]:
        return [*self._verbs, *self.remote_commands]

    def screen(self, screen_id: str) -> ScreenDefinition | None:
        return self.screens.get(screen_id)

    def providers(self) -> list[str]:
        """Nothing yet. Not enforced (§5 rule 6): the model picker over a socket
        has no projection — `llm.list_providers()` is not on the wire — so a remote
        front end offers no `/model` choices. In process it lists them."""
        return []

    async def browse_sessions(self) -> list[SessionSummary]:
        """The daemon's own list — stored logs and its live roots, already merged."""
        reply = await self.client.call(verbs.SESSIONS_BROWSE, NoParams())
        return list(reply.sessions)

    async def presets(self) -> list[PresetSchema]:
        """The postures this root offers, asked when the picker opens.

        Which is the only moment anything needs them — and the moment a fold
        would have been wrong, because this client may have attached long after
        the posture was last changed.
        """
        reply = await self.client.call(
            verbs.PRESETS_LIST, SessionParams(session_id=self.session_id)
        )
        return list(reply.presets)

    def credential_held(self, name: str) -> bool:
        """From the last `credentials/held` answer — a fact about the *daemon's*
        store, and empty until `refresh_credentials` has asked. The login screen
        asks, because it is the only thing that reads this."""
        return bool(self.held.get(name))

    async def refresh_credentials(self, names: Sequence[str]) -> None:
        reply = await self.client.call(
            verbs.CREDENTIALS_HELD,
            HeldCredentialsParams(session_id=self.session_id, names=list(names)),
        )
        self.held = dict(reply.held)

    # ---------------------------------------------------------------- turns --

    async def submit(self, text: str) -> None:
        """Queue a turn and wait for the root to go idle.

        Two halves because the wire's are two: `session/prompt` returns as soon as
        the prompt is *in the inbox*, which is what makes it survive this client
        dying, and the turn's end arrives later as a `session.status`. The status
        is set to `running` here as well as by that frame, because `_until_idle`
        would otherwise return before the frame announcing the turn had crossed —
        and because a person who pressed enter should see the spinner before a
        socket round trip.
        """
        self.state.status = "running"
        self.host.state_changed()
        await self.client.prompt(self.session_id, text)
        await self._until_idle()

    async def _until_idle(self) -> None:
        """Wait for this root to stop working, or for the daemon to go away.

        Both, because waiting on only the first is a hang whenever it is the
        second that happens. Woken by `_moved` rather than polled: the status is
        still the one place the answer lives — the event is only how this finds
        out it changed. `waiting` ends the wait too: a root parked on a person is
        waiting for *this screen's* modal, and treating that as work in flight
        would leave `submit` awaiting the very thing it is blocking.
        """
        while not self.client.closed.is_set() and self.state.busy:
            await first_of(self._moved, self.client.closed)

    def queue(self, text: str) -> None:
        """Add to the inbox mid-turn, without waiting. Sync because the app calls
        it from a key handler, so the frame is sent by a worker — and the person's
        own text reaches them back off `session.event` like everybody else's."""
        self.state.queued += 1
        self.host.state_changed()
        self._spawn(self.client.prompt(self.session_id, text))

    def cancel(self) -> None:
        self._spawn(
            self.client.call(verbs.SESSION_CANCEL, SessionParams(session_id=self.session_id))
        )

    def _spawn(self, work: Any) -> None:
        """Run an awaitable from a sync caller, owned by the app's worker pool so
        it is cancelled with the app. The sync members of `FrontSession` exist
        for key handlers, and key handlers exist only once there is an app."""
        if self.app is None:
            raise RuntimeError("attach_surfaces first: nothing owns background work yet")
        self.app.run_worker(work, exclusive=False)

    async def run_command(self, line: str) -> str | None:
        """Dispatch a `/name` line — the merge in `commands()`, read back.

        One path for both ends: a local definition's `run` dispatches a Textual
        action, a remote one's sends `session/command`. `parse_command_line` is
        the registry's own split, so a line tokenises the same way here as it
        would on the daemon.
        """
        name, argument = parse_command_line(line)
        for definition in self.commands():
            if definition.name == name:
                shown = await definition.run(argument, None)
                return str(shown) if shown else None
        raise KeyError(f'unknown command "/{name}"')

    async def shell(self, command: str) -> None:
        """`!!` in the session's workspace — the daemon's shell, not this one.

        A browser tab has no shell, and "the session's shell" is the honest
        meaning either way. Nothing is returned because the command and its
        output arrive as `shell/*` events, so the person who typed it reads it
        back off the same log as everybody else.
        """
        await self.client.mutate(
            verbs.SESSION_SHELL, ShellParams(session_id=self.session_id, command=command)
        )

    async def attach(self, paths: Sequence[str]) -> list[AttachmentRef]:
        """Read these files here and stage them on the root.

        **This client reads the bytes** — the human door (I-9), through the same
        `read_for_attach` the in-process store uses, so a person's file is
        classified the same way from either terminal. The daemon learns content
        and a name, never a path.
        """
        for path in paths:
            name, mime, content = await read_for_attach(path)
            await stage_bytes(self.client, self.session_id, name, mime, content)
        return self._staged.refs

    # ------------------------------------------------------------ lifecycle --

    def attach_surfaces(self, app: Any) -> list[Callable[[], Any]]:
        """Take the app, and build the local verbs that dispatch into it.

        Built once here rather than per `commands()` call, which the completion
        source makes on every keystroke. The daemon's own verbs belong to its
        registry and unwind with the row that made them; what unwinds here is
        what this client made — the local verbs and each screen's key binding.

        Imported inside, because `screens.py` is the Textual-shaped half of the
        front end and this module is driven headless by tests with no terminal.
        """
        self.app = app
        self._wire_screens(app)
        self.host.state_changed()

        def release() -> None:
            self._release_screens()
            self.app = None
            self._verbs = []

        return [release]

    def _wire_screens(self, app: Any) -> None:
        """Build this client's verbs: the table's, plus one per drawable screen.

        A screen buys a verb, a palette entry and a key — the three routes it
        buys in process too — and the key is a binding on the app, so replacing
        the set means releasing the old bindings first. Called again whenever
        `session.screens` says the deployment's list changed.
        """
        self._release_screens()
        self._verbs = local_commands(app)
        for screen in self.screens.values():
            action = open_screen_action(screen.id)
            self._verbs.append(action_command(app, screen.id, screen.label, action))
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

    def store_credential(self, name: str, value: str) -> bool:
        """Hand a secret to the daemon. Never logged, on either side. `True`
        optimistically: the member is sync, so the answer lands after this
        returns; `False` in process means "nowhere to put it", which over a
        socket `daemon/config` has already answered."""
        self._spawn(
            self.client.mutate(
                verbs.CREDENTIALS_STORE,
                StoreCredentialParams(session_id=self.session_id, name=name, value=value),
            )
        )
        return True

    async def flush(self) -> None:
        """Nothing to do: the daemon owns the log and flushes on its checkpoint
        policy, on passivation and in teardown. A client asking for a flush would
        be asking for a durability guarantee it neither provides nor can verify."""

    async def close(self) -> None:
        """Detach and stop reading. **The root keeps running** — the point of P5-01.

        No flush and no shutdown: this front end is leaving, not ending the
        session. What happens to the root afterwards is the daemon's business.

        **The detach is a courtesy, so its failure is not one.** The daemon drops
        a connection's subscriptions when the socket closes either way — that is
        `_Connection.serve`'s `finally` — and this only lets it happen a moment
        earlier, which matters to a client that keeps one connection across
        several sessions. So a connection already gone is a detach already done:
        checking `closed` cannot close the race, because the pump can stop
        between the check and the reply, and Textual cancelling its workers at
        shutdown is exactly when it does.
        """
        with suppress(DaemonGone), anyio.move_on_after(2.0):
            await self.client.call(verbs.SESSION_DETACH, SessionParams(session_id=self.session_id))


async def attach_session(
    client: DaemonClient,
    session_id: str,
    *,
    host: ModalHost,
    cwd: Path | None = None,
    trust: TrustAnswer = "",
) -> DaemonSession:
    """Start or resume a session on the daemon and catch this client up on it.

    The order is the one `session/attach`'s own docstring argues for: say what
    this client can answer, start the root, read its projections, subscribe, page
    the history from the cursor the attach reply named, then go live. The
    projections are independent reads of a mounted root and are fetched together;
    the attach follows them only so `front` can be built whole, and its reply is
    what seeds the status, the route and the footer.

    Credentials are deliberately not among them: which names a deployment *has*
    comes from `daemon/config`, which is fetched here, so a caller could not name
    them yet — and whether one is held can change while a session is open. The
    login screen asks for them when it opens, which is where the answer is read.
    """
    state = TuiState()
    client.handlers[ApprovalAsk.METHOD] = answering(ApprovalAsk, _asking_approval(host))
    client.handlers[QuestionAsk.METHOD] = answering(QuestionAsk, _asking_question(host))
    # `asks` **before** the attach: the desk joins a front end as it attaches, and
    # a client that declared nothing is never asked.
    await client.initialize("asks")
    # `trust` is the person's answer, which this client asked for and the daemon
    # enforces — it refuses a `cwd` nobody has vouched for (P5-14).
    # The reply carries this root's cursor, and so its generation — which is what
    # keys the mirror below. Read rather than discarded: `created_at` is stable
    # across a resume (`cursor_of` says so), so the number here is the one the
    # attach reply will name, and taking it now is what lets the mirror be built
    # whole instead of re-keyed afterwards.
    created = await client.call(
        verbs.SESSION_NEW,
        NewSessionParams(session_id=session_id, cwd=str(cwd) if cwd else None, trust=trust),
    )
    generation = created.cursor.generation

    # Three startup reads at once, each through the one typed door and each
    # landing in a slot of its own reply's type. The first draft kept a
    # `dict[str, dict[str, Any]]` and decoded afterwards with
    # `verbs.X.read(replies[verbs.X.name])` — which named the verb twice per
    # line, so `verbs.COMMANDS_LIST.read(replies[verbs.SCREENS_LIST.name])`
    # type-checked. That is the two-values-must-agree shape `Verb` exists to
    # delete, reintroduced at the three sites that had opted out of `call`, and
    # failing as a `ValidationError` at TUI startup rather than as a type error.
    # A one-slot list per read is what a heterogeneous dict could not be.
    async def fetch[P: WireModel, R: WireModel](verb: Verb[P, R], params: P, into: list[R]) -> None:
        into.append(await client.call(verb, params))

    configs: list[DaemonConfigReply] = []
    listed_commands: list[SessionCommandsNotice] = []
    listed_screens: list[SessionScreensNotice] = []
    listed_tools: list[SessionToolsReply] = []
    listed_skills: list[SessionSkillsReply] = []
    asked = SessionParams(session_id=session_id)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(fetch, verbs.DAEMON_CONFIG, NoParams(), configs)
        tasks.start_soon(fetch, verbs.COMMANDS_LIST, asked, listed_commands)
        tasks.start_soon(fetch, verbs.SCREENS_LIST, asked, listed_screens)
        # Two more reads in the same group rather than on demand, because both
        # are drawn in the sidebar from the first frame — a panel that filled in
        # a moment later would be a second kind of "not yet" beside the history
        # still paging. Neither changes while a session is open: a row that
        # registers a tool does it at mount, and installing a skill is a restart.
        tasks.start_soon(fetch, verbs.TOOLS_LIST, asked, listed_tools)
        tasks.start_soon(fetch, verbs.SKILLS_LIST, asked, listed_skills)

    config = configs[0]
    state.tools = _catalog(listed_tools[0].tools)
    state.skills = _catalog(listed_skills[0].skills)
    front = DaemonSession(
        client=client,
        session_id=session_id,
        state=state,
        # `tools=None`: the registry that renders a card is in the daemon, which
        # sends the rendered view beside each event — see `Frame.view`.
        adapter=TuiEventAdapter(state=state),
        host=host,
        config_rows=tuple(config.rows),
        remote_commands=[
            _remote_command(client, session_id, one) for one in listed_commands[0].commands
        ],
        screens=_screens_of(listed_screens[0].screens),
        generation=int(generation) if generation.isdigit() else None,
    )
    client.peer.on_notify = front.dispatch
    # The attach reply carries the status, the route and the footer, so this is
    # the one frame the front end starts from.
    attached = await client.call(verbs.SESSION_ATTACH, SessionParams(session_id=session_id))
    # Through the feed, which owns the rule that this reply is the first status
    # frame — the CLI reached for it separately and got a different answer.
    front.feed.seed(attached)
    # The attach reply's **cursor**, wound back to the start — not its `from`,
    # which is the *index* the live stream begins at and is not a cursor at all.
    # Passing it as one cost the client seq 0 of every session: `session/snapshot`
    # read a bare int as "no cursor", paged from 1, and the loss was invisible
    # until `Session(seed=…)` refused a log that did not start at 0 — which only
    # happens when somebody opens a screen. `ph agents attach` builds the same
    # shape for `--since`.
    await front.feed.catch_up(client, attached.cursor.model_copy(update={"sequence": 0}))
    front.feed.live()
    return front


class _Described(Protocol):
    """The two fields a catalog panel draws, whatever carries them.

    A Protocol rather than `Any`: `ToolSchema` and `Skill` differ in everything
    the sidebar does not show — a tool's parameter schema, a skill's path and
    version — and agree on these two. Named, the agreement is checked; as `Any`
    it was a comment, and a wire model that dropped `description` would have
    failed at the first draw instead of at the call.
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
    """One of the daemon's commands, with a `run` that actually runs it — there.

    The body is a `session/command`, so the palette, the completer and
    `run_command` see one kind of thing and dispatch it one way; which end
    executes a verb is the definition's business, not the caller's.
    """

    async def elsewhere(argument: str, _context: Any) -> str | None:
        line = f"/{schema.name} {argument}".rstrip()
        reply = await client.mutate(
            verbs.SESSION_COMMAND, CommandParams(session_id=session_id, line=line)
        )
        # A repeat answers with a description and no `shown`, which is the one
        # thing this caller wanted — so the union is narrowed rather than
        # ignored, and a re-sent command says nothing instead of saying the
        # wrong thing.
        return reply.shown if isinstance(reply, CommandShown) else None

    return CommandDefinition(
        name=schema.name,
        summary=schema.summary,
        run=elsewhere,
        argument_hint=schema.argument_hint,
    )


def _screens_of(schemas: Sequence[ScreenSchema]) -> dict[str, ScreenDefinition]:
    """The screens this deployment has *and* this client can draw.

    The wire supplies `label`, `order` and `key` — the deployment's own — and the
    local definition supplies `build`, the one field that cannot travel.
    """
    local = {definition.id: definition for definition in LOCAL_SCREENS}
    found: dict[str, ScreenDefinition] = {}
    for schema in schemas:
        mine = local.get(schema.id)
        if mine is None:
            log.debug("ph_app.tui: no local builder for screen %r", schema.id)
            continue
        found[schema.id] = ScreenDefinition(build=mine.build, **schema.model_dump())
    return found


def _asking_approval(host: ModalHost) -> Callable[[ApprovalAsk], Awaitable[ApprovalAskReply]]:
    """`approval/ask` → the modal, in a worker.

    `reason` travels back on the wire rather than being steered from here: the
    daemon holds the agent, and a client steering a turn it does not own would be
    writing into somebody else's session.
    """

    async def ask(asked: ApprovalAsk) -> ApprovalAskReply:
        outcome, reason = await host.ask_approval(asked.request)
        # Through the seam's own encoder: `Edited` and `Responded` are frozen
        # dataclasses, and putting one in a frame unencoded is a `TypeError`
        # inside the task group that answers the ask — which the desk reads as
        # "this front end cannot answer" and drops it for.
        return ApprovalAskReply(answer=answer_to_wire(outcome), reason=reason or "")

    return ask


def _asking_question(host: ModalHost) -> Callable[[QuestionAsk], Awaitable[QuestionAskReply]]:
    """`question/ask` → the ask-user modal, in a worker."""

    async def ask(asked: QuestionAsk) -> QuestionAskReply:
        return QuestionAskReply(answer=await host.ask_question(asked.question))

    return ask
