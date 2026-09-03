"""The terminal as a daemon client — the same screen, a harness elsewhere (P5-14).

`HarnessSession` mounts pH in the terminal's own process, so closing the terminal
ends the turn. This is the other implementation of the same `FrontSession`: every
method answers by asking the daemon, and the app above it cannot tell which one it
has. That equality is the point — one layout, one fold, one set of verbs, whether
the person is at a tty or in a browser tab.

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

**The `Session` is rebuilt from the log, on demand.** Only one thing wants it:
`ScreenDefinition.build(session)` when somebody opens a screen. `Session` has no
way to admit an already-numbered event, and inventing one to make a client-side
mirror writable would be a second append path into the type whose whole contract
is that appends are its own. The id is on the protocol separately so the sidebar
never asks for the whole thing.

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
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from ph.llm.types import AttachmentRef
from ph.seams.approval import ApprovalRequest
from ph.seams.attachments import read_for_attach
from ph.seams.commands import CommandDefinition, CommandSchema, parse_command_line
from ph.seams.tui_screens import ScreenDefinition, ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import Session, SessionEvent

from ..attach import Tray
from ..daemon.client import DaemonClient
from ..daemon.follow import Followed, first_of
from ..wire import obj, seq, view_of
from .adapter import Frame, TuiEventAdapter
from .commands import local_commands
from .frontend import ModalHost
from .state import TuiState
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
    directory: str = ""
    held: dict[str, bool] = field(default_factory=dict)
    feed: Followed = field(init=False)
    app: Any = None
    """The Textual app, once `attach_surfaces` has been given it — so a local verb
    has something to dispatch into and async work has an owner. `None` until then,
    which is the state a headless test drives this in."""
    _local: list[CommandDefinition] = field(default_factory=list)
    _log: list[SessionEvent] = field(default_factory=list)
    _session: Session | None = None
    _built_from: int = -1
    _readings: list[StatusReading] = field(default_factory=list)
    _moved: anyio.Event = field(default_factory=anyio.Event)
    """Set and replaced whenever the root's status changes — a wake-up for
    `_until_idle`, not a second copy of the status, which it re-reads."""
    _staged: Tray = field(default_factory=Tray)
    """This client's view of the root's tray, kept in step by `session.staged`.

    A mirror rather than the truth: the *root* holds the tray, because two people
    looking at one conversation must see one composer."""

    def __post_init__(self) -> None:
        self.feed = Followed(
            session_id=self.session_id, on_events=self._apply, on_status=self._status
        )

    # -------------------------------------------------------------- the log --

    @property
    def session(self) -> Session:
        """The log as a `Session`, rebuilt when it has grown.

        Keyed on the length this was built *from*, not on the built session's
        own length: `Session(seed=…)` appends a `session/end-seed` marker, so
        comparing the two lengths never matched and the cache never hit.
        """
        if self._built_from != len(self._log):
            self._session = Session(self.session_id, seed=self._log)
            self._built_from = len(self._log)
        assert self._session is not None
        return self._session

    def _apply(self, events: Sequence[tuple[Mapping[str, Any], Any]]) -> None:
        """Fold a run of wire events into the transcript and this client's log."""
        for wire, sidecar in events:
            try:
                event = SessionEvent.from_wire(wire)
            except Exception:
                log.debug("ph_app.tui: a frame would not rebuild as an event", exc_info=True)
                continue
            self._log.append(event)
            try:
                # `view_of` at the wire edge, so the fold is handed a type rather
                # than a mapping it would have to distrust; `tools` stays `None`
                # because the registry that renders a card is in the daemon.
                self.adapter.apply(event, Frame(live=True, view=view_of(event.type, sidecar)))
            except Exception:
                log.exception("ph_app.tui: the adapter refused an event")
        self.host.state_changed()

    def _status(self, params: Mapping[str, Any]) -> None:
        """`session.status`, which carries the footer beside it.

        Pushed rather than polled because a reading is a fold of the log, so the
        moment worth re-reading them is the moment the agent moved. The TUI's own
        30 Hz tick stays client-local: it exists for the spinner.
        """
        status = str(params.get("status") or "")
        if status:
            self.state.status = status
            self._moved.set()
            self._moved = anyio.Event()
        self._readings = [
            StatusReading.model_validate(obj(one)) for one in seq(params.get("readings"))
        ]
        self.host.state_changed()

    def dispatch(self, method: str, params: dict[str, Any]) -> None:
        """Every notification this front end reads, in one place.

        The feed owns the two it buffers; `session.staged` is a *snapshot* of the
        tray rather than a delta, so it is correct whatever order it arrives in
        and needs no buffer of its own.
        """
        if method in ("session.event", "session.status"):
            self.feed(method, params)
            return
        if method == "session.staged" and params.get("sessionId") == self.session_id:
            self._staged = Tray()
            for wire in seq(params.get("staged")):
                self._staged.stage(AttachmentRef.model_validate(obj(wire)))
            self.host.state_changed()

    # ----------------------------------------------------------- projections --

    def status_readings(self) -> list[StatusReading]:
        return list(self._readings)

    def commands(self) -> list[Any]:
        return [*self._local, *self.remote_commands]

    def screen(self, screen_id: str) -> Any:
        return self.screens.get(screen_id)

    def providers(self) -> list[Any]:
        """Nothing yet. Not enforced (§5 rule 6): the model picker over a socket
        has no projection — `llm.list_providers()` is not on the wire — so a remote
        front end offers no `/model` choices. In process it lists them."""
        return []

    def sessions_directory(self, fallback: Path) -> Path:
        return Path(self.directory) if self.directory else fallback

    def credential_held(self, name: str) -> bool:
        """From the last `credentials/held` answer — a fact about the *daemon's*
        store, fetched with the other projections at attach time."""
        return bool(self.held.get(name))

    async def refresh_credentials(self, names: Sequence[str]) -> None:
        reply = await self.client.call(
            "credentials/held", sessionId=self.session_id, names=list(names)
        )
        self.held = {str(key): bool(value) for key, value in obj(reply.get("held")).items()}

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
        self._spawn(self.client.call("session/cancel", sessionId=self.session_id))

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
        await self.client.mutate("session/shell", self.session_id, command=command)

    async def attach(self, paths: Sequence[str]) -> list[AttachmentRef]:
        """Read these files here and stage them on the root.

        **This client reads the bytes** — the human door (I-9), through the same
        `read_for_attach` the in-process store uses, so a person's file is
        classified the same way from either terminal. The daemon learns content
        and a name, never a path.
        """
        for path in paths:
            name, mime, content = await read_for_attach(path)
            put = await self.client.call(
                "attachment/put",
                sessionId=self.session_id,
                name=name,
                mime=mime,
                contentB64=b64encode(content).decode(),
            )
            await self.client.mutate(
                "session/stage", self.session_id, attachment=obj(put.get("attachment"))
            )
        return self._staged.refs

    # ------------------------------------------------------------ lifecycle --

    def attach_surfaces(self, app: Any) -> list[Callable[[], Any]]:
        """Take the app, and build the local verbs that dispatch into it.

        Built once here rather than per `commands()` call, which the completion
        source makes on every keystroke. There is nothing to register into — the
        remote verbs belong to the daemon's registry and unwind with the row that
        made them — so the disposer only puts this back.
        """
        self.app = app
        self._local = local_commands(app)
        self.host.state_changed()

        def release() -> None:
            self.app = None
            self._local = []

        return [release]

    def set_preset(self, name: str) -> None:
        self._spawn(self.client.mutate("session/preset", self.session_id, preset=name))

    def store_credential(self, name: str, value: str) -> bool:
        """Hand a secret to the daemon. Never logged, on either side. `True`
        optimistically: the member is sync, so the answer lands after this
        returns; `False` in process means "nowhere to put it", which over a
        socket `daemon/config` has already answered."""
        self._spawn(
            self.client.mutate("credentials/store", self.session_id, name=name, value=value)
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
        """
        if not self.client.closed.is_set():
            with anyio.move_on_after(2.0):
                await self.client.call("session/detach", sessionId=self.session_id)


async def attach_session(
    client: DaemonClient,
    session_id: str,
    *,
    host: ModalHost,
    cwd: Path | None = None,
    credentials: Sequence[str] = (),
) -> DaemonSession:
    """Start or resume a session on the daemon and catch this client up on it.

    The order is the one `session/attach`'s own docstring argues for: say what
    this client can answer, start the root, read its projections, subscribe, page
    the history from the cursor the attach reply named, then go live. The
    projections are independent reads of a mounted root and are fetched together;
    the attach follows them only so `front` can be built whole.
    """
    state = TuiState()
    client.handlers["approval/ask"] = _asking_approval(host)
    client.handlers["question/ask"] = _asking_question(host)
    # `asks` **before** the attach: the desk joins a front end as it attaches, and
    # a client that declared nothing is never asked.
    await client.initialize("asks")
    await client.call("session/new", sessionId=session_id, cwd=str(cwd) if cwd else None)

    replies: dict[str, dict[str, Any]] = {}

    async def fetch(method: str, **params: Any) -> None:
        replies[method] = await client.call(method, **params)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(fetch, "daemon/config")
        for method in ("commands/list", "screens/list", "session/readings"):
            tasks.start_soon(partial(fetch, method, sessionId=session_id))
        if credentials:
            tasks.start_soon(
                partial(fetch, "credentials/held", sessionId=session_id, names=list(credentials))
            )

    config = replies["daemon/config"]
    front = DaemonSession(
        client=client,
        session_id=session_id,
        state=state,
        # `tools=None`: the registry that renders a card is in the daemon, which
        # sends the rendered view beside each event — see `Frame.view`.
        adapter=TuiEventAdapter(state=state),
        host=host,
        config_rows=tuple(seq(config.get("rows"))),
        directory=str(config.get("sessionsDirectory") or ""),
        remote_commands=[
            _remote_command(client, session_id, obj(one))
            for one in seq(replies["commands/list"].get("commands"))
        ],
        screens=_screens_of(seq(replies["screens/list"].get("screens"))),
        held={
            str(k): bool(v) for k, v in obj(replies.get("credentials/held", {}).get("held")).items()
        },
    )
    client.peer.on_notify = front.dispatch
    attached = await client.call("session/attach", sessionId=session_id)
    front._status({**attached, "readings": replies["session/readings"].get("readings")})
    await front.feed.catch_up(client, attached.get("from"))
    front.feed.live()
    return front


def _remote_command(
    client: DaemonClient, session_id: str, wire: Mapping[str, Any]
) -> CommandDefinition:
    """One of the daemon's commands, with a `run` that actually runs it — there.

    The body is a `session/command`, so the palette, the completer and
    `run_command` see one kind of thing and dispatch it one way; which end
    executes a verb is the definition's business, not the caller's.
    """
    schema = CommandSchema.model_validate(wire)

    async def elsewhere(argument: str, _context: Any) -> str | None:
        line = f"/{schema.name} {argument}".rstrip()
        reply = await client.mutate("session/command", session_id, line=line)
        shown = reply.get("shown")
        return str(shown) if shown else None

    return CommandDefinition(
        name=schema.name,
        summary=schema.summary,
        run=elsewhere,
        argument_hint=schema.argument_hint,
    )


def _screens_of(wire: Sequence[Any]) -> dict[str, ScreenDefinition]:
    """The screens this deployment has *and* this client can draw.

    The wire supplies `label`, `order` and `key` — the deployment's own — and the
    local definition supplies `build`, the one field that cannot travel.
    """
    local = {definition.id: definition for definition in LOCAL_SCREENS}
    found: dict[str, ScreenDefinition] = {}
    for entry in wire:
        schema = ScreenSchema.model_validate(obj(entry))
        mine = local.get(schema.id)
        if mine is None:
            log.debug("ph_app.tui: no local builder for screen %r", schema.id)
            continue
        found[schema.id] = ScreenDefinition(build=mine.build, **schema.model_dump())
    return found


def _asking_approval(host: ModalHost) -> Any:
    """`approval/ask` → the modal, in a worker.

    `reason` travels back on the wire rather than being steered from here: the
    daemon holds the agent, and a client steering a turn it does not own would be
    writing into somebody else's session.
    """

    async def ask(params: dict[str, Any]) -> dict[str, Any]:
        request = ApprovalRequest.model_validate(obj(params.get("request")))
        outcome, reason = await host.ask_approval(request)
        return {"answer": outcome, "reason": reason}

    return ask


def _asking_question(host: ModalHost) -> Any:
    """`question/ask` → the ask-user modal, in a worker."""

    async def ask(params: dict[str, Any]) -> dict[str, Any]:
        question = UserQuestion.model_validate(obj(params.get("question")))
        return {"answer": await host.ask_question(question)}

    return ask
