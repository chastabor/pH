"""The supervisor: roots that outlive the clients watching them (P5-01, I-2).

Every mode before this one ties an agent's life to a connection. `ph -p` exits
when the turn ends; the TUI's root dies with the terminal; `--mode rpc` lives as
long as stdin. That is right for a one-shot and wrong for the thing Phase 5 is
about — a run that takes an hour, that a person checks on from two machines, and
that must not stop because a laptop lid closed.

**One `anyio` task per root, and the client is not it.** A root owns a mounted
profile, a session, an agent and a queue; its task drains that queue whether or
not anybody is attached. Attaching subscribes a connection to the root's events;
detaching unsubscribes it. Neither starts nor stops the work, which is the whole
of the row's gate: *TUI close leaves the root running*.

**Addressed by id, never by connection.** A client says which root it means, so
two clients may watch one root and one client may watch two — and a reconnect is
an attach rather than a resume, because the root never stopped.

What this row deliberately does not do: capabilities, cursors and command
journaling (P5-02), leases against a second daemon (P5-03), crash retries
(P5-04), passivation (P5-05). The supervisor is the thing those attach to.

@module ph_app.daemon.supervisor
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.llm.types import create_user_message
from ph.persistence import resume_session, resumption_of, session_path
from ph.session import Session, SessionEvent

from ..protocol import cursor_of
from ..runtime import compose, mounted

__all__ = ["Root", "Supervisor"]

log = logging.getLogger("ph_app.daemon")

COMMAND_ACCEPTED = "client/command"
"""The record that makes a mutating command idempotent (P5-02)."""

WAKE_SLOTS = 8
"""How many un-acted-on doorbell rings to hold.

Small on purpose: a wake carries no payload, so a dropped one costs nothing as
long as *some* wake reaches an idle task — the message itself is already in the
agent's inbox and on disk."""

Subscriber = Callable[[str, dict[str, Any]], None]
"""`(event_name, payload)` — how a connection hears about a root it watches."""


@dataclass(slots=True)
class Root:
    """One long-running agent, and the wake channel its task waits on.

    A dataclass rather than a closure over the task, because the things a client
    asks about — is it busy, what is its session, who is watching — are read
    from *outside* the task that owns them.
    """

    id: str
    ctx: Context
    session: Session
    agent: Any
    wake: MemoryObjectSendStream[None]
    """Tells the root's task there is something in the inbox.

    A *signal*, not a queue: the durable queue is the agent's own `Inbox`, which
    logs `agent/inbox/spliced` before the projection changes precisely so a
    prompt survives a crash. A prompt held in a private memory stream would be
    lost with no record — in the one pH process built to outlive its clients.
    """
    waiting: MemoryObjectReceiveStream[None]
    exits: AsyncExitStack
    commands: set[str] = field(default_factory=set)
    """Commands already run, folded from this session's own log.

    A set rather than a scan, and derived rather than remembered: it is rebuilt
    from `client/command` events when a root starts, so a resumed root knows
    what it already did."""
    subscribers: set[Subscriber] = field(default_factory=set)
    """Attached connections. A set of bound methods, which compare by
    `(__self__, __func__)`, so attaching and detaching are symmetric without a
    token table to keep in step."""

    def subscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.discard(subscriber)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        """Tell every watcher. A failing subscriber is dropped, not raised.

        A client whose socket died — or one that cannot keep up — must not take
        the root down with it, which is the inversion this whole row exists to
        prevent. Dropping happens *here*, where the subscriber list is, so the
        policy has one owner: an earlier draft decided it in the connection and
        returned quietly, so the watcher it announced as dropped stayed
        subscribed and re-paid the fan-out for every later event.
        """
        for subscriber in list(self.subscribers):
            try:
                subscriber(event, payload)
            except Exception:
                log.debug("ph_app.daemon: dropping a watcher of root %s", self.id)
                self.subscribers.discard(subscriber)

    @property
    def status(self) -> str:
        """What the *agent* says, not a copy kept beside it.

        `agent.status` is a property over the driver's own phase and every
        transition emits `agent/status`; a second field set around this class's
        own `await` would be true only at that call boundary, and a steer, a
        cancel or an inbox wake between turns would move one and not the other.
        """
        return str(self.agent.status)

    @property
    def generation(self) -> str:
        """Which incarnation of this session a cursor belongs to.

        A cursor is `{generation, sequence}` and not a bare sequence because a
        sequence alone is only meaningful against the log it counted. Two things
        can invalidate it: a session forked from a prefix, and a log replaced
        rather than continued. The header's `createdAt` identifies the log a
        `seq` was counted against — stable across a resume (which continues the
        same log, by P5-01's fix) and different for anything that is not that
        log, which is exactly the distinction a client needs.
        """
        return str(self.session.header.created_at)

    def cursor(self) -> dict[str, Any]:
        """Where a client that has seen everything should resume from."""
        return {"generation": self.generation, "sequence": self.session.seq}

    def accepted(self, command: str) -> bool:
        """Whether this exact command already ran for this client.

        **Folded once, then kept.** The set is built from the log when the root
        starts — so it survives a restart, which is the one moment a client is
        most likely to retry: it reconnects, cannot know whether its last
        `session/prompt` landed, and sends it again. The first draft re-scanned
        the whole log per command instead, which measured 4.9 ms at 200 000
        events on the daemon's own event loop, stalling every other connection
        for that long.
        """
        return command in self.commands

    def remember(self, command: str) -> None:
        """Record a command in the log and in the fold that reads it back."""
        self.session.append(COMMAND_ACCEPTED, {"command": command})
        self.commands.add(command)

    def describe(self) -> dict[str, Any]:
        """What a client is told about this root.

        One name per fact: `rootId` and `sessionId` were the same string (a root
        *is* its session here) and `events` was `cursor.sequence`, which left a
        client picking which of two spellings was authoritative.
        """
        return {
            "sessionId": self.session.id,
            "status": self.status,
            "watchers": len(self.subscribers),
            "cursor": cursor_of(self.session),
        }


@dataclass(slots=True)
class Supervisor:
    """Every root this daemon is running, and the task group they live in."""

    documents: Sequence[Path]
    tasks: TaskGroup
    provider: str = "fake"
    model: str = "fake-1"
    roots: dict[str, Root] = field(default_factory=dict)
    _parsed: Sequence[tuple[str, Any]] | None = None

    async def start(self, root_id: str) -> Root:
        """Mount a profile, create its agent, and give it its own task.

        Through `runtime.mounted`, which exists so "a mode cannot drift from the
        profile semantics" — and whose `finally` is what a hand-rolled
        `Context()` + `mount()` pair loses: a mount that raised partway left a
        live `Context` that was never in `self.roots` and so was never disposed.
        An `AsyncExitStack` holds it because a root's lifetime is longer than
        any one `async with`, which is `HarnessSession`'s arrangement in the TUI.

        The mount is per root and not shared: two roots are two deployments as
        far as every seam is concerned — separate sessions, workspaces and
        kernels — and sharing one `Context` would make a row's `ctx.provide`
        visible to a root that never asked for it. The *parse* is shared, since
        the YAML cannot differ between them and re-reading it was most of the
        cost of starting one.
        """
        if root_id in self.roots:
            return self.roots[root_id]
        if self._parsed is None:
            self._parsed = compose(self.documents)
        exits = AsyncExitStack()
        run = await exits.enter_async_context(mounted(self.documents, parsed=self._parsed))
        ctx = run.ctx
        session = await self._session_for(ctx, root_id)
        agent = ctx.agents.create(session, AgentOptions(provider=self.provider, model=self.model))
        wake, waiting = anyio.create_memory_object_stream[None](max_buffer_size=WAKE_SLOTS)
        root = Root(
            id=root_id,
            ctx=ctx,
            session=session,
            agent=agent,
            wake=wake,
            waiting=waiting,
            exits=exits,
        )
        # What this session already did, read back from its own log — which is
        # what makes a retry after a restart safe rather than a second turn.
        root.commands.update(
            str(event.data.get("command", ""))
            for event in session.events_from(0)
            if event.type == COMMAND_ACCEPTED
        )
        self.roots[root_id] = root

        def relay(source: Session, event: SessionEvent) -> None:
            # Nothing is built before there is somebody to send it to: this runs
            # once per streamed chunk, and rendering a payload for zero watchers
            # measured 6.6 µs an event — 13 ms of a 2 000-chunk turn, discarded.
            if not root.subscribers:
                return
            root.publish(
                "session.event",
                # `thaw=False`: this payload's only destination is `dumps`,
                # which handles the frozen forms, and thawing deep-copies the
                # tree for nobody.
                {"sessionId": root.id, "event": event.to_wire(thaw=False)},
            )

        def announce(agent_: Any, status: str) -> None:
            if agent_ is agent:
                root.publish("session.status", {"sessionId": root.id, "status": status})

        # The session's own feed, not the store-wide `session/event` bus: a
        # child agent's events belong to its own transcript, and subscribing
        # here means never receiving them rather than receiving and discarding.
        exits.callback(session.observe(relay))
        ctx.on("agent/status", announce)
        self.tasks.start_soon(self._run, root)
        return root

    async def _session_for(self, ctx: Context, root_id: str) -> Session:
        """The root's session — resumed from disk when there is one to resume.

        **Creating unconditionally corrupted the log.** `sessions.create` mints a
        fresh session and the JSONL store appends, so restarting a daemon with
        the same root id concatenated a second session onto the first: one file,
        one header, and `seq` restarting at zero halfway through, which breaks
        A1 and makes every fold over that file double-count.

        Resuming is also what connects this to F6 — a root that died holding a
        worktree gets its `workspace/acquired` reconciled on the way back up,
        because `session/created` fires for an adopted session too.

        The resume is announced, not silent: whoever started this daemon may not
        know a previous run crashed, and "picked up where something left off" is
        the kind of surprise that should cost a line on stderr. The durable
        record is the `session/resumed` event `resume_session` appends — a cron
        job leaves the fact in the trace whether or not anyone reads stderr.
        """
        store = ctx.get("session_persistence")
        if store is not None and session_path(store.root, root_id).is_file():
            session: Session = await resume_session(ctx, root_id)
            resumed = resumption_of(session) or {}
            log.warning(
                "ph_app.daemon: resumed root %s from %s existing events%s",
                root_id,
                resumed.get("events", "?"),
                " — the previous run was interrupted" if resumed.get("interrupted") else "",
            )
            return session
        fresh: Session = ctx.sessions.create(root_id)
        return fresh

    async def _run(self, root: Root) -> None:
        """The root's own task: drive the agent whenever its inbox has work.

        This is the sentence the row is built on. Nothing here refers to a
        client, so nothing a client does — connecting, disconnecting, dying —
        appears in this loop at all.

        A wake that arrives while the agent is already running is dropped on
        purpose: `run()` drains the inbox until it is empty, so the turn already
        in flight will pick the new message up, and calling it twice raises.
        """
        async with root.waiting:
            async for _ in root.waiting:
                if root.agent.status != "idle":
                    continue
                await root.agent.run()
                await root.ctx.sessions.flush(root.session)

    async def prompt(self, root_id: str, text: str, *, command: str = "") -> Root:
        """Splice a turn into the agent's inbox and wake its task.

        Returns as soon as the message is *logged*, not when the turn is done:
        the caller is a socket handler serving other clients, and "the root is
        working" is a thing a watcher learns from `agent/status` rather than
        from a reply that never came.

        `command` is the client's idempotence key, already joined at the wire
        edge — one join, in the module that owns the wire, rather than the same
        f-string in two places that have to stay in step.
        """
        root = await self.start(root_id)
        if command:
            if root.accepted(command):
                # The retry a reconnecting client cannot avoid sending: it does
                # not know whether the first one landed. Answering "yes, that
                # one" is what makes asking twice safe.
                return root
            # Written *before* the splice, so a crash between the two re-runs a
            # command rather than losing it — the same write-ahead ordering A10
            # applies to blobs. A duplicated turn is visible in the transcript;
            # a dropped one is not.
            root.remember(command)
        root.agent.followup(
            create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})
        )
        # A full channel means the task has wakes pending and has not reached
        # them yet, so it will drain this message too — the inbox is the queue,
        # and this is only the doorbell.
        with suppress(anyio.WouldBlock):
            root.wake.send_nowait(None)
        return root

    def describe(self) -> list[dict[str, Any]]:
        return [root.describe() for root in self.roots.values()]

    async def aclose(self) -> None:
        """Close every root's wake channel and unwind its context.

        Channels first, so each task leaves its own loop rather than being
        cancelled mid-turn. Then per root, because one root's teardown failing
        must not strand the others (I2) — and each is flushed *before* it
        unwinds, since disposal appends events of its own and a session lost on
        exit is the worst way to learn that.
        """
        for root in self.roots.values():
            await root.wake.aclose()
        for root in list(self.roots.values()):
            try:
                await root.ctx.sessions.flush(root.session)
                await root.exits.aclose()
            except Exception:
                log.warning("ph_app.daemon: root %s did not unwind cleanly", root.id, exc_info=True)
        self.roots.clear()
