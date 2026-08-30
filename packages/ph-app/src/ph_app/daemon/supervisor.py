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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from filelock import FileLock, Timeout

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.llm.types import create_user_message
from ph.persistence import resume_session, resumption_of
from ph.seams.subagents import child_is_live
from ph.seams.workspace import workspace_of
from ph.seams.workspace_git import latest_checkpoint, restore
from ph.session import Session, SessionEvent, now_ms
from ph.tools.errors import error_message

from ..protocol import Refusal, cursor_of
from ..runtime import compose, mounted
from .recovery import (
    FAILED,
    PASSIVATE_AFTER,
    PASSIVATED,
    RECOVERED,
    RETRY,
    Recovery,
    recovery_of,
)

__all__ = ["Root", "SessionBusy", "Supervisor"]


class SessionBusy(Refusal):
    """Another process holds this session's lease (I-5).

    Its own type so the wire can name it — the gate is that a concurrent open
    comes back as `session_already_active` rather than as a generic failure a
    client cannot branch on.
    """

    code = "session_already_active"


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
    recovery: Recovery = field(default_factory=lambda: Recovery(attempts=0, failed=False))
    """The retry ladder's state (P5-04), folded from the log when this root
    starts and maintained by `retry`/`give_up`/`recovered` from there.

    Held rather than re-folded, for the measurement `accepted` records below: a
    whole-log scan per read is 4.9 ms at 200 000 events, and `status` is read
    for every root on every `sessions/list`."""
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
        live = str(self.agent.status)
        # `retrying` is derived here too, not only published. Announcing it as a
        # notification while `sessions/list` still said "idle" gave a client
        # attaching mid-backoff two answers to one question — the drift this
        # property derives rather than copies in order to avoid.
        # The agent is authoritative whenever it has work in hand. It is only
        # *between* turns that "idle" is ambiguous — a root that exhausted the
        # ladder is idle in exactly the same way as one waiting for a prompt,
        # and P5-01 made this property derive rather than copy precisely so the
        # answer could not drift; so the second reading is derived too, from the
        # log, rather than from a flag set beside this class's own `await`.
        if live != "idle":
            return live
        if self.recovery.failed:
            return "failed"
        return "retrying" if self.recovery.attempts else live

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

    def retry(self, *, reason: str, restored: bool) -> None:
        """Record that a failed turn is being run again (P5-04).

        Written *before* the attempt, not after it: a daemon that died during
        the retry must come back knowing the attempt was made, or it resumes
        with a shorter ladder than it had actually spent. The same write-ahead
        ordering A10 applies to blobs and `remember` applies to commands.
        """
        self.session.append(
            RETRY,
            {
                "attempt": self.recovery.attempts + 1,
                "of": self.recovery.total,
                "delayMs": int(self.recovery.delay * 1000),
                "reason": reason,
                # Said out loud, because "false" is the ordinary case for an
                # advisory-tier root and a transcript that implied a rollback
                # nobody performed would misread the attempt that follows.
                "restored": restored,
            },
        )
        self.recovery = replace(self.recovery, attempts=self.recovery.attempts + 1)
        self.publish("session.status", {"sessionId": self.id, "status": "retrying"})

    def recovered(self) -> None:
        """A retry worked, so the ladder clears — and says so in the log.

        Recorded rather than simply forgotten, because forgetting is not durable:
        a root that recovered, then had the daemon restart, would otherwise fold
        its old `supervisor/retry` records back and resume with a ladder it had
        already climbed out of. It is also the *only* thing that resets the
        count, which is what keeps a failing retry from clearing its own bound.
        """
        self.session.append(RECOVERED, {"afterAttempts": self.recovery.attempts})
        self.recovery = Recovery(attempts=0, failed=False)
        self.publish("session.status", {"sessionId": self.id, "status": self.status})

    def idle_for(self, now: int) -> int:
        """Milliseconds since anything happened in this session (P5-05).

        From the log's own last event, so it means "nothing has happened" rather
        than "no client called us", survives a restart, and cannot drift from
        what the transcript shows. A log with no events yet is treated as busy:
        a root that has just been created has not been idle for ninety minutes,
        whatever the clock says.
        """
        last = self.session.last_event
        return 0 if last is None else max(0, now - int(last.time))

    def passivated(self, idle_ms: int) -> None:
        """Say in the log that this root is being released, before releasing it.

        Write-ahead, like every other record here: the append has to reach the
        buffer before the flush that carries it, or the last thing the log says
        is whatever the root was doing ninety minutes ago and the pause reads as
        a crash.
        """
        self.session.append(PASSIVATED, {"idleMs": idle_ms})
        self.publish("session.status", {"sessionId": self.id, "status": "passivated"})

    def give_up(self, reason: str, *, attempts: int) -> None:
        """Record that the ladder is spent, and tell whoever is watching.

        In the log first. A root that stopped working is exactly the fact an
        unattended run needs to leave behind — a cron-started agent reports it
        in its own trace whether or not a client was ever attached — and
        `Root.status` reads it straight back rather than keeping a copy.
        """
        self.session.append(FAILED, {"attempts": attempts, "reason": reason})
        self.recovery = replace(self.recovery, failed=True)
        self.publish("session.status", {"sessionId": self.id, "status": "failed"})
        log.error("ph_app.daemon: root %s failed after %d attempts — %s", self.id, attempts, reason)

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
    passivate_after: float | None = PASSIVATE_AFTER
    """Seconds of quiet before a root is released, or `None` to keep them all.

    A field beside `provider` and `model` because it is the same kind of thing —
    per-daemon configuration — and because threading it through `serve`, the
    sweeper task and two method signatures spelled one constant in eight places,
    two of them positionally unchecked through `start_soon`."""
    roots: dict[str, Root] = field(default_factory=dict)
    _starting: anyio.Lock = field(default_factory=anyio.Lock)
    _parsed: Sequence[tuple[str, Any]] | None = None

    async def start(self, root_id: str) -> Root:
        """Take the lease for this root, then mount it (I-5).

        Serialized per id, because the check-then-mount below spans two awaits:
        two clients asking for the same new root at once both pass the
        membership test, both mount a profile, and both reach for the same
        session file — I-5's hazard stated exactly ("two writers on one JSONL").
        The store's own uniqueness check does not save it: each root gets its
        *own* `Context` and therefore its own `SessionStore`, so neither knows
        about the other.

        The lease would catch that pair too, but it would catch it as a
        *refusal*, and a refusal is the wrong answer to the question these two
        clients asked. Two clients naming one root want the same root; only a
        second *process* is a conflict. So the ordering is here and the refusal
        is there, and the second racer gets the root the first one built.

        One lock rather than one per id, and a fast path that never reaches it.
        `prompt` calls this on *every turn*, so a per-id table minted an
        `anyio.Lock` per call to throw it away — 832 B and 0.65 µs each — and
        retained an entry per id ever started, cleared nowhere, in the one
        process built to run for weeks. A single lock costs a serialized mount
        (~2 ms) between two *different* new roots, which happens at most once
        per root, while the returning-client path now skips the lock's
        checkpoint entirely. `_start` re-checks membership under it, which is
        the ordinary double-checked build.
        """
        root = self.roots.get(root_id)
        if root is not None:
            return root
        async with self._starting:
            return await self._start(root_id)

    async def _start(self, root_id: str) -> Root:
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
        async with AsyncExitStack() as exits:
            run = await exits.enter_async_context(mounted(self.documents, parsed=self._parsed))
            ctx = run.ctx
            session = await self._session_for(ctx, root_id)
            options = AgentOptions(provider=self.provider, model=self.model)
            agent = ctx.agents.create(session, options)
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
            # Both folds are read back from this session's own log, once, here:
            # what it already did (so a retry after a restart is not a second
            # turn) and how far up the retry ladder it got (so a root resumed
            # mid-ladder does not start the count over and retry forever).
            root.recovery = recovery_of(session)
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
            # Ours now: the `async with` unwinds a stack that has been emptied,
            # so a failure anywhere above disposes everything it entered and a
            # success hands the whole stack to the root. The hand-rolled
            # `except BaseException: aclose()` this replaces guarded only the
            # three lines it wrapped — `agents.create`, the channel and the
            # `session.observe` callback all sat outside it, and which side of
            # that boundary a new line lands on was invisible.
            root.exits = exits.pop_all()
            return root

    async def _lease(self, ctx: Context, path: Path, root_id: str) -> None:
        """Claim one session log against every other writer (I-5).

        The lock in `start` orders the racers *inside* this process; this is what
        stops a second daemon from appending to a log this one is writing. Two
        writers on one JSONL produce exactly the corruption P5-01 measured when
        the daemon concatenated sessions onto each other: `seq` going backwards
        mid-file, which breaks A1 and makes every fold double-count.

        **Daemon against daemon, and no further.** The lease is taken here, so a
        `ph -p --session x` run against a session a daemon holds still opens it
        — the hazard belongs to `JsonlSessionStore`, which is the thing that
        actually writes, and leasing there would cover every mode at once. I-5
        names the second daemon and that is what this row gates; the CLI half is
        left for the row that moves the lease down into the store rather than
        claimed here by a docstring.

        Taken beside the log rather than at a path of its own construction: the
        store owns where sessions live, and a lease derived independently is one
        `PH_HOME` change away from guarding a file nobody writes.

        `timeout=0` — "somebody else holds it" is known immediately and is a
        refusal, not something to wait out; blocking here would also stall the
        event loop for every *other* root this supervisor is running.

        Taken through `ctx.effect`, which is the repo's one mechanism for this
        and names the case verbatim — *"every external artifact an agent takes —
        a child process, a worktree, a temp path, a lock — is acquired through
        here, so cleanup is structural rather than remembered (§4.9, I2)"*. The
        first draft registered `lock.release` on a hand-held `AsyncExitStack`,
        which made the lease the one artifact in the tree invisible to
        `ctx.dispose()` and its labelled disposal log, and threaded the stack
        down through `_session_for` to get there.

        Acquired *inline*, not on a worker thread. Wrapping it in
        `to_thread.run_sync` measured **+340 µs on a 1.9 ms root start — twice
        the 166 µs the acquire itself costs** — because a real start is seconds
        after the last one and pays a cold thread plus a cold selector wakeup
        every time. At `timeout=0` the acquire is one `os.open` and a
        non-blocking `flock`: it cannot wait, so there is no blocking to move
        off the loop. Its neighbours settle it — `path.is_file()` two lines down
        and `resume_session`'s whole-log read are both on the loop thread, so a
        200 µs threshold is not one this function was holding.

        `thread_local=False` is load-bearing, and its absence is silent.
        filelock keeps its re-entrancy counter in a thread-local by default, so
        a lease acquired on a worker thread and released from the event loop
        finds a counter of zero and returns *having released nothing* — no
        error, no warning, and a lock file held until the process dies. It cost
        three tests, two of them P5-01's, all failing as "this session is
        already active" against a daemon that had cleanly shut down. The lease
        belongs to the process, not to whichever thread happened to take it.
        """

        def acquire() -> Callable[[], None]:
            # No mkdir: filelock's own `ensure_directory_exists` is the same
            # `parents=True, exist_ok=True` call on the same directory one
            # statement later, and `JsonlSessionStore.track` makes a third. All
            # three fired on every root start.
            lock = FileLock(f"{path}.lock", timeout=0, thread_local=False)
            try:
                lock.acquire()
            except Timeout as error:
                raise SessionBusy(f'session "{root_id}" is already active') from error
            return lock.release

        await ctx.effect(acquire, label=f"session-lease({root_id})")

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
        # A profile with no persistence writes nothing, so it has no log to
        # lease and none to resume — `path` stays `None` and both fall through
        # to the one `create` below.
        store = ctx.get("session_persistence")
        # `locate` may honestly answer `None` — a backend with no per-session
        # file has nothing to lease, and P5-03's lease must decline rather than
        # invent a path, because a lock on a file nobody writes protects nothing
        # while looking like it does. That backend brings its own concurrency
        # story; this one is the filesystem's.
        path = store.locate(root_id) if store is not None else None
        if path is not None:
            await self._lease(ctx, path, root_id)
        elif store is not None:
            # Said out loud. A backend with no per-session file gets no I-5
            # lease, and a *silent* skip is the shape that hides it: two daemons
            # would open one session and nothing would refuse. The store owning
            # its own claim — `claim(session_id)` on the Protocol rather than a
            # path accessor — is the real answer and is its own row.
            log.warning(
                "ph_app.daemon: %s provides no lease path; I-5 is not enforced for %s",
                type(store).__name__,
                root_id,
            )
        if store is not None and store.exists(root_id):
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

        Crashes and their ladder belong to `_drive`, which this calls once per
        wake — so nothing raising out of a root can cancel the supervisor's task
        group and take every *other* root down with it.
        """
        async with root.waiting:
            async for _ in root.waiting:
                if root.agent.status != "idle":
                    continue
                await self._drive(root)

    async def _drive(self, root: Root) -> None:
        """One wake, and the ladder if the root's task crashes (P5-04).

        **One root's crash is not the daemon's.** This runs in the supervisor's
        task group, so anything raising out of here cancels the group and takes
        every *other* root down with it — the failure a supervisor exists to
        prevent. `run()` contains its own turn failures, so what reaches this
        boundary is the unanticipated kind: a flush that cannot write, a
        disposed context, a bug.

        Those are worth retrying because the work is still in the inbox — the
        crash happened around the turn rather than inside it — which is exactly
        what a *turn* failure is not: that one already claimed its message, so
        running again would produce an empty turn reporting false success. See
        `recovery` for why that distinction decides the whole row.

        The delay is spent before the retry, not after the failure, so a root
        that gives up does so immediately rather than sleeping first.
        """
        while True:
            try:
                await root.agent.run()
                await root.ctx.sessions.flush(root.session)
                if root.recovery.attempts:
                    # Only after a ladder was actually climbed, so an ordinary
                    # turn writes nothing. This is what clears the count, and it
                    # has to be a record only success can write.
                    root.recovered()
                    await self._flush(root)
                return
            except (anyio.get_cancelled_exc_class(), anyio.ClosedResourceError):
                # Teardown, not failure. Recording a give-up here would write to
                # a session that is being disposed and would libel a root that
                # was only ever asked to stop.
                raise
            except Exception as error:
                log.exception("ph_app.daemon: root %s crashed outside a turn", root.id)
                state = root.recovery
                if state.spent:
                    root.give_up(error_message(error), attempts=state.attempts)
                    # Flushed, like the retries before it. This is the record
                    # that matters most and it was the one write-through missed:
                    # a daemon stopping right after giving up left the give-up
                    # in a buffer, so the next daemon resumed the log, saw no
                    # `agent/failed`, and reported the root idle — a ladder that
                    # forgets it was spent is one that starts over forever.
                    await self._flush(root)
                    return
                restored = await self._restore(root)
                root.retry(reason=error_message(error), restored=restored)
                await self._flush(root)
                await anyio.sleep(state.delay)

    def _live_schedules(self, root: Root) -> list[Any]:
        """This root's schedules that could still fire, or an empty list.

        The seam lookup and its `None` guard written once: three callers wanted
        it — the tick, the heartbeat and the passivation predicate — and each
        had its own copy of the name, the guard and the read.
        """
        schedule = root.ctx.get("schedule")
        return [] if schedule is None else list(schedule.live(root.session))

    async def tick(self, *, now: int | None = None) -> list[str]:
        """Fire whatever is due on every mounted root (P5-06). Returns their ids.

        **Claim then deliver, never the other way round.** `ScheduleService.claim`
        appends `schedule/tick` before returning, so a crash between the two
        costs a skipped run rather than a repeated one — and a repeated run of a
        scheduled prompt bills twice and puts a turn in the transcript nobody
        asked for. The delivery is `prompt`, which is the same path a person's
        message takes, so a scheduled turn is an ordinary turn with a record
        saying why it started.

        One stamp for the whole pass, so two schedules due in the same second
        agree about what "now" was.
        """
        stamp = now if now is not None else now_ms()
        fired: list[str] = []
        for root in list(self.roots.values()):
            schedule = root.ctx.get("schedule")
            if schedule is None:
                continue
            # The whole per-root body, not just the claim. The guard used to
            # cover `claim` alone — which barely raises, since a bad expression
            # logs and declines — while `prompt` and `_flush`, the two calls
            # that genuinely fail, sat outside it. One bad root taking the pass
            # down is what this is for.
            try:
                claimed = schedule.claim(root.session, now=stamp)
                for entry in claimed:
                    log.info("ph_app.daemon: root %s firing schedule %s", root.id, entry.id)
                    await self.prompt(root.id, entry.prompt)
                    fired.append(entry.id)
                # Only when something was appended. The condition also read
                # `or schedule.live(...)`, which folded the whole log a second
                # time to decide to flush a buffer the first fold had just left
                # empty — 24 ms a root at 500 000 events, every five seconds.
                if claimed:
                    await self._flush(root)
            except Exception:
                log.exception("ph_app.daemon: root %s failed its schedule tick", root.id)
        return fired

    async def heartbeat(self, *, now: int | None = None) -> None:
        """Leave a liveness record on every root that has work scheduled.

        A record, not a keep-alive: a schedule that fires monthly otherwise
        leaves a log whose last line is a month old, which reads exactly like a
        log nobody is writing any more.
        """
        stamp = now if now is not None else now_ms()
        for root in list(self.roots.values()):
            live = self._live_schedules(root)
            if live:
                root.ctx.schedule.heartbeat(root.session, now=stamp, live=len(live))
                await self._flush(root)

    async def _release(self, root: Root) -> None:
        """Flush a root's log, then unwind everything it took.

        One spelling, because there are two callers — `aclose` at shutdown and
        `passivate` when a root goes quiet — and this pair has diverged here
        before: `_flush` exists because `aclose`'s earlier copy skipped
        `exits.aclose()` when the flush raised, leaving an unwritable root's
        context undisposed. P5-05 reintroduced the same two-copies shape, down
        to the identical warning string. Whatever joins root teardown next — a
        lease, a reclaim, a cache eviction — now has one place to be added and
        cannot land in only one of them.
        """
        await self._flush(root)
        try:
            await root.exits.aclose()
        except Exception:
            log.warning("ph_app.daemon: root %s did not unwind cleanly", root.id, exc_info=True)

    async def _flush(self, root: Root) -> None:
        """Get the ladder's own record to disk, or carry on without it.

        The crash being retried may well *be* a failing flush, and a ladder that
        raised while recording that it was retrying would turn one broken root
        into a task that dies with no account of why. `aclose` wants the same
        thing for the same reason — and had its own copy, which additionally
        skipped `exits.aclose()` when the flush raised, so one root's unwritable
        log left its context undisposed.
        """
        try:
            await root.ctx.sessions.flush(root.session)
        except Exception:
            log.warning("ph_app.daemon: root %s could not flush its retry", root.id, exc_info=True)

    async def _restore(self, root: Root) -> bool:
        """Put the root's tree back to its last restore point, if it has one.

        A retry that ran against a half-mutated tree would be a different turn
        from the one that failed — the model would see edits from an attempt
        nobody kept, and a ladder that compounds its own damage is worse than no
        ladder. `workspace/checkpoint` is P4-09's record and already a fold, so
        this asks the log rather than remembering anything.

        Best-effort by construction: an advisory-tier root has no worktree and
        nothing to restore, and a restore that fails must not cost the retry. In
        both cases the attempt goes ahead against the tree as it stands, and the
        `agent/retry` record says `restored: false` so the transcript does not
        imply a rollback that did not happen.
        """
        # `workspace_of`, not `ctx.workspace.of`: this runs inside `_drive`'s
        # `except`, and `ctx.workspace` *raises* on a profile that layers no
        # workspace row — so the raw lookup would escape the handler whose whole
        # job is keeping the task group alive. The helper is fail-soft, and is
        # the one spelling of this question the rest of the tree uses.
        workspace = workspace_of(root.ctx, root.agent)
        if workspace is None:
            return False
        tree = latest_checkpoint(root.session, root.agent.id)
        if not tree:
            return False
        try:
            await restore(root.ctx, workspace, tree)
        except Exception:
            log.warning(
                "ph_app.daemon: root %s could not be restored to %s", root.id, tree, exc_info=True
            )
            return False
        return True

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

    def passivatable(self, root: Root, *, now: int, after: float) -> bool:
        """Whether this root may be released (P5-05).

        Every condition is a reason a root is still *wanted*, and each is read
        from something that already exists rather than from a flag set beside
        it:

        * **it is doing something** — `status` covers `running`, and covers
          `retrying`, which matters: a root in P5-04's backoff is idle between
          attempts and releasing it there would passivate a root mid-ladder;
        * **somebody is watching** — `subscribers`, which is the root's own
          attachment set, so a client that attached and never detached keeps its
          session alive by the same fact that makes it receive events;
        * **it has live children** — folded from `subagent/*`, because a parent
          released while a child is still running would be rehydrated by the
          child's own events arriving at a root that no longer exists;
        * **it has been quiet long enough**, from the log.

        * **it has work scheduled** — a root with a live schedule is going to
          be needed again, and releasing it would be releasing something that
          has already said when it comes back (P5-06).

        Heartbeats were the fourth condition and turn out not to be one: a
        heartbeat is a *record* the scheduler leaves so an operator can tell
        "waiting for Wednesday" from "died on Tuesday", not a claim on the
        root's life. What keeps a scheduled root mounted is the schedule.
        """
        if root.status != "idle":
            return False
        if root.subscribers:
            return False
        # **The quiet check before the fold**, which is not merely tidier. The
        # root reaching this line is idle and unwatched — exactly the steady
        # state the sweeper exists for — so a fold above it runs on every sweep
        # of the whole ninety-minute window and is discarded eighty-nine times
        # out of ninety. Measured over one idle window at 500 000 events across
        # 50 roots: **60.9 s of event loop as written, 1.3 ms with this line
        # first**, and `idle_for` costs 43 ns.
        if root.idle_for(now) < after * 1000:
            return False
        # Through the seam's cached fold rather than the bare function:
        # `SessionFoldCache` keys on `session.seq`, and an idle root's log does
        # not grow, so every sweep after the first is a dict hit instead of a
        # whole-log walk (0.09 µs against 13.5 ms at 500 000 events). That is
        # what saves the root this returns `False` for — idle, unwatched, one
        # unsettled child — which would otherwise re-fold every sixty seconds
        # for the life of the daemon: 16 minutes of event loop a day at 50 such
        # roots.
        subagents = root.ctx.get("subagents")
        roster = subagents.roster(root.session) if subagents is not None else {}
        if any(child_is_live(child) for child in roster.values()):
            return False
        # Last, and cached the same way: this one was inserted *above* the
        # comment describing the cached fold, so the paragraph arguing against
        # a bare whole-log walk sat directly on top of one.
        return not self._live_schedules(root)

    async def sweep(self, *, after: float | None = None, now: int | None = None) -> list[str]:
        """Release every root that has been quiet long enough. Returns their ids.

        Returns rather than logs so a test — and P5-10's `ph agents` — can ask
        what happened without parsing a log line, and so the caller decides
        whether a sweep that released nothing is worth saying out loud.

        Iterates a copy: passivation removes from `self.roots`.
        """
        stamp = now if now is not None else now_ms()
        window = self.passivate_after if after is None else after
        if window is None:
            return []
        released: list[str] = []
        for root in list(self.roots.values()):
            if self.passivatable(root, now=stamp, after=window):
                await self.passivate(root, now=stamp)
                released.append(root.id)
        return released

    async def passivate(self, root: Root, *, now: int) -> None:
        """Release a root's process-side state, keeping its session on disk.

        **Rehydration is already written**: `start()` resumes any root whose log
        exists (P5-01), so waking one is the ordinary path rather than a second
        mechanism — which is why this row is mostly a sweeper and a record, and
        why the round-trip is a property rather than a feature.

        Order matters and is the same order `aclose` uses: record, flush, then
        unwind. Unwinding disposes the mounted `Context` and, with it, the
        P5-03 lease on this session's log — so a passivated session is one
        another process may legitimately open, which is the point rather than an
        oversight.

        The wake channel closes first so the root's task leaves its own loop
        instead of being cancelled mid-turn, exactly as in `aclose`.
        """
        async with self._starting:
            # The same lock `start` orders itself with, and for the mirror-image
            # reason. Passivation removes the root from `self.roots` and *then*
            # unwinds — so without this, a `start` arriving in between finds no
            # root, tries to open the session, and is refused by the P5-03 lease
            # the outgoing root has not released yet: `session_already_active`
            # for a session nobody is using. `session/attach` made that
            # reachable by waking a passivated root through `start`.
            if self.roots.get(root.id) is not root:
                return
            await self._passivate(root, now=now)

    async def _passivate(self, root: Root, *, now: int) -> None:
        idle_ms = root.idle_for(now)
        root.passivated(idle_ms)
        log.info(
            "ph_app.daemon: passivating root %s after %d minutes idle", root.id, idle_ms // 60_000
        )
        self.roots.pop(root.id, None)
        subagents = root.ctx.get("subagents")
        if subagents is not None:
            # The cached roster would outlive the root otherwise: the seam keys
            # its fold by session id and nothing else tells it this one is gone.
            subagents.forget_session(root.id)
        schedule = root.ctx.get("schedule")
        if schedule is not None:
            schedule.forget_session(root.id)
        await root.wake.aclose()
        await self._release(root)

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
            await self._release(root)
        self.roots.clear()
