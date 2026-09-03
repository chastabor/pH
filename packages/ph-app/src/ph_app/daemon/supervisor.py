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
from collections.abc import Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from filelock import FileLock, Timeout

from ph.agent.types import AgentOptions
from ph.cordis import Context, Profile
from ph.llm.types import user_text
from ph.paths import resolve_roots
from ph.persistence import resume_session, resumption_of
from ph.seams.schedule import Schedule, state_to_wire
from ph.seams.schedule_index import ScheduleIndex
from ph.seams.subagents import child_is_live
from ph.seams.workspace import workspace_of
from ph.seams.workspace_git import latest_checkpoint, restore
from ph.session import Session, SessionEvent, now_ms
from ph.tools.errors import error_message

from ..protocol import Refusal, cursor_of
from ..runtime import mounted
from .cards import CARD_EVENTS, presentation_of
from .frontend import AskDesk
from .projections import readings_of
from .recovery import (
    FAILED,
    PASSIVATE_AFTER,
    PASSIVATED,
    RECOVERED,
    RETRY,
    UNREACHABLE,
    WAKE_WITHIN,
    Recovery,
    recovery_of,
)

__all__ = ["NON_GUARANTEES", "Root", "ScheduleUnavailable", "SessionBusy", "Supervisor"]


class ScheduleUnavailable(Refusal):
    """This root's profile did not mount the `schedule` seam (P5-06).

    Its own type for `SessionBusy`'s reason: "there is nothing to schedule
    against here" is a fact about the profile a client can act on — mount the
    row, or stop asking — and a generic failure would have it guessing whether
    the schedule was rejected or the id was wrong.
    """

    code = "schedule_unavailable"


class SessionBusy(Refusal):
    """Another process holds this session's lease (I-5).

    Its own type so the wire can name it — the gate is that a concurrent open
    comes back as `session_already_active` rather than as a generic failure a
    client cannot branch on.
    """

    code = "session_already_active"


log = logging.getLogger("ph_app.daemon")


NON_GUARANTEES: tuple[tuple[str, str], ...] = (
    (
        "worker model",
        "one anyio task per root, all in this process — not a process per root (Q7)",
    ),
    (
        "per-root memory",
        "not capped. A root that allocates without bound is killed by the OS as one "
        "process, and every other root goes with it",
    ),
    (
        "crash containment",
        "per root, not per process. A root's own crash is caught, retried and given up "
        "on in its log (P5-04); a segfault in a C extension, an OOM kill or a SIGKILL "
        "ends every root at once, and SIGKILL runs no teardown at all (N7)",
    ),
    (
        "CPU",
        "shared. One root's CPU-bound work delays every other root's turn — only "
        "ph-rlm's kernels run model code in child processes with limits of their own",
    ),
    (
        "restart",
        "not rolling. Stopping the daemon stops every root; the next daemon resumes one "
        "from its log when a client asks for it, or when a schedule of its own comes due",
    ),
    (
        "while the daemon is down",
        "nothing fires. A schedule is kept only while `ph daemon` runs; on start it "
        "catches up, one run per missed window. For work that must happen whether or "
        "not the daemon is up, start it from cron, anacron or a systemd timer — pH "
        "schedules inside a conversation and does not replace them",
    ),
    (
        "a question a person walked away from",
        "re-posed only while this daemon runs. `AskDesk` holds an open ask in memory "
        "and puts it to whoever attaches next; the log keeps the question either way "
        "(`question/asked` with no `question/answered`, which `pending_questions` "
        "folds), but nothing reads that fold on resume yet — so a daemon that stopped "
        "mid-question does not re-ask by itself, and the turn that was waiting is gone "
        "(P7-09)",
    ),
    (
        "per user",
        "one daemon per $PH_RUNTIME — a second on the same socket is refused rather "
        "than merged (P5-01), and a second writer on one session log is refused by its "
        "lease (I-5). Isolation *between users* is the operator's layer",
    ),
)
"""What this supervisor does **not** promise, in the module that would imply it (N5).

Rule 6: *state what is not enforced, next to where it would be assumed* — and a
caveat only in the docs is a defect. What is assumed here is what "one daemon,
many long-running agents" sounds like it means, and every row is a place where it
does not mean it.

Data, not prose: `ph doctor` and `ph agents doctor` print these verbatim, so the
sentences a person reads and the sentences this file is responsible for keeping
true are the same strings. A paragraph in a docstring cannot be printed, and a
sentence nobody can print is one nobody checks.
"""

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

    Held rather than re-folded, for the reason `accepted` gives below: `status` is
    read for every root on every `sessions/list`, and a whole-log scan per read puts
    that on the daemon's event loop."""
    commands: set[str] = field(default_factory=set)
    """Commands already run, folded from this session's own log.

    A set rather than a scan, and derived rather than remembered: it is rebuilt
    from `client/command` events when a root starts, so a resumed root knows
    what it already did."""
    subscribers: set[Subscriber] = field(default_factory=set)
    """Attached connections. A set of bound methods, which compare by
    `(__self__, __func__)`, so attaching and detaching are symmetric without a
    token table to keep in step."""
    desk: AskDesk | None = None
    """This root's `AskDesk` — who gets asked when a turn needs a person (P5-13).

    On the root rather than on a connection because the *question* outlives any
    client: nobody attached means the ask waits, and the log holds it either
    way."""

    def subscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.discard(subscriber)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        """Tell every watcher. A failing subscriber is dropped, not raised.

        A client whose socket died — or one that cannot keep up — must not take the root
        down with it, which is the inversion this row exists to prevent. Dropping happens
        *here*, where the subscriber list is, so the policy has one owner.
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
        # Parked on a person outranks what the agent says: it reports `running`
        # while an approval is open, which is true and not useful. See
        # `test_a_root_parked_on_a_person_may_be_released`.
        if self.desk is not None and self.desk.waiting:
            return "waiting"
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

        **Folded once, then kept.** Built from the log when the root starts, so it
        survives a restart — the one moment a client is most likely to retry, because it
        reconnects, cannot know whether its last `session/prompt` landed, and sends it
        again. Re-scanning per command would put that read on the daemon's event loop,
        stalling every other connection.
        """
        return command in self.commands

    def remember(self, command: str) -> None:
        """Record a command in the log and in the fold that reads it back."""
        self.session.append(COMMAND_ACCEPTED, {"command": command})
        self.commands.add(command)

    def once(self, command: str) -> bool:
        """Claim this command, or say it was already claimed. `True` means act.

        The write-ahead ordering, next to the two halves it orders, because there
        are two mutating verbs now — `session/prompt` and `session/command` — and
        each spelling it out is one that can drift. The record is written
        **before** the act, so a crash between them re-runs the command rather
        than losing it: the same reasoning A10 applies to blobs, and a duplicated
        turn is visible in the transcript where a dropped one is not.

        An empty key means the caller offered no identity and wants no
        deduplication; it always acts.
        """
        if not command:
            return True
        if self.accepted(command):
            # The retry a reconnecting client cannot avoid sending: it does not
            # know whether the first one landed. Answering "yes, that one" is
            # what makes asking twice safe.
            return False
        self.remember(command)
        return True

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

    def unreachable(self, note: dict[str, Any]) -> None:
        """Record that the supervisor lost the socket it was bound to (P5-11).

        An append and nothing else, where every other record here appends *and*
        publishes: `relay` observes this session's own feed, so the `session.event` frame
        goes out on the same append, and a client attached before the path went away is
        still on a live stream. A second `publish` would send one fact twice under two
        names.

        **Not a status change.** This root is doing exactly what it was doing; what broke
        is the door. Reporting it as `failed` would put the recovery ladder to work
        climbing over a socket.
        """
        self.session.append(UNREACHABLE, note)

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

    def detail(self) -> dict[str, Any]:
        """`describe`, plus what a client asking about *one* root wants.

        Here rather than assembled at the wire edge, which is where the two
        recovery fields were read from: a root's client-facing projection split
        across two files is how `sessions/list` and `session/status` come to use
        different names for the same object. `status` already collapses the
        ladder to `"retrying"` or `"failed"`, and this is where the rungs behind
        that answer are — so a new one is added once.
        """
        return {
            **self.describe(),
            "attempts": self.recovery.attempts,
            "failed": self.recovery.failed,
        }


@dataclass(slots=True)
class Supervisor:
    """Every root this daemon is running, and the task group they live in."""

    profile: Profile
    tasks: TaskGroup
    provider: str = "fake"
    model: str = "fake-1"
    passivate_after: float | None = PASSIVATE_AFTER
    """Seconds of quiet before a root is released, or `None` to keep them all.

    A field beside `provider` and `model` because it is the same kind of thing —
    per-daemon configuration — and because threading it through `serve`, the
    sweeper task and two method signatures spelled one constant in eight places,
    two of them positionally unchecked through `start_soon`."""
    wake_within: float | None = WAKE_WITHIN
    """How stale an indexed appointment may be and still wake its root (P6-23).

    `None` — the default — catches up on whatever was missed, however long the
    daemon was down, which is the scheduler's whole promise. A deployment that
    would rather not resurrect a long-abandoned session sets a bound here."""
    roots: dict[str, Root] = field(default_factory=dict)
    _starting: anyio.Lock = field(default_factory=anyio.Lock)

    async def start(self, root_id: str, *, cwd: str | None = None) -> Root:
        """Take the lease for this root, then mount it (I-5).

        **Serialized per id**, because the check-then-mount below spans two awaits: two
        clients asking for the same new root at once both pass the membership test, both
        mount a profile, and both reach for the same session file — I-5's hazard exactly.
        The store's own uniqueness check does not save it, because each root gets its own
        `Context` and therefore its own `SessionStore`.

        The lease would catch that pair as a *refusal*, which is the wrong answer to the
        question these two clients asked: two clients naming one root want the same root,
        and only a second *process* is a conflict. So the ordering is here and the
        refusal is in `_lease`, and the second racer gets the root the first one built.

        One lock rather than one per id, with a fast path that never reaches it —
        `prompt` calls this on *every turn*. `_start` re-checks membership under it,
        which is the ordinary double-checked build.
        """
        root = self.roots.get(root_id)
        if root is not None:
            return root
        async with self._starting:
            return await self._start(root_id, cwd=cwd)

    async def _start(self, root_id: str, *, cwd: str | None = None) -> Root:
        """Mount a profile, create its agent, and give it its own task.

        Through `runtime.mounted`, so a mode cannot drift from the profile semantics —
        and its `finally` is what a hand-rolled `Context()` + `mount()` pair loses: a
        mount that raised partway left a live `Context` that was never in `self.roots`
        and so was never disposed. An `AsyncExitStack` holds it because a root's lifetime
        is longer than any one `async with`.

        The mount is **per root and not shared**: two roots are two deployments as far as
        every seam is concerned, and sharing one `Context` would make a row's
        `ctx.provide` visible to a root that never asked for it. The *composition* is shared:
        `profile` arrives composed, and each root mounts it.
        """
        if root_id in self.roots:
            return self.roots[root_id]
        async with AsyncExitStack() as exits:
            ctx = await exits.enter_async_context(mounted(self.profile))
            session = await self._session_for(ctx, root_id, cwd=cwd)
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
            # The one thing standing between a gated tool call and a person
            # (P5-13). Registered here rather than by a row: the answerers are
            # this *deployment's* front-end channel, not a capability a profile
            # contributes, and they unwind with the root that owns them.
            root.desk = AskDesk(root=root)
            for dispose in root.desk.attach():
                exits.callback(dispose)
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
                # Nothing is built before there is somebody to send it to: this
                # runs once per streamed chunk, so a payload rendered for zero
                # watchers is work thrown away per event.
                if not root.subscribers:
                    return
                payload: dict[str, Any] = {
                    "sessionId": root.id,
                    # `thaw=False`: this payload's only destination is `dumps`,
                    # which handles the frozen forms, and thawing deep-copies the
                    # tree for nobody.
                    "event": event.to_wire(thaw=False),
                }
                # Beside the event, never inside it: a rendered card is derived
                # from the definitions mounted right now, and an event is what
                # the log said. Gated *here* rather than inside `presentation_of`
                # because this runs per appended event — every streamed chunk —
                # and Python evaluates `ctx.get("tools")` before the function
                # that would have rejected the event anyway.
                if event.type in CARD_EVENTS:
                    view = presentation_of(ctx.get("tools"), source, event)
                    if view is not None:
                        payload["presentation"] = view
                root.publish("session.event", payload)

            def announce(agent_: Any, status: str) -> None:
                # Guarded like `relay` above, and for the same reason: reading
                # the footer folds every registered status field over the log,
                # and doing that for nobody is the work this check exists to
                # skip.
                if agent_ is agent and root.subscribers:
                    root.publish(
                        "session.status",
                        {
                            "sessionId": root.id,
                            "status": status,
                            # Beside the status because they change together and
                            # for the same reason: every reading is a fold of
                            # this log, so the moment worth re-reading them is
                            # the moment the agent moved. A client polling them
                            # on its own clock would ask constantly and learn
                            # nothing between turns.
                            "readings": readings_of(root),
                        },
                    )

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

        The lock in `start` orders racers *inside* this process; this stops a second
        daemon appending to a log this one is writing. Two writers on one JSONL put `seq`
        backwards mid-file, which breaks A1 and makes every fold double-count.

        **Daemon against daemon, and no further.** A `ph -p --session x` run against a
        session a daemon holds still opens it: the hazard belongs to
        `JsonlSessionStore`, and leasing there would cover every mode at once. I-5 names
        the second daemon and that is what this gates.

        Taken beside the log rather than at a path of its own construction — the store
        owns where sessions live, and a lease derived independently is one `PH_HOME`
        change away from guarding a file nobody writes.

        `timeout=0`: "somebody else holds it" is a refusal, not something to wait out,
        and blocking here would stall the event loop for every *other* root.

        Through `ctx.effect`, the repo's one mechanism for an acquired external artifact,
        so cleanup is structural rather than remembered (§4.9, I2).

        **`thread_local=False` is load-bearing, and its absence is silent.** filelock
        keeps its re-entrancy counter in a thread-local by default, so a lease acquired
        on a worker thread and released from the event loop finds a counter of zero and
        returns *having released nothing* — no error, no warning, and a lock file held
        until the process dies. The lease belongs to the process, not to whichever thread
        took it.
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

    async def _session_for(self, ctx: Context, root_id: str, *, cwd: str | None = None) -> Session:
        """The root's session — resumed from disk when there is one to resume.

        **Creating unconditionally corrupted the log.** `sessions.create` mints a fresh
        session and the JSONL store appends, so restarting a daemon with the same root id
        concatenated a second session onto the first: one file, one header, and `seq`
        restarting at zero halfway through.

        Resuming is also what connects this to F6 — a root that died holding a worktree
        gets its `workspace/acquired` reconciled on the way back up, because
        `session/created` fires for an adopted session too.

        The resume is announced, not silent: whoever started this daemon may not know a
        previous run crashed. The durable record is the `session/resumed` event, so a
        cron job leaves the fact in the trace whether or not anyone reads stderr.
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
        # `cwd` reaches the *header*, which is storage metadata beside the log
        # rather than an event in it — so it describes where this conversation
        # happened without becoming something the model reads or a replay has to
        # re-apply. `SessionHeader` validates that it is absolute, so a relative
        # path is refused here rather than resolved against the daemon's own
        # working directory, which is not the client's and never was.
        fresh: Session = ctx.sessions.create(root_id, meta={"cwd": cwd} if cwd else None)
        return fresh

    async def _run(self, root: Root) -> None:
        """The root's own task: drive the agent whenever its inbox has work.

        **Nothing here refers to a client**, so nothing a client does — connecting,
        disconnecting, dying — appears in this loop at all.

        A wake arriving while the agent is already running is dropped on purpose: `run()`
        drains the inbox until it is empty, so the turn in flight picks the new message
        up, and calling it twice raises.

        Crashes and their ladder belong to `_drive`, so nothing raising out of a root can
        cancel the supervisor's task group and take every *other* root down with it.
        """
        async with root.waiting:
            async for _ in root.waiting:
                if root.agent.status != "idle":
                    continue
                await self._drive(root)

    async def _drive(self, root: Root) -> None:
        """One wake, and the ladder if the root's task crashes (P5-04).

        **One root's crash is not the daemon's.** This runs in the supervisor's task
        group, so anything raising out of here would cancel the group and take every
        other root with it. `run()` contains its own turn failures, so what reaches this
        boundary is the unanticipated kind: a flush that cannot write, a disposed
        context, a bug.

        Those are worth retrying because the work is still in the inbox — the crash
        happened *around* the turn rather than inside it, which a turn failure is not:
        that one already claimed its message, so running again would produce an empty
        turn reporting false success. See `recovery`.

        The delay is spent before the retry, not after the failure, so a root that gives
        up does so immediately rather than sleeping first.
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

    def _schedule_seam(self, root: Root) -> Any:
        """This root's schedule seam, or a refusal naming why there is none.

        The read `_live_schedules` does quietly — a `None` seam means "no
        schedules", which is the right answer for a *tick* — is the wrong answer
        for a person who just asked to create one: they would get a silent
        success and a schedule that never fires.
        """
        seam = root.ctx.get("schedule")
        if seam is None:
            raise ScheduleUnavailable(f'root "{root.id}" has no schedule seam mounted')
        return seam

    async def schedule(self, root_id: str, entry: Schedule) -> Schedule:
        """Record a schedule on a root, bringing it up if it is not running.

        Through `start`, like `prompt` and `session/attach`, and for a sharper
        reason than either: `tick` only fires schedules on *mounted* roots, so
        scheduling against a passivated one and leaving it passivated would
        write a schedule that can never come due. A root with a live schedule is
        also one `passivatable` refuses to release, so bringing it up is what
        keeps it up.

        Flushed before returning, for `tick`'s reason: a schedule that exists
        only in a buffer is one a daemon restart silently forgets, and the whole
        point of recording it is that nobody has to remember.
        """
        root = await self.start(root_id)
        created: Schedule = self._schedule_seam(root).create(root.session, entry)
        await self._flush(root)
        return created

    async def unschedule(self, root_id: str, schedule_id: str) -> bool:
        """Cancel a schedule. `False` when this log never knew that id."""
        root = await self.start(root_id)
        cancelled: bool = self._schedule_seam(root).cancel(root.session, schedule_id)
        if cancelled:
            await self._flush(root)
        return cancelled

    def scheduled(self, root: Root) -> list[dict[str, Any]]:
        """What is still going to fire on this root, and when.

        One stamp for the whole listing, so two schedules read in the same call
        agree about when "now" was — the same rule `tick` states for a pass.
        """
        stamp = now_ms()
        return [state_to_wire(state, now=stamp) for state in self._live_schedules(root)]

    def _live_schedules(self, root: Root) -> list[Any]:
        """This root's schedules that could still fire, or an empty list.

        The seam lookup and its `None` guard written once: three callers wanted
        it — the tick, the heartbeat and the passivation predicate — and each
        had its own copy of the name, the guard and the read.
        """
        schedule = root.ctx.get("schedule")
        return [] if schedule is None else list(schedule.live(root.session))

    async def rehydrate(self, *, now: int | None = None) -> list[str]:
        """Mount the unmounted roots that have an appointment due (P6-23).

        **The gap this closes is silence.** A schedule outlives its process and
        `tick` iterates `self.roots`, which a boot has none of — so the log kept
        the appointment and nothing kept the log, and the only symptom was a run
        that did not happen.

        **An index, not a scan, and the reasons are three separate failures.**
        Reading every stored log to find which hold a schedule is 500 reads
        before the first connection is answered. Mounting them all is a profile,
        a workspace and possibly a kernel each, which is the cost P5-05 exists to
        release. And `start` takes P5-03's **lease**, so waking everything claims
        every session on the machine and the next `ph -p` over any of them is
        refused — loud, immediate, and hitting sessions with no schedule at all.
        So only what is due is mounted, and every other session stays unleased.

        **No second delivery path.** This mounts; the ordinary `tick` fires. That
        is what makes a missed window behave as P5-06 already settled it — `claim`
        coalesces to the most recent due moment and records it write-ahead — and
        it is why a machine off from Tuesday to Thursday runs Wednesday's work
        once rather than twice or never.

        Failures are per root and logged: a session the index names but that will
        not mount costs its own appointment, not the pass.
        """
        index = self._index()
        if index is None:
            return []
        stamp = now if now is not None else now_ms()
        woken: list[str] = []
        for entry in sorted(index.read().values(), key=lambda one: one.next_at):
            if entry.session_id in self.roots or entry.next_at > stamp:
                continue
            if self.wake_within is not None and stamp - entry.updated > self.wake_within * 1000:
                log.info(
                    "ph_app.daemon: not waking %s; its appointment was last confirmed %d s ago",
                    entry.session_id,
                    (stamp - entry.updated) // 1000,
                )
                continue
            try:
                await self.start(entry.session_id)
            except Exception:
                log.warning(
                    "ph_app.daemon: could not wake %s for its schedule",
                    entry.session_id,
                    exc_info=True,
                )
                continue
            log.info(
                "ph_app.daemon: woke %s for a schedule due at %d",
                entry.session_id,
                entry.next_at,
            )
            woken.append(entry.session_id)
        return woken

    def _index(self) -> ScheduleIndex | None:
        """The index this daemon reads, or `None` when nothing indexes.

        Built from the same `$PH_HOME` the seam writes under, rather than asked
        of a mounted root: at boot there are none, which is the whole situation.
        """
        try:
            return ScheduleIndex(resolve_roots().home)
        except Exception:
            log.warning("ph_app.daemon: no schedule index; nothing will be woken", exc_info=True)
            return None

    async def wake_and_tick(self, *, now: int | None = None) -> list[str]:
        """One scheduler pass: mount what is due, then fire what is mounted.

        The two are separate methods because they answer to different owners —
        `rehydrate` reads an index this daemon does not write, `tick` drives roots
        it does — and one cadence because a root woken for an appointment that is
        not then fired in the same pass waits a whole interval to be noticed.
        """
        await self.rehydrate(now=now)
        return await self.tick(now=now)

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
            # The whole per-root body, not just the claim: `claim` barely raises
            # (a bad expression logs and declines) where `prompt` and `_flush` are
            # the two calls that genuinely fail. One bad root must not take the
            # whole pass down.
            try:
                claimed = schedule.claim(root.session, now=stamp)
                for entry in claimed:
                    log.info("ph_app.daemon: root %s firing schedule %s", root.id, entry.id)
                    await self.prompt(root.id, entry.prompt)
                    fired.append(entry.id)
                # Only when something was appended: adding `or schedule.live(...)`
                # folds the whole log a second time to decide whether to flush a
                # buffer the first fold has just left empty, every five seconds.
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

    async def announce_unreachable(self, note: dict[str, Any]) -> None:
        """Write the "nobody can reach me" record into every root, and flush it.

        **Flushed rather than left in the buffer**, which is not the usual bar here:
        every other record survives a crash because the log is written on the way out,
        and this one is written precisely when the way out has stopped being reliable. If
        the person's answer to an unreachable daemon is `kill`, an unflushed record never
        explains anything to anyone.

        Every root, not the busy ones: what became unreachable is the daemon, and a
        transcript that stops without a word is the same puzzle either way.
        """
        for root in list(self.roots.values()):
            root.unreachable(note)
            await self._flush(root)

    async def _release(self, root: Root) -> None:
        """Flush a root's log, then unwind everything it took.

        One spelling, because there are two callers — `aclose` at shutdown and
        `passivate` when a root goes quiet — and this pair has diverged twice: an earlier
        `aclose` skipped `exits.aclose()` when the flush raised, leaving an unwritable
        root's context undisposed. Whatever joins root teardown next — a lease, a
        reclaim, a cache eviction — has one place to be added.
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

        A retry against a half-mutated tree would be a different turn from the one that
        failed — the model would see edits from an attempt nobody kept, and a ladder that
        compounds its own damage is worse than no ladder. `workspace/checkpoint` is
        P4-09's record and already a fold, so this asks the log rather than remembering.

        Best-effort by construction: an advisory-tier root has nothing to restore, and a
        failed restore must not cost the retry. Either way the attempt goes ahead against
        the tree as it stands, and `agent/retry` says `restored: false` so the transcript
        does not imply a rollback that did not happen.
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
        if not root.once(command):
            return root
        root.agent.followup(user_text(text))
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

        Every condition is a reason a root is still *wanted*, and each is read from
        something that already exists rather than from a flag set beside it:

        * **it is doing something** — `status`, which covers `retrying`: a root in
          P5-04's backoff is idle between attempts, and releasing it there would
          passivate a root mid-ladder;
        * **somebody is watching** — `subscribers`, the root's own attachment set, so a
          client that attached and never detached keeps its session alive by the same
          fact that makes it receive events;
        * **it has live children** — folded from `subagent/*`, because a parent released
          while a child is still running would be rehydrated by the child's own events
          arriving at a root that no longer exists;
        * **it has work scheduled** (P5-06) — a root with a live schedule has already
          said when it comes back;
        * **it has been quiet long enough**, from the log.

        A heartbeat is deliberately *not* a condition: it is a record the scheduler
        leaves so an operator can tell "waiting for Wednesday" from "died on Tuesday",
        not a claim on the root's life. What keeps a scheduled root mounted is the
        schedule.
        """
        # `waiting` joins `idle`, which is the whole reason that status exists.
        if root.status not in ("idle", "waiting"):
            return False
        if root.subscribers:
            return False
        # **The quiet check before the fold**, which is not merely tidier. The
        # root reaching this line is idle and unwatched — exactly the steady
        # state the sweeper exists for — so a fold above it runs on every sweep
        # of the whole ninety-minute window and is discarded eighty-nine times
        # out of ninety.
        if root.idle_for(now) < after * 1000:
            return False
        # Through the seam's cached fold rather than the bare function:
        # `SessionFoldCache` keys on `session.seq`, and an idle root's log does
        # not grow, so every sweep after the first is a dict hit instead of a
        # whole-log walk. That is what saves the root this returns `False` for —
        # idle, unwatched, one unsettled child — which would otherwise re-fold
        # every sixty seconds for the life of the daemon.
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

        **Rehydration is already written**: `start()` resumes any root whose log exists
        (P5-01), so waking one is the ordinary path rather than a second mechanism.

        Order matters, and is `aclose`'s order: record, flush, then unwind. Unwinding
        disposes the mounted `Context` and with it the P5-03 lease, so a passivated
        session is one another process may legitimately open — the point rather than an
        oversight. The wake channel closes first so the root's task leaves its own loop
        instead of being cancelled mid-turn.
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
