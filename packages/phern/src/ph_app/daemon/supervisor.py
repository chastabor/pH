"""The supervisor: roots that outlive the clients that watch them (P5-01, I-2).

Every mode before this one ties the life of an agent to a connection. `phern -p`
exits when the turn ends. The root of the TUI dies with the terminal.
`--mode rpc` lives as long as stdin. That is right for a one-shot run. It is
wrong for the subject of Phase 5: a run that takes an hour, that a person checks
from two machines, and that must not stop because a laptop lid closed.

**One `anyio` task for each root, and the client is not that task.** A root owns
a mounted profile, a session, an agent and a queue. Its task drains that queue
whether or not a client is attached. To attach is to subscribe a connection to
the events of the root, and to detach is to unsubscribe it. Neither starts or
stops the work. That is the complete gate of the row: *TUI close leaves the root
running*.

**A root is addressed by id, never by connection.** A client says which root it
means. Two clients can therefore watch one root, and one client can watch two. A
reconnect is an attach and not a resume, because the root never stopped.

The parts that the row started without now exist:

- Capabilities, cursors and the command journal belong to the server
  (`server.py`, P5-02).
- The session lease that holds the log of each root refuses a second daemon
  (P5-03).
- The crash ladder belongs to `recovery.py`, and this module applies it to the
  task of each root (P5-04).
- This module releases a quiet root (`passivate`, P5-05).
- This module also brings a root back after a restart (`rehydrate`,
  `resume_owed`).

`NON_GUARANTEES` below says what a daemon still does not promise.

@module ph_app.daemon.supervisor
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from ph.agent.types import AgentDriver
from ph.cancel import reason_of
from ph.cordis import Context, LoaderError, Profile
from ph.json import as_obj, as_seq, as_str
from ph.keys import (
    AGENTS,
    COMMANDS,
    COMPACTION,
    GOALS,
    INVARIANTS,
    MOUNT,
    SCHEDULE,
    SESSION_PERSISTENCE,
    SESSIONS,
    SUBAGENTS,
    TOOLS,
    WORKSPACE,
)
from ph.llm.types import AttachmentRef, PluginSource, create_user_message
from ph.paths import resolve_roots
from ph.persistence import (
    JsonlArchive,
    cut_short_note,
    cut_turn_owner,
    resumption_of,
    unwritten_at_unwind,
)
from ph.persistence.protocol import SessionArchive, SessionPersistence
from ph.seams.credentials import hold_for_credential, waiting_for
from ph.seams.invariants import Violation
from ph.seams.live_index import (
    Hold,
    LiveEntry,
    LiveIndex,
    LiveIndexWriter,
    ordered_holds,
    rebuild_live_index,
)
from ph.seams.models import (
    ModelChoice,
    ModelChoiceError,
    ModelEntry,
    ModelList,
    choose,
    move_to,
    start_on,
)
from ph.seams.schedule import (
    CANCELED,
    CREATED,
    REBUILD_TYPES,
    TICK,
    Schedule,
    ScheduleService,
    ScheduleState,
    fire_due,
    rebuild_index,
    scheduled_source,
    state_to_wire,
)
from ph.seams.schedule_index import Appointment, ScheduleIndex
from ph.seams.shell import ShellService
from ph.seams.subagents import CHILD_EVENT_TYPES, DELETED, STATUS, child_is_live
from ph.seams.workspace import Workspace, latest_checkpoint, workspace_of
from ph.session import (
    IntentNotDurable,
    Session,
    SessionEvent,
    now_ms,
    session_written,
)
from ph.session.kinds import SESSION_HOLDER
from ph.session.writers import log_writer
from ph.session_profile import (
    OverrideSource,
    ProfileBase,
    ProfileChange,
    logged_environment,
    record_adopted,
    record_declined,
)
from ph.text import count_of
from ph.tools.errors import error_message

from ..attach import Tray, prompt_message
from ..payloads import (
    ChildRow,
    CompactionFacts,
    OwedListing,
    OwedRoot,
    ProfileAsk,
    ProfileAskReply,
    ResumedRoot,
    ResumeReport,
    RootDescription,
    RootDetail,
    SessionChildrenNotice,
    SessionCommandsNotice,
    SessionEventNotice,
    SessionNotice,
    SessionScreensNotice,
    SessionStagedNotice,
    SessionStatusNotice,
    ShellReply,
)
from ..profiles import (
    composed_like,
    kept_note,
)
from ..protocol import Refusal, cursor_of
from ..runtime import mount_session, open_root, read_start
from ..sessions import RecordedStart, not_a_root
from ..shell import run_shell
from ..wire import stated
from .cards import CARD_EVENTS, presentation_of
from .frontend import AskDesk, FrontEnd
from .planner import Planner
from .profile_command import profile_command
from .projections import commands_of, family_of, family_rows, readings_of, screens_of
from .recovery import (
    CHILD_RETRY_LIMIT,
    FAILED,
    LADDER,
    PASSIVATE_AFTER,
    PASSIVATED,
    RECOVERED,
    RETRY,
    ROOT_RESUME_LIMIT,
    UNREACHABLE,
    VERIFY_AFTER,
    VIOLATED,
    WAKE_RETRY_DELAYS,
    WAKE_WITHIN,
    Recovery,
    floored,
    interrupted_resumes,
    recovery_of,
    resume_children,
)

_LOG = log_writer(__name__)

__all__ = [
    "NON_GUARANTEES",
    "NotARoot",
    "Root",
    "RootStartAbandoned",
    "ScheduleUnavailable",
    "Supervisor",
]


class RootStartAbandoned(Refusal):
    """The daemon stopped before it finished the mount of this root.

    This refusal has its own code for the same reason that `ScheduleUnavailable`
    has one. "The process went away under your request" is a different statement
    from "the session is held elsewhere" or "that profile has no such seam". A
    client that cannot tell them apart cannot decide whether a retry is sensible.
    Here a retry is sensible, against the next daemon: the mount unwound after
    the stop, so nothing still holds the session.

    Only a shutdown can raise it. A mount is a task of the *supervisor*, so the
    only thing that ends it early is the end of the task group. A caller that
    still waits at that moment gets this refusal, and not an empty answer.
    """

    code = "root_start_abandoned"


class NotARoot(Refusal):
    """This id names the log of a sub-agent. Only the mount of its root writes that
    log (P11-08).

    This refusal has its own code for the same reason that `RootStartAbandoned`
    has one: the next action of the client is specific. A retry never helps. The
    root that this refusal names does help: attach to that root, and its mount
    readmits the child. A second root on the file of the child would be a second
    writer on a log that is the only record of the child.
    """

    code = "not_a_root"


@dataclass(slots=True, eq=False)
class _Mounting:
    """One mount in flight, and the answer that it will leave.

    **This record makes two callers share one mount.** `start` no longer holds a
    lock across the build. The build moved onto the task group of the supervisor
    so that a caller can leave without ending the build. The lock used to
    serialize the check-then-mount sequence, and the `_mounting` table now does
    that. The first caller registers a record. Every later caller finds the
    record and waits on the same event. `start` awaits nothing between its lookup
    and the registration, so no second caller can come between them.
    """

    done: anyio.Event = field(default_factory=anyio.Event)
    root: Root | None = None
    error: Exception | None = None
    """What the mount raised. `start` raises it again in every caller that waits
    on the mount. A cancellation is not stored. It leaves through the task group,
    and the waiters get `RootStartAbandoned` from the empty record."""


class ScheduleUnavailable(Refusal):
    """The profile of this root did not mount the `schedule` seam (P5-06).

    This refusal has its own type for the same reason that `SessionBusy` has one.
    (`SessionBusy` is in `ph.persistence.lease`, where the I-5 refusal moved with
    its `session_already_active` code.) "There is nothing to schedule against
    here" is a fact about the profile, and a client can act on it: mount the row,
    or stop asking. With a generic failure, the client would have to guess
    whether the schedule was rejected or the id was wrong.
    """

    code = "schedule_unavailable"


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
        "not rolling, and not this daemon's to do. Stopping the daemon stops every root, "
        "and nothing here starts the next one: an init system or `phern service` does "
        "(`contrib/`). The next daemon resumes owed roots when asked — `phern agents "
        "resume --all`, `phern daemon --resume-owed`, `phern service start` — and "
        "otherwise one from its log when a client asks for it, or when a schedule of its "
        "own comes due",
    ),
    (
        "while the daemon is down",
        "nothing fires. A schedule is kept only while `phern daemon` runs; on start it "
        "catches up, one run per missed window. For work that must happen whether or "
        "not the daemon is up, start it from cron, anacron or a systemd timer — pH "
        "schedules inside a conversation and does not replace them",
    ),
    (
        "an auto-started daemon",
        "leaves when nothing needs it. One a UI spawned exits on the last detach unless a "
        "turn is running, a schedule is on the books, or a `--keep-alive` window was asked "
        "for — so an indexed appointment keeps it resident and a schedule still fires. What "
        "it does not survive is the process ending: a `kill -9`, a logout reap or a reboot "
        "takes the daemon and the appointment with it until a UI opens again, which is the "
        "row above seen from the other side. Timely work on a machine that reboots wants a "
        "systemd or launchd unit owning `phern daemon`",
    ),
    (
        "`--keep-alive`",
        "a floor on how long the process stays, not a ceiling. It is how long a daemon waits "
        "*after the last client leaves* before going; a turn, a schedule or another client "
        "that outruns it keeps it up, which is the point. It is also the client's argv and "
        "not a daemon-side setting — a daemon may not read a front end's preferences, and "
        "after P5-14 may not share a filesystem with it",
    ),
    (
        "what `!!` puts in the log",
        "whatever the command printed. The environment is scrubbed — `ctx.shell` drops every "
        "name matching KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL, and pH reads provider keys "
        "through `ctx.credentials` rather than the child's environment — so `!!env` prints no "
        "harness secret. A command that reads one from somewhere pH does not know about, a file "
        "or a keychain or a fetched token, logs it like any other output: `!!` is the person's "
        "own shell with the person's own reach (P7-10)",
    ),
    (
        "how long `!!` may run",
        "without limit. `ctx.subprocess` bounds what a child may *print* (P7-13, 8 MiB per "
        "stream by default) and `timeout_ms` bounds how long it may run — but `!!` passes "
        "none, so a command that hangs holds its root until somebody cancels it. A model's "
        "`bash` call can set one; a person's `!!` cannot, yet",
    ),
    (
        "who may run `!!`",
        "anyone holding the web token, deliberately — the same authority a terminal already "
        "grants, where a person can approve any tool call the model makes",
    ),
    (
        "a question a person walked away from",
        "re-posed only while this daemon runs. `AskDesk` holds an open ask in memory "
        "and puts it to whoever attaches next; across a restart it is not re-asked, "
        "because the turn that was waiting is gone (P7-09). The log is not left "
        "half-written, though: repair settles an unanswered `question/asked` on "
        "resume, so no question nobody can answer is left open, and the transcript "
        "says it was interrupted rather than declined",
    ),
    (
        "an outside effect after a crash",
        "unknown unless the tool can say. A call the daemon died during reads as outcome "
        "unknown after a resume, and the model is told to check before retrying — unless "
        "its tool names its effect (`idempotency_key`, so a repeat is answered from the "
        "log) or can check it (`reconcile`, which `write` and `agent_message_send` do), "
        "asked of a Code Mode dispatch as of a top-level call. A tool with neither — "
        "every MCP tool today — keeps the unknown (P10-12, P10-13, L6b)",
    ),
    (
        "credentials across a restart",
        "not kept. A credential given over the wire (`credentials/store`) lives in this "
        "daemon's memory and is gone after a restart; a root or child whose route names it "
        "is held — `needs-credential`, no retry spent — and waits for it by name until it is "
        "supplied again (T5). A credential a *tool* needs, such as an MCP token, is not "
        "checked: the log records which tools were called, not which credential each would ask for",
    ),
    (
        "facts across two logs",
        "eventual, not atomic. Each log is written on its own schedule, so the order "
        "is what a crash can keep. A message reaches its receiver's log on disk before "
        "its sender is told (L6), and on resume the send is checked against that log: "
        "found, it reads as done, and missing from every log it could have reached, as "
        "not done (L6b). Only a root's send to a sibling root not yet resumed can stay "
        "unknown. A child's ending is on its own disk before its parent is handed the "
        "result (F1), so a crash can leave a child that ended under a parent that never "
        "heard, and never a parent holding an answer from a child that reads as "
        "unfinished. Within one log a `Session.batch()` lands whole or not at all "
        "(P10-14, P10-15); across two logs there is no batch",
    ),
    (
        "per user",
        "one daemon per $PH_RUNTIME — a second on the same socket is refused rather "
        "than merged (P5-01), and a second writer on one session log is refused by its "
        "lease (I-5). Isolation *between users* is the operator's layer",
    ),
)
"""What this supervisor does **not** promise, stated in the module that would
imply the promise (N5).

Rule 6 says: *state what is not enforced, next to where it would be assumed*. A
caveat that is only in the docs is a defect. The assumption here is what "one
daemon, many long-running agents" appears to mean. Each row names one place where
it does not mean that.

These are data and not prose. `phern doctor` and `phern agents doctor` print them
verbatim. The sentences that a person reads are therefore the same strings that
this file must keep true. A paragraph in a docstring cannot be printed, and
nobody checks a sentence that nobody can print.
"""


WAKE_SLOTS = 8
"""How many wakes the channel holds before the task of a root acts on them.

The number is small on purpose. A wake carries no payload, so a dropped wake
costs nothing as long as *some* wake reaches an idle task. The message itself is
already in the inbox of the agent and on disk."""

Subscriber = Callable[[str, dict[str, Any]], None]
"""`Subscriber` takes `(event_name, payload)`. It is how a connection hears about a
root that it watches."""


def _nothing() -> None:
    """The default for a callback that nobody set. The callbacks are:

    - `Supervisor.recheck_lifetime`, on a supervisor that no server watches
    - `Root.recheck`, on a root that no supervisor started
    - `Audience.emptied`, on an audience that no supervisor made
    """


@dataclass(slots=True, eq=False)
class Audience:
    """Who is attached to one session id, and what they share.

    **An audience is kept by id, not by root.** A connection attaches to an id,
    and a root is one mount of that id. A root that is released and mounted
    again, or restarted on a new profile, is a new object under the same id.
    These fields used to be on the root, and each such change was then a
    hand-off that somebody had to write and keep complete. A restart copied the
    watchers, the tray, the count of the desk and its front ends one at a time. A
    detach during the restart had to be pointed at the root that was being
    replaced. A field added here was one more field to remember there. Now the
    supervisor holds the audience by id (`Supervisor.audiences`). A root reads
    the audience of its id, and a restart has nothing to carry.

    **An audience lives while somebody is attached or its id has a root**,
    mounted or about to be. `Supervisor._unattended` is the only door that it
    leaves by. The supervisor calls it when the last watcher goes (`emptied`),
    when a root is unmounted, and when a mount settles, whether the mount
    succeeded or failed.
    """

    subscribers: set[Subscriber] = field(default_factory=set)
    """The attached connections. This is a set of bound methods, and bound methods
    compare by `(__self__, __func__)`. Attach and detach are therefore symmetric,
    with no token table to keep in step."""
    front_ends: set[FrontEnd] = field(default_factory=set)
    """The attached connections that can answer for a person (`AskDesk`)."""
    staged: Tray = field(default_factory=Tray)
    """The attachments that wait for the next prompt: **the tray of the composer,
    shared.**

    The tray is not in the log. The private-composer rule draws the same
    distinction: text that nobody has sent is not an act in the session, and
    neither is a file that nobody has sent. The tray is *shared*, and not kept
    for each client, for this reason: the person who dropped a PNG on a browser
    tab and the person who watches a terminal look at one conversation. If only
    one of them can see a chip, the other cannot reason about the composer.

    Rule 6, stated plainly: **the tray does not survive the daemon.** A root
    that is released when it goes quiet, or a daemon that exits, loses the staged
    attachments. The *blob* stays in the store, because `attachment/put` wrote
    it. The cost of the loss is therefore to drop the file again, not to find it
    again, and nothing durable is hidden here.
    """
    asked: int = 0
    """How many asks have been named under this id. The next is `ask-<asked + 1>`.

    `AskDesk` names each ask itself, because nothing that the seams pass to it is
    unique. The `refine` approvals of the Continual Harness have no call id,
    which leaves only the tool name. The `ask_id` of a question is whatever its
    caller passed. Two asks under one name used to share an entry in the `asks`
    of the desk. The first ask to finish removed the entry of the other, and the
    root was reported as idle while it still waited for a person.

    A name is unique within this id only, on purpose: a root is self-contained
    and coordinates with no other root. Every ask frame and `ask.settled` carries
    the `sessionId` of the root, so a front end that is attached to two roots
    files an ask by both values (`AskKey`, the `key` of each frame). The count is
    here and not on the desk, because a desk belongs to one mount. A root that
    started again and counted from 1 could meet a front end that still files an
    ask under that name."""
    emptied: Callable[[], None] = _nothing
    """Report that the last watcher left. That can let the root be released, and
    the audience with it. `Supervisor` sets this callback when it makes the
    audience."""

    def drop(self, subscriber: Subscriber) -> None:
        """Stop sending notices to `subscriber`, and call `emptied` when nobody is
        left. This is the only way that a watcher leaves. It covers a detach and a
        watcher that could not be told."""
        self.subscribers.discard(subscriber)
        if not self.subscribers:
            self.emptied()

    def publish(self, notice: SessionNotice) -> None:
        """Tell every watcher. A subscriber that fails is dropped, and the failure
        is not raised.

        A client whose socket died, or a client that cannot keep up, must not
        end the root. This row exists to prevent that inversion. The drop happens
        *here*, where the subscriber list is, so the policy has one owner.

        The notice carries the name that it travels under
        (`SessionNotice.METHOD`), so the method and its payload cannot disagree.
        They used to be two arguments at eight call sites, and nothing checked
        that they matched.
        """
        # The notice is dumped **one time**, above the loop, and every watcher
        # gets the same dict. The hand-built payload did the same before it
        # became a model. Inside the loop, the dump ran again for each
        # subscriber. Measured for one turn at 1 / 3 / 5 watchers, that cost
        # 3.70 / 5.00 / 6.26 ms, against a flat 3.7 ms when hoisted. The cost of
        # a fan-out must not grow linearly with the number of watchers.
        wire = notice.to_wire()
        for subscriber in list(self.subscribers):
            try:
                subscriber(notice.METHOD, wire)
            except Exception:
                log.debug("ph_app.daemon: dropping a watcher that could not be told")
                self.drop(subscriber)


@dataclass(slots=True)
class _FamilyFeed:
    """What `session.children` has told the watchers of a root, and whether another
    push is due to them (P11-07).

    The stream of the root itself no longer carries anything about its children,
    because each child writes only its own log. The supervisor therefore listens
    to every log in the mount of the root, and pushes the family when a row
    would change. **Two filters apply, and both are cheap enough to run for a
    streamed chunk:**

    - Only a record that the state of a child is folded from
      (`CHILD_EVENT_TYPES`) can change a row. Every other record costs one set
      test.
    - The rows are compared with the last rows that were sent. A change that no
      row shows sends nothing.
    """

    sent: list[ChildRow] | None = None
    """The rows last pushed, or `None` before the first."""
    due: bool = False
    """A push is scheduled and has not run yet. This flag merges a burst into one
    push. Examples of a burst: the `canceled` record of a batch and its
    tombstone, or eight children that settle in one step."""


@dataclass(slots=True)
class Root:
    """One long-running agent, and the wake channel that its task waits on.

    This is a dataclass and not a closure over the task. A client asks whether
    the root is busy, what its session is, and who watches it. Those facts are
    read from *outside* the task that owns them.
    """

    id: str
    ctx: Context
    session: Session
    agent: AgentDriver
    wake: MemoryObjectSendStream[None]
    """Tells the task of the root that there is something in the inbox.

    This is a *signal*, not a queue. The durable queue is the `Inbox` of the
    agent. The inbox logs `agent/inbox/spliced` before the projection changes,
    so that a prompt survives a crash. A prompt that was held in a private
    memory stream would be lost with no record. That must not happen in the one
    pH process that is built to outlive its clients.
    """
    waiting: MemoryObjectReceiveStream[None]
    exits: AsyncExitStack
    driving: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    """The scope of the task of this root (`Supervisor._run`). `_unmount` cancels
    it after the root is released (D1).

    To close the wake ends only a task that waits for work. A task inside a
    turn, or one that waited for a person, used to continue against a disposed
    mount for the life of the daemon. The scope is canceled **after** the
    release, so the log on disk is the log that the release left. A turn that a
    shutdown cut therefore stays open for the resumer of the next daemon, and
    nothing that the task writes as it ends reaches a log."""
    stopped: anyio.Event = field(default_factory=anyio.Event)
    """Set when this root's task has ended, which `_unmount` waits for."""
    recovery: Recovery = field(default_factory=lambda: Recovery(attempts=0, failed=False))
    """The state of the retry ladder (P5-04). It is folded from the log when this
    root starts, and `retry`, `give_up` and `recovered` maintain it after that.

    The state is held and not folded again at each read. `status` is read for
    every root on every `sessions/list`, and a scan of the whole log for each
    read would run on the event loop of the daemon. The commands that a root
    already ran are now in the index of the journal (`CLIENT_COMMAND`), which is
    cached and extended, not scanned."""
    audience: Audience = field(default_factory=Audience)
    """Who is attached to the id of this root, and what they share. The audience
    belongs to the id, and the supervisor gives it to each mount of that id
    (`Supervisor.audiences`). A root that is built by hand has its own
    audience."""
    desk: AskDesk | None = None
    """The `AskDesk` of this root: who is asked when a turn needs a person (P5-13).

    The desk is on the root and not on a connection, because the *question*
    outlives any client. When nobody is attached, the ask waits. The log holds
    the ask in both cases."""
    change: ProfileChange | None = None
    """How the named profile of this root had moved when the root started,
    compared with the version that the root runs on (S6). `None` when it had not
    moved. The root runs on its own version until a version is adopted.
    `describe` reports how many settings differ."""
    withdrew: str | None = None
    """What the start of this root withdrew, or `None` when it withdrew nothing.
    It is an adopted version that the loader refused, in the words of
    `withdrawn_note` (`StartingProfile.withdrawn`). The command that asked for
    the start reports it (`_Connection._after_command`)."""
    restart_wanted: bool = False
    """Set by a command that moved the environment of this root to where a live
    mount cannot follow: `/profile use`, or a `/profile clear` of an override
    that turned a row on or off (S7). The `after` step of the `session/command`
    mutation (`_Connection._after_command`) consumes it. That step starts the
    root again from its log after the command and its key are durable, or after
    the command failed. The restart does not run inside the command, because the
    records of the command are still being written to the session of this
    root."""
    held_on_profile: bool = False
    """Whether this root waits for a person to decide about `change` (S6). That is
    the case when the user's own profile moved and the root was started for a
    front end that can be asked. The root is held as it is for a missing
    credential: nothing is driven, and a prompt waits in the inbox. `status`
    says `needs-profile-decision`."""
    family: _FamilyFeed = field(default_factory=_FamilyFeed)
    """What `session.children` last told this root's watchers (P11-07)."""
    unverified_at: int | None = None
    """When something in the mount of this root was last made durable since its
    invariants were checked, in epoch ms. `None` when nothing has been (P12-04).

    The value is set from `session/durable`. ph-core emits that event after
    every flush that wrote something, for every session in the mount: the
    session of the root and the sessions of its children, whichever path wrote
    (a turn, a command, a credential, a schedule). A cache drifts only when
    something writes, and every writer that has finished has flushed. This is
    therefore the one place that sees every write. `verify_root` clears the
    value."""
    recheck: Callable[[], None] = _nothing
    """Report that the status of this root may have moved. Both the release of the
    root and the life of the daemon depend on the status. `Supervisor._start`
    points this callback at `Supervisor._recheck`.

    It is called on every status transition of the agent. It is also called for
    the changes that the status of the agent does not announce:

    - The desk opens or closes an ask. (A root that waits for a person reports
      `waiting`.)
    - The ladder ends, as `failed` or as `recovered` back to `idle`.

    Without this callback, each of those changes waited until some unrelated
    event caused a check. A sixty-second sweep used to cover that gap."""

    @property
    def held(self) -> bool:
        """Whether this root waits for a person before anything runs. The root waits
        for a credential that its route names (T5), or for a decision about its
        moved profile (S6). While the root is held, its task drives nothing and
        its children are not readmitted under it (P13-06). Whatever lifts the
        hold calls `ring`."""
        return self.needs_credential is not None or self.held_on_profile

    @property
    def needs_credential(self) -> str | None:
        """The credential that the route of this root names and that its deployment
        cannot supply, or `None` (T5). While it is set, the root is held. Its
        task does not drive the agent, so a prompt waits in the inbox and does
        not fail at the adapter edge. `status` says `needs-credential`.

        **The value is read from the log**, like `status`, and is not kept in a
        field beside the log. The hold is a `credential/needed` record. The check
        writes it at start and on every `credentials/store`, and the cached index
        of the journal answers from it.
        """
        return waiting_for(self.ctx, self.session).get(SESSION_HOLDER)

    def ring(self) -> None:
        """Send a wake: tell the task of the root that there may be work in the
        inbox.

        A full channel means that the task has wakes pending and has not reached
        them yet, so it will drain this work too. The inbox is the queue, and
        this is only the signal. **A closed channel means that the root has been
        released** (D6). The caller, for example a listener during shutdown or
        the arrival of a credential, then signaled nobody, and the message is in
        the log for the next mount.
        """
        with suppress(anyio.WouldBlock, anyio.ClosedResourceError, anyio.BrokenResourceError):
            self.wake.send_nowait(None)

    @property
    def subscribers(self) -> set[Subscriber]:
        """The connections attached to this root's id (`Audience.subscribers`)."""
        return self.audience.subscribers

    @property
    def staged(self) -> Tray:
        """The tray of the composer, which the watchers of this root share
        (`Audience.staged`)."""
        return self.audience.staged

    def subscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.add(subscriber)

    def publish(self, notice: SessionNotice) -> None:
        """Tell every watcher of this root's id (`Audience.publish`)."""
        self.audience.publish(notice)

    @property
    def status(self) -> str:
        """What the *agent* reports, and not a copy that is kept beside it.

        `agent.status` is a property over the phase of the driver, and every
        transition emits `agent/status`. A second field, set around the `await`
        of this class, would be true only at that call boundary. A steer, a
        cancel or an inbox wake between turns would change one and not the
        other.
        """
        # A root that waits for a person takes priority over what the agent
        # reports. The agent reports `running` while an approval is open, which
        # is true and not useful. See
        # `test_a_root_parked_on_a_person_may_be_released`.
        # A root that is held on its profile also waits for a person, and the
        # ask that it waits on belongs to the desk. This answer therefore comes
        # first, because it is the one that a person can act on.
        if self.held_on_profile:
            return "needs-profile-decision"
        if self.desk is not None and self.desk.waiting:
            return "waiting"
        live = str(self.agent.status)
        # `retrying` is also derived here, and not only published. When it was
        # announced as a notification while `sessions/list` still said "idle",
        # a client that attached during the backoff got two answers to one
        # question. This property derives its answer and does not copy it, to
        # avoid that drift.
        # The agent is authoritative whenever it has work in hand. "idle" is
        # ambiguous only *between* turns: a root that exhausted the ladder is
        # idle in the same way as a root that waits for a prompt. P5-01 made
        # this property derive and not copy, so that the answer could not
        # drift. The second reading is therefore derived too, from the ladder
        # state that is folded from the log (`recovery`), and not from a flag
        # that is set beside the `await` of this class.
        if live != "idle":
            return live
        if self.needs_credential is not None:
            return "needs-credential"
        if self.recovery.failed:
            return "failed"
        if self.recovery.attempts:
            return "retrying"
        # **Work in the inbox is work in hand**, before the driver has taken it.
        # `prompt` logs the message, a wake is sent, and `prompt` returns. The
        # task of the root starts the turn at its next checkpoint. Between the
        # two, the phase of the driver still says idle. A client that attached
        # in that window (`phern agents attach --until-idle` directly after a
        # `send`) was told that the work was done before it began, and stopped
        # without it. This answer is derived from the inbox, which is itself
        # folded from the log, and not from a flag that is set here.
        if self.agent.inbox.has_pending:
            return "running"
        return live

    @property
    def last_turn(self) -> str | None:
        """How the most recent turn ended, as the agent reported it.

        `status` is deliberately the status of the agent, and it stays `idle`
        after a turn that ended in error. The ladder does not count turn
        failures, and a second status invented here would be the opinion of the
        supervisor about the work of the agent. But a client that polled
        `sessions/list` could not tell a root that answered from a root whose
        last answer was an error, and `--until-idle` treated both as done. The
        `turn/end` of the agent is therefore reported beside the status:
        `completed`, `error`, `aborted`, `blocked`, `max-tokens` or
        `interrupted`. It is read from the log and not kept in a field (P5-04).

        The read goes through `as_obj` (`ph.json`), which holds the rule for
        frozen and plain payloads. The session freezes payloads to
        `mappingproxy`. A reader that tests for `dict` therefore works live and
        silently sees nothing on resume. The fold of P5-04 itself got that
        guard wrong. The two `turn/end` readers of the TUI already go through
        `as_obj`. A fourth way to write the read here would be the copy that
        somebody misses.
        """
        event = self.session.latest("turn/end")
        kind = as_obj(event.data.get("reason")).get("kind") if event is not None else None
        return str(kind) if kind else None

    def retry(self, *, reason: str) -> None:
        """Record that a failed turn is run again (P5-04).

        The record is written *before* the attempt, not after it. A daemon that
        died during the retry must come back with the knowledge that the attempt
        was made. Otherwise it resumes with a shorter ladder than it had really
        spent. A10 applies the same write-ahead order to blobs, and
        `CLIENT_COMMAND` applies it to commands.

        **The record is also written before the restore** (S12), which is part
        of the attempt. The restore used to run first. A daemon that died during
        the restore had then rewritten the tree with no record of the attempt,
        and a crash loop there never advanced the count. The seam records the
        restore as its own pair (`WORKSPACE_RESTORE`).
        """
        _LOG.append(
            self.session,
            RETRY,
            {
                "attempt": self.recovery.attempts + 1,
                "of": self.recovery.total,
                "delayMs": int(self.recovery.delay * 1000),
                "reason": reason,
            },
        )
        self.recovery = replace(self.recovery, attempts=self.recovery.attempts + 1)
        self.publish(SessionStatusNotice(session_id=self.id, status="retrying"))

    def recovered(self) -> None:
        """A retry worked, so the ladder is cleared, and the log records that.

        The recovery is recorded and not only forgotten, because to forget is
        not durable. Suppose a root recovered and the daemon then restarted.
        Without the record, the root would fold its old `supervisor/retry`
        records back and resume with a ladder that it had already left. This
        record is also the *only* thing that resets the count. A retry that
        fails therefore cannot clear its own limit.
        """
        _LOG.append(self.session, RECOVERED, {"afterAttempts": self.recovery.attempts})
        self.recovery = Recovery(attempts=0, failed=False)
        self.publish(SessionStatusNotice(session_id=self.id, status=self.status))
        # The status goes from `retrying` to `idle` with no agent transition:
        # the turn that worked already reported idle while the ladder still said
        # retrying.
        self.recheck()

    def idle_for(self, now: int) -> int:
        """The milliseconds since anything happened in this session (P5-05).

        The value is measured from the last event of the log. It therefore means
        "nothing has happened" and not "no client called us". It survives a
        restart, and it cannot drift from what the transcript shows.

        **A log with no events is measured from the `created_at` of the
        header.** That is still the log, and it is a clock that runs. When "no
        last event" was read as zero, an *unused* session lasted forever. A
        `session/new` followed by a client that vanished left a root that
        nothing could ever release. The root held a mounted profile for the life
        of the daemon and, under P7-08, kept the process alive with it.
        """
        return max(0, now - self.quiet_since)

    @property
    def quiet_since(self) -> int:
        """When this session last did anything, in epoch ms. It is the time of the
        last event, or the `created_at` of the header for a log with no events
        (`idle_for` says why).

        `Supervisor.next_release` plans a release from this value: the window
        ends at this value plus `passivate_after`."""
        last = self.session.last_event
        return self.session.header.created_at if last is None else int(last.time)

    def passivated(self, idle_ms: int) -> None:
        """Record in the log that this root is about to be released, before the
        release.

        The record is write-ahead, like every other record here. The append must
        reach the buffer before the flush that carries it. Otherwise the last
        thing in the log is whatever the root did ninety minutes ago, and the
        pause looks like a crash.
        """
        _LOG.append(self.session, PASSIVATED, {"idleMs": idle_ms})
        self.publish(SessionStatusNotice(session_id=self.id, status="passivated"))

    def unreachable(self, note: dict[str, Any]) -> None:
        """Record that the supervisor lost the socket that it was bound to (P5-11).

        This method appends and does nothing else. The other records here,
        except `violated`, append *and* publish. `relay` observes the feed of
        this session, so the `session.event` frame goes out on the same append,
        and a client that attached before the path went away is still on a live
        stream. A second `publish` would send one fact twice under two names.

        **This is not a status change.** The root does exactly what it did
        before. The socket is what broke. To report the root as `failed` would
        start the recovery ladder because of a socket problem.
        """
        _LOG.append(self.session, UNREACHABLE, note)

    def violated(self, note: dict[str, Any]) -> None:
        """Record that a pollable invariant did not hold for this root (I6).

        This method appends and does nothing else, for the same reason as
        `unreachable`: `relay` observes the feed of this session, so the frame
        goes out on the same append.

        **This is not a status change**, also for the same reason as
        `unreachable`, and the reason is stronger here. A stale projection says
        that the bookkeeping of the root is wrong, not that its work is wrong.
        If the record started the recovery ladder, the ladder would restart a
        root because of a cache, and lose the turn in flight *and* the evidence.
        """
        _LOG.append(self.session, VIOLATED, note)

    def give_up(self, reason: str, *, attempts: int) -> None:
        """Record that the ladder is spent, and tell the watchers.

        The record goes in the log first. An unattended run must leave exactly
        this fact behind: that a root stopped working. An agent that cron
        started reports it in its own trace, whether or not a client was ever
        attached. `Root.status` then reports `failed` from `recovery`, which
        this method updates and which the next start folds from the log.
        """
        _LOG.append(self.session, FAILED, {"attempts": attempts, "reason": reason})
        self.recovery = replace(self.recovery, failed=True)
        self.publish(SessionStatusNotice(session_id=self.id, status="failed"))
        log.error("ph_app.daemon: root %s failed after %d attempts — %s", self.id, attempts, reason)
        self.recheck()

    def describe(self) -> RootDescription:
        """What a client is told about this root.

        Each fact has one name. `rootId` and `sessionId` used to be the same
        string (a root *is* its session here), and `events` was
        `cursor.sequence`. A client then had to choose which of two names was
        authoritative.
        """
        options = self.agent.options
        compaction = self.ctx.get(COMPACTION)
        return RootDescription(
            session_id=self.session.id,
            status=self.status,
            last_turn=self.last_turn,
            watchers=len(self.subscribers),
            cursor=cursor_of(self.session),
            # The route that this root is on. Without it, a client could not
            # know the route until the first turn, because the only other
            # statement of it is `request/context`, which is appended when a
            # request is *built*. A front end that has just attached must draw
            # a footer now. "no model" over a socket, where the in-process
            # front end said `fake-1`, was the complete content of a snapshot
            # diff that nobody could read.
            #
            # The options of the agent and the profile spell "none" as `""`.
            # `stated` turns each into `None` for the wire.
            provider=stated(options.provider),
            model=stated(options.model),
            model_key=stated(options.model_key),
            profile=stated(self.ctx.require(MOUNT).profile.name),
            profile_changes=len(self.change.differences) if self.change is not None else 0,
            # `None` while the root is held: the question that its modal asks
            # is the account, and the "runs on the version it started on" of
            # the note is not yet true of either version.
            profile_note=(
                kept_note(self.change, self.id, door="session")
                if self.change is not None and not self.held_on_profile
                else None
            ),
            # The share at which this root compacts on its own, as its seam
            # answers it. A front end draws its context gauge in the warning
            # color from this share. `ph_app` may not import the bundle that
            # owns the engine, so it asks the seam and copies no default.
            compaction=CompactionFacts(
                trigger_fraction=compaction.trigger_fraction() if compaction is not None else None
            ),
        )

    def status_notice(self) -> SessionStatusNotice:
        """The complete `session.status` for this root: the facts of its description
        and its footer, as attach seeds them. It is for a watcher that is told
        that the root changed under it."""
        facts = self.describe().facts()
        return SessionStatusNotice(
            session_id=self.id,
            **facts.model_copy(update={"readings": readings_of(self)}).model_dump(),
        )

    def detail(self) -> RootDetail:
        """`describe`, plus what a client that asks about *one* root needs.

        The detail is assembled here and not at the wire edge, where the two
        recovery fields used to be read. When the client-facing projection of a
        root is split across two files, `sessions/list` and `session/status`
        come to use different names for the same object. `status` already
        reduces the ladder to `"retrying"` or `"failed"`. The ladder fields
        behind that answer are here, so a new field is added in one place.
        """
        return RootDetail(
            **self.describe().model_dump(),
            attempts=self.recovery.attempts,
            failed=self.recovery.failed,
        )


def _recorded_violations(session: Session) -> set[str]:
    """The ids of the invariants that the log of this root already records as
    broken.

    `verify_root` records a *change* against this set. The set is read from the
    log and not remembered. A root that was passivated overnight and woken in
    the morning therefore compares against its own transcript, and not against
    an empty memory. An empty memory would announce again a condition that is
    already on the record.

    **The set holds ids and deliberately not the details.** A detail is the
    sentence that a check builds, and `Session.stale()` builds it from *counts*:
    "derive_messages holds 3 message(s) where a fresh derivation gives 4". Those
    numbers change each time the session grows. A root that drifts and is still
    busy is the only interesting case. A comparison of details would make such a
    root differ from its own last record on every check, and append again each
    time. This comparison exists to prevent exactly that flood, and a key on the
    detail would bring the flood back in the one case that matters.

    The invariant is what broke. The detail is how the break looked at one
    moment. The record carries the detail, and the id decides whether a record
    is written.

    The set is empty for a root that has never violated an invariant *and* for a
    root whose last record said that the invariants hold again. Those are the
    same fact: the record names nothing as broken now.
    """
    event = session.latest(VIOLATED)
    if event is None:
        return set()
    return {as_str(as_obj(one).get("invariant")) for one in as_seq(event.data.get("violations"))}


QUIET: tuple[str, ...] = (
    "idle",
    "waiting",
    "failed",
    "needs-credential",
    "needs-profile-decision",
)
"""The statuses that are not work in hand. `_held` (for `passivatable`), `busy` and
`_verify_at` read this tuple.

`waiting` is here beside `idle`, and that is the purpose of the `waiting`
status. A root that waits for a person who has closed the terminal reports
`running` from the agent, which is true and not useful. `retrying` is
deliberately absent: a root in the backoff of P5-04 is between attempts, not
finished.

**`failed` is here because it is the one status that can never change by
itself** (E3). `give_up` is the end of the recovery ladder: no further attempt is
scheduled, and nothing in the process will move that root again. When `failed`
was absent from this tuple, it counted as work in hand forever, and `busy()` was
permanently true. An ephemeral daemon is started for a run and must exit when
the work is done. Such a daemon then stayed up for the life of the machine
because of a root that had already given up. This tuple asks whether a person
waits for the root, and nobody waits for that one.

**`needs-credential` is here for the same reason as `waiting`** (T5). The root
cannot move until a person supplies a key. Its hold is in its log, so a release
loses nothing, and the next start asks again. **`needs-profile-decision` is here
for that reason too** (S6): a root that is released before anybody answered is
asked again at its next start.

The tuple has a name so that it is not written out in each reader. The readers
answer *different* questions from the same rule, among them "may this root be
released" and "may this process end". A status that was added to one copy and
not to another would change one answer and not the other.
"""


def _say_what_waits(root: Root) -> None:
    """Log one line for each missing credential, with the sessions that are held
    on it (T5).

    The line is logged when a root comes back. After a restart, that is the
    moment when a reader of the daemon log needs it: from outside, a child that
    is held for a key looks exactly like a child that is only slow.
    """
    for name, holders in _waiting_on([root]):
        log.warning(
            "ph_app.daemon: credential %s is missing; held until it is supplied: %s",
            name,
            holders,
        )


def _say_profile_moved(root: Root) -> None:
    """Log one line when a root starts and its named profile has moved since the
    root began (S6). The root is then held for a person, or it keeps its
    version. The daemon log must give that account in both cases."""
    change = root.change
    if change is None:
        return
    if root.held_on_profile:
        log.warning(
            "ph_app.daemon: root %s is held: %s has changed since it started (%s); "
            "waiting for a person to decide",
            root.id,
            change.now.name,
            count_of(len(change.differences), "setting"),
        )
        return
    log.info("ph_app.daemon: %s", kept_note(change, root.id))


def _waiting_on(roots: Iterable[Root]) -> list[tuple[str, str]]:
    """`(credential, "root, root/run, root/run/run, …")` for every hold that these
    roots and the sub-agents beneath them record (T5).

    **Each hold is in the log of the session that waits** (Phase 11). The hold
    for the route of a root is in the log of the root, and the hold of a child
    is in the log of the child. Both are recorded as `SESSION_HOLDER`, which is
    what `ChildState.awaiting` folds. The log of a root therefore answers for
    the root alone, and every level of its family is asked too: a grandchild
    that is readmitted beneath a readmitted child is held for its key in the
    same way. Each holder is named by the run ids from the root down, which is
    how a parent addresses each child.
    """
    waiting: dict[str, list[str]] = {}
    for root in roots:
        name = root.needs_credential
        if name is not None:
            waiting.setdefault(name, []).append(root.id)
        where = {root.id: root.id}
        for state in family_of(root):
            where[state.session_id] = f"{where.get(state.parent_id, root.id)}/{state.run_id}"
            if state.awaiting is not None:
                waiting.setdefault(state.awaiting, []).append(where[state.session_id])
    return [(name, ", ".join(holders)) for name, holders in sorted(waiting.items())]


def _working_beneath(root: Root) -> bool:
    """Whether a sub-agent beneath `root` is still working. A working sub-agent
    keeps the root mounted.

    The answer is read from the log of each child (Phase 11): `child_is_live`
    over the whole family, stored members included. Stored members count
    because the resume sweep has decided each of them by the time that a
    release can ask. A member was readmitted, was ended in its own log, or is
    held for a credential. A held member waits, and the root must not be
    released under it. The sweep has also revoked what an ended child left
    unfinished beneath it. Nothing that is still live is therefore an orphan,
    and every level counts in the same way.
    """
    return any(child_is_live(state) for state in family_of(root))


def model_for(ctx: Context, choice: ModelChoice) -> ModelEntry:
    """`choose`, but a choice that names nothing runnable raises the `Refusal` that
    a client is sent."""
    try:
        return choose(ctx, choice)
    except ModelChoiceError as error:
        raise Refusal(str(error)) from error


@dataclass(slots=True)
class Supervisor:
    """Every root this daemon is running, and the task group they live in."""

    profile: Profile
    tasks: TaskGroup
    choice: ModelChoice = field(default_factory=ModelChoice)
    """What `phern daemon --provider/--model` asked every root to run on. An empty
    choice means the profile default of each root. The `ctx.models` of each root
    resolves the choice at mount, so a root runs on what its profile lists, not
    on a string."""
    passivate_after: float | None = PASSIVATE_AFTER
    """The seconds of quiet before a root is released, or `None` to keep every
    root.

    This is a field beside `choice` because it is the same kind of thing:
    configuration for one daemon. Before, it was passed through `serve`, the old
    sweeper task and two method signatures. That wrote one constant in eight
    places, and two of them were positional arguments through `start_soon` that
    nothing checked."""
    check_invariants: bool = True
    """Whether the pollable invariants of a root are checked when it settles (I6,
    P12-04).

    Turn this off for a soak or a benchmark that must not pay for an O(events)
    refold on each turn, and in tests whose subject is something else.
    `verify_root` still answers a caller that asks, whatever the setting."""
    verify_after: float = VERIFY_AFTER
    """The seconds that a root must be settled, with nothing written in its mount,
    before its check runs. `recovery.VERIFY_AFTER` explains why there is a
    wait."""
    wake_within: float | None = WAKE_WITHIN
    """How stale an indexed appointment can be and still wake its root (P6-23).

    `None`, the default, catches up on whatever was missed, however long the
    daemon was down. That is the complete promise of the scheduler. A deployment
    that does not want to bring back a long-abandoned session sets a limit
    here."""
    wake_retry_delays: tuple[float, ...] = WAKE_RETRY_DELAYS
    """The delays before the scheduler tries again after a pass that could not
    mount a session for its appointment. Each consecutive failing pass takes the
    next step, and the last step repeats. `recovery.WAKE_RETRY_DELAYS` explains
    why there is a retry."""
    continue_interrupted: bool = False
    """Whether a resume continues a turn of a *person* that the restart cut short,
    as it continues a turn of a goal or of a schedule (P13-04,
    `phern daemon --continue-interrupted`).

    The default is off. A person whose prompt was interrupted is at a keyboard
    somewhere and sends the prompt again. A turn that runs for that person
    without a request is a turn that the person did not ask for. Turn it on for
    a deployment where nobody is at a keyboard: a fleet of roots that are driven
    only over the wire, where every prompt comes from a client."""
    roots: dict[str, Root] = field(default_factory=dict)
    audiences: dict[str, Audience] = field(default_factory=dict)
    """Who is attached to each session id, and what they share (`Audience`). The
    table is keyed by id, so every mount of an id reads the same audience.
    `_unattended` removes an entry."""
    recheck_lifetime: Callable[[], None] = _nothing
    """Report that what holds the *daemon* may have moved.
    `DaemonServer.__post_init__` points this at `DaemonServer.check_lifetime`
    (P9-06, P9-07).

    This is a callback, and the supervisor does not know its server, because
    the dependency runs in one direction only. The server holds a supervisor,
    and a supervisor that held a server back would make a cycle. The default
    does nothing, so a supervisor that a test builds, or anything else that
    builds one without a `DaemonServer`, has nothing to set.

    **The name states the question, not the cause**, because more than one
    thing raises the question:

    - A turn starts or ends. The socket cannot see that event. Its listener is
      registered in `_start` beside the other `ctx.on` listeners, and is not
      hidden inside one of them.
    - A schedule is made, withdrawn or fired. `_watch_schedules` hears that from
      the log, whichever path wrote it.
    - The status changes in a way that the agent does not announce.
      `Root.recheck` reports those changes.

    `DaemonServer.holds` lists the terms and their events (§5 rule 6)."""
    _starting: anyio.Lock = field(default_factory=anyio.Lock)
    _mounting: dict[str, _Mounting] = field(default_factory=dict)
    """The mounts in flight, by the root id that they were asked for. This table
    makes two callers of one id share one mount. See `_Mounting` and `start`."""
    _in_flight: set[_Mounting] = field(default_factory=set)
    """Every mount in flight. That includes the mount of a new root, which has no
    id to be found by until it opens. `aclose` waits on this set."""
    _closing: bool = False
    """Whether `aclose` has begun. It is set before `aclose` waits, so the set
    that `aclose` waits on cannot grow after that."""
    _schedules: ScheduleIndex | None = None
    """The appointment index, built on first use. See `_index`."""
    _live: LiveIndexWriter | None = field(init=False, default=None)
    """The writer of the live-roots index (P13-01), or `None` when `$PH_HOME` did
    not resolve. That failure is logged one time, at construction, and not at
    every transition."""
    _live_guessed: bool = False
    """Whether the live-roots index was last rebuilt from a guessed directory, with
    no mounted store to vouch for it. In that case, an `owed` call that does not
    ask for a rebuild reads the file and does not read every log again, until a
    store mounts or a rebuild is asked for. `_tried` applies the same rule to
    the schedule index."""
    unwritten: dict[str, str] = field(default_factory=dict)
    """The logs that could not be written at the last attempt (P13-08), by session
    id, each with its root. A log here is the log of the root or of a child,
    because a root resumes into whatever its family lost. Two sources add a log:

    - A write through any path fails, including a barrier before a model
      request. The store reports it (`session/unwritten`, heard in `_start`).
    - The last write of the mount fails at the unwind, and the mount keeps the
      failure (`_unmount`).

    A log leaves when a later write of it succeeds. `daemon/status` lists the
    roots, and readiness is no while there is one. An orchestrator must stop
    sending work to a daemon that cannot write, and it must learn that plainly,
    not from a warning in a log file."""
    scheduler: Planner = field(init=False)
    """The scheduler (P5-06, P6-23). It runs a pass that mounts what is due and
    fires what is mounted (`wake_and_tick`), and then sleeps until the next
    appointment (`next_wake`).

    The scheduler sleeps and does not poll. With nothing scheduled, it sleeps
    until it is told: to make a schedule is the opt-in (docs/seams/schedule.md).
    The first pass runs at once, so a boot wakes what came due while the daemon
    was down. A wake by the clock is a due wake, and it may retry a rebuild that
    failed (`_rebuild`).

    Two things tell the scheduler to plan again: `notice_schedules` (the log
    shows a schedule made or canceled), and the mount of a root, which brings
    its store and its own appointments. A session that did not mount for its
    appointment is retried on a bounded backoff (`_wake_retry`)."""
    verifier: Planner = field(init=False)
    """The invariant check (I6, P12-04). It runs a pass that checks every root that
    is settled and has had nothing written for `verify_after` (`_verify_quiet`),
    and then sleeps until the next root will be due (`next_verify`).

    Two things tell the verifier to plan again. One is `session/durable` from
    any session in a mount, which marks the root of that mount as unverified.
    The other is `_recheck`, because a root that was written to while it worked
    becomes due only when it settles. The verifier replaced a five-minute poll
    over every live root."""
    releaser: Planner = field(init=False)
    """Root release (P5-05, P12-01). It runs a pass that releases every root that
    has been quiet long enough (`sweep`), and then sleeps until the quiet window
    of the next root ends (`next_release`).

    Everything that can make a root releasable sooner tells the releaser
    (`_recheck`):

    - a turn starts or ends
    - a root mounts
    - the last watcher of a root leaves
    - a root starts to wait for a person, or stops
    - a child settles
    - a schedule ends
    - the ladder ends, as failed or as recovered

    Each has a test in `test_daemon.py` that changes only that one thing."""
    _occasions: int = 0
    """How many times a schedule changed. Each change, like each clock wake of the
    scheduler (`scheduler.rang`), is a reason why an index rebuild that failed
    may now succeed (`_rebuild`)."""
    _tried: tuple[bool, int, int] | None = None
    """The inputs of the last rebuild that left the index in doubt: whether a store
    was available, `_occasions` and `scheduler.rang`. A rebuild with the same
    inputs is not tried again."""
    _last_pass: int = 0
    """When the last scheduler pass began. Anything that was due by then was
    attempted, so it does not hold the next sleep at zero (`next_wake`)."""
    _wake_failures: int = 0
    """How many scheduler passes in a row left an appointment behind because its
    session did not mount. It selects the step of `wake_retry_delays` that the
    next retry takes."""
    _wake_retry: int | None = None
    """When the next pass retries a wake that failed, in epoch ms, or `None` while
    no wake has failed. `next_wake` plans it beside the appointments."""
    _last_verify: int = 0
    """When the last invariant pass began. The next pass begins no sooner than
    `PASS_FLOOR` after it (`next_verify`)."""
    _last_release: int = 0
    """When the last release pass began. Its only use is the floor for the next
    pass (`next_release`).

    The "attempted by then" rule of the scheduler does not apply here. A root
    that was held at that pass, for example by a child that still worked, had no
    chance to be released. When the hold ends, the quiet window of the root has
    usually ended long before. If the plan left that root out, the root would
    stay mounted."""

    def __post_init__(self) -> None:
        self.scheduler = Planner("the scheduler pass", run=self.wake_and_tick, plan=self.next_wake)
        self.releaser = Planner(
            "the release pass", run=self.sweep, plan=lambda _now: self.next_release()
        )
        self.verifier = Planner(
            "the invariant check", run=self._verify_quiet, plan=lambda _now: self.next_verify()
        )
        try:
            self._live = LiveIndexWriter(LiveIndex(resolve_roots().home), self.tasks)
        except Exception:
            log.warning(
                "ph_app.daemon: no live-roots index; nothing records what is owed", exc_info=True
            )

    async def start(
        self,
        root_id: str | None,
        *,
        cwd: str | None = None,
        choice: ModelChoice = ModelChoice(),
        profile: str | None = None,
        asks: bool = False,
    ) -> Root:
        """Return this root, and mount it if it is not mounted. For `None`, return a
        new root, which is named as it opens (I-5).

        **The mount is the work of the supervisor, not of the caller.** It runs
        as a task of `self.tasks`, and this method only waits for it.
        `Peer.serve` cancels every handler in flight at the moment that the
        socket of its client reaches EOF, and a handler can be anywhere inside
        the build: the rows of the profile, the I-5 claim, the cut of a worktree,
        the readmission of children. When the build ran inline, a person who
        closed the terminal during a mount canceled the build *and* the teardown
        of what the build had already taken. An `AsyncExitStack` that unwinds
        under a scope that is already canceled stops at its first await. The
        lease therefore stayed held, and every later `start` of that id answered
        `session_already_active` until the daemon was restarted.

        **The supervisor owns the work and does not shield it**, and so the fix
        costs nothing elsewhere. A shield needs a bound, because a shield with
        no deadline is a daemon that will not exit. The bound must then be
        guessed against the slowest legitimate mount, and a shutdown that
        arrives during a mount waits for the whole guess. Here a cancellation
        lands on the *await*. The caller leaves immediately, the daemon finishes
        the root that it started, and a shutdown ends the mount through the task
        group like any other task.

        **`_mounting` serializes the mount for each id**, because the
        check-then-mount sequence spans awaits. Two clients that ask for the
        same new root at the same time both pass the membership test, both
        mount a profile, and both reach for the same session file. That is
        exactly the hazard of I-5. The uniqueness check of the store does not
        prevent it, because each root gets its own `Context` and therefore its
        own `SessionStore`. The lease would catch the pair as a *refusal*,
        which is the wrong answer to their question: two clients that name one
        root want the same root, and only a second *process* is a conflict. The
        second caller therefore waits on the record of the first, and gets the
        root that the first one built.

        There is a fast path that touches none of this, because `prompt` calls
        this method on every turn.

        `choice` is what a fresh root mounts on. It comes from `session/new`,
        and the choice of the daemon applies when it is absent or empty. A root
        that is already mounted keeps its route here, and `set_model` is the
        way to move it. `profile` is the named profile that a *new* session is
        created on (S5), and the `--profile` of the daemon applies when it is
        `None`. A session with a log runs on what its log records, whatever is
        asked here. `asks` says that a front end that can answer for a person
        asked for this start (S6). If the user's own named profile moved since
        the root began, the root is then held for the user to decide. A root
        that is started for a schedule keeps its version.

        **`None` is a session that the daemon names** (protocol 8):
        `open_session` mints its id as it opens. Nobody can ask for that root
        before this method returns, so it takes neither the fast path nor
        `_mounting`. `_in_flight` still holds it for `aclose`.

        **The table holds only complete roots** (D5), so the fast path cannot
        return a root that is still being built: `_start` puts a root in the
        table as its last step. A mount or a restart of this id in flight is
        waited for before the fast path. A caller that arrives as a restart
        begins therefore gets the new root, and not the root that is about to be
        unmounted.
        """
        asked = root_id or "a new root"
        if self._closing:
            # This check comes first, before the fast path (D7). Handlers
            # outlive the start of teardown, and `aclose` would release a root
            # that is returned now while the handler still uses it. A root that
            # is built now is work that nothing would ever release.
            raise RootStartAbandoned(f"the daemon is shutting down and will not mount {asked}")
        # The record is registered and the task is spawned with no await. That
        # lets two callers share one mount: nothing can run between the lookup
        # and the assignment, so the second caller finds the record of the
        # first.
        pending = self._mounting.get(root_id) if root_id is not None else None
        if pending is None and root_id is not None:
            root = self.roots.get(root_id)
            if root is not None:
                return root
        if pending is None:
            pending = _Mounting()
            self._in_flight.add(pending)
            if root_id is not None:
                self._mounting[root_id] = pending
            wanted = self.choice if choice.is_default else choice
            self.tasks.start_soon(self._mount, root_id, cwd, wanted, profile, asks, pending)
        await pending.done.wait()
        if pending.error is not None:
            raise pending.error
        if pending.root is None:
            raise RootStartAbandoned(
                f"the daemon stopped before it finished mounting {asked}; "
                "nothing was left holding its session"
            )
        return pending.root

    async def _mount(
        self,
        root_id: str | None,
        cwd: str | None,
        choice: ModelChoice,
        profile: str | None,
        asks: bool,
        pending: _Mounting,
    ) -> None:
        """Build one root on a task of the supervisor, and publish the answer.

        `_starting` is held *here* and not in `start`. The lock orders a mount
        against `passivate`, and that order must hold even when the caller has
        left.

        Only `Exception` is caught. A cancellation means that the task group is
        ending, and it leaves through this task as it would through any other.
        To store it would pass the `CancelledError` of one task to another task,
        and that produces an error in the bookkeeping of cancel scopes.
        """
        try:
            async with self._starting:
                pending.root = await self._start(
                    root_id, cwd=cwd, choice=choice, profile=profile, asks=asks
                )
        except Exception as error:
            pending.error = error
        finally:
            self._settled(root_id, pending)

    def _settled(self, root_id: str | None, pending: _Mounting) -> None:
        """A mount or a restart is done, whether it succeeded or failed. Its record
        leaves both tables, and its waiters are woken. The record is removed by
        identity, because it is the only record that this call registered."""
        if root_id is not None and self._mounting.get(root_id) is pending:
            del self._mounting[root_id]
            # A mount or a restart that failed leaves no root under the id. Its
            # audience goes too, unless somebody is still attached.
            self._unattended(root_id)
        self._in_flight.discard(pending)
        pending.done.set()

    async def _start(
        self,
        root_id: str | None,
        *,
        cwd: str | None,
        choice: ModelChoice,
        profile: str | None = None,
        asks: bool = False,
    ) -> Root:
        """Mount a profile, create its agent, and give it its own task.

        The mount goes through `runtime.mount_session`, which enters
        `runtime.mounted`, so a mode cannot drift from the profile semantics.
        The `finally` of `mounted` does what a hand-written `Context()` and
        `mount()` pair loses: a mount that raised partway left a live `Context`
        that was never in `self.roots`, and so was never disposed. An
        `AsyncExitStack` holds the mount, because the lifetime of a root is
        longer than any one `async with`.

        The mount is **for one root and is not shared**. Two roots are two
        deployments as far as every seam is concerned, and one shared `Context`
        would make the `ctx.provide` of a row visible to a root that never asked
        for it. **The profile is also for one root** (S5). A root with a log is
        mounted from the environment of that log (`session_profile`), and a new
        root from the profile that it was asked for. Two roots on two profiles
        can therefore share a daemon, and a root that went idle comes back as it
        was. A named profile that moved since the root began is kept, or is put
        to a person (`asks`, S6) before anything runs. `runtime.mount_session`
        withdraws an adopted version that the loader refuses, and the root
        starts on the version that it had. `phern -p` and rpc keep the same
        rule.
        """
        if root_id in self.roots:
            return self.roots[root_id]
        async with AsyncExitStack() as exits:
            # **Where this root works is decided before the mount.** The fs
            # seam fixes its root while it applies, and `workspace-lifecycle`
            # reads the root there to discover the provisioning of the project.
            # A root that was mounted first and rebased afterwards would
            # already have branched its worktree from the wrong tree. The
            # directory of a new session comes from the client. The directory
            # of a resumed session is what its own header recorded. It is read
            # from disk, because there is no store to ask until this mount
            # exists.
            start = await read_start(root_id)
            if root_id is not None and start.owner is not None:
                # This check runs before the mount and the claim, so the refusal
                # leaves the file of the child exactly as its root left it.
                raise NotARoot(not_a_root(root_id, start.owner.root))
            where = cwd or start.cwd
            ctx, starting = await mount_session(
                exits,
                start,
                await anyio.to_thread.run_sync(self._requested, profile),
                project=Path(where) if where else None,
            )
            session = await self._session_for(ctx, start, cwd=cwd)
            # The model of the session itself comes first: the base, and any
            # `/model` that the session logged, which the open restored. Then
            # comes the choice of this start, which is an override where it
            # differs (S4).
            try:
                entry = await start_on(ctx, session, choice, source="cli", command=choice.flags)
            except ModelChoiceError as error:
                raise Refusal(str(error)) from error
            agent = ctx.require(AGENTS).create(session, entry.options())
            wake, waiting = anyio.create_memory_object_stream[None](max_buffer_size=WAKE_SLOTS)
            root = Root(
                id=session.id,
                ctx=ctx,
                session=session,
                agent=agent,
                wake=wake,
                waiting=waiting,
                exits=exits,
                # The audience belongs to the id, not to this mount. Whoever
                # attached to an earlier mount of the id watches this one, with
                # the tray and the ask count that they had (`Audience`).
                audience=self._audience(session.id),
            )
            # No failure between here and the table may leave an audience for a
            # root that never existed (Holdings rule 1). An audience that
            # somebody is attached to stays, and an empty one goes.
            exits.callback(self._unattended, session.id)
            # The desk is the only thing between a gated tool call and a person
            # (P5-13). It is registered here and not by a row. The answerers
            # are the front-end channel of this *deployment*, not a capability
            # that a profile contributes, and they unwind with the root that
            # owns them.
            root.desk = AskDesk(root=root)
            for dispose in root.desk.attach():
                exits.callback(dispose)
            root.recheck = self._recheck

            def woken(agent_: object) -> None:
                """A message that asks this root to run reached its inbox (P13-05).

                The driver reports every delivery that wakes: a prompt, a steer,
                the ending notice of a child, the continuation of a turn that
                was cut short. Here the daemon turns that report into a ring. A
                result that a readmitted child gives to an idle parent is
                therefore run, and is not left pending for the life of the
                daemon. The listener is registered as soon as there is a root to
                ring, so a delivery by this mount itself also rings: the wake
                channel holds the ring until the task starts. `_run` drops a
                ring that the task cannot act on yet, because the root is held
                or is inside a turn. The message is still in the inbox.
                """
                if agent_ is agent:
                    root.ring()

            ctx.on("agent/woken", woken)
            # `/profile` (S7) is registered in the same way, for the same
            # reason: this host, not a row, saves and switches the profile of a
            # session.
            commands = ctx.get(COMMANDS)
            if commands is not None:
                exits.callback(commands.register(profile_command(root)))
            # How far up the retry ladder this session got is read from its own
            # log, one time, here. A root that is resumed partway up the ladder
            # therefore does not restart the count and retry forever. What the
            # root already did is in the `CLIENT_COMMAND` index of the journal,
            # which is folded from the same log.
            root.recovery = recovery_of(session)
            # **A named profile that moved since this root began** (S6) is kept,
            # unless the user's own file moved and a person is here to ask. In
            # that case the root is held, and the person is asked, before
            # anything runs on either version.
            root.change, root.withdrew = starting.change, starting.withdrawn
            root.held_on_profile = (
                asks and starting.change is not None and starting.change.worth_asking
            )
            _say_profile_moved(root)
            # **A turn that the restart cut short is continued, or the root
            # gives up on it** (P13-04), and the repair that closed the turn is
            # written one time (P13-10). The closing records that the resume
            # synthesized are in the buffer, not on disk. When they were left
            # there, a second kill before the first barrier found the same open
            # turn and repaired it again, and a root that ran no turn was
            # repaired at every boot. The flush is here, before any work, so
            # that the record of the crash, with the continuation or the
            # give-up, is on disk before what follows it.
            resumed = resumption_of(session)
            if resumed is not None and resumed.interrupted:
                if resumed.turn is not None:
                    self._continue_interrupted(root)
                await self._flush(root)
            # **Nothing starts without its credentials** (T5). The check runs
            # before the root can be driven and before its children are
            # readmitted. A key that a restart lost therefore holds the work
            # that needs it, by name, and does not fail that work at the first
            # request.
            await self._check_credential(root)
            # What the *children* of this root are owed, now that there is an
            # agent for them to attach to (P5-04). A daemon that stopped
            # between the admission of a child and its first turn left that
            # work described in the log and running nowhere. The queued
            # children are driven again, and the children that were caught
            # inside a turn are settled and not run again. See
            # `resume_children`. Children are not readmitted while the root is
            # held on its profile: a "yes" starts the root again, which would
            # cut them short. **They are also not readmitted while the root is
            # held on a credential** (P13-06). A child that is readmitted under
            # a root that cannot run would spend its ladder on a key that a
            # person could supply in a second, and the arrival of that key
            # readmits them (`credential_supplied`). Meanwhile they stay live
            # in their own logs, which keeps the root mounted and on the
            # live-roots index. The root is not in the table yet (D5). Nothing
            # that a sweep readmits looks the root up by id, and a caller that
            # got the root now would miss the events that its relay,
            # registered below, exists to send.
            if not root.held:
                await self._resume_children(root)
            _say_what_waits(root)

            def relay(source: Session, event: SessionEvent) -> None:
                # Nothing is built until there is somebody to send it to. This
                # runs one time for each streamed chunk, so a payload that is
                # rendered for zero watchers is wasted work on every event.
                if not root.subscribers:
                    return
                # The card travels beside the event, never inside it. A rendered
                # card is derived from the definitions that are mounted now,
                # and an event is what the log said. The gate is *here* and not
                # inside `presentation_of`, because this runs for each appended
                # event, which means every streamed chunk, and Python evaluates
                # `ctx.get(TOOLS)` before it calls the function that would
                # reject the event anyway. The card is resolved *before* the
                # notice is built, so an event with a card costs one model, and
                # not a model plus a `model_copy` of it.
                view = (
                    presentation_of(ctx.get(TOOLS), source, event)
                    if event.type in CARD_EVENTS
                    else None
                )
                root.publish(
                    SessionEventNotice(
                        session_id=root.id,
                        # `thaw=False`: the only destination of this payload is
                        # `dumps`, which handles the frozen forms. A thaw would
                        # deep-copy the tree for no reader.
                        event=event.to_wire(thaw=False),
                        presentation=view,
                    )
                )

            def verbs() -> None:
                """The command set changed, so tell the watchers. See `announce` below."""
                if root.subscribers:
                    root.publish(
                        SessionCommandsNotice(session_id=root.id, commands=commands_of(root))
                    )

            def drew() -> None:
                """The screen set changed, so tell the watchers.

                The seam emits `screens/change` from the `then` of
                `claim_key`, which runs after the entry has left the table. So
                the list that this reads on an unload no longer holds the
                screen. A hook that ran on the scope of the registering row
                would unwind before the removal, read the old table, and
                publish the list unchanged.

                The notice carries the whole list and not the one screen,
                because `ScreenDefinition.build` cannot travel. A client keeps
                only the ids that it can draw, so it reads the projection again
                and does not apply a delta.
                """
                if root.subscribers:
                    root.publish(SessionScreensNotice(session_id=root.id, screens=screens_of(root)))

            def beneath(source: Session, event: SessionEvent) -> None:
                """The log of a child in the mount of this root changed a row: push
                the family (P11-07).

                This is the store-wide feed, filtered. `relay` above is the feed
                of the root itself. The log of a root holds nothing about its
                children (Phase 11), so their state reaches a watcher only from
                their own logs. A mount holds one root and its family, so a log
                with a parent link is a descendant. Only a record that the state
                of a child is folded from can change its row, so a streamed
                chunk costs one set test. The guard is like the guard of
                `relay`: nothing is read when nobody watches, and `_tell_family`
                sends only what the rows show as changed.

                **The release notice comes before that guard.** A child that
                settles is what lets an unwatched root be released, and an
                unwatched root is exactly the root that the guard skips. The
                notice is sent only for a status or a tombstone, the two records
                on which `child_is_live` can become false. The spending records
                of a child arrive with every reply and can hold nothing.
                """
                if event.type not in CHILD_EVENT_TYPES or source.header.delegating_parent is None:
                    return
                if event.type in (STATUS, DELETED):
                    self.releaser.notice()
                if not root.subscribers:
                    return
                if not root.family.due:
                    root.family.due = True
                    self.tasks.start_soon(self._tell_family, root)

            def lifetime(agent_: object, _status: str) -> None:
                """A turn that starts or ends changes what holds the *daemon*.

                This is its own listener and not a line inside `announce`,
                because it is a different subscriber with a different audience.
                `announce` tells the watchers of this root what the root does.
                This listener tells the process whether it is still wanted. The
                most important case has no watchers at all: a detached
                `phern -p`, whose daemon must leave when the work that it was
                started for is finished. The guard of `announce` must therefore
                not cover this listener.

                The end of a turn is also the most common moment at which the
                quiet window of a root starts, so the releaser plans again on
                the same transition.
                """
                if agent_ is agent:
                    root.recheck()

            def announce(agent_: object, status: str) -> None:
                # This is guarded like `relay` above, for the same reason. To
                # read the footer folds every registered status field over the
                # log, and this check exists to skip that work when nobody
                # watches.
                if agent_ is agent and root.subscribers:
                    root.publish(
                        SessionStatusNotice(
                            session_id=root.id,
                            status=status,
                            # The account that the agent gives of the turn that
                            # just ended (P5-04). `idle` alone looks like
                            # success, and this is the moment at which a client
                            # that waits for idle decides.
                            last_turn=root.last_turn,
                            # The readings are beside the status because they
                            # change together, for the same reason. Every
                            # reading is a fold of this log, so the moment to
                            # read them again is the moment when the agent
                            # moved. A client that polled them on its own clock
                            # would ask constantly and learn nothing between
                            # turns.
                            readings=readings_of(root),
                        )
                    )

            # This observes the feed of the session itself, not the store-wide
            # `session/event` bus. The events of a child agent belong to its
            # own transcript. With a subscription here, they are never
            # received, which is better than to receive and discard them.
            exits.callback(session.observe(relay))
            # A schedule is made or canceled on this root, by whichever path.
            # That happens in the log, so the scheduler hears it from the log.
            exits.callback(session.observe(self._watch_schedules))
            # The events of the children are heard on the bus, and only as a
            # row. A watcher is sent `session.children` about them, never their
            # chunks.
            ctx.on("session/event", beneath)

            def written(source: Session, _through: int) -> None:
                """Something in this mount is now durable and has not been checked.

                **This is also the one moment at which the live-roots index must
                be updated** (P13-01). Every hold reaches disk through a flush:
                the first request of a turn, a goal record, an acked prompt, the
                status of a child in its own log, the records of the ladder. The
                index therefore reflects what is durable plus what this process
                knows, sampled where both just changed. The event is heard only
                for a flush that wrote something (`SessionStore.flush`), so a
                hold that moved with no record after it would wait for the next
                flush. A log that was written is also no longer unwritten
                (P13-08)."""
                root.unverified_at = now_ms()
                self.verifier.notice()
                self.unwritten.pop(source.id, None)
                self._note_live(root)

            def could_not_write(source: Session, error: Exception) -> None:
                self._could_not_write(root, source.id, reason_of(error))

            ctx.on("session/durable", written)
            ctx.on("session/unwritten", could_not_write)
            ctx.on("agent/status", lifetime)
            ctx.on("agent/status", announce)
            ctx.on("commands/change", verbs)
            ctx.on("screens/change", drew)
            if root.agent.inbox.has_pending and not root.recovery.failed:
                # This is work that the log holds and that no turn claimed: a
                # prompt that was acked and flushed before the last process
                # died, or one that a passivation left behind partway up the
                # ladder. `Inbox` replays it, and `status` reports it as
                # `running`. Nothing else rings for it. The next prompt or tick
                # would ring, and would then run both as one turn. A `restart`
                # comes through here too. **There is no ring after a spent
                # ladder.** The message that a root gave up on is still in its
                # inbox. A ring here would be one more attempt for each daemon
                # lifetime, and the folded count exists to end exactly that
                # endless retry. That root waits for a person, as `give_up`
                # said. The wake channel holds the ring until the task below
                # starts.
                root.ring()
            asking: anyio.CancelScope | None = None
            if root.held_on_profile:
                # The ask ends with the root. If the root is released first, the
                # ask is made again at the next start, which finds the hold
                # again.
                asking = anyio.CancelScope()
                exits.callback(asking.cancel)
            # The root owns the stack now. The `async with` unwinds a stack
            # that has been emptied. A failure anywhere above therefore
            # disposes everything that it entered, and a success gives the
            # whole stack to the root. This replaces a hand-written
            # `except BaseException: aclose()`, which guarded only the three
            # lines that it wrapped. `agents.create`, the channel and the
            # `session.observe` callback were all outside it, and nobody could
            # see on which side of that boundary a new line was.
            root.exits = exits.pop_all()
            # **The root is complete, so it goes in the table, with its task**
            # (D5, Holdings rule 1). This is one step with nothing awaited,
            # after everything above that could fail. When the root went in
            # earlier, a second caller got a root that was still being built:
            # its children were not yet readmitted, and its relay was not yet
            # registered. A build that then failed left that caller with a
            # root that never ran. `_unmount` is the only door out of the
            # table.
            self.roots[root.id] = root
            self.tasks.start_soon(self._run, root)
            if asking is not None:
                self.tasks.start_soon(self._decide_profile, root, asking)
            # A root brings its store, and a rebuild of an index in doubt can
            # read through that store. It also brings its own quiet window,
            # which may be the first to end.
            self.scheduler.notice()
            self.releaser.notice()
            # The root is recorded in the live-roots index with what it owes, or
            # is not listed at all (P13-01). The next daemon reads that record
            # to learn whether this root is worth mounting.
            self._note_live(root)
            return root

    async def _session_for(
        self, ctx: Context, start: RecordedStart, *, cwd: str | None = None
    ) -> Session:
        """The session of the root. It is claimed, and then resumed from disk when a
        log exists.

        This is one call into `open_session`, through `open_root`, with the
        family in which the start of the root found its log. Every host opens a
        session through `open_session`. The I-5 claim, the decision to resume or
        to create, and the `session/resumed` record are all there. A root
        therefore cannot drift from what `phern -p --session x` does to the same
        file (P5-03). The claim belongs to the store. It is taken on the context
        of this root, so that the unwind of the root (a passivation or a
        shutdown) gives the session back.

        **An unconditional create corrupted the log.** `sessions.create` mints a
        new session, and the store appends. A daemon that was restarted with the
        same root id therefore added a second session to the file of the first:
        one file, one header, and a `seq` that restarted at zero halfway
        through. The resume also connects this method to F6. A root that died
        while it held a worktree gets its `workspace/acquired` reconciled when
        it comes back, because a resumed log awaits `session/loaded` before it
        is handed out.

        The resume is announced. Whoever started this daemon may not know that
        a previous run crashed. The durable record is the `session/resumed`
        event, so a cron job leaves the fact in the trace whether or not anybody
        reads stderr.
        """
        session = await open_root(ctx, start, meta={"cwd": cwd} if cwd else None)
        resumed = resumption_of(session)
        if resumed is not None:
            log.warning(
                "ph_app.daemon: resumed root %s from %s existing events%s",
                session.id,
                resumed.events,
                " — the previous run was interrupted" if resumed.interrupted else "",
            )
        return session

    async def _run(self, root: Root) -> None:
        """The task of the root: drive the agent whenever its inbox has work.

        **Nothing here refers to a client.** A client that connects, disconnects
        or dies therefore does not appear in this loop at all.

        A wake that arrives while the agent is already running is dropped on
        purpose. `run()` drains the inbox until it is empty, so the turn in
        flight takes the new message, and a second call to `run()` raises.

        Crashes and their ladder belong to `_drive`. Nothing that a root raises
        can therefore cancel the task group of the supervisor and end every
        *other* root with it.

        The loop runs within `root.driving`, which the release of the root
        cancels (`_unmount`). To close the wake ends this loop only while it
        waits for work.
        """
        try:
            with root.driving:
                async with root.waiting:
                    async for _ in root.waiting:
                        # The root is held for a credential (T5) or for a profile
                        # decision (S6). The message stays in the inbox, and
                        # whatever lifts the hold rings again. A ring whose
                        # message a running turn already took has nothing to
                        # do: `run()` on an empty inbox opens and closes an
                        # empty turn.
                        agent = root.agent
                        if agent.status != "idle" or not agent.inbox.has_pending or root.held:
                            continue
                        await self._drive(root)
        finally:
            root.stopped.set()

    async def set_model(
        self, root: Root, chosen: ModelEntry, *, source: OverrideSource, command: str
    ) -> None:
        """Run `root` on `chosen` from its next request. This is `/model` (S1,
        item 8).

        The method takes the entry that `model_for` already resolved. A key that
        the profile does not list is therefore refused before the idempotence
        key of the daemon is claimed. **The change is an override of the
        `models` row of the session** (S4, `move_to`). It is recorded before it
        is made, so the next start of the session runs on it too. A route that
        is not listed joins the list of the session under a name of its own.
        The change applies from the next request, because a request in flight
        is already on the wire.

        The credential is checked again, because the new route may name another
        one. Every watcher is told the route now, and not at the next request.
        A footer that kept the old model until then would be wrong for exactly
        as long as somebody reads it.
        """
        entry = await move_to(root.ctx, root.session, chosen, source=source, command=command)
        root.agent.reroute(entry.options())
        await self._check_credential(root)
        if root.subscribers:
            root.publish(
                # The three fields are one fact (`StatusFacts`), and `model`
                # says that this frame states the route. `move_to` names every
                # route that it moves to. The seam spells an unlisted route as
                # a key of `""`, which must not reach the wire as a name.
                SessionStatusNotice(
                    session_id=root.id,
                    provider=entry.route.provider,
                    model=entry.route.model,
                    model_key=stated(entry.key),
                )
            )

    def _requested(self, name: str | None) -> Profile:
        """The profile that a new root starts on: `name`, or the profile of the
        daemon when `name` is `None`. It is composed with the start options of
        the daemon, so a `--patch` given to `phern daemon` reaches every root,
        whatever profile the root was asked for. A root with a log mounts the
        environment of that log over this profile (`session_profile`)."""
        if name is None:
            return self.profile
        try:
            return composed_like(name, self.profile)
        except (LoaderError, OSError, ValueError) as error:
            raise Refusal(f'profile "{name}" does not compose: {error}') from error

    def starts_on(self) -> ModelEntry | None:
        """What a new root runs on. It is read from the composed profile, without a
        mount.

        The result is `None` when nothing resolves: the profile lists no models
        and the daemon was started without a route. `phern daemon` refuses that
        case before it binds. A served daemon is therefore in that case only
        when it is built by hand, as a test builds it.
        """
        try:
            return ModelList.of(self.profile).resolve(self.choice)
        except ModelChoiceError:
            return None

    async def _resume_children(self, root: Root) -> None:
        """Put back to work what this root's children are owed (P5-04)."""
        revived = await resume_children(root.ctx, root.agent)
        if revived:
            log.info(
                "ph_app.daemon: root %s put %d admitted child(ren) back to work",
                root.id,
                len(revived),
            )

    async def _tell_family(self, root: Root) -> None:
        """Push `session.children` to the watchers of this root, one time for each
        burst (P11-07).

        The first change of a burst schedules this task, and the task runs at
        the next checkpoint. Every change that the burst made is therefore in
        the one list that the task reads. A batch writes its `canceled` record
        and its tombstone together, and the ending of a child and the next step
        of its parent are one loop turn apart. The list is read whole, from the
        logs of the children. It is sent only if it differs from the last list
        that was sent, so a change that the rows do not show sends nothing.

        A root that was released while this task waited is told nothing. Its
        watchers were told `passivated`, and its context is gone.
        """
        root.family.due = False
        if not self._holds(root) or not root.subscribers:
            return
        rows = family_rows(root)
        if rows == root.family.sent:
            return
        root.family.sent = rows
        root.publish(SessionChildrenNotice(session_id=root.id, children=rows))

    async def _decide_profile(self, root: Root, asking: anyio.CancelScope) -> None:
        """Put the moved named profile of a root to a person, and act on the answer
        (S6).

        The question goes through the desk of the root, so whoever attaches is
        asked. It runs within `asking`, which the release of the root cancels.
        A root that is released first is asked again at its next start. The
        answers:

        - **Adopt** records the version and starts the root again from its log.
          The start makes the version the base, with the overrides kept, and
          then applies the start options of this daemon.
        - **Keep** records the refusal, so that version is not asked about
          again.
        - **Later** records nothing.

        After keep or later, the root runs on the version that it has.

        This is its own task in the group of the supervisor, so nothing that is
        raised here may escape. After a failure, the root continues on its own
        version, which is the version that it had.
        """
        change = root.change
        if change is None or root.desk is None:
            return
        try:
            with asking:
                decision = await root.desk.decide_profile(change)
            if asking.cancelled_caught:
                return
            if decision == "adopt":
                # The exception is a version that was adopted while this
                # question was open (`adopt`). That version is the one that the
                # user saw and accepted, so it stays.
                adopted = logged_environment(root.session).adopted is not None
                if not adopted and not await record_adopted(root.ctx, root.session, change.now):
                    log.warning("ph_app.daemon: root %s: the adoption did not reach disk", root.id)
                await self.restart(root)
                return
            if decision == "keep":
                await record_declined(root.ctx, root.session, change.now)
        except Exception:
            log.exception("ph_app.daemon: root %s: the profile decision failed", root.id)
        if not self._holds(root):
            return
        root.held_on_profile = False
        if not root.held:
            await self._resume_children(root)
        root.publish(root.status_notice())
        root.ring()

    async def restart(self, root: Root) -> Root | None:
        """Release `root` and start it again from its log, and keep its watchers
        (S6).

        This is what a "yes" to a moved profile does, and what `/profile use`
        does (S7). The new base is in the log, and a base takes effect at a
        start. The root is therefore unmounted as a passivation would unmount
        it, and then mounted again. The record of a passivation is not written,
        because the root did not go quiet. Whoever is attached stays attached.
        The audience belongs to the id (`Audience`), so the new root has its
        watchers, its answerers, its tray and its ask count from the moment
        that it is built. The watchers are told what changed: the route, the
        verbs and the screens are those of the new mount. `_start` rings for a
        prompt that is still in the inbox.

        `_start` withdraws an adopted version that does not mount. The root then
        comes back on the version that it ran on, and its log says why
        (`profile/withdrawn`). The method returns the new root, or `None` when
        the root did not start at all.

        **The restart is a mount, so it is the work of the supervisor** (D2), as
        the mount that `start` makes is. The remount runs on `self.tasks` under
        a `_Mounting` record, and this method only waits for it. When the
        remount ran on the task of the caller, a client that disconnected
        during a restart canceled it between the unmount and the mount. The
        root was then gone from the table, and nothing reported it. A `start`
        of this id that arrives during the restart waits on the same record.
        The record also keeps the audience of the id while no root is under the
        id. A second restart of the same root joins the first.
        """
        pending = self._mounting.get(root.id)
        if pending is None:
            if self._closing or not self._holds(root):
                return None
            pending = _Mounting()
            self._in_flight.add(pending)
            self._mounting[root.id] = pending
            self.tasks.start_soon(self._remount, root, pending)
        await pending.done.wait()
        return pending.root

    async def _remount(self, root: Root, pending: _Mounting) -> None:
        """The work of `restart`, on a task of the supervisor: unmount `root`, mount
        its log again, and tell whoever is attached what the new mount
        changed."""
        try:
            async with self._starting:
                if not self._holds(root):
                    return
                await self._unmount(root, why="restarted from its log")
                try:
                    back = await self._start(root.id, cwd=None, choice=ModelChoice())
                except Exception as error:
                    log.exception(
                        "ph_app.daemon: root %s did not start again on its new profile", root.id
                    )
                    # The failure is published to the audience of the id, which
                    # the unmounted root has left. The watchers stay attached to
                    # the id. Nothing is mounted for them to watch now, and
                    # they will hear whatever is mounted under the id next.
                    audience = self.audiences.get(root.id)
                    if audience is not None:
                        audience.publish(SessionStatusNotice(session_id=root.id, status="failed"))
                    pending.error = error
                    return
                back.publish(back.status_notice())
                back.publish(SessionCommandsNotice(session_id=back.id, commands=commands_of(back)))
                back.publish(SessionScreensNotice(session_id=back.id, screens=screens_of(back)))
                pending.root = back
        finally:
            self._settled(root.id, pending)

    def part(self, root_id: str, watcher: Subscriber, front_end: FrontEnd) -> None:
        """A connection stops watching `root_id`, and stops answering for it.

        The detach is by id, as the attach was. A root may be mounted under the
        id, in a restart, or gone. In each case the connection is in the
        audience. No window therefore exists in which a detach finds nothing
        and the watcher is left behind (D8).
        """
        audience = self.audiences.get(root_id)
        if audience is None:
            return
        audience.front_ends.discard(front_end)
        audience.drop(watcher)

    def _audience(self, root_id: str) -> Audience:
        """The audience of `root_id`, made if there is none. When its last watcher
        leaves, two things follow (`_emptied`). A root that nobody watches may
        be released, so the releaser is told. This is not a `recheck`, because
        watchers are not a term of what holds the daemon. And an audience that
        nothing holds is removed."""
        audience = self.audiences.get(root_id)
        if audience is None:
            audience = self.audiences[root_id] = Audience()
            audience.emptied = partial(self._emptied, root_id)
        return audience

    def _emptied(self, root_id: str) -> None:
        self.releaser.notice()
        self._unattended(root_id)

    def _unattended(self, root_id: str) -> None:
        """Remove the audience of `root_id` when nothing is left of it: nobody is
        attached, no root is under the id, and no mount of the id is in flight.
        This is the only door that an audience leaves by (Holdings rule 2).
        Each thing that could have been the last to hold the audience calls
        it."""
        audience = self.audiences.get(root_id)
        if audience is None or audience.subscribers or audience.front_ends:
            return
        if root_id in self.roots or root_id in self._mounting:
            return
        del self.audiences[root_id]

    def _holds(self, root: Root) -> bool:
        """Whether `root` is the one this supervisor holds under its id."""
        return self.roots.get(root.id) is root

    async def adopt(self, root: Root, version: ProfileBase) -> None:
        """Accept `version` for the next start of `root`. This is
        `phern profiles adopt` (S6).

        A running root is not interrupted (decision 18). It keeps its base, and
        the start after its release makes this version the base, with the
        overrides kept. **This call answers a root that is held on that same
        question.** The pending ask is settled as "adopt". That removes its
        modal from every front end, and starts the root on the version that is
        recorded here, which is the version that the user was shown.
        """
        if not await record_adopted(root.ctx, root.session, version):
            raise Refusal(f"{root.id}: the adoption could not be written to its log")
        if root.held_on_profile and root.desk is not None:
            root.desk.settle(ProfileAsk.METHOD, ProfileAskReply(decision="adopt").to_wire())

    def behind(self) -> list[tuple[str, str]]:
        """`(root, what)` for every mounted root whose named profile moved since
        the root started (S6). These are rows of `phern agents doctor`, beside
        the credentials that are awaited."""
        rows: list[tuple[str, str]] = []
        for root_id, root in sorted(self.roots.items()):
            change = root.change
            if change is None:
                continue
            what = f"{change.now.name}: {count_of(len(change.differences), 'setting')} differ"
            rows.append(
                (root_id, f"{what} — held for a decision" if root.held_on_profile else what)
            )
        return rows

    async def _check_credential(self, root: Root) -> None:
        """Hold `root` for the credential of its own route, or lift that hold, and
        record the result in its log."""
        options = root.agent.options
        await hold_for_credential(
            root.ctx, root.session, SESSION_HOLDER, options.provider, options.model
        )

    async def credential_supplied(self, root: Root) -> None:
        """A credential reached `root`: lift every hold that it ends (T5).

        The hold of the root itself is lifted when the credential that its own
        route names is here now. The root is then rung, because a prompt that
        arrived while it was held still waits in its inbox. The hold of every
        child that waited for a credential that has arrived is lifted too. The
        check reads the log, and not the name that was just stored. One value
        can lift the hold of a root and of its children at the same time, and a
        second check costs nothing when nothing changed.
        """
        held = root.needs_credential is not None
        await self._check_credential(root)
        if held and root.needs_credential is None:
            root.ring()
        subagents = root.ctx.get(SUBAGENTS)
        # The children are not readmitted while the root itself still waits, for
        # another credential or for its profile decision (P13-06). The hold
        # that kept its children out in `_start` keeps them out here, and
        # whatever lifts that hold readmits them.
        if subagents is not None and not root.held:
            revived = await subagents.readmit_waiting(root.agent, retry_limit=CHILD_RETRY_LIMIT)
            if revived:
                log.info(
                    "ph_app.daemon: root %s put %d child(ren) back to work once their "
                    "credential arrived",
                    root.id,
                    len(revived),
                )
        await self._flush(root)

    async def _drive(self, root: Root) -> None:
        """One wake, and the ladder if the task of the root crashes (P5-04).

        **The crash of one root is not the crash of the daemon.** This runs in
        the task group of the supervisor, so anything raised out of here would
        cancel the group and end every other root with it. `run()` contains its
        own turn failures. What reaches this boundary is therefore the kind that
        nobody anticipated: a flush that cannot write, a disposed context, a
        bug.

        Those failures are worth a retry, because the work is still in the
        inbox. The crash happened *around* the turn and not inside it. A turn
        failure is different: the turn already claimed its message, so a second
        run would produce an empty turn that reports false success. See
        `recovery.py`.

        The delay is spent before the retry, not after the failure. A root that
        gives up therefore does so at once, and does not sleep first.

        **The loop runs only while the root is in the table** (D1, `_holds`). A
        root that was released under its turn has nothing left to retry. Its
        release wrote what there was and unwound what the root held. A ladder
        that was climbed for such a root wrote `supervisor/retry` into a log
        that was already given back. It also told `retrying` to watchers that a
        restart had already given to the new root.
        """
        while self._holds(root):
            try:
                await root.agent.run()
                if not self._holds(root):
                    return
                await root.ctx.require(SESSIONS).flush(root.session)
                if root.recovery.attempts or root.recovery.failed:
                    # The record is written only after a ladder was climbed, so
                    # an ordinary turn writes nothing. This record clears the
                    # count, and it must be a record that only success can
                    # write. `failed` with no attempts is the give-up of the
                    # resume ladder (P13-04). A prompt from a person that then
                    # ran clears it in the same way.
                    root.recovered()
                    await self._flush(root)
                return
            except (anyio.get_cancelled_exc_class(), anyio.ClosedResourceError):
                # This is teardown, not failure. A give-up recorded here would
                # write to a session that is being disposed. It would also
                # report as failed a root that was only asked to stop.
                raise
            except Exception as error:
                if not self._holds(root):
                    log.debug(
                        "ph_app.daemon: root %s was released under a turn that then failed: %s",
                        root.id,
                        error_message(error),
                    )
                    return
                log.exception("ph_app.daemon: root %s crashed outside a turn", root.id)
                state = root.recovery
                if state.spent:
                    root.give_up(error_message(error), attempts=state.attempts)
                    # The give-up is flushed, like the retries before it. This
                    # record matters most, and it was the one record that
                    # write-through missed. A daemon that stopped directly
                    # after the give-up left the record in a buffer. The next
                    # daemon resumed the log, saw no `supervisor/failed`, and
                    # reported the root as idle. A ladder that forgets that it
                    # was spent starts again forever.
                    await self._flush(root)
                    return
                root.retry(reason=error_message(error))
                workspace, tree = self._restore_point(root)
                if workspace is not None and tree is not None:
                    # The durable opening of the restore writes this log, and
                    # the retry record with it, before the tree is touched.
                    await self._restore(root, workspace, tree)
                # This flush writes the retry, or the record that settles the
                # restore. In both cases the records of the attempt follow a
                # ladder that the log already has, and not a restore that a
                # repair would report as unknown.
                await self._flush(root)
                await anyio.sleep(state.delay)

    def _schedule_seam(self, root: Root) -> ScheduleService:
        """The schedule seam of this root, or a refusal that says why there is
        none.

        `_live_schedules` reads the seam quietly: a seam of `None` means "no
        schedules". That is the correct answer for a *tick*. It is the wrong
        answer for a person who has just asked to create a schedule. That
        person would get a silent success and a schedule that never fires.
        """
        seam = root.ctx.get(SCHEDULE)
        if seam is None:
            raise ScheduleUnavailable(f'root "{root.id}" has no schedule seam mounted')
        return seam

    async def schedule(self, root_id: str, entry: Schedule) -> Schedule:
        """Record a schedule on a root, and mount the root if it is not mounted.

        The call goes through `start`, like `prompt` and `session/attach`, and
        it has a stronger reason than either. `tick` fires schedules only on
        *mounted* roots. Before `rehydrate` mounted a root for a due
        appointment (P6-23), a schedule on a root that stayed passivated could
        never come due. A root with a live schedule is also a root that
        `passivatable` refuses to release, so the mount is what keeps the root
        mounted.

        The schedule is flushed before the method returns, for the same reason
        as in `tick`. A schedule that exists only in a buffer is a schedule
        that a daemon restart silently forgets. The purpose of the record is
        that nobody has to remember.
        """
        root = await self.start(root_id)
        created: Schedule = self._schedule_seam(root).create(root.session, entry)
        await self._flush(root)
        return created

    async def unschedule(self, root_id: str, schedule_id: str) -> bool:
        """Cancel a schedule. `False` when this log never knew that id."""
        root = await self.start(root_id)
        canceled: bool = self._schedule_seam(root).cancel(root.session, schedule_id)
        if canceled:
            await self._flush(root)
        return canceled

    def scheduled(self, root: Root) -> list[dict[str, Any]]:
        """What will still fire on this root, and when.

        The whole listing uses one stamp, so two schedules that are read in the
        same call agree about when "now" was. `tick` states the same rule for a
        pass.
        """
        stamp = now_ms()
        return [state_to_wire(state, now=stamp) for state in self._live_schedules(root)]

    def _live_schedules(self, root: Root) -> list[ScheduleState]:
        """The schedules of this root that can still fire, or an empty list.

        The seam lookup and its `None` guard are written one time, here. The
        listing, the `schedule` hold of the daemon and the release predicate
        all need them, and each used to have its own copy of the name, the
        guard and the read.
        """
        schedule = root.ctx.get(SCHEDULE)
        return [] if schedule is None else list(schedule.live(root.session))

    async def rehydrate(self, *, now: int) -> None:
        """Mount the unmounted roots that have an appointment due (P6-23).

        **This closes a gap that made no noise.** A schedule outlives its
        process, and `tick` iterates `self.roots`, which is empty at boot. The
        log kept the appointment, and nothing had the log mounted. The only
        symptom was a run that did not happen.

        **The method reads an index and does not scan, for three separate
        reasons.**

        - To read every stored log and find which logs hold a schedule is 500
          reads before the first connection is answered.
        - To mount them all costs a profile, a workspace and possibly a kernel
          for each. P5-05 exists to release that cost.
        - `start` takes the **lease** of P5-03. To wake everything would claim
          every session on the machine, and the next `phern -p` over any of
          them would be refused. That refusal is loud and immediate, and it
          hits sessions that have no schedule at all.

        So only what is due is mounted, and every other session stays unleased.

        **There is no second delivery path.** This method mounts, and the
        ordinary `tick` fires. A missed window therefore behaves as P5-06
        already settled it: `claim` coalesces to the most recent due moment and
        records it write-ahead. A machine that was off from Tuesday to Thursday
        therefore runs the work of Wednesday one time, and not twice or never.

        A failure is for one root, and it is logged. A session that the index
        names and that does not mount costs its own appointment, not the pass.
        **It is tried again**, `wake_retry_delays` after this pass. Each
        consecutive pass that leaves a session behind takes the next step,
        until a pass wakes everything that it meant to wake. Without the retry,
        `next_wake` would never plan for that session, because it drops the
        moments that were already attempted. The lease that a `phern -p` held
        at the appointed minute would then have cost the appointment for as
        long as nothing else woke the scheduler.

        **An index that cannot vouch for itself is rebuilt from the logs** (S18)
        before it is read, so this pass wakes what the rebuild found
        (`_rebuild`).
        """
        stamp = now
        appointments: dict[str, Appointment] = {}
        index = self._index()
        if index is not None:
            if not index.survey().trusted:
                await self._rebuild(index)
            appointments = index.read()
        left_behind = False
        for entry in sorted(appointments.values(), key=lambda one: one.next_at):
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
                left_behind = True
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
        if left_behind:
            rung = min(self._wake_failures, len(self.wake_retry_delays) - 1)
            self._wake_failures += 1
            self._wake_retry = stamp + int(self.wake_retry_delays[rung] * 1000)
        else:
            self._wake_failures = 0
            self._wake_retry = None

    # ------------------------------------------------------ the live index --

    def _note_live(self, root: Root) -> None:
        """Record what `root` owes now, or that it owes nothing (P13-01).

        The method is called on every flush in the mount of the root that wrote
        something (`written` in `_start`), at the mount, at passivation and at
        shutdown. The writer drops an entry that it already wrote, so a flush
        that moved no hold costs the fold in `_live_entry` and nothing else.
        """
        writer = self._live
        if writer is None:
            return
        try:
            writer.note(root.id, self._live_entry(root))
        except Exception:
            log.warning("ph_app.daemon: could not note root %s as live", root.id, exc_info=True)

    def _live_entry(self, root: Root) -> LiveEntry | None:
        """What `root` owes: its entry, or `None` for a root that owes nothing."""
        holds: list[Hold] = []
        if root.agent.status != "idle" or root.status in ("waiting", "retrying"):
            holds.append("turn")
        elif root.agent.inbox.has_pending:
            holds.append("inbox")
        goals = root.ctx.get(GOALS)
        if goals is not None and goals.open(root.session) is not None:
            holds.append("goal")
        subagents = root.ctx.get(SUBAGENTS)
        damaged: tuple[str, ...] = ()
        if subagents is not None:
            if any(child_is_live(child) for child in subagents.children(root.id).values()):
                holds.append("children")
            # A child log that cannot be read (P13-09). The root is listed for
            # it, whether or not the root has holds.
            damaged = subagents.unreadable(root.id)
        header = root.session.header
        entry = LiveEntry(
            session_id=root.id,
            holds=ordered_holds(holds),
            updated=now_ms(),
            family=header.family,
            cwd=header.cwd or "",
            profile=root.ctx.require(MOUNT).profile.name,
            failed=root.recovery.failed,
            damaged=damaged,
        )
        return entry if entry.listed else None

    async def owed(self, *, rebuild: bool = False) -> OwedListing:
        """Every root that the live-roots index lists as owing work, and why
        (P13-02).

        The index is rebuilt first when the caller asks, or when the index
        cannot vouch for itself. The rebuild reads through the store of a live
        root when one is mounted. Otherwise it reads the daemon's own
        `$PH_HOME/sessions` as JSONL. That is the guess that `_rebuild` makes
        for the schedule index, and it is marked in the same way (`vouch`). The
        rebuild runs off the event loop, because it reads the log of every
        root.
        """
        writer = self._live
        if writer is None:
            return OwedListing(sessions=[], trusted=False, rebuilt=False)
        index = writer.index
        survey = await anyio.to_thread.run_sync(index.survey)
        archive, vouched = self._archive()
        rebuilt = False
        if rebuild or (not survey.complete and (vouched or not self._live_guessed)):
            # The log of a mounted root lags what this process knows until its
            # next barrier. The entry of a mounted root is therefore taken from
            # this process and not from disk.
            live = {root.id: self._live_entry(root) for root in self.roots.values()}
            try:
                rebuilt = await anyio.to_thread.run_sync(
                    partial(
                        rebuild_live_index,
                        index,
                        archive,
                        now=now_ms(),
                        since=survey,
                        live=live,
                        vouch=vouched,
                        ladder=LADDER,
                    )
                )
            except Exception:
                log.warning("ph_app.daemon: could not rebuild the live-roots index", exc_info=True)
            self._live_guessed = not vouched
            survey = await anyio.to_thread.run_sync(index.survey)
        rows: list[OwedRoot] = []
        for entry in sorted(survey.entries.values(), key=lambda one: one.session_id):
            root = self.roots.get(entry.session_id)
            if root is None:
                rows.append(OwedRoot.of(entry, mounted=False, status="stored"))
            else:
                rows.append(OwedRoot.of(entry, mounted=True, status=root.status, held=root.held))
        return OwedListing(sessions=rows, trusted=survey.complete, rebuilt=rebuilt)

    async def resume_owed(
        self, sessions: Sequence[str] = (), *, all_owed: bool = False, force: bool = False
    ) -> ResumeReport:
        """Mount the named roots, and every owed root with `all_owed`, so that each
        continues without a client (P13-03). The callers are `sessions/resume`,
        `phern agents resume`, and the pass that `phern daemon --resume-owed`
        runs after its socket is open.

        Each root goes through `start`, which already does everything that a
        resume needs: the log is repaired, the children are readmitted, a turn
        that was cut short is continued (P13-04), and the root is rung for a
        pending prompt. This method only chooses the roots and reports what
        became of each.

        A root whose ladder is spent is left on disk unless `force` is set. A
        mount of it rings nothing, and a resumer that runs every minute would
        otherwise mount it every minute. A root that is listed only for a
        damaged log (P13-09) is left on disk too. It holds nothing that a mount
        would resume, and only a repair changes it. The exception is a root
        that is named, because a person who names a root wants it mounted.

        The roots are mounted one at a time, which is the order that `_mount`
        gives them in any case. A root that does not mount costs its own row
        and not the pass. Examples: another process holds it, it is not a root,
        or its log cannot be read.
        """
        listing = await self.owed()
        listed = {row.session_id: row for row in listing.sessions}
        named = set(sessions)
        wanted = dict.fromkeys([*sessions, *(listed if all_owed else ())])
        rows = [
            await self._resume_one(one, listed.get(one), force=force, named=one in named)
            for one in wanted
        ]
        return ResumeReport(sessions=rows, trusted=listing.trusted)

    async def _resume_one(
        self, session_id: str, row: OwedRoot | None, *, force: bool, named: bool
    ) -> ResumedRoot:
        """One root of a resume pass, and the row that says what became of it."""
        mounted = self.roots.get(session_id)
        if mounted is not None:
            return ResumedRoot(session_id=session_id, outcome="mounted", status=mounted.status)
        left = row.left_by_a_pass(force=force, named=named) if row is not None else None
        if left == "spent":
            return ResumedRoot(
                session_id=session_id,
                outcome="spent",
                detail="its ladder is spent; it waits for a person, or for `force`",
            )
        if left == "damaged" and row is not None:
            return ResumedRoot(
                session_id=session_id,
                outcome="damaged",
                detail=f"nothing to resume; a log beneath it will not read: "
                f"{', '.join(row.damaged)} — `phern doctor sessions --repair`",
            )
        try:
            root = await self.start(session_id)
        except Exception as error:
            log.warning("ph_app.daemon: could not resume %s: %s", session_id, error_message(error))
            return ResumedRoot(
                session_id=session_id, outcome="refused", detail=error_message(error)
            )
        if (name := root.needs_credential) is not None:
            detail = f"waiting for credential {name}"
        elif root.held_on_profile:
            detail = "waiting for a decision about its profile"
        else:
            return ResumedRoot(session_id=session_id, outcome="resumed", status=root.status)
        return ResumedRoot(session_id=session_id, outcome="held", status=root.status, detail=detail)

    def _continue_interrupted(self, root: Root) -> None:
        """Put a turn that the restart cut short back to work, or give up on the
        root (P13-04).

        **Which turns.** A turn that nobody will send again is continued. That
        is the case when a goal is open, or when the turn was owed to nobody at
        a keyboard (`cut_turn_owner`): the turn of a schedule, the notice of a
        child, or an earlier continuation by this supervisor. The setting
        `continue_interrupted` adds the turn of a person. Without the setting,
        the interrupted prompt of a person is left. The person is at a keyboard
        somewhere, and the transcript shows that the turn closed as
        `interrupted`. Two other cases are not continued:

        - A prompt that the turn had not claimed. It is still in the inbox, and
          the ring at the end of `_start` runs it.
        - A turn that was cut before it claimed a prompt, whatever else holds.
          It owes nobody. When it was continued too, one prompt got a second
          turn.

        **How many times.** The ladder of the root is `interrupted_resumes`. It
        is the counterpart of the ladder of a child (`restarts_since_progress`):
        an answer forgives the restarts before it. Past `ROOT_RESUME_LIMIT`, the
        root is failed in its log. `status` then says so, the live-roots index
        carries it, and a resumer skips the root. A turn that was cut before
        its claim climbs the ladder too, although nothing is sent for it. The
        ring runs its prompt again at every boot, and the limit is what ends a
        prompt whose turn dies before its claim each time.

        **What the agent is told** is what a readmitted child is told
        (`cut_short_note`): that the turn above was cut short, which attempt
        this is, and to continue from the transcript and not start again. The
        message is tagged as from this supervisor and not from a person, so a
        transcript draws it as a notice. It is delivered for the next turn,
        which the ring at the end of `_start` starts.
        """
        if root.recovery.failed:
            return
        owner = cut_turn_owner(root.session)
        goals = root.ctx.get(GOALS)
        unasked = (
            owner is None
            or self.continue_interrupted
            or (goals is not None and goals.open(root.session) is not None)
            or owner == "unattended"
        )
        if not unasked:
            log.info(
                "ph_app.daemon: root %s: its turn was cut short; left for the person who sent it",
                root.id,
            )
            return
        attempt = interrupted_resumes(root.session)
        if attempt > ROOT_RESUME_LIMIT:
            root.give_up(
                f"its turn was cut short by {attempt} restarts in a row with no answer between",
                attempts=attempt,
            )
            return
        if owner is None:
            return
        log.warning(
            "ph_app.daemon: root %s: its turn was cut short; continuing it (attempt %d of %d)",
            root.id,
            attempt,
            ROOT_RESUME_LIMIT,
        )
        root.agent.followup(
            create_user_message(
                content=[
                    {"type": "text", "text": cut_short_note(attempt, limit=ROOT_RESUME_LIMIT)}
                ],
                source=PluginSource(
                    plugin="ph_app.daemon.supervisor",
                    form="notice",
                    summary="the turn was cut short by a restart; continuing it",
                ),
            )
        )

    def _index(self) -> ScheduleIndex | None:
        """The index that this daemon reads, or `None` when nothing indexes.

        The index is built from the same `$PH_HOME` that the seam writes under.
        It is not asked of a mounted root, because at boot there are none, and
        boot is the case that this index is for.

        **It is built one time.** `$PH_HOME` is fixed for the life of a daemon,
        and `resolve_roots()` is not free. It reads the environment again and
        canonicalizes three paths, which is a handful of syscalls. To derive
        the index again on each call cost a measurable amount after `booked()`
        joined the turn path. The index object holds a path, not a snapshot.
        `read()` still goes to disk every time, so each pass plans from the
        index as it is.
        """
        if self._schedules is None:
            try:
                self._schedules = ScheduleIndex(resolve_roots().home)
            except Exception:
                log.warning(
                    "ph_app.daemon: no schedule index; nothing will be woken", exc_info=True
                )
        return self._schedules

    async def _rebuild(self, index: ScheduleIndex) -> None:
        """Rebuild an index that cannot vouch for itself (S18). The rebuild is
        skipped when nothing that it depends on has moved since the last try
        left the index in doubt. No timer drives it (docs/seams/schedule.md).

        The rebuild reads through the store of a live root, whatever its kind
        (`SessionArchive.holding`). With no root mounted, it reads the daemon's
        own `$PH_HOME/sessions` as JSONL. That is a guess, so what it finds is
        written and not vouched for, and the first root that mounts is the
        reason to try again.
        """
        archive, vouched = self._archive()
        attempt = (vouched, self._occasions, self.scheduler.rang)
        if attempt == self._tried:
            return
        self._tried = attempt
        try:
            written = await anyio.to_thread.run_sync(
                lambda: rebuild_index(
                    index, archive.holding(REBUILD_TYPES, gate=CREATED), now=now_ms(), vouch=vouched
                )
            )
        except Exception:
            log.warning("ph_app.daemon: could not rebuild the schedule index", exc_info=True)
            return
        if written:
            log.info(
                "ph_app.daemon: rebuilt the schedule index%s",
                "" if vouched else ", from a guess no store vouches for",
            )

    def _archive(self) -> tuple[SessionArchive, bool]:
        """What a rebuild reads, and whether the rebuild may vouch for what it
        finds. It is the store of a live root when one is mounted. Otherwise it
        is the daemon's own `$PH_HOME/sessions`, read as JSONL. That is a guess,
        which `sessions_directory` explains, so what it finds is written and not
        vouched for."""
        store = self._store()
        if store is not None:
            return store, True
        return JsonlArchive(resolve_roots().sessions_dir()), False

    def _store(self) -> SessionPersistence | None:
        """The store of a live root, which answers for the daemon.

        One root can answer for all of them because every root of a daemon
        mounts the same store. The store is a deployment row
        (`session-persistence`). A named profile sets environment rows only,
        and the loader refuses one that sets a deployment row. The row
        therefore comes from the base bundle or from `daemon.yaml`, and every
        composition carries both. Two roots on two profiles (S5) share it.
        `test_two_roots_on_two_profiles_mount_one_store` holds this.
        """
        for root in self.roots.values():
            store = root.ctx.get(SESSION_PERSISTENCE)
            if store is not None:
                return store
        return None

    def appointments(self) -> dict[str, Appointment]:
        """Every appointment on record, or empty when nothing indexes."""
        index = self._index()
        return index.read() if index is not None else {}

    def sessions_directory(self) -> Path | None:
        """Where the store of this daemon keeps sessions.

        The store of a live root answers for the daemon, because every root
        mounts the same store (`_store`). That reply comes from a mounted
        store, so it is tried first.

        **With nothing mounted, the daemon answers for itself**, and does not
        return `None`. P9-05 forced that change. A daemon that has just started
        holds no roots, so `sessions/browse` before the first attach used to
        answer with an empty list. The startup session picker runs *before* the
        first attach, and that is its whole job. It was therefore permanently
        empty in its main case.

        The fallback is the daemon's **own** `$PH_HOME/sessions`. That path is
        the literal default that `session-persistence-jsonl` computes
        (`resolve_roots().sessions_dir()`, in the `apply` of jsonl.py). In any
        deployment that has not overridden the row, it is therefore the same
        path that the store would have named. It is also the home of the daemon
        and never the home of a client. The P5-14 rule is therefore untouched:
        the two must not disagree about which `$PH_HOME` they meant.

        **To move the logs, set `paths.sessions` in `daemon.yaml`.** This
        default comes from that setting too, so the two cannot disagree. Rule 6
        for the other way: a deployment that sets
        `session-persistence-jsonl.root` instead gets a cold browse of the
        wrong directory, until its first root mounts and the answer of the
        store takes over. In practice the wrong directory is absent, and an
        absent directory lists nothing. The `root` of the row stays for a test
        or an embedding that mounts one store somewhere private. To read it
        from outside the row would be a worse trade for a picker.
        """
        store = self._store()
        if store is not None:
            return store.directory()
        return resolve_roots().sessions_dir()

    def busy(self) -> bool:
        """Whether any root is doing something that a person waits for.

        This is the first condition of `passivatable` and **nothing else from
        it**. That predicate also requires a quiet window in the log
        (`passivate_after`, ninety minutes by default). A quiet window is right
        for the release of a *root* and wrong for the end of a *process*. When
        the exit asked the whole predicate, with a window of sixty seconds, it
        put a minute between the last detach and the exit.

        The condition itself is `QUIET`. It is read here and not written out
        again, so the two predicates cannot come to disagree about what "doing
        something" means.
        """
        return any(root.status not in QUIET for root in self.roots.values())

    def booked(self) -> bool:
        """Whether anything is scheduled to happen, on disk or on a live root.

        Both halves are needed, because they answer at different times. The
        index is what a daemon reads at boot, with no roots mounted. The seam
        of a root is what knows about a schedule that was created since.

        **The in-memory half comes first.** `appointments()` opens a file, and
        the seam lookup is a dict hit. This method now runs on every turn
        transition (`DaemonServer.check_lifetime`), so a mounted root that
        already says yes must not pay for the index too. The common case, in
        which nothing is scheduled anywhere, still reaches the index. There it
        costs one failed open and not a parse.

        **The index answers only for sessions that nothing has mounted.** The
        index is a cache, and the logs stay authoritative (I-6). The seam of a
        mounted root has already said no, and an index entry for that root can
        only be staler. Since K10 the index is stale by construction for a
        moment: the write runs off the event loop, after the `cancel` that
        caused it. When the index was asked about a mounted root, a withdrawn
        appointment therefore kept its hold on the daemon until the next sweep.
        """
        if any(self._live_schedules(root) for root in self.roots.values()):
            return True
        return bool(self._unmounted_appointments())

    def _unmounted_appointments(self) -> list[Appointment]:
        """The appointments in the index for sessions that no root has mounted. The
        seam of a mounted root answers for that root, and the index can only be
        staler (`booked`)."""
        return [
            entry for entry in self.appointments().values() if entry.session_id not in self.roots
        ]

    async def wake_and_tick(self, *, now: int | None = None) -> list[str]:
        """One scheduler pass: mount what is due, then fire what is mounted.

        The two steps are separate methods because they answer to different
        owners: `rehydrate` reads an index this daemon does not write, `tick`
        drives roots it does. They are one pass because a root that was woken
        for an appointment and not fired in the same pass would wait for the
        next pass to be noticed.

        The pass is marked as begun before either step runs. A pass that raises
        therefore does not leave what it attempted to wake the scheduler again
        at once (`next_wake`).
        """
        stamp = now if now is not None else now_ms()
        self._last_pass = stamp
        await self.rehydrate(now=stamp)
        return await self.tick(now=stamp)

    def notice_schedules(self) -> None:
        """A schedule changed. Wake the scheduler so that it plans again, and allow
        another rebuild of an index in doubt (`_rebuild`)."""
        self._occasions += 1
        self.scheduler.notice()

    def _watch_schedules(self, _session: Session, event: SessionEvent) -> None:
        """A schedule was made, withdrawn or fired, by whichever path. That happens
        in the log, so the scheduler, root release and the `schedule` hold of
        the daemon all hear it here."""
        if event.type in (CREATED, CANCELED):
            self.notice_schedules()
        if event.type in (CREATED, CANCELED, TICK):
            # Made: the sidebar learns that the daemon is held. Withdrawn, or a
            # `once` that has fired: either can be the last thing that keeps
            # this root mounted, and the daemon up with it.
            self._recheck()

    def _recheck(self) -> None:
        """Something that the release of a root, or the life of the daemon, depends
        on may have moved. Plan the release again, and ask again what holds the
        process.

        `Root.recheck` points here. It is called on every status transition of
        the agent, and for the changes that the status of the agent does not
        announce: a root that starts to wait for a person or stops, and the
        ladder that ends as failed or as recovered.

        The verifier is told too. A root that was written to while it worked
        becomes due only when it settles, and to settle is a status change."""
        self.releaser.notice()
        self.verifier.notice()
        self.recheck_lifetime()

    def next_wake(self, now: int) -> int | None:
        """When the scheduler next has work, or `None` when it has none at all.

        The answer is the earlier of two moments: a schedule that is due on a
        mounted root, by its log, and an appointment of a session that is not
        mounted, by the index. Anything that was due by the last pass was
        attempted then: it was fired, or declined and logged. It therefore does
        not hold the sleep at zero. No pass begins sooner than `PASS_FLOOR`
        after the last pass, so an interval of a millisecond is not a busy
        loop. An appointment whose session did not mount is the one attempt
        that is planned again, at the moment that `rehydrate` set for it.
        """
        moments = [
            due
            for root in self.roots.values()
            if (seam := root.ctx.get(SCHEDULE)) is not None
            and (due := seam.next_due(root.session, now=now)) is not None
        ]
        moments.extend(entry.next_at for entry in self._unmounted_appointments())
        if self._wake_retry is not None:
            moments.append(self._wake_retry)
        ahead = [moment for moment in moments if moment > self._last_pass]
        if not ahead:
            return None
        return floored(min(ahead), self._last_pass)

    async def tick(self, *, now: int | None = None) -> list[str]:
        """Fire whatever is due on every mounted root (P5-06). Returns the ids of
        the schedules that fired.

        **The firing goes through the door of the seam** (`fire_due`, P13-07),
        which keeps the order that a crash needs: the tick is ahead of the
        prompt in the log, one write follows the delivery, and the index is
        told only after both are on disk. A write that fails marks the root as
        unwritten here. The index stays where it was, and the turn that the
        prompt starts is refused at its first request barrier. No run therefore
        happens without its tick on disk.

        The delivery is `prompt`, which is the path that a message from a
        person takes. A scheduled turn is therefore an ordinary turn with a
        record that says why it started. It carries the source of the schedule
        (`scheduled_source`), so a reader can tell it from the turn of a
        person.

        The whole pass uses one stamp, so two schedules that are due in the
        same second agree about what "now" was.
        """
        stamp = now if now is not None else now_ms()
        fired: list[str] = []
        for root in list(self.roots.values()):
            if root.ctx.get(SCHEDULE) is None:
                continue
            # The `try` covers the whole body for one root. `claim` rarely
            # raises (a bad expression logs and declines). `prompt` and the
            # write are the two calls that really fail. One bad root must not
            # end the whole pass.
            try:
                done = await fire_due(
                    root.ctx, root.session, now=stamp, deliver=partial(self._fire, root)
                )
            except Exception:
                log.exception("ph_app.daemon: root %s failed its schedule tick", root.id)
                continue
            fired.extend(entry.id for entry in done)
        return fired

    async def _fire(self, root: Root, entry: Schedule) -> None:
        """Deliver the prompt of one schedule to its root, for the next turn.

        The reach is `next-turn`, because a scheduled turn is an ordinary turn.
        That is the contract of `tick`. It also prevents two schedules that are
        due in one pass from becoming one turn.
        """
        log.info("ph_app.daemon: root %s firing schedule %s", root.id, entry.id)
        await self.prompt(root.id, entry.prompt, reach="next-turn", source=scheduled_source(entry))

    def _verify_at(self, root: Root) -> int | None:
        """When the check of `root` is due, in epoch ms, or `None`. It is `None`
        when the check is off, when nothing was written since the last check, or
        when the root is not settled.

        A settled root is quiet and has nothing working beneath it. A flush in
        the middle of a turn is not a settle. Without the second condition, a
        child that flushes at every step under an idle parent would cause a
        refold of the whole mount of the parent at each flush."""
        if not self.check_invariants or root.unverified_at is None:
            return None
        if root.status not in QUIET or _working_beneath(root):
            return None
        return root.unverified_at + int(self.verify_after * 1000)

    def next_verify(self) -> int | None:
        """When the next root is due its check, epoch ms, or `None` while none is."""
        moments = [at for root in self.roots.values() if (at := self._verify_at(root)) is not None]
        if not moments:
            return None
        return floored(min(moments), self._last_verify)

    async def _verify_quiet(self) -> None:
        """The verifier's pass: check every root whose moment has come."""
        stamp = now_ms()
        self._last_verify = stamp
        for root in list(self.roots.values()):
            at = self._verify_at(root)
            if at is None or at > stamp:
                continue
            # **One checkpoint for each root.** One `verify()` takes 12 ms on a
            # log of 25k events and 161 ms on a log of 500k events. Nothing else
            # here awaits, so several roots that are due together would run
            # one after another with the loop blocked. A yield between them
            # limits the stall to one root. It does not weaken the reason why
            # this is not threaded. That argument is about a torn read
            # *inside* one `verify`, which is still atomic with respect to the
            # loop.
            await anyio.lowlevel.checkpoint()
            await self.verify_root(root)

    async def verify_settled(self, root: Root) -> None:
        """Check `root` now if anything in its mount was written since its last
        check.

        This is the last check before a release (`passivate`). The verifier
        would reach the root after `verify_after`, and by then its caches are
        gone. The call costs nothing when nothing new was written.
        """
        if self.check_invariants and root.unverified_at is not None:
            await self.verify_root(root)

    async def verify_root(self, root: Root) -> list[Violation]:
        """Check the pollable invariants of one root. Record the set of broken
        invariants when it differs from what the log of the root already says.
        Returns what is broken.

        **The gap that this closes.** `ph.seams.invariants` divides invariants
        into two kinds. An inline invariant, such as I3, refuses on the path
        that it governs. A *pollable* invariant carries a `check`, because a
        projection either equals its fold now or it does not. `phern doctor`
        mounts a fresh profile, so it reports that the checks run, and never
        that the live state of a deployment passed them. The caches that those
        checks exist to catch can drift only in a process that has been up for
        a while, and this is that process. **The check is for one root, because
        the mount is for one root.**

        **The check runs on the event loop and is deliberately not threaded.**
        The checks refold live session state, and only this loop changes that
        state. On another thread they would race an append and report a torn
        read as a violation. That is the worst possible failure for a check
        whose only job is to be believed.

        The record is flushed, for the reason that `announce_unreachable` gives,
        and the reason is stronger here. This is a finding about bookkeeping
        that the same process owns, so the buffer of that process is the one
        place where it must not stay.

        A root whose profile mounts no `invariants` row is skipped, and is not
        reported as holding. An invariant that is not mounted is absent from
        the report, which is the rule of the seam itself.
        """
        registry = root.ctx.get(INVARIANTS)
        if registry is None:
            return []
        # `verify` itself contains a check that raises, and turns it into a
        # finding and does not drop it. There is therefore nothing to guard
        # here, and a guard would hide the one outcome that this method is
        # for.
        current = registry.verify()
        # **The record is a change, not a sample.** That is the difference
        # between a record and a leak. A cache that drifted does not repair
        # itself. A check that appended what it saw would write the same
        # finding at every settle for the life of the root, in the log that a
        # person opens *because* something is wrong.
        #
        # It is not a latch either, as `DaemonServer.unreachable_since` is. That
        # transition is one-way by construction, and this one is not. A
        # generation bump rebuilds the derivation, so a cache can start to hold
        # again. A latch would then miss the next drift, which is the one
        # failure that this check exists to catch.
        #
        # The set is read back from the log and not held on `Root`, so the
        # comparison survives passivation and resume. A root that was released
        # overnight and restarted must not announce again what its own
        # transcript already says. `latest` is an incremental fold, so the
        # read costs nothing.
        if {one.invariant for one in current} != _recorded_violations(root.session):
            if current:
                log.error(
                    "ph_app.daemon: root %s violates %d invariant(s): %s",
                    root.id,
                    len(current),
                    "; ".join(f"{one.invariant}: {one.detail}" for one in current),
                )
            else:
                # The clearing is recorded too. After a transcript that says
                # "violated" and then goes quiet, a reader cannot tell a fixed
                # cache from a daemon that stopped looking. The
                # `supervisor/passivated` record exists for the same reason: a
                # gap that nothing explains.
                log.info("ph_app.daemon: root %s holds its invariants again", root.id)
            root.violated(
                {
                    "violations": [
                        {"invariant": one.invariant, "detail": one.detail} for one in current
                    ],
                    # The pid is here for the same reason as in the note that
                    # `announce_unreachable` writes, which
                    # `DaemonServer.check_reachable` builds: ten roots in one
                    # incident can be correlated afterwards.
                    "pid": os.getpid(),
                }
            )
            await self._flush(root)
        # This line runs after the flush of the record, which marked the root as
        # unverified again. The record is nothing new to the next check.
        root.unverified_at = None
        return current

    async def announce_unreachable(self, note: dict[str, Any]) -> None:
        """Write the "nobody can reach me" record into every root, and flush it.

        **The record is flushed and not left in the buffer.** That is not the
        usual standard here. Every other record survives a crash because the
        log is written when the daemon stops, and this record is written
        exactly when the stop is no longer reliable. If the answer of a person
        to an unreachable daemon is `kill`, a record that was not flushed
        explains nothing to anybody.

        The record goes to every root, not only to the busy roots. The daemon
        is what became unreachable, and a transcript that stops with no
        explanation is the same puzzle in both cases.
        """
        for root in list(self.roots.values()):
            root.unreachable(note)
            await self._flush(root)

    async def _flush(self, root: Root) -> None:
        """Write the log of `root`, or continue without the write. The crash that a
        ladder retries can be a flush that fails. A failure marks the root
        through the store (`session/unwritten`, heard in `_start`)."""
        await session_written(root.ctx, root.session)

    def _could_not_write(self, root: Root, session_id: str, reason: str) -> None:
        """A write of one of the logs of `root` raised: its own log or the log of a
        child (P13-08).

        The failure is logged one time for each log, at error level, with the
        reason that the write gave. The store, or the last write of the mount,
        logs a warning for each attempt, and that warning belongs to the
        mechanism. This line is the fact that an operator reads a daemon log
        for."""
        if session_id not in self.unwritten:
            log.error(
                "ph_app.daemon: session %s of root %s could not be written (%s); readiness is no",
                session_id,
                root.id,
                reason,
            )
        self.unwritten[session_id] = root.id

    def _restore_point(self, root: Root) -> tuple[Workspace | None, str | None]:
        """The workspace of the root, and the tree that a retry starts from. The
        tree is `None` when the root has none.

        A retry against a tree that was partly changed would be a different
        turn from the turn that failed. The model would see edits from an
        attempt that nobody kept, and a ladder that adds to its own damage is
        worse than no ladder. `workspace/checkpoint` is the record of P4-09 and
        is already a fold, so this method asks the log and does not remember.
        A root on the advisory tier has nothing to restore, which is the
        ordinary answer.
        """
        # This uses `workspace_of`, not `ctx.workspace.of`. The method runs
        # inside the `except` of `_drive`, and `ctx.workspace` *raises* on a
        # profile that layers no workspace row. The raw lookup would therefore
        # escape the handler whose job is to keep the task group alive. The
        # helper is fail-soft, and it is the one form of this question that
        # the rest of the tree uses.
        workspace = workspace_of(root.ctx, root.agent)
        if workspace is None:
            return None, None
        # The seam answers `""` for an agent that took no restore point.
        return workspace, stated(latest_checkpoint(root.session, root.agent.id))

    async def _restore(self, root: Root, workspace: Workspace, tree: str) -> None:
        """Put the tree of the root back to `tree`, through the seam that records
        the restore.

        **The record belongs to the seam** (`WORKSPACE_RESTORE`, opened inside
        `WorkspaceSeam.restore`). It is on disk before the tree is touched, and
        it is settled after. A repair closes a restore that the daemon died
        inside as "may be partly restored". The retry is already in the log
        ahead of the restore, and the durable opening of the restore writes
        both. The attempt is therefore counted whatever happens here.

        The restore is best-effort by construction: a restore that fails must
        not cost the retry. The attempt goes ahead against the tree as it is,
        and `workspace/restored` says `ok: false`, so the transcript does not
        imply a rollback that did not happen. A restore that the log cannot
        record is not run, for the same reason.
        """
        try:
            # The call goes through the seam. `require` is safe here, although
            # the raw lookup in `_restore_point` is not: that method found a
            # workspace, so the row is mounted.
            await root.ctx.require(WORKSPACE).restore(root.agent.id, tree, session=root.session)
        except IntentNotDurable:
            log.warning(
                "ph_app.daemon: root %s was not restored to %s, because the log could not "
                "record it",
                root.id,
                tree,
            )
        except Exception:
            log.warning(
                "ph_app.daemon: root %s could not be restored to %s", root.id, tree, exc_info=True
            )

    async def prompt(
        self,
        root_id: str,
        text: str,
        *,
        attachments: Sequence[AttachmentRef] = (),
        reach: Literal["soonest", "next-turn"] = "soonest",
        source: PluginSource | None = None,
    ) -> Root:
        """Splice a turn into the inbox of the agent, and wake its task.

        The method returns as soon as the message is *logged*, not when the
        turn is done. The caller is a socket handler that serves other clients.
        A watcher learns that the root is working from `agent/status`, and not
        from a reply that never came.

        **There is no idempotence here any more.** The `MUTATIONS` table of the
        daemon applies the write-ahead guard to every mutating method, before
        this method is called. The scheduler calls this method directly and
        carries no key. The parameter is therefore gone, and not optional.

        **`reach` says whose message this is**, and the two answers differ. A
        person who types while the agent works means *"also this"*, and wants
        it read at the next step. The message of a child already had that reach
        (`rlm-messaging` steers, so "a running agent picks it up without
        finishing first"), and the message of a person did not. A **schedule**
        means something different. The contract of `tick` is that a scheduled
        turn is an *ordinary turn with a record saying why it started*. A
        schedule therefore takes `next-turn`, which keeps that true. If it
        joined a turn that it has nothing to do with, it would also share the
        per-turn ceilings of that turn, and two schedules that are due in one
        pass would merge into one turn.

        The consequence of `soonest`: an interjection *joins* the running turn,
        so no `turn/start` is appended. `CallBudget.turn_limit` says what that
        costs a per-turn ceiling, beside the setting that it is about, so it is
        not written a third time here.

        `source` says whose words these are when they are not from a person
        just now: the words of a schedule (`tick`). The wake comes from the
        driver. Both deliveries emit `agent/woken`, and the listener of the
        root turns that event into its ring (P13-05).
        """
        root = await self.start(root_id)
        # The list of the client comes first, because it is what that person
        # just named. The tray follows in the order in which it was filled, and
        # `take` drains it. See `Tray`.
        taken = root.staged.take()
        if taken:
            root.publish(SessionStagedNotice(session_id=root.id))
        message = prompt_message(text, [*attachments, *taken], source=source)
        if reach == "next-turn":
            root.agent.followup(message)
        else:
            # The loop decides whether this lands in the middle of a turn. The
            # phase belongs to the loop, and a status that is read here could
            # be stale by the time that the verb lands.
            root.agent.interject(message)
        return root

    async def shell(
        self, root_id: str, shell: ShellService, command: str, *, surface: bool = False
    ) -> ShellReply:
        """Run a shell command of the user in the workspace of the session (P7-10).

        **The command runs in the root, not in the client.** `ctx.shell`
        resolves the working directory and the containment tier from the agent.
        The command therefore runs where the session lives, and is bounded as
        the session is. A browser tab has no shell, and "the shell of the
        session" is the correct meaning of the verb in both front ends.

        The output travels as *events*, not in the reply. Every attached UI
        therefore learns what happened by the one route that they all watch
        already. The seam is passed in because the caller resolved it before it
        claimed the idempotence key. See `shell_of`.
        """
        root = await self.start(root_id)
        result = await run_shell(shell, root.session, root.agent, command, surface=surface)
        return ShellReply(session_id=root.id, exit_code=result.exit_code)

    def describe(self) -> list[RootDescription]:
        return [root.describe() for root in self.roots.values()]

    def awaited(self) -> list[tuple[str, str]]:
        """Every credential that a mounted root, or a sub-agent beneath it, waits
        for, and who waits. The result is `(name, "root, root/run, …")` rows for
        `phern agents doctor` (T5).

        The rows are read from the logs of the sessions that wait: the log of a
        root, and the log of each child (`_waiting_on`). The resume check
        recorded the holds there, so the doctor and a transcript say the same
        thing. Roots that are not mounted are not asked. Their holds are on
        disk, and are checked again when the roots are next started.
        """
        return _waiting_on(self.roots.values())

    def passivatable(self, root: Root, *, now: int, after: float) -> bool:
        """Whether this root may be released (P5-05).

        Every condition is a reason why a root is still *wanted*. Each is read
        from something that already exists, and not from a flag that is set
        beside it:

        - **It is doing something.** This is `status`, which covers `retrying`.
          A root in the backoff of P5-04 is idle between attempts, and a
          release there would passivate a root in the middle of its ladder.
        - **Somebody is watching.** This is `subscribers`, the connections that
          are attached to the id of the root. A client that attached and never
          detached keeps its session alive by the same fact that makes it
          receive events.
        - **It has work beneath it.** A child is still working, by the log of
          the child (`_working_beneath`). A root that is released under a
          running child takes the mount of the child with it. The child is
          suspended in the middle of its task, and comes back only when
          something wakes the root.
        - **It has work scheduled** (P5-06). A root with a live schedule has
          already said when it comes back.
        - **It has not been quiet long enough**, by the log.

        The first four are `_held`, which `next_release` reads too. The moment
        that a root is planned for and the predicate that releases it therefore
        cannot disagree.
        """
        return root.idle_for(now) >= after * 1000 and not self._held(root)

    def _held(self, root: Root) -> bool:
        """Whether something other than recent activity keeps `root` mounted.

        The cheap terms come first. The two folds go through the caches of the
        seams and not through the bare functions. Each `SessionFoldCache` keys
        on the `seq` of its log, and an idle log does not grow, so a second
        call is a dict hit and not a walk of the whole log. Since P12-01 this
        is asked when a release is planned or a pass runs, and not on a
        sixty-second sweep. A root whose only claim is an unsettled child is
        still asked on every plan.
        """
        return (
            root.status not in QUIET
            or bool(root.subscribers)
            or _working_beneath(root)
            or bool(self._live_schedules(root))
        )

    def next_release(self) -> int | None:
        """When the next root may be released, in epoch ms, or `None` while none can
        be.

        The answer is the earliest end of a quiet window, by the log, among the
        roots that nothing else keeps mounted (`_held`). A root that something
        else keeps mounted has no moment. Whatever lets it go sends a notice
        (`Supervisor.releaser`), and the plan is made again then. The answer is
        no sooner than `PASS_FLOOR` after the last pass, so a root whose
        release raised is not retried in a busy loop.
        """
        if self.passivate_after is None:
            return None
        window = int(self.passivate_after * 1000)
        moments = [
            root.quiet_since + window for root in self.roots.values() if not self._held(root)
        ]
        if not moments:
            return None
        return floored(min(moments), self._last_release)

    async def sweep(self, *, after: float | None = None, now: int | None = None) -> list[str]:
        """Release every root that has been quiet long enough. Returns their ids.

        The method returns the ids and does not log them. A test, and the
        `phern agents` of P5-10, can then ask what happened without a parse of
        a log line, and the caller decides whether a sweep that released
        nothing is worth a message.

        The loop iterates a copy, because passivation removes from
        `self.roots`.
        """
        stamp = now if now is not None else now_ms()
        self._last_release = stamp
        window = self.passivate_after if after is None else after
        if window is None:
            return []
        released: list[str] = []
        for root in list(self.roots.values()):
            if self.passivatable(root, now=stamp, after=window) and await self.passivate(
                root, now=stamp, after=window
            ):
                released.append(root.id)
        return released

    async def passivate(self, root: Root, *, now: int, after: float | None = None) -> bool:
        """Release the process-side state of a root, and keep its session on disk.
        Returns whether the root was released.

        With `after`, the root is released **only if it is still quiet after the
        lock is held** (D4). The release pass decides and then waits for
        `_starting`. Before this check, a client that attached and prompted
        during that wait had its root recorded as `passivated` and cut short.
        The check uses the same `now`, so a prompt during the wait, which is a
        newer last event, reads as no quiet at all. Without `after`, the root
        is released as asked.

        **Rehydration is already written.** `start()` resumes any root whose
        log exists (P5-01), so to wake a root is the ordinary path and not a
        second mechanism.

        The order matters, and it is the order of `aclose`: record, flush, then
        unwind. The unwind disposes the mounted `Context` and the P5-03 lease
        with it. Another process may therefore legitimately open a passivated
        session. That is the purpose, not an oversight. The wake channel
        closes first, so that a task that waits for work leaves its own loop.
        A task inside a turn is canceled after the release (`_unmount`, D1).
        """
        async with self._starting:
            # This is the lock that orders a mount (`_mount`), and it is taken
            # here for the same reason, seen from the other side. Passivation
            # removes the root from `self.roots` and *then* unwinds. Without
            # the lock, a `start` that arrives between the two finds no root
            # and tries to open the session. The P5-03 lease, which the
            # outgoing root has not given back yet, refuses it:
            # `session_already_active` for a session that nobody uses.
            # `session/attach` made that case reachable when it began to wake
            # a passivated root through `start`.
            if not self._holds(root) or (
                after is not None and not self.passivatable(root, now=now, after=after)
            ):
                return False
            # The root leaves the table before anything is awaited, as
            # `_unmount` would remove it. A caller from here on mounts the log
            # again, after this release.
            del self.roots[root.id]
            # The idle time is measured first. A violation that the check below
            # records is a new last event, and the release would then report a
            # root that was idle for no time at all.
            idle_ms = root.idle_for(now)
            # This is the last check of its caches before they are gone
            # (P12-04). A drift that is found now is recorded in the log that a
            # resume will read, and is not lost.
            await self.verify_settled(root)
            root.passivated(idle_ms)
            log.info(
                "ph_app.daemon: passivating root %s after %d minutes idle",
                root.id,
                idle_ms // 60_000,
            )
            # A quiet root owes nothing that a resumer must mount it for. The
            # exception is a goal that is still open. The entry keeps that
            # hold, so that a watchdog can report it.
            self._note_live(root)
            await self._unmount(root, why="released after going quiet")
            return True

    async def _unmount(self, root: Root, *, why: str) -> None:
        """The one door that a root leaves by. A passivation, a restart and a
        shutdown all use it.

        The steps run in this order, each for a reason (Holdings rules 2 and
        4):

        1. **The root leaves the table, before anything is awaited**, and only
           if this root is the one in the table. A caller from here on mounts
           its log again, after this release (`_starting` orders the two), and
           is not given a root that is leaving.
        2. Its wake is closed, so a task that waits for work leaves its loop.
        3. The root is released: flushed, then unwound. Every way out uses this
           one sequence. When shutdown and passivation each had their own, the
           pair diverged twice. An earlier `aclose` skipped `exits.aclose()`
           when the flush raised, and left the context of an unwritable root
           undisposed.
        4. **Its task is ended, after the release** (D1). A task inside a turn,
           or one that waited for a person, used to continue against a disposed
           mount. The task ends after the release so that the log on disk is
           the log that the release left. A turn that a shutdown cut stays
           open for the resumer of the next daemon, and nothing that the task
           writes as it ends reaches a log.
        5. **The root leaves its audience**, last. Whoever is attached
           therefore hears the release itself, and nothing that this root says
           after it. The audience belongs to the id, and the next mount speaks
           to it.
        """
        if self._holds(root):
            del self.roots[root.id]
        subagents = root.ctx.get(SUBAGENTS)
        if subagents is not None:
            # Without this call, the children that the seam read for this root
            # would outlive the root. The seam keys them by the id of the
            # parent, and nothing else tells the seam that this parent is gone.
            subagents.forget_session(root.id)
        await root.wake.aclose()
        await self._flush(root)
        try:
            await root.exits.aclose()
        except Exception:
            log.warning("ph_app.daemon: root %s did not unwind cleanly", root.id, exc_info=True)
        for session_id, reason in unwritten_at_unwind(root.ctx).items():
            self._could_not_write(root, session_id, reason)
        root.driving.cancel(why)
        await root.stopped.wait()
        # The root has ended, so it leaves the audience of its id (Holdings
        # rule 3). The audience belongs to the id and to the next mount, and
        # anything that this root still says reaches nobody.
        root.audience = Audience()

    async def aclose(self, *, deadline: float | None = None) -> None:
        """Close the wake channel of every root, and unwind its context.

        The channels close first, so that each task that waits for work leaves
        its own loop. A task inside a turn is ended after the release of its
        root (`_unmount`, D1). The roots are then released one at a time,
        because a teardown that fails for one root must not strand the others
        (I2). Each root is flushed *before* it unwinds. That flush cannot cover
        what the unwind itself appends (`workspace/disposed`, the tombstone of
        a child). The last act of the mount writes those records. The `claim`
        of the store registered that act beside the lease (`write_on_unwind`,
        F2).

        **`deadline` is one budget for the whole shutdown.** It is set on each
        root before the root is released. Every root is its own `Context` with
        its own runtime, and `Context.dispose` shields itself. Without the
        shared deadline, each root took a fresh `GRACE_SECONDS`, and the
        ten-second bound of the caller became ten seconds *for each root*. The
        deadline is set on the context and not passed, because a root unwinds
        through its `AsyncExitStack` and there is no argument to pass it
        through.

        **Mounts in flight are waited for first, and none is admitted after.**
        `self.roots` is the set that this method can release, and a mount in
        flight is by definition not in it yet. A root that finished after the
        loop below would hold a session that nothing flushes and a worktree
        that nothing reclaims (F6). Teardown exists to prevent that. The wait
        takes the same budget as the rest. A mount that is still building when
        the budget ends is left to the final `tasks.cancel_scope.cancel()` of
        `serve`. That cancel ends the mount through the same shielded `finally`
        that every other unwind goes through. The worst case is therefore a
        root that never existed, and not a root that is left half-built.
        """
        self._closing = True
        if self._in_flight:
            log.info("ph_app.daemon: waiting for %d mount(s) in flight", len(self._in_flight))
            bound = math.inf if deadline is None else deadline
            with anyio.CancelScope(deadline=bound, shield=True):
                for pending in list(self._in_flight):
                    await pending.done.wait()
        roots = list(self.roots.values())
        for root in roots:
            await root.wake.aclose()
        # This records what each root still owes, as the last statement of this
        # daemon about it (P13-01). The resumer of the next daemon must mount a
        # root that was stopped in the middle of a turn. Every root is noted
        # before any root is released. The index then takes them in one write
        # and not one write for each root, because the writer runs during the
        # awaits of the first release. And what one root owes does not change
        # when another root is released.
        for root in roots:
            self._note_live(root)
        for root in roots:
            if deadline is not None:
                root.ctx.unwind_by(deadline)
            # Each root goes through the one door. It therefore leaves the table
            # as it is released, and not all roots after the last release
            # (D7). Its task ends after its release, so the turn that a
            # shutdown cuts stays cut on disk.
            await self._unmount(root, why="the daemon stopped")
        if self._live is not None:
            # This runs under the same `deadline` as the rest, with no bound of
            # its own. The write runs in this task, and a write on a thread is
            # not abandoned on a cancel. A second clock would therefore bound
            # only the wait before the write began.
            await self._live.settle()
