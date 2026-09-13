"""`$PH_RUNTIME/daemon.sock` — the supervisor's front door (P5-01).

A unix socket rather than stdio, which is the whole point: stdio has exactly one
peer and dies with it, and a daemon exists to be reconnected to. The method names
extend `--mode rpc`'s shape rather than starting a second vocabulary.

**The socket is state, and stale state is a lie.** A path left behind by a crashed
daemon makes every client hang on a connect that will never be answered, so
binding removes an unresponsive one first and **refuses a responsive one** — the
second is another daemon, which is P5-03's lease to arbitrate rather than this
row's to overwrite.

The same sentence read the other way is P5-11: a path that stopped being *this*
daemon's socket is a lie about this daemon. `$PH_RUNTIME` sits under
`$XDG_RUNTIME_DIR`, which logind reaps at logout for a user who is not lingering,
so the door can be removed while the process behind it keeps running — and every
later client is told "no daemon socket" and to start one, which the leases the
first is still holding will refuse. `watch` compares the socket's inode against
the one bound here, and says so in each root's own log, because by then the
surfaces that could carry the news are exactly the ones that went away.

@module ph_app.daemon.server
"""

from __future__ import annotations

import logging
import os
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from anyio.abc import ByteStream

from ph.agent.types import AgentCancelCause
from ph.cordis import Profile
from ph.keys import ATTACHMENTS, COMMANDS, CREDENTIALS, PERMISSION_PRESETS, TOOLS
from ph.lingering import RuntimeLifetime, lifetime, socket_identity
from ph.llm.types import AttachmentRef
from ph.paths import resolve_roots
from ph.resources import GRACE_SECONDS
from ph.seams.attachments import mime_for
from ph.seams.schedule import Schedule
from ph.seams.shell import ShellService
from ph.session import now_ms
from ph.wire import WireModel

from .. import verbs
from ..params import (
    CancelScheduleParams,
    CommandParams,
    CreateScheduleParams,
    InitializeParams,
    MutationParams,
    NewSessionParams,
    PresetParams,
    PromptParams,
    PutAttachmentParams,
    ShellParams,
    SnapshotParams,
    StageParams,
    StoreCredentialParams,
    TrustAnswer,
)
from ..payloads import (
    AttachmentStored,
    AttachReply,
    CommandShown,
    CredentialsHeldReply,
    CredentialStored,
    DaemonConfigReply,
    DaemonStatusReply,
    DiagnosticRow,
    DiagnosticSection,
    MutationRepeated,
    PresetApplied,
    RootDescription,
    RootListing,
    RootStatusReply,
    ScheduleCancelled,
    SessionBrowse,
    SessionDetached,
    SessionNotice,
    SessionSchedulesReply,
    SessionStagedNotice,
    SessionStatusNotice,
    ShellReply,
    SnapshotPage,
)
from ..protocol import (
    PROTOCOL_VERSION,
    SNAPSHOT_EVENTS,
    CapabilityBlock,
    NoParams,
    Notify,
    Refusal,
    SeamAbsent,
    SessionParams,
    UnknownMethod,
    Verb,
    capabilities,
    cursor_of,
    resume_at,
    served,
)
from ..shell import shell_of
from ..trust import TrustStore, trust_path
from .cards import CARD_EVENTS, presentation_of
from .duplex import Peer
from .framing import MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_SIZE
from .launch import listening
from .projections import (
    browse_of,
    commands_of,
    credentials_of,
    presets_of,
    readings_of,
    screens_of,
    skills_of,
    tools_of,
)
from .recovery import EPHEMERAL_QUIET, PASSIVATE_AFTER, WAKE_WITHIN
from .supervisor import NON_GUARANTEES, Root, Supervisor

if TYPE_CHECKING:
    from ph.seams.attachments import AttachmentStore
    from ph.seams.commands import CommandRegistry
    from ph.seams.credentials import CredentialService
    from ph.seams.permission_presets import PermissionPresetService

__all__ = ["DaemonServer", "serve"]

log = logging.getLogger("ph_app.daemon")

HEARTBEAT_EVERY = 5 * 60.0
"""Seconds between liveness records for a root that has work scheduled.

Not a keep-alive and not a health check: a record, so an operator reading a
cron-driven trace can tell "waiting for Wednesday" from "died on Tuesday". A
schedule that fires monthly otherwise leaves a log whose last line is a month
old, which is indistinguishable from a log nobody is writing.

Beside the other two cadences rather than in the seam, where it was: ph-core
held a constant only this loop read."""

TICK_EVERY = 5.0
"""How often due schedules are checked. The floor on a schedule's resolution.

Five seconds rather than the sweeper's sixty: this decides when work *starts*,
and a minute of slack on "run at 09:00" is a minute somebody notices."""

SWEEP_EVERY = 60.0
"""How often the passivation sweep runs. A coarse tick, not a second timeout."""

WATCH_EVERY = 30.0
"""How often the daemon checks that the socket at its path is still its own (P5-11).

The thing being watched changes at most once in the life of a process — a
logout, a reboot's worth of directory — so this is not a poll on a hot fact. It
is a bound on how long the log takes to say what happened, and thirty seconds
means the record's timestamp still lines up with the logout a person is trying
to correlate it with. Cheaper than the sweep it sits beside: one `lstat`, no
roots walked.
"""


@dataclass(frozen=True, slots=True)
class Mutation[P: MutationParams, R: WireModel]:
    """One method that changes a root: what it takes, how to validate it, how to do it.

    Two halves rather than one body, and the seam is the idempotence key. The
    daemon claims the key *between* them — after `prepare` has had its chance to
    refuse, before `act` has had its chance to do anything — so a refusal never
    consumes a retry and a crash never loses one. Both halves receive the
    connection, the root and the **parsed** params; `prepare` hands `act`
    whatever it resolved, so a seam is looked up once and a refusal happens
    before any record is written.

    The `Verb` is the row's identity: it carries the name, the params model the
    wire dict is checked against, and the reply. The check is the wrapper's
    first move — before `start` resolves a root — so a malformed call never
    mounts anything. Generic in both so each half is typed against its own
    method: `_act_prompt` reads `.prompt` and `.attachments`, and a half that
    read a field its method does not carry is a type error rather than a
    `KeyError` at the first real call.

    **`act` returns the verb's reply and no longer `Any`** (issue 74). It was
    `Awaitable[Any]` because "a verb answers with whatever its reply's type is —
    a model where the reply has one, a dict where it is a per-method shape", and
    the dicts are gone: every reply is a model, so the return is checkable and
    an `act` that answered with another verb's model is now a type error here
    rather than a client's `ValidationError` at the first real call.
    """

    verb: Verb[P, R]
    prepare: Callable[[_Connection, Root, P], Awaitable[Any]]
    """`Any` is the *plan*, not the reply: what `prepare` resolved and hands to
    `act`. Each pair agrees on it by being written together, and no third party
    ever sees it."""
    act: Callable[[_Connection, Root, P, Any], Awaitable[R]]


@dataclass(frozen=True, slots=True)
class Method[P: WireModel, R: WireModel]:
    """One method that reads, or acts without an idempotence key.

    The same shape as `Mutation` minus the two-halves discipline, for the same
    reason: a row names its verb beside its handler, so the handler is typed
    against both what the wire may carry and what it must answer with, and the
    *set* of methods is a value a test can hold against the vocabulary — the
    argument `tui/adapter.py`'s `HANDLERS` makes about event types.
    """

    verb: Verb[P, R]
    handle: Callable[[_Connection, P], Awaitable[R]]


@dataclass(frozen=True, slots=True)
class Announced[P: WireModel]:
    """One method that answers **nothing**: `shutdown`, and only ever it.

    A sibling of `Method` rather than a `Method` whose `R` is some unread
    model, because that model was the problem: `ShutdownAck` existed to give
    the verb a reply type and the handler something to return, for a frame
    `respond` computes and then drops because the id is `None`.

    Its handler returns `None`, and `respond` sends nothing — which is what
    lets the *type* carry the contract that "stop" is not a question. See
    `ph_app.protocol.Notify`.

    Rows of this shape sit in `METHODS` beside the others rather than in a
    third table: `_dispatch` reads `.verb.parse` and `.handle` off either, so
    one lookup still answers "is this a method the daemon knows".
    """

    verb: Notify[P]
    handle: Callable[[_Connection, P], Awaitable[None]]


type _Row = Method[Any, Any] | Announced[Any]
"""A `METHODS` value: a method that answers, or the one that does not.

A union rather than a base class both satisfy. `_dispatch` reads `.verb.parse`
and `.handle` off either, which structural attribute access on a union gives
for free — where a shared base would have to declare a `handle` returning
`Awaitable[Any]`, putting back the `Any` the row types exist to remove.
"""


def _command_key(params: MutationParams) -> str:
    """A client's idempotence key, joined at the wire edge and only here.

    Two mutating methods take one now — `session/prompt` and `session/command` —
    and the f-string was written in both, which is what a comment three lines
    above one of them already claimed was not the case."""
    return f"{params.client_id}:{params.command_id}" if params.command_id else ""


CAPABILITIES = (
    "roots",
    "attach",
    "cursors",
    "snapshots",
    "asks",
    "browse",
    "projections",
    "attachments",
    "staging",
    "shell",
)
"""What this transport adds to the two both of them have.

A constant because two callers say it now — `initialize`, and the `daemon/status`
a client runs when it wants to know what it is talking to — and a capability
block that disagreed with itself depending on which method you asked would be
worse than having none.
"""


class DaemonUnavailable(Refusal):
    """This daemon cannot take its socket, and the reason is worth a sentence.

    One type over both startup preconditions — a socket another daemon is
    listening on, and one the kernel will not bind (a `$PH_RUNTIME` past
    `AF_UNIX`'s 107-byte path limit, a directory this user cannot write). The
    CLI caught `(RuntimeError, OSError)` for them, which is two builtins wide
    enough to swallow a `typer.Exit` — `typer.Exit` subclasses `RuntimeError`,
    and `cli.py` already carries a comment about being bitten by exactly that.
    """

    code = "daemon_unavailable"


class AttachmentTooLarge(Refusal):
    """One frame cannot carry this file.

    Its own refusal because the caller's next move is specific and knowable: this
    is not "too big to attach", it is "too big to attach *in one frame*", and the
    limit is named so a client can say so rather than guessing. Chunked upload is
    the fix and it is not built (§5 rule 6) — a browser dropping a 40 MB video
    gets a sentence, not a truncated blob.
    """

    code = "attachment_too_large"


class AttachmentUnknown(Refusal):
    """A prompt referenced an attachment this deployment has never stored.

    Refused rather than dropped, because the alternative is the silent failure
    P7-01 exists to end: a person attaches a diagram, the reference is stale or
    from another machine, and the turn goes out as plain text with nothing saying
    the picture was never sent.
    """

    code = "attachment_unknown"


class UntrustedProject(Refusal):
    """This `cwd` has not been trusted, and mounting it would read its config.

    Its own type because the client's next move is specific: *ask*. A front end
    catching this shows the trust modal and retries with `trust`; one that has
    no person to ask — `ph agents send` naming a new session in a checkout
    nobody has vouched for — stops, which is the point.
    """

    code = "untrusted_project"


class NoSuchSession(Refusal):
    """No root is running under that id — which is not the same as it being
    busy, and a client that wants to start one branches differently."""

    code = "no_such_session"


@dataclass(slots=True, eq=False)
class _Connection:
    """One client, and the roots it is watching.

    `eq=False` so identity is identity: a connection goes into the ask desk's set
    of front ends, and a dataclass's generated `__eq__` takes `__hash__` away with
    it. Two clients are never the same client, whatever their fields hold.

    The framing, the pending table, the write ordering and the in-flight bound
    all live in `Peer`, which the *client* is built from too — see
    `duplex.py` for why one object rather than two.
    """

    stream: ByteStream
    server: DaemonServer
    attached: set[str] = field(default_factory=set)
    declared: frozenset[str] = frozenset()
    """What this *client* said it can do, at `initialize`.

    Capabilities go both ways for the same reason they go one way: a client
    reads the server's block rather than inferring from which socket it opened,
    and the server reads this rather than inferring from which method arrived.
    `asks` is the name in both directions — the daemon offering to ask, and a
    client offering to answer — because it is one feature, and half of it is
    useless without the other half."""
    peer: Peer = field(init=False)

    def __post_init__(self) -> None:
        # Eagerly: `_Connection` is only ever built inside `serve`'s accept loop,
        # which is already in a running event loop — which is what `anyio.Event`
        # needs. A lazy property bought nothing and left `peer` reachable in a
        # half-built state.
        self.peer = Peer(
            stream=self.stream,
            dispatch=self._dispatch,
            dispatch_notifications=True,
            id_prefix="s",
        )

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or *raise* so the root drops this watcher.

        Raising is the point: catching `WouldBlock` here and logging "dropped"
        drops nothing, and the watcher that cannot keep up re-pays the whole
        fan-out for every later event. The subscriber list belongs to the root,
        so the root is what removes from it; this only has to fail loudly enough
        to be noticed.
        """
        self.peer.tell(method, params)

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Put a question to this client and wait for the person behind it."""
        return await self.peer.ask(method, params)

    async def serve(self) -> None:
        try:
            await self.peer.serve()
        finally:
            for root_id in self.attached:
                root = self.server.supervisor.roots.get(root_id)
                if root is not None:
                    root.unsubscribe(self.notify)
                    if root.desk is not None:
                        root.desk.leave(self)
            self.attached.clear()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:  # noqa: ANN401
        """The daemon's half of the vocabulary — dsh's names (P5-02).

        `session/*` rather than P5-01's `root/*`: a supervised root *is* a
        session here, and the dsh client already ships against these names. The
        supervisory additions (`daemon/hello`, `session/attach`) are declared in
        the capability block rather than inferred from which socket answered.

        **Two tables, not a chain (P8-07).** Every method is a row naming its
        `Verb` and its handler — `MUTATIONS` for the ones that change a root
        under an idempotence key, `METHODS` for everything else — so an
        unknown method is a missed lookup rather than the last `else`, the
        vocabulary is a value `test_daemon_methods` holds against the verbs it
        is derived from, and each handler receives its own parsed type. That
        last point is what makes the handlers typeable at all: one function
        reading `params["sessionId"]` under twenty-one `if`s could not give
        each branch a different static type without a cast per branch, which is
        a dict with extra steps.
        """
        mutation = MUTATIONS.get(method)
        if mutation is not None:
            return await self._mutate(mutation, params)
        entry = METHODS.get(method)
        if entry is None:
            raise UnknownMethod(f'unknown method "{method}"')
        return await entry.handle(self, entry.verb.parse(params))

    async def _mutate(
        self,
        mutation: Mutation[Any, Any],
        params: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Every method that changes a root goes through this one wrapper.

        Parse, resolve the root (through `start`, so acting on a passivated one
        brings it back), validate, claim the idempotence key, act. The parse is
        first so a malformed call is refused before a root is mounted for it.
        See `MUTATIONS` for why the key is claimed here and nowhere else.
        """
        parsed = mutation.verb.parse(params)
        root = await self.server.supervisor.start(parsed.session_id)
        plan = await mutation.prepare(self, root, parsed)
        if not root.once(_command_key(parsed)):
            return MutationRepeated(**root.describe().model_dump())
        return await mutation.act(self, root, parsed, plan)

    # --- the methods -----------------------------------------------------------
    # One per row of `METHODS`, each taking its own params model. Read-only
    # projections are not here: `_projection` builds both their handler and
    # their row.

    async def _initialize(self, params: InitializeParams) -> CapabilityBlock:
        # A property of the *client*, so it is said once here rather than
        # per-attach: whether a UI can put a modal in front of a person does
        # not vary by which session it is watching, and a per-attach flag let
        # the same client claim it for one root and not another — two answers
        # to a question with one true answer.
        self.declared = frozenset(params.capabilities)
        # Named, because the failure is otherwise invisible: a client that
        # declares `"ask"` is never joined to any desk, and every question
        # its person should have answered is simply never asked.
        unknown = self.declared - set(CAPABILITIES)
        if unknown:
            log.warning(
                "ph_app.daemon: a client declared %s, which this daemon does not serve; "
                "it serves %s",
                ", ".join(sorted(unknown)),
                ", ".join(CAPABILITIES),
            )
        return capabilities(*CAPABILITIES)

    async def _sessions_list(self, _params: NoParams) -> RootListing:
        return RootListing(sessions=self.server.supervisor.describe())

    async def _new_session(self, params: NewSessionParams) -> RootDescription:
        # `cwd` is the client's, and the daemon is the one that mounts — so
        # it is said here rather than assumed from the daemon's own process,
        # which is somewhere neither the person nor their files are. `or None`
        # because a client that has no directory to name sends `""` or omits it,
        # and both mean the same thing.
        cwd = params.cwd or None
        # The daemon mounts, so the daemon enforces — `ph_app.trust` says
        # why. Only for a session created here: resuming one that exists is
        # not a new decision about a new directory.
        self._check_trust(cwd, params.trust)
        root = await self.server.supervisor.start(params.session_id, cwd=cwd)
        if params.trust == "always" and cwd is not None:
            # After the mount, not before: a directory is only worth
            # recording once its profile has actually composed.
            TrustStore(path=trust_path()).trust(Path(cwd))
        return root.describe()

    async def _cancel(self, params: SessionParams) -> RootDescription:
        # Not a `MUTATIONS` row: cancel is idempotent by construction, and a
        # key would make an honest retry answer `repeated` and leave the turn
        # running. `_root`, not `start`: nothing to stop on a passivated one.
        root = self._root(params.session_id)
        root.agent.cancel(AgentCancelCause(kind="user"), keep_inbox=True)
        return root.describe()

    async def _sessions_browse(self, _params: NoParams) -> SessionBrowse:
        # Daemon-level, not a `PROJECTIONS` row: it is not about one root.
        # Every root mounts the same profile and so the same store, and which
        # roots are held is the supervisor's own answer.
        return SessionBrowse(sessions=browse_of(self.server.supervisor))

    async def _daemon_config(self, _params: NoParams) -> DaemonConfigReply:
        # The composed profile, which is a property of the *daemon* and not
        # of any root: every root mounts the same composition.
        return DaemonConfigReply(rows=list(self.server.supervisor.profile.dump()))

    async def _credentials_held(self, params: SessionParams) -> CredentialsHeldReply:
        # `credentials/held` and `credentials/store`, not `session/credential`
        # and `session/credentials`: those were two names one letter apart for
        # opposite kinds, and the one that writes a secret is the last method
        # that should be easy to reach by typo.
        #
        # No `names`: the daemon composes the profile, so it knows which
        # credentials this deployment names — see `credentials_of`. Asking the
        # client to supply them is what made the whole composed configuration
        # something a front end had to be sent.
        root = self._root(params.session_id)
        return CredentialsHeldReply(
            session_id=root.id, held=credentials_of(root, self.server.supervisor)
        )

    # The schedule seam over the wire (P5-06, P5-10). Create and cancel go
    # through the supervisor rather than the seam directly: both need the
    # root mounted and the append flushed, and a schedule that lives only in
    # a buffer is one a restart forgets.

    async def _schedule_create(self, params: CreateScheduleParams) -> Schedule:
        return await self.server.supervisor.schedule(params.session_id, params.to_schedule())

    async def _schedule_cancel(self, params: CancelScheduleParams) -> ScheduleCancelled:
        cancelled = await self.server.supervisor.unschedule(params.session_id, params.schedule_id)
        return ScheduleCancelled(
            session_id=params.session_id,
            schedule_id=params.schedule_id,
            cancelled=cancelled,
        )

    async def _schedule_list(self, params: SessionParams) -> SessionSchedulesReply:
        root = self._root(params.session_id)
        return SessionSchedulesReply(
            session_id=root.id, schedules=self.server.supervisor.scheduled(root)
        )

    async def _daemon_status(self, _params: NoParams) -> DaemonStatusReply:
        return self.server.status()

    async def _shutdown(self, _params: NoParams) -> None:
        # Actually stops it, and takes no id by contract: a client awaiting
        # a reply would be waiting on a frame the daemon is concurrently
        # losing the ability to write. "Stop" is not a question.
        self.server.stop.set()

    async def _put(self, params: PutAttachmentParams) -> AttachmentStored:
        """Store bytes a client read, and answer with the reference to them.

        **The client reads the file, not the daemon** — I-9's human door. A person
        may attach anything they can already open, and their permissions are the
        only ones that should decide; it is also the only path a browser has,
        since the bytes reach the page before they reach anything else. The
        daemon never learns a path, which is the property that makes this the
        same method for a terminal on this machine and a tab on another.

        Content-addressed, so putting a file twice stores it once and the second
        put is a cheap way to *learn* the reference.

        Not a `MUTATIONS` row on purpose: content-addressed, so a retry is
        already a no-op — and its reply *is* the reference the client came
        for, which a `repeated` envelope would withhold.
        """
        root = self._root(params.session_id)
        store = self._store(root)
        encoded = params.content_b64
        # Checked before decoding, on the encoded length: base64 is 4/3 of the
        # bytes, so decoding first to measure would be the allocation this
        # refusal exists to avoid. Off by the two padding bytes, immaterial here.
        if len(encoded) * 3 // 4 > MAX_ATTACHMENT_BYTES:
            raise AttachmentTooLarge(
                f"an attachment must be at most {MAX_ATTACHMENT_SIZE} "
                "in one frame; chunked upload is not implemented"
            )
        try:
            content = b64decode(encoded, validate=True)
        except BinasciiError as error:
            raise Refusal(f"contentB64 is not valid base64: {error}") from error
        name = params.name or None
        ref = await store.save_bytes(
            content=content,
            # A client that sent no `mime` still sent a name, and `mime_for` is
            # the one ladder both doors climb — an octet-stream default here made
            # a `.png` a document while `--attach` called it an image.
            mime=mime_for(params.mime or "", name or ""),
            name=name,
        )
        return AttachmentStored(session_id=root.id, attachment=ref)

    # --- the mutation halves --------------------------------------------------
    # `prepare` validates and may refuse; nothing it does is an effect. `act` is
    # the effect. The wrapper claims the idempotence key between them, which is
    # what makes a refusal *not* consume a retry: a prompt refused for an unknown
    # attachment, re-sent with a known one under the same key, must act.

    async def _prepare_prompt(self, root: Root, params: PromptParams) -> list[AttachmentRef]:
        return self._attachments(root, params.attachments)

    async def _act_prompt(
        self, root: Root, params: PromptParams, refs: list[AttachmentRef]
    ) -> RootDescription:
        root = await self.server.supervisor.prompt(root.id, params.prompt, attachments=refs)
        return root.describe()

    async def _prepare_stage(self, root: Root, params: StageParams) -> AttachmentRef:
        return self._known(self._store(root), params.attachment)

    async def _act_stage(
        self, root: Root, params: StageParams, ref: AttachmentRef
    ) -> SessionStagedNotice:
        """Put an attachment in the composer's tray, for every attached UI.

        Not appended: un-submitted intent is not an act in the session, which is
        the same rule that keeps a half-typed prompt off the log. Broadcast,
        because the tray is shared — a chip only the uploader can see is a
        composer nobody else can reason about.
        """
        notice = SessionStagedNotice(session_id=root.id, staged=list(root.staged.stage(ref)))
        root.publish(notice)
        return notice

    async def _prepare_command(self, root: Root, params: CommandParams) -> CommandRegistry:
        registry = root.ctx.get(COMMANDS)
        if registry is None:
            raise SeamAbsent("this deployment has no commands")
        return registry

    async def _act_command(
        self, root: Root, params: CommandParams, registry: CommandRegistry
    ) -> CommandShown:
        """Run one `/name argument` line in the root's own context.

        **In the daemon, not the client**, because a command body reaches for
        seams that only exist where the profile is mounted — and because the
        record of having run it belongs in this session's log, where every other
        attached UI will see it.
        """
        shown = await registry.dispatch(
            params.line,
            scope=root.agent.ctx,
            session=root.session,
            agent=root.agent,
        )
        # A command can move a reading without moving the agent — `/sandbox` is
        # the one that does — and nothing else would re-read them. See
        # `_refresh_readings`; it costs one fold, and only when somebody is
        # attached to see it.
        _refresh_readings(root)
        return CommandShown(session_id=root.id, shown=shown)

    async def _prepare_shell(self, root: Root, params: ShellParams) -> tuple[ShellService, str]:
        """Resolve the seam and the command **before** the key is claimed.

        The seam check used to happen inside `act`, which is after `once()` has
        recorded the request — so a `!!` sent to a deployment with no shell burnt
        the client's retry on a refusal. That is precisely what the two halves
        are for.
        """
        command = params.command.strip()
        if not command:
            raise Refusal("a shell command cannot be empty")
        return shell_of(root.ctx), command

    async def _act_shell(
        self, root: Root, params: ShellParams, prepared: tuple[ShellService, str]
    ) -> ShellReply:
        """`!!<command>` — the person's own shell, in the session's workspace.

        The reply is the exit code; the *output* arrives as `shell/command` and
        `shell/result` events, so every attached UI reads it by the one route
        they all already watch and the person who typed it reads it back off the
        same event as everybody else.
        """
        shell, command = prepared
        return await self.server.supervisor.shell(root.id, shell, command)

    async def _prepare_preset(self, root: Root, params: PresetParams) -> PermissionPresetService:
        presets = root.ctx.get(PERMISSION_PRESETS)
        if presets is None:
            raise SeamAbsent("this deployment has no permission presets")
        return presets

    async def _act_preset(
        self, root: Root, params: PresetParams, presets: PermissionPresetService
    ) -> PresetApplied:
        applied = presets.apply_preset(params.preset, session=root.session)
        _refresh_readings(root)
        return PresetApplied(session_id=root.id, preset=applied.name)

    async def _prepare_credential(
        self, root: Root, params: StoreCredentialParams
    ) -> CredentialService:
        service = root.ctx.get(CREDENTIALS)
        if service is None:
            raise SeamAbsent("this deployment stores no credentials")
        return service

    async def _act_credential(
        self, root: Root, params: StoreCredentialParams, service: CredentialService
    ) -> CredentialStored:
        # **The value is used and not kept.** It is never logged, never echoed
        # in the reply, and never reaches `describe()` — the reply is the name
        # and a boolean, which is everything a UI needs to redraw.
        service.provide_value(params.name, params.value)
        return CredentialStored(session_id=root.id, name=params.name)

    def _attachments(self, root: Root, refs: list[AttachmentRef]) -> list[AttachmentRef]:
        """The refs a prompt named, checked against what this deployment holds.

        Refused rather than dropped: a reference from another machine, or to a
        blob a `gc` took, would otherwise send the turn as plain text with
        nothing saying the picture never went.
        """
        if not refs:
            return []
        store = self._store(root)
        return [self._known(store, ref) for ref in refs]

    def _store(self, root: Root) -> AttachmentStore:
        """The attachment store, or the one refusal for its absence.

        One resolution for the three methods that need it, so "this deployment
        stores no attachments" has one author — the two that grew separately had
        opposite rules for a missing store, one staging anything and one refusing
        everything.
        """
        store = root.ctx.get(ATTACHMENTS)
        if store is None:
            raise SeamAbsent("this deployment stores no attachments")
        return store

    def _known(self, store: AttachmentStore, ref: AttachmentRef) -> AttachmentRef:
        """A reference the client sent, checked against what this deployment holds.

        Presence only: the *shape* was the params model's to check, which is why
        a malformed reference is `invalid_params` and this refusal is reserved
        for a well-formed one nobody stored here.
        """
        if not store.exists(ref):
            raise AttachmentUnknown(f"no attachment {ref.attachment_id} is stored here")
        return ref

    def _check_trust(self, cwd: str | None, answer: TrustAnswer) -> None:
        """Refuse a `cwd` nobody has vouched for. `ph_app.trust` says why.

        `"once"` mounts without recording; `"always"` is recorded by the caller,
        after the mount has actually succeeded.
        """
        if cwd is None or answer == "once":
            return
        store = TrustStore(path=trust_path())
        project = Path(cwd)
        if answer == "always":
            # Recorded by the *caller*, once the mount has actually succeeded:
            # writing here would vouch for a directory whose profile then failed
            # to compose, and the next client would walk straight in.
            return
        if not store.trusted(project):
            raise UntrustedProject(f"{cwd} has not been trusted; ask, then send trust")

    def _root(self, session_id: str) -> Root:
        root = self.server.supervisor.roots.get(session_id)
        if root is None:
            raise NoSuchSession(f'no session "{session_id}"')
        return root

    async def _attach(self, params: SessionParams) -> AttachReply:
        """Subscribe to what happens *next*, and say where that starts.

        **Attach does not replay.** Streaming the gap here — one `session.event` frame per
        event into a fixed-size outbox with no await point — makes a client reattaching to
        a root that has moved on get a `WouldBlock` out of its own attach, *after* the
        subscription has been made.

        So catch-up has one mechanism, and it is the paged one: the reply says where the
        live stream begins, and the client reads `session/snapshot` from its cursor up to
        that point. That also makes the 512 KiB-class bound apply to replay.
        """
        # Through `start`, so attaching to a *passivated* root brings it back
        # rather than reporting it gone (P5-05). `start` returns the live root
        # untouched when there is one, and resumes from the log when there is
        # not — the same path `session/prompt` takes, which is what keeps
        # rehydration one mechanism instead of two. There is no cursor to read:
        # attach does not replay, so it does not take one — `SnapshotParams`
        # says why the field was removed rather than accepted and ignored.
        root = await self.server.supervisor.start(params.session_id)
        if root.id not in self.attached:
            self.attached.add(root.id)
            root.subscribe(self.notify)
        # Watching and answering are different claims, and the second is the
        # client's `asks` capability rather than anything about this attach: a
        # follower like `ph agents attach` watches without ever declaring it, and
        # is never asked. See `test_a_watcher_that_is_not_a_front_end_is_never_asked`.
        if "asks" in self.declared and root.desk is not None:
            root.desk.join(self)
        # The footer, with the status it belongs to — the same pairing
        # `session.status` makes, so a client that has just attached draws a
        # complete one without a second call and without assembling a frame
        # shape no wire message has.
        return AttachReply(**root.describe().model_dump(), readings=readings_of(root))

    async def _snapshot(self, params: SnapshotParams) -> SnapshotPage:
        """One bounded page of a session's history, and the cursor for the next."""
        root = self._root(params.session_id)
        start = resume_at(root.session, params.cursor)
        page = root.session.events_from(start, SNAPSHOT_EVENTS)
        tools = root.ctx.get(TOOLS)
        events = [event.to_wire(thaw=False) for event in page]
        # **The same sidecar the relay attaches**, because a client must not see
        # one transcript live and a different one on replay. Keyed by seq and
        # **sparse**, because a page is 2048 events and a turn contributes a
        # handful of cards: the positional list this replaced serialized two
        # thousand `null`s — twelve kilobytes of nothing — on every attach,
        # reconnect and page-forward.
        presentations = {
            str(event.seq): view
            for event in page
            if event.type in CARD_EVENTS
            and (view := presentation_of(tools, root.session, event)) is not None
        }
        return SnapshotPage(
            session_id=root.id,
            events=events,
            presentations=presentations,
            # Where the read actually began, which is not always where the cursor
            # asked: `resume_at` answers a cursor from another incarnation of the
            # log with 0. Sent rather than left to be inferred from the first
            # event's seq — that inference is `None` for an empty page, and
            # `resume_at`'s own docstring already promises the client will be told.
            started_at=start,
            cursor=cursor_of(root.session, start + len(events)),
            more=start + len(events) < root.session.seq,
        )

    async def _status(self, params: SessionParams) -> RootStatusReply:
        """One root in detail — what `sessions/list` says, and why it says it.

        The listing carries what a table needs for every root; this carries what
        a person asks about *one*. The retry ladder is the reason it exists:
        `status` collapses to `"retrying"` or `"failed"`, and the two questions
        that follow — how many attempts, and what is still going to fire — have
        no other way to be asked.
        """
        root = self._root(params.session_id)
        return RootStatusReply(
            **root.detail().model_dump(), schedules=self.server.supervisor.scheduled(root)
        )

    async def _detach(self, params: SessionParams) -> SessionDetached:
        was_attached = params.session_id in self.attached
        self.attached.discard(params.session_id)
        root = self.server.supervisor.roots.get(params.session_id)
        if root is not None:
            root.unsubscribe(self.notify)
        # Deliberately *not* an error when nothing was attached: detach is what a
        # client does while tidying up, often twice, and a teardown path that
        # raises is one nobody can write correctly.
        return SessionDetached(session_id=params.session_id, detached=was_attached)


def _refresh_readings(root: Root) -> None:
    """Re-read the footer and push it, for a change the *agent* did not make.

    Readings ride `session.status`, and the supervisor sends one when the agent
    moves — which is right for a budget or a token count, and wrong for the two
    postures. A preset switch and `/sandbox` change what the footer says without
    the agent doing anything at all, so nothing re-read them: the client redrew
    from the list it was holding and repainted the posture it already had.

    Status left empty on purpose. `StatusFacts` is "as much of it as changed",
    and a reader keeps every field a frame does not state — so this says the
    readings moved and claims nothing about where the agent is.

    Guarded like `Supervisor.announce`'s own publish, and for its reason: a
    reading is a fold over the log, and folding for nobody is the work the
    check exists to skip.
    """
    if root.subscribers:
        root.publish(SessionStatusNotice(session_id=root.id, readings=readings_of(root)))


def _projection[N: SessionNotice](
    verb: Verb[SessionParams, N], key: str, fold: Callable[[Root], list[Any]]
) -> tuple[str, _Row]:
    """One `METHODS` row answering a projection with its verb's reply model.

    Returns the row rather than the handler, so each projection names its verb
    **once**: wrapped in `_reading(...)` at the table it appeared twice on one
    line, which is the shape `_unkeyed`/`_mutating` exist to remove.

    A factory rather than four near-identical handlers because the four differ
    only in a model, a key and a fold — and the closure holds exactly those,
    nothing of the request. (`Method.handle` receives no method name, so a
    single table-reading handler would have to add that parameter to all 22
    handlers in order to serve four.)

    The reply models are the *same types* the notifications use for two of the
    four: a client validates `commands/list`'s reply as a `SessionCommandsNotice`
    because the reply and the notification are one shape. The server was the only
    party that did not say so, and dumped rows by hand at this return.
    """

    async def handle(connection: _Connection, params: SessionParams) -> N:
        root = connection._root(params.session_id)
        # Through the verb, so the model is named once: the row that routes this
        # handler and the handler itself cannot disagree about what it answers.
        # `read` rather than `reply(session_id=…, **{key: …})` because the field
        # name is a parameter here, so a keyword spread is untypeable — and
        # validating is what checks that the fold's rows match the model's
        # declared element type, which is the whole reason the model is named.
        return verb.read({"session_id": root.id, key: fold(root)})

    return _unkeyed(verb, handle)


def _mutating[P: MutationParams, R: WireModel](
    verb: Verb[P, R],
    prepare: Callable[[_Connection, Root, P], Awaitable[Any]],
    act: Callable[[_Connection, Root, P, Any], Awaitable[R]],
) -> tuple[str, Mutation[Any, Any]]:
    """One `MUTATIONS` row, keyed by its verb's name and **checked against it**.

    A function rather than `{verb.name: Mutation(verb, ...)}` written inline,
    and the reason is worth stating because the inline form looks identical and
    checks nothing: the table's own annotation is the expected type of each
    value, so `dict[str, Mutation[Any, Any]]` solves `P` and `R` to `Any` and
    every handler fits every verb. Sabotage-checked — pairing `session/status`
    with `_sessions_list` passed `mypy --strict` silently. Here the return type
    names no type variable, so `P` and `R` can only come from the verb, and the
    two halves are checked against it.
    """
    return verb.name, Mutation(verb, prepare, act)


def _unkeyed[P: WireModel, R: WireModel](
    verb: Verb[P, R], handle: Callable[[_Connection, P], Awaitable[R]]
) -> tuple[str, _Row]:
    """One `METHODS` row, checked against its verb. `_mutating` says why.

    Named for the absence of an idempotence key rather than for reading:
    `session/new`, `session/cancel`, `attachment/put` and both schedule verbs
    all write. `verbs.UNKEYED` carries the same correction.
    """
    return verb.name, Method(verb, handle)


def _announcing[P: WireModel](
    verb: Notify[P], handle: Callable[[_Connection, P], Awaitable[None]]
) -> tuple[str, _Row]:
    """One `METHODS` row for a method with no reply. `_mutating` says why a
    builder and not a literal; `Announced` says why a separate row type."""
    return verb.name, Announced(verb, handle)


MUTATIONS: dict[str, Mutation[Any, Any]] = dict(
    (
        _mutating(verbs.SESSION_PROMPT, _Connection._prepare_prompt, _Connection._act_prompt),
        _mutating(verbs.SESSION_COMMAND, _Connection._prepare_command, _Connection._act_command),
        _mutating(verbs.SESSION_STAGE, _Connection._prepare_stage, _Connection._act_stage),
        _mutating(verbs.SESSION_SHELL, _Connection._prepare_shell, _Connection._act_shell),
        _mutating(verbs.SESSION_PRESET, _Connection._prepare_preset, _Connection._act_preset),
        _mutating(
            verbs.CREDENTIALS_STORE,
            _Connection._prepare_credential,
            _Connection._act_credential,
        ),
    )
)
"""Every method that changes a root, and the one place their idempotence lives.

A client's `clientId`/`commandId` names a request; a reconnecting client that
cannot know whether its last one landed re-sends it, and the same key must not
run the effect twice. That rule used to be applied per handler by memory — two
handlers spelled it, the third forgot — so it is applied here by construction:
a method in this table gets the write-ahead guard whether its author thought of
it or not, and one that is not in it cannot claim to be idempotent by key.

The repeat reply has one shape — `MutationRepeated`, a description carrying
`repeated: true` — so a client branches on one field for every verb.
Deliberately absent: `attachment/put`
(content-addressed, so a retry is a no-op already, and its reply *is* the
reference a repeat must still return) and `session/new` (`start` is idempotent
by id).
"""

METHODS: dict[str, _Row] = dict(
    (
        _unkeyed(verbs.INITIALIZE, _Connection._initialize),
        _unkeyed(verbs.DAEMON_HELLO, _Connection._initialize),
        _unkeyed(verbs.SESSIONS_LIST, _Connection._sessions_list),
        _unkeyed(verbs.SESSIONS_BROWSE, _Connection._sessions_browse),
        _unkeyed(verbs.SESSION_NEW, _Connection._new_session),
        _unkeyed(verbs.SESSION_ATTACH, _Connection._attach),
        _unkeyed(verbs.SESSION_DETACH, _Connection._detach),
        _unkeyed(verbs.SESSION_STATUS, _Connection._status),
        _unkeyed(verbs.SESSION_CANCEL, _Connection._cancel),
        _unkeyed(verbs.SESSION_SNAPSHOT, _Connection._snapshot),
        _unkeyed(verbs.ATTACHMENT_PUT, _Connection._put),
        _unkeyed(verbs.CREDENTIALS_HELD, _Connection._credentials_held),
        # The read-only projections (P5-14). What a front end used to read straight
        # off `ctx`; each is a fold computed now, so a reconnecting client gets
        # today's answer rather than one cached when somebody last wrote it down.
        # See `projections.py`. Rows rather than a third table spliced in: they are
        # `METHODS` rows like any other, and the table they came from was read
        # exactly once, to build this one.
        _projection(verbs.SESSION_READINGS, "readings", readings_of),
        _projection(verbs.COMMANDS_LIST, "commands", commands_of),
        _projection(verbs.SCREENS_LIST, "screens", screens_of),
        _projection(verbs.TOOLS_LIST, "tools", tools_of),
        _projection(verbs.SKILLS_LIST, "skills", skills_of),
        _projection(verbs.PRESETS_LIST, "presets", presets_of),
        _unkeyed(verbs.SCHEDULE_CREATE, _Connection._schedule_create),
        _unkeyed(verbs.SCHEDULE_CANCEL, _Connection._schedule_cancel),
        _unkeyed(verbs.SCHEDULE_LIST, _Connection._schedule_list),
        _unkeyed(verbs.DAEMON_CONFIG, _Connection._daemon_config),
        _unkeyed(verbs.DAEMON_STATUS, _Connection._daemon_status),
        _announcing(verbs.SHUTDOWN, _Connection._shutdown),
    )
)
"""Every method that is not a mutation: what it takes, and what answers it.

Disjoint from `MUTATIONS` by construction — a name in both would be dispatched as
a mutation and the row here never reached — and the two together are the
daemon's whole vocabulary, which `test_daemon_methods` holds against
`verbs.VOCABULARY`: that set is derived from the verb declarations, so what the
check catches is a verb declared and never routed. `initialize` and
`daemon/hello` are two names for one handler because the dsh SDK says the first
and P5-01 said the second.
"""


@dataclass(slots=True)
class DaemonServer:
    """The supervisor behind the socket, and the event that ends the run."""

    supervisor: Supervisor
    stop: anyio.Event
    path: Path
    """The socket this is answering on, so `daemon/status` can say so.

    A client resolves the same path to connect, but "which socket am I actually
    talking to" is the first question anyone debugging two daemons asks, and an
    answer derived a second time on the client side would agree with the server
    by assumption rather than by evidence."""
    ephemeral: bool = False
    """Whether this daemon should exit once nothing needs it (P7-08).

    **Decided by who started it, and that is the whole rule.** `ph daemon` typed
    at a prompt is a service: somebody chose to run a supervisor, and a service
    that exits when idle is a service that is not there when the next client
    arrives. One a UI spawned because the socket was absent was nobody's
    decision, and leaving a process resident on a person's machine after they
    closed the thing that started it is the kind of accretion nobody attributes
    to the right cause a week later.

    A flag rather than a subclass or a second `serve`: every other behaviour is
    identical, and the difference is one predicate on a cadence that already
    runs."""
    open_connections: int = 0
    """How many clients are connected right now — *connected*, not attached.

    The exit predicate reads this, and the distinction is deliberate: a
    `ph agents doctor` mid-call has no subscription and no root, so an
    attachment-based count would hang up on it between the request and the
    reply. Counted rather than a set of connections, because the only question
    asked of it is whether it is zero."""
    tick_every: float = TICK_EVERY
    sweep_every: float = SWEEP_EVERY
    heartbeat_every: float = HEARTBEAT_EVERY
    watch_every: float = WATCH_EVERY
    """The four cadences, named rather than a tuple: `serve` already threads
    them past each other positionally into `start_soon`, and this is the one
    place they are read back by a person."""
    started: int = field(default_factory=now_ms)
    """When this daemon came up, for the uptime a status reply carries."""
    identity: tuple[int, int] | None = None
    """`(st_dev, st_ino)` of the socket at bind time — what `watch` compares against.

    Captured by `serve` rather than read here, because "the socket I bound" is a
    fact about a moment and this object outlives it: reading the path on first
    use would adopt whatever is there by then, which is the exact substitution
    the watch exists to catch."""
    unreachable_since: int | None = None
    """When the socket stopped being this daemon's, or `None` while it still is.

    A latch, not a sample. The transition is one-way by construction — an
    unlinked socket's inode does not come back, and a path re-created by anyone
    else is somebody else's — so this is set once, announced once, and read
    thereafter by `status` for whoever eventually gets to ask."""

    def status(self) -> DaemonStatusReply:
        """What this daemon is, for `ph agents doctor`.

        Everything here is read from the running process rather than re-derived
        by the client: the socket it bound, the policy it was started with, the
        cadences it is actually running. A doctor that reported what the *client*
        would have chosen would agree with a daemon started differently and say
        nothing at all.
        """
        supervisor = self.supervisor
        return DaemonStatusReply(
            protocol_version=PROTOCOL_VERSION,
            capabilities=served(*CAPABILITIES),
            pid=os.getpid(),
            socket=str(self.path),
            uptime_ms=now_ms() - self.started,
            roots=len(supervisor.roots),
            provider=supervisor.provider,
            model=supervisor.model,
            passivate_after=supervisor.passivate_after,
            tick_every=self.tick_every,
            sweep_every=self.sweep_every,
            heartbeat_every=self.heartbeat_every,
            watch_every=self.watch_every,
            unreachable_since=self.unreachable_since,
            # `report()`'s shape as models — a list of sections, each a title
            # and `(label, value)` rows — carrying one built-in section today
            # (P5-11's socket lifetime). One encoding of one fact, so nothing on
            # the wire can disagree with itself and the client needs no bespoke
            # decoder; P5-12 fills the same envelope from the daemon's *mounted*
            # registry with no client change.
            sections=[
                DiagnosticSection(
                    title=title,
                    rows=[DiagnosticRow(label=label, value=value) for label, value in rows],
                )
                for title, rows in self.report()
            ],
        )

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The daemon's own diagnostic sections, in `report()`'s shape.

        A list because there is already more than one, and P5-12's arrival is
        what the shape was built for: `sections` reached the client through one
        loop, so the isolation section cost a tuple entry here and nothing at
        all on the other side.
        """
        return [
            ("socket lifetime", self.socket_lifetime().describe()),
            # The daemon's own non-guarantees (N5, I-2), printed by the command a
            # person runs to ask what this daemon is. Rule 6 wants them beside
            # where they would be assumed, and this reply *is* that place: it
            # says "roots: 7" two rows up, which is the sentence that invites
            # every assumption these rows correct.
            ("isolation", list(NON_GUARANTEES)),
        ]

    def socket_lifetime(self) -> RuntimeLifetime:
        """Whether *this* socket survives logout — asked of the path it bound.

        `serve()` takes an explicit path, so the socket a daemon is answering on
        and the one `resolve_roots()` derives are not always the same file. A
        daemon that reported the lifetime of a directory it is not using would
        be the client-side re-derivation `daemon/status` exists to prevent,
        moved inside the server.

        Read per call rather than captured at start: lingering can be enabled
        while a daemon runs, and a doctor that answered from a snapshot taken at
        boot would keep telling a person to run the command they just ran.
        """
        return lifetime(self.path)

    async def check_reachable(self) -> str:
        """One watch pass: is the socket at our path still ours? (P5-11)

        `""` when it is. Two shapes of no, wanting the same record and not the same
        sentence: `removed` is logout reaping `$XDG_RUNTIME_DIR` out from under a daemon
        that keeps running, and `replaced` is what happens next — the person logs back in,
        `ph daemon` binds a *new* socket at the same path, and two supervisors now believe
        they own this user's roots. An existence check reads the second as a recovery,
        which is why the identity is a `(dev, inode)` pair rather than a boolean.

        **Deliberately not a shutdown.** The roots keep working — their tasks hold no
        reference to a connection, which is P5-01's whole inversion — and ending an hour
        of in-flight work over a socket problem would be this row's own failure mode
        arriving from the other side.
        """
        if self.identity is None or self.unreachable_since is not None:
            # Nothing to compare against (a caller that built this by hand), or
            # already latched. Either way there is no transition to find, and the
            # `lstat` below is skipped for the life of the process.
            return ""
        current = socket_identity(self.path)
        if current == self.identity:
            return ""
        reason = "removed" if current is None else "replaced"
        life = self.socket_lifetime()
        self.unreachable_since = now_ms()
        note = {
            "reason": reason,
            "socket": str(self.path),
            # Which daemon, and which incident. Ten roots get ten records with
            # ten `Session`-assigned timestamps, and without these the only way
            # to ask "were these two sessions in the same incident" afterwards is
            # to correlate on clock times and a payload that happens to be equal.
            "pid": os.getpid(),
            "since": self.unreachable_since,
            "tier": life.tier,
            "linger": life.linger,
            "advice": life.advice,
        }
        # `error`, not `warning`: for a daemon started from a terminal this line
        # is the only place the news lands at the moment it happens, and it is
        # the difference between "the agents stopped answering" and a sentence
        # naming the command that prevents it next time.
        log.error(
            "ph_app.daemon: %s is no longer this daemon's socket (%s) — "
            "clients cannot reach %d root(s); %s",
            self.path,
            reason,
            len(self.supervisor.roots),
            life.advice or "the roots keep running",
        )
        await self.supervisor.announce_unreachable(note)
        return reason

    def spent(self, *, now: int | None = None) -> bool:
        """Whether there is anything left for this daemon to be up for (P7-08).

        Auto-started, nobody connected, and nothing the supervisor holds is
        wanted. **Connected, not attached**: see `open_connections`. And the
        supervisor's half asks P5-05's own `passivatable` with a window of its own
        (`EPHEMERAL_QUIET`) rather than waiting for the sweep to empty `roots` —
        the obvious version quietly made the exit depend on `--passivate-after`,
        so `off` pinned an ephemeral daemon forever and `90` held it ninety
        minutes, neither of which is what either flag says.

        A predicate, with a `now`, for the reason `passivatable` is: a test asks
        the question rather than waiting out a cadence.
        """
        if not self.ephemeral or self.open_connections:
            return False
        stamp = now if now is not None else now_ms()
        return self.supervisor.unwanted(now=stamp, after=EPHEMERAL_QUIET)

    async def sweep(self) -> list[str]:
        """The passivation sweep, and then — if this daemon is spent — the exit.

        One cadence rather than a fourth timer: both halves ask "is anyone still
        using this", one about a root and one about the process, and a second
        timer for the second question would be a second answer to when to ask it.

        The sweep runs first, and no longer because the exit depends on it — see
        `spent` — but because releasing a root this daemon is about to stop
        anyway is what flushes its log and drops its lease on the ordinary path,
        rather than in teardown's shielded window.

        Teardown already unlinks the socket, so the next client sees an *absent*
        one and starts a daemon of its own rather than hitting the
        present-but-refusing diagnosis, which is the aftermath of a crash and
        says something quite different.
        """
        released = await self.supervisor.sweep()
        if self.spent():
            log.info("ph_app.daemon: nothing left to serve; stopping (auto-started)")
            self.stop.set()
        return released

    async def _handle(self, stream: ByteStream) -> None:
        self.open_connections += 1
        try:
            async with stream:
                await _Connection(stream=stream, server=self).serve()
        finally:
            # In `finally`, because the count is a claim on the *process* now: a
            # connection that ended by crashing and was never subtracted would
            # keep an ephemeral daemon resident forever, and the symptom — a
            # daemon that will not leave — points nowhere near here.
            self.open_connections -= 1


async def _every(
    seconds: float, stop: anyio.Event, work: Callable[..., Awaitable[Any]], what: str
) -> None:
    """Run `work` on a fixed cadence until the daemon stops.

    **One reading of `stop`, not three**: the timeout falls through to the work
    and a set event returns, so the loop condition and a trailing guard cannot
    disagree about what "stopped" means. That reasoning was written once and
    then depended on twice — the sweeper and the ticker were the same seven
    lines with different bodies — so any change to shutdown semantics needed
    two edits and nothing would have noticed one of them being missed.

    A failing pass is logged and the cadence continues: work that raised would
    otherwise take the task group with it, and with it every root, over a
    housekeeping pass. That is the reasoning `_drive` contains a crash for.
    """
    while True:
        with anyio.move_on_after(seconds):
            await stop.wait()
            return
        try:
            await work()
        except Exception:
            log.exception("ph_app.daemon: %s failed", what)


async def _clear_stale(path: Path) -> None:
    """Remove a socket nobody is listening on; refuse one somebody is.

    A stale path is the ordinary aftermath of a crash and makes every client
    hang on a connect that is never answered. A *live* one is another daemon,
    and taking its socket would leave two supervisors both believing they own
    this user's roots — which is I-5's question and P5-03's to answer, so here
    it is a refusal rather than a race.
    """
    if not path.exists():
        return
    if await listening(path):
        raise DaemonUnavailable(f"a daemon is already listening on {path}")
    path.unlink(missing_ok=True)


async def serve(
    profile: Profile,
    *,
    provider: str = "fake",
    model: str = "fake-1",
    passivate_after: float | None = PASSIVATE_AFTER,
    wake_within: float | None = WAKE_WITHIN,
    ephemeral: bool = False,
    sweep_every: float = SWEEP_EVERY,
    tick_every: float = TICK_EVERY,
    heartbeat_every: float = HEARTBEAT_EVERY,
    watch_every: float = WATCH_EVERY,
    path: Path | None = None,
    ready: anyio.Event | None = None,
    started: Callable[[DaemonServer], None] | None = None,
) -> None:
    """Run the supervisor until `shutdown`.

    `ready` is an `anyio.Event` set once the socket is accepting, so a caller —
    a test, or `ph agents` starting a daemon on demand — can wait for the door
    to open rather than poll for the file to appear. The file exists before it
    is listening, which is exactly the window a poll would land in.
    """
    socket_path = path or resolve_roots().ensure().daemon_socket()
    await _clear_stale(socket_path)
    # Bound *before* the task group, beside the stale check it belongs with:
    # binding is a precondition, and a precondition that fails inside a group
    # comes back wrapped in an `ExceptionGroup` that `ph daemon`'s `except`
    # cannot see. A `$PH_RUNTIME` deep enough to exceed `AF_UNIX`'s 107-byte
    # path limit printed a full traceback for exactly that reason. Nothing has
    # been built yet at this point, so there is nothing for the teardown below
    # to have cleaned up either.
    try:
        listener = await anyio.create_unix_listener(socket_path)
        # The socket carries every command this user's agents will take, so it
        # is theirs alone — the same reasoning `$PH_RUNTIME` is 0o700 for.
        os.chmod(socket_path, 0o600)
    except OSError as error:
        raise DaemonUnavailable(f"cannot listen on {socket_path}: {error}") from error
    async with anyio.create_task_group() as tasks:
        # Built inside the group so `tasks` is a required field rather than an
        # Optional with a "not serving" guard: a supervisor that cannot start a
        # root is a state that should not be representable.
        supervisor = Supervisor(
            profile=profile,
            tasks=tasks,
            provider=provider,
            model=model,
            passivate_after=passivate_after,
            wake_within=wake_within,
        )
        try:
            server = DaemonServer(
                supervisor=supervisor,
                stop=anyio.Event(),
                path=socket_path,
                tick_every=tick_every,
                sweep_every=sweep_every,
                heartbeat_every=heartbeat_every,
                watch_every=watch_every,
                ephemeral=ephemeral,
                # Taken here, immediately after the bind and before anything can
                # have replaced it — the one moment at which "the socket at this
                # path" and "the socket this daemon is listening on" are the
                # same file by construction rather than by assumption (P5-11).
                identity=socket_identity(socket_path),
            )
            # Four cadences, four tasks, one primitive: a cadence riding another's
            # counter advances only when that one *succeeds*, so a run of failing
            # ticks would starve an unrelated record.
            if passivate_after is not None or ephemeral:
                # `server.sweep`, not `supervisor.sweep`: the pass now ends with
                # "and is there anything left to be up for", and the answer can
                # set `stop` — which belongs to the server. Started for an
                # ephemeral daemon even with passivation off, because the two
                # questions are separate: `--passivate-after off` says keep the
                # roots, not stay resident after the last one is gone.
                tasks.start_soon(_every, sweep_every, server.stop, server.sweep, "the sweep")
            if tick_every > 0:
                # Woken before fired, and on the tick's own cadence rather than
                # once at boot: a session can gain an appointment at any moment
                # from a `ph -p` run in another process, and a root the sweeper
                # released is unmounted again by the time its next one is due
                # (P6-23). One small file read per pass buys both.
                await supervisor.rehydrate()
                tasks.start_soon(
                    _every, tick_every, server.stop, supervisor.wake_and_tick, "the tick"
                )
                tasks.start_soon(
                    _every, heartbeat_every, server.stop, supervisor.heartbeat, "the heartbeat"
                )
            if watch_every > 0:
                # Its own cadence and its own `if`, not a rider on the tick: a test
                # that turns the scheduler off to keep a timer out of its assertions
                # must not thereby turn off the thing that notices the daemon has no
                # door.
                tasks.start_soon(
                    _every, watch_every, server.stop, server.check_reachable, "the socket watch"
                )
            if started is not None:
                # Handed out rather than reachable through the socket: a test
                # whose subject is the supervisor's own concurrency has no wire
                # question to ask, and reaching it through a client would be
                # testing the transport to get at something behind it.
                started(server)
            async with listener:
                tasks.start_soon(listener.serve, server._handle)
                if ready is not None:
                    ready.set()
                await server.stop.wait()
        finally:
            # Shielded, and bounded. `shutdown` is a notification — the caller
            # does not wait for it — so a cancel from whoever started `serve()`
            # routinely arrives *while* this is unwinding, and an unwinding cut
            # short here loses everything teardown is for: sessions unflushed,
            # worktrees unreclaimed (F6), and P5-03's leases never released, so
            # the next daemon refuses a session whose holder is already gone.
            # `move_on_after` rather than a bare shield, because a root that
            # will not unwind must not become a process that will not exit.
            # `GRACE_SECONDS` rather than a number of its own: it is the same
            # budget `install_lifecycle` spends on `ctx.dispose()`, with the
            # same `move_on_after(shield=True)`, and two constants for one
            # tunable is how an inner budget silently becomes dead code (when
            # it exceeds the outer one) or the only one that ever fires.
            with anyio.move_on_after(GRACE_SECONDS, shield=True):
                await supervisor.aclose()
            socket_path.unlink(missing_ok=True)
            # Last: the accept loop and any root task still in flight. Roots are
            # unwound above by their own channels closing, so this cancels a
            # listener rather than a turn.
            tasks.cancel_scope.cancel()
