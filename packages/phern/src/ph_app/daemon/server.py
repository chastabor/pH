"""`$PH_RUNTIME/daemon.sock`: the socket where clients reach the supervisor (P5-01).

The daemon listens on a unix socket and not on stdio. Stdio has exactly one peer
and ends with that peer, and a daemon exists so that clients can reconnect to
it. The method names extend the names of `--mode rpc`. They do not start a
second vocabulary.

**The socket is state, and stale state misleads clients.** A crashed daemon
leaves its path behind, and every client then hangs on a connect that is never
answered. Binding therefore removes a socket that does not respond, and
**refuses a socket that responds**. A socket that responds is another daemon.
The lease of P5-03 decides between two daemons, and this row must not overwrite
the other daemon.

P5-11 is the same rule from the other side: a path that is no longer the socket
of *this* daemon misleads clients about this daemon. `$PH_RUNTIME` is under
`$XDG_RUNTIME_DIR`, and logind removes that directory at logout for a user who
is not lingering. The socket file can therefore be removed while the process
behind it keeps running. Every later client is then told "no daemon socket" and
told to start a daemon, and the leases that the first daemon still holds refuse
that start. `keep_watching` runs whenever the kernel reports that the entry of
the socket moved. It compares the inode of the socket with the inode that was
bound here (`check_reachable`), and it records a change in the log of each
root. It uses the logs because, by then, the surfaces that could carry the news
are exactly the ones that are gone.

@module ph_app.daemon.server
"""

from __future__ import annotations

import logging
import os
import signal
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from anyio.abc import (
    ByteStream,
    SocketAttribute,
    SocketStream,
    TaskStatus,
)
from anyio.streams.stapled import MultiListener

from ph.agent.types import AgentCancelCause
from ph.cordis import Profile
from ph.framing import listening
from ph.keys import ATTACHMENTS, COMMANDS, CREDENTIALS, PERMISSION_PRESETS, TOOLS
from ph.lingering import RuntimeLifetime, lifetime, socket_identity
from ph.llm.types import AttachmentRef
from ph.path_watch import EntryWatch, Mechanism, WatchUnavailable
from ph.paths import make_directories, resolve_roots
from ph.persistence.lease import lease_directory
from ph.resources import GRACE_SECONDS, stop_on_signals
from ph.seams.attachments import mime_for
from ph.seams.models import ModelChoice, ModelEntry
from ph.seams.schedule import Schedule
from ph.seams.shell import ShellService
from ph.session import Prior, intents_of, now_ms, session_written
from ph.session_profile import ProfileBase
from ph.text import duration
from ph.wire import WireModel

from .. import verbs
from ..kinds import CLIENT_COMMAND, command_settled
from ..params import (
    AdoptParams,
    BrowseParams,
    CancelScheduleParams,
    CommandParams,
    CreateScheduleParams,
    InitializeParams,
    ModelParams,
    MutationParams,
    NewSessionParams,
    OwedParams,
    PresetParams,
    ProbeParams,
    PromptParams,
    PutAttachmentParams,
    ResumeParams,
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
    DaemonLifetime,
    DaemonMode,
    DaemonStatusReply,
    DiagnosticRow,
    DiagnosticSection,
    Hold,
    MutationRepeated,
    NewSessionReply,
    OwedListing,
    PresetApplied,
    ProbeReply,
    RepeatOutcome,
    ResumeReport,
    RootDescription,
    RootListing,
    RootStatusReply,
    ScheduleCanceled,
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
    MethodResult,
    NoParams,
    Notify,
    ProbeName,
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
from .launch import SPAWN_TIMEOUT
from .notify import notify
from .planner import Planner
from .probes import (
    HEARTBEAT_SECONDS,
    SCHEDULER_GRACE_SECONDS,
    STALE_HEARTBEATS,
    answer_probe,
    keep_heartbeat,
    parse_bind,
)
from .projections import (
    browse_of,
    commands_of,
    credentials_of,
    family_rows,
    models_of,
    presets_of,
    readings_of,
    screens_of,
    skills_of,
    tools_of,
)
from .recovery import PASSIVATE_AFTER, WAKE_WITHIN
from .supervisor import NON_GUARANTEES, Root, Supervisor, model_for

if TYPE_CHECKING:
    from ph.seams.attachments import AttachmentStore
    from ph.seams.commands import CommandRegistry
    from ph.seams.permission_presets import PermissionPresetService

__all__ = ["DaemonServer", "serve"]

log = logging.getLogger("ph_app.daemon")


@dataclass(frozen=True, slots=True)
class Mutation[P: MutationParams, R: WireModel]:
    """One method that changes a root: what it takes, how to validate it, how to do it.

    The method has two halves and not one body, and the idempotence key is
    claimed *between* them. The daemon claims the key after `prepare` has had
    its chance to refuse, and before `act` has had its chance to do anything. A
    refusal therefore never uses up a retry, and a crash never loses one. Both
    halves receive the connection, the root and the **parsed** params. `prepare`
    passes to `act` whatever it resolved, so a seam is looked up one time and a
    refusal happens before any record is written.

    The `Verb` is the identity of the row. It carries the name, the params model
    that the wire dict is checked against, and the reply. The wrapper makes that
    check first, before `start` resolves a root, so a malformed call never
    mounts anything. The class is generic in both types so that each half is
    typed against its own method: `_prepare_prompt` reads `.attachments` and
    `_act_prompt` reads `.prompt`. A half that reads a field that its method
    does not carry is a type error, and not a `KeyError` at the first real call.

    **`act` returns the reply of the verb, and no longer `Any`** (issue 74). It
    was `Awaitable[Any]` because a verb answered with whatever the type of its
    reply was: a model where the reply had one, and a dict where the reply was a
    per-method shape. The dicts are gone and every reply is a model, so the
    return type can be checked. An `act` that answers with the model of another
    verb is now a type error here, and not a `ValidationError` in a client at
    the first real call.
    """

    verb: Verb[P, R]
    prepare: Callable[[_Connection, Root, P], Awaitable[Any]]
    """`Any` is the *plan*, not the reply: it is what `prepare` resolved and passes
    to `act`. The two halves of a pair agree on it because they are written
    together, and no other code ever sees it."""
    act: Callable[[_Connection, Root, P, Any], Awaitable[R]]
    after: Callable[[_Connection, Root, R | None], Awaitable[R | None]] | None = None
    """A step that the act leaves until it is on disk. The step also decides the
    reply.

    The step runs after the flush that `_mutate` makes. A step that unwinds the
    root (a restart) therefore never races the records of the act, the key and
    the settle of the key. The step runs **on the error path too**, with no
    reply, so that a request that the act made before it failed does not stay on
    the root for an unrelated mutation to find. The step returns the reply to
    send, or `None` to send the reply of the act. Only `session/command` has
    this step. A command that moved the environment of the root past what a live
    mount can follow (`/profile use`) asks for the root to start again from its
    log."""


@dataclass(frozen=True, slots=True)
class Method[P: WireModel, R: WireModel]:
    """One method that reads, or that acts without an idempotence key.

    It has the shape of `Mutation` without the two halves, and for the same
    reason. A row names its verb beside its handler, so the handler is typed
    against both what the wire may carry and what the handler must answer with.
    The *set* of methods is also a value that a test can compare with the
    vocabulary. `HANDLERS` in `tui/adapter.py` makes the same argument about
    event types.
    """

    verb: Verb[P, R]
    handle: Callable[[_Connection, P], Awaitable[R]]


@dataclass(frozen=True, slots=True)
class Announced[P: WireModel]:
    """One method that answers **nothing**. `shutdown` is the only such method.

    This is a sibling of `Method`, and not a `Method` whose `R` is a model that
    nobody reads, because that model was the problem. `ShutdownAck` existed to
    give the verb a reply type and the handler something to return. `respond`
    computed a frame from it and then dropped the frame, because the id is
    `None`.

    The handler returns `None`, and `respond` sends nothing. The *type* can
    therefore carry the contract that "stop" is not a question. See
    `ph_app.protocol.Notify`.

    Rows of this shape are in `METHODS` beside the others, and not in a third
    table. `_dispatch` reads `.verb.parse` and `.handle` from either row type,
    so one lookup still answers the question "is this a method that the daemon
    knows".
    """

    verb: Notify[P]
    handle: Callable[[_Connection, P], Awaitable[None]]


type _Row = Method[Any, Any] | Announced[Any]
"""A `METHODS` value: a method that answers, or the one method that does not.

This is a union, and not a base class that both satisfy. `_dispatch` reads
`.verb.parse` and `.handle` from either, and structural attribute access on a
union allows that with no more code. A shared base would have to declare a
`handle` that returns `Awaitable[Any]`, which puts back the `Any` that the row
types exist to remove.
"""


def _repeat_outcome(prior: Prior) -> RepeatOutcome:
    """`settled` only for a verb whose act was seen to finish. A verb that is still
    acting in this process is `unknown`, and so is a verb that repair or a
    raising act closed. The reply says `unknown` explicitly."""
    return "settled" if prior.outcome == "done" else "unknown"


def _command_key(params: MutationParams) -> str:
    """The idempotence key of a client, joined at the wire edge and only here.

    Every method in `MUTATIONS` takes one. Earlier the f-string was written in
    two handlers, `session/prompt` and `session/command`, although a comment
    three lines above one of them claimed that this was not the case."""
    return f"{params.client_id}:{params.command_id}" if params.command_id else ""


CAPABILITIES = (
    "roots",
    "attach",
    "cursors",
    "snapshots",
    "asks",
    "browse",
    "browse-cwd",
    "projections",
    "attachments",
    "staging",
    "shell",
)
"""The capabilities that this transport adds to the two that both transports have
(`sessions` and `streaming`).

This is a constant because two callers now use it: `initialize`, and the
`daemon/status` that a client runs when it wants to know what it is talking to.
A capability block that gave a different answer to each method would be worse
than no block.
"""


class DaemonUnavailable(Refusal):
    """This daemon cannot take its socket. The message gives the reason.

    One type covers every startup precondition:

    - A socket that another daemon is listening on.
    - A socket that the kernel does not bind: a `$PH_RUNTIME` longer than the
      107-byte path limit of `AF_UNIX`, or a directory that this user cannot
      write.
    - A probe address that cannot be bound (`--probe-bind`).

    Earlier the CLI caught `(RuntimeError, OSError)` for the two socket cases.
    Those two builtins are wide enough to catch a `typer.Exit`, because
    `typer.Exit` subclasses `RuntimeError`. `cli.py` already has a comment about
    exactly that bug.
    """

    code = "daemon_unavailable"


class AttachmentTooLarge(Refusal):
    """One frame cannot carry this file.

    This is a separate refusal because the next step of the caller is specific
    and known. The refusal does not mean "too big to attach". It means "too big
    to attach *in one frame*". The message names the limit, so a client can
    report it and does not have to guess. The fix is chunked upload, which is
    not built (§5 rule 6). A browser that drops a 40 MB video gets a sentence,
    and not a truncated blob.
    """

    code = "attachment_too_large"


class AttachmentUnknown(Refusal):
    """A prompt referenced an attachment that this deployment has never stored.

    The prompt is refused, and the reference is not dropped. To drop it is the
    silent failure that P7-01 exists to end: a user attaches a diagram, the
    reference is stale or comes from another machine, and the turn goes out as
    plain text with nothing to say that the picture was never sent.
    """

    code = "attachment_unknown"


class UntrustedProject(Refusal):
    """This `cwd` has not been trusted, and mounting it would read its config.

    This is a separate type because the next step of the client is specific:
    *ask*. A front end that catches this shows the trust modal and retries with
    `trust`. A client that has no user to ask stops, which is the purpose. An
    example is `phern agents send` when it names a new session in a checkout
    that nobody has vouched for.
    """

    code = "untrusted_project"


class NoSuchSession(Refusal):
    """No root is running under that id. That is not the same as a busy root, and
    a client that wants to start a root takes a different branch."""

    code = "no_such_session"


@dataclass(slots=True, eq=False)
class _Connection:
    """One client, and the roots that it watches.

    `eq=False` keeps identity as identity. A connection goes into the set of
    front ends of the ask desk, and the `__eq__` that a dataclass generates
    removes `__hash__`. Two clients are never the same client, whatever their
    fields hold.

    The framing, the pending table, the write ordering and the in-flight bound
    are all in `Peer`. The *client* is built from `Peer` too. See `duplex.py`
    for the reason why there is one object and not two.
    """

    stream: ByteStream
    server: DaemonServer
    attached: set[str] = field(default_factory=set)
    declared: frozenset[str] = frozenset()
    """What this *client* said that it can do, at `initialize`.

    Capabilities go in both directions for the same reason that they go in one
    direction. A client reads the block of the server and does not infer from
    which socket it opened. The server reads this field and does not infer from
    which method arrived. `asks` is the name in both directions: the daemon
    offers to ask, and a client offers to answer. It is one feature, and each
    half is useless without the other half."""
    spoke: bool = False
    """Whether this connection has ever sent a frame. A connection that has sent
    one is a client. A connection that has not is only a knock.

    `ph.framing.listening()` connects and closes at once. Every spawn, every
    stale-socket check and every `_await_socket` poll runs it. Nothing at the
    socket layer can tell such a knock from a front end, so the first frame
    does.

    The field has one job: **a client that leaves arms the keep-alive window,
    and a knock does not**. If a knock armed the window again, each check for
    the daemon would hold the daemon open for another full keep-alive. The
    *exit* asks `DaemonServer.served` instead. That is the same question about
    the process and not about one socket, so it also covers the lifetime clock
    and a turn that finishes, as well as this teardown."""
    peer: Peer = field(init=False)

    def __post_init__(self) -> None:
        # `peer` is built eagerly. `_Connection` is only ever built inside the
        # accept loop of `serve`, which already runs in an event loop, and a
        # running event loop is what `anyio.Event` needs. A lazy property gave
        # no benefit and left `peer` reachable in a half-built state.
        self.peer = Peer(
            stream=self.stream,
            dispatch=self._dispatch,
            dispatch_notifications=True,
            id_prefix="s",
        )

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or *raise* so that the root drops this watcher.

        Raising is the purpose. Code that catches `WouldBlock` here and logs
        "dropped" drops nothing, and a watcher that cannot keep up costs the
        whole fan-out again for every later event. The subscriber list belongs
        to the root, so the root removes the watcher from it. This method only
        has to fail loudly enough to be noticed.
        """
        self.peer.tell(method, params)

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Put a question to this client and wait for the user behind it."""
        return await self.peer.ask(method, params)

    async def serve(self) -> None:
        try:
            await self.peer.serve()
        finally:
            for root_id in list(self.attached):
                self._part(root_id)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> MethodResult:
        """The daemon half of the vocabulary, under the names of dsh (P5-02).

        The names are `session/*` and not the `root/*` of P5-01. A supervised
        root *is* a session here, and the dsh client already ships against these
        names. The supervisory additions (`daemon/hello`, `session/attach`) are
        declared in the capability block. A client does not infer them from
        which socket answered.

        **Two tables, not a chain (P8-07).** Every method is a row that names its
        `Verb` and its handler. `MUTATIONS` holds the methods that change a root
        under an idempotence key, and `METHODS` holds everything else. This has
        three results:

        - An unknown method is a missed lookup, and not the last `else`.
        - The vocabulary is a value, which `test_daemon_methods` compares with
          the verbs that it is derived from.
        - Each handler receives its own parsed type.

        The last result is what makes it possible to type the handlers. One
        function that read `params["sessionId"]` under twenty-one `if`s could
        not give each branch a different static type without a cast in each
        branch, and that is no better than a plain dict.
        """
        await self.server.check_unwatched()
        if not self.spoke:
            self.spoke = True
            self.server.served = True
            # A client is here, so any window that the last client armed when
            # it left is over. The keep-alive means "since you left", not
            # "since the daemon started". This is done here and not on the
            # accept. A knock that reset the window would hold a daemon open
            # for another full keep-alive each time something checked for the
            # daemon.
            self.server.keep_alive_until = None
            # The terminal that is already attached also learns that it is no
            # longer alone. `clients` moved, and that turns its `exits on
            # detach` into `held · client`. The arrival is a lifetime transition
            # like the departure in `_handle`, and it was the only transition
            # that was not announced.
            self.server.check_lifetime()
            # Both clock terms just moved: the keep-alive is disarmed and the spawn
            # window is closed.
            self.server.lifetime_clock.notice()
        mutation = MUTATIONS.get(method)
        if mutation is not None:
            return await self._mutate(mutation, params)
        entry = METHODS.get(method)
        if entry is None:
            raise UnknownMethod(f'unknown method "{method}"')
        return await entry.handle(self, entry.verb.parse(params))

    async def _mutate(
        self,
        mutation: Mutation[Any, WireModel],
        params: dict[str, Any],
    ) -> MethodResult:
        """Every method that changes a root goes through this one wrapper.

        The steps are: parse, resolve the root, validate, claim the idempotence
        key, act. The root is resolved through `start`, so an act on a
        passivated root brings it back. The parse is first, so a malformed call
        is refused before a root is mounted for it. See `MUTATIONS` for the
        reason why the key is claimed here and nowhere else.
        """
        parsed = mutation.verb.parse(params)
        root = await self.server.supervisor.start(parsed.session_id)
        plan = await mutation.prepare(self, root, parsed)
        try:
            reply = await self._acted(mutation, root, parsed, plan)
        except Exception:
            if mutation.after is not None:
                await mutation.after(self, root, None)
            raise
        if mutation.after is not None:
            reply = await mutation.after(self, root, reply) or reply
        return reply

    async def _acted(
        self, mutation: Mutation[Any, WireModel], root: Root, parsed: MutationParams, plan: object
    ) -> WireModel:
        """Run the act under its idempotence key, and put it on disk before any
        client is told."""
        key = _command_key(parsed)
        if not key:
            # The client offered no identity, so it wants no deduplication.
            # The act always runs.
            reply = await mutation.act(self, root, parsed, plan)
        else:
            journal = intents_of(root.ctx)
            # The key is claimed between the halves (`Mutation` gives the
            # reason). When an act raises, `claim_once` settles the key
            # `unknown`. Otherwise the key would refuse the retry, with nothing
            # to say that the act did not finish.
            async with journal.claim_once(root.session, CLIENT_COMMAND, {"command": key}) as held:
                if isinstance(held, Prior):
                    # A client that reconnects cannot avoid this retry, because
                    # it does not know whether the first request landed. The
                    # answer says "yes, that one", and whether it finished.
                    # That answer is what makes it safe to ask twice.
                    return MutationRepeated(
                        **root.describe().model_dump(), outcome=_repeat_outcome(held)
                    )
                reply = await mutation.act(self, root, parsed, plan)
                journal.settle(root.session, held, command_settled(key))
        # **Durable before the client is told** (F7). A client that has its reply
        # never sends the request again. When the reply went out while the key
        # and the record of the act were still in memory, a crash lost the act
        # completely, and "the message is logged" for `session/prompt` meant
        # logged in a buffer. The flush comes after the act, which is the reason
        # why `CLIENT_COMMAND` is `buffered`. A key that was made durable first
        # would turn a crash before the act into a refused retry for work that
        # never began. An act that records its own start before an effect that
        # it cannot take back (`!!`, `run_shell`) carries the key to disk with
        # that record. A crash then leaves an open key, repair settles it
        # `unknown`, and the retry is told so. One flush of the root carries the
        # key, the act and its settle.
        #
        # This is `written`, not `flush`. The act has happened by now, and a
        # write that fails does not undo it. An error answer to the verb would
        # tell the client the one thing that is false. A disk that refuses
        # writes makes the flushes of the root itself fail next, and the retry
        # ladder reports that (`supervisor/failed`).
        # A failure marks the root unwritten through the store (P13-08).
        await session_written(root.ctx, root.session)
        return reply

    # --- the methods -----------------------------------------------------------
    # One per row of `METHODS`, each taking its own params model. Read-only
    # projections are not here: `_projection` builds both their handler and
    # their row.

    async def _initialize(self, params: InitializeParams) -> CapabilityBlock:
        # This is a property of the *client*, so the client declares it one time
        # here and not at each attach. Whether a UI can put a modal in front of
        # a user does not depend on which session it is watching. A per-attach
        # flag let the same client claim the capability for one root and not for
        # another, which is two answers to a question that has one true answer.
        self.declared = frozenset(params.capabilities)
        # Unknown names are logged, because the failure is invisible otherwise.
        # A client that declares `"ask"` is never joined to any desk, and every
        # question that its user should have answered is never asked.
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

    async def _new_session(self, params: NewSessionParams) -> NewSessionReply:
        # `cwd` belongs to the client, and the daemon is what mounts. The client
        # therefore states it here. The daemon does not assume it from its own
        # process, which runs in a place where neither the user nor the files of
        # the user are. `or None` is used because a client that has no directory
        # to name sends `""` or omits the field, and both mean the same thing.
        cwd = params.cwd or None
        # The daemon mounts, so the daemon enforces trust. `ph_app.trust` gives
        # the reason. The check runs on every `session/new`: when a session is
        # named by id, one that the daemon would create and one that it would
        # resume look the same until the session is open.
        self._check_trust(cwd, params.trust)
        supervisor = self.server.supervisor
        # `asks`: a front end that can answer for a user asked for this start.
        # When the named profile of that user moved, the root is therefore held
        # for the user (S6). A request with no id is a new session, and the
        # supervisor names it as it opens (protocol 8).
        root = await supervisor.start(
            params.session_id,
            cwd=cwd,
            choice=params.choice,
            profile=params.profile,
            asks="asks" in self.declared,
        )
        if not params.choice.is_default:
            # A root that was already running (for another terminal) is on its
            # own route, so the choice moves it. A root that just mounted on
            # the chosen route is left alone.
            chosen = model_for(root.ctx, params.choice)
            # Compare the route and its call settings, and not the key. A route
            # that the list did not hold gets its key only when it is chosen
            # (`route_key`).
            running_on = replace(root.agent.options, model_key=chosen.key)
            if chosen.options() != running_on:
                await supervisor.set_model(root, chosen, source="cli", command=params.choice.flags)
        if params.trust == "always" and cwd is not None:
            # After the mount, not before: a directory is only worth
            # recording once its profile has actually composed.
            TrustStore(path=trust_path()).trust(Path(cwd))
        return NewSessionReply(**root.describe().model_dump(), header=root.session.header)

    async def _cancel(self, params: SessionParams) -> RootDescription:
        # Cancel is not a `MUTATIONS` row. It is idempotent by construction, and
        # with a key a genuine retry would be answered `repeated` while the turn
        # kept running. It uses `_root` and not `start`, because a passivated
        # root has nothing to stop.
        root = self._root(params.session_id)
        root.agent.cancel(AgentCancelCause(kind="user"), keep_inbox=True)
        return root.describe()

    async def _sessions_browse(self, params: BrowseParams) -> SessionBrowse:
        # This is daemon-level and not a `_projection` row, because it is not
        # about one root. The supervisor names the directory of the store
        # (`sessions_directory`), and it knows which roots are held.
        return SessionBrowse(sessions=browse_of(self.server.supervisor, cwd=params.cwd))

    async def _daemon_config(self, _params: NoParams) -> DaemonConfigReply:
        # The composed profile of the *daemon*: what a new root starts on when
        # `session/new` names no profile. A root can be on another profile
        # (S5), so these rows are the default and not the rows of every root.
        return DaemonConfigReply(rows=list(self.server.supervisor.profile.dump()))

    async def _credentials_held(self, params: SessionParams) -> CredentialsHeldReply:
        # The names are `credentials/held` and `credentials/store`, and not
        # `session/credential` and `session/credentials`. Those were two names
        # one letter apart for opposite kinds of method, and the method that
        # writes a secret must be the hardest one to reach by a typo.
        #
        # There is no `names` param. The daemon composes the profile of the
        # root, so it knows which credentials that profile names (see
        # `credentials_of`). When the client had to supply the names, a front
        # end had to be sent the whole composed configuration.
        root = self._root(params.session_id)
        return CredentialsHeldReply(session_id=root.id, held=credentials_of(root))

    # The schedule seam over the wire (P5-06, P5-10). Create and cancel go
    # through the supervisor and not directly through the seam. Both need the
    # root mounted and the append flushed, and a restart forgets a schedule
    # that is only in a buffer.

    async def _schedule_create(self, params: CreateScheduleParams) -> Schedule:
        # The `schedule` hold moves when the log does, not here.
        # `_watch_schedules` receives the record and rechecks, whichever path
        # wrote it.
        return await self.server.supervisor.schedule(params.session_id, params.to_schedule())

    async def _schedule_cancel(self, params: CancelScheduleParams) -> ScheduleCanceled:
        canceled = await self.server.supervisor.unschedule(params.session_id, params.schedule_id)
        return ScheduleCanceled(
            session_id=params.session_id,
            schedule_id=params.schedule_id,
            canceled=canceled,
        )

    async def _schedule_list(self, params: SessionParams) -> SessionSchedulesReply:
        root = self._root(params.session_id)
        return SessionSchedulesReply(
            session_id=root.id, schedules=self.server.supervisor.scheduled(root)
        )

    async def _daemon_status(self, _params: NoParams) -> DaemonStatusReply:
        return self.server.status()

    async def _sessions_owed(self, params: OwedParams) -> OwedListing:
        # Daemon-level, like `sessions/browse`. The index is one file under the
        # `$PH_HOME` of this daemon, and the supervisor itself knows which of
        # its roots are mounted.
        return await self.server.supervisor.owed(rebuild=params.rebuild)

    async def _sessions_resume(self, params: ResumeParams) -> ResumeReport:
        # Unkeyed, like `session/new`. `start` is idempotent by id, so a pass
        # that is sent twice mounts each root one time, and answers `mounted`
        # the second time.
        return await self.server.supervisor.resume_owed(
            params.sessions, all_owed=params.all_owed, force=params.force
        )

    async def _daemon_probe(self, params: ProbeParams) -> ProbeReply:
        return self.server.probe(params.name)

    async def _daemon_lifetime(self, _params: NoParams) -> DaemonLifetime:
        return self.server.lifetime()

    async def _shutdown(self, _params: NoParams) -> None:
        # This stops the daemon, and by contract it takes no id. A client that
        # awaited a reply would wait for a frame that the daemon is losing the
        # ability to write at the same moment. "Stop" is not a question.
        self.server.stop.set()

    async def _put(self, params: PutAttachmentParams) -> AttachmentStored:
        """Store bytes that a client read, and answer with the reference to them.

        **The client reads the file, not the daemon.** This is the human route
        of I-9. A user may attach anything that the user can already open, and
        only the permissions of the user decide. It is also the only path that
        a browser has, because the bytes reach the page before they reach
        anything else. The daemon never learns a path. For that reason this is
        the same method for a terminal on this machine and for a tab on another
        machine.

        The store is content-addressed. A file that is put twice is stored one
        time, and the second put is a cheap way to *learn* the reference.

        This is not a `MUTATIONS` row, on purpose. A retry is already a no-op,
        and the reply *is* the reference that the client came for. A `repeated`
        envelope would withhold it.
        """
        root = self._root(params.session_id)
        store = self._store(root)
        encoded = params.content_b64
        # The size is checked before decoding, on the encoded length. Base64 is
        # 4/3 of the bytes, so decoding first to measure would make the
        # allocation that this refusal exists to avoid. The estimate is off by
        # the two padding bytes, which does not matter here.
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
            # the one rule that both routes use. With an octet-stream default
            # here, a `.png` was a document, while `--attach` called it an
            # image.
            mime=mime_for(params.mime or "", name or ""),
            name=name,
        )
        return AttachmentStored(session_id=root.id, attachment=ref)

    # --- the mutation halves --------------------------------------------------
    # `prepare` validates and may refuse. Nothing that it does is an effect.
    # `act` is the effect. The wrapper claims the idempotence key between them,
    # so a refusal does *not* use up a retry. A prompt that is refused for an
    # unknown attachment, and sent again with a known one under the same key,
    # must act.

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
        """Put an attachment in the tray of the composer, for every attached UI.

        Nothing is appended. Intent that is not submitted is not an act in the
        session, which is the same rule that keeps a half-typed prompt out of
        the log. The notice is broadcast because the tray is shared. If only
        the uploader could see a chip, nobody else could reason about the
        composer.
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
        """Run one `/name argument` line in the context of the root.

        **In the daemon, not the client.** A command body uses seams that exist
        only where the profile is mounted. The record that the command ran also
        belongs in the log of this session, where every other attached UI sees
        it.
        """
        shown = await registry.dispatch(
            params.line,
            scope=root.agent.ctx,
            session=root.session,
            agent=root.agent,
        )
        # A command can move a reading without moving the agent, and `/sandbox`
        # is the command that does. Nothing else would read the readings again.
        # See `_refresh_readings`. It costs one fold, and only when a client is
        # attached to see the result.
        _refresh_readings(root)
        return CommandShown(session_id=root.id, shown=shown)

    async def _after_command(self, root: Root, reply: CommandShown | None) -> CommandShown | None:
        """Start the root again after a command moved it past what a live mount follows.

        `/profile use` asks for this with `Root.restart_wanted`. So does a
        `/profile clear` of an override that turned a row on or off. This step
        consumes the request after the command is on disk, and the root starts
        again from its log. When that start had to withdraw the version that it
        was given (`runtime.mount_session`), the reply says so (`Root.withdrew`).
        The user who asked is listening here, before any attach reads it.
        """
        if not root.restart_wanted:
            return None
        root.restart_wanted = False
        back = await self.server.supervisor.restart(root)
        if reply is None or back is None or not back.withdrew:
            return None
        said = f"{reply.shown}\n" if reply.shown else ""
        return reply.model_copy(update={"shown": f"{said}{back.withdrew}"})

    async def _prepare_shell(self, root: Root, params: ShellParams) -> tuple[ShellService, str]:
        """Resolve the seam and the command **before** the key is claimed.

        Earlier the seam check was inside `act`, which ran after `once()` had
        recorded the request. A `!!` that was sent to a deployment with no shell
        therefore used up the retry of the client on a refusal. The two halves
        exist to prevent exactly that.
        """
        command = params.command.strip()
        if not command:
            raise Refusal("a shell command cannot be empty")
        return shell_of(root.ctx), command

    async def _act_shell(
        self, root: Root, params: ShellParams, prepared: tuple[ShellService, str]
    ) -> ShellReply:
        """`!!<command>`: the shell of the user, in the workspace of the session.

        The reply is the exit code. The *output* arrives as `shell/command` and
        `shell/result` events. Every attached UI therefore reads it by the one
        route that they all already watch, and the user who typed the command
        reads it from the same event as every other client.
        """
        shell, command = prepared
        return await self.server.supervisor.shell(root.id, shell, command, surface=params.surface)

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

    async def _prepare_model(self, root: Root, params: ModelParams) -> ModelEntry:
        """Resolve the choice **before** the key is claimed, by the rule of
        `_prepare_shell`. A key that this profile does not list, or a provider
        that nothing here serves, is a refusal, and a refusal must not use up
        the retry of the client."""
        return model_for(root.ctx, params.choice)

    async def _act_model(
        self, root: Root, params: ModelParams, chosen: ModelEntry
    ) -> RootDescription:
        """The root as it now stands. A repeat of this key gets that reply too."""
        await self.server.supervisor.set_model(
            root, chosen, source="verb", command=f"/model {params.choice.spelled}"
        )
        return root.describe()

    async def _credentials_store(self, params: StoreCredentialParams) -> CredentialStored:
        """Give a root the value of a credential. The root uses it and does not keep it.

        **Unkeyed** (T5). One value that is stored twice is stored one time, so a
        retry needs no key. A key would also outlive what it named: the value is
        in the memory of this process, so after a restart the only way to bring
        it back is to send it again. This uses `start` and not `_root`, because
        the value belongs to the seam of a mounted root.
        """
        root = await self.server.supervisor.start(params.session_id)
        service = root.ctx.get(CREDENTIALS)
        if service is None:
            raise SeamAbsent("this deployment stores no credentials")
        # **The value is used and not kept.** It is never logged, never echoed
        # in the reply, and never reaches `describe()`. The reply is the name
        # and a boolean, which is everything that a UI needs to redraw.
        service.provide_value(params.name, params.value)
        # Whatever was waiting for the value now proceeds (T5): the root, if
        # its route named this credential, and every child that was held for it.
        await self.server.supervisor.credential_supplied(root)
        return CredentialStored(session_id=root.id, name=params.name)

    def _attachments(self, root: Root, refs: list[AttachmentRef]) -> list[AttachmentRef]:
        """The refs that a prompt named, checked against what this deployment holds.

        An unknown ref is refused and not dropped. A reference from another
        machine, or to a blob that a `gc` removed, would otherwise send the turn
        as plain text with nothing to say that the picture was never sent.
        """
        if not refs:
            return []
        store = self._store(root)
        return [self._known(store, ref) for ref in refs]

    def _store(self, root: Root) -> AttachmentStore:
        """The attachment store, or the one refusal for its absence.

        The three methods that need the store all resolve it here, so the
        message "this deployment stores no attachments" is written in one place.
        The two resolutions that grew separately had opposite rules for a
        missing store: one staged anything and one refused everything.
        """
        store = root.ctx.get(ATTACHMENTS)
        if store is None:
            raise SeamAbsent("this deployment stores no attachments")
        return store

    def _known(self, store: AttachmentStore, ref: AttachmentRef) -> AttachmentRef:
        """A reference that the client sent, checked against what this deployment
        holds.

        This checks presence only. The params model checked the *shape*, so a
        malformed reference is `invalid_params`. This refusal is reserved for a
        well-formed reference that nobody stored here.
        """
        if not store.exists(ref):
            raise AttachmentUnknown(f"no attachment {ref.attachment_id} is stored here")
        return ref

    def _check_trust(self, cwd: str | None, answer: TrustAnswer) -> None:
        """Refuse a `cwd` that nobody has vouched for. `ph_app.trust` gives the reason.

        `"once"` mounts and records nothing. For `"always"`, the caller records
        the answer after the mount has succeeded.
        """
        if cwd is None or answer == "once":
            return
        store = TrustStore(path=trust_path())
        project = Path(cwd)
        if answer == "always":
            # The *caller* records this answer, after the mount has succeeded.
            # A write here would vouch for a directory whose profile then
            # failed to compose, and the next client would be let in without a
            # question.
            return
        if not store.trusted(project):
            raise UntrustedProject(f"{cwd} has not been trusted; ask, then send trust")

    async def _adopt(self, params: AdoptParams) -> RootDescription:
        """Accept the version of a named profile for a root that this daemon holds (S6).

        `phern profiles adopt` writes the log of a stored session itself, under
        the lease of that session. A daemon that holds a session holds its
        lease, so the daemon writes that log. This does not go through `start`.
        A root that is not mounted here is for the caller to write, and to mount
        it in order to record a version would start what the record says to
        start later. The version is the one that the user was shown, and it is
        carried whole.
        """
        root = self._root(params.session_id)
        await self.server.supervisor.adopt(root, ProfileBase.of(params.version))
        return root.describe()

    def _root(self, session_id: str) -> Root:
        root = self.server.supervisor.roots.get(session_id)
        if root is None:
            raise NoSuchSession(f'no session "{session_id}"')
        return root

    async def _attach(self, params: SessionParams) -> AttachReply:
        """Subscribe to what happens *next*, and say where that starts.

        **Attach does not replay.** A replay here would stream the gap as one
        `session.event` frame for each event, into a fixed-size outbox with no
        await point. A client that attached again to a root that had moved on
        would then get a `WouldBlock` from its own attach, *after* the
        subscription was made.

        Catch-up therefore has one mechanism, and it is the paged one. The reply
        says where the live stream begins, and the client reads
        `session/snapshot` from its cursor up to that point. The 512 KiB-class
        bound then applies to replay too.
        """
        # The root comes through `start`, so an attach to a *passivated* root
        # brings it back and does not report it gone (P5-05). `start` returns
        # the live root untouched when there is one, and resumes from the log
        # when there is not. `session/prompt` takes the same path, which keeps
        # rehydration as one mechanism and not two. There is no cursor to read:
        # attach does not replay, so it does not take one. `SnapshotParams`
        # says why the field was removed and not accepted and ignored.
        root = await self.server.supervisor.start(params.session_id, asks="asks" in self.declared)
        self._join(root)
        # The footer comes with the status that it belongs to, which is the
        # same pairing that `session.status` makes. A client that has just
        # attached can then draw a complete footer without a second call, and
        # without assembling a frame shape that no wire message has.
        return AttachReply(**root.describe().model_dump(), readings=readings_of(root))

    async def _snapshot(self, params: SnapshotParams) -> SnapshotPage:
        """One bounded page of a session's history, and the cursor for the next."""
        root = self._root(params.session_id)
        start = resume_at(root.session, params.cursor)
        page = root.session.events_from(start, SNAPSHOT_EVENTS)
        tools = root.ctx.get(TOOLS)
        events = [event.to_wire(thaw=False) for event in page]
        # **The same sidecar that the relay attaches**, because a client must
        # not see one transcript live and a different one on replay. It is keyed
        # by seq and **sparse**, because a page is 2048 events and a turn
        # contributes only a few cards. The positional list that this replaced
        # serialized two thousand `null`s (twelve kilobytes of nothing) on every
        # attach, reconnect and page-forward.
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
            # Where the read actually began. That is not always where the cursor
            # asked: `resume_at` answers a cursor from another log with 0, and
            # can move a cursor from an earlier incarnation of this log back.
            # The value is sent, and the client does not have to infer it from
            # the seq of the first event. That inference gives `None` for an
            # empty page, and the docstring of `resume_at` already promises
            # that the client is told.
            started_at=start,
            cursor=cursor_of(root.session, start + len(events)),
            more=start + len(events) < root.session.seq,
        )

    async def _status(self, params: SessionParams) -> RootStatusReply:
        """One root in detail: what `sessions/list` says, and the reason for it.

        The listing carries what a table needs for every root. This reply carries
        what a user asks about *one* root. It exists because of the retry
        ladder. `status` collapses to `"retrying"` or `"failed"`, and two
        questions follow: how many attempts were made, and what will still
        fire. No other method can ask them.
        """
        root = self._root(params.session_id)
        return RootStatusReply(
            **root.detail().model_dump(), schedules=self.server.supervisor.scheduled(root)
        )

    def _join(self, root: Root) -> None:
        """Start watching this root, and answer for it if this client may.

        Watching and answering are different claims. The second claim is the
        `asks` capability of the client, and is not a property of the attach. A
        follower such as `phern agents attach` watches, never declares `asks`,
        and is never asked. See
        `test_a_watcher_that_is_not_a_front_end_is_never_asked`.

        **Subscribed on every attach** (D3). The subscription goes into the
        audience of the id of the root, which every mount of that id shares
        (`Audience`). A set add needs no guard.
        """
        self.attached.add(root.id)
        root.subscribe(self.notify)
        if "asks" in self.declared and root.desk is not None:
            root.desk.join(self)

    def _part(self, root_id: str) -> bool:
        """Stop watching and stop being asked. Returns whether this client was
        attached.

        **One pair, three callers** (E5). Attaching does three things, and three
        places each undid some of them. `_detach` unsubscribed and kept the desk
        seat. A client that detached and stayed connected was therefore still
        put in front of a user: it got an approval modal for a session that it
        had said it no longer watched, and so it never answered. Only the
        teardown in `serve` undid all three, and only because the TUI closes the
        socket right after it detaches.

        The detach is by id, as the attach was (`Supervisor.part`, D8). The
        audience belongs to the id. A detach therefore takes effect whether a
        root is mounted under the id, or a restart is between the unmount of one
        root and the mount of the next.
        """
        was_attached = root_id in self.attached
        self.attached.discard(root_id)
        self.server.supervisor.part(root_id, self.notify, self)
        return was_attached

    async def _detach(self, params: SessionParams) -> SessionDetached:
        was_attached = self._part(params.session_id)
        # It is deliberately *not* an error when nothing was attached. A client
        # detaches while it tidies up, often twice, and nobody can write a
        # teardown path correctly if that path raises.
        return SessionDetached(session_id=params.session_id, detached=was_attached)


def _refresh_readings(root: Root) -> None:
    """Read the footer again and push it, for a change that the *agent* did not make.

    Readings are carried by `session.status`, and the supervisor sends one when
    the agent moves. That is right for a budget or a token count, and wrong for
    the two postures. A preset switch and `/sandbox` change what the footer says
    while the agent does nothing, so nothing read the readings again. The
    client redrew from the list that it held and painted the posture that it
    already had.

    The status is left empty on purpose. `StatusFacts` is "as much of it as
    changed", and a reader keeps every field that a frame does not state. This
    frame therefore says that the readings moved, and claims nothing about
    where the agent is.

    The publish has the same guard as the publish in the `announce` of the
    supervisor, and for the same reason. A reading is a fold over the log, and
    the guard exists to skip a fold that nobody reads.
    """
    if root.subscribers:
        root.publish(SessionStatusNotice(session_id=root.id, readings=readings_of(root)))


def _projection[N: SessionNotice](
    verb: Verb[SessionParams, N], key: str, fold: Callable[[Root], list[Any]]
) -> tuple[str, _Row]:
    """One `METHODS` row that answers a projection with the reply model of its verb.

    This returns the row and not the handler, so each projection names its verb
    **one time**. When the handler was wrapped in `_reading(...)` at the table,
    the verb appeared twice on one line, which is the shape that `_unkeyed` and
    `_mutating` exist to remove.

    This is a factory, and not eight nearly identical handlers, because the
    eight differ only in a model, a key and a fold. The closure holds exactly
    those, and nothing of the request. (`Method.handle` receives no method
    name. A single handler that read a table would therefore have to add that
    parameter to every other handler in order to serve these eight.)

    For three of the eight, the reply model is the *same type* that the
    notification uses. A client validates the reply of `commands/list` as a
    `SessionCommandsNotice`, because the reply and the notification are one
    shape. The server was the only party that did not say so, and it dumped
    rows by hand at this return.
    """

    async def handle(connection: _Connection, params: SessionParams) -> N:
        root = connection._root(params.session_id)
        # The reply is built through the verb, so the model is named one time.
        # The row that routes this handler and the handler itself cannot
        # disagree about what it answers. `read` is used and not
        # `reply(session_id=…, **{key: …})`: the field name is a parameter here,
        # so a keyword spread cannot be typed. Validation also checks that the
        # rows of the fold match the declared element type of the model, which
        # is the reason why the model is named.
        return verb.read({"session_id": root.id, key: fold(root)})

    return _unkeyed(verb, handle)


def _mutating[P: MutationParams, R: WireModel](
    verb: Verb[P, R],
    prepare: Callable[[_Connection, Root, P], Awaitable[Any]],
    act: Callable[[_Connection, Root, P, Any], Awaitable[R]],
    *,
    after: Callable[[_Connection, Root, R | None], Awaitable[R | None]] | None = None,
) -> tuple[str, Mutation[Any, Any]]:
    """One `MUTATIONS` row, keyed by the name of its verb and **checked against it**.

    This is a function, and not `{verb.name: Mutation(verb, ...)}` written
    inline. The inline form looks identical and checks nothing, so the reason
    is stated here. The annotation of the table is the expected type of each
    value. `dict[str, Mutation[Any, Any]]` therefore solves `P` and `R` to
    `Any`, and every handler fits every verb. A sabotage check confirmed this:
    a row that paired `session/status` with `_sessions_list` passed
    `mypy --strict` with no error. Here the return type names no type variable,
    so `P` and `R` can come only from the verb, and the two halves are checked
    against it.
    """
    return verb.name, Mutation(verb, prepare, act, after)


def _unkeyed[P: WireModel, R: WireModel](
    verb: Verb[P, R], handle: Callable[[_Connection, P], Awaitable[R]]
) -> tuple[str, _Row]:
    """One `METHODS` row, checked against its verb. `_mutating` gives the reason.

    The name says that the row has no idempotence key. It does not say that the
    row reads: `session/new`, `session/cancel`, `attachment/put`,
    `schedule/create` and `schedule/cancel` all write. `verbs.UNKEYED` carries
    the same correction.
    """
    return verb.name, Method(verb, handle)


def _announcing[P: WireModel](
    verb: Notify[P], handle: Callable[[_Connection, P], Awaitable[None]]
) -> tuple[str, _Row]:
    """One `METHODS` row for a method with no reply. `_mutating` says why this is
    a builder and not a literal. `Announced` says why it is a separate row type."""
    return verb.name, Announced(verb, handle)


MUTATIONS: dict[str, Mutation[Any, Any]] = dict(
    (
        _mutating(verbs.SESSION_PROMPT, _Connection._prepare_prompt, _Connection._act_prompt),
        _mutating(
            verbs.SESSION_COMMAND,
            _Connection._prepare_command,
            _Connection._act_command,
            after=_Connection._after_command,
        ),
        _mutating(verbs.SESSION_STAGE, _Connection._prepare_stage, _Connection._act_stage),
        _mutating(verbs.SESSION_SHELL, _Connection._prepare_shell, _Connection._act_shell),
        _mutating(verbs.SESSION_PRESET, _Connection._prepare_preset, _Connection._act_preset),
        _mutating(verbs.SESSION_MODEL, _Connection._prepare_model, _Connection._act_model),
    )
)
"""Every method that changes a root, and the one place that makes them idempotent.

The `clientId` and `commandId` of a client name a request. A client that
reconnects cannot know whether its last request landed, so it sends the request
again, and the same key must not run the effect twice. Each handler used to
apply that rule from memory: two handlers wrote it and the third forgot. The
rule is now applied here by construction. A method in this table gets the
write-ahead guard whether or not its author thought of it, and a method that is
not in the table cannot claim to be idempotent by key.

The repeat reply has one shape: `MutationRepeated`, a description that carries
`repeated: true` and whether the first attempt finished (`outcome`). A client
therefore branches on one field for every verb. The keys and their outcomes
belong to the journal (`CLIENT_COMMAND`, P10-10).

Two methods are absent on purpose:

- `attachment/put` is content-addressed, so a retry is already a no-op, and its
  reply *is* the reference that a repeat must still return.
- `session/new` goes through `start`, which is idempotent by id.
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
        _unkeyed(verbs.CREDENTIALS_STORE, _Connection._credentials_store),
        _unkeyed(verbs.SESSION_ADOPT, _Connection._adopt),
        # The read-only projections (P5-14). A front end used to read these
        # directly from `ctx`. Each is a fold that is computed now, so a client
        # that reconnects gets the current answer, and not one that was cached
        # when somebody last wrote it down. See `projections.py`. They are rows
        # here, and not a third table that is spliced in. They are `METHODS`
        # rows like any other, and the table that they came from was read
        # exactly one time, to build this one.
        _projection(verbs.SESSION_READINGS, "readings", readings_of),
        _projection(verbs.COMMANDS_LIST, "commands", commands_of),
        _projection(verbs.SCREENS_LIST, "screens", screens_of),
        _projection(verbs.SESSION_CHILDREN, "children", family_rows),
        _projection(verbs.TOOLS_LIST, "tools", tools_of),
        _projection(verbs.SKILLS_LIST, "skills", skills_of),
        _projection(verbs.PRESETS_LIST, "presets", presets_of),
        _projection(verbs.MODELS_LIST, "models", models_of),
        _unkeyed(verbs.SCHEDULE_CREATE, _Connection._schedule_create),
        _unkeyed(verbs.SCHEDULE_CANCEL, _Connection._schedule_cancel),
        _unkeyed(verbs.SCHEDULE_LIST, _Connection._schedule_list),
        _unkeyed(verbs.DAEMON_CONFIG, _Connection._daemon_config),
        _unkeyed(verbs.DAEMON_STATUS, _Connection._daemon_status),
        _unkeyed(verbs.DAEMON_LIFETIME, _Connection._daemon_lifetime),
        _unkeyed(verbs.DAEMON_PROBE, _Connection._daemon_probe),
        _unkeyed(verbs.SESSIONS_OWED, _Connection._sessions_owed),
        _unkeyed(verbs.SESSIONS_RESUME, _Connection._sessions_resume),
        _announcing(verbs.SHUTDOWN, _Connection._shutdown),
    )
)
"""Every method that is not a mutation: what it takes, and what answers it.

The table is disjoint from `MUTATIONS` by construction. A name in both would be
dispatched as a mutation, and the row here would never be reached. The two
tables together are the whole vocabulary of the daemon. `test_daemon_methods`
compares them with `verbs.VOCABULARY`. That set is derived from the verb
declarations, so the check catches a verb that is declared and never routed.
`initialize` and `daemon/hello` are two names for one handler, because the dsh
SDK uses the first name and P5-01 used the second.
"""


@dataclass(slots=True)
class DaemonServer:
    """The supervisor behind the socket, and the event that ends the run."""

    supervisor: Supervisor
    stop: anyio.Event
    path: Path
    """The socket that this server answers on, so that `daemon/status` can report it.

    A client resolves the same path to connect. But "which socket am I actually
    talking to" is the first question of anyone who debugs two daemons. An
    answer that the client derived a second time would agree with the server by
    assumption and not by evidence."""
    ephemeral: bool = False
    """Whether this daemon exits when nothing needs it any more (P7-08).

    **Whoever started the daemon decides, and that is the complete rule.** A
    `phern daemon` that is typed at a prompt is a service. Somebody chose to run
    a supervisor, and a service that exits when it is idle is not there when the
    next client arrives. Nobody decided to start a daemon that a UI spawned
    because the socket was absent. If that process stayed resident on the
    machine of a user after the user closed the UI that started it, nobody
    would connect the leftover process with its cause a week later.

    This is a flag, and not a subclass or a second `serve`. Every other
    behavior is identical, and the difference is one predicate (`spent`)."""
    connections: set[_Connection] = field(default_factory=set)
    """The clients that are connected right now. *Connected* does not mean attached.

    The distinction is deliberate. A `phern agents doctor` in the middle of a
    call has no subscription and no root, so a count of attachments would
    disconnect it between the request and the reply.

    This is a set, and not the count that it used to be, because it now answers
    two questions: whether anything is connected, and **whom to tell** when the
    lifetime changes (P9-07)."""
    keep_alive: float = 0.0
    """Seconds to stay up after the last client leaves, when nothing else holds
    the daemon.

    The default is zero, which means leave at once. A user sets the flag when a
    switch between two terminals must not cost a new mount. It is the
    preference of the *client*, and it reaches the daemon through the argv that
    the client composed. That is the only route. A daemon may not read the
    `tui.json` of a front end, and after P5-14 it may not even share a
    filesystem with the front end."""
    keep_alive_until: int | None = None
    """When the armed keep-alive expires, or `None` when none is armed.

    The last client arms it when it leaves, and the first frame of the next
    client clears it. The window is therefore "since you left" and not "since
    the daemon started"."""
    resumed: bool = True
    """Whether the first resume pass has finished (P13-03). It is true from the
    start when no pass was asked for. `serve(resume_owed=True)` builds the server
    with `False`, and the pass sets it. Readiness is no until then: an
    orchestrator that sends work to a daemon that is still mounting what the
    last daemon owed would race the mounts."""
    served: bool = False
    """Whether any client has ever spoken to this daemon (P9-06).

    **This closes the spawn window in a place that every caller inherits.** A
    fresh ephemeral daemon holds nothing: no connection, no turn, no
    appointment. It is therefore spent from its first instant. It does not exit
    only because nothing asks, in the moment between `serve()` binding the
    socket and the UI that spawned it sending a frame. Three callers can ask in
    that moment: the disconnect of the knock, the first pass of the lifetime
    clock, and a scheduled root that finishes a turn. A guard on one of them
    left the other two.

    `spent()` therefore reads this field, and the answer is bounded and not
    latched. A daemon that nobody has spoken to may not exit until
    `launch.SPAWN_TIMEOUT` has passed, which is the same number that the
    launcher waits. After that time the launcher has given up, so nobody is
    left to protect. One constant serves both halves, and not two constants
    that must agree.

    The field is deliberately not on the wire. It is never false when anybody is
    watching, because the frame that makes it true is the frame of the watcher."""

    announced: DaemonLifetime | None = None
    """The last lifetime frame that was sent, so that an unchanged answer is not
    sent again.

    This is a remembered value and not a dirty flag. The comparison is over the
    whole frame, and a flag would have to be set by every writer of every field
    that goes into a frame."""
    watching: Mechanism | None = None
    """How the entry of the socket is watched (P5-11, P12-05), or `None` when it is
    not watched: the watch was refused (`watch_refused`), or the server was built
    by hand. `serve` sets it."""
    watch_refused: str | None = None
    """Why the watch could not be armed, or why it stopped, or `None`. When this is
    set, nothing hears the socket move, and `check_unwatched` checks at each
    client request instead."""
    lifetime_clock: Planner = field(init=False)
    """The two deadlines of the daemon itself (P7-08, P12-02): the expiry of an armed
    keep-alive, and the moment when an auto-started daemon that nobody has spoken
    to reaches `SPAWN_TIMEOUT` (`lifetime_deadline`). Every other change to what
    holds the process arrives as an event, and `check_lifetime` runs on each.
    These two end on a clock, when nobody is left in the process to notice. The
    planner therefore sleeps until the sooner of the two and asks then.

    The planner is told wherever either deadline moves: at the first frame of a
    connection (which disarms the keep-alive and closes the spawn window) and
    when a connection closes (which may arm the keep-alive)."""
    started: int = field(default_factory=now_ms)
    """When this daemon came up, for the uptime a status reply carries."""
    identity: tuple[int, int] | None = None
    """`(st_dev, st_ino)` of the socket at bind time. `check_reachable` compares
    against it.

    `serve` captures it, and this object does not read it. "The socket that I
    bound" is a fact about one moment, and this object outlives that moment. A
    read of the path on first use would adopt whatever is there by then, which
    is the exact substitution that the watch exists to catch."""
    serving: bool = False
    """Whether the unix listener is accepting (P13-12). It is the complete answer
    of the `startup` probe. `serve` sets it beside `ready`."""
    heartbeat_at: int = field(default_factory=now_ms)
    """When the event loop last stamped the clock (`keep_heartbeat`), in epoch ms.
    `live` measures a stuck loop by it."""
    probe_address: str = ""
    """`host:port` the probes answer on, or `""` when not bound (`--probe-bind`)."""
    _leases: Path | None = field(default=None, init=False)
    """The lease directory. The first `ready` probe resolves and makes it, so each
    later probe asks the kernel one question and does not resolve the roots
    again."""
    unreachable_since: int | None = None
    """When the socket stopped being the socket of this daemon, or `None` while it
    still is.

    This is a latch, not a sample. The transition is one-way by construction: the
    inode of an unlinked socket does not come back, and a path that anyone else
    created again belongs to them. The field is therefore set one time and
    announced one time. After that, `status` reads it for whoever asks later."""

    def __post_init__(self) -> None:
        # The two are wired here and not passed in, because they hold each
        # other. The server needs the supervisor to ask what is running, and the
        # supervisor needs the server to say that the answer moved. The wiring
        # is here and not in `serve`, so no server is ever built half-wired.
        self.supervisor.recheck_lifetime = self.check_lifetime
        self.lifetime_clock = Planner(
            "the lifetime check", run=self._lifetime_pass, plan=self.lifetime_deadline
        )

    async def _lifetime_pass(self) -> None:
        self.check_lifetime()

    def lifetime_deadline(self, now: int) -> int | None:
        """The sooner of the two clock terms that are still ahead of `now`, in epoch
        ms, or `None`.

        Only moments that are still ahead count. The pass that just ran already
        asked about a moment that is past, and only a notice moves it again."""
        moments: list[int] = []
        if self.keep_alive_until is not None:
            moments.append(self.keep_alive_until)
        if self.ephemeral and not self.served:
            moments.append(self.spawn_window_ends)
        return min((moment for moment in moments if moment > now), default=None)

    @property
    def spawn_window_ends(self) -> int:
        """When a daemon that nobody has spoken to may leave, in epoch ms (`served`
        gives the reason).

        The clock that wakes at this moment and the predicate that asks after it
        (`spent`) use this one expression, so the two cannot disagree."""
        return self.started + int(SPAWN_TIMEOUT * 1000)

    @property
    def socket_watch(self) -> str:
        """`inotify`, `kqueue`, `off`, or `unavailable: <why>`, for `daemon/status`."""
        if self.watching is not None:
            return self.watching
        return "off" if self.watch_refused is None else f"unavailable: {self.watch_refused}"

    async def check_unwatched(self) -> None:
        """Check the socket as a client request arrives, when nothing watches it
        (`_Connection._dispatch`).

        A watch that could not be armed must not silently become a cadence (§12
        decision 4). A request is the moment that can still be acted on. A
        daemon that lost its path cannot be reached by new clients, the doctor
        included. The clients that can be told are therefore the ones that are
        already connected, and those clients send the requests."""
        if self.watch_refused is not None:
            await self.check_reachable()

    def status(self) -> DaemonStatusReply:
        """What this daemon is, for `phern agents doctor`.

        Everything here is read from the running process, and the client does
        not derive it again: the socket that the daemon bound, the policy that
        it was started with, the jobs that it is running. A doctor that reported
        what the *client* would have chosen would agree with a daemon that was
        started differently, and would tell the user nothing.
        """
        supervisor = self.supervisor
        return DaemonStatusReply(
            protocol_version=PROTOCOL_VERSION,
            capabilities=served(*CAPABILITIES),
            pid=os.getpid(),
            socket=str(self.path),
            uptime_ms=now_ms() - self.started,
            roots=len(supervisor.roots),
            starts_on=supervisor.starts_on(),
            passivate_after=supervisor.passivate_after,
            next_wake=supervisor.scheduler.planned,
            next_release=supervisor.releaser.planned,
            socket_watch=self.socket_watch,
            check_invariants=supervisor.check_invariants,
            unreachable_since=self.unreachable_since,
            unwritten=sorted(set(supervisor.unwritten.values())),
            probe=self.probe_address,
            # The shape of `report()` as models: a list of sections, each a
            # title and `(label, value)` rows. It carries the built-in sections
            # of `DaemonServer.report`, of which the socket lifetime of P5-11
            # was the first. Each fact has one encoding, so nothing on the wire
            # can disagree with itself, and the client needs no special
            # decoder. P5-12 fills the same envelope from the *mounted*
            # registry of the daemon with no client change.
            sections=[
                DiagnosticSection(
                    title=title,
                    rows=[DiagnosticRow(label=label, value=value) for label, value in rows],
                )
                for title, rows in self.report()
            ],
        )

    def probe(self, name: ProbeName) -> ProbeReply:
        """Answer one of the three orchestrator questions. Both routes get their
        answer here (P13-12): the TCP responder and `daemon/probe` call this and
        nothing else.

        - *startup*: the unix listener is serving.
        - *ready*: startup, the socket is still the socket of this daemon, there
          is no root that the daemon could not write, the lease directory is
          writable, and the first resume pass has finished when one was asked
          for (P13-03).
        - *live*: the event loop stamped a heartbeat within `STALE_HEARTBEATS`
          intervals, the socket is still the socket of this daemon, and the
          scheduler is no later than `SCHEDULER_GRACE_SECONDS` past its own plan.

        Each reason is a sentence that an operator reads from a 503.
        """
        now = now_ms()
        reasons: list[str] = []
        if self.stop.is_set():
            reasons.append("the daemon is stopping")
        if name in ("startup", "ready") and not self.serving:
            reasons.append("the socket is not accepting yet")
        if name in ("ready", "live") and self.unreachable_since is not None:
            reasons.append(f"the socket at {self.path} is no longer this daemon's")
        if name == "ready":
            unwritten = sorted(set(self.supervisor.unwritten.values()))
            if unwritten:
                reasons.append(f"root(s) could not be written: {', '.join(unwritten)}")
            leases = self._leases_writable()
            if leases:
                reasons.append(leases)
            if not self.resumed:
                reasons.append("the first resume pass has not finished")
        if name == "live":
            stale = STALE_HEARTBEATS * int(HEARTBEAT_SECONDS * 1000)
            if now - self.heartbeat_at > stale:
                reasons.append(
                    f"the event loop has not run for {(now - self.heartbeat_at) // 1000} s"
                )
            planned = self.supervisor.scheduler.planned
            if planned is not None and now - planned > SCHEDULER_GRACE_SECONDS * 1000:
                reasons.append(f"the scheduler is {(now - planned) // 1000} s past its plan")
        return ProbeReply(name=name, ok=not reasons, reasons=reasons, checked_at=now)

    def _leases_writable(self) -> str:
        """`""` when this daemon can take a lease, else why not."""
        if self._leases is None:
            directory = self.supervisor.sessions_directory()
            if directory is None:
                return ""
            leases = lease_directory(directory)
            try:
                make_directories(leases)
            except OSError as error:
                return f"the lease directory {leases} cannot be made: {error}"
            self._leases = leases
        if not os.access(self._leases, os.W_OK):
            return f"the lease directory {self._leases} is not writable"
        return ""

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The diagnostic sections of the daemon itself, in the shape of `report()`.

        This is a list because there is already more than one section, and the
        shape was built for the arrival of P5-12. `sections` reached the client
        through one loop, so the isolation section cost one tuple entry here
        and nothing on the client side.
        """
        life = self.lifetime()
        return [
            # This section is named for the *process*. It is beside the section
            # of the socket and not merged with it. One says whether this
            # daemon intends to stay, and the other says whether the socket
            # that it bound survives a logout. A user who reads "lifetime"
            # needs to know which of the two it is.
            (
                "daemon lifetime",
                [
                    ("mode", life.mode),
                    ("holds", " · ".join(life.holds) or "nothing"),
                    ("keep alive", duration(life.keep_alive_ms) if life.keep_alive_ms else "off"),
                ],
            ),
            ("socket lifetime", self.socket_lifetime().describe()),
            # What is held for a credential, by name (T5). A user asks this
            # when a child resumed after a restart and does not move.
            ("credentials awaited", self.supervisor.awaited() or [("held", "nothing")]),
            # The roots whose named profile moved since they started, and which
            # of them wait for a user (S6). `sessions/list` gives the same
            # count for each root.
            ("session profiles", self.supervisor.behind() or [("behind", "nothing")]),
            # The non-guarantees of the daemon itself (N5, I-2). The command
            # that a user runs to ask what this daemon is prints them. Rule 6
            # wants them beside the place where they would be assumed, and this
            # reply *is* that place. Two rows up it says "roots: 7", and that
            # sentence invites every assumption that these rows correct.
            ("isolation", list(NON_GUARANTEES)),
        ]

    def socket_lifetime(self) -> RuntimeLifetime:
        """Whether *this* socket survives logout, asked of the path that it bound.

        `serve()` takes an explicit path, so the socket that a daemon answers
        on and the socket that `resolve_roots()` derives are not always the
        same file. A daemon that reported the lifetime of a directory that it
        does not use would repeat, inside the server, the client-side
        derivation that `daemon/status` exists to prevent.

        The value is read at each call and not captured at start. Lingering can
        be enabled while a daemon runs. A doctor that answered from a snapshot
        taken at boot would keep telling a user to run the command that the
        user just ran.
        """
        return lifetime(self.path)

    async def check_reachable(self) -> str:
        """Is the socket at our path still ours? (P5-11)

        Returns `""` when it is. There are two kinds of no. They need the same
        record and not the same sentence. `removed` is a logout that removes
        `$XDG_RUNTIME_DIR` from under a daemon that keeps running. `replaced` is
        what happens next: the user logs back in, `phern daemon` binds a *new*
        socket at the same path, and two supervisors now believe that they own
        the roots of this user. An existence check reads the second case as a
        recovery, which is the reason why the identity is a `(dev, inode)` pair
        and not a boolean.

        **Deliberately not a shutdown.** The roots keep working, because their
        tasks hold no reference to a connection. That is the inversion that
        P5-01 makes. If a socket problem ended an hour of work in flight, that
        would be the failure that this row guards against, arriving from the
        other side.
        """
        if self.identity is None or self.unreachable_since is not None:
            # Nothing to compare against (a caller that built this by hand), or
            # already latched. Either way there is no transition to find.
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
            # ten timestamps that `Session` assigned. Without these fields, the
            # only way to ask later "were these two sessions in the same
            # incident" is to correlate clock times and a payload that happens
            # to be equal.
            "pid": os.getpid(),
            "since": self.unreachable_since,
            "tier": life.tier,
            "linger": life.linger,
            "advice": life.advice,
        }
        # `error`, not `warning`. For a daemon that was started from a terminal,
        # this line is the only place where the news arrives at the moment it
        # happens. It is the difference between "the agents stopped answering"
        # and a sentence that names the command that prevents it next time.
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

    async def keep_watching(self, watch: EntryWatch) -> None:
        """The socket watch (P5-11, P12-05): run `check_reachable` whenever the kernel
        reports that the entry of the socket may have moved, and never on a clock.

        The check runs one time at the start, for the moment between `serve`
        taking `identity` and arming `watch`. The watch ends when the answer is
        no, because the latch is one-way and there is nothing more to hear. It
        also ends when the watched directory is gone, after the check that finds
        this. `serve` owns the watch and closes it.
        """
        if await self._asked():
            return
        try:
            async for _ in watch.changes():
                if await self._asked():
                    return
        except OSError:
            # The watch itself broke (its descriptor, or the kernel). This is
            # logged one time, and the socket is then checked at each request
            # instead.
            log.exception("ph_app.daemon: the socket watch failed; checking on requests instead")
            self.watch_refused = "the watch failed"
            self.watching = None

    async def _asked(self) -> bool:
        """`check_reachable`, contained: returns whether it found the socket gone.

        A check that fails is logged and the watch continues, as with a planner
        pass that fails. If the failure were raised, it would end the task group
        and every root with it."""
        try:
            return bool(await self.check_reachable())
        except Exception:
            log.exception("ph_app.daemon: the socket check failed")
            return False

    def holds(self, *, now: int | None = None) -> list[Hold]:
        """Why this daemon is still up: every reason, in the order asked.

        **The reasons, and not a yes or no**, because the same four facts answer
        two questions: whether to stop, and what to *tell* a user about why the
        daemon has not stopped (P9-07). They are computed one time, so the
        sidebar and the exit cannot disagree about the life of a daemon.

        `client` means *connected*, not attached. See `connections`. The
        supervisor answers `task` and `schedule`. This object answers
        `keep-alive`, which is last because it is the only reason that expires
        with no event.

        An empty list means that nothing wants the daemon. For a **service**
        daemon that is still not a reason to leave: somebody chose to run a
        supervisor. `spent` decides that.

        `now` is a parameter for the reason that `passivatable` takes one: a
        test asks the question and does not wait for a cadence.

        **Every term has an event behind it** (§5 rule 6). `client`, `task` and
        `schedule` are announced as they move: the first frame of a connection
        and its close, an `agent/status` transition, a `schedule/create` or
        `schedule/cancel`, and `Root.recheck` for the status changes that the
        agent does not announce (a root that parks on a user, the end of the
        ladder). An expiring `keep-alive` ends on a clock, and `lifetime_clock`
        wakes at that moment to ask (P12-02).

        The type is `list[Hold]` and not `list[str]`. The vocabulary is closed
        and the sidebar uses it, so a fifth reason is a type error here, and not
        a blank row in the terminal of a user.
        """
        stamp = now if now is not None else now_ms()
        reasons: list[Hold] = []
        if self.connections:
            reasons.append("client")
        if self.supervisor.busy():
            reasons.append("task")
        if self.supervisor.booked():
            reasons.append("schedule")
        if self.keep_alive_until is not None and stamp < self.keep_alive_until:
            reasons.append("keep-alive")
        return reasons

    def spent(self, *, now: int | None = None, holds: Sequence[Hold] | None = None) -> bool:
        """Whether nothing is left for this daemon to stay up for (P7-08).

        The daemon is spent when it was auto-started and nothing holds it.
        **`EPHEMERAL_QUIET` is no longer part of this.** When this asked the
        whole `passivatable` of P5-05, the exit inherited a sixty-second
        log-quiet window. That window is right for the release of a root and
        wrong for the end of a process: it put a minute between the close of the
        last terminal and the exit of the daemon. The quiet window keeps its job
        on the root side, where it belongs.

        **`served` is the other half of the answer**, and it is here and not in
        `holds()` on purpose. A daemon that still waits for the client that
        spawned it is not *held* by anything, because nobody could name a
        reason. It only may not leave yet. As one of the reasons, it would have
        put a `starting` on the wire and `held · starting` in a sidebar for the
        first thirty seconds of every session. That is a word about the
        internals of the daemon, in the row that answers the question of a user.

        `holds` is the answer that the caller has already computed, for the one
        caller that has one. `check_lifetime` needs it for the announcement in
        any case. A second call would be a second trip to the schedule index
        and, worse, a second *sample*: a turn that ended between the two would
        announce one answer and act on another. It is a parameter, and the
        caller does not inline `ephemeral and not holds`, so that the rule is
        written only here.
        """
        if not self.ephemeral:
            return False
        if self.holds(now=now) if holds is None else holds:
            return False
        stamp = now if now is not None else now_ms()
        return self.served or stamp >= self.spawn_window_ends

    @property
    def mode(self) -> DaemonMode:
        """`ephemeral` or `service`, from the flag that decided it."""
        return "ephemeral" if self.ephemeral else "service"

    def lifetime(self) -> DaemonLifetime:
        """Why this daemon is still here, as the frame that the verb and the
        notification both carry.

        One builder serves the verb, the notification and the section of the
        doctor, so the three cannot describe the same process differently. That
        is the argument of `holds()`, one layer up.
        """
        return DaemonLifetime(
            mode=self.mode,
            holds=self.holds(),
            clients=len(self.connections),
            keep_alive_ms=int(self.keep_alive * 1000),
        )

    def check_lifetime(self) -> None:
        """Announce what holds this daemon, and stop if nothing does.

        **On the event, never on a cadence.** The answer can change at these
        moments, and each of them calls this: the first frame of a connection,
        a connection closing, a turn starting or ending, and `Root.recheck`.
        The close of the last terminal therefore ends an auto-started daemon at
        once. The sidebar of the terminal *beside* it also learns that a turn
        holds the process while the turn is still running. `lifetime_clock`
        calls this at the two moments that end on a clock: the expiry of a
        keep-alive, and the close of the spawn window.

        The announcement comes first and runs even when the answer is
        "nothing". A front end needs that frame most: it turns `daemon exits on
        detach` into the sentence that the user acts on.

        **One sample, two uses.** The announcement and the exit read the same
        `DaemonLifetime` and do not ask twice. `holds()` promises this: its
        reasons are computed one time, so that the sidebar and the exit cannot
        disagree. A second call would have silently broken that promise. One
        sample is also cheaper: `holds()` reaches `booked()`, which touches the
        schedule index, and this runs on every turn transition.
        """
        current = self.lifetime()
        self._announce_lifetime(current)
        if self.spent(holds=current.holds):
            log.info("ph_app.daemon: nothing left to serve; stopping (auto-started)")
            self.stop.set()

    def _announce_lifetime(self, current: DaemonLifetime) -> None:
        """Tell every connected client, when the answer has moved.

        **On change**, because this runs on every turn transition. Without the
        check, a daemon with four terminals open would write four identical
        frames for each prompt. That is the fan-out that `SessionEventNotice` is
        careful about, for a fact that moves a hundred times less often.

        The frame goes to `connections` and not to the subscribers of a root.
        It is a fact about the process, so the audience is every client that is
        connected to the process, including a client that watches another
        session. That is the second question that the connection *set* exists
        to answer.
        """
        if current == self.announced:
            return
        self.announced = current
        wire = current.to_wire()
        for connection in list(self.connections):
            try:
                connection.notify(DaemonLifetime.METHOD, wire)
            except Exception:
                # A client that cannot keep up is for the root to drop, and not
                # for this method. `notify` raises so that the *subscriber*
                # list removes the client, and this list is membership of the
                # socket and not of a feed. One lost lifetime frame costs a
                # stale sidebar line, and the next change sends another frame.
                log.debug("ph_app.daemon: a client did not take the lifetime frame")

    async def _handle(self, stream: ByteStream) -> None:
        connection = _Connection(stream=stream, server=self)
        self.connections.add(connection)
        try:
            async with stream:
                await connection.serve()
        finally:
            # This is in `finally`, because membership is a claim on the
            # *process*. A connection that ended in a crash and was never
            # removed would keep an ephemeral daemon resident forever. The
            # symptom, a daemon that does not leave, points nowhere near here.
            self.connections.discard(connection)
            # **A knock is not a departure** (see `_Connection.spoke`), so it
            # arms no window. The *check* is unconditional, because `spent()`
            # now knows about the spawn window itself. A knock that outlived
            # the last real client used to leave the exit to a sweep. A check
            # here is both correct and immediate.
            if connection.spoke and not self.connections and self.keep_alive > 0:
                self.keep_alive_until = now_ms() + int(self.keep_alive * 1000)
                self.lifetime_clock.notice()
            self.check_lifetime()


async def _stop_on_signals(
    server: DaemonServer, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
) -> None:
    """`SIGTERM` or `SIGINT` asks for the same stop that a `shutdown` frame asks for.

    Without this, the default action ends the process at once, with the effect
    of a `SIGKILL`: what the logs of the roots still owe is not flushed, no
    lease is given back on the ordinary path, and the socket is left for the
    `_clear_stale` of the next daemon. `kill`, `systemctl stop`, `launchctl`
    and a Ctrl-C in a foreground `phern daemon` all arrive this way. On the
    first signal, the teardown runs under its own budget (`GRACE_SECONDS`, in
    the `finally` of `serve`), and the children of every root are suspended in
    their own logs as it runs. A **second** signal while the teardown runs, or
    a signal after a `shutdown` frame has already started it, means that the
    user wants the process to leave now.

    **The shared rule** (`ph.resources.stop_on_signals`): to leave at once kills
    what this process still has in the orphan journal before the process goes.
    The hard stop that the first signal arms does the same. That stop fires if
    the teardown has not ended within `SHUTDOWN_SECONDS`. A teardown that does
    not end therefore cannot become a process that does not end, and cannot
    leave a shell command running behind it.

    This is started as a task in the group of `serve`. It is therefore alive for
    the whole teardown. It is canceled with the listener at the end, and that
    cancel disarms the hard stop.
    """

    def stopping(signum: int) -> bool:
        if server.stop.is_set():
            return False
        log.info("ph_app.daemon: stopping on %s", signal.Signals(signum).name)
        server.stop.set()
        return True

    await stop_on_signals(stopping, task_status=task_status)


@asynccontextmanager
async def _cancel_is_a_stop(stop: anyio.Event) -> AsyncIterator[None]:
    """Run the block shielded from a cancel that is meant for `serve`, and turn that
    cancel into `stop`: the orderly teardown first, then the cancel passed on to
    the caller.

    **A cancel reached every task at once.** The turn of a root was canceled
    before the `finally` of `serve` began, so the turn closed as `aborted` and
    as stopped by the user. The note of the shutdown on the live-roots index
    then found a root that owed nothing, and the next daemon never resumed work
    that nobody had stopped. The listener, the planners and any mount in flight
    ended the same way, before the teardown that was written to end them in
    order. Shielded, they see only `stop`. The teardown answers `stop` as it
    answers a `shutdown` frame, under `GRACE_SECONDS`.

    **The relay is outside the shield**, and it is the one task that the cancel
    still reaches. It is *started* and not spawned, so a cancel before it runs
    fails `serve` here, and does not leave a shielded daemon that nothing can
    end. Nothing bounds the teardown except its own deadline, as after a
    `shutdown` frame. What the teardown waits on is shielded itself (`aclose`,
    `Context.dispose`), so a clock here would reach none of it.
    """

    async def relay(*, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
        task_status.started()
        try:
            await anyio.sleep_forever()
        finally:
            stop.set()

    async with anyio.create_task_group() as relaying:
        await relaying.start(relay)
        with anyio.CancelScope(shield=True):
            yield
        relaying.cancel_scope.cancel()
    # The group absorbs its own cancel and exits with no checkpoint. The cancel
    # of the caller would therefore end here unreported, and `serve` would
    # return into a call that was abandoned.
    await anyio.lowlevel.checkpoint_if_cancelled()


async def _clear_stale(path: Path) -> None:
    """Remove a socket that nobody listens on. Refuse one that somebody listens on.

    A stale path is the ordinary result of a crash, and it makes every client
    hang on a connect that is never answered. A *live* path is another daemon.
    To take its socket would leave two supervisors that both believe they own
    the roots of this user. That is the question of I-5, and P5-03 answers it,
    so here it is a refusal and not a race.
    """
    if not path.exists():
        return
    if await listening(path):
        raise DaemonUnavailable(f"a daemon is already listening on {path}")
    path.unlink(missing_ok=True)


async def serve(
    profile: Profile,
    *,
    choice: ModelChoice = ModelChoice(),
    passivate_after: float | None = PASSIVATE_AFTER,
    wake_within: float | None = WAKE_WITHIN,
    ephemeral: bool = False,
    keep_alive: float = 0.0,
    scheduling: bool = True,
    check_invariants: bool = True,
    path: Path | None = None,
    ready: anyio.Event | None = None,
    started: Callable[[DaemonServer], None] | None = None,
    signals: bool = False,
    probe: str | None = None,
    resume_owed: bool = False,
    continue_interrupted: bool = False,
) -> int:
    """Run the supervisor until `shutdown`. Returns the number of session logs that
    could not be written when it stopped (P13-08), which is the exit status that
    `phern daemon` reports.

    `probe` is a `host:port` on which to answer the startup, readiness and
    liveness probes over TCP (P13-12, `ph_app.daemon.probes`). With `None`, they
    are answered only on the unix socket.

    `resume_owed` runs `Supervisor.resume_owed` over every owed root after the
    socket is accepting (P13-03). Readiness is no until the pass ends. It is the
    pass that `phern agents resume --all` would run, for a unit that has no
    `ExecStartPost`. `continue_interrupted` is `Supervisor.continue_interrupted`
    (P13-04).

    `ready` is an `anyio.Event` that is set when the socket is accepting. A
    caller (a test, or `phern agents` when it starts a daemon on demand) can
    wait for the socket to open and does not have to poll for the file to
    appear. The file exists before the socket is listening, and a poll would
    land in exactly that window.

    `signals` makes `SIGTERM` and `SIGINT` the same stop as a `shutdown` frame
    (`_stop_on_signals`). It is off by default and on only for `phern daemon`.
    A process-wide handler belongs to the process that *is* the daemon, and not
    to a test or an embedding host that runs `serve` inside something else.

    **A cancel is that stop too** (`_cancel_is_a_stop`), for a host that runs
    `serve` and ends it as it ends any task. The call still ends canceled, after
    the daemon has stopped.
    """
    socket_path = path or resolve_roots().ensure().daemon_socket()
    await _clear_stale(socket_path)
    # The socket is bound *before* the task group, beside the stale check that
    # it belongs with. Binding is a precondition. A precondition that fails
    # inside a group comes back wrapped in an `ExceptionGroup`, which the
    # `except` of `phern daemon` cannot see. For exactly that reason, a
    # `$PH_RUNTIME` deep enough to exceed the 107-byte path limit of `AF_UNIX`
    # printed a full traceback. Nothing has been built at this point, so the
    # teardown below would have had nothing to clean up either.
    try:
        listener = await anyio.create_unix_listener(socket_path)
        # The socket carries every command that the agents of this user take,
        # so only this user may use it. `$PH_RUNTIME` is 0o700 for the same
        # reason.
        os.chmod(socket_path, 0o600)
    except OSError as error:
        raise DaemonUnavailable(f"cannot listen on {socket_path}: {error}") from error
    # The identity is taken here, immediately after the bind and before
    # anything can have replaced the socket. This is the one moment at which
    # "the socket at this path" and "the socket this daemon is listening on"
    # are the same file by construction and not by assumption (P5-11). It is a
    # local as well as a field, because the teardown below needs it whether or
    # not the server was ever built.
    identity = socket_identity(socket_path)
    probes = None
    if probe is not None:
        try:
            host, port = parse_bind(probe)
            probes = await anyio.create_tcp_listener(local_host=host, local_port=port)
        except (OSError, ValueError) as error:
            # The socket that this daemon just bound is removed with the
            # refusal, so the next start finds no socket, and not a stale one
            # to clear.
            await listener.aclose()
            socket_path.unlink(missing_ok=True)
            raise DaemonUnavailable(f"cannot listen for probes on {probe}: {error}") from error
    # The watch is armed right after, for the same reason: nothing would hear a
    # change between the two (P12-05).
    watch: EntryWatch | None = None
    refused: str | None = None
    try:
        watch = EntryWatch(socket_path)
    except WatchUnavailable as error:
        refused = str(error)
        log.error(
            "ph_app.daemon: cannot watch %s (%s); it is checked at each request from "
            "a connected client instead",
            socket_path,
            error,
        )
    # The watch has its own scope, so that teardown can end the watch before it
    # unlinks the socket. That unlink is this daemon closing its own socket,
    # not losing it.
    watching = anyio.CancelScope()
    stop = anyio.Event()
    async with _cancel_is_a_stop(stop), anyio.create_task_group() as tasks:
        # The supervisor is built inside the group, so that `tasks` is a
        # required field and not an Optional with a "not serving" guard. A
        # supervisor that cannot start a root is a state that must not be
        # representable.
        supervisor = Supervisor(
            profile=profile,
            tasks=tasks,
            choice=choice,
            passivate_after=passivate_after,
            wake_within=wake_within,
            check_invariants=check_invariants,
            continue_interrupted=continue_interrupted,
        )
        try:
            server = DaemonServer(
                supervisor=supervisor,
                stop=stop,
                path=socket_path,
                watching=watch.mechanism if watch is not None else None,
                watch_refused=refused,
                ephemeral=ephemeral,
                keep_alive=keep_alive,
                identity=identity,
                probe_address=_bound_address(probes) if probes is not None else "",
                resumed=not resume_owed,
            )
            # The jobs that end on a clock. Each is its own task, and it sleeps
            # until its next moment and does not poll (`ph_app.daemon.planner`).
            # Release, the invariant check and the lifetime clock are always
            # started. When one has nothing to plan for (passivation off, the
            # check off, a service daemon with no keep-alive), it sleeps until
            # it is told, which costs nothing.
            tasks.start_soon(supervisor.releaser.keep, server.stop)
            tasks.start_soon(supervisor.verifier.keep, server.stop)
            tasks.start_soon(server.lifetime_clock.keep, server.stop)
            if signals:
                # Started rather than scheduled: a signal that came before the
                # receiver was in place would take the default action.
                await tasks.start(_stop_on_signals, server)
            if scheduling:
                # See `Supervisor.scheduler` and docs/seams/schedule.md.
                tasks.start_soon(supervisor.scheduler.keep, server.stop)
            if watch is not None:

                async def keep_watching(armed: EntryWatch) -> None:
                    with watching:
                        await server.keep_watching(armed)

                tasks.start_soon(keep_watching, watch)
            if probes is not None:
                log.info("ph_app.daemon: probes on %s", server.probe_address)
                tasks.start_soon(_keep_probing, server, probes)
            tasks.start_soon(keep_heartbeat, server, server.stop)
            if started is not None:
                # The server is handed out here, and is not reached through the
                # socket. A test whose subject is the concurrency of the
                # supervisor has no wire question to ask. To reach the
                # supervisor through a client would test the transport in order
                # to get at something behind it.
                started(server)
            # The accept loop has its own group. The group is ended at the stop
            # and waited for before the listener closes. When the listener was
            # closed under the loop, a client that connected at that moment
            # woke `accept` on a closed socket and failed the group of the
            # daemon. The handlers run in `tasks`, so the end of the loop ends
            # no exchange in flight, and that includes the `shutdown` request.
            async with listener, anyio.create_task_group() as accepting:
                accepting.start_soon(partial(listener.serve, server._handle, task_group=tasks))
                server.serving = True
                if resume_owed:
                    # The pass starts after the socket is open. A client that
                    # attaches during the pass then shares the mount and does
                    # not wait for the pass. Readiness says no until it ends.
                    tasks.start_soon(_resume_at_boot, server)
                if ready is not None:
                    ready.set()
                await server.stop.wait()
                accepting.cancel_scope.cancel()
        finally:
            notify("STOPPING=1")
            # **The watch ends before anything below touches the socket.** The
            # unlink is this daemon closing its own socket. A watch that still
            # ran would hear it as the socket being taken, and would say so in
            # every root. The watch is canceled before the first await below,
            # so it cannot run in between.
            watching.cancel()
            # **The socket goes first, and only if it is still ours** (E2).
            #
            # *Only if ours*, because `check_reachable` latches `replaced` for a
            # reason: the user logged back in, a second `phern daemon` bound a
            # new socket at this path, and the file here is now the socket of
            # **that** daemon. With an unconditional unlink, the exit of this
            # daemon removed the socket of the live one. The roots that the
            # live daemon served became unreachable, for no reason that anybody
            # could see from either process.
            #
            # *First*, because the drain below may take `GRACE_SECONDS` and the
            # listener has already stopped accepting. A file that exists but
            # refuses is the one state that a client cannot act on. The client
            # spawns a daemon, the new daemon binds this same path, and the old
            # teardown then deleted the socket of the new daemon. Removing the
            # file first turns that window into a plain "no daemon", which is a
            # state that the client already handles.
            if identity is not None and socket_identity(socket_path) == identity:
                socket_path.unlink(missing_ok=True)
            # Shielded, and bounded. A cancel from whoever started `serve()`
            # stops at `_cancel_is_a_stop`. But the group cancels itself when
            # one of its tasks fails, and an unwind that is cut short here
            # loses everything that teardown is for: sessions are not flushed,
            # worktrees are not reclaimed (F6), and the leases of P5-03 are
            # never released, so the next daemon refuses a session whose holder
            # is already gone.
            # **Both halves, and they do different jobs.** The scope shields the
            # work of `aclose` itself: the flushes, the channel closes, the loop
            # between roots. An outer cancellation would otherwise cut that
            # work, leave a lease held, and make the next daemon refuse the
            # session. The *deadline* is what bounds the roots.
            # `Context.dispose` shields itself, so a deadline out here could
            # never reach inside one, and each root would have started a fresh
            # `GRACE_SECONDS` of its own: ten roots, ten budgets, from a line
            # that says ten seconds.
            #
            # Both use one instant, so they end together. A root that does not
            # unwind must not become a process that does not exit. With two
            # constants for one tunable, the budget that was set stops being
            # the budget that fires.
            until = anyio.current_time() + GRACE_SECONDS
            with anyio.CancelScope(deadline=until, shield=True):
                await supervisor.aclose(deadline=until)
            # Last, cancel whatever still runs: the planners, a connection that
            # is still open, and a turn whose log the release of its root has
            # already written.
            tasks.cancel_scope.cancel()
            if watch is not None:
                watch.close()
            if probes is not None:
                await probes.aclose()
    return len(supervisor.unwritten)


async def _resume_at_boot(server: DaemonServer) -> None:
    """`--resume-owed`: one pass over every owed root, with readiness no until the
    pass ends. Whatever the pass finds is written to the log of the daemon, one
    line for each root, in the way that `phern agents resume` prints it. No
    client is attached to be told in another way."""
    try:
        report = await server.supervisor.resume_owed(all_owed=True)
    except Exception:
        log.exception("ph_app.daemon: the resume pass failed")
    else:
        for row in report.sessions:
            log.info(
                "ph_app.daemon: resume pass: %s %s status=%s detail=%s",
                row.session_id,
                row.outcome,
                row.status or "-",
                row.detail or "-",
            )
        if not report.sessions:
            log.info("ph_app.daemon: resume pass: no root owes work")
        elif not report.trusted:
            log.warning(
                "ph_app.daemon: resume pass: the live-roots index could not vouch for itself"
            )
    finally:
        server.resumed = True


async def _keep_probing(server: DaemonServer, probes: MultiListener[SocketStream]) -> None:
    """Answer probes until the listener is closed under the accept loop. Teardown
    ends the job that way, so a closed listener is the end of the job and not a
    failure."""
    with suppress(anyio.ClosedResourceError):
        await probes.serve(partial(answer_probe, server))


def _bound_address(probes: MultiListener[SocketStream]) -> str:
    """The `host:port` that a TCP listener actually bound. Port 0 asks the kernel
    for a port, and the first of the per-family listeners is the one that a
    loopback bind made."""
    try:
        address = probes.listeners[0].extra(SocketAttribute.local_address)
    except LookupError:
        return "?"
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)
