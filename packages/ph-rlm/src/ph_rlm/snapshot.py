"""`rlm-kernel-snapshot`: the persistent namespace, in the log (D17).

This module is what makes the declaration `persistence: "namespace"` true. The
seam takes the *promise* of the provider at registration (D6). These events are
the provider keeping that promise. Without them the runtime would be exactly what
dsh refused to ship: a REPL whose state between calls is invisible to the log,
with a declaration on top.

There are four decisions, in order of importance.

**A snapshot is per variable, not per namespace.** An unchanged 200 MiB DataFrame
emits nothing, because its digest did not change. A snapshot of the namespace as
one blob would append that DataFrame again on every cell that changed anything.
The log would then grow with the *size of the namespace* and not with the size of
the change.

**A blob is written, then recorded** (§4.9, S11). A blob is durable at its
locator before the event that names it is appended (`SpillStore.try_save`). A run
that dies between the two leaves a blob that nothing names, and the next read of
the log collects it. A blob that could not be written is recorded as a `clear`. A
blob that cannot be read on restore is reported by `kernel/restored` as failed.
Nothing is dropped silently.

**`patch` is deliberately absent.** D17 allows a `bsdiff4` delta chain against an
anchor *and* says to benchmark first. `dill` output is not byte-stable across
processes in the way that a QuickJS heap image is. Memo ordering and bytes
derived from `id()` change even when the value does not. The digest of each
variable is what actually keeps growth linear. `SNAPSHOT_KINDS` names `patch` so
that the vocabulary is ready.

**The tag is provenance, not secrecy.** The tag is an HMAC-SHA256 keyed by the
session id. A blob from another session, or a mangled file, fails verification
and is not unpickled. Anyone who can write the log can write the tag too, so the
tag is not a defense against a hostile writer to the filesystem. It is a defense
against the mistake that actually happens: a blob restored into the wrong
session, or a half-written file, unpickled into a namespace as if it were sound.

@module ph_rlm.snapshot
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from ph.cordis import Context, plugin
from ph.json import as_str
from ph.keys import AGENTS, COMPACTION, SESSIONS, SPILL_STORE
from ph.seams.compaction import CompactionNote
from ph.seams.spill import PlannedBlob, SpillClaim
from ph.session import Session
from ph.session.writers import log_writer
from ph.wire import WireModel

from .keys import KERNEL_SNAPSHOTS, PYTHON_RUNTIME

_LOG = log_writer(__name__)

__all__ = [
    "INLINE_BLOB_MAX",
    "RESERVED_KINDS",
    "SNAPSHOT_KINDS",
    "KernelSnapshotPolicy",
    "SnapshotKind",
    "apply",
    "fold_namespace",
    "namespaces_in",
    "render_live_variables",
]

log = logging.getLogger("ph_rlm.snapshot")

SnapshotKind: TypeAlias = Literal["snap", "patch", "clear", "recipe"]
SNAPSHOT_KINDS: Final[tuple[SnapshotKind, ...]] = ("snap", "patch", "clear", "recipe")
"""The full vocabulary D17 defines."""

RESERVED_KINDS: Final[frozenset[str]] = frozenset({"patch", "recipe"})
"""The kinds that are defined here and that nothing writes, with what each one
waits for.

`patch` waits for a benchmark. A `bsdiff4` chain over `dill` bytes is worth its
complexity only if the delta ratio justifies it, and the module docstring argues
why that is not obvious. `recipe` needs a *harness-owned declarative load of a
kernel variable*. The corpus of P3-17 is the obvious candidate, and it is
deliberately not one, because it never enters the kernel (see
`ph_rlm.context_loader`).

The set is declared beside the vocabulary and not in a test. Whoever implements a
kind reads this module, and the conformance suite reads this set."""

INLINE_BLOB_MAX: Final = 64 * 1024
"""Above this size a payload goes to `ctx.spill_store`, and the event carries the
locator. Below it the event carries the bytes, so a small namespace needs no
second file to be restorable."""


class SnapshotRecord(WireModel):
    """One `kernel/snapshot` payload."""

    kind: SnapshotKind
    var: str
    digest: str | None = None
    bytes: int | None = None
    blob: str | None = None
    """Base64, when the payload is small enough to be in the event."""
    locator: str | None = None
    """The locator of the payload, when the payload is not in the event."""
    tag: str | None = None
    reason: str | None = None
    """Why a variable was cleared: `deleted`, `too-large`, `unpicklable`."""


def _tag(session_id: str, payload: bytes) -> str:
    return hmac.new(session_id.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def fold_namespace(session: Session, namespace: str) -> dict[str, SnapshotRecord]:
    """The current state of one namespace, folded from its events.

    This is a fold, so `ctx.sessions.fork(source, boundary)` rebuilds the
    namespace *as of the boundary* by folding only the events at or before it. A
    side file could not express that. It would give every fork the latest
    namespace of the parent, and D17 was chosen to avoid exactly that.
    """
    state: dict[str, SnapshotRecord] = {}
    for event in session.events:
        if event.type != "kernel/snapshot":
            continue
        if as_str(event.data.get("namespace")) != namespace:
            continue
        try:
            record = SnapshotRecord.model_validate(event.data.get("record"))
        except Exception:
            continue
        if record.kind == "clear":
            state.pop(record.var, None)
        else:
            state[record.var] = record
    return state


def namespaces_in(session: Session) -> set[str]:
    """Every namespace that the log of this session has a snapshot for."""
    return {
        str(namespace)
        for event in session.events
        if event.type == "kernel/snapshot"
        if isinstance(namespace := event.data.get("namespace"), str)
    }


@dataclass(slots=True)
class KernelSnapshotPolicy:
    """Turns the snapshot frames of the runtime into events, and back again.

    The provider knows about processes and frames. This class knows about the
    log. Because of the split, neither has to know both.
    """

    ctx: Context
    inline_blob_max: int = INLINE_BLOB_MAX

    def _session(self, namespace: str) -> Session | None:
        """The session that the events of a namespace belong in.

        The session is read from `ctx.agents`. It is not kept in a side table
        that `agent/created` feeds. The namespace key *is* the agent id, and the
        registry already knows every live agent. A second copy of that mapping
        would be a second thing to keep correct.
        """
        agent = self.ctx.require(AGENTS).get(namespace)
        return agent.session if agent is not None else None

    # ------------------------------------------------------------- recording --

    async def record(self, namespace: str, run_id: int, variables: list[dict[str, Any]]) -> None:
        """Append one `kernel/snapshot` for each changed variable, and put the
        blobs where the records say they are.

        **Each blob is written first. Then the records follow, as one batch for
        the cell** (S11). Every other producer keeps this order
        (`SpillStore.try_save`). When the blobs were written after the records, a
        blob that the store could not take left a `snap` that named nothing, and
        the next restore dropped it silently. Now a blob that cannot be written
        is recorded as a `clear` that says so, and a restore tells the model
        that.

        The records are in the log but not yet on disk. The barrier of the next
        request puts the records on disk before the model reads the result of the
        cell, and a blob is durable before any record names it. The blobs are
        written together (`try_save_all`): all at the same time, each file
        synced, and their one directory synced one time and not one time for each
        variable. The `done` of the guest waits for all of it.
        """
        session = self._session(namespace)
        if session is None:
            return
        encoded = [
            one for raw in variables if (one := self._encode(session, namespace, raw)) is not None
        ]
        if not encoded:
            return
        spill = self.ctx.get(SPILL_STORE)
        # A blob comes back only when a store is mounted to take it.
        blobs = [
            (blob, f"kernel variable {record.var}") for record, blob in encoded if blob is not None
        ]
        saved = iter(
            await spill.try_save_all(blobs) if spill is not None and blobs else [None] * len(blobs)
        )
        with session.batch() as batch:
            for record, blob in encoded:
                if blob is not None and next(saved) is None:
                    record = SnapshotRecord(
                        kind="clear",
                        var=record.var,
                        reason="too-large (the blob could not be stored)",
                    )
                _LOG.append(
                    batch,
                    "kernel/snapshot",
                    {"namespace": namespace, "run": run_id, "record": record.to_wire()},
                )

    def _encode(
        self, session: Session, namespace: str, raw: dict[str, Any]
    ) -> tuple[SnapshotRecord, PlannedBlob | None] | None:
        """The record to append, and the blob to write for it. The blob is `None`
        for an inline record or a `clear`, which have nothing for the store.

        The payload is decoded **one time** and passed on. To base64-decode it
        for the tag, decode it again to write the blob, and hash it a third time
        to derive the path would be three passes that recompute what is already
        available.
        """
        name = raw.get("var")
        if not isinstance(name, str):
            return None
        skipped = raw.get("skipped")
        if skipped is not None:
            # The guest could not store the variable. It is recorded as a `clear`
            # with the reason, so that a restore *tells the model that the name
            # is gone*. Otherwise the model would find an undefined name in the
            # middle of a session and read that as a bug.
            return SnapshotRecord(kind="clear", var=name, reason=str(skipped)), None
        blob = raw.get("blob")
        digest = raw.get("digest")
        if not isinstance(blob, str) or not isinstance(digest, str):
            return None
        payload = base64.b64decode(blob)
        tag = _tag(session.id, payload)
        if len(payload) <= self.inline_blob_max:
            inline = SnapshotRecord(
                kind="snap", var=name, digest=digest, bytes=len(payload), blob=blob, tag=tag
            )
            return inline, None
        spill = self.ctx.get(SPILL_STORE)
        if spill is None:
            return (
                SnapshotRecord(
                    kind="clear", var=name, reason="too-large (no spill store is mounted)"
                ),
                None,
            )
        # The store derives the locator, so the record can name it and this
        # module does not mirror the naming rule of the store. The blob is
        # planned one time: the locator here and the write in `record` share one
        # digest of the payload.
        planned = spill.plan(
            owner=_owner(namespace), suggested_name=f"{name}.dill", content=payload
        )
        spilled = SnapshotRecord(
            kind="snap",
            var=name,
            digest=digest,
            bytes=len(payload),
            locator=str(planned.locator),
            tag=tag,
        )
        return spilled, planned

    # ------------------------------------------------------------ restoring --

    async def materialize(self, namespace: str) -> tuple[list[dict[str, Any]], list[str]]:
        """The payloads to give a kernel that just started, and the names whose
        payload could not be read.

        **The second half is the one that was missing** (S11). A blob that was
        gone, torn, or from another session was skipped. It then appeared
        nowhere, neither as restored nor as failed. When every variable dropped,
        nothing at all was recorded: `df` was undefined after a restart, with no
        notice. Now each such name is returned, so `kernel/restored` reports it
        as failed.
        """
        session = self._session(namespace)
        if session is None:
            return [], []
        variables: list[dict[str, Any]] = []
        unreadable: list[str] = []
        for record in fold_namespace(session, namespace).values():
            payload = await self._payload(session, record)
            if payload is None:
                if record.kind == "snap":
                    unreadable.append(record.var)
                continue
            variables.append({"var": record.var, "blob": base64.b64encode(payload).decode("ascii")})
        return variables, unreadable

    async def _payload(self, session: Session, record: SnapshotRecord) -> bytes | None:
        if record.blob is not None:
            payload = base64.b64decode(record.blob)
        elif record.locator is not None:
            spill = self.ctx.get(SPILL_STORE)
            if spill is None:
                return None
            try:
                payload = await spill.load_bytes(record.locator)
            except OSError:
                return None
        else:
            return None
        if record.tag is not None and not hmac.compare_digest(
            record.tag, _tag(session.id, payload)
        ):
            # A blob from another session, or a mangled one. It is refused and
            # not unpickled, because `dill.loads` on arbitrary bytes executes
            # arbitrary code.
            log.warning("ph_rlm.snapshot: %s failed verification and was not restored", record.var)
            return None
        return payload

    async def restored(self, namespace: str, outcome: dict[str, Any]) -> None:
        """Record what came back, so the model can be told what did not."""
        session = self._session(namespace)
        if session is None:
            return
        _LOG.append(session, "kernel/restored", {"namespace": namespace, **outcome})


def render_live_variables(session: Session) -> str:
    """The names that the kernel still holds, for a compaction summary prompt
    (G10).

    Compaction rewrites a *surface*. A REPL is not on that surface, so every
    variable here survives the cut completely untouched. That is exactly why the
    summary has to say so. The model is about to read "the conversation history
    has been summarized". From the conversation alone, it cannot tell whether the
    namespace went too. A model that assumed that it did would spend the next
    cell on recomputing a DataFrame that it already has. D17 exists so that the
    model does not have to.

    The names are read from the same fold that `materialize` restores from, so
    what the summary names and what a restart brings back cannot disagree. The
    result is empty when there is no kernel state. An empty result contributes
    nothing, and not a paragraph that says so.
    """
    lines: list[str] = []
    for namespace in sorted(namespaces_in(session)):
        names = sorted(record.var for record in fold_namespace(session, namespace).values())
        if names:
            lines.append(f"- namespace `{namespace}`: " + ", ".join(f"`{name}`" for name in names))
    if not lines:
        return ""
    return (
        "The Python kernel is still running and still holds these variables. "
        "Summarizing the conversation does not clear them: every name below is "
        "still defined and can be used in the next cell without recomputing it.\n"
        + "\n".join(lines)
    )


def _owner(namespace: str) -> str:
    """The spill owner that holds the blobs of one namespace."""
    return f"kernel/{namespace}"


class Config(WireModel):
    """Row config for the snapshot policy."""

    inline_blob_max: int = INLINE_BLOB_MAX


@plugin(
    "rlm-kernel-snapshot", affects="environment", config=Config, inject=[SESSIONS, PYTHON_RUNTIME]
)
async def apply(ctx: Context, config: Config) -> None:
    """Wire the policy to the runtime provider and to session open.

    The row injects `python_runtime` and not `code_runtime`. A mounted seam says
    nothing about *which* provider answered it, and this row exists to keep the
    promise of one specific provider. Activation is driven by service
    availability, so naming the real dependency is also what orders the two rows.
    """
    policy = KernelSnapshotPolicy(ctx=ctx, inline_blob_max=config.inline_blob_max)
    ctx.provide(KERNEL_SNAPSHOTS, policy)
    runtime = ctx.require(PYTHON_RUNTIME)

    def take_back() -> None:
        # The policy is taken back with the row. Without that, the runtime went
        # on recording into a policy whose row was gone. It is taken back only
        # while it is still the policy of this row, as `claim_slot` releases
        # (Holdings rule 4).
        if runtime.snapshots is policy:
            runtime.snapshots = None

    # The take-back is registered before the write that it undoes
    # (Holdings rule 1).
    ctx.add_disposer(take_back, label="rlm-kernel-snapshot(runtime)")
    runtime.snapshots = policy

    def claim_kernel_blobs(scope: Context) -> None:
        """Contribute the blobs of this row to the open-time sweep of the seam
        (P6-15).

        This goes through `ctx.inject` for the same reason as
        `announce_live_variables`. The store is optional here, because a small
        namespace is stored inline. A bare `ctx.get` at mount reads `None` under
        a profile that orders `spill-local` later, and the sweep would then
        silently collect nothing.
        """
        scope.require(SPILL_STORE).claim(
            SpillClaim(
                label="rlm-kernel-snapshot",
                event_type="kernel/snapshot",
                owner=lambda data: _owner(as_str(data["namespace"])),
                locator=lambda data: (data.get("record") or {}).get("locator"),
            ),
            scope=scope,
        )

    ctx.inject([SPILL_STORE], claim_kernel_blobs, label="rlm-kernel-spill-claim")

    def announce_live_variables(scope: Context) -> None:
        """Tell a compaction summary what the kernel still holds (G10, P4-03).

        This goes through `ctx.inject` and not through the `inject` list of this
        row. The compaction seam is a *nice to have* here, and a deployment that
        removed the compaction row must not lose kernel snapshots because of it.
        When the seam appears, the note registers. When the seam goes, the scope
        disposes and takes the registration with it.
        """
        scope.require(COMPACTION).note(
            CompactionNote(name="rlm:live-variables", text=render_live_variables), scope=scope
        )

    ctx.inject([COMPACTION], announce_live_variables, label="rlm-compaction-note")
