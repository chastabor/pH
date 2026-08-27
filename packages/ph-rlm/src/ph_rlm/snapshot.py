"""`rlm-kernel-snapshot` — the persistent namespace, in the log (D17).

This is the module that earns `persistence: "namespace"`. The seam takes the
provider's *promise* at registration (D6); these events are the promise being
kept. Without them the runtime would be exactly what dsh refused to ship — a
REPL whose cross-call state is invisible to the log — with a declaration on top.

Four decisions, in the order they matter.

**Per variable, not per namespace.** An unchanged 200 MiB DataFrame emits
nothing, because its digest did not move. Snapshotting the namespace as one blob
would append that DataFrame again on every cell that touched anything at all,
and the log would grow with the *size of the namespace* rather than the size of
the change.

**The event is appended before the blob is written.** Write-ahead ordering
(§4.9): a death between the two yields an event whose blob is missing — which
`kernel/restored` reports as a failed variable — rather than a blob nothing
references, which nothing would ever find or collect. The orphan case is swept
at session open (F7); the dangling-reference case is self-describing.

**`patch` is deliberately absent.** D17 allows a `bsdiff4` delta chain against
an anchor, *and* says to benchmark first because `dill` output is not byte-stable
across processes the way a QuickJS heap image is — memo ordering and
`id()`-derived bytes move even when the value does not. The plan's own fallback
is "snap-only, still log-resident". Per-variable digesting is what actually keeps
growth linear, and it is here; a delta chain over unstable bytes would add a
re-anchoring policy and a second failure mode for a gain nobody has measured.
`SNAPSHOT_KINDS` names `patch` so the vocabulary is ready when someone does.

**The tag is provenance, not secrecy.** HMAC-SHA256 keyed by the session id, so a
blob from another session or a mangled file fails verification instead of being
unpickled. Anyone who can write the log can write the tag too — this is not a
defence against a hostile filesystem writer, it is a defence against the mistake
that actually happens: a blob restored into the wrong session, or a half-written
file, unpickled into a namespace as if it were sound.

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
from ph.session import Session
from ph.wire import WireModel

__all__ = [
    "INLINE_BLOB_MAX",
    "SNAPSHOT_KINDS",
    "KernelSnapshotPolicy",
    "SnapshotKind",
    "apply",
    "fold_namespace",
    "namespaces_in",
    "referenced_locators",
]

log = logging.getLogger("ph_rlm.snapshot")

SnapshotKind: TypeAlias = Literal["snap", "patch", "clear", "recipe"]
SNAPSHOT_KINDS: Final[tuple[SnapshotKind, ...]] = ("snap", "patch", "clear", "recipe")
"""The full vocabulary D17 defines. `patch` is reserved, not emitted — see the
module docstring for why a delta chain over `dill` bytes is benchmark-gated."""

INLINE_BLOB_MAX: Final = 64 * 1024
"""Above this a payload goes to `ctx.spill_store` and the event carries the
locator. Below it the event carries the bytes, so a small namespace needs no
second file to be restorable."""


class SnapshotRecord(WireModel):
    """One `kernel/snapshot` payload."""

    kind: SnapshotKind
    var: str
    digest: str | None = None
    bytes: int | None = None
    blob: str | None = None
    """Base64, when the payload is small enough to live in the event."""
    locator: str | None = None
    """Where the payload went, when it was not."""
    tag: str | None = None
    reason: str | None = None
    """Why a variable was cleared: `deleted`, `too-large`, `unpicklable`."""


def _tag(session_id: str, payload: bytes) -> str:
    return hmac.new(session_id.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def fold_namespace(session: Session, namespace: str) -> dict[str, SnapshotRecord]:
    """The current state of one namespace, folded from its events.

    A fold, so `ctx.sessions.fork(source, boundary)` reconstructs the namespace
    *as of the boundary* by folding only the events at or before it. A side file
    could not express that — it would hand every fork the parent's latest
    namespace, which is the thing D17 was chosen over.
    """
    state: dict[str, SnapshotRecord] = {}
    for event in session.events:
        if event.type != "kernel/snapshot":
            continue
        if str(event.data.get("namespace")) != namespace:
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
    """Every namespace this session's log has snapshotted."""
    return {
        str(namespace)
        for event in session.events
        if event.type == "kernel/snapshot"
        if isinstance(namespace := event.data.get("namespace"), str)
    }


def referenced_locators(session: Session) -> set[str]:
    """Every spill locator the log still points at, for the open-time sweep (F7)."""
    found: set[str] = set()
    for event in session.events:
        if event.type != "kernel/snapshot":
            continue
        locator = (event.data.get("record") or {}).get("locator")
        if isinstance(locator, str):
            found.add(locator)
    return found


@dataclass(slots=True)
class KernelSnapshotPolicy:
    """Turns the runtime's snapshot frames into events, and back again.

    The provider knows about processes and frames; this knows about the log. The
    split is why neither has to know both.
    """

    ctx: Context
    inline_blob_max: int = INLINE_BLOB_MAX

    def _session(self, namespace: str) -> Session | None:
        """The session a namespace's events belong in.

        Read from `ctx.agents` rather than kept in a side table fed by
        `agent/created`: the namespace key *is* the agent id, and the registry
        already knows every live agent. A second copy of that mapping is a second
        thing to keep correct — and for a subagent the agent id and the session
        id are not the same string, so deriving one from the other is not
        available either.
        """
        agent = self.ctx.agents.get(namespace)
        session = getattr(agent, "session", None)
        return session if isinstance(session, Session) else None

    # ------------------------------------------------------------- recording --

    async def record(self, namespace: str, run_id: int, variables: list[dict[str, Any]]) -> None:
        """Append one `kernel/snapshot` per changed variable."""
        session = self._session(namespace)
        if session is None:
            return
        for raw in variables:
            encoded = self._encode(session, namespace, raw)
            if encoded is None:
                continue
            record, payload = encoded
            session.append(
                "kernel/snapshot",
                {"namespace": namespace, "run": run_id, "record": record.to_wire()},
            )
            if record.locator is not None and payload is not None:
                await self._write_blob(namespace, record, payload)

    def _encode(
        self, session: Session, namespace: str, raw: dict[str, Any]
    ) -> tuple[SnapshotRecord, bytes | None] | None:
        """The record to append, and the payload to write after appending it.

        The payload is decoded once and handed on. It used to be base64-decoded
        here for the tag, decoded again to write the blob, and hashed a third
        time to derive the path — 35 ms of the 64 ms a 16 MiB variable cost,
        spent re-deriving what was already in hand.
        """
        name = raw.get("var")
        if not isinstance(name, str):
            return None
        skipped = raw.get("skipped")
        if skipped is not None:
            # The guest could not store it. Recorded as a `clear` with the reason,
            # so a restore *tells the model the name is gone* instead of letting
            # it find an undefined name mid-session and read it as a bug.
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
        spill = self.ctx.get("spill_store")
        if spill is None:
            return (
                SnapshotRecord(
                    kind="clear", var=name, reason="too-large (no spill store is mounted)"
                ),
                None,
            )
        # The store derives the locator, so the event can name it *before* the
        # blob exists — write-ahead ordering (§4.9) — without this module
        # mirroring the store's naming rule.
        spilled = SnapshotRecord(
            kind="snap",
            var=name,
            digest=digest,
            bytes=len(payload),
            locator=str(
                spill.locator_for(
                    owner=_owner(namespace), suggested_name=f"{name}.dill", content=payload
                )
            ),
            tag=tag,
        )
        return spilled, payload

    async def _write_blob(self, namespace: str, record: SnapshotRecord, payload: bytes) -> None:
        spill = self.ctx.get("spill_store")
        if spill is None:
            return
        try:
            await spill.save_bytes(
                owner=_owner(namespace),
                source=f"kernel variable {record.var}",
                suggested_name=f"{record.var}.dill",
                content=payload,
            )
        except OSError:
            # The event is already durable and names a blob that is not there;
            # `kernel/restored` will report the variable as failed, which is the
            # recoverable half of the ordering choice.
            log.warning("ph_rlm.snapshot: could not write the blob for %s", record.var)

    # ------------------------------------------------------------ restoring --

    async def materialize(self, namespace: str) -> list[dict[str, Any]]:
        """The payloads to hand a freshly started kernel."""
        session = self._session(namespace)
        if session is None:
            return []
        variables: list[dict[str, Any]] = []
        for record in fold_namespace(session, namespace).values():
            payload = await self._payload(session, record)
            if payload is None:
                continue
            variables.append({"var": record.var, "blob": base64.b64encode(payload).decode("ascii")})
        return variables

    async def _payload(self, session: Session, record: SnapshotRecord) -> bytes | None:
        if record.blob is not None:
            payload = base64.b64decode(record.blob)
        elif record.locator is not None:
            spill = self.ctx.get("spill_store")
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
            # A blob from another session, or a mangled one. Refused rather than
            # unpickled: `dill.loads` on arbitrary bytes executes arbitrary code.
            log.warning("ph_rlm.snapshot: %s failed verification and was not restored", record.var)
            return None
        return payload

    async def restored(self, namespace: str, outcome: dict[str, Any]) -> None:
        """Record what came back, so the model can be told what did not."""
        session = self._session(namespace)
        if session is None:
            return
        session.append("kernel/restored", {"namespace": namespace, **outcome})

    # ------------------------------------------------------------------- GC --

    async def sweep(self, session: Session) -> list[str]:
        """Drop blobs no event references (F7), at session open.

        Both halves come from the same log: the namespaces to visit *and* the
        locators to keep are read from this session's own events. Sweeping every
        namespace the process had seen against one session's reference set
        deleted other sessions' blobs — for them the set was empty, so everything
        they owned looked unreferenced.
        """
        spill = self.ctx.get("spill_store")
        if spill is None:
            return []
        referenced = referenced_locators(session)
        removed: list[str] = []
        for namespace in sorted(namespaces_in(session)):
            removed.extend(await spill.sweep(owner=_owner(namespace), referenced=referenced))
        return removed


def _owner(namespace: str) -> str:
    """The spill owner one namespace's blobs live under."""
    return f"kernel/{namespace}"


class Config(WireModel):
    """Row config for the snapshot policy."""

    inline_blob_max: int = INLINE_BLOB_MAX


@plugin("rlm-kernel-snapshot", config=Config, inject=["sessions", "python_runtime"])
async def apply(ctx: Context, config: Config) -> None:
    """Wire the policy to the runtime provider and to session open.

    `python_runtime` is injected rather than `code_runtime`: the seam being
    mounted says nothing about *which* provider answered it, and this row exists
    to keep one specific provider's promise. Activation is service-availability
    driven, so naming the real dependency is also what orders the two rows.
    """
    policy = KernelSnapshotPolicy(ctx=ctx, inline_blob_max=config.inline_blob_max)
    ctx.provide("kernel_snapshots", policy)
    ctx.python_runtime.snapshots = policy

    async def sweep_on_open(session: Session) -> None:
        removed = await policy.sweep(session)
        if removed:
            log.info("ph_rlm.snapshot: swept %d unreferenced kernel blob(s)", len(removed))

    ctx.on("session/created", sweep_on_open)
