"""Applying a refinement: validate, then append, then project (P3-16).

The order is the design. **Validation happens before anything durable is
written.** Every refusal is recorded *on the event* and is not dropped, so a
refinement that applied in part says which part and why.

Four checks are specific to pH. Each one closes a hole that prime-agent leaves
open:

* **H1: the reference must resolve.** An entry that names an import or a
  callable that does not exist teaches the model to call nothing. The check runs
  a probe in the *runtime that the model uses* (`_probe`), not against this
  process. `/refine` cannot create capability, and an unresolvable reference is
  the knowledge layer trying to create it (I7, Q13). The rejection is on the
  event.
* **H2: the call pattern is rendered, never accepted.** Wherever a binding of
  that name exists, the `call_pattern` of an entry becomes
  `await tools.<name>(...)`. A proposal that could write its own call pattern
  could steer the model onto the raw-namespace path, and C2 exists to close that
  path.
* **H3: a global edit needs approval.** An entry with `scope: "global"` is
  injected into every future session, including other projects, so it goes
  through `ctx.approval`. A local entry does not.
* **H5: the doctrine is not editable.** `base_system_prompt` is refused by id.

**The projection is written and never read.** `harness_state.json` exists for
humans, the trajectory viewer and export. Deleting it loses nothing.
`verify_projection` asserts that the file equals the fold. That is the only way
that a projection stays a projection and does not quietly become a second
authority.

@module ph_rlm.harness.service
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import anyio

from ph.agent.types import AgentHandle
from ph.cordis import Context
from ph.json import dumps
from ph.keys import APPROVAL, CODE_RUNTIME, SESSIONS, TOOLS
from ph.locks import file_lock
from ph.paths import write_atomic
from ph.persistence import append_records
from ph.seams.code_runtime import CodeRunRequest
from ph.session import (
    IntentNotDurable,
    Session,
    SessionFoldCache,
    SessionTable,
    intents_of,
    session_written,
)
from ph.session.writers import log_writer

from ..kinds import HARNESS_PROBE, probe_settled
from .state import (
    GLOBAL_LOG_NAME,
    PROJECTION_NAME,
    REFINED,
    RESERVED_IDS,
    AppliedEdit,
    HarnessEdit,
    HarnessEntry,
    HarnessReference,
    HarnessScope,
    HarnessState,
    RefinementProposal,
    RefinementRecord,
    fold_events,
    local_fold_cache,
    read_global_events,
)

_LOG = log_writer(__name__)

__all__ = ["HarnessService", "RefinementRefused", "slugify"]

log = logging.getLogger("ph_rlm.harness")

IMMUTABLE_ID = 'entry id "{entry_id}" is not editable'
"""H5, in the form that prime-agent uses: the id is refused, not the content."""

PROBE_NAMESPACE = "harness-probe:{agent}"
"""Where H1 resolves a reference for one agent. It is a kernel that belongs to
the agent and stands beside its working kernel. It is within the same boundary
(`CodeRunRequest.agent`), and it is closed with the agent."""


class RefinementRefused(Exception):
    """The whole refinement was refused before anything was written.

    This differs from a rejection of one edit, which is recorded on the event
    and lets the rest apply. This exception means that no record was appended at
    all. For example, the user declined a global edit, or the proposal had
    nothing valid left.
    """


def slugify(text: str) -> str:
    """A stable id from a title, so a create without an id is still addressable."""
    words = "".join(c if c.isalnum() else " " for c in text).split()[:6]
    return "-".join(words).lower()[:48] or "entry"


def _invert(edit: AppliedEdit) -> AppliedEdit:
    """One applied edit, undone: the snapshots are swapped.

    The action is derived from the swap and is not carried over. The action is
    what a reader of the log sees, and `after is None` is what the fold acts on.
    """
    return AppliedEdit(
        action="delete" if edit.before is None else "create" if edit.after is None else "update",
        kind=edit.kind,
        id=edit.id,
        before=edit.after,
        after=edit.before,
    )


@dataclass(slots=True)
class HarnessService:
    """The service published as `ctx.harness`."""

    ctx: Context
    directory: Path
    """`$PH_HOME/harness`: the global log and the projection for humans."""
    _local: SessionFoldCache[HarnessState] = field(default_factory=local_fold_cache)
    _global_cache: tuple[int, HarnessState] | None = None
    _merged: SessionTable[tuple[HarnessState, HarnessState, HarnessState]] = field(
        default_factory=SessionTable
    )
    """For each session: `(local, global, merged)`, the memo that `state()` reads
    through. It is released with the session, as `_local` is."""

    # ------------------------------------------------------------------ read --

    def local(self, session: Session) -> HarnessState:
        """The refinements of this session, folded and cached on `session.seq`."""
        return self._local.read(session)

    def stale_folds(self, sessions: Iterable[Session]) -> list[str]:
        """Cached local harness states that no longer equal their fold (I6).

        This is the sibling of `stale_projections`, one layer further in. That
        method asks whether the file on disk equals the fold. This method asks
        whether the value in memory does. Both matter, because the projection is
        written *from* the cache. A cache that drifted writes a file that agrees
        with the cache and with nothing else.
        """
        return self._local.stale(sessions)

    def globals(self) -> HarnessState:
        """The deployment-wide refinements, folded from `$PH_HOME`.

        The cache key is the size of the log, not its content. The log is
        append-only, so a byte count that has not changed cannot hide a new
        refinement. A stat is also cheaper than a new read on every prompt
        assembly.
        """
        path = self.directory / GLOBAL_LOG_NAME
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        if self._global_cache is not None and self._global_cache[0] == size:
            return self._global_cache[1]
        state = fold_events(read_global_events(self.directory))
        self._global_cache = (size, state)
        return state

    def state(self, session: Session | None) -> HarnessState:
        """What the model should be told: local layered over global.

        The result is memoized on the *identity* of the two folds that it
        merges. Each cache above returns the same object until its log gains a
        refinement, so `is` on the pair is an exact "nothing changed" test. This
        method runs on every model step. A rebuilt merge would copy every entry
        and the whole refinement history again, to produce an answer with the
        same bytes.
        """
        world = self.globals()
        if session is None:
            return world
        local = self.local(session)
        memo = self._merged.get(session)
        if memo is not None and memo[0] is local and memo[1] is world:
            return memo[2]
        merged = local.merged_with(world)
        self._merged.keep(session, (local, world, merged))
        return merged

    # -------------------------------------------------------------- validate --

    async def _validate(
        self,
        proposal: RefinementProposal,
        *,
        agent: AgentHandle,
        refine_id: str,
    ) -> tuple[list[HarnessEdit], list[str]]:
        """`(accepted, rejected)`: the checks, which run before the refinement is
        written. The checks write nothing, except the probes of H1 (`_probe`),
        which record themselves under `refine_id`."""
        accepted: list[HarnessEdit] = []
        rejected: list[str] = []
        current = self.state(agent.session)
        for edit in proposal.edits:
            entry_id = edit.id or slugify(edit.title or edit.content)
            if entry_id in RESERVED_IDS:
                rejected.append(IMMUTABLE_ID.format(entry_id=entry_id))
                continue
            if edit.action == "delete":
                if current.entry(edit.kind, entry_id) is None:
                    rejected.append(f'cannot delete "{entry_id}": no such {edit.kind}')
                    continue
                accepted.append(edit.model_copy(update={"id": entry_id}))
                continue
            if edit.kind == "skill":
                if edit.reference is None:
                    rejected.append(
                        f'skill "{entry_id}" has no reference; a skill entry must name the '
                        "capability it describes, or it is teaching the model to call nothing"
                    )
                    continue
                unresolved = await self._probe(
                    edit.reference, agent=agent, entry_id=entry_id, refine_id=refine_id
                )
                if unresolved is not None:
                    rejected.append(f'skill "{entry_id}" does not resolve: {unresolved}')
                    continue
            accepted.append(edit.model_copy(update={"id": entry_id}))
        return accepted, rejected

    async def _probe(
        self,
        reference: HarnessReference,
        *,
        agent: AgentHandle,
        entry_id: str,
        refine_id: str,
    ) -> str | None:
        """H1: resolve a reference in the runtime, or say why it does not resolve.

        **Within the boundary of the agent that proposed it** (S21). An import
        runs the top level of the module, and the model named the module. So the
        probe gets the workspace of the agent, its confinement, and its scope as
        the owner of the kernel. In a namespace that no agent owned, the probe
        had none of them. It was code that the model chose, run with more
        permission than the model has, by auto-refine with nobody asking.

        **Beside the kernel of the agent, not in it** (`PROBE_NAMESPACE`). A
        module that hangs or crashes on import costs the kernel of the probe and
        not the kernel of the model. Nothing that the module binds reaches the
        namespace or the snapshots of the model.

        **On disk before it runs, in the log of the agent** (`HARNESS_PROBE`).
        The probe is then settled with what it found. A probe that hangs or
        takes its kernel down therefore still shows what was started.
        """
        runtime = self.ctx.get(CODE_RUNTIME)
        if runtime is None:
            # When the seam is mounted but has no provider, the seam words the
            # error itself: `runtime.run` raises it, and the except below
            # reports it.
            return "no code runtime is mounted to resolve it against"
        session = agent.session
        if session is None:
            return "the agent has no log to record the probe in"

        intents = intents_of(self.ctx)
        try:
            async with intents.claim(
                session,
                HARNESS_PROBE,
                {
                    "refineId": refine_id,
                    "entry": entry_id,
                    "module": reference.module,
                    "callable": reference.callable,
                },
            ) as held:
                outcome = await runtime.run(
                    CodeRunRequest(
                        program=reference.probe(),
                        agent=agent.id,
                        namespace=PROBE_NAMESPACE.format(agent=agent.id),
                    )
                )
                unresolved = (
                    None if outcome.error is None else str(outcome.error).strip().splitlines()[-1]
                )
                intents.settle(session, held, probe_settled(held.opened.seq, unresolved))
        except IntentNotDurable:
            return "the probe could not be recorded, so it did not run"
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return unresolved

    def render_call_pattern(self, entry: HarnessEntry, scope: Context) -> str | None:
        """H2: the binding form, wherever a binding of that name exists.

        The pattern is derived and not accepted, so a refinement cannot write
        prompt text that points at the ungoverned path.
        """
        if entry.reference is None:
            return None
        name = entry.reference.callable
        if self.ctx.require(TOOLS).view(scope).visible.get(name) is not None:
            return f"await tools.{name}(...)"
        return f"{entry.reference.module}.{name}(...)"

    # ----------------------------------------------------------------- apply --

    async def apply(
        self,
        proposal: RefinementProposal,
        *,
        agent: AgentHandle,
        scope: HarnessScope = "local",
    ) -> RefinementRecord:
        """Validate, then record. The record is the state, and nothing else is.

        An agent makes the refinement, into the harness of its session. A skill
        entry is probed as that agent: within its boundary, and recorded in its
        log (S21). A global edit asks the user of that agent (H3).

        :raises RefinementRefused: the user declined a global edit, or nothing
            valid was left to apply.
        """
        refine_id, approving = await self._authorize(
            scope,
            agent,
            "a global refinement edits every future session, including other projects, and "
            "was not approved",
        )
        session = agent.session
        accepted, rejected = await self._validate(proposal, agent=agent, refine_id=refine_id)
        if not accepted:
            raise RefinementRefused(
                "; ".join(rejected) or "the proposal contained no edits to apply"
            )

        current = self.state(session)
        # The scope of the agent is used, because whether a binding of that name
        # is visible is a question for each agent (B7).
        target_scope = agent.ctx
        applied: list[AppliedEdit] = []
        for edit in accepted:
            # `_validate` stamped an id on every accepted edit. It is the one
            # place where ids are derived, so apply cannot disagree with what
            # `_validate` checked.
            entry_id = edit.id
            assert entry_id is not None
            before = current.entry(edit.kind, entry_id)
            after = None if edit.action == "delete" else self._entry(edit, entry_id, before, scope)
            if after is not None:
                after = after.model_copy(
                    update={"call_pattern": self.render_call_pattern(after, target_scope)}
                )
            applied.append(
                AppliedEdit(
                    action=edit.action, kind=edit.kind, id=entry_id, before=before, after=after
                )
            )

        record = RefinementRecord(
            refine_id=refine_id,
            scope=scope,
            summary=proposal.summary,
            rationale=proposal.rationale,
            expected_outcome=proposal.expected_outcome,
            applied_edits=applied,
            rejected=rejected,
            approved_in=approving.id if approving is not None else None,
        )
        await self._commit(record, scope=scope, session=session, approving=approving)
        return record

    def _entry(
        self, edit: HarnessEdit, entry_id: str, before: HarnessEntry | None, scope: HarnessScope
    ) -> HarnessEntry:
        """The entry an edit produces. An update bumps the version it replaced."""
        return HarnessEntry(
            kind=edit.kind,
            id=entry_id,
            title=edit.title or (before.title if before else entry_id),
            content=edit.content,
            version=(before.version + 1) if before is not None else 1,
            scope=scope,
            path=edit.path or (before.path if before else None),
            reference=edit.reference or (before.reference if before else None),
            metadata=edit.metadata or (before.metadata if before else {}),
        )

    async def _authorize(
        self, scope: HarnessScope, agent: AgentHandle | None, refused: str
    ) -> tuple[str, Session | None]:
        """The id of a new refinement. For a global refinement, also the session
        whose log holds the approval that allowed it (H3, L4). Both writers
        (`apply` and `rollback`) go through here.

        The id is made first so that the ask can carry it. The `callId` of the
        approval then names the record that it allowed.

        :raises RefinementRefused: `refused`, for a global edit that was not approved.
        """
        refine_id = f"refine-{secrets.token_hex(4)}"
        if scope != "global":
            return refine_id, None
        if not await self._approved(agent, refine_id):
            raise RefinementRefused(refused)
        return refine_id, agent.session if agent is not None else None

    async def _approved(self, agent: AgentHandle | None, refine_id: str) -> bool:
        """H3: a global edit asks. A local edit never reaches here.

        The ask carries the id of the refinement as its `callId`. So the
        `approval/decided` event in the log of the approving session names the
        record that it allowed, and the global record names that session
        (`approved_in`, L4).
        """
        approval = self.ctx.get(APPROVAL)
        if approval is None or agent is None:
            # Fail closed (B3): a global edit with nowhere to ask is not approved.
            return False
        outcome = await approval.request(
            agent=agent,
            tool_name="refine",
            call_id=refine_id,
            reason="a global refinement is injected into every future session, "
            "including other projects",
        )
        # `allowed-once` is the only outcome that proceeds. The other three are
        # distinct so that a caller can tell a refusal from a missing channel.
        # (`bool()` is here because `approval` arrives untyped through `ctx.get`.)
        return bool(outcome == "allowed-once")

    async def _commit(
        self,
        record: RefinementRecord,
        *,
        scope: HarnessScope,
        session: Session | None,
        approving: Session | None,
    ) -> None:
        """The one durable write, to the log that owns this scope, then the projection.

        Both callers go through here, so that a rollback and an apply cannot
        come to differ in what they leave behind.

        **The session reaches disk before anything outside it does** (L4). For a
        global edit, that session is the session of the approving agent, whose
        `approval/decided` allowed the edit. Every future session reads the
        global log. An edit that was written there first could outlive a crash
        that destroyed its approval. The result would be an edit in force with
        no record that anyone allowed it. So a global edit whose approval cannot
        be written is refused. For a local edit, that session is the session
        that holds `harness/refined`, and it is written ahead of the projection
        that restates it.
        """
        if scope == "local":
            if session is None:
                raise RefinementRefused("a local refinement needs a session to record it in")
            _LOG.append(session, REFINED, record.to_wire())
            # The record is the state, and it is in memory already. A write that
            # fails here means that the next flushes of the session fail too,
            # and the projection is only a restatement.
            await session_written(self.ctx, session)
        else:
            if approving is not None and not await session_written(self.ctx, approving):
                raise RefinementRefused(
                    "the approval of this global refinement could not be written, so the "
                    "refinement was not applied"
                )
            await anyio.to_thread.run_sync(self._append_global, record)
        await self.write_projection(session)

    def _append_global(self, record: RefinementRecord) -> None:
        """Append one record to `$PH_HOME/harness/events.jsonl`, under a lock.

        Concurrent sessions share this log. The lock makes two refinements at
        the same time two records and not one torn line. The lock is held for
        the append only. A reader folds whatever complete lines it finds.

        **The write is `fsync`ed, and a failed write is taken back**
        (`append_records`, L4). Every future session reads this log. A record
        that stayed in the page cache could be gone after a power loss, while
        the sessions that acted on it are not.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / GLOBAL_LOG_NAME
        with file_lock(path.with_suffix(".lock"), timeout=30, what="the harness log"):
            append_records(path, [record.to_wire()])
        self._global_cache = None

    # ------------------------------------------------------------- rollback --

    async def rollback(
        self, refine_id: str, *, session: Session | None, agent: AgentHandle | None = None
    ) -> RefinementRecord:
        """H6: the inverse of one refinement, which is that record read backwards.

        Each applied edit carries both snapshots. The inverse is therefore the
        same edits in reverse order, with `before` and `after` swapped. The
        inverse is built as a record and is not proposed again through `apply`,
        for two reasons:

        * **An inverse must invert.** A new proposal would construct fresh
          entries and bump their versions, so the restored entry would differ
          from the entry that it restores. The churn belongs in the refinement
          history, which the log keeps in any case.
        * **It is not validated again.** This state was validated when it was
          written. A refusal to undo a bad refinement because the world has
          changed since then would trap a user in the state that the user is
          trying to leave.

        A global rollback still asks (H3). Undoing a deployment-wide entry
        changes every future session exactly as writing one does.
        """
        state = self.state(session)
        target = next((one for one in state.refinements if one.refine_id == refine_id), None)
        if target is None:
            raise RefinementRefused(f'no refinement "{refine_id}" in this harness')
        if any(one.rollback_of == refine_id for one in state.refinements):
            raise RefinementRefused(f'"{refine_id}" has already been rolled back')
        rollback_id, approving = await self._authorize(
            target.scope,
            agent,
            f'rolling back "{refine_id}" changes every future session and was not approved',
        )

        record = RefinementRecord(
            refine_id=rollback_id,
            scope=target.scope,
            summary=f"rollback of {refine_id}",
            applied_edits=[_invert(edit) for edit in reversed(target.applied_edits)],
            rollback_of=refine_id,
            approved_in=approving.id if approving is not None else None,
        )
        await self._commit(record, scope=target.scope, session=session, approving=approving)
        return record

    # ---------------------------------------------------------- projection --

    def projection_path(self, session: Session | None) -> Path:
        """Where the projection of this session is. There is one file per session.

        The projection is of *local layered over global*. With a single shared
        path, two sessions would each overwrite the file of the other, and the
        "file equals the fold" invariant (P6-01) would flap in any deployment
        that runs more than one session. The plan names
        `<session artifacts>/harness/`. Until artifact roots exist, a directory
        named for the session under the harness root has the same shape.
        """
        if session is None:
            return self.directory / PROJECTION_NAME
        return self.directory / session.id / PROJECTION_NAME

    async def write_projection(self, session: Session | None) -> Path:
        """Write `harness_state.json` for humans. Nothing reads it back.

        The write goes through a rename. A human can read this file while a
        refinement is being applied, and under the daemon two sessions project
        at the same time. A truncating write would make a torn file in both
        cases.
        """
        path = self.projection_path(session)
        payload = dumps(self.state(session).to_wire())
        # Not durable. Nothing reads the file back: the log is the state, and
        # the file restates it.
        await anyio.to_thread.run_sync(partial(write_atomic, path, f"{payload}\n", durable=False))
        return path

    def stale_projections(self) -> list[str]:
        """Every written projection that no longer equals the fold behind it (I6, P6-01).

        The check is here and not in the invariant row that declares it, because
        the layout of the projection belongs to this service: which sessions
        have one, where it is, and what it projects. A row that enumerated
        `[None, *sessions]` from outside encoded a write path that this service
        does not have, because nothing calls `write_projection(None)`. Such a
        row would also silently miss a third projection on the day that one is
        added.

        The check is per session because the projection is per session. **A
        missing file is not drift.** Nothing requires a session to have
        projected. An alarm that is loudest where the feature is used least is
        an alarm that people learn to ignore.
        """
        sessions = self.ctx.get(SESSIONS)
        if sessions is None:
            return []
        return [
            f"session {session.id}: {path} does not equal the fold it projects"
            for session in sessions.list()
            if (path := self.projection_path(session)).exists()
            and not self.verify_projection(session)
        ]

    def verify_projection(self, session: Session | None) -> bool:
        """Whether the file on disk equals the fold.

        This invariant keeps a projection a projection. If the file can drift,
        something has started to treat the file as state.
        """
        path = self.projection_path(session)
        if not path.exists():
            return False
        return path.read_text(encoding="utf-8").strip() == dumps(self.state(session).to_wire())
