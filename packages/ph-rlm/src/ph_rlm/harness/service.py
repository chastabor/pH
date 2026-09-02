"""Applying a refinement: validate, then append, then project (P3-16).

The order is the design. **Validation happens before anything durable is
written**, and every refusal is recorded *on the event* rather than dropped — so
a refinement that half-applied says which half and why.

Four checks are pH's own, and each one closes a hole prime-agent leaves open:

* **H1 — the reference must resolve.** An entry naming an import or callable that
  does not exist teaches the model to call nothing. It is checked by running a
  probe in the *runtime the model actually uses*, not against this process:
  `/refine` cannot conjure capability, and an unresolvable reference is the
  knowledge layer trying to (I7, Q13). The rejection is on the event.
* **H2 — the call pattern is rendered, never accepted.** Wherever a binding of
  that name exists, an entry's `call_pattern` becomes `await tools.<name>(...)`.
  A proposal that could write its own would be able to steer the model onto the
  raw-namespace path, which is what C2 exists to close.
* **H3 — a global edit is approval-gated.** A `scope: "global"` entry is injected
  into every future session, including other projects, so it goes through
  `ctx.approval`. A local one does not.
* **H5 — the doctrine is not editable.** `base_system_prompt` is refused by id.

**The projection is written and never read.** `harness_state.json` exists for
humans, `ph trace` and export. Deleting it loses nothing, and
`verify_projection` asserts the file equals the fold — which is the only way a
projection stays a projection rather than quietly becoming a second authority.

@module ph_rlm.harness.service
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from filelock import FileLock

from ph.cordis import Context
from ph.paths import write_text_under
from ph.session import Session, SessionFoldCache
from ph.session.json import dumps

from .state import (
    GLOBAL_LOG_NAME,
    PROJECTION_NAME,
    REFINED,
    RESERVED_IDS,
    AppliedEdit,
    HarnessEdit,
    HarnessEntry,
    HarnessScope,
    HarnessState,
    RefinementProposal,
    RefinementRecord,
    fold_events,
    local_fold_cache,
    read_global_events,
)

__all__ = ["HarnessService", "RefinementRefused", "slugify"]

log = logging.getLogger("ph_rlm.harness")

IMMUTABLE_ID = 'entry id "{entry_id}" is not editable'
"""H5, in prime-agent's shape: the id is refused, not the content."""

PROBE_NAMESPACE = "harness-probe"
"""Where H1 resolves a reference.

A namespace of its own, not the agent's: a probe is the harness checking itself,
and leaving `_m`/`_c` behind in the namespace the model is using would put them
in its snapshots."""


class RefinementRefused(Exception):
    """The whole refinement was refused before anything was written.

    Distinct from an edit-level rejection, which is recorded on the event and
    lets the rest apply: this one means no record was appended at all — a global
    edit the human declined, or a proposal with nothing valid left.
    """


def slugify(text: str) -> str:
    """A stable id from a title, so a create without an id is still addressable."""
    words = "".join(c if c.isalnum() else " " for c in text).split()[:6]
    return "-".join(words).lower()[:48] or "entry"


def _invert(edit: AppliedEdit) -> AppliedEdit:
    """One applied edit, undone: the snapshots swapped.

    The action is derived from the swap rather than carried, because it is what a
    reader of the log sees and `after is None` is what the fold acts on.
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
    """`$PH_HOME/harness` — the global log and the human-facing projection."""
    _local: SessionFoldCache[HarnessState] = field(default_factory=local_fold_cache)
    _global_cache: tuple[int, HarnessState] | None = None
    _merged: dict[str, tuple[HarnessState, HarnessState, HarnessState]] = field(
        default_factory=dict
    )
    """Per session: `(local, global, merged)` — the memo `state()` reads through."""

    # ------------------------------------------------------------------ read --

    def local(self, session: Session) -> HarnessState:
        """This session's own refinements, folded and cached on `session.seq`."""
        return self._local.read(session)

    def globals(self) -> HarnessState:
        """The deployment-wide refinements, folded from `$PH_HOME`.

        Cached on the log's size rather than its content: it is append-only, so a
        byte count that has not moved cannot hide a new refinement, and a stat is
        cheaper than a re-read on every prompt assembly.
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

        Memoized on the *identity* of the two folds it merges: each cache above
        hands back the same object until its log actually gained a refinement,
        so `is` on the pair is an exact "nothing changed" test — and this runs
        every model step, where rebuilding the merge would re-copy every entry
        and the whole refinement history to reproduce a byte-identical answer.
        """
        world = self.globals()
        if session is None:
            return world
        local = self.local(session)
        memo = self._merged.get(session.id)
        if memo is not None and memo[0] is local and memo[1] is world:
            return memo[2]
        merged = local.merged_with(world)
        self._merged[session.id] = (local, world, merged)
        return merged

    # -------------------------------------------------------------- validate --

    async def validate(
        self, proposal: RefinementProposal, *, scope: HarnessScope, session: Session | None
    ) -> tuple[list[HarnessEdit], list[str]]:
        """`(accepted, rejected)` — the checks, before anything durable happens."""
        accepted: list[HarnessEdit] = []
        rejected: list[str] = []
        current = self.state(session)
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
                unresolved = await self._probe(edit.reference)
                if unresolved is not None:
                    rejected.append(f'skill "{entry_id}" does not resolve: {unresolved}')
                    continue
            accepted.append(edit.model_copy(update={"id": entry_id}))
        return accepted, rejected

    async def _probe(self, reference: Any) -> str | None:
        """H1: resolve a reference in the runtime, or say why it does not.

        A silent cell — no bindings, no dispatch records — because this is the
        harness checking itself, not the model calling something.
        """
        runtime = self.ctx.get("code_runtime")
        if runtime is None:
            # A mounted seam with no provider is the seam's own error to word:
            # `runtime.run` raises it, and the except below reports it.
            return "no code runtime is mounted to resolve it against"
        from ph.seams.code_runtime import CodeRunRequest

        try:
            outcome = await runtime.run(
                CodeRunRequest(program=reference.probe(), namespace=PROBE_NAMESPACE)
            )
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None if outcome.error is None else str(outcome.error).strip().splitlines()[-1]

    def render_call_pattern(self, entry: HarnessEntry, scope: Context) -> str | None:
        """H2: the binding form wherever a binding of that name exists.

        Derived rather than accepted, so a refinement cannot author prompt text
        that points at the ungoverned path.
        """
        if entry.reference is None:
            return None
        name = entry.reference.callable
        if self.ctx.tools.view(scope).visible.get(name) is not None:
            return f"await tools.{name}(...)"
        return f"{entry.reference.module}.{name}(...)"

    # ----------------------------------------------------------------- apply --

    async def apply(
        self,
        proposal: RefinementProposal,
        *,
        scope: HarnessScope = "local",
        session: Session | None = None,
        agent: Any = None,
    ) -> RefinementRecord:
        """Validate, then record. The record is the state; nothing else is.

        :raises RefinementRefused: a global edit the human declined, or nothing
            valid left to apply.
        """
        if scope == "global" and not await self._approved(agent):
            raise RefinementRefused(
                "a global refinement edits every future session, including other "
                "projects, and was not approved"
            )
        accepted, rejected = await self.validate(proposal, scope=scope, session=session)
        if not accepted:
            raise RefinementRefused(
                "; ".join(rejected) or "the proposal contained no edits to apply"
            )

        current = self.state(session)
        # The *agent's* scope when there is one, because whether a binding of
        # that name is visible is a per-agent question (B7).
        target_scope = getattr(agent, "ctx", None) or self.ctx
        applied: list[AppliedEdit] = []
        for edit in accepted:
            # `validate` stamped an id on every accepted edit — it is the one
            # place ids are derived, so apply cannot disagree with what it checked.
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
            refine_id=f"refine-{secrets.token_hex(4)}",
            scope=scope,
            summary=proposal.summary,
            rationale=proposal.rationale,
            expected_outcome=proposal.expected_outcome,
            applied_edits=applied,
            rejected=rejected,
        )
        await self._commit(record, scope=scope, session=session)
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

    async def _approved(self, agent: Any) -> bool:
        """H3: a global edit asks. A local one never reaches here."""
        approval = self.ctx.get("approval")
        if approval is None or agent is None:
            # Fail closed (B3): a global edit with nowhere to ask is not approved.
            return False
        outcome = await approval.request(
            agent=agent,
            tool_name="refine",
            reason="a global refinement is injected into every future session, "
            "including other projects",
        )
        # `allowed-once` is the only outcome that proceeds; the other three
        # are distinct so a caller can tell a refusal from a missing channel.
        # (`bool()` because `approval` arrives untyped through `ctx.get`.)
        return bool(outcome == "allowed-once")

    async def _commit(
        self, record: RefinementRecord, *, scope: HarnessScope, session: Session | None
    ) -> None:
        """The one durable write, to whichever log owns this scope, then project.

        Both callers go through here so a rollback and an apply cannot come to
        differ in what they leave behind.
        """
        if scope == "local":
            if session is None:
                raise RefinementRefused("a local refinement needs a session to record it in")
            session.append(REFINED, record.to_wire())
        else:
            await anyio.to_thread.run_sync(self._append_global, record)
        await self.write_projection(session)

    def _append_global(self, record: RefinementRecord) -> None:
        """Append one record to `$PH_HOME/harness/events.jsonl`, under a lock.

        Concurrent sessions share this log, so the lock is what makes two
        refinements at once two records rather than one torn line. Held for the
        append alone — a reader folds whatever complete lines it finds.
        """
        # The directory must exist before FileLock can create its lock file.
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / GLOBAL_LOG_NAME
        with FileLock(str(path.with_suffix(".lock")), timeout=30):
            write_text_under(path, f"{dumps(record.to_wire())}\n", append=True)
        self._global_cache = None

    # ------------------------------------------------------------- rollback --

    async def rollback(
        self, refine_id: str, *, session: Session | None, agent: Any = None
    ) -> RefinementRecord:
        """H6: the inverse of one refinement — that record, read backwards.

        Each applied edit carries both snapshots, so the inverse is the same
        edits reversed with `before` and `after` swapped. Built as a record rather
        than re-proposed through `apply`, for two reasons:

        * **An inverse must actually invert.** Re-proposing would construct fresh
          entries and bump their versions, so the restored entry would differ from
          the one it restores. The churn belongs in the refinement history, which
          the log keeps anyway.
        * **It is not revalidated.** This state was validated when it was written,
          and refusing to undo a bad refinement because the world has since moved
          would trap a user in the state they are trying to leave.

        A global rollback still asks (H3): undoing a deployment-wide entry changes
        every future session exactly as writing one does.
        """
        state = self.state(session)
        target = next((one for one in state.refinements if one.refine_id == refine_id), None)
        if target is None:
            raise RefinementRefused(f'no refinement "{refine_id}" in this harness')
        if any(one.rollback_of == refine_id for one in state.refinements):
            raise RefinementRefused(f'"{refine_id}" has already been rolled back')
        if target.scope == "global" and not await self._approved(agent):
            raise RefinementRefused(
                f'rolling back "{refine_id}" changes every future session and was not approved'
            )

        record = RefinementRecord(
            refine_id=f"refine-{secrets.token_hex(4)}",
            scope=target.scope,
            summary=f"rollback of {refine_id}",
            applied_edits=[_invert(edit) for edit in reversed(target.applied_edits)],
            rollback_of=refine_id,
        )
        await self._commit(record, scope=target.scope, session=session)
        return record

    # ---------------------------------------------------------- projection --

    def projection_path(self, session: Session | None) -> Path:
        """Where this session's projection lives — one file per session.

        Per session because the projection is of *local layered over global*, so
        a single shared path would have two sessions overwriting each other's and
        the "file equals the fold" invariant (P6-01) flapping in any deployment
        that runs more than one. `<session artifacts>/harness/` in the plan;
        until artifact roots exist, a directory named for the session under the
        harness root is the same shape.
        """
        if session is None:
            return self.directory / PROJECTION_NAME
        return self.directory / session.id / PROJECTION_NAME

    async def write_projection(self, session: Session | None) -> Path:
        """Write `harness_state.json` for humans. Nothing reads it back."""
        path = self.projection_path(session)
        payload = dumps(self.state(session).to_wire())
        await anyio.to_thread.run_sync(self._write_projection, path, payload)
        return path

    def _write_projection(self, path: Path, payload: str) -> None:
        """Written through a rename, so a reader never sees half a projection.

        A human can read this file while a refinement is applying, and under the
        daemon two sessions project at once; a truncating write makes both of
        those a torn file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
        staged.write_text(f"{payload}\n", encoding="utf-8")
        staged.replace(path)

    def stale_projections(self) -> list[str]:
        """Every written projection that no longer equals the fold behind it (I6, P6-01).

        Here rather than in the invariant row that declares it, because the
        projection's layout is this service's own: which sessions have one, where
        it lives, what it projects. A row that enumerated `[None, *sessions]` from
        outside encoded a write path this service does not have — nothing calls
        `write_projection(None)` — and would silently miss a third projection the
        day one is added.

        Per session because the projection is per session. **A missing file is not
        drift**: nothing requires a session to have projected, and an alarm loudest
        where the feature is used least is one people learn to ignore.
        """
        sessions = self.ctx.get("sessions")
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

        The invariant that keeps a projection a projection: if this can drift,
        something has started treating the file as state.
        """
        path = self.projection_path(session)
        if not path.exists():
            return False
        return path.read_text(encoding="utf-8").strip() == dumps(self.state(session).to_wire())
