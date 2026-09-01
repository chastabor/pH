"""`ctx.compaction` — replacing history with a summary, without losing it.

The seam only. *Definition, Provider, Consumer* (invariant I5): the definition
is `CompactionEngine`, a provider is any plugin calling `ctx.compaction.register`
(`compaction-summarize` in `ph-stabilize`, P4-03), and the consumers are the
`/compact` command and whatever policy row decides a session is under pressure.
The loop knows nothing about any of it — compaction attaches to `agent/pre-step`
and `agent/request-error` like every other stabilization feature (D12).

**Compaction is a surface `replace`, and that is the whole safety story (A3).**
The engine appends one message whose `surfaceOp` shadows the range it stands
for: `derive_messages()` yields the summary, the log keeps every shadowed event,
and `transcript()` still shows the person the conversation they had. Nothing is
deleted, so a summary that turns out to have dropped something important is a
bad *reading* of the log rather than a hole in it (I4).

**Notes are the part that is pH's own.** A summary replaces *conversation*, and
pH has state that is not conversation: an RLM session's kernel namespace
survives the cut completely untouched, because compaction rewrites a surface and
a REPL is not on it. A summary that did not say so would leave the model
believing its variables went wherever the conversation went — so a plugin that
owns such state registers a `CompactionNote`, and the engine puts it in the
summary prompt (G10). The state itself is never touched from here.

@module ph.seams.compaction
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from ..cordis import Boundary, Context, Disposer, Running, boundary_of, plugin, running
from ..session import Session
from ._registry import claim_entry, claim_slot

__all__ = [
    "CompactionEngine",
    "CompactionError",
    "CompactionNote",
    "CompactionResult",
    "CompactionSeam",
    "CompactionTrigger",
    "apply",
]

log = logging.getLogger("ph.seams.compaction")

CompactionTrigger: TypeAlias = Literal["pressure", "overflow", "manual"]
"""Why compaction is being considered.

`pressure` is the estimate before a request is made, `overflow` is the
provider's own `CONTEXT_WINDOW_EXCEEDED` after one was, and `manual` is a person
typing `/compact`. An engine is allowed to answer them differently: the first
two are policy, the third is an instruction.
"""


class CompactionError(Exception):
    """A compaction that was attempted and failed, with a class a caller can act on.

    `code` is a stable, small vocabulary so a front end can phrase the outcome
    without parsing the message: `busy` (a compaction is already running, or the
    agent is not idle), `summary` (the summarizing model call failed or returned
    nothing usable), `unavailable` (no engine is registered).

    Distinct from returning `None`, which means *nothing was compacted and that
    is fine* — an empty session, or no cut that keeps every tool call with its
    result. Merging the two would make "your conversation is short" and "your
    summarizer is broken" the same outcome.
    """

    def __init__(self, code: Literal["busy", "summary", "unavailable"], message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """What one landed compaction did."""

    trigger: CompactionTrigger
    summary: str
    shadowed_seqs: tuple[int, ...]
    """Every surface node the replacement shadows, in surface order. The same
    seqs the replacement event cites as `sourceEventSeqs`, which is what makes
    the substitution reversible reading rather than a deletion."""
    shadowed_tokens: int
    """Estimated cost of what left the derivation — the number that says whether
    the compaction was worth its model call."""
    replacement_seq: int
    locator: str | None = None
    """Where the replaced conversation was written, when it could be. `None`
    when no spill store took it: the summary is then all there is, and the
    engine must not claim otherwise."""


@runtime_checkable
class CompactionEngine(Protocol):
    """A compaction backend: it owns trigger policy, retention and summarizing."""

    async def compact_if_needed(
        self, agent: Any, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """Compact only if this engine's own policy says to, else `None`."""
        ...

    async def compact_now(self, agent: Any, *, instructions: str = "") -> CompactionResult | None:
        """Compact useful history regardless of pressure; `None` when there is none.

        `instructions` is the person's own account of what they are about to
        work on. An engine may weight the summary towards it; one that cannot is
        free to ignore it, which is why it has a default rather than being a
        second method.
        """
        ...


@dataclass(frozen=True, slots=True)
class CompactionNote:
    """State that survives a compaction, and the sentence that says so.

    `text` is called with the session at summarize time and returns the block to
    put in the summary prompt, or `""` to contribute nothing — the same
    empty-means-absent rule `PromptContext` uses, so a note about a namespace
    that holds nothing costs no prompt.
    """

    name: str
    text: Callable[[Session], str]
    order: int = 0


@dataclass(frozen=True, slots=True)
class _NoteRegistration:
    """A note and who registered it (P6-29).

    A `Running` rather than a single scope, because it cannot be read as the wrong
    half: `by.layer` is what `notes()` filters with, and `by.owner` is what the
    note's body runs as. One field holding one of those under a name that reads like
    the other is the P6-12 conflation."""

    note: CompactionNote
    by: Running


@dataclass(slots=True)
class CompactionSeam:
    """The service published as `ctx.compaction`."""

    ctx: Context
    engine: CompactionEngine | None = None
    """At most one. Two answers to "when and how is history replaced" is a
    contradiction, and a profile picks its backend."""
    engine_by: Running | None = None
    """Who registered the engine (P6-29). Set and cleared with it by `claim_slot`.

    Entered around every call into the engine, so anything its body registers
    unwinds with the row that registered the engine rather than with this seam.
    The layer stays the registration's own — **not the agent being compacted**,
    which both call sites have in hand — because reading it means a fourteenth
    copy of P6-24's `getattr(agent, "ctx", None)`, and an engine is one
    deployment-wide object rather than a per-agent one. `ph.seams.fs` is the one
    provider whose target is already derived, and it passes it. P6-24 did not
    change that: it fixed the boundary a *policy* call is judged in, and an engine
    is invoked on a lifetime question rather than a policy one. If this seam is
    ever handed a scope, these are the lines that take it."""
    _notes: list[_NoteRegistration] = field(default_factory=list)

    # ------------------------------------------------------------ the engine --

    def register(self, engine: CompactionEngine, *, scope: Context | None = None) -> Disposer:
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "engine",
            engine,
            label="compaction",
        )

    def require(self) -> CompactionEngine:
        if self.engine is None:
            raise CompactionError("unavailable", "no compaction engine is registered")
        return self.engine

    async def compact_if_needed(
        self, agent: Any, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """Automatic policy. Absent an engine this is a no-op, never an error.

        A profile that layered no compaction row has *chosen* not to compact, and
        a policy hook that raised at it would turn that choice into a broken
        session on the first long conversation.
        """
        if self.engine is None:
            return None
        with running(self.engine_by):
            return await self.engine.compact_if_needed(agent, trigger)

    async def compact_now(self, agent: Any, *, instructions: str = "") -> CompactionResult | None:
        """An explicit request. Absent an engine this *is* an error.

        Unlike the automatic path: somebody asked for something the deployment
        cannot do, and silence would read as "nothing to compact".
        """
        engine = self.require()
        with running(self.engine_by):
            return await engine.compact_now(agent, instructions=instructions)

    # ------------------------------------------------------------- the notes --

    def note(self, note: CompactionNote, *, scope: Context | None = None) -> Disposer:
        """Contribute a block to every summary prompt this context can reach."""
        # Both questions, kept together (P6-29): `by.layer` is what `notes()`
        # filters with `reaches`, `by.owner` is the disposer's scope and the one
        # the body runs as. Through `claim_entry` because `_NoteRegistration` is
        # a frozen dataclass and therefore compares by *value*: the
        # `self._notes.remove(entry)` this replaces would have had one row's
        # disposal take another's equal entry, which is the defect `_registry`
        # exists to name.
        by = self.ctx.running_for(scope)
        entry = _NoteRegistration(note=note, by=by)
        return claim_entry(
            by.owner,
            self._notes,
            entry,
            label=f"compaction-note({note.name})",
        )

    def notes(self, session: Session, *, scope: Boundary) -> list[str]:
        """The rendered blocks for one session, in `order` then registration order.

        Scope-filtered by the same rule as event dispatch and every other scoped
        registry: a global registration reaches every agent, an agent-scoped one
        reaches that agent alone (B7).

        **The boundary is stated** (P6-32), and this reader is the one that shows
        which way the axis runs. It resolved `scope or self.ctx`, and for a
        registry that bears *restrictions* the mount is the widest answer — that
        is the fail-open the row is named for. This registry bears none: `notes`
        filters on `reaches` alone, so the mount is the **smallest** set, and the
        old default could never over-include. It could only drop an agent's own
        note from that agent's summary — silent-*narrow*, which the row rejects
        for the same reason in the other direction ("degrades a model quietly
        rather than erroring, and nobody notices").

        So the test for a new registry is not "can I name the harm" but **does
        this reader bear restrictions**: if it does, an unstated boundary widens;
        if it filters on `reaches` alone, it under-includes. Either way a policy
        reader takes a boundary. See `Deployment`, which states the axis from the
        other side — widest along the restriction axis only.

        A note that raises is dropped with its
        traceback rather than taking the compaction down — a summary without one
        block is worth more than an uncompacted session.
        """
        target = boundary_of(scope, self.ctx)
        rendered: list[str] = []
        visible = [entry for entry in self._notes if entry.by.layer.reaches(target)]
        for entry in sorted(visible, key=lambda one: one.note.order):
            note = entry.note
            try:
                # **As the row that registered it, for the scope being
                # compacted** (P6-29). `text` is a row's body invoked by a
                # registry — the same category as a tool's `execute` — and it ran
                # unbound, so anything it registered landed on the seam and
                # outlived its row. This is one of the four P6-26 could not reach
                # while the binding held a single `Context`: the target here is a
                # *layer* and the owner is still the registering row.
                with running(entry.by, target):
                    text = note.text(session)
            except Exception:
                log.warning(
                    "ph.seams.compaction: note %r failed to render", note.name, exc_info=True
                )
                continue
            if text:
                rendered.append(text)
        return rendered


@plugin("compaction")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the compaction seam definition. No engine ships in `ph-base`."""
    ctx.provide("compaction", CompactionSeam(ctx=ctx))
