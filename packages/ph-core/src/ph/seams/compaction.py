"""`ctx.compaction`: replace history with a summary, and lose nothing.

This module holds the seam only. *Definition, Provider, Consumer* (invariant
I5): the definition is `CompactionEngine`. A provider is any plugin that calls
`ctx.compaction.register` (`compaction-summarize` in `ph-stabilize`, P4-03).
The consumer through this seam is the `/compact` command (`compact_now`). The
automatic triggers are the engine's own business: the engine row attaches to
`agent/pre-step`, `agent/request` and `agent/request-error`, like every other
stabilization feature (D12), and decides there whether a session is under
pressure. The loop knows nothing about compaction.

**Compaction is a surface `replace`, and the safety of the seam depends on
that (A3).** The engine appends one message. The `surfaceOp` of that message
shadows the range that the message stands for. `derive_messages()` yields the
summary, the log keeps every shadowed event, and `transcript()` still shows the
user the conversation that they had. Nothing is deleted. So a summary that
dropped something important is a bad *reading* of the log and not a hole in it
(I4).

**Notes are the part that is specific to pH.** A summary replaces
*conversation*, and pH has state that is not conversation. The kernel namespace
of an RLM session survives the cut untouched, because compaction rewrites a
surface and a REPL is not on a surface. A summary that did not say so would let
the model believe that its variables went where the conversation went. So a
plugin that owns such state registers a `CompactionNote`, and the engine puts
the note in the summary prompt (G10). This module never touches the state
itself.

@module ph.seams.compaction
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from ..agent.types import AgentHandle
from ..cordis import Boundary, Context, Disposer, Running, boundary_of, plugin, running
from ..keys import COMPACTION
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

CompactionTrigger: TypeAlias = Literal["pressure", "window", "overflow", "manual"]
"""Why compaction is being considered.

- `pressure`: the estimate before a step.
- `window`: the estimate at the request itself, when it says that the request
  fills the whole window. A provider would only refuse such a request.
- `overflow`: the provider answered `CONTEXT_WINDOW_EXCEEDED` after a request
  was sent.
- `manual`: a user typed `/compact`.

An engine may answer them differently. The first three are policy, and the last
is an instruction.
"""


class CompactionError(Exception):
    """A compaction that was attempted and failed, with a class that a caller can act on.

    `code` is a small, stable vocabulary, so a front end can phrase the outcome
    without parsing the message:

    - `busy`: a compaction is already running, or the agent is not idle.
    - `summary`: the summarizing model call failed or returned nothing usable.
    - `unavailable`: no engine is registered.

    This is distinct from a return of `None`, which means *nothing was compacted
    and that is fine*: the session is empty, or no cut keeps every tool call
    with its result. If the two were merged, "your conversation is short" and
    "your summarizer is broken" would be the same outcome.
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
    """Every surface node that the replacement shadows, in surface order. The
    replacement event cites the same seqs as `sourceEventSeqs`. That citation
    makes the substitution a reversible reading and not a deletion."""
    shadowed_tokens: int
    """The estimated cost of what left the derivation. This number shows whether
    the compaction was worth its model call."""
    replacement_seq: int
    locator: str | None = None
    """Where the replaced conversation was written, when it could be written.
    `None` when no spill store took it. The summary is then all that exists, and
    the engine must not claim otherwise."""


@runtime_checkable
class CompactionEngine(Protocol):
    """A compaction backend. It owns the trigger policy, retention and summarizing."""

    async def compact_now(
        self, agent: AgentHandle, *, instructions: str = ""
    ) -> CompactionResult | None:
        """Compact useful history regardless of pressure. Return `None` when there is none.

        `instructions` is the user's own account of the work that comes next. An
        engine may weight the summary towards it. An engine that cannot do so
        may ignore it. That is why it is a parameter with a default and not a
        second method.
        """
        ...

    def trigger_fraction(self) -> float | None:
        """The share of the context window at which this engine compacts on its own,
        or `None` when it does not compact on pressure.

        A row that warns the model before a summary is written for it asks this
        (the readouts of `ph-clm`). The warning can then arrive ahead of the
        summary on any window size, and the row does not write the share down a
        second time beside this one. The `limits` row and the daemon ask too:
        they color a budget reading and the context gauge at the same share.
        """
        ...


@dataclass(frozen=True, slots=True)
class CompactionNote:
    """State that survives a compaction, and the sentence that says so.

    The engine calls `text` with the session at summarize time. `text` returns
    the block to put in the summary prompt, or `""` to contribute nothing.
    `PromptContext` uses the same rule (empty means absent), so a note about a
    namespace that holds nothing costs no prompt.
    """

    name: str
    text: Callable[[Session], str]
    order: int = 0


@dataclass(frozen=True, slots=True)
class _NoteRegistration:
    """A note and who registered it (P6-29).

    `by` is a `Running` and not a single scope, because a `Running` cannot be
    read as the wrong half: `notes()` filters with `by.layer`, and the body of
    the note runs as `by.owner`. One field that holds one of the two, under a
    name that reads like the other, is the conflation of P6-12."""

    note: CompactionNote
    by: Running


@dataclass(slots=True)
class CompactionSeam:
    """The service published as `ctx.compaction`."""

    ctx: Context
    engine: CompactionEngine | None = None
    """At most one. Two answers to "when and how is history replaced" would
    contradict each other, and a profile picks its backend."""
    engine_by: Running | None = None
    """Who registered the engine (P6-29). `claim_slot` sets it and clears it
    together with the engine.

    The seam enters it around `compact_now`. So anything that the body of the
    engine registers unwinds with the row that registered the engine, and not
    with this seam. The automatic triggers already run as that row, because the
    engine row calls its engine from its own listeners.

    The layer stays the layer of the registration. It is **not the agent that
    is being compacted**, although both call sites have that agent in hand. The
    reason is that an engine is one object for the whole deployment and not one
    object per agent. `ph.seams.fs` is the one provider whose target is already
    derived, and it passes that target. P6-24 did not change this. P6-24 fixed
    the boundary in which a *policy* call is judged, and a call into an engine
    is a lifetime question and not a policy question. If this seam is ever
    handed a scope, the `running(self.engine_by)` line is where it goes."""
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

    def trigger_fraction(self) -> float | None:
        """The answer of the engine (`CompactionEngine.trigger_fraction`), or `None`
        when no engine is registered. A profile that compacts nothing has no
        share to warn ahead of."""
        return None if self.engine is None else self.engine.trigger_fraction()

    async def compact_now(
        self, agent: AgentHandle, *, instructions: str = ""
    ) -> CompactionResult | None:
        """An explicit request. With no engine this *is* an error.

        A profile that layered no compaction row has *chosen* not to compact, and
        nothing compacts on its own. But somebody who asked has asked for
        something that the deployment cannot do, and silence would read as
        "nothing to compact".
        """
        engine = self.require()
        with running(self.engine_by):
            return await engine.compact_now(agent, instructions=instructions)

    # ------------------------------------------------------------- the notes --

    def note(self, note: CompactionNote, *, scope: Context | None = None) -> Disposer:
        """Contribute a block to every summary prompt that this context can reach."""
        # Both questions are kept together (P6-29). `notes()` filters with
        # `by.layer` through `reaches`. `by.owner` is the scope of the disposer
        # and the scope that the body runs as. The entry goes through
        # `claim_entry` because `_NoteRegistration` is a frozen dataclass, so it
        # compares by *value*. The `self._notes.remove(entry)` that this
        # replaces would have let the disposal of one row take the equal entry
        # of another row. That is the defect that `_registry` describes.
        by = self.ctx.running_for(scope)
        entry = _NoteRegistration(note=note, by=by)
        return claim_entry(
            by.owner,
            self._notes,
            entry,
            label=f"compaction-note({note.name})",
        )

    def notes(self, session: Session, *, scope: Boundary) -> list[str]:
        """The rendered blocks for one session, by `order` and then by registration order.

        The filter by scope follows the same rule as event dispatch and every
        other scoped registry: a global registration reaches every agent, and an
        agent-scoped registration reaches that agent alone (B7).

        **The boundary is stated** (P6-32), and this reader shows which way the
        axis runs. It used to resolve `scope or self.ctx`. For a registry that
        bears *restrictions*, the mount is the widest answer, and that is the
        fail-open case that P6-32 is named for. This registry bears none:
        `notes` filters on `reaches` alone. So the mount is the **smallest** set,
        and the old default could never include too much. It could only drop a
        note that is scoped to one agent from the summary of that agent. That
        failure is silently *narrow*, and P6-32 rejects it for the same reason
        in the other direction: an empty tool set or an empty prompt degrades a
        model quietly with no error, and nobody notices.

        So the test for a new registry is not "can I name the harm". The test is
        **does this reader bear restrictions**. If it does, an unstated boundary
        widens. If it filters on `reaches` alone, an unstated boundary includes
        too little. In both cases a policy reader takes a boundary. See
        `Deployment`, which states the axis from the other side: widest along
        the restriction axis only.

        A note that raises is dropped, and its traceback is logged. It does not
        stop the compaction, because a summary without one block is worth more
        than a session that is not compacted.
        """
        target = boundary_of(scope, self.ctx)
        rendered: list[str] = []
        visible = [entry for entry in self._notes if entry.by.layer.reaches(target)]
        for entry in sorted(visible, key=lambda one: one.note.order):
            note = entry.note
            try:
                # **Run as the row that registered the note, for the scope that
                # is being compacted** (P6-29). `text` is the body of a row that
                # a registry invokes, in the same category as the `execute` of a
                # tool. It used to run unbound, so anything that it registered
                # landed on the seam and outlived its row. This is one of the
                # four sites that P6-26 could not reach while the binding held a
                # single `Context`: the target here is a *layer*, and the owner
                # is still the registering row.
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


@plugin("compaction", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the compaction seam definition. No engine ships in `ph-base`."""
    ctx.provide(COMPACTION, CompactionSeam(ctx=ctx))
