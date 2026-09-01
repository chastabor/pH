"""The ordered surface over the append-only log.

The log never changes; the *surface* does. Compaction, pruning and offloading
append an event whose `surfaceOp` replaces a range of nodes — the shadowed
events leave the derivation and stay in the log (invariant I4). That one
mechanism gives compaction, offload, rollback and fork/replay the same shape,
and is why a checkpointer could not substitute for it (port plan §8).

Ported from dsh `packages/core/session/src/surface.ts`. One simplification:
dsh's manager can fold a loaded *window* of a log (`baseSeq`); pH always holds
the whole log, so `seq` is the list index everywhere (A1).

@module ph.session.surface
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .events import (
    SessionEvent,
    SurfaceOp,
    SurfaceReplace,
    is_surface_eligible_type,
)
from .json import thaw_json

__all__ = [
    "SurfaceError",
    "SurfaceFoldReplacement",
    "SurfaceFoldResult",
    "SurfaceManager",
    "fold_surface",
    "is_append_surface_event",
    "is_in_place_rewrite",
    "is_replacement_surface_event",
    "is_surface_event",
]


class SurfaceError(ValueError):
    """A candidate event violates the surface contract."""


def is_surface_event(event: SessionEvent) -> bool:
    """Whether an event is on the model-visible surface."""
    return is_surface_eligible_type(event.type) and event.surface_op is not None


def is_append_surface_event(event: SessionEvent) -> bool:
    """Whether the event entered the surface at its own log position.

    The surface deliberately shadows replaced ranges, which makes it the wrong
    source for a *human* transcript — a landed compaction would erase
    conversation the user already saw. Append-origin events are that
    transcript's source material; replacement copies stay model-only.
    """
    return is_surface_event(event) and event.surface_op == "append"


def is_replacement_surface_event(event: SessionEvent) -> bool:
    return is_surface_event(event) and event.surface_op != "append"


def is_in_place_rewrite(event: SessionEvent) -> bool:
    """Whether a replacement stands for exactly the one node it replaces.

    The two shapes a `replace` comes in, told apart structurally rather than by
    guessing from the producer. A **substitution** stands in for a range with
    something new — a compaction summary, an offloaded paste — and the rows it
    shadows leave the model's view. An **in-place rewrite** replaces a single
    node with a near-copy of itself: a tool result whose content was relocated,
    an assistant message whose call arguments were elided. Nothing leaves the
    conversation, so a reader should update the row it already has rather than
    draw a second one and dim the first.

    A consumer that keyed on `is_replacement_surface_event` alone had to hold a
    list of which producers do which — exactly the shape-matching that the
    `form` discriminator replaced one layer up.
    """
    operation = event.surface_op
    if not is_replacement_surface_event(event) or not isinstance(operation, SurfaceReplace):
        return False
    return (
        len(operation.replaces) == 1 and tuple(event.source_event_seqs or ()) == operation.replaces
    )


@dataclass(frozen=True, slots=True)
class SurfaceFoldReplacement:
    seq: int
    shadowed_seqs: tuple[int, ...]
    """The nodes this event took out, in **surface order**.

    `start`/`end` used to sit beside this and no consumer ever read them — the
    range was an encoding of the set, and the set is what anyone asking "what did
    this compaction shadow" actually wants.
    """


@dataclass(frozen=True, slots=True)
class SurfaceFoldResult:
    nodes: tuple[int, ...]
    replacements: tuple[SurfaceFoldReplacement, ...]


@dataclass(slots=True)
class _FoldState:
    nodes: list[int] = field(default_factory=list)
    replace_generation: int = 0


@dataclass(frozen=True, slots=True)
class _AppendPlan:
    seq: int


@dataclass(frozen=True, slots=True)
class _ReplacePlan:
    seq: int
    shadowed_seqs: tuple[int, ...]


def _surface_op_of(event: SessionEvent) -> SurfaceOp | None:
    """Validate event-local eligibility and return its operation."""
    if not is_surface_eligible_type(event.type):
        if event.surface_op is not None:
            raise SurfaceError(
                f'session event "{event.type}" is not surface-eligible and cannot carry surfaceOp'
            )
        if event.source_event_seqs is not None:
            raise SurfaceError(
                f'session event "{event.type}" is not surface-eligible and cannot '
                "carry sourceEventSeqs"
            )
        return None
    if event.surface_op is None:
        raise SurfaceError(
            f'session event "{event.type}" is surface-eligible and requires a surfaceOp marker'
        )
    return event.surface_op


def _assert_provenance(event: SessionEvent, shadowed_seqs: Sequence[int]) -> None:
    """Check cited source seqs against the log order and the shadowed range."""
    sources: set[int] = set()
    raw = event.source_event_seqs
    if raw is not None:
        if len(raw) == 0 and event.type != "assistant/message":
            raise SurfaceError("sourceEventSeqs must not be empty except on assistant/message")
        non_earlier = next((source for source in raw if source >= event.seq), None)
        sources.update(raw)
        if len(sources) != len(raw):
            raise SurfaceError("sourceEventSeqs must not contain duplicates")
        if non_earlier is not None:
            raise SurfaceError(
                f"sourceEventSeqs must reference earlier events: {non_earlier} >= "
                f"current seq {event.seq}"
            )
    missing = [seq for seq in shadowed_seqs if seq not in sources]
    if missing:
        raise SurfaceError(
            "surface replace: sourceEventSeqs must include every shadowed surface "
            f"node; missing {', '.join(str(seq) for seq in missing)}"
        )


def _shadowed(state: _FoldState, op: SurfaceReplace) -> tuple[int, ...]:
    """Which nodes this replacement takes out, in surface order.

    **Membership, never arithmetic.** Every named seq has to be a node right now or
    the whole operation is refused — there is no "between", so a name that has already
    been shadowed cannot be silently re-shadowed and a range cannot drift over
    messages nobody cited.

    Returned in surface order rather than the order the writer listed them, so a
    consumer reading `shadowed_seqs` sees the conversation's order and not an accident
    of how the set was built.

    **One pass with a set, not `.index` per name** — the alternative is quadratic in
    the number of names, which is precisely the direction compaction grows. Measured
    in `tests/test_surface.py`.
    """
    if len(op.replaces) == 1:
        # **The common case keeps the cheap test.** A membership scan stops at
        # the node it finds, so replacing an early one costs its position; the
        # set below always costs the whole surface. Building it here made a pass
        # of 200 in-place rewrites on a 2 000-node surface *slower* than the
        # range op it replaced — 16.7 ms against 9.8 — because every one of them
        # names exactly one node.
        only = op.replaces[0]
        if only not in state.nodes:
            raise SurfaceError(f"surface replace: seq {only} is not a current surface node")
        return (only,)
    present = set(state.nodes)
    absent = next((seq for seq in op.replaces if seq not in present), None)
    if absent is not None:
        raise SurfaceError(f"surface replace: seq {absent} is not a current surface node")
    named = set(op.replaces)
    return tuple(node for node in state.nodes if node in named)


def _assert_tool_result_rewrite(
    event: SessionEvent, shadowed_seqs: Sequence[int], log: Sequence[SessionEvent]
) -> None:
    """A `tool/result` replacement may change only the result's content.

    Offloading rewrites one result in place (G2/C5). Letting it change the call
    id or the error identity as well would let a "spill" silently rewrite what
    the model is told happened.
    """
    if event.type != "tool/result":
        return
    if len(shadowed_seqs) != 1:
        raise SurfaceError("tool/result surface replacement must rewrite exactly one current node")
    original = log[shadowed_seqs[0]] if 0 <= shadowed_seqs[0] < len(log) else None
    if original is None or original.type != "tool/result":
        raise SurfaceError("tool/result surface replacement must target a current tool/result")
    if _blank_result_content(original.data) != _blank_result_content(event.data):
        raise SurfaceError("tool/result surface replacement may change only content")


def _blank_result_content(data: Any) -> Any:
    """A `tool/result` payload with the result block's content blanked out."""
    plain = thaw_json(data)
    message = plain.get("message") if isinstance(plain, dict) else None
    if isinstance(message, dict):
        blocks = message.get("content")
        if isinstance(blocks, list) and blocks:
            first = blocks[0]
            if isinstance(first, (dict, MappingProxyType)):
                message["content"] = [{**dict(first), "content": None}]
    return plain


def _plan(
    state: _FoldState, event: SessionEvent, expected_seq: int, log: Sequence[SessionEvent]
) -> _AppendPlan | _ReplacePlan | None:
    if event.seq != expected_seq:
        raise SurfaceError(
            f"session event seq {event.seq} is not contiguous; expected {expected_seq}"
        )
    op = _surface_op_of(event)
    if op is None:
        return None
    if op == "append":
        _assert_provenance(event, ())
        return _AppendPlan(seq=event.seq)
    assert isinstance(op, SurfaceReplace)
    shadowed = _shadowed(state, op)
    _assert_provenance(event, shadowed)
    _assert_tool_result_rewrite(event, shadowed, log)
    return _ReplacePlan(seq=event.seq, shadowed_seqs=shadowed)


def _apply(
    state: _FoldState, plan: _AppendPlan | _ReplacePlan | None
) -> SurfaceFoldReplacement | None:
    if isinstance(plan, _AppendPlan):
        state.nodes.append(plan.seq)
        return None
    if isinstance(plan, _ReplacePlan):
        # `shadowed_seqs` is in surface order, so its first element is the
        # earliest node this takes out — which is where the replacement lands.
        # Nothing ahead of that was shadowed, so the count of survivors before it
        # is unchanged and the index needs no adjusting.
        insert_at = state.nodes.index(plan.shadowed_seqs[0])
        if len(plan.shadowed_seqs) == 1:
            # **The common case, and the one that must stay O(1).** Every
            # `truncate_arguments`, every offloaded tool result and every elided
            # paste is one node standing in for one node. Rebuilding the whole
            # list for those measured 3.3x slower across a realistic pass than
            # the same-length slice assignment the range op used to do.
            state.nodes[insert_at] = plan.seq
        else:
            shadowed = set(plan.shadowed_seqs)
            state.nodes[:] = [node for node in state.nodes if node not in shadowed]
            state.nodes.insert(insert_at, plan.seq)
        state.replace_generation += 1
        return SurfaceFoldReplacement(seq=plan.seq, shadowed_seqs=plan.shadowed_seqs)
    return None


def fold_surface(events: Sequence[SessionEvent]) -> SurfaceFoldResult:
    """Replay a complete log through the canonical surface fold.

    The offline counterpart of `SurfaceManager`: an external reconstructor folds
    the same rules over a stored log and must reach the same nodes.
    """
    state = _FoldState()
    replacements: list[SurfaceFoldReplacement] = []
    for index, event in enumerate(events):
        replacement = _apply(state, _plan(state, event, index, events))
        if replacement is not None:
            replacements.append(replacement)
    return SurfaceFoldResult(nodes=tuple(state.nodes), replacements=tuple(replacements))


class SurfaceManager:
    """Incremental ordered surface view and append-boundary validator.

    Holds a live reference to the session's log list and folds lazily, so the
    cost of reading the surface is O(events appended since the last read).
    """

    __slots__ = ("_log", "_processed", "_state")

    def __init__(self, log: list[SessionEvent]) -> None:
        self._log = log
        self._state = _FoldState()
        self._processed = 0

    def validate_next(self, event: SessionEvent) -> None:
        """Validate a candidate *before* it enters the log.

        Planning the transition without committing it is what makes a rejected
        append leave the surface untouched — a partially mutated surface would
        be unrecoverable. The committed fold re-plans the event once it is in
        the log; the state is unchanged in between, so the outcome is the same.
        """
        self._process_delta()
        _plan(self._state, event, len(self._log), self._log)

    @property
    def replace_generation(self) -> int:
        """Monotonic count of committed positional replacements."""
        self._process_delta()
        return self._state.replace_generation

    @property
    def node_count(self) -> int:
        self._process_delta()
        return len(self._state.nodes)

    @property
    def nodes(self) -> tuple[int, ...]:
        """Surface event seqs in model-visible order."""
        return self.nodes_from(0)

    def nodes_from(self, index: int) -> tuple[int, ...]:
        """The surface from position `index` on — what an incremental reader needs."""
        self._process_delta()
        return tuple(self._state.nodes[index:])

    def _process_delta(self) -> None:
        log = self._log
        while self._processed < len(log):
            seq = self._processed
            _apply(self._state, _plan(self._state, log[seq], seq, log))
            self._processed = seq + 1
