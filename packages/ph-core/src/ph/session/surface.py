"""The ordered surface over the append-only log.

The log never changes. The *surface* does. Compaction, pruning and offloading
append an event whose `surfaceOp` replaces a range of nodes. The shadowed
events leave the derivation and stay in the log (invariant I4). That one
mechanism gives compaction, offload, rollback and fork/replay the same shape.
It is also why a checkpointer could not substitute for it (port plan §8).

Ported from dsh `packages/core/session/src/surface.ts`, with one
simplification. The manager of dsh can fold a loaded *window* of a log
(`baseSeq`). pH always holds the whole log, so `seq` is the list index
everywhere (A1).

@module ph.session.surface
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..json import PlainJsonValue, thaw_json
from .events import (
    SessionEvent,
    SurfaceOp,
    SurfaceReplace,
    is_surface_eligible_type,
)

__all__ = [
    "REWRITTEN_IN_PLACE",
    "SurfaceError",
    "SurfaceFoldReplacement",
    "SurfaceFoldResult",
    "SurfaceManager",
    "fold_surface",
    "is_append_surface_event",
    "is_in_place_rewrite",
    "is_replacement_surface_event",
    "is_stand_in",
    "is_surface_event",
    "shadowed_by",
]


class SurfaceError(ValueError):
    """A candidate event violates the surface contract."""


def is_surface_event(event: SessionEvent) -> bool:
    """Whether an event is on the model-visible surface."""
    return is_surface_eligible_type(event.type) and event.surface_op is not None


def is_append_surface_event(event: SessionEvent) -> bool:
    """Whether the event entered the surface at its own log position.

    The surface deliberately shadows replaced ranges. That makes it the wrong
    source for a *human* transcript, because a landed compaction would erase
    conversation that the user already saw. Append-origin events are the source
    material of that transcript. Replacement copies stay model-only.
    """
    return is_surface_event(event) and event.surface_op == "append"


def is_replacement_surface_event(event: SessionEvent) -> bool:
    return is_surface_event(event) and event.surface_op != "append"


def shadowed_by(event: SessionEvent) -> tuple[int, ...]:
    """The nodes that `event` took off the surface. `()` for anything but a replacement."""
    operation = event.surface_op
    if is_replacement_surface_event(event) and isinstance(operation, SurfaceReplace):
        return operation.replaces
    return ()


def is_stand_in(event: SessionEvent) -> bool:
    """Whether this node is text that stands in for conversation that left the surface.

    This is the claim that `PluginSource.form == "compaction"` makes on a
    replacement: the text is a summary that compaction wrote, or a revision by
    a model of its own context. The claim separates the node from the other
    plugin-authored replacement, an offloaded paste, and from a row that only
    names a plugin. A reader that keys on it treats the text as already
    condensed: compaction declines to summarize a summary, and a context editor
    lists it as a revision.

    The function reads `data["source"]`, not `data["message"]["source"]`. A
    `user/message` payload *is* the message, where an `assistant/message` wraps
    one. An assistant node therefore has no top-level source and answers
    `False`, which is the right answer for it.
    """
    source = event.data.get("source")
    return (
        is_replacement_surface_event(event)
        and isinstance(source, Mapping)
        and source.get("kind") == "plugin"
        and source.get("form") == "compaction"
    )


def is_in_place_rewrite(event: SessionEvent) -> bool:
    """Whether a replacement stands for exactly the one node that it replaces.

    A `replace` has two shapes. This function tells them apart by structure and
    does not guess from the producer:

    * A **substitution** stands in for a range with something new: a compaction
      summary, an offloaded paste, or a revision by a model of its context. The
      rows that it shadows leave the view of the model.
    * An **in-place rewrite** replaces a single node with a near-copy of
      itself: a tool result whose content was relocated, or an assistant
      message whose call arguments were elided. Nothing leaves the
      conversation, so a reader should update the row that it already has, and
      not draw a second row and dim the first.

    **The test is by type, because the commit makes the type exact.** An
    `assistant/message` or `tool/result` replacement is refused unless it
    rewrites exactly one node of its own type, cites it, and keeps its message
    id (`_assert_assistant_rewrite`, `_assert_tool_result_rewrite`). So every
    other replacement, which is a substitution, is a `user/message`. The shape
    alone (one node, cited as the source) cannot tell them apart, because a
    substitution that stands for a single message has that shape too.

    A consumer that keyed on `is_replacement_surface_event` alone had to hold a
    list of which producers do which. That is exactly the shape-matching that
    the `form` discriminator replaced one layer up.
    """
    replaced = shadowed_by(event)
    return (
        event.type in REWRITTEN_IN_PLACE
        and len(replaced) == 1
        and tuple(event.source_event_seqs or ()) == replaced
    )


REWRITTEN_IN_PLACE = frozenset({"assistant/message", "tool/result"})
"""The types that a replacement may rewrite in place. It may rewrite no other type in place."""


@dataclass(frozen=True, slots=True)
class SurfaceFoldReplacement:
    seq: int
    shadowed_seqs: tuple[int, ...]
    """The nodes that this event took out, in **surface order**.

    This is the set and not a `start`/`end` range. A range is only an encoding
    of the set, and a caller that asks "what did this compaction shadow" wants
    the set.
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
    """The nodes that this replacement takes out, in surface order.

    **Membership, never arithmetic.** Every named seq must be a node now, or
    the whole operation is refused. There is no "between". A name that is
    already shadowed cannot be silently shadowed again, and a range cannot
    drift over messages that nobody cited.

    The result is in surface order and not in the order that the writer listed
    the names. A consumer that reads `shadowed_seqs` therefore sees the order
    of the conversation, and not an accident of how the set was built.

    **One pass with a set, not `.index` per name.** To resolve each name by a
    scan of the node list from 0 is O(names x nodes). That is quadratic in the
    number of names, and the number of names is what grows with compaction.
    """
    if len(op.replaces) == 1:
        # **The common case keeps the cheap test.** A membership scan stops at
        # the node that it finds, so the cost to replace an early node is its
        # position. The set below always costs the whole surface. For a
        # single-name replace, it is slower to build the set than to go without
        # it, and almost every replace names exactly one node.
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
    """A `tool/result` replacement may change only the content of the result.

    Offloading rewrites one result in place (G2/C5). If it could also change
    the call id or the error identity, a "spill" could silently rewrite what
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


def _assert_assistant_rewrite(
    event: SessionEvent, shadowed_seqs: Sequence[int], log: Sequence[SessionEvent]
) -> None:
    """An `assistant/message` replacement rewrites one reply in place, or nothing.

    This is the reply-side counterpart of `_assert_tool_result_rewrite`. It is
    what lets `is_in_place_rewrite` decide by type. A replacement in assistant
    role that named a range, or carried a message of its own, would be new
    speech that the model never made. Readers that skip the calls of a rewrite
    as already answered (repair, the todo row) would also skip the calls of a
    substitution. Text that stands in for a range is a `user/message`. Every
    log that this format reads already agrees: the one producer of assistant
    replacements has always rewritten a single reply in place.
    """
    if event.type != "assistant/message":
        return
    original = log[shadowed_seqs[0]] if len(shadowed_seqs) == 1 else None
    if original is None or original.type != "assistant/message":
        raise SurfaceError(
            "assistant/message surface replacement must rewrite exactly one current "
            "assistant/message; text standing in for a range is a user/message"
        )
    if tuple(event.source_event_seqs or ()) != tuple(shadowed_seqs):
        raise SurfaceError(
            "assistant/message surface replacement must cite only the reply it rewrites"
        )
    if _message_id(event) != _message_id(original):
        raise SurfaceError(
            "assistant/message surface replacement must keep the message id; a new "
            "message in place of a reply is a substitution, a user/message"
        )


def _message_id(event: SessionEvent) -> object:
    message = event.data.get("message")
    return message.get("id") if isinstance(message, Mapping) else None


def _blank_result_content(data: object) -> PlainJsonValue:
    """A `tool/result` payload with the content of every result block blanked out.

    **Every block is blanked, and the list keeps its length.** An earlier
    version blanked only the first block and collapsed the list to it, so it
    compared two payloads by their first block alone. A replacement could drop
    the blocks behind the first, or add one with a different `toolCallId` or
    `isError`, and still read as "content only". This narrowing is what stops a
    spill from rewriting what the model is told happened. It must therefore
    hold for the whole message and not only for its first block.
    """
    plain = thaw_json(data)
    message = plain.get("message") if isinstance(plain, dict) else None
    if isinstance(message, dict):
        blocks = message.get("content")
        if isinstance(blocks, list):
            # `thaw_json` above rebuilds every mapping as a plain `dict`, so a
            # proxy cannot reach here. A non-mapping element can, from a
            # hand-edited log. It is kept and not dropped, because a lost block
            # is what this comparison exists to catch.
            message["content"] = [
                {**block, "content": None} if isinstance(block, dict) else block for block in blocks
            ]
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
    _assert_assistant_rewrite(event, shadowed, log)
    return _ReplacePlan(seq=event.seq, shadowed_seqs=shadowed)


def _replace_reads_the_batch(events: Sequence[SessionEvent]) -> bool:
    """Whether a replace in `events` follows a member that moved the surface.

    That is the one case in which the plan depends on the effects of the batch
    itself (`validate_batch`)."""
    moved = False
    for event in events:
        op = _surface_op_of(event)
        if isinstance(op, SurfaceReplace) and moved:
            return True
        moved = moved or op is not None
    return False


def _apply(
    state: _FoldState, plan: _AppendPlan | _ReplacePlan | None
) -> SurfaceFoldReplacement | None:
    if isinstance(plan, _AppendPlan):
        state.nodes.append(plan.seq)
        return None
    if isinstance(plan, _ReplacePlan):
        # `shadowed_seqs` is in surface order, so its first element is the
        # earliest node that this takes out, and the replacement lands there.
        # Nothing before that node was shadowed, so the count of survivors
        # before it is unchanged and the index needs no adjustment.
        insert_at = state.nodes.index(plan.shadowed_seqs[0])
        if len(plan.shadowed_seqs) == 1:
            # **The common case, and the one that must stay O(1).** Every
            # `truncate_arguments`, every offloaded tool result and every elided
            # paste is one node that stands in for one node. To rebuild the whole
            # list for those is work proportional to the surface per rewrite.
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

    This is the offline counterpart of `SurfaceManager`. An external
    reconstructor folds the same rules over a stored log and must reach the
    same nodes.
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

    The manager holds a live reference to the log list of the session and folds
    lazily. The cost to read the surface is therefore O(events appended since
    the last read).
    """

    __slots__ = ("_log", "_processed", "_state")

    def __init__(self, log: list[SessionEvent]) -> None:
        self._log = log
        self._state = _FoldState()
        self._processed = 0

    def validate_next(self, event: SessionEvent) -> None:
        """Validate a candidate *before* it enters the log.

        The transition is planned and not committed. That is why a rejected
        append leaves the surface untouched. A partially mutated surface could
        not be recovered. The committed fold plans the event again after it is
        in the log. The state does not change in between, so the outcome is the
        same.
        """
        self._process_delta()
        _plan(self._state, event, len(self._log), self._log)

    def validate_batch(self, events: Sequence[SessionEvent]) -> None:
        """Validate candidates that will enter the log together (P10-14).

        Each candidate is planned against the state that the candidates before
        it would leave, on a **scratch copy** of the fold state. A member may
        therefore replace a node that an earlier member added, and a refusal
        anywhere leaves the live state as it was. `validate_next` needs no copy
        because it plans one event and applies nothing. A batch must apply each
        plan to plan the next.

        **There is no copy unless a replace can see the effects of the batch
        itself**, and almost no batch has such a replace. The batch of every
        step has no replace at all: the driver takes a prompt from the inbox in
        one batch with `step/start` and its `user/message` (S3). An append, or
        a record that is not on the surface, reads neither the fold state nor
        the log, so each is planned where it stands. That costs O(members) and
        not O(nodes + events). A batch per step would otherwise pay the larger
        cost on a log that grows by one `assistant/chunk` per token.

        A replace with nothing before it that moved the surface shadows only
        committed nodes. Examples are the `offload/spilled` of the overflow
        clip beside its rewrite, and an elided paste beside its accounting. The
        live state and the live log answer such a replace exactly as the copy
        would. Only a replace behind an append or behind another replace needs
        the scratch copy: O(nodes) for the state, and O(events) for the log
        that its rewrite check may index.
        """
        self._process_delta()
        base = len(self._log)
        if not _replace_reads_the_batch(events):
            for offset, event in enumerate(events):
                _plan(self._state, event, base + offset, self._log)
            return
        state = _FoldState(list(self._state.nodes), self._state.replace_generation)
        log = [*self._log, *events]
        for offset, event in enumerate(events):
            _apply(state, _plan(state, event, base + offset, log))

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
        """The surface from position `index` on. An incremental reader needs this."""
        self._process_delta()
        return tuple(self._state.nodes[index:])

    def _process_delta(self) -> None:
        log = self._log
        while self._processed < len(log):
            seq = self._processed
            _apply(self._state, _plan(self._state, log[seq], seq, log))
            self._processed = seq + 1
