"""The section map: the model's view of its own context, cut where it may be edited.

A **section** is the run of surface nodes between two consecutive balanced cuts
(`ph.session.balance`): a user message on its own, an assistant reply on its own, or
an assistant message together with every tool result it opened. It is the smallest
unit that can leave the surface without orphaning a call or a result, so it is the
unit every edit names.

**Named by the seq its first node was appended at** (`S412`), which never moves
(A1). The model can read an id on one call and use it on the next, however much was
appended in between. An in-place rewrite keeps the id — the node it lands is a
near-copy of the one it replaced, so the section is the same section — and only a
substitution retires one, after which the refusal names what stands for it.

**Folded once per event.** What a node contributes to the map — its size, its
calls, its preview, where it came from — depends only on its event and the log
before it, neither of which ever changes, so `SectionMap` keeps those facts in a
`SessionFoldCache` and a call pays only for what was appended since the last one.
Only the cut into sections is redone, because the surface moves.

@module ph_clm.sections
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

from ph.llm.types import (
    MediaBlock,
    Message,
    PluginSource,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserSource,
    text_of,
)
from ph.seams.subprocess import first_line
from ph.seams.token_meter import TokenMeter
from ph.session import (
    Session,
    SessionEvent,
    SessionFoldCache,
    cuts_of,
    derive_event_message,
    is_stand_in,
    open_call_delta,
    origin_of,
    shadowed_by,
)
from ph.system_prompt.assembly import is_context_snapshot
from ph.text import block_marker, one_line
from ph.tools.errors import HarnessError

__all__ = [
    "EditRefused",
    "Section",
    "SectionMap",
    "label",
    "render_messages",
    "resolve_one",
    "resolve_run",
    "span_label",
]

SectionKind = Literal["task", "user", "assistant", "step", "context", "revised"]
"""What a section is, for the map and for the protection rules.

`task` is the first message the person typed; `user` any later one. `step` is an
assistant message that made calls, with their results. `context` is plugin-authored
text — the context snapshot, a relayed message. `revised` is text standing in for
conversation that has left the surface (`is_stand_in`): a ph-clm edit or a
compaction summary."""

PREVIEW_CHARS = 80


class EditRefused(HarnessError):
    """A context edit or read declined, with the reason the model reads — a name that
    is not a section, a protected section, a run with a gap, an edit the gate stops."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CONTEXT_EDIT_REFUSED")


@dataclass(frozen=True, slots=True)
class Section:
    """One editable unit of the surface."""

    nodes: tuple[int, ...]
    """Every surface node in the section, in surface order."""
    origins: tuple[int, ...]
    """The seq each node was first appended at, through in-place rewrites of it."""
    kind: SectionKind
    tokens: int
    after: int
    """Tokens in every section after this one — what an edit here makes a provider
    read again, since a prefix cache cannot serve anything past the first change."""
    protected: str | None
    """Why this section may not be edited, or `None`."""
    calls: tuple[str, ...]
    """The names of the calls a `step` made."""
    preview: str
    stands_for: tuple[int, ...]
    """The nodes the replacements in this section took off the surface."""

    @property
    def id(self) -> int:
        return self.origins[0]

    @property
    def name(self) -> str:
        return label(self.id)


def label(seq: int) -> str:
    """A section id as the model reads and writes it."""
    return f"S{seq}"


def span_label(first: int, last: int) -> str:
    """Two section ids as one span, or one id when they are the same."""
    return label(first) if first == last else f"{label(first)}..{label(last)}"


# ------------------------------------------------------------------ the map --


@dataclass(frozen=True, slots=True)
class _Node:
    """What one surface event contributes to a map."""

    origin: int
    shown: bool
    """Whether it projects to a message at all."""
    kind: SectionKind
    """`user` for anything the person typed; the map decides which one is the task."""
    snapshot: bool
    tokens: int
    delta: int
    """`open_call_delta`: how it moves the count of calls awaiting a result."""
    calls: tuple[str, ...]
    preview: str
    replaced: tuple[int, ...]


class SectionMap:
    """The section map of any session, under one row's protection settings."""

    __slots__ = ("_nodes", "meter", "protect_task")

    def __init__(self, meter: TokenMeter, *, protect_task: bool) -> None:
        self.meter = meter
        self.protect_task = protect_task
        self._nodes: SessionFoldCache[dict[int, _Node]] = SessionFoldCache(
            self.fold, extend=self.extend
        )

    def stale(self, sessions: Iterable[Session]) -> list[str]:
        """Every cached map whose facts no longer equal the fold of its log (I6) —
        the delegate `contribute_fold_cache` polls."""
        return self._nodes.stale(sessions)

    def forget(self, session_id: str) -> None:
        """A session left the store: its facts go with it, rather than staying for a
        session nobody can reach."""
        self._nodes.forget(session_id)

    def tokens(self, session: Session) -> int:
        """What every section adds up to, without cutting the surface into them."""
        known = self._nodes.read(session)
        return sum(known[seq].tokens for seq in session.surface.nodes)

    def __call__(self, session: Session) -> tuple[Section, ...]:
        """The session's current surface, as sections, oldest first."""
        known = self._nodes.read(session)
        order = session.surface.nodes
        facts = [known[seq] for seq in order]
        cuts = cuts_of(fact.delta for fact in facts)
        groups = _groups(facts, cuts)
        remaining = sum(fact.tokens for fact in facts)
        sections: list[Section] = []
        task_seen = False
        for start, stop in groups:
            group = facts[start:stop]
            shown = [fact for fact in group if fact.shown]
            kind = shown[0].kind
            if kind == "user" and not task_seen:
                kind, task_seen = "task", True
            tokens = sum(fact.tokens for fact in group)
            remaining -= tokens
            sections.append(
                Section(
                    nodes=tuple(order[start:stop]),
                    origins=tuple(fact.origin for fact in group),
                    kind=kind,
                    tokens=tokens,
                    after=remaining,
                    protected=self._protection(kind, shown, open_call=not cuts[stop]),
                    calls=tuple(name for fact in shown for name in fact.calls),
                    preview=next((fact.preview for fact in shown if fact.preview), ""),
                    stands_for=tuple(seq for fact in group for seq in fact.replaced),
                )
            )
        return tuple(sections)

    def _protection(
        self, kind: SectionKind, shown: Sequence[_Node], *, open_call: bool
    ) -> str | None:
        if open_call:
            return "it holds the step in flight"
        if kind == "task" and self.protect_task:
            return "it is the task (the clm-context row's protectTask)"
        if any(fact.snapshot for fact in shown):
            return "it is the context snapshot, which is re-added whenever it is missing"
        return None

    def fold(self, session: Session) -> dict[int, _Node]:
        """Every surface event's facts, from the start of the log."""
        return self.extend({}, session, 0)

    def extend(self, known: dict[int, _Node], session: Session, start: int) -> dict[int, _Node]:
        """`known`, with the facts of every surface event from `start` on — in place,
        which `SessionFoldCache` allows (`ph.testing.check_fold_laws` says why)."""
        for event in session.events_from(start):
            if event.surface_op is not None:
                known[event.seq] = self._node(session, event)
        return known

    def _node(self, session: Session, event: SessionEvent) -> _Node:
        # A rewrite in place keeps its section's name; a substitution is a new one
        # (`origin_of`, which follows only what `is_in_place_rewrite` calls a rewrite).
        origin = origin_of(session, event.seq)
        replaced = shadowed_by(event)
        message = derive_event_message(event)
        if message is None:
            return _Node(origin, False, "assistant", False, 0, 0, (), "", replaced)
        calls = tuple(block.name for block in message.content if isinstance(block, ToolCallBlock))
        return _Node(
            origin=origin,
            shown=True,
            kind=_kind(event, message, calls),
            snapshot=is_context_snapshot(message),
            tokens=self.meter.measure(message),
            delta=open_call_delta(message),
            calls=calls,
            preview=_preview(message),
            replaced=replaced,
        )


# ---------------------------------------------------------------- resolving --


def resolve_one(
    session: Session, sections: Sequence[Section], name: str, *, editing: bool
) -> Section:
    """The one section `name` means — or a refusal that says why.

    An id that is no longer a section gets an answer the model can act on: which
    section now stands for it, or which section it sits inside. `editing` refuses a
    protected section, which a read does not.
    """
    parts = _span(name)
    if len(parts) != 1:
        raise EditRefused(f"name one section here, not the span {name!r}")
    section = sections[_position(session, sections, _id(parts[0]))]
    if editing and section.protected is not None:
        raise EditRefused(f"{section.name} cannot be edited: {section.protected}")
    return section


def resolve_run(
    session: Session, sections: Sequence[Section], names: Sequence[str]
) -> tuple[int, int]:
    """`[start, stop)`: the positions of the one unbroken, editable run `names` mean.

    A name is one id (`S412`) or a span (`S412..S431`, `S412-S431`) meaning every
    section from one to the other. One run, because a replacement lands where the
    earliest of its nodes was; nothing protected, because the edit would take it.
    """
    chosen: set[int] = set()
    for name in names:
        positions = [_position(session, sections, _id(part)) for part in _span(name)]
        chosen.update(range(min(positions), max(positions) + 1))
    if not chosen:
        raise EditRefused("name at least one section (they look like S412)")
    start, stop = min(chosen), max(chosen) + 1
    if len(chosen) != stop - start:
        gap = next(sections[at].name for at in range(start, stop) if at not in chosen)
        raise EditRefused(
            f"the sections must be one unbroken run, and {gap} sits between them; "
            "edit each run on its own, or include it"
        )
    for section in sections[start:stop]:
        if section.protected is not None:
            raise EditRefused(f"{section.name} cannot be edited: {section.protected}")
    return start, stop


# --------------------------------------------------------------- reading back --


def render_messages(messages: Iterable[Message]) -> str:
    """Messages as plain text, one role-headed block each — for recall and diffs."""
    blocks: list[str] = []
    for message in messages:
        lines: list[str] = []
        for block in message.content:
            match block:
                case TextBlock():
                    lines.append(block.text)
                case ToolCallBlock():
                    lines.append(f"[call {block.name} {block.arguments}]")
                case ToolResultBlock():
                    lines.append(f"[result {block.tool_call_id}]")
                    lines.append(text_of(block.content, placeholder=block_marker))
                case ReasoningBlock() | MediaBlock():
                    lines.append(block_marker(block.type))
                case _ as unhandled:
                    assert_never(unhandled)
        blocks.append(f"[{message.role}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ helpers --


def _span(name: str) -> list[str]:
    return re.split(r"\.\.|-", name, maxsplit=1)


def _id(text: str) -> int:
    """`S412`, `s412` or `412` → 412."""
    stripped = text.strip()
    digits = stripped[1:] if stripped[:1] in ("S", "s") else stripped
    if not digits.isdigit():
        raise EditRefused(f"{text!r} is not a section id (they look like S412)")
    return int(digits)


def _position(session: Session, sections: Sequence[Section], seq: int) -> int:
    """Where section `seq` sits in the map, or a refusal naming what replaced it."""
    for position, section in enumerate(sections):
        if section.id == seq:
            return position
    for section in sections:
        if seq in section.nodes or seq in section.origins:
            raise EditRefused(f"{label(seq)} is inside {section.name}; name the section")
    for section in sections:
        if seq in _ever_stood_for(session, section.nodes):
            raise EditRefused(f"{label(seq)} has been revised; {section.name} stands for it now")
    raise EditRefused(f"{label(seq)} is not a section of this context")


def _ever_stood_for(session: Session, nodes: Iterable[int]) -> set[int]:
    """Every seq the nodes replaced, recursively. Only asked on the refusal path."""
    seen: set[int] = set()
    pending = list(nodes)
    while pending:
        event = session.at(pending.pop())
        fresh = [seq for seq in (() if event is None else shadowed_by(event)) if seq not in seen]
        seen.update(fresh)
        pending.extend(fresh)
    return seen


def _groups(facts: Sequence[_Node], cuts: Sequence[bool]) -> list[tuple[int, int]]:
    """`[start, stop)` index ranges of the sections, in surface order.

    A node that projects to no message (an empty assistant reply hosting a
    max-tokens step's usage) is balanced on both sides, so the cuts alone would make
    it a section of nothing. It joins its neighbor instead: every node belongs to
    exactly one section, so a span of sections is always a span of the surface.
    """
    groups: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(facts) + 1):
        if not (cuts[index] or index == len(facts)):
            continue
        if any(fact.shown for fact in facts[start:index]):
            groups.append((start, index))
        elif groups:
            groups[-1] = (groups[-1][0], index)
        else:
            continue  # leading empty nodes: the first section takes them
        start = index
    return groups


def _kind(event: SessionEvent, message: Message, calls: tuple[str, ...]) -> SectionKind:
    if message.role == "assistant":
        return "step" if calls else "assistant"
    if isinstance(message.source, PluginSource):
        return "revised" if is_stand_in(event) else "context"
    return "user" if isinstance(message.source, UserSource) else "context"


def _preview(message: Message) -> str:
    for block in message.content:
        if isinstance(block, TextBlock):
            line = first_line(block.text)
        elif isinstance(block, ToolResultBlock):
            line = first_line(text_of(block.content))
        else:
            continue
        if line:
            return one_line(line, PREVIEW_CHARS)
    return ""
