"""The section map: the model's view of its own context, divided where it can be edited.

A **section** is the run of surface nodes between two consecutive balanced cuts
(`ph.session.balance`). It is a user message alone, an assistant reply alone, or
an assistant message together with every tool result that it opened. A section
is the smallest unit that can leave the surface without leaving a call or a
result with no partner. It is therefore the unit that every edit names.

**A section is named by the seq at which its first node was appended** (`S412`).
That seq never changes (A1). The model can read an id on one call and use it on
the next call, whatever was appended in between. An in-place rewrite keeps the
id. The node that it lands is a near-copy of the node that it replaced, so the
section is the same section. Only a substitution retires an id. After that, the
refusal names the section that stands for the retired one.

**The facts of each event are folded one time.** A node contributes facts to the
map: its size, its calls, its preview, and its origin. Those facts depend only on
the event of the node and the log before it, and neither ever changes.
`SectionMap` therefore keeps the facts in a `SessionFoldCache`, and a call pays
only for the events that were appended since the last call. Only the division
into sections is computed again, because the surface changes.

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
from ph.text import one_line
from ph.tools.errors import HarnessError

__all__ = [
    "EditRefused",
    "Section",
    "SectionMap",
    "label",
    "resolve_one",
    "resolve_run",
    "span_label",
]

SectionKind = Literal["task", "user", "assistant", "step", "context", "revised"]
"""The kind of a section, for the map and for the protection rules.

`task` is the first message that the user typed, and `user` is any later one.
`step` is an assistant message that made calls, together with their results.
`context` is text that a plugin wrote, such as the context snapshot or a relayed
message. `revised` is text that stands in for conversation that has left the
surface (`is_stand_in`): a ph-clm edit or a compaction summary."""

PREVIEW_CHARS = 80


class EditRefused(HarnessError):
    """A context edit or read that was declined, with the reason that the model
    reads. Examples: a name that is not a section, a protected section, a run
    with a gap, or an edit that the gate stops."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CONTEXT_EDIT_REFUSED")


@dataclass(frozen=True, slots=True)
class Section:
    """One editable unit of the surface."""

    nodes: tuple[int, ...]
    """Every surface node in the section, in surface order."""
    origins: tuple[int, ...]
    """The seq at which each node was first appended, traced back through its
    in-place rewrites."""
    kind: SectionKind
    tokens: int
    after: int
    """The tokens in every section after this one. An edit here makes a provider
    read them again, because a prefix cache cannot serve anything past the first
    change."""
    protected: str | None
    """Why this section may not be edited, or `None`."""
    calls: tuple[str, ...]
    """The names of the calls that a `step` made."""
    preview: str
    stands_for: tuple[int, ...]
    """The nodes that the replacements in this section took off the surface."""

    @property
    def id(self) -> int:
        return self.origins[0]

    @property
    def name(self) -> str:
        return label(self.id)


def label(seq: int) -> str:
    """Format a section id as the model reads and writes it."""
    return f"S{seq}"


def span_label(first: int, last: int) -> str:
    """Format two section ids as one span, or as one id when they are the same."""
    return label(first) if first == last else f"{label(first)}..{label(last)}"


# ------------------------------------------------------------------ the map --


@dataclass(frozen=True, slots=True)
class _Node:
    """The facts that one surface event contributes to a map."""

    origin: int
    shown: bool
    """Whether the event projects to a message at all."""
    kind: SectionKind
    """`user` for everything that the user typed. The map decides which one is the
    task."""
    snapshot: bool
    tokens: int
    delta: int
    """`open_call_delta`: how the event changes the count of calls that wait for a
    result."""
    calls: tuple[str, ...]
    preview: str
    replaced: tuple[int, ...]


class SectionMap:
    """The section map of any session, with the protection settings of one row."""

    __slots__ = ("_nodes", "meter", "protect_task")

    def __init__(self, meter: TokenMeter, *, protect_task: bool) -> None:
        self.meter = meter
        self.protect_task = protect_task
        self._nodes: SessionFoldCache[dict[int, _Node]] = SessionFoldCache(
            self.fold, extend=self.extend
        )

    def stale(self, sessions: Iterable[Session]) -> list[str]:
        """Each cached map whose facts no longer equal the fold of its log (I6).
        `contribute_fold_cache` polls this method."""
        return self._nodes.stale(sessions)

    def __call__(self, session: Session) -> tuple[Section, ...]:
        """Return the current surface of the session as sections, oldest first."""
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
        """Compute the facts of every surface event, from the start of the log."""
        return self.extend({}, session, 0)

    def extend(self, known: dict[int, _Node], session: Session, start: int) -> dict[int, _Node]:
        """Add to `known` the facts of every surface event from `start` onward.
        `known` is changed in place, which `SessionFoldCache` allows
        (`ph.testing.check_fold_laws` says why)."""
        for event in session.events_from(start):
            if event.surface_op is not None:
                known[event.seq] = self._node(session, event)
        return known

    def _node(self, session: Session, event: SessionEvent) -> _Node:
        # An in-place rewrite keeps the name of its section. A substitution gets a
        # new name. (`origin_of` follows only what `is_in_place_rewrite` calls a
        # rewrite.)
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
            tokens=self.meter.node_tokens(session, event.seq),
            delta=open_call_delta(message),
            calls=calls,
            preview=_preview(message),
            replaced=replaced,
        )


# ---------------------------------------------------------------- resolving --


def resolve_one(
    session: Session, sections: Sequence[Section], name: str, *, editing: bool
) -> Section:
    """Return the one section that `name` means, or raise a refusal that gives the
    reason.

    An id that is no longer a section gets an answer that the model can act on:
    the section that now stands for it, or the section that contains it. With
    `editing`, a protected section is refused. A read does not refuse it.
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
    """Return `[start, stop)`: the positions of the one unbroken, editable run
    that `names` mean.

    A name is one id (`S412`) or a span (`S412..S431`, `S412-S431`). A span means
    every section from the first id to the second. The names must give one run,
    because a replacement lands where the earliest of its nodes was. The run must
    contain nothing protected, because the edit would remove it.
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
    """Return the position of section `seq` in the map, or raise a refusal that
    names what replaced it."""
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
    """Every seq that the nodes replaced, recursively. Called only on the refusal
    path."""
    seen: set[int] = set()
    pending = list(nodes)
    while pending:
        event = session.at(pending.pop())
        fresh = [seq for seq in (() if event is None else shadowed_by(event)) if seq not in seen]
        seen.update(fresh)
        pending.extend(fresh)
    return seen


def _groups(facts: Sequence[_Node], cuts: Sequence[bool]) -> list[tuple[int, int]]:
    """Return the `[start, stop)` index ranges of the sections, in surface order.

    Some nodes project to no message. An example is an empty assistant reply that
    carries the usage of a max-tokens step. Such a node is balanced on both
    sides, so the cuts alone would make it a section with no content. It joins
    its neighbor instead. Every node then belongs to exactly one section, so a
    span of sections is always a span of the surface.
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
    """The first line of text that a message shows: its words, or the output of a
    result."""
    for block in message.content:
        match block:
            case TextBlock():
                line = first_line(block.text)
            case ToolResultBlock():
                line = first_line(text_of(block.content))
            case ToolCallBlock() | ReasoningBlock() | MediaBlock():
                continue
            case _ as unhandled:
                assert_never(unhandled)
        if line:
            return one_line(line, PREVIEW_CHARS)
    return ""
