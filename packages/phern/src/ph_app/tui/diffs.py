"""What changed, grouped as git groups commits by file: the projection behind `/diff`.

Each group is one thing as it is now, with every change that led to it, oldest
first. That is what `git log -p -- file` shows for a file. Each entry is one
change: a `before` and an `after`, the seq that shows it in the transcript, and
the records that its producer wrote for it. A group also holds its whole history
in one diff: the thing as first written against the thing now, as
`git diff A..B -- file` does. The screen (`ph_app.tui.diff_screen`) draws
whatever this list holds, so it knows nothing about where a group came from.

**Meant to grow more kinds.** Context revisions are the one source today. A
workspace file change or a profile diff would be one more `DiffKind` member and
one more source function in `SOURCES`, beside `revisions`. Neither the screen
nor the groups that the other sources produce would change. `SOURCES` is a tuple
and not a registry. The sources belong to this package, and a seam with a single
member is one that nobody can check. `ph.seams.tui_screens` says the same about
its one slot.

**A revision is any surface replacement, whoever wrote it.** That covers the
summary of compaction, its argument elision and its overflow clip, the paste
preview of input-offload, and the edit that a model makes through ph-clm. All of
them go through `ph.session.revise`, so one reading of the log covers every one.
The app may not import a bundle to learn its types (`test_app_layering.py`).
Each revision is classified from the event alone, as a substitution, a stand-in
or a rewrite. `ph.session.shape_of` does that, through
`ph_app.revisions.read_revision`, which the training export shares.

**The group of a revision is the surface node that it led to**
(`Session.history`). Every revision ends up in exactly one group. A node is
shadowed one time, so the revisions behind the current surface form a tree for
each node. The history merges, unlike the one line per file of git: a summary
over two sections that a model had edited holds both edits and itself. Grouping
by the sections of ph-clm would need the app to read the map of a bundle, which
it may not do (`test_app_layering.py`). A section is a run of these nodes in any
case.

**The before of an entry is what the change replaced, not what was first
written.** A revision of a revision is diffed against the revision that it
replaced, so stepping through a group shows each edit one time. The whole
history of the group is the conversation as first written (`originals`) against
the node now, which is also what the `context_diff` of ph-clm compares. Both
sides render through `render_messages`, and both diffs are `unified_diff`, so a
revision reads the same here as in that tool.

**Producer records are shown, not interpreted.** These are the records that the
producer of a change wrote through the revision door, which stamps each record
with the replacements that it explains (`Session.records_of`). A record can be
`clm/revised`, `compaction/summarized`, `offload/input-spilled`, or a type that
this process has never heard of. Each record is listed field by field, the way
the trajectory describes a type that it has no phrase for (`wire.fields_of`).
What the door wrote on the record is kept apart from those fields: what the
record revises, and its `size`. The `size` becomes the recorded size of the
entry when the record explains this change alone. A replacement that was written
before the door stamped its records has none.

**Nothing is rendered until a row is shown, and nothing rendered is kept.** The
before of a summary is every node that it shadowed, which can be most of the
conversation. So building the list renders no text. An entry holds how to render
its two sides (`texts`), and a group holds how to render its whole history
(`whole`). The screen renders the row that it shows, and releases the strings
when the row changes. An entry with no recorded size has its characters counted
at that time. No tokenizer runs. The client has no meter for the session, and a
count that disagreed with the count of the producer would be worse than
characters that are labeled as characters.

**A limit, recorded and not fixed.** The transcript draws nothing for an
in-place rewrite of an assistant message, because the text is already on screen
(`TuiEventAdapter._on_assistant_message`). So this list is the only place where
a person sees one.

**Textual-free**, like `trajectory`. It takes a `Session`. A stored log becomes
one through `ph.persistence.read_session` with nothing mounted, so a viewer over
a file can reuse this module.

@module ph_app.tui.diffs
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias

from ph.json import as_obj, as_str
from ph.session import (
    RevisionShape,
    RevisionSize,
    Session,
    SessionEvent,
    is_replacement_surface_event,
    messages_of,
    origin_of,
    originals,
    producer_data,
    render_messages,
    revises_of,
)
from ph.text import count_of

from ..revisions import read_revision
from ..wire import fields_of

__all__ = [
    "SOURCES",
    "DiffEntry",
    "DiffGroup",
    "DiffKind",
    "DiffSource",
    "ProducerRecord",
    "build_diffs",
    "revisions",
]

DiffKind: TypeAlias = Literal["revision"]
"""Where an entry came from. The screen shows it as a column, so that a person
who reads a mixed list can tell the kinds apart."""


@dataclass(frozen=True, slots=True)
class ProducerRecord:
    """A record that was written beside a change, field by field and not interpreted."""

    type: str
    seq: int
    fields: tuple[tuple[str, str], ...]
    """The result of `wire.fields_of`, in the order of the payload."""
    revises: tuple[int, ...] = ()
    """The replacements that the record explains. It names more than this one when
    it is one record for several changes, as compaction writes one for every
    argument it elided. The numbers of such a record describe all of those
    changes, so none of them is the size of this entry."""


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One change: what it was, and what it is."""

    seq: int
    """The log seq at which the change was written. The entries of a group are
    ordered by it."""
    title: str
    summary: str
    producer: str | None
    """Who made the change, as the log reports it: the name of a plugin, or the
    types of the records beside the change. It is `None` when the log does not
    say."""
    texts: Callable[[], tuple[str, str]] = field(compare=False, repr=False)
    """`(before, after)`, rendered on each call and kept by nobody here."""
    facts: tuple[str, ...] = ()
    """Lines that the source wrote about this change, shown above the diff."""
    records: tuple[ProducerRecord, ...] = ()
    recorded: RevisionSize | None = None
    """The size a producer record gave, when one did."""


@dataclass(frozen=True, slots=True)
class DiffGroup:
    """One thing as it is now, and every change that led to it."""

    kind: DiffKind
    seq: int
    """Where the thing is now. Groups are ordered by it, and a screen opened at a
    transcript position selects the nearest group at or before it."""
    reveal: int
    """The seq of the transcript row that shows the thing now."""
    entries: tuple[DiffEntry, ...]
    """Oldest first. The last entry is the change that made the thing what it is
    now."""
    whole: Callable[[], tuple[str, str]] = field(compare=False, repr=False)
    """`(as first written, now)`, rendered on each call and kept by nobody here."""

    @property
    def latest(self) -> DiffEntry:
        return self.entries[-1]

    @property
    def producers(self) -> str:
        """Who made its changes, each producer one time, oldest first."""
        named = (entry.producer for entry in self.entries if entry.producer is not None)
        return ", ".join(dict.fromkeys(named))


DiffSource: TypeAlias = Callable[[Session], list[DiffGroup]]
"""`source(session) -> groups`, in log order. One per `DiffKind`."""


def build_diffs(session: Session) -> list[DiffGroup]:
    """Every source's groups, ordered by seq."""
    return sorted(
        (group for source in SOURCES for group in source(session)), key=lambda one: one.seq
    )


# ---------------------------------------------------------------- revisions --


def revisions(session: Session) -> list[DiffGroup]:
    """One group per surface node a revision led to, in surface order."""
    groups: list[DiffGroup] = []
    for node in session.surface.nodes:
        history = session.history(node)
        if not history:
            continue
        groups.append(
            DiffGroup(
                kind="revision",
                seq=node,
                reveal=origin_of(session, node),
                entries=tuple(_entry(session, event) for event in history),
                whole=partial(_whole, session, node),
            )
        )
    return groups


def _entry(session: Session, event: SessionEvent) -> DiffEntry:
    """One revision, against what it replaced."""
    revision = read_revision(session, event)
    records = tuple(_described(record) for record in revision.records)
    stated = as_str(as_obj(event.data.get("source")).get("summary"))
    return DiffEntry(
        seq=event.seq,
        title=revision.shape,
        summary=stated or _summary(revision.shape, event, revision.replaced),
        producer=revision.producer or _beside(records),
        texts=revision.texts,
        facts=tuple(_facts(session, event, revision.shape, revision.replaced)),
        records=records,
        recorded=revision.size,
    )


def _beside(records: Sequence[ProducerRecord]) -> str | None:
    """The types of the records beside a change, each type one time. `None` when
    no record is beside it."""
    return ", ".join(dict.fromkeys(record.type for record in records)) or None


SOURCES: tuple[DiffSource, ...] = (revisions,)
"""Every source of diff groups. A new kind of change is a function added here."""


def _whole(session: Session, node: int) -> tuple[str, str]:
    """The whole history of a node: the conversation that it stands for as first
    written, and the node now."""
    return (
        render_messages(originals(session, node)),
        render_messages(messages_of(session, (node,))),
    )


def _summary(shape: RevisionShape, event: SessionEvent, shadowed: tuple[int, ...]) -> str:
    """A line for a revision whose producer gave none."""
    if shape == "rewrite":
        return f"{event.type} rewritten in place"
    return f"stands for {count_of(len(shadowed), 'node')}"


def _facts(
    session: Session, event: SessionEvent, shape: RevisionShape, shadowed: tuple[int, ...]
) -> Iterable[str]:
    yield f"replaces seq {_seqs(shadowed)}"
    if shape == "rewrite":
        reveal = origin_of(session, event.seq)
        yield f"the transcript shows it on the row it rewrote (seq {reveal})"
    for seq in shadowed:
        replaced = session.at(seq)
        if replaced is not None and is_replacement_surface_event(replaced):
            yield f"seq {seq} was itself a revision; the entry before this has what it replaced"


def _described(record: SessionEvent) -> ProducerRecord:
    """A record as the screen lists it: the fields of its producer and, apart from
    them, the stamp of the revision door that says what the record revises."""
    return ProducerRecord(
        record.type, record.seq, tuple(fields_of(producer_data(record))), revises_of(record)
    )


def _seqs(seqs: Sequence[int]) -> str:
    """`12..14, 17`: seqs as runs."""
    runs: list[list[int]] = []
    for seq in seqs:
        if runs and seq == runs[-1][-1] + 1:
            runs[-1].append(seq)
        else:
            runs.append([seq])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}..{run[-1]}" for run in runs)
