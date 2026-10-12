"""What changed, grouped as git groups commits by file: the projection behind `/diff`.

Each group is one thing as it is now, and every change that led to it, oldest first:
what `git log -p -- file` shows for a file. Each entry is one change: a `before` and an
`after`, the seq that shows it in the transcript, and the records its producer wrote
for it. A group also holds its whole history in one diff, the thing as first written
against the thing now, as `git diff A..B -- file` does. The screen
(`ph_app.tui.diff_screen`) draws whatever this list holds, so it knows nothing about
where a group came from.

**Meant to grow more kinds.** Context revisions are the one source today. A workspace
file change or a profile diff would be one more `DiffKind` member and one more source
function in `SOURCES`, beside `revisions`. Neither the screen nor the groups the other
sources produce would change. A tuple, not a registry: the sources are this package's
own, and a seam with a single member is one nobody can check (`ph.seams.tui_screens`
says the same about its one slot).

**A revision is any surface replacement, whoever wrote it.** That covers compaction's
summary, its argument elision and overflow clip, input-offload's paste preview, and a
model's own edit through ph-clm. All of them go through `ph.session.revise`, so one
reading of the log covers every one, and the app may not import a bundle to learn its
types (`test_app_layering.py`). Each one is classified from the event alone, as a
substitution, a stand-in or a rewrite (`ph_app.revisions.shape_of`, which the
training export shares).

**A revision's group is the surface node it led to** (`Session.history`). Every
revision ends up in exactly one: a node is shadowed once, so the revisions behind
today's surface form a tree per node. The history merges, unlike git's one line per
file: a summary over two sections a model had edited holds both edits and itself.
Grouping by ph-clm's sections instead would need the app to read a bundle's map, which
it may not (`test_app_layering.py`), and a section is a run of these nodes anyway.

**An entry's before is what the change replaced, not what was first written.** A
revision of a revision is diffed against the revision it replaced, so stepping through
a group shows each edit once. The group's whole history is the conversation as first
written (`originals`) against the node now, which is what ph-clm's `context_diff`
compares too. Both sides render through `render_messages`, and both diffs are
`unified_diff`, so a revision reads the same here as in that tool.

**Producer records are shown, not interpreted.** The records its producer wrote
through the revision door, which stamps each with the replacements it explains
(`Session.records_of`): `clm/revised`, `compaction/summarized`, `offload/input-spilled`,
or a type this process has never heard of. Each is listed field by field, the way the
trajectory describes a type it has no phrase for (`wire.fields_of`), apart from what the
door wrote on it: what it revises, and its `size`, which becomes the entry's recorded
size when the record explains this change alone. A replacement written before the door
stamped its records has none.

**Nothing is rendered until a row is shown, and nothing rendered is kept.** A summary's
before is every node it shadowed, which can be most of the conversation, so building the
list renders no text: an entry holds how to render its two sides (`texts`), a group its
whole history (`whole`), and the screen renders the row it shows and lets the strings go
when the row changes. An entry
with no recorded size has its characters counted then. No tokenizer runs: the client
has no meter for the session, and a count that disagreed with the producer's own would
be worse than characters labeled as characters.

**A limit, recorded rather than fixed.** The transcript draws nothing for an in-place
rewrite of an assistant message, because the text is already on screen
(`TuiEventAdapter._on_assistant_message`). So this list is the only place a person sees
one.

**Textual-free**, like `trajectory`. It takes a `Session`, which a stored log becomes
through `ph.persistence.read_session` with nothing mounted, so a viewer over a file can
reuse it.

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
"""Where an entry came from. The screen shows it as a column, so a person reading a
mixed list can tell the kinds apart."""


@dataclass(frozen=True, slots=True)
class ProducerRecord:
    """A record that landed beside a change, field by field and uninterpreted."""

    type: str
    seq: int
    fields: tuple[tuple[str, str], ...]
    """`wire.fields_of`, in the payload's own order."""
    revises: tuple[int, ...] = ()
    """The replacements the record explains. More than this one when it is one record
    for several, as compaction writes one for every argument it elided: its numbers
    describe all of them, so none of them is this entry's size."""


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One change: what it was, and what it is."""

    seq: int
    """The log seq the change landed at; a group's entries are ordered by it."""
    title: str
    summary: str
    producer: str
    """Who made the change, as the log says: a plugin's name, or the types of the
    records beside it. Empty when the log does not say."""
    texts: Callable[[], tuple[str, str]] = field(compare=False, repr=False)
    """`(before, after)`, rendered on each call and kept by nobody here."""
    facts: tuple[str, ...] = ()
    """Lines of the source's own about this change, shown above the diff."""
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
    """Oldest first; the last is the change that made it what it is now."""
    whole: Callable[[], tuple[str, str]] = field(compare=False, repr=False)
    """`(as first written, now)`, rendered on each call and kept by nobody here."""

    @property
    def latest(self) -> DiffEntry:
        return self.entries[-1]

    @property
    def producers(self) -> str:
        """Who made its changes, each once, oldest first."""
        return ", ".join(dict.fromkeys(entry.producer for entry in self.entries if entry.producer))


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
        producer=revision.producer or ", ".join(dict.fromkeys(record.type for record in records)),
        texts=revision.texts,
        facts=tuple(_facts(session, event, revision.shape, revision.replaced)),
        records=records,
        recorded=revision.size,
    )


SOURCES: tuple[DiffSource, ...] = (revisions,)
"""Every source of diff groups. A new kind of change is a function added here."""


def _whole(session: Session, node: int) -> tuple[str, str]:
    """A node's whole history: the conversation it stands for as first written, and
    the node now."""
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
    """A record as the screen lists it: its producer's fields, and apart from them the
    door's stamp of what it revises."""
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
