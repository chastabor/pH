"""What changed, as entries a person can step through: the projection behind `/diff`.

Each entry is one change: a `before` and an `after`, the seq that shows it in the
transcript, and the records its producer wrote beside it. The screen
(`ph_app.tui.diff_screen`) draws whatever this list holds, so it knows nothing about
where an entry came from.

**Meant to grow more kinds.** Context revisions are the one source today. A workspace
file change or a profile diff would be one more `DiffKind` member and one more source
function in `SOURCES`, beside `revisions`. Neither the screen nor the entries the other
sources produce would change. A tuple, not a registry: the sources are this package's
own, and a seam with a single member is one nobody can check (`ph.seams.tui_screens`
says the same about its one slot).

**A revision is any surface replacement, whoever wrote it.** That covers compaction's
summary, its argument elision and overflow clip, input-offload's paste preview, and a
model's own edit through ph-clm. All of them go through `ph.session.revise`, so one
reading of the log covers every one, and the app may not import a bundle to learn its
types (`test_app_layering.py`). Each one is classified from the event alone:

* a **substitution** stands for a run of nodes with text of its own, and a
  **stand-in** is the substitution that claims the run left the surface as
  condensed conversation (`is_stand_in`): a summary, or a model's revision;
* a **rewrite** replaces one node with a near-copy of itself
  (`is_in_place_rewrite`).

**Before is what the change replaced, not what was first written.** A revision of a
revision is diffed against the revision it replaced, so stepping through the list shows
each edit once. The conversation as first written is the before of the earliest entry
in that chain, and it is what `originals` reads back and what ph-clm's `context_diff`
compares against. Both sides render through `render_messages`, the rendering they
share, so a revision reads the same here as in that tool.

**Producer records are shown, not interpreted.** The records that landed in the
replacement's batch (`clm/revised`, `compaction/summarized`, `offload/input-spilled`,
or a type this process has never heard of) are listed field by field, the way the
trajectory describes a type it has no phrase for (`wire.fields_of`). The one field
convention read here is a `<unit>Before`/`<unit>After` pair of integers, which becomes
the entry's recorded size.

**Nothing is rendered until a row is shown, and nothing rendered is kept.** A summary's
before is every node it shadowed, which can be most of the conversation, so building the
list renders no text: an entry holds how to render its two sides (`texts`), and the
screen renders the row it shows and lets the strings go when the row changes. An entry
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

import difflib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias, TypeGuard

from ph.json import JsonValue, as_obj, as_str
from ph.session import (
    Session,
    SessionEvent,
    is_in_place_rewrite,
    is_replacement_surface_event,
    is_stand_in,
    is_surface_event,
    messages_of,
    origin_of,
    render_messages,
    shadowed_by,
)
from ph.text import count_of

from ..wire import fields_of, source_of

__all__ = [
    "SOURCES",
    "DiffEntry",
    "DiffKind",
    "DiffSource",
    "ProducerRecord",
    "RevisionShape",
    "Size",
    "build_diffs",
    "revisions",
    "unified",
]

DiffKind: TypeAlias = Literal["revision"]
"""Where an entry came from. The screen shows it as a column, so a person reading a
mixed list can tell the kinds apart."""

RevisionShape: TypeAlias = Literal["substitution", "stand-in", "rewrite"]
"""How a revision changed the surface. The module docstring says how each is told."""


@dataclass(frozen=True, slots=True)
class ProducerRecord:
    """A record that landed beside a change, field by field and uninterpreted."""

    type: str
    seq: int
    fields: tuple[tuple[str, str], ...]
    """`wire.fields_of`, in the payload's own order."""
    shared: bool = False
    """The record belongs to the whole batch rather than to this change alone: it is
    one record for several replacements, as compaction writes one for every argument
    it elided. Its numbers describe all of them, so none of them is this entry's
    size."""


@dataclass(frozen=True, slots=True)
class Size:
    """A change's size before and after, in `unit`."""

    before: int
    after: int
    unit: str

    def label(self) -> str:
        return f"{self.before:,}→{self.after:,} {self.unit}"


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One change: what it was, what it is, and where the transcript shows it."""

    kind: DiffKind
    seq: int
    """The log seq the change landed at. Entries are ordered by it, and a screen
    opened at a transcript position selects the nearest entry at or before it."""
    reveal: int
    """The seq of the transcript row that shows the change, which is not always
    `seq`: a rewrite has no row of its own and shows on the row it rewrote."""
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
    recorded: Size | None = None
    """The size a producer record gave, when one did."""


DiffSource: TypeAlias = Callable[[Session], list[DiffEntry]]
"""`source(session) -> entries`, in log order. One per `DiffKind`."""


def build_diffs(session: Session) -> list[DiffEntry]:
    """Every source's entries, ordered by seq."""
    return sorted(
        (entry for source in SOURCES for entry in source(session)), key=lambda one: one.seq
    )


def unified(before: str, after: str) -> tuple[str, ...]:
    """The unified diff of one entry's two sides, every line of it. A reader that draws
    it bounds it."""
    return tuple(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), "before", "after", lineterm=""
        )
    )


# ---------------------------------------------------------------- revisions --


_Landed: TypeAlias = tuple[tuple[ProducerRecord, ...], Size | None]
"""What one replacement's batch says about it: its records, and the size one gave."""


def revisions(session: Session) -> list[DiffEntry]:
    """One entry per surface replacement in the log, in log order."""
    log = session.events
    batches: dict[int, dict[int, _Landed]] = {}
    entries: list[DiffEntry] = []
    for event in log:
        if not is_replacement_surface_event(event):
            continue
        shadowed = shadowed_by(event)
        shape: RevisionShape = (
            "rewrite"
            if is_in_place_rewrite(event)
            else "stand-in"
            if is_stand_in(event)
            else "substitution"
        )
        records, recorded = _landed(log, event, batches)
        reveal = origin_of(session, event.seq) if shape == "rewrite" else event.seq
        kind, plugin, _ = source_of(event.data)
        stated = as_str(as_obj(event.data.get("source")).get("summary"))
        entries.append(
            DiffEntry(
                kind="revision",
                seq=event.seq,
                reveal=reveal,
                title=shape,
                summary=stated or _summary(shape, event, shadowed),
                producer=(plugin if kind == "plugin" else "")
                or ", ".join(dict.fromkeys(record.type for record in records)),
                texts=partial(_texts, session, shadowed, event.seq),
                facts=tuple(_facts(session, event, shadowed, reveal)),
                records=records,
                recorded=recorded,
            )
        )
    return entries


SOURCES: tuple[DiffSource, ...] = (revisions,)
"""Every source of diff entries. A new kind of change is a function added here."""


def _texts(session: Session, shadowed: tuple[int, ...], seq: int) -> tuple[str, str]:
    """A revision's two sides: what it replaced, and itself."""
    return (
        render_messages(messages_of(session, shadowed)),
        render_messages(messages_of(session, (seq,))),
    )


def _summary(shape: RevisionShape, event: SessionEvent, shadowed: tuple[int, ...]) -> str:
    """A line for a revision whose producer gave none."""
    if shape == "rewrite":
        return f"{event.type} rewritten in place"
    return f"stands for {count_of(len(shadowed), 'node')}"


def _facts(
    session: Session, event: SessionEvent, shadowed: tuple[int, ...], reveal: int
) -> Iterable[str]:
    yield f"replaces seq {_seqs(shadowed)}"
    if reveal != event.seq:
        yield f"the transcript shows it on the row it rewrote (seq {reveal})"
    for seq in shadowed:
        replaced = session.at(seq)
        if replaced is not None and is_replacement_surface_event(replaced):
            yield f"seq {seq} was itself a revision; its own entry has what it replaced"


def _landed(
    log: Sequence[SessionEvent], event: SessionEvent, batches: dict[int, dict[int, _Landed]]
) -> _Landed:
    """What `event`'s batch says about it, the batch read once for all its members."""
    batch = event.batch
    if batch is None:
        return (), None
    if batch.first not in batches:
        # `seq` is the index (A1), so the batch is a slice.
        batches[batch.first] = _batch(log[batch.first : batch.last + 1])
    return batches[batch.first][event.seq]


def _batch(members: Sequence[SessionEvent]) -> dict[int, _Landed]:
    """Each replacement in a batch, with the records that landed with it.

    A batch holding one replacement owns every record in it. A batch holding several
    pairs them by type and order: a type with one record per replacement gives each
    replacement its own (ph-clm writes each edit's record after it). Any other record
    is the batch's, shown on each replacement and sized for none of them (compaction's
    one record for every argument it elided). One pass, so a batch of a thousand
    elisions costs a thousand and not a million.
    """
    replacements = [one.seq for one in members if is_replacement_surface_event(one)]
    records = [one for one in members if not is_surface_event(one)]
    several = len(replacements) > 1
    counts = Counter(record.type for record in records)
    taken: Counter[str] = Counter()
    shown: dict[int, list[ProducerRecord]] = {seq: [] for seq in replacements}
    sizes: dict[int, list[Size]] = {seq: [] for seq in replacements}
    for record in records:
        shared = several and counts[record.type] != len(replacements)
        described = ProducerRecord(record.type, record.seq, tuple(fields_of(record.data)), shared)
        if shared:
            for seq in replacements:
                shown[seq].append(described)
            continue
        owner = replacements[taken[record.type]] if several else replacements[0]
        taken[record.type] += 1
        shown[owner].append(described)
        sizes[owner].extend(_pairs(record.data))
    # One pair across the records: two would be two claims about one change, and
    # choosing between them is interpreting a producer.
    return {
        seq: (tuple(shown[seq]), sizes[seq][0] if len(sizes[seq]) == 1 else None)
        for seq in replacements
    }


def _pairs(data: Mapping[str, JsonValue]) -> Iterable[Size]:
    """Each `<unit>Before`/`<unit>After` pair of integers in a record."""
    for key, before in data.items():
        unit = key.removesuffix("Before")
        if not unit or unit == key:
            continue
        after = data.get(f"{unit}After")
        if _integer(before) and _integer(after):
            yield Size(before, after, unit)


def _integer(value: JsonValue) -> TypeGuard[int]:
    """An integer and not a boolean, which Python counts as one."""
    return isinstance(value, int) and not isinstance(value, bool)


def _seqs(seqs: Sequence[int]) -> str:
    """`12..14, 17`: seqs as runs."""
    runs: list[list[int]] = []
    for seq in seqs:
        if runs and seq == runs[-1][-1] + 1:
            runs[-1].append(seq)
        else:
            runs.append([seq])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}..{run[-1]}" for run in runs)
