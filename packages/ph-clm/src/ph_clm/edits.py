"""The one entry point that every context edit goes through.

There are four verbs. Each verb writes one surface `replace` and one
`clm/revised` record. Both are committed in one `Session.batch()`, so the record
never lands without the edit that it describes.

* **tombstone**: a run of whole sections becomes a one-line marker that says what
  was removed and why. The marker is visible on purpose
  (`plans/Ph_Clm_Context_Editing_Plan.md`, decision 1), so that the model still
  knows that work happened there. A tombstone next to an earlier tombstone
  absorbs it, so a run of deletions reads as one marker.
* **replace**: a run of whole sections becomes text that the model wrote, either
  a summary or a rewrite of the span. The message has two text blocks. The first
  is the label of the harness, which says what the text stands for. The second
  is the text of the model. A front end that shows only what the model may edit
  can then separate them without reading the words (`labeled`).
* **rewrite**: one passage inside one tool result or one assistant reply changes
  in place. The call ids and the message id stay the same, so the pairing and
  the row do not change.
* **restore**: the latest of these edits in one section is undone, one level. A
  rewrite goes back to the exact text before it. A tombstone or a replace comes
  back as the text of what it stood for, in one message under a label (option A
  of decision 3). An exact restore of a substitution needs a surface op that
  brings shadowed nodes back. That is a format change, and the plan defers it
  until sessions show that it is wanted. A restore is also an edit, so restoring
  a restore redoes what the restore undid.

A substitution (a tombstone or a replace) is a `user/message` from
`PluginSource(plugin="clm", form="compaction")`. That form states "this stands in
for conversation that has left the surface" (`is_stand_in`). The TUI draws such a
message as a revision, and compaction recognizes it as a summary that it does not
need to summarize again. The role is user, never assistant (decision 2), because
text that stands for several turns does not belong to the speaker of one turn.

**Whole sections, one unbroken run, nothing protected.** `sections.resolve_run`
enforces these rules. A replacement lands where its earliest node was, and a run
that split a call from its result would give the provider an orphan. Core
enforces the rest. Every write goes through the revision functions of core
(`ph.session.revise`). They keep the message id of a rewrite, and they leave the
usage and the step of an assistant reply with the original. The surface refuses a
node that is no longer current, and it refuses a `tool/result` rewrite that
changes more than content.

**The gate is off unless a row asks for it** (decision 4). `fit` refuses growth
past the window, and `shrink` refuses any growth. The limits of ph ship unset,
for the reason that the `limits` row gives. Compaction is the backstop in either
case.

@module ph_clm.edits
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, assert_never, cast, get_args

from ph.json import JsonObject, as_int, as_obj, as_seq, as_str
from ph.llm.types import (
    ContentBlock,
    ContextForm,
    MediaBlock,
    Message,
    PluginSource,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    create_user_message,
    text_of,
)
from ph.seams.token_meter import TokenBaseline
from ph.session import (
    RevisionRecord,
    RevisionSize,
    Session,
    SessionBatch,
    SessionEvent,
    editable_message,
    is_in_place_rewrite,
    messages_of,
    render_messages,
    revises_of,
    rewrite,
    shadowed_by,
    substitute,
)
from ph.session.writers import log_writer
from ph.text import NO_OUTPUT, thousands

from .kinds import REVISED
from .sections import EditRefused, Section, SectionMap, label, resolve_one, resolve_run, span_label

_LOG = log_writer(__name__)

__all__ = [
    "Editor",
    "Gate",
    "Pending",
    "Removed",
    "Revision",
    "Verb",
    "Via",
    "labeled",
    "node_text",
    "receipt",
    "revision_of",
]

PLUGIN = "clm"
"""The plugin name that the `PluginSource` of a substitution carries."""

Verb = Literal["tombstone", "replace", "rewrite", "restore"]
Gate = Literal["none", "fit", "shrink"]
Via = Literal["tool", "mirror"]
"""The front end that an edit came through: a context tool, or the context file."""

_UNLANDED = -1
"""The `replacement` of a draft revision before `Editor.land` knows the seq."""


@dataclass(frozen=True, slots=True)
class Removed:
    """One span that a tombstone removed, and the reason that was given for it."""

    first: int
    last: int
    reason: str

    def describe(self) -> str:
        span = span_label(self.first, self.last)
        return f"{span} ({self.reason})" if self.reason else span


@dataclass(frozen=True, slots=True)
class Revision:
    """What one edit did, as its `clm/revised` record says it and as its tool
    reports it."""

    verb: Verb
    first: int
    last: int
    replacement: int
    shadowed: tuple[int, ...]
    tokens_before: int
    tokens_after: int
    reread: int
    """The tokens after the edit that a provider reads again, because its prefix
    cache stopped at the first change."""
    context_before: int = 0
    """The size of the context before the batch in which this edit landed, as
    `TokenMeter.baseline` counts it: the last count of the provider, adjusted by
    what changed since. A receipt and a readout therefore give the same number.
    `Editor.land` sets this field, as it sets `call_id`."""
    call_id: str | None = None
    via: Via = "tool"
    removed: tuple[Removed, ...] = ()
    """For a tombstone: every span that its marker stands for, with the reason
    that was given."""
    in_place: bool = False
    """Whether the edit kept the id of its section. That is true for a rewrite and
    for a restore of a rewrite. It is false for an edit that landed a replacement
    with its own id. This field is not recorded. The replacement event shows it
    (`is_in_place_rewrite`), and a reader of the log asks the event."""

    @property
    def context_after(self) -> int:
        return self.context_before - self.tokens_before + self.tokens_after

    def size(self) -> RevisionSize:
        """What the edit replaced and what it put there, in tokens. The revision
        function writes this on the record as its `size`."""
        return RevisionSize("tokens", self.tokens_before, self.tokens_after)

    def record(self) -> JsonObject:
        return {
            "verb": self.verb,
            "via": self.via,
            "callId": self.call_id,
            "first": self.first,
            "last": self.last,
            "shadowed": list(self.shadowed),
            "reread": self.reread,
            "contextBefore": self.context_before,
            "removed": [
                {"first": one.first, "last": one.last, "reason": one.reason} for one in self.removed
            ],
        }


def revision_of(event: SessionEvent | None) -> Revision | None:
    """Read a `clm/revised` record back. Returns `None` for any other event. The
    replacement and the size are the values that the revision function stamped on
    the record (`revises`, `size`)."""
    if event is None or event.type != REVISED:
        return None
    data = event.data
    verb = as_str(data.get("verb"))
    revised = revises_of(event)
    size = RevisionSize.of(event)
    if verb not in get_args(Verb) or len(revised) != 1 or size is None:
        return None
    via: Via = "mirror" if as_str(data.get("via")) == "mirror" else "tool"
    return Revision(
        verb=cast("Verb", verb),
        first=as_int(data.get("first")),
        last=as_int(data.get("last")),
        replacement=revised[0],
        shadowed=tuple(as_int(seq) for seq in as_seq(data.get("shadowed"))),
        tokens_before=size.before,
        tokens_after=size.after,
        reread=as_int(data.get("reread")),
        context_before=as_int(data.get("contextBefore")),
        call_id=as_str(data.get("callId")) or None,
        via=via,
        removed=tuple(
            Removed(
                first=as_int(as_obj(one).get("first")),
                last=as_int(as_obj(one).get("last")),
                reason=as_str(as_obj(one).get("reason")),
            )
            for one in as_seq(data.get("removed"))
        ),
    )


def receipt(revisions: Sequence[Revision]) -> str:
    """Describe what some edits did and what they cost. This is the one wording,
    used for the result of a tool call and for a write of the context file."""
    done = []
    for revision in revisions:
        span = span_label(revision.first, revision.last)
        match revision.verb:
            case "tombstone":
                done.append(f"removed {span}, now {label(revision.replacement)}")
            case "replace":
                done.append(f"replaced {span}, now {label(revision.replacement)}")
            case "rewrite":
                done.append(f"rewrote a passage in {span}")
            case "restore" if revision.in_place:
                done.append(f"restored {span} to its text before its last edit")
            case "restore":
                done.append(
                    f"restored what {span} stood for, now {label(revision.replacement)}; "
                    "restore that to undo this"
                )
    before = revisions[0].context_before
    after = before + sum(one.tokens_after - one.tokens_before for one in revisions)
    reread = max(one.reread for one in revisions)
    said = "; ".join(done)
    lines = [
        f"{said[:1].upper()}{said[1:]}. Context ~{thousands(before)} → ~{thousands(after)} tokens.",
        f"The ~{thousands(reread)} tokens after it are read once more on the next request: "
        "a prefix cache cannot serve past an edit.",
    ]
    if after > before and any(one.verb != "restore" for one in revisions):
        # A restore grows the context by design. The warning is for a condensing
        # edit that kept the old text beside the new text.
        lines.append(
            "This GREW the context. If you meant to condense, you may have kept the old text "
            "as well as the new."
        )
    return "\n".join(lines)


def node_text(message: Message) -> str:
    """The text of one node that an edit reads and replaces: the words of a reply
    or of a user message, or the output of a result. This is the read half of
    `_with_text`. Calls, reasoning, and media are not text that an edit changes."""
    return "\n".join(text for text in map(_block_text, message.content) if text is not None)


def labeled(message: Message) -> tuple[str, str] | None:
    """Split a ph-clm substitution into its label and the text that the model may
    edit. Returns `None` for any other message, including a compaction summary.

    The label is the first text block. The harness wrote it
    (`Editor._substituting` writes it first), and it is returned without its
    brackets. The rest is the text of the model. A tombstone is all label. The
    function reads only the message, so a fold of the log can call it for each
    node as the node lands.
    """
    source = message.source
    if not (isinstance(source, PluginSource) and source.plugin == PLUGIN):
        return None
    texts = [block.text for block in message.content if isinstance(block, TextBlock)]
    if not texts:
        return None
    return texts[0].removeprefix("[").removesuffix("]"), "\n".join(texts[1:])


@dataclass(frozen=True, slots=True)
class Pending:
    """One edit that is validated and drafted but not yet written. `Editor.land`
    writes several in one batch, so a front end that makes many edits at one time
    lands all of them or none."""

    write: Callable[[SessionBatch, RevisionRecord], SessionEvent]
    """Lands the edit and its record. The revision function writes the record
    after the edit."""
    draft: Revision


@dataclass(frozen=True, slots=True)
class Editor:
    """The four verbs, with the section map and the gate of one row.

    Each verb has a draft method (`tombstoning`, `tombstoning_runs`, `replacing`,
    `rewriting_text`, `restoring`) that validates and measures without writing.
    `land` then applies the gate and writes. The tools land one draft for each
    call. The context file compiles one write into several drafts and lands them
    together. A draft method takes sections that the caller has already resolved.
    The caller is responsible for protection and for an unbroken run.
    """

    sections: SectionMap
    gate: Gate

    # ------------------------------------------------------- one edit, landed --

    def tombstone(
        self, session: Session, names: Sequence[str], reason: str, *, call_id: str | None
    ) -> Revision:
        """Replace a run of whole sections with a one-line marker."""
        sections = self.sections(session)
        start, stop = resolve_run(session, sections, names)
        draft = self.tombstoning(session, sections, start, stop, reason)
        return self.land(session, [draft], call_id=call_id)[0]

    def replace(
        self, session: Session, names: Sequence[str], text: str, *, call_id: str | None
    ) -> Revision:
        """Replace a run of whole sections with text that the model wrote."""
        sections = self.sections(session)
        start, stop = resolve_run(session, sections, names)
        return self.land(session, [self.replacing(sections, start, stop, text)], call_id=call_id)[0]

    def rewrite(
        self, session: Session, name: str, old: str, new: str, *, call_id: str | None
    ) -> Revision:
        """Change one passage inside the tool output or the reply of one section,
        in place."""
        if not old:
            raise EditRefused("say which text to change: `old` is empty")
        if old == new:
            raise EditRefused("`old` and `new` are the same text")
        sections = self.sections(session)
        section = resolve_one(session, sections, name, editing=True)
        sites = [site for seq in section.nodes for site in _text_sites(session, seq)]
        count = sum(text.count(old) for _, _, text in sites)
        if count == 0:
            raise EditRefused(
                f"that text does not occur in the tool output or replies of {section.name}"
            )
        if count > 1:
            raise EditRefused(
                f"that text occurs {count} times in {section.name}; include more of it so "
                "it names one place"
            )
        seq, path = next((seq, path) for seq, path, text in sites if old in text)
        event = _at(session, seq)
        message = _rewritten(event, path, old, new)
        draft = self._rewriting(session, section, event, message)
        return self.land(session, [draft], call_id=call_id)[0]

    def restore(self, session: Session, name: str, *, call_id: str | None) -> Revision:
        """Undo the latest edit of one section, one level."""
        sections = self.sections(session)
        section = resolve_one(session, sections, name, editing=True)
        return self.land(session, [self.restoring(session, sections, section)], call_id=call_id)[0]

    # ----------------------------------------------------------------- drafts --

    def tombstoning(
        self,
        session: Session,
        sections: Sequence[Section],
        start: int,
        stop: int,
        reason: str,
        *,
        keep: frozenset[int] = frozenset(),
    ) -> Pending:
        """Draft the marker for `sections[start:stop]`.

        A tombstone beside an earlier tombstone absorbs it, so a run of deletions
        reads as one marker. The exception is a section in `keep`, which another
        draft in the same batch already changes.
        """
        if start > 0 and _absorbed(session, sections[start - 1], keep):
            start -= 1
        if stop < len(sections) and _absorbed(session, sections[stop], keep):
            stop += 1
        removed = tuple(_removals(session, sections[start:stop], reason.strip()))
        text = "removed from context: " + "; ".join(one.describe() for one in removed)
        return self._substituting(
            sections, start, stop, _note((f"[{text}]",), text), verb="tombstone", removed=removed
        )

    def tombstoning_runs(
        self,
        session: Session,
        sections: Sequence[Section],
        positions: Sequence[int],
        reason: str,
        *,
        keep: frozenset[int],
    ) -> list[Pending]:
        """Draft one marker for each run of `positions`, in a batch with other
        drafts.

        Two runs with only an earlier, untouched marker between them count as one
        run. Without this rule, each of the two new markers would absorb that
        earlier marker, and the absorption rule of `tombstoning` would apply twice
        to one node.
        """
        runs: list[list[int]] = []
        for position in sorted(positions):
            if runs and position == runs[-1][-1] + 1:
                runs[-1].append(position)
            elif (
                runs
                and position == runs[-1][-1] + 2
                and _absorbed(session, sections[position - 1], keep)
            ):
                runs[-1] += [position - 1, position]
            else:
                runs.append([position])
        return [
            self.tombstoning(session, sections, run[0], run[-1] + 1, reason, keep=keep)
            for run in runs
        ]

    def replacing(self, sections: Sequence[Section], start: int, stop: int, text: str) -> Pending:
        """Draft text that the model wrote, to stand in for `sections[start:stop]`."""
        body = text.strip()
        if not body:
            raise EditRefused("a replacement needs text; to drop the sections, tombstone them")
        span = span_label(sections[start].id, sections[stop - 1].id)
        return self._substituting(
            sections,
            start,
            stop,
            _note((f"[revised context, standing for {span}]", body), f"revised {span}"),
            verb="replace",
        )

    def restoring(self, session: Session, sections: Sequence[Section], section: Section) -> Pending:
        """Draft the undo, one level, of the latest edit of this row in `section`.

        A rewrite goes back to the exact text that it replaced. The message is the
        same message, so the checks of the revision function hold. A substitution
        comes back as what it shadowed, read one level down and in surface order
        (its record keeps that order). When that was one user-role message, the
        result is a copy of it. This is the case for another marker, or for a
        restore that is being undone. Otherwise the result is the text of those
        messages in one message, under a label. The form of that message is
        `recall`, not `compaction`. It is the conversation itself, not a summary
        of it, so compaction may summarize it again.
        """
        edits = [
            (seq, revision)
            for seq in section.nodes
            if (revision := _edit_at(session, seq)) is not None
        ]
        if not edits:
            if section.stands_for:
                raise EditRefused(
                    f"{section.name} was revised by the harness (a compaction summary, or a "
                    "paste moved to a file), not by a context edit; context_recall reads "
                    "what it stands for"
                )
            raise EditRefused(
                f"{section.name} has not been revised; it is in your context as it was written"
            )
        seq, revision = max(edits, key=lambda edit: edit[0])
        event = _at(session, seq)
        if is_in_place_rewrite(event):
            before = editable_message(_at(session, shadowed_by(event)[0]))
            return self._rewriting(session, section, event, before, verb="restore")
        stood = list(messages_of(session, revision.shadowed))
        position = sections.index(section)
        if len(stood) == 1 and stood[0].role == "user":
            again = create_user_message(content=list(stood[0].content), source=stood[0].source)
            return self._substituting(sections, position, position + 1, again, verb="restore")
        span = span_label(revision.first, revision.last)
        message = _note(
            (f"[restored context, standing for {span}]", render_messages(stood)),
            f"restored {span}",
            form="recall",
        )
        return self._substituting(sections, position, position + 1, message, verb="restore")

    def rewriting_text(self, session: Session, section: Section, seq: int, text: str) -> Pending:
        """Draft an in-place change of every text in node `seq` of `section` to
        `text`. A reply keeps its calls. A result keeps its call id and everything
        that is not text."""
        event = _at(session, seq)
        return self._rewriting(session, section, event, _with_text(event, text))

    def land(
        self,
        session: Session,
        drafts: Sequence[Pending],
        *,
        call_id: str | None,
        via: Via = "tool",
    ) -> list[Revision]:
        """Apply the gate to every draft, then write all of them in one batch,
        each with its record. One call lands them, through one front end."""
        baseline = self.sections.meter.baseline(session)
        # Gate the batch as one change. Two drafts that each fit can fail to fit
        # together, and the receipt reports them together.
        self._admit(
            baseline,
            sum(draft.draft.tokens_before for draft in drafts),
            sum(draft.draft.tokens_after for draft in drafts),
        )
        landed: list[Revision] = []
        with session.batch() as batch:
            for draft in drafts:
                revision = dataclasses.replace(
                    draft.draft, call_id=call_id, via=via, context_before=baseline.tokens
                )
                record = RevisionRecord(_LOG, REVISED, revision.record(), revision.size())
                replacement = draft.write(batch, record)
                landed.append(dataclasses.replace(revision, replacement=replacement.seq))
        return landed

    # ------------------------------------------------------------------ inner --

    def _substituting(
        self,
        sections: Sequence[Section],
        start: int,
        stop: int,
        message: Message,
        *,
        verb: Verb,
        removed: tuple[Removed, ...] = (),
    ) -> Pending:
        """Draft `message`, a user message, to stand for `sections[start:stop]`."""
        span = sections[start:stop]
        shadowed = tuple(seq for section in span for seq in section.nodes)
        return Pending(
            lambda batch, record: substitute(batch, shadowed, message, record),
            Revision(
                verb=verb,
                first=span[0].id,
                last=span[-1].id,
                replacement=_UNLANDED,
                shadowed=shadowed,
                tokens_before=sum(section.tokens for section in span),
                tokens_after=self.sections.meter.measure(message),
                reread=span[-1].after,
                removed=removed,
            ),
        )

    def _rewriting(
        self,
        session: Session,
        section: Section,
        event: SessionEvent,
        message: dict[str, Any],
        *,
        verb: Verb = "rewrite",
    ) -> Pending:
        meter = self.sections.meter
        later = section.nodes[section.nodes.index(event.seq) + 1 :]
        return Pending(
            lambda batch, record: rewrite(batch, event, message, record),
            Revision(
                verb=verb,
                in_place=True,
                first=section.id,
                last=section.id,
                replacement=_UNLANDED,
                shadowed=(event.seq,),
                tokens_before=meter.node_tokens(session, event.seq),
                tokens_after=meter.measure(Message.model_validate(message)),
                reread=section.after + sum(meter.node_tokens(session, seq) for seq in later),
            ),
        )

    def _admit(self, baseline: TokenBaseline, before: int, after: int) -> None:
        """Refuse growth that the gate of the row does not allow. `none` allows
        everything."""
        if self.gate == "none" or after <= before:
            return
        if self.gate == "shrink":
            raise EditRefused(
                f"this edit grows the context (~{before} → ~{after} tokens), and the "
                "clm-context row's gate is `shrink`: replace stale text with something shorter"
            )
        window = baseline.context_window
        if window is None or baseline.tokens - before + after > window:
            # With no window there is no "fits" to test, so growth stays refused.
            # The harness of the paper uses the same rule for its own `fit` gate.
            limit = "an unknown window" if window is None else f"the {window}-token window"
            raise EditRefused(
                f"this edit grows the context (~{before} → ~{after} tokens) past {limit}, "
                "and the clm-context row's gate is `fit`"
            )


# ------------------------------------------------------------------ helpers --


def _at(session: Session, seq: int) -> SessionEvent:
    event = session.at(seq)
    assert event is not None, f"seq {seq} came off this session's surface"
    return event


def _note(texts: Sequence[str], summary: str, *, form: ContextForm = "compaction") -> Message:
    """Build a substitution of this row: the label, then any text of the model,
    each in its own block (`labeled`)."""
    return create_user_message(
        content=[{"type": "text", "text": text} for text in texts],
        source=PluginSource(plugin=PLUGIN, form=form, summary=summary),
    )


def _edit_at(session: Session, seq: int) -> Revision | None:
    """The record of the edit of this row that put node `seq` on the surface, or
    `None` when no such edit did. The revision function stamps each record with
    the replacement that it explains."""
    return next(filter(None, map(revision_of, session.records_of(seq))), None)


def _removed_by(session: Session, section: Section) -> tuple[Removed, ...] | None:
    """What an earlier tombstone removed, when `section` is the marker of that
    tombstone."""
    if len(section.nodes) != 1:
        return None
    revision = _edit_at(session, section.nodes[0])
    return revision.removed if revision is not None and revision.verb == "tombstone" else None


def _absorbed(session: Session, neighbor: Section, keep: frozenset[int]) -> bool:
    """Whether a new tombstone absorbs `neighbor`: an earlier marker that no other
    draft in the batch changes."""
    return neighbor.id not in keep and _removed_by(session, neighbor) is not None


def _removals(session: Session, span: Sequence[Section], reason: str) -> Iterator[Removed]:
    """What a marker for `span` says that it removed. Each run of ordinary
    sections is one span with `reason`. Each earlier marker that it absorbs
    contributes what that marker said."""
    run: list[Section] = []
    for section in span:
        earlier = _removed_by(session, section)
        if earlier is None:
            run.append(section)
            continue
        if run:
            yield Removed(run[0].id, run[-1].id, reason)
            run = []
        yield from earlier
    if run:
        yield Removed(run[0].id, run[-1].id, reason)


def _block_text(block: ContentBlock) -> str | None:
    match block:
        case TextBlock():
            return block.text
        case ToolResultBlock():
            return text_of(block.content)
        case ToolCallBlock() | ReasoningBlock() | MediaBlock():
            return None
        case _ as unhandled:
            assert_never(unhandled)


def _with_text(event: SessionEvent, text: str) -> dict[str, Any]:
    """The message of `event` with all of its text replaced by `text`. Everything
    else is kept: the calls and the reasoning of a reply, and the call id and all
    non-text content of a result. This is the write half of `node_text`."""
    message = editable_message(event)
    blocks = cast("list[dict[str, Any]]", message["content"])
    if event.type == "assistant/message":
        message["content"] = _replaced_text(blocks, text)
    else:
        for index, block in enumerate(blocks):
            if block.get("type") == "tool-result":
                parts = cast("list[dict[str, Any]]", block.get("content") or [])
                blocks[index] = {**block, "content": _replaced_text(parts, text or NO_OUTPUT)}
    return message


def _replaced_text(blocks: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """`blocks` with every text block removed, and one block that holds `text` at
    the position of the first text block (at the front when there was none). When
    `text` is empty, the result has no text block."""
    first = next((i for i, block in enumerate(blocks) if block.get("type") == "text"), 0)
    kept = [block for block in blocks if block.get("type") != "text"]
    if not text:
        return kept
    return [*kept[:first], {"type": "text", "text": text}, *kept[first:]]


_Path = tuple[int, ...]
"""The position of a text in the payload of an event: a block index in an
assistant reply, or the index of a result block and the index of the text inside
it."""


def _text_sites(session: Session, seq: int) -> list[tuple[int, _Path, str]]:
    """The editable texts of one node: the text blocks of an assistant reply, or
    the text content of a tool result. Reasoning, calls, and media are not text
    that an edit changes."""
    event = _at(session, seq)
    blocks = as_seq(as_obj(event.data.get("message")).get("content"))
    sites: list[tuple[int, _Path, str]] = []
    if event.type == "assistant/message":
        for index, block in enumerate(blocks):
            entry = as_obj(block)
            if entry.get("type") == "text":
                sites.append((seq, (index,), as_str(entry.get("text"))))
    elif event.type == "tool/result":
        for index, block in enumerate(blocks):
            entry = as_obj(block)
            if entry.get("type") != "tool-result":
                continue
            for inner, part in enumerate(as_seq(entry.get("content"))):
                piece = as_obj(part)
                if piece.get("type") == "text":
                    sites.append((seq, (index, inner), as_str(piece.get("text"))))
    return sites


def _rewritten(event: SessionEvent, path: _Path, old: str, new: str) -> dict[str, Any]:
    """The message of `event` with the one passage changed. This is what
    `ph.session.revise.rewrite` lands in its place. Everything else that a rewrite
    must keep is kept."""
    message = editable_message(event)
    blocks = cast("list[dict[str, Any]]", message["content"])
    if event.type == "assistant/message":
        (index,) = path
        blocks[index] = {**blocks[index], "text": blocks[index]["text"].replace(old, new, 1)}
    else:
        index, inner = path
        parts = cast("list[dict[str, Any]]", blocks[index]["content"])
        parts[inner] = {**parts[inner], "text": parts[inner]["text"].replace(old, new, 1)}
    return message
