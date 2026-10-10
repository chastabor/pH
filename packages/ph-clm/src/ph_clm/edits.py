"""The one door every context edit goes through.

Three verbs, each one surface `replace` and one `clm/revised` record, committed in
one `Session.batch()` so the record never lands without the edit it describes:

* **tombstone** — a run of whole sections becomes a one-line marker saying what was
  removed and why. Visible on purpose (`plans/Ph_Clm_Context_Editing_Plan.md`,
  decision 1): the model keeps knowing that work happened there. A tombstone next
  to an earlier one absorbs it, so a run of deletions reads as one marker.
* **replace** — a run of whole sections becomes text the model wrote: a summary, or
  a rewrite of the span.
* **rewrite** — one passage inside one tool result or one assistant reply changes,
  in place. The call ids and the message id stay, so nothing about the pairing or
  the row moves.

A substitution (tombstone, replace) is a `user/message` from `PluginSource(plugin=
"clm", form="compaction")` — the form that claims "this stands in for conversation
that has left the surface" (`is_stand_in`), which is what the TUI draws as a
revision and what compaction recognizes as a summary it need not summarize again.
User role, never assistant (decision 2): text standing for several turns is no one
turn's speaker.

**Whole sections, one unbroken run, nothing protected** — `sections.resolve_run`
holds those, because a replacement lands where its earliest node was and a run that
split a call from its result would hand the provider an orphan. Core holds the rest:
every write goes through its revision door (`ph.session.revise`), which keeps a
rewrite's message id and leaves an assistant reply's usage and step behind with the
original, and the surface refuses a node that is no longer current and a
`tool/result` rewrite that changes more than content.

**The gate is off unless a row asks** (decision 4): `fit` refuses growth past the
window, `shrink` refuses any growth. ph's limits ship unset for the reason the
`limits` row gives, and compaction is the backstop either way.

@module ph_clm.edits
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, assert_never, cast

from ph.json import JsonObject, as_int, as_obj, as_seq, as_str
from ph.llm.types import (
    CONTEXT_SUMMARY_MAX_CHARS,
    ContentBlock,
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
from ph.session import (
    Session,
    SessionBatch,
    SessionEvent,
    derive_event_message,
    editable_message,
    rewrite,
    substitute,
)
from ph.session.writers import log_writer
from ph.text import NO_OUTPUT, one_line, thousands

from .sections import EditRefused, Section, SectionMap, label, resolve_one, resolve_run, span_label

_LOG = log_writer(__name__)

__all__ = [
    "REVISED",
    "Editor",
    "Gate",
    "Pending",
    "Removed",
    "Revision",
    "Verb",
    "Via",
    "node_text",
    "receipt",
    "revision_of",
]

PLUGIN = "clm"
"""The plugin name a substitution's `PluginSource` carries."""

REVISED = "clm/revised"

Verb = Literal["tombstone", "replace", "rewrite"]
Gate = Literal["none", "fit", "shrink"]
Via = Literal["tool", "mirror"]
"""Which front end an edit came through: a context tool, or the context file."""

_LABEL = re.compile(r"^\[revised context, standing for [^\]]*\]$")
"""The first line a replacement carries, which a re-revision does not repeat."""

_UNLANDED = -1
"""A draft revision's `replacement` before `Editor.land` knows the seq."""


@dataclass(frozen=True, slots=True)
class Removed:
    """One span a tombstone took out, and the reason given for it."""

    first: int
    last: int
    reason: str

    def describe(self) -> str:
        span = span_label(self.first, self.last)
        return f"{span} ({self.reason})" if self.reason else span


@dataclass(frozen=True, slots=True)
class Revision:
    """What one edit did, as its `clm/revised` record says and its tool reports."""

    verb: Verb
    first: int
    last: int
    replacement: int
    shadowed: tuple[int, ...]
    tokens_before: int
    tokens_after: int
    reread: int
    """Tokens after the edit that a provider reads again, its prefix cache having
    stopped at the first change."""
    context_before: int
    call_id: str | None = None
    via: Via = "tool"
    removed: tuple[Removed, ...] = ()
    """For a tombstone: every span its marker stands for, with the reason given."""

    @property
    def context_after(self) -> int:
        return self.context_before - self.tokens_before + self.tokens_after

    def record(self) -> JsonObject:
        return {
            "verb": self.verb,
            "via": self.via,
            "callId": self.call_id,
            "first": self.first,
            "last": self.last,
            "replacement": self.replacement,
            "shadowed": list(self.shadowed),
            "tokensBefore": self.tokens_before,
            "tokensAfter": self.tokens_after,
            "reread": self.reread,
            "contextBefore": self.context_before,
            "removed": [
                {"first": one.first, "last": one.last, "reason": one.reason} for one in self.removed
            ],
        }


def revision_of(event: SessionEvent | None) -> Revision | None:
    """A `clm/revised` record read back, or `None` for any other event."""
    if event is None or event.type != REVISED:
        return None
    data = event.data
    verb = as_str(data.get("verb"))
    if verb not in ("tombstone", "replace", "rewrite"):
        return None
    via: Via = "mirror" if as_str(data.get("via")) == "mirror" else "tool"
    return Revision(
        verb=cast("Verb", verb),
        first=as_int(data.get("first")),
        last=as_int(data.get("last")),
        replacement=as_int(data.get("replacement")),
        shadowed=tuple(as_int(seq) for seq in as_seq(data.get("shadowed"))),
        tokens_before=as_int(data.get("tokensBefore")),
        tokens_after=as_int(data.get("tokensAfter")),
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
    """What some edits did and what they cost — the one wording, for a tool call's
    result and for a context-file write's alike."""
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
    before = revisions[0].context_before
    after = before + sum(one.tokens_after - one.tokens_before for one in revisions)
    reread = max(one.reread for one in revisions)
    said = "; ".join(done)
    lines = [
        f"{said[:1].upper()}{said[1:]}. Context ~{thousands(before)} → ~{thousands(after)} tokens.",
        f"The ~{thousands(reread)} tokens after it are read once more on the next request: "
        "a prefix cache cannot serve past an edit.",
    ]
    if after > before:
        lines.append(
            "This GREW the context. If you meant to condense, you may have kept the old text "
            "as well as the new."
        )
    return "\n".join(lines)


def node_text(message: Message) -> str:
    """The text of one node an edit reads and replaces: a reply's or a user's words, a
    result's output — the read half of `_with_text`. Calls, reasoning and media are
    not text an edit touches."""
    return "\n".join(text for text in map(_block_text, message.content) if text is not None)


@dataclass(frozen=True, slots=True)
class Pending:
    """One edit validated and drafted, not yet written — `Editor.land` writes several
    in one batch, so a front end that makes many at once lands all or none."""

    write: Callable[[SessionBatch], SessionEvent]
    draft: Revision


@dataclass(frozen=True, slots=True)
class Editor:
    """The three verbs, under one row's section map and gate.

    Each verb is a draft (`tombstoning`, `tombstoning_runs`, `replacing`,
    `rewriting_text`) that validates and measures without writing, and `land`, which
    gates and writes. The tools land one draft per call; the context file compiles
    one write into several and lands them together. Drafts take sections the caller
    has already resolved — protection and contiguity are the caller's to have held.
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
        """Replace a run of whole sections with text the model wrote."""
        sections = self.sections(session)
        start, stop = resolve_run(session, sections, names)
        return self.land(session, [self.replacing(sections, start, stop, text)], call_id=call_id)[0]

    def rewrite(
        self, session: Session, name: str, old: str, new: str, *, call_id: str | None
    ) -> Revision:
        """Change one passage inside one section's tool output or reply, in place."""
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
        draft = self._rewriting(session, sections, section, event, message)
        return self.land(session, [draft], call_id=call_id)[0]

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
        """The marker for `sections[start:stop]`.

        A tombstone beside an earlier one takes it in, so a run of deletions reads as
        one marker — except a section in `keep`, which another draft in the same batch
        is already changing.
        """
        if start > 0 and _absorbed(session, sections[start - 1], keep):
            start -= 1
        if stop < len(sections) and _absorbed(session, sections[stop], keep):
            stop += 1
        removed = tuple(_removals(session, sections[start:stop], reason.strip()))
        text = "removed from context: " + "; ".join(one.describe() for one in removed)
        return self._substituting(
            sections, start, stop, text=f"[{text}]", summary=text, verb="tombstone", removed=removed
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
        """One marker for each run of `positions`, in a batch with other drafts.

        Two runs with only an earlier, untouched marker between them are one run, so
        neither marker-to-be takes that one in on its own — the rule `tombstoning`'s
        absorption would otherwise apply twice to one node.
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
        """Text the model wrote, standing in for `sections[start:stop]`."""
        body = _unlabeled(text)
        if not body:
            raise EditRefused("a replacement needs text; to drop the sections, tombstone them")
        span = span_label(sections[start].id, sections[stop - 1].id)
        return self._substituting(
            sections,
            start,
            stop,
            text=f"[revised context, standing for {span}]\n{body}",
            summary=f"revised {span}",
            verb="replace",
        )

    def rewriting_text(
        self,
        session: Session,
        sections: Sequence[Section],
        section: Section,
        seq: int,
        text: str,
    ) -> Pending:
        """Every text in node `seq` of `section` becomes `text`, in place — a reply
        keeps its calls, a result its call id and anything that is not text."""
        event = _at(session, seq)
        return self._rewriting(session, sections, section, event, _with_text(event, text))

    def land(
        self,
        session: Session,
        drafts: Sequence[Pending],
        *,
        call_id: str | None,
        via: Via = "tool",
    ) -> list[Revision]:
        """Gate every draft, then write them all, each followed by its record, in one
        batch — for one call, through one front end."""
        for draft in drafts:
            self._admit(session, draft.draft.tokens_before, draft.draft.tokens_after)
        landed: list[Revision] = []
        with session.batch() as batch:
            for draft in drafts:
                revision = dataclasses.replace(
                    draft.draft, replacement=draft.write(batch).seq, call_id=call_id, via=via
                )
                _LOG.append(batch, REVISED, revision.record())
                landed.append(revision)
        return landed

    # ------------------------------------------------------------------ inner --

    def _substituting(
        self,
        sections: Sequence[Section],
        start: int,
        stop: int,
        *,
        text: str,
        summary: str,
        verb: Verb,
        removed: tuple[Removed, ...] = (),
    ) -> Pending:
        span = sections[start:stop]
        message = create_user_message(
            content=[{"type": "text", "text": text}],
            source=PluginSource(
                plugin=PLUGIN,
                form="compaction",
                summary=one_line(summary, CONTEXT_SUMMARY_MAX_CHARS),
            ),
        )
        shadowed = tuple(seq for section in span for seq in section.nodes)
        return Pending(
            lambda batch: substitute(batch, shadowed, message),
            Revision(
                verb=verb,
                first=span[0].id,
                last=span[-1].id,
                replacement=_UNLANDED,
                shadowed=shadowed,
                tokens_before=sum(section.tokens for section in span),
                tokens_after=self.sections.meter.measure(message),
                reread=span[-1].after,
                context_before=sum(section.tokens for section in sections),
                removed=removed,
            ),
        )

    def _rewriting(
        self,
        session: Session,
        sections: Sequence[Section],
        section: Section,
        event: SessionEvent,
        message: dict[str, Any],
    ) -> Pending:
        meter = self.sections.meter
        before = derive_event_message(event)
        later = section.nodes[section.nodes.index(event.seq) + 1 :]
        later_tokens = sum(
            meter.measure(one)
            for one in (derive_event_message(_at(session, seq)) for seq in later)
            if one is not None
        )
        return Pending(
            lambda batch: rewrite(batch, event, message),
            Revision(
                verb="rewrite",
                first=section.id,
                last=section.id,
                replacement=_UNLANDED,
                shadowed=(event.seq,),
                tokens_before=0 if before is None else meter.measure(before),
                tokens_after=meter.measure(Message.model_validate(message)),
                reread=section.after + later_tokens,
                context_before=sum(one.tokens for one in sections),
            ),
        )

    def _admit(self, session: Session, before: int, after: int) -> None:
        """Refuse growth the row's gate does not allow. `none` allows everything."""
        if self.gate == "none" or after <= before:
            return
        if self.gate == "shrink":
            raise EditRefused(
                f"this edit grows the context (~{before} → ~{after} tokens), and the "
                "clm-context row's gate is `shrink`: replace stale text with something shorter"
            )
        baseline = self.sections.meter.baseline(session)
        window = baseline.context_window
        if window is None or baseline.tokens - before + after > window:
            # With no window there is no "fits" to test, so growth stays refused — the
            # paper harness's rule for its own `fit` gate.
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


def _removed_by(session: Session, section: Section) -> tuple[Removed, ...] | None:
    """What an earlier tombstone removed, when `section` is that tombstone's marker.

    O(1): `Editor.land` writes the record immediately after the replacement it
    describes, in the same batch, so the record of a marker at `seq` is the event at
    `seq + 1`.
    """
    if len(section.nodes) != 1:
        return None
    marker = section.nodes[0]
    revision = revision_of(session.at(marker + 1))
    if revision is None or revision.verb != "tombstone" or revision.replacement != marker:
        return None
    return revision.removed


def _absorbed(session: Session, neighbor: Section, keep: frozenset[int]) -> bool:
    """Whether a new tombstone takes `neighbor` in: an earlier marker no other draft in
    the batch is changing."""
    return neighbor.id not in keep and _removed_by(session, neighbor) is not None


def _removals(session: Session, span: Sequence[Section], reason: str) -> Iterator[Removed]:
    """What a marker for `span` says it removed: each run of ordinary sections as one
    span with `reason`, and each earlier marker it takes in as what that one said."""
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


def _unlabeled(text: str) -> str:
    """`text` without the label a replacement already carries, so a revision of a
    revision says what it stands for once."""
    lines = text.strip().splitlines()
    while lines and _LABEL.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


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
    """`event`'s message with all of its text replaced by `text`, everything else kept:
    a reply's calls and reasoning, a result's call id and anything not text — the
    write half of `node_text`."""
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
    """`blocks` with every text block gone and one holding `text` where the first was
    (at the front when there was none); no text block at all when `text` is empty."""
    first = next((i for i, block in enumerate(blocks) if block.get("type") == "text"), 0)
    kept = [block for block in blocks if block.get("type") != "text"]
    if not text:
        return kept
    return [*kept[:first], {"type": "text", "text": text}, *kept[first:]]


_Path = tuple[int, ...]
"""Where a text sits in an event's payload: a block index in an assistant reply, or
a result block's index and the index of the text inside it."""


def _text_sites(session: Session, seq: int) -> list[tuple[int, _Path, str]]:
    """The editable texts of one node: an assistant reply's text blocks, a tool
    result's text content. Reasoning, calls and media are not text an edit touches."""
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
    """`event`'s message with the one passage changed — what `ph.session.revise.rewrite`
    lands in its place, keeping everything else the door says a rewrite keeps."""
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
