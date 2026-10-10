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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ph.json import JsonObject, as_int, as_obj, as_seq, as_str
from ph.llm.types import CONTEXT_SUMMARY_MAX_CHARS, Message, PluginSource, create_user_message
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
from ph.text import one_line

from .sections import EditRefused, Section, SectionMap, resolve_one, resolve_run, span_label

_LOG = log_writer(__name__)

__all__ = ["REVISED", "Editor", "Gate", "Removed", "Revision", "Verb", "revision_of"]

PLUGIN = "clm"
"""The plugin name a substitution's `PluginSource` carries."""

REVISED = "clm/revised"

Verb = Literal["tombstone", "replace", "rewrite"]
Gate = Literal["none", "fit", "shrink"]

_UNLANDED = -1
"""A draft revision's `replacement` before `Editor._land` knows the seq."""


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
    call_id: str | None
    removed: tuple[Removed, ...] = ()
    """For a tombstone: every span its marker stands for, with the reason given."""

    def record(self) -> JsonObject:
        return {
            "verb": self.verb,
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
        removed=tuple(
            Removed(
                first=as_int(as_obj(one).get("first")),
                last=as_int(as_obj(one).get("last")),
                reason=as_str(as_obj(one).get("reason")),
            )
            for one in as_seq(data.get("removed"))
        ),
    )


@dataclass(frozen=True, slots=True)
class Editor:
    """The three verbs, under one row's section map and gate."""

    sections: SectionMap
    gate: Gate

    def tombstone(
        self, session: Session, names: Sequence[str], reason: str, *, call_id: str | None
    ) -> Revision:
        """Replace a run of whole sections with a one-line marker."""
        sections = self.sections(session)
        start, stop = resolve_run(session, sections, names)
        removed = [Removed(sections[start].id, sections[stop - 1].id, reason.strip())]
        # A tombstone beside an earlier one takes it in: one marker for the run.
        if start > 0 and (before := _removed_by(session, sections[start - 1])) is not None:
            start -= 1
            removed = [*before, *removed]
        if stop < len(sections) and (behind := _removed_by(session, sections[stop])) is not None:
            stop += 1
            removed = [*removed, *behind]
        text = "removed from context: " + "; ".join(one.describe() for one in removed)
        return self._substitute(
            session,
            sections[start:stop],
            sections,
            f"[{text}]",
            text,
            "tombstone",
            call_id,
            tuple(removed),
        )

    def replace(
        self, session: Session, names: Sequence[str], text: str, *, call_id: str | None
    ) -> Revision:
        """Replace a run of whole sections with text the model wrote."""
        if not text.strip():
            raise EditRefused("a replacement needs text; to drop the sections, tombstone them")
        sections = self.sections(session)
        start, stop = resolve_run(session, sections, names)
        span = span_label(sections[start].id, sections[stop - 1].id)
        body = f"[revised context, standing for {span}]\n{text.strip()}"
        return self._substitute(
            session, sections[start:stop], sections, body, f"revised {span}", "replace", call_id
        )

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
        meter = self.sections.meter
        before = derive_event_message(event)
        later = section.nodes[section.nodes.index(seq) + 1 :]
        later_tokens = sum(
            meter.measure(message)
            for message in (derive_event_message(_at(session, one)) for one in later)
            if message is not None
        )
        return self._land(
            session,
            lambda batch: rewrite(batch, event, message),
            Revision(
                verb="rewrite",
                first=section.id,
                last=section.id,
                replacement=_UNLANDED,
                shadowed=(seq,),
                tokens_before=0 if before is None else meter.measure(before),
                tokens_after=meter.measure(Message.model_validate(message)),
                reread=section.after + later_tokens,
                context_before=sum(one.tokens for one in sections),
                call_id=call_id,
            ),
        )

    def _substitute(
        self,
        session: Session,
        span: Sequence[Section],
        sections: Sequence[Section],
        text: str,
        summary: str,
        verb: Verb,
        call_id: str | None,
        removed: tuple[Removed, ...] = (),
    ) -> Revision:
        """Land `text` where `span` was, as a revision standing in for it."""
        message = create_user_message(
            content=[{"type": "text", "text": text}],
            source=PluginSource(
                plugin=PLUGIN,
                form="compaction",
                summary=one_line(summary, CONTEXT_SUMMARY_MAX_CHARS),
            ),
        )
        shadowed = tuple(seq for section in span for seq in section.nodes)
        return self._land(
            session,
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
                call_id=call_id,
                removed=removed,
            ),
        )

    def _land(
        self, session: Session, write: Callable[[SessionBatch], SessionEvent], draft: Revision
    ) -> Revision:
        """Gate the edit, then land it and its record in one batch, the record second."""
        self._admit(session, draft.tokens_before, draft.tokens_after)
        with session.batch() as batch:
            revision = dataclasses.replace(draft, replacement=write(batch).seq)
            _LOG.append(batch, REVISED, revision.record())
        return revision

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

    O(1): `_land` writes the record immediately after the replacement it describes,
    in the same batch, so the record of a marker at `seq` is the event at `seq + 1`.
    """
    if len(section.nodes) != 1:
        return None
    marker = section.nodes[0]
    revision = revision_of(session.at(marker + 1))
    if revision is None or revision.verb != "tombstone" or revision.replacement != marker:
        return None
    return revision.removed


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
