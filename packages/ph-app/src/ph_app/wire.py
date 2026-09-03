"""Reading the log's plain JSON without a model in the way.

Every reader of a stored or streamed log — the TUI's transcript adapter and session
lister, `ph agents attach` — sees payloads in one of two shapes: **frozen**, when
the event is in memory (a `MappingProxyType` over tuples), or **plain**, when it
was persisted and read back (`dict`s over `list`s). The helpers here accept both,
and the distinction is load-bearing: a reader that tested for `dict` would work on
resume and silently see nothing live.

Absence is normal too. The log is JSON, every field is optional to a reader, and a
missing one must cost a row rather than the transcript.

**In `ph_app`, not `ph_app.tui`.** `ph_app.tui.__init__` imports the Textual app,
so a module under it cannot be read from without paying for the terminal framework
— which a headless `ph agents attach` should never do to render a line of a log it
just received.

@module ph_app.wire
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from ph.tools import ToolCallView, ToolResultView
from ph.tools.presentation import CARD_VIEWS

__all__ = [
    "describe",
    "first",
    "index_at_or_before",
    "matches_terms",
    "media_labels",
    "message_of",
    "obj",
    "one_line",
    "result_block",
    "seq",
    "source_of",
    "split_terms",
    "text_of_wire",
    "view_of",
]


def obj(value: Any) -> Mapping[str, Any]:
    """A wire object, or an empty one."""
    return value if isinstance(value, Mapping) else {}


def seq(value: Any) -> Sequence[Any]:
    """A wire list, or an empty one. A tuple in memory, a list on disk."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return value


def first(value: Any) -> Mapping[str, Any]:
    """The first object of a wire list, or an empty one."""
    items = seq(value)
    return obj(items[0]) if items else {}


def text_of_wire(
    blocks: Any, *, kind: str = "text", placeholder: Callable[[str], str] | None = None
) -> str:
    """Join the text of wire-form content blocks of one kind.

    The wire-form twin of `ph.llm.types.text_of`, which needs models. `kind`
    selects `text` or `reasoning`. `placeholder` names the blocks it skips —
    an auditor wants to see that an image was there; a transcript row does not
    — and mirrors the same argument on `text_of` so the two joins stay one
    behaviour described twice rather than two behaviours.
    """
    parts: list[str] = []
    for block in seq(blocks):
        if not isinstance(block, Mapping):
            continue
        block_kind = block.get("type")
        if block_kind == kind:
            parts.append(str(block.get("text", "")))
        elif placeholder is not None and isinstance(block_kind, str):
            parts.append(placeholder(block_kind))
    return "\n".join(parts)


def message_of(event: Any) -> Mapping[str, Any]:
    """The message inside an event's payload, whichever shape it takes.

    `user/message`'s payload *is* the message; `assistant/message` wraps one
    beside `turn`, `step` and `usage`. Both projections need to know that, and
    getting it wrong is silent — a message with no content rather than an error
    — so the knowledge lives here instead of in each reader.
    """
    payload = obj(getattr(event, "data", event))
    return obj(payload.get("message")) if "message" in payload else payload


def view_of(event_type: str, sidecar: Any) -> ToolCallView | ToolResultView | None:
    """The card view a daemon sent beside an event, validated **here** (P7-12).

    Here because this is the wire edge: every other frame a client reads is
    validated in this module or its callers, and a fold two layers down should be
    handed a type, not a `Mapping` it has to distrust. The event type chooses the
    model — a `tool/call` carries a `ToolCallView`, a `tool/result` a
    `ToolResultView` — and anything else, or anything that does not parse,
    is `None`: the adapter then draws the generic card it draws with no daemon at
    all, which is a plain card and not a wrong one.
    """
    model = CARD_VIEWS.get(event_type)
    if model is None or not isinstance(sidecar, Mapping):
        return None
    try:
        return model.model_validate(dict(sidecar))
    except ValidationError:
        return None


def result_block(message: Any) -> Mapping[str, Any]:
    """The `tool_result` block inside a tool-result message, or an empty one.

    One block carries both the visible text and the error flag, and it sits one
    level deeper than it looks: the text is `message.content[0].content`, not
    `message.content`. Three readers already agreed on that hop and a fourth —
    `ph agents attach` — skipped it, so `text_of_wire` selected `type: "text"`
    against blocks of `type: "tool-result"` and every tool result followed from
    a terminal rendered blank. `message_of`'s docstring already names this class
    of mistake: getting it wrong is silent, so the knowledge lives here.
    """
    return first(obj(message).get("content"))


def describe(data: Any) -> str:
    """A one-line `key=value` account of a payload, or `""` when it has none.

    Deliberately generic, which is the whole point: a type this build has no
    phrase for still gets a readable line, because a view that silently hid an
    event it did not recognize would be exactly the omission A11 forbids. The
    auditor's projection has read payloads this way since P3-24; a second reader
    outside the TUI is what moved it here.

    Read frozen: this builds one truncated line, and deep-copying the payload to
    iterate its top level was a second full copy of the same tree.
    """
    payload = obj(data)
    return one_line(", ".join(f"{key}={value}" for key, value in payload.items() if value != ""))


def source_of(message: Any) -> tuple[str, str, str]:
    """`(kind, name, form)` for a message's producer.

    The wire keys `plugin`/`model`/`callId` are the discriminated members of
    `MessageSource`; naming them in one place is what stops two readers
    disagreeing about who produced a record.
    """
    source = obj(obj(message).get("source"))
    kind = str(source.get("kind") or "")
    name = str(source.get("plugin") or source.get("model") or source.get("callId") or "")
    return kind, name, str(source.get("form") or "")


def split_terms(query: str, prefix: str) -> tuple[list[str], str]:
    """Split a query into `prefix`-tagged terms and the free text that is left.

    Here rather than in the one widget that needs it, because `matches_terms`
    below claims to be "one definition of what filtering means" and a caller that
    re-lexes the query before handing it over quietly makes that false: a quoted
    phrase or a `-negation` rule added there would be split apart by the caller's
    own splitter before it ever arrived. One tokenizer, and the picker gets the
    tagged form for free the day it wants it.

    Case-folded once, here, because `matches_terms` folds too and a tag compared
    against an unfolded term would miss `Type:workspace`.
    """
    terms = query.lower().split()
    tagged = [term.removeprefix(prefix) for term in terms if term.startswith(prefix)]
    return tagged, " ".join(term for term in terms if not term.startswith(prefix))


def matches_terms(haystack: str, query: str) -> bool:
    """Every whitespace-separated term appears in `haystack`, case-folded.

    One definition of what filtering means, because the TUI now filters in two
    places — the choice picker and the trajectory table — and `modals/base.py`
    predicted exactly this: bespoke screens "would mean five places to get the
    escape key, the focus order and the filter semantics subtly different".
    """
    return all(term in haystack.lower() for term in query.lower().split())


def one_line(text: str, limit: int = 120) -> str:
    """Whitespace collapsed and truncated with an ellipsis — a card subtitle."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


def index_at_or_before(seqs: Iterable[int], target: int) -> int:
    """The position of the last seq at or before `target`, or `-1`.

    The one definition of the join between the two projections (P4-17). Both
    readers need the *nearest preceding* entry rather than an exact hit, because
    they do not render the same events: a `request/header` is a record with no
    transcript row by design, and a streamed chunk is a row with no record. A
    reader that answered "nothing" for those would strand someone who asked to
    be taken somewhere.

    Written once because two readers with their own version of "nearest" is how
    the two views come to disagree about which row that is. The sequence is in
    log order, so this stops at the first entry past the target.
    """
    found = -1
    for index, value in enumerate(seqs):
        if value > target:
            break
        if value >= 0:
            found = index
    return found


def media_labels(blocks: Any) -> list[str]:
    """One human label per media block: `image/png · diagram.png`.

    A separate read rather than a `placeholder` on `text_of_wire`, because that
    argument is `(kind) -> str` and deliberately the twin of `ph.llm.types`'s —
    widening it to see the block would change two joins to describe one row.
    What a person needs here is the *name they attached*, which only the block
    has.
    """
    labels: list[str] = []
    for block in seq(blocks):
        if not isinstance(block, Mapping) or block.get("type") != "media":
            continue
        attachment = obj(block.get("attachment"))
        name = attachment.get("name") or attachment.get("attachmentId")
        labels.append(f"{attachment.get('mime') or 'file'} · {name}")
    return labels
