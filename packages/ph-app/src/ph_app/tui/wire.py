"""Reading the log's plain JSON without a model in the way.

Every reader in the TUI — the transcript adapter, the session lister — sees
payloads in one of two shapes: **frozen**, when the event is in memory (a
`MappingProxyType` over tuples), or **plain**, when it was persisted and read
back (`dict`s over `list`s). The helpers here accept both, and the distinction
is load-bearing: a reader that tested for `dict` would work on resume and
silently see nothing live. Phase 1 hit the same trap with `run_code` arguments.

Absence is normal too. The log is JSON, every field is optional to a reader, and
a missing one must cost a row rather than the transcript.

@module ph_app.tui.wire
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "first",
    "index_at_or_before",
    "matches_terms",
    "media_labels",
    "message_of",
    "obj",
    "one_line",
    "seq",
    "source_of",
    "text_of_wire",
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
