"""Reading the plain JSON of the log without a model in the way.

Every reader of a stored or streamed log sees payloads in one of two shapes. The
readers are the transcript adapter of the TUI, the session lister, and
`phern agents attach`. A payload is **frozen** when the event is in memory (a
`MappingProxyType` over tuples). It is **plain** when it was persisted and read
back (`dict`s over `list`s). The helpers here accept both shapes, and the
distinction matters: a reader that tested for `dict` would work on resume and
silently see nothing live.

Absence is normal too. The log is JSON, and every field is optional to a reader.
A missing field must cost one row and not the transcript.

`as_obj` and `as_seq` started here as `obj`/`seq` and now live in `ph.json`,
which every reader imports directly. This module re-exported them for a while,
so that its own importers kept working. That shim is retired, because it gave
one function two import paths.

**This module is in `ph_app`, not `ph_app.tui`.** `ph_app.tui.__init__` imports
the Textual app, so a caller that reads from a module under it pays for the
terminal framework. A headless `phern agents attach` must never pay that to
render a line of a log that it just received.

@module ph_app.wire
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from ph.json import JsonObject, JsonValue, as_obj, as_seq, as_str
from ph.text import brief_value, one_line, redacted_marker
from ph.tools import ToolCallView, ToolResultView
from ph.tools.presentation import CARD_VIEWS

__all__ = [
    "describe",
    "fields_of",
    "first",
    "index_at_or_before",
    "matches_terms",
    "media_labels",
    "message_of",
    "named",
    "result_block",
    "source_of",
    "split_terms",
    "stated",
    "text_of_wire",
    "view_of",
]


def first(value: object) -> Mapping[str, Any]:
    """The first object of a wire list, or an empty one."""
    items = as_seq(value)
    return as_obj(items[0]) if items else {}


def text_of_wire(
    blocks: object,
    *,
    kind: str = "text",
    placeholder: Callable[[str], str] | None = None,
) -> str:
    """Join the text of wire-form content blocks of one kind.

    This is the wire-form twin of `ph.llm.types.text_of`, which needs models.
    `kind` selects `text` or `reasoning`. `placeholder` names the blocks that
    the join skips. An auditor wants to see that an image was there, and a
    transcript row does not. The argument mirrors the same argument on
    `text_of`, so that the two joins stay one behavior that is described twice
    and do not become two behaviors.
    """
    parts: list[str] = []
    for block in as_seq(blocks):
        if not isinstance(block, Mapping):
            continue
        block_kind = block.get("type")
        if block_kind == kind:
            # The `text` of a redacted reasoning block is the ciphertext of the
            # provider, not prose (G8). The answer is here and not at each
            # caller, for the reason that this function exists: it and
            # `ph.llm.types.text_of` are one behavior that is described twice,
            # and a branch at the call site would make it three.
            parts.append(redacted_marker() if block.get("redacted") else as_str(block.get("text")))
        elif placeholder is not None and isinstance(block_kind, str):
            parts.append(placeholder(block_kind))
    return "\n".join(parts)


def message_of(payload: JsonObject) -> Mapping[str, Any]:
    """The message inside the payload of an event, whichever shape it takes.

    The payload of `user/message` *is* the message. `assistant/message` wraps
    one beside `turn`, `step` and `usage`. Both projections need to know that. A
    mistake here is silent: it gives a message with no content and not an error.
    So the knowledge lives here and not in each reader.

    The argument is the payload, never the event. Both callers already hold
    `event.data` where they ask. `agents.py` narrowed it a line above, and
    `trajectory.py` binds it as `outer` for its own reads. A function that took
    either would need a `getattr` check of the shape, in place of the one type
    that both sides actually have.
    """
    return as_obj(payload.get("message")) if "message" in payload else payload


def view_of(event_type: str, sidecar: JsonValue) -> ToolCallView | ToolResultView | None:
    """The card view that a daemon sent beside an event, validated **here** (P7-12).

    It is validated here because this is the wire edge. Every other frame that a
    client reads is validated in this module or its callers. A fold two layers
    down should receive a type, and not a `Mapping` that it has to distrust. The
    event type chooses the model: a `tool/call` carries a `ToolCallView`, and a
    `tool/result` carries a `ToolResultView`. Anything else, and anything that
    does not parse, is `None`. The adapter then draws the generic card that it
    draws with no daemon at all, which is a plain card and not a wrong one.

    The parameter is `JsonValue` and not `Any`, and the `isinstance` below
    stays. The card is *unvalidated* JSON, not an untyped object, and those are
    different promises. `Any` let a caller pass anything at all. `JsonValue`
    says that a daemon sent the value and nothing has looked at it yet, and that
    is exactly what this function narrows.
    """
    model = CARD_VIEWS.get(event_type)
    if model is None or not isinstance(sidecar, Mapping):
        return None
    try:
        return model.model_validate(dict(sidecar))
    except ValidationError:
        return None


def result_block(message: object) -> Mapping[str, Any]:
    """The `tool_result` block inside a tool-result message, or an empty one.

    One block carries both the visible text and the error flag. It sits one
    level deeper than it looks: the text is `message.content[0].content`, not
    `message.content`. Three readers already agreed on that hop. A fourth,
    `phern agents attach`, skipped it. So `text_of_wire` selected
    `type: "text"` against blocks of `type: "tool-result"`, and every tool
    result that was followed from a terminal rendered blank. The docstring of
    `message_of` already names this class of mistake: it is silent, so the
    knowledge lives here.
    """
    return first(as_obj(message).get("content"))


def describe(data: object) -> str:
    """A one-line `key=value` account of a payload, or `""` when it has none.

    It is deliberately generic, and that is the purpose. A type that this build
    has no phrase for still gets a readable line. A view that silently hid an
    event that it did not recognize would be exactly the omission that A11
    forbids. The projection of the auditor has read payloads this way since
    P3-24. A second reader outside the TUI is what moved it here.

    The payload is read frozen. This builds one truncated line, and a deep copy
    of the payload to iterate its top level was a second full copy of the same
    tree.
    """
    return one_line(", ".join(f"{key}={value}" for key, value in fields_of(data)))


def fields_of(data: object) -> list[tuple[str, str]]:
    """The top-level fields of a payload as `(key, value)` pairs.

    Each value is on one line (`brief_value`), and an empty value is left out.
    `describe` joins these pairs, and a view that lists the fields one per line
    shows them."""
    return [(key, brief_value(value)) for key, value in as_obj(data).items() if value != ""]


def source_of(message: object) -> tuple[str | None, str | None, str | None]:
    """`(kind, name, form)` for the producer of a message. A part that the
    message does not name is `None`.

    The wire keys `plugin`/`model`/`callId` are the discriminated members of
    `MessageSource`. They are named in one place so that two readers cannot
    disagree about who produced a record.

    A part that is not a string, or that is an empty string, names nothing. So
    a reader tests a part against `None` and never against `""`.
    """
    source = as_obj(as_obj(message).get("source"))
    name = source.get("plugin") or source.get("model") or source.get("callId")
    return named(source.get("kind")), named(name), named(source.get("form"))


def named(value: object) -> str | None:
    """A JSON value that names something, or `None` for anything else.

    A value that is not a string, and an empty string, name nothing. A number is
    not turned into a name, for the reason that `ph.json.as_str` gives: `str()`
    of a wrong shape invents a value."""
    return value if isinstance(value, str) and value else None


def stated(text: str) -> str | None:
    """`text`, or `None` when it is empty.

    This is the one conversion at the edge of a `ph-core` type that still spells
    absence as `""`, such as `AgentOptions.model`, `Profile.name` or
    `ModelEntry.key`. It takes `str` and not `str | None` on purpose. When such a
    field becomes `str | None`, mypy names every call here, and each call is then
    deleted."""
    return text or None


def split_terms(query: str, prefix: str) -> tuple[list[str], str]:
    """Split a query into `prefix`-tagged terms and the free text that is left.

    It is here and not in the one widget that needs it. `matches_terms` below
    claims to be "one definition of what filtering means". A caller that lexes
    the query again before it passes the query on makes that claim false without
    notice. If a quoted phrase or a `-negation` rule were added there, the
    splitter of the caller would split it apart before it arrived. So there is
    one tokenizer, and the picker gets the tagged form at no cost when it wants
    it.

    The query is case-folded one time, here. `matches_terms` folds too, and a
    tag that was compared against an unfolded term would miss `Type:workspace`.
    """
    terms = query.lower().split()
    tagged = [term.removeprefix(prefix) for term in terms if term.startswith(prefix)]
    return tagged, " ".join(term for term in terms if not term.startswith(prefix))


def matches_terms(haystack: str, query: str) -> bool:
    """Every whitespace-separated term appears in `haystack`, case-folded.

    This is one definition of what filtering means, because the TUI now filters
    in two places: the choice picker and the trajectory table. `modals/base.py`
    predicted exactly this: five separate screens would be five places where the
    escape key, the focus order and the filter rules could differ slightly.
    """
    return all(term in haystack.lower() for term in query.lower().split())


def index_at_or_before(seqs: Iterable[int | None], target: int) -> int | None:
    """The position of the last seq at or before `target`, or `None` when no entry
    is that early. An entry whose seq is `None` has no position in the log, and
    it is never the answer.

    This is the one definition of the join between the two projections (P4-17).
    It was written for the transcript and the trajectory table, and the diff
    screen uses it too. Each reader needs the *nearest preceding* entry and not
    an exact hit, because the views do not render the same events. A
    `request/header` is a record that has no transcript row by design, and a
    streamed chunk is a row that has no record. A reader that answered "nothing"
    for those would strand a user who asked to be taken somewhere.

    It is written one time, because readers that each have their own version of
    "nearest" make the views disagree about which row that is. The sequence is
    in log order, so the loop stops at the first entry past the target.
    """
    found: int | None = None
    for index, value in enumerate(seqs):
        if value is None:
            continue
        if value > target:
            break
        found = index
    return found


def media_labels(blocks: object) -> list[str]:
    """One human label for each media block: `image/png · diagram.png`.

    This is a separate read and not a `placeholder` on `text_of_wire`. That
    argument is `(kind) -> str`, and it is deliberately the twin of the argument
    in `ph.llm.types`. To widen it so that it sees the block would change two
    joins in order to describe one row. What a user needs here is the *name that
    they attached*, and only the block has it.
    """
    labels: list[str] = []
    for block in as_seq(blocks):
        if not isinstance(block, Mapping) or block.get("type") != "media":
            continue
        attachment = as_obj(block.get("attachment"))
        name = attachment.get("name") or attachment.get("attachmentId")
        labels.append(f"{attachment.get('mime') or 'file'} · {name}")
    return labels
