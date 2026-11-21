"""Ask a model for a *value* and not for prose (P7-17).

Some calls do not want a reply. They want a shape. The review gate of
`rlm-harness` asks "should this conversation be refined, and why", and its
planner asks for a list of edits. Both asked for JSON in the prompt and then
relied on a tolerant parser, `parse_json_object`, which is now removed. The
docstring of that parser admitted that an instruction to return only JSON is
not a guarantee. The parser stripped fences, retried on a substring, and failed
closed when the model wrote a sentence first.

**Three separate things replace that reliance.** The wire constrains the reply
where it can (`response_format`, and `ResolvedModel.structured_output` says
whether this route does). The reply is *validated* against the schema whether
or not the wire helped. And a reply that fails is sent back with the specific
violation named. The second is what makes this correct on a route with no wire
support. Anthropic has no equivalent today, so a caller there gets the
instruction, the validation and the retry, and not the guarantee.

**A schema that cannot constrain causes a warning at the call. It is not
discovered on the third identical retry.** A converter from JSON Schema to a
grammar needs a *type* for each field to build a real constraint. `required`
without a matching `properties` entry therefore degrades to a grammar that
constrains almost nothing: the model returns whatever shape it likes, and every
correction fails in the same way. This is ported from the playbook executor of
OpenMono, which documents the failure and warns at step start. Without the
warning, the cost is three model calls and an abort.

@module ph.llm.structured
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import replace
from typing import Any, overload

from pydantic import BaseModel

from ..json import JsonObject, JsonValue, first_json_value, loads
from ..tools.json_schema import (
    CONSTRAINING_KEYWORDS,
    schema_of,
    validate_json_schema_value,
)
from .adapter import LlmError
from .assembler import BlockAssembler
from .types import GenerateOptions, LlmFailure, StreamChunk, create_message, text_of

__all__ = [
    "SchemaViolation",
    "ask_for_shape",
    "structural_warning",
    "validated_shape",
]

log = logging.getLogger("ph.llm.structured")

ATTEMPTS = 3
"""How many times a reply may be asked for before the call gives up.

A route that enforces the schema gets one attempt fewer (`ask_for_shape`).
Where the wire enforces the schema, the correction is itself guaranteed valid,
so a second attempt is already generous. The full budget is for the routes that
cannot enforce it. There, the ordinary failure is a model that writes a
confirmation sentence in place of the document."""


class SchemaViolation(Exception):
    """A reply that is not the shape that was asked for, with what was wrong.

    The exception carries the violations and not one sentence, because the next
    step is to read them back to the model. "missing required field(s):
    severity" is a correction that the model can act on. "invalid response" is
    not.
    """

    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations) or "the reply was not the requested shape")
        self.violations = violations


def structural_warning(schema: dict[str, Any]) -> str | None:
    """Why this schema may not constrain anything, reported before the schema is used.

    The case that matters is `required` naming a field that `properties` does
    not type. A reply to such a schema passes validation here: the field is
    present, so nothing complains. But the schema produces a grammar that
    constrains nothing on the wire. A caller sees the same malformed answer
    three times and no reason.

    A caller that passes a pydantic model cannot reach this, and `ask_for_shape`
    takes only a pydantic model. The warning is for a dict schema: the `output`
    block of a skill (`ph.seams.skills`), and the output that a child is asked
    for (`ph.seams.subagents`).
    """
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(required, list):
        return None
    typed = properties if isinstance(properties, dict) else {}
    untyped = [str(name) for name in required if not _constrains(typed.get(str(name)))]
    if not untyped:
        return None
    return (
        f"required field(s) {', '.join(untyped)} have no typed `properties` entry, "
        "so a grammar built from this schema will not constrain the reply"
    )


def _constrains(entry: object) -> bool:
    """Whether one `properties` entry says anything about its value (C11).

    The test is `CONSTRAINING_KEYWORDS`. It was `"type" in entry`. An optional
    field has no top-level `type`, because pydantic renders `str | None` as an
    `anyOf` and a `Literal` as an `enum`. `required` is about *presence* and
    not about nullability, so a field that is both required and optional is
    ordinary. With the old test, every schema with such a field warned that its
    grammar would not constrain the reply. People learn to ignore a warning
    that fires on correct input, and the warning then fails in the case that it
    was written for.
    """
    return isinstance(entry, dict) and any(key in entry for key in CONSTRAINING_KEYWORDS)


def _object_in(text: str) -> JsonValue:
    """The JSON value in a reply. A fence or a sentence around it is tolerated.

    This is kept even where the wire enforces the schema, because "enforced" is
    a property of the *route* and this function runs on every route. It is now
    a fallback and not the whole mechanism. That is the difference that P7-17
    is about.
    """
    for candidate in _candidates(text):
        try:
            return loads(candidate)
        except json.JSONDecodeError:
            continue
    raise SchemaViolation(["the reply is not JSON"])


def _candidates(text: str) -> Iterator[str]:
    """Spans of `text` that might be the document, widest first, produced lazily (C11).

    There are three readings, in the order of how likely and how cheap they are:

    * The whole reply, for a model that answered with nothing else.
    * The first `{` to the last `}`, which is one object wrapped in prose or in
      a fence.
    * The first *complete* object, for a reply that holds two. `here it is:
      {"a": 1} and also {"b": 2}` spans both objects under the second reading
      and parses as neither. The caller was then told "the reply is not JSON"
      about a reply that was mostly JSON.

    **This is a generator, because the third reading costs a parse.**
    `raw_decode` scans at ~32 ns/char, which is 628 µs on a 20 KB reply. That
    cost would be paid on every attempt, and `ask_for_shape` retries.
    Widest-first also matters by itself. One object that contains another is
    the common shape, and taking the inner object there would return a fragment
    that validates against nothing.
    """
    yield text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return
    widest = text[start : end + 1]
    yield widest
    first = first_json_value(text, start)
    if first is not None and first != widest:
        yield first


@overload
def validated_shape[Shape: BaseModel](text: str, shape: type[Shape]) -> Shape: ...
@overload
def validated_shape(text: str, shape: dict[str, Any]) -> JsonObject: ...
def validated_shape(text: str, shape: type[BaseModel] | dict[str, Any]) -> BaseModel | JsonObject:
    """The reply as a validated value, or a `SchemaViolation` that names what is wrong.

    `shape` is the same union that `ToolOutput.schema` takes, for the same
    reason. A pydantic model is one declaration that produces both the wire
    schema and the validator. A caller that has one therefore does not get a
    *second*, weaker check here and then the real check in its own code. Given
    a model, this returns the instance. Given a dict, it returns the dict that
    it validated. The overloads make that statement a *type*, so a caller does
    not have to restate it at the call site.
    """
    value = _object_in(text)
    if not isinstance(value, dict):
        raise SchemaViolation(["the reply is not a JSON object"])
    violations = validate_json_schema_value(shape, value)
    if violations:
        raise SchemaViolation(violations)
    if isinstance(shape, type) and issubclass(shape, BaseModel):
        return shape.model_validate(value)
    return value


async def ask_for_shape[Shape: BaseModel](
    stream: Callable[[GenerateOptions], Awaitable[AsyncIterator[StreamChunk]]],
    options: GenerateOptions,
    shape: type[Shape],
    *,
    enforced: bool = False,
    attempts: int = ATTEMPTS,
) -> Shape:
    """One call that must come back as `shape`. A reply that fails is sent back for correction.

    `stream` is `ctx.llm.stream`. It is passed in and not looked up here, so
    this stays a function over the seam and does not become a second thing that
    resolves the seam. `shape` is a pydantic model. It becomes the wire schema
    *and* the validator, so the caller gets a typed instance. The reply is
    checked one time, and not half-checked here and then checked properly by
    the caller.

    `enforced` is `ResolvedModel.structured_output`: whether this route builds
    a grammar from the schema. An enforced route gets one attempt fewer,
    because a constrained correction is itself guaranteed valid. A violation
    from a route that claimed enforcement is logged as a bug of the adapter or
    of the provider, which it is. It must not read as a model that would not
    comply.

    The correction turn **offers no tools**. That is a requirement and not a
    courtesy. A tool call is not a document in the shape of the schema, so a
    route that is asked for both has to break one promise. For that reason,
    `options` that carry tools are refused outright and are not stripped
    silently.

    `BlockAssembler` is the assembly that the loop uses, so a caller cannot
    disagree with the transcript about what a reply said. `text_of` drops
    reasoning blocks and does not hand them to a JSON parser.
    """
    if options.tools:
        raise ValueError("a structured reply and tools cannot be asked for in one call")
    wire = schema_of(shape)
    warning = structural_warning(wire)
    if warning is not None:
        log.warning("ph.llm.structured: %s", warning)

    budget = max(1, attempts - 1 if enforced else attempts)
    attempt = replace(options, response_schema=wire)
    for turn in range(budget):
        assembler = BlockAssembler()
        async for chunk in await stream(attempt):
            assembler.push(chunk)
        if assembler.finish.kind == "error":
            # **The call failed. The shape was never in question** (C11). When
            # this was raised as a `SchemaViolation`, every caller was told that
            # the model had answered badly. A rate limit, a context overflow or
            # a dropped connection arrived under the one exception that means
            # "the reply was the wrong shape", and it carried none of the code
            # of the provider. `llm-retry` branches on that code, and
            # `SchemaViolation` has none.
            failure = assembler.finish.failure or LlmFailure(
                message="the model call failed", code="UNKNOWN"
            )
            raise LlmError(failure.message, failure.code, failure)
        reply = text_of(assembler.blocks()).strip()
        try:
            return validated_shape(reply, shape)
        except SchemaViolation as violation:
            if enforced and turn == 0:
                log.warning(
                    "ph.llm.structured: %s/%s declares structured output but the reply "
                    "violated the schema: %s",
                    options.provider,
                    options.model,
                    "; ".join(violation.violations),
                )
            if turn == budget - 1:
                raise
            attempt = _corrected(attempt, reply, violation)
    raise SchemaViolation(["the reply was not the requested shape"])  # pragma: no cover


def _corrected(options: GenerateOptions, reply: str, violation: SchemaViolation) -> GenerateOptions:
    """The follow-up request: what came back, and exactly what was wrong with it.

    This uses `replace` and not a field-by-field rebuild, which is what a frozen
    dataclass is for. The hand-written version had already dropped `stop`, and
    it would have dropped every field that was added after it.
    """
    return replace(
        options,
        messages=(
            *options.messages,
            create_message(
                role="assistant",
                content=[{"type": "text", "text": reply}],
                # The source is the route that produced the reply, because
                # `ModelSource` names one. This message is the reply of the
                # model, handed back to it. Attributing it to anything else
                # would be the false statement that the source field exists to
                # prevent.
                source={"kind": "model", "provider": options.provider, "model": options.model},
            ),
            create_message(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"That reply is not valid against the requested schema: "
                            f"{'; '.join(violation.violations)}. Reply with the JSON "
                            "document only — no prose, no code fence."
                        ),
                    }
                ],
                source={"kind": "plugin", "plugin": "ph.llm.structured", "form": "instructions"},
            ),
        ),
    )
