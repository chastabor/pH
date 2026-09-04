"""Asking a model for a *value* rather than prose (P7-17).

Some calls do not want a reply, they want a shape: `rlm-harness`'s review gate
asks "should this conversation be refined, and why", and its planner asks for a
list of edits. Both asked for JSON in the prompt and then hoped — `parse_json_object`'s
own docstring conceded it, in as many words: *"return only JSON is an instruction
and not a guarantee"*. It stripped fences, retried on a substring, and failed
closed when the model wrote a sentence first.

**Three things replace the hoping, and they are not the same thing.** The wire
constrains where it can (`response_format`, and `ResolvedModel.structured_output`
says whether this route does); the reply is *validated* against the schema
whether or not the wire helped; and a reply that fails is sent back once with the
specific violation named. The middle one is what makes this honest on a route
with no wire support — Anthropic has no equivalent today, so there a caller gets
the instruction, the validation and the retry, and not the guarantee.

**A schema that cannot constrain is warned about at the call, not discovered on
the third identical retry.** A JSON-Schema-to-grammar converter needs a *type*
for each field to build a real constraint, so `required` without a matching
`properties` entry degrades to a near-unconstrained grammar: the model returns
whatever shape it likes and every correction fails the same way. Ported from
OpenMono's playbook executor, which documents the failure and warns at step
start; the cost of not saying it is three model calls and an abort.

@module ph.llm.structured
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from ..tools.json_schema import schema_of, validate_json_schema_value
from .assembler import BlockAssembler
from .types import GenerateOptions, create_message, text_of

__all__ = [
    "SchemaViolation",
    "ask_for_shape",
    "structural_warning",
    "validated_shape",
]

log = logging.getLogger("ph.llm.structured")

ATTEMPTS = 3
"""How many times a reply may be asked for before the call gives up.

One more than the *one* a constrained route needs: where the wire enforces the
schema the correction is itself guaranteed valid, so a second attempt is already
generous. The budget is for the routes that cannot, where a model writing a
confirmation sentence instead of the document is the ordinary failure."""


class SchemaViolation(Exception):
    """A reply that is not the shape it was asked for, with what was wrong.

    Carries the violations rather than a sentence, because the next thing that
    happens to it is being read back to the model — "missing required field(s):
    severity" is a correction it can act on, where "invalid response" is not.
    """

    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations) or "the reply was not the requested shape")
        self.violations = violations


def structural_warning(schema: dict[str, Any]) -> str | None:
    """Why this schema may not constrain anything, said before it is used.

    `required` naming a field that `properties` does not type is the one that
    matters: it survives validation here — the field is present, so nothing
    complains — while producing a grammar that constrains nothing on the wire.
    A caller sees the same malformed answer three times and no reason.

    Unreachable from a caller that passes a pydantic model, which is every caller
    in this tree; it is here for the dict schemas a row may bring.
    """
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(required, list):
        return None
    typed = properties if isinstance(properties, dict) else {}
    untyped = [
        str(name)
        for name in required
        if not isinstance(typed.get(str(name)), dict) or "type" not in typed[str(name)]
    ]
    if not untyped:
        return None
    return (
        f"required field(s) {', '.join(untyped)} have no typed `properties` entry, "
        "so a grammar built from this schema will not constrain the reply"
    )


def _object_in(text: str) -> Any:
    """The JSON value in a reply, tolerating a fence or a sentence around it.

    Kept even where the wire enforces the schema, because "enforced" is a
    property of the *route* and this runs on every route. It is a fallback now
    rather than the whole mechanism, which is the difference P7-17 is about.
    """
    for candidate in (text.strip(), *_braced(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise SchemaViolation(["the reply is not JSON"])


def _braced(text: str) -> list[str]:
    start, end = text.find("{"), text.rfind("}")
    return [text[start : end + 1]] if start != -1 and end > start else []


def validated_shape(text: str, shape: type[BaseModel] | dict[str, Any]) -> Any:
    """The reply as a validated value, or `SchemaViolation` naming what is wrong.

    `shape` is the same union `ToolOutput.schema` takes, and for the same reason:
    a pydantic model is one declaration that produces both the wire schema and
    the validator, so a caller that has one does not get a *second*, weaker check
    here and then the real one at home. Given a model it returns the instance;
    given a dict it returns the dict it validated.
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
    stream: Callable[[GenerateOptions], Awaitable[Any]],
    options: GenerateOptions,
    shape: type[Shape],
    *,
    enforced: bool = False,
    attempts: int = ATTEMPTS,
) -> Shape:
    """One call that must come back as `shape`, corrected once if it does not.

    `stream` is `ctx.llm.stream`, passed rather than reached for, so this stays a
    function over the seam instead of a second thing that resolves it. `shape` is
    a pydantic model: it becomes the wire schema *and* the validator, so the
    caller gets a typed instance and the reply is checked once rather than
    half-checked here and properly at home.

    `enforced` is `ResolvedModel.structured_output` — whether this route builds a
    grammar from the schema. It buys one fewer attempt, because a constrained
    correction is itself guaranteed valid; and a violation from a route that
    claimed enforcement is logged as the adapter or provider bug it is, rather
    than reading as a model that would not comply.

    The correction turn **offers no tools**. That is not politeness: a tool call
    is not a document in the schema's shape, so a route asked for both has to
    break one promise — which is why `options` carrying tools is refused outright
    rather than being quietly stripped.

    `BlockAssembler` is the loop's own assembly, so a caller cannot disagree with
    the transcript about what a reply said, and `text_of` drops reasoning blocks
    rather than handing them to a JSON parser.
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
            failure = assembler.finish.failure
            raise SchemaViolation([failure.message if failure else "the model call failed"])
        reply = text_of(assembler.blocks()).strip()
        try:
            settled: Shape = validated_shape(reply, shape)
            return settled
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
    """The follow-up: what came back, and exactly what was wrong with it.

    `replace` rather than a field-by-field rebuild, which is what a frozen
    dataclass is for — the hand-written version had already dropped `stop`, and
    would have dropped every field added after it.
    """
    return replace(
        options,
        messages=(
            *options.messages,
            create_message(
                role="assistant",
                content=[{"type": "text", "text": reply}],
                # The route that said it, because `ModelSource` names one — this
                # message is the model's own reply being handed back to it, and
                # attributing it to anything else would be the falsehood the
                # source field exists to prevent.
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
