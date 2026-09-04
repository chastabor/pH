"""P7-17 — asking a model for a value, and what happens when it answers prose.

Three claims, and they are deliberately separable because a route may honour one
and not the others.

**The schema rides the request.** A route that supports `response_format` builds
a grammar from it and cannot answer another shape; `ResolvedModel.structured_output`
is how a caller learns whether it got that, and it defaults to *no* so a caller
never assumes a guarantee it did not receive.

**Validation is on every route.** The wire's help is a bonus, not the mechanism —
which is what makes this honest on Anthropic, where there is no equivalent field
today.

**A near-miss is corrected, not declined.** "Return only JSON" is an instruction,
not a guarantee, and the reply that fails is sent back once with the specific
violation named rather than costing the whole call.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import BaseModel

from ph.llm.structured import (
    SchemaViolation,
    ask_for_shape,
    structural_warning,
    validated_shape,
)
from ph.llm.types import GenerateOptions, create_message
from ph.testing import text_chunks

pytestmark = pytest.mark.anyio


class Verdict(BaseModel):
    """The shape a caller asks for: one declaration, wire schema and validator.

    A pydantic model rather than a dict, which is what every caller in the tree
    passes — the dict path stays for a row that brings a foreign schema."""

    verdict: str
    score: float | None = None


SHAPE: dict[str, Any] = Verdict.model_json_schema()


def _options(**over: Any) -> GenerateOptions:
    base: dict[str, Any] = {
        "provider": "fake",
        "model": "fake-1",
        "messages": (
            create_message(
                role="user", content=[{"type": "text", "text": "judge it"}], source={"kind": "user"}
            ),
        ),
    }
    return GenerateOptions(**{**base, **over})


def _replies(*texts: str) -> Any:
    """A stream answering each call with the next text. Returns `(stream, seen)`.

    `text_chunks` is the shared quartet a text reply emits — spelling it out here
    would be a fourth copy of what a chunk stream looks like, and the one that
    stops matching when the vocabulary moves.
    """
    remaining = list(texts)
    seen: list[GenerateOptions] = []

    async def stream(options: GenerateOptions) -> Any:
        seen.append(options)

        async def chunks() -> Any:
            for chunk in text_chunks(remaining.pop(0) if remaining else ""):
                yield chunk

        return chunks()

    return stream, seen


# ------------------------------------------------------------- validation --


def test_a_model_shape_comes_back_as_the_model() -> None:
    """One declaration, one validation, a typed result.

    The caller used to get a dict here and re-validate it with its own model at
    home — two passes, the first of which never saw the `anyOf` a pydantic
    schema emits for an optional field, so the near-miss it was meant to correct
    died at the second instead."""
    settled = validated_shape('{"verdict": "keep", "score": 3}', Verdict)

    assert isinstance(settled, Verdict)
    assert (settled.verdict, settled.score) == ("keep", 3)


def test_a_dict_shape_still_returns_the_dict_it_validated() -> None:
    """The path a row bringing a foreign schema takes."""
    assert validated_shape('{"verdict": "keep"}', SHAPE) == {"verdict": "keep"}


def test_prose_around_the_document_is_read_through() -> None:
    """The tolerant read stays, as the fallback rather than the whole mechanism.

    A route that enforces nothing is still a route pH runs on, and a fence is the
    single most common near-miss.
    """
    assert validated_shape('Sure!\n```json\n{"verdict": "drop"}\n```', Verdict).verdict == "drop"


def test_a_missing_required_field_names_itself() -> None:
    """The violation is carried, not flattened to a sentence — the next thing that
    happens to it is being read back to the model."""
    with pytest.raises(SchemaViolation) as raised:
        validated_shape('{"score": 2}', SHAPE)

    assert any("verdict" in one for one in raised.value.violations)


def test_a_reply_that_is_not_json_at_all_is_a_violation() -> None:
    with pytest.raises(SchemaViolation):
        validated_shape("I'm afraid I can't do that.", SHAPE)


def test_a_json_value_that_is_not_an_object_is_refused() -> None:
    """A list parses as JSON and would otherwise validate as an empty object —
    which is how a proposal with no edits half-applies."""
    with pytest.raises(SchemaViolation, match="not a JSON object"):
        validated_shape("[1, 2, 3]", SHAPE)


# ------------------------------------------------------- the useless schema --


def test_a_schema_that_cannot_constrain_says_so_before_it_is_used() -> None:
    """The failure that costs three identical retries and explains nothing.

    A grammar converter needs a *type* per field to build a real constraint, so
    `required` without a matching `properties` entry degrades to a near-
    unconstrained grammar — the model returns whatever it likes and every
    correction fails the same way. Ported from OpenMono's executor, which
    documents exactly this and warns at step start.

    It passes validation here, which is the point: nothing else would catch it.
    """
    degenerate = {"type": "object", "required": ["severity", "file"]}

    assert validated_shape('{"severity": "high", "file": "x"}', degenerate) == {
        "severity": "high",
        "file": "x",
    }
    warning = structural_warning(degenerate)
    assert warning is not None and "severity" in warning and "file" in warning
    assert structural_warning(SHAPE) is None, "a pydantic-derived schema warns about nothing"


# ---------------------------------------------------------------- the wire --


# -------------------------------------------------------------- the retry --


async def test_a_near_miss_is_corrected_rather_than_failed() -> None:
    """The whole point: one bad reply costs a turn, not the call."""
    stream, seen = _replies("Certainly!", '{"verdict": "keep"}')

    assert (await ask_for_shape(stream, _options(), Verdict)).verdict == "keep"
    assert len(seen) == 2


async def test_the_correction_names_the_violation_and_offers_no_tools() -> None:
    """A tool call is not a document in the schema's shape.

    So a route asked for both would have to break one promise — the correction
    turn carries the schema and no tools, and says exactly what was wrong rather
    than "try again", which is a correction the model can act on.
    """
    stream, seen = _replies('{"score": 1}', '{"verdict": "keep"}')

    await ask_for_shape(stream, _options(), Verdict)

    correction = seen[1]
    assert correction.tools == ()
    assert correction.response_schema == SHAPE
    text = str(correction.messages[-1].content[0].text)
    assert "verdict" in text, "the specific violation, not a generic retry"
    # The model's own words are handed back as the model's, on the route that
    # said them — anything else would misattribute them.
    echoed = correction.messages[-2]
    assert echoed.role == "assistant" and echoed.source.provider == "fake"


async def test_the_correction_keeps_the_rest_of_the_request() -> None:
    """A retry is the same call again, not a new one built from three fields.

    The first version rebuilt `GenerateOptions` field by field and had already
    dropped `stop`; every field added after it would have gone the same way,
    silently, on the correction turn only. `dataclasses.replace` is what a frozen
    dataclass is for.

    Sabotage: rebuild the options by hand and the budget, the stop list and the
    purpose vanish from the second call.
    """
    stream, seen = _replies("nope", '{"verdict": "keep"}')

    await ask_for_shape(
        stream,
        _options(max_tokens=512, stop=("</done>",), purpose="refine", session_id="s1"),
        Verdict,
    )

    first, correction = seen
    for field in ("max_tokens", "stop", "purpose", "session_id", "system", "temperature"):
        assert getattr(correction, field) == getattr(first, field), field


async def test_a_model_that_never_complies_gives_up_with_the_reason() -> None:
    """Bounded, and the reason survives to the caller.

    A planner that keeps missing the shape must cost a fixed number of calls,
    not one per turn forever.
    """
    stream, seen = _replies("nope", "still nope", "nope again")

    with pytest.raises(SchemaViolation):
        await ask_for_shape(stream, _options(), Verdict, attempts=3)

    assert len(seen) == 3


async def test_structure_and_tools_are_refused_together() -> None:
    """Not stripped quietly — a caller that asked for both has a bug, and a
    silently dropped tool list is the kind that surfaces three layers away."""
    from ph.llm.types import ToolSchema

    with pytest.raises(ValueError, match="tools"):
        await ask_for_shape(
            _replies("{}")[0],
            _options(tools=(ToolSchema(name="t", description="d", parameters={}),)),
            Verdict,
        )


async def test_a_degenerate_schema_warns_on_the_call(caplog: Any) -> None:
    """Once, at the call — not on the third retry, where nobody is reading."""
    stream, _seen = _replies('{"severity": "high"}')

    with caplog.at_level(logging.WARNING, logger="ph.llm.structured"):
        # It *validates* — the field is present — which is exactly why nothing
        # else would catch a schema that constrains nothing on the wire.
        await ask_for_shape(stream, _options(), {"required": ["severity"], "type": "object"})

    assert any("will not constrain" in record.message for record in caplog.records)
