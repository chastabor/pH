"""P6-02 — the JSON-Schema subset pH validates, and the honesty around its edges.

This validator is what stands between the harness and a schema **pH did not
author**: an MCP server's tool declaration, a subagent's declared shape. The
pydantic path needs no tests of its own — pydantic is the validator there, and a
test of it would be a test of pydantic — so everything here is the raw-dict path,
which had none at all.

Two properties carry the module, and both are about what it does *not* do:

* **A keyword outside the subset is ignored, never treated as satisfied.** The
  failure that matters is the quiet one — `allOf` accepted as though checked
  means a tool believes an argument was validated against a rule nothing read.
  `unsupported_keywords` is how a caller finds that out in advance, so it has to
  see into the places a schema nests.
* **A malformed schema refuses to enforce rather than refusing the value.** An
  uncompilable `pattern`, a `$ref` that goes nowhere, a `type` nobody has heard
  of: each leaves the value unjudged instead of inventing a violation. A
  validator that failed closed on its own confusion would reject the tool calls
  of every server whose dialect is wider than this one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ph.tools.json_schema import (
    SUPPORTED_KEYWORDS,
    schema_of,
    unsupported_keywords,
    validate_json_schema_value,
)


def _violations(schema: dict[str, Any], value: Any) -> list[str]:
    return validate_json_schema_value(schema, value)


# ------------------------------------------------------------ the reporting --


def test_a_keyword_outside_the_subset_is_reported_rather_than_enforced() -> None:
    """The module's central promise, from both sides.

    `allOf` is not in `SUPPORTED_KEYWORDS`, so nothing checks it — and a value
    that plainly violates it produces no complaint. That is the correct outcome
    and also the dangerous one, which is why the same schema names the keyword
    when asked.
    """
    schema = {"type": "object", "allOf": [{"required": ["never"]}]}

    assert "allOf" not in SUPPORTED_KEYWORDS
    assert _violations(schema, {}) == [], "an unenforced keyword must not be half-enforced"
    assert unsupported_keywords(schema) == {"allOf"}


def test_unsupported_keywords_looks_everywhere_a_schema_nests() -> None:
    """A report that only read the top level would be worse than none.

    A schema's teeth are usually in its properties, and `$defs`/`definitions` are
    where a `$ref` sends the reader — so a keyword hiding one hop down is exactly
    the one a caller would have been reassured about.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "format": "email"}},
        "items": {"not": {"type": "null"}},
        "additionalProperties": {"oneOf": []},
        "$defs": {"D": {"type": "object", "patternProperties": {}}},
        "definitions": {"E": {"type": "object", "unevaluatedItems": True}},
    }

    assert unsupported_keywords(schema) == {
        "format",
        "not",
        "oneOf",
        "patternProperties",
        "unevaluatedItems",
    }


def test_a_schema_that_is_not_a_dict_reports_nothing() -> None:
    """`True` is a legal JSON Schema meaning "anything". There is no keyword to
    report and nothing to enforce, so both answers are empty rather than an
    error — this is asked about schemas pH did not write."""
    assert unsupported_keywords(True) == set()
    assert _violations(True, {"anything": 1}) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------- the types --


def test_every_declared_type_name_is_checked() -> None:
    """The table, exercised as a table — including the two `bool` traps.

    `True` is an `int` in Python and `1` is not a `bool`, so a naive `isinstance`
    makes `{"type": "integer"}` accept `True` and `{"type": "boolean"}` accept
    `1`. Both would let a tool receive an argument of the wrong kind from a
    server whose schema was right.
    """
    accepted = {
        "null": None,
        "boolean": True,
        "string": "s",
        "number": 1.5,
        "integer": 3,
        "object": {},
        "array": [],
    }
    for name, value in accepted.items():
        assert _violations({"type": name}, value) == [], name

    assert _violations({"type": "integer"}, True) == ["<root>: expected integer"]
    assert _violations({"type": "number"}, True) == ["<root>: expected number"]
    assert _violations({"type": "boolean"}, 1) == ["<root>: expected boolean"]


def test_a_type_union_passes_when_any_member_matches() -> None:
    """`["string", "null"]` is how every schema spells an optional field."""
    schema = {"type": ["string", "null"]}

    assert _violations(schema, "s") == [] and _violations(schema, None) == []
    assert _violations(schema, 3) == ["<root>: expected string or null"]


def test_a_type_nobody_has_heard_of_leaves_the_value_alone() -> None:
    """Fail *open* on an unknown dialect, which is the module's whole posture.

    A draft-specific or vendor type name pH does not implement must not become a
    violation, or every call from a server using one is refused for a rule this
    validator never read. The value goes unjudged and `unsupported_keywords` is
    where the gap is reported.
    """
    assert _violations({"type": "integer64"}, "not a number at all") == []


def test_a_failed_type_check_stops_before_the_rest() -> None:
    """One complaint per value, not a cascade.

    A string where an object was declared fails `required`, `properties` and
    `additionalProperties` too, and reporting all four describes one mistake four
    times — the model then has to work out that they are the same thing.
    """
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}

    assert _violations(schema, "nope") == ["<root>: expected object"]


# --------------------------------------------------------------- the values --


def test_const_and_enum_name_what_was_expected() -> None:
    assert _violations({"const": "go"}, "stop") == ["<root>: must equal 'go'"]
    assert _violations({"const": "go"}, "go") == []
    assert _violations({"enum": ["a", "b"]}, "c") == ["<root>: must be one of ['a', 'b']"]
    assert _violations({"enum": ["a", "b"]}, "b") == []


def test_every_numeric_bound_compares_in_its_own_direction() -> None:
    """Four keywords, two of them strict, and the strictness is the point.

    `exclusiveMinimum` written as `minimum` accepts the boundary value, which is
    the one value the schema's author took the trouble to exclude.
    """
    assert _violations({"minimum": 5}, 4) == ["<root>: fails minimum 5"]
    assert _violations({"minimum": 5}, 5) == []
    assert _violations({"maximum": 5}, 6) == ["<root>: fails maximum 5"]
    assert _violations({"maximum": 5}, 5) == []
    assert _violations({"exclusiveMinimum": 5}, 5) == ["<root>: fails exclusiveMinimum 5"]
    assert _violations({"exclusiveMinimum": 5}, 6) == []
    assert _violations({"exclusiveMaximum": 5}, 5) == ["<root>: fails exclusiveMaximum 5"]
    assert _violations({"exclusiveMaximum": 5}, 4) == []


def test_a_non_finite_number_is_refused_wherever_it_appears() -> None:
    """JSON has no `NaN` or `Infinity`, so one here came from a producer that
    invented them — and it would be rejected at the provider, or worse, silently
    reshaped. `snapshotJsonValue` refuses them entering the log for the same
    reason (A1); this is the same refusal on the way out."""
    assert _violations({"type": "number"}, float("nan")) == ["<root>: must be a finite number"]
    assert _violations({"type": "number"}, float("inf")) == ["<root>: must be a finite number"]


def test_a_bound_that_is_not_a_number_is_ignored() -> None:
    """`{"minimum": true}` is a schema bug, and `True >= 1` would otherwise make
    it a *value* bug reported against the caller."""
    assert _violations({"minimum": True}, 0) == []
    assert _violations({"maximum": "five"}, 6) == []


def test_string_length_and_pattern() -> None:
    assert _violations({"minLength": 2}, "a") == ["<root>: fails minLength 2"]
    assert _violations({"maxLength": 2}, "abc") == ["<root>: fails maxLength 2"]
    assert _violations({"pattern": "^a+$"}, "bbb") == ["<root>: does not match '^a+$'"]
    assert _violations({"pattern": "^a+$"}, "aaa") == []


def test_a_pattern_that_will_not_compile_is_not_enforced() -> None:
    """A schema pH cannot compile is one it does not enforce.

    The alternative is refusing every value against a regex whose author meant a
    different dialect — a broken schema turning into an unusable tool, when the
    honest answer is that the constraint went unchecked.
    """
    assert _violations({"pattern": "([unclosed"}, "anything") == []


def test_array_bounds_and_per_item_paths() -> None:
    """The index is in the path, because "one of these is wrong" is not
    actionable when the array has forty entries."""
    assert _violations({"minItems": 2}, [1]) == ["<root>: fails minItems 2"]
    assert _violations({"maxItems": 1}, [1, 2]) == ["<root>: fails maxItems 1"]
    assert _violations({"items": {"type": "string"}}, ["a", 2, "c"]) == ["[1]: expected string"]


# --------------------------------------------------------------- the objects --


def test_a_missing_required_property_is_named() -> None:
    schema = {"type": "object", "required": ["a", "b"], "properties": {}}

    assert _violations(schema, {"a": 1}) == ["<root>: missing required property 'b'"]


def test_additional_properties_false_refuses_what_it_did_not_declare() -> None:
    schema = {"type": "object", "properties": {"a": {}}, "additionalProperties": False}

    assert _violations(schema, {"a": 1, "b": 2}) == ["<root>: unexpected property 'b'"]
    assert _violations(schema, {"a": 1}) == []


def test_additional_properties_as_a_schema_validates_the_rest() -> None:
    """The map-shaped schema: named properties keep their own rules and
    everything else answers to one."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": {"type": "integer"},
    }

    assert _violations(schema, {"name": "x", "count": 2}) == []
    assert _violations(schema, {"name": "x", "count": "two"}) == ["count: expected integer"]


def test_a_nested_path_reads_as_the_caller_wrote_it() -> None:
    """Dotted for properties, bracketed for indexes, and no leading separator at
    the top level — the path is for a person reading a refusal."""
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            }
        },
    }

    assert _violations(schema, {"outer": {"items": ["a", 1]}}) == [
        "outer.items[1]: expected string"
    ]


# ----------------------------------------------------------------- the refs --


def test_a_local_ref_is_followed() -> None:
    """`$defs` plus `$ref` is what pydantic emits for any nested model, so a
    validator that did not follow one would check nothing below the first."""
    schema = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "$defs": {"Inner": {"type": "object", "required": ["id"]}},
    }

    assert _violations(schema, {"inner": {"id": 1}}) == []
    assert _violations(schema, {"inner": {}}) == ["inner: missing required property 'id'"]


def test_a_ref_that_goes_nowhere_leaves_the_value_alone() -> None:
    """Four ways to be unresolvable, one answer to all of them.

    A foreign `$ref` names a document pH does not have; a dangling one names a
    definition that is not there; a cycle would not terminate; a target that is
    not a schema cannot be applied. Each leaves the value unjudged, because the
    alternative — refusing it — invents a violation out of the schema's problem.
    """
    unresolvable: list[dict[str, Any]] = [
        {"$ref": "https://example.com/schema.json"},
        {"$ref": "#/$defs/Missing", "$defs": {}},
        {"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}},
        {"$ref": "#/$defs/A", "$defs": {"A": ["not", "a", "schema"]}},
    ]
    for schema in unresolvable:
        assert _violations(schema, {"anything": True}) == [], schema["$ref"]


def test_a_ref_chain_is_followed_to_the_end() -> None:
    """One hop at a time, so an alias of an alias still resolves."""
    schema = {
        "$ref": "#/$defs/A",
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}},
    }

    assert _violations(schema, 3) == ["<root>: expected string"]
    assert _violations(schema, "s") == []


# -------------------------------------------------------------- the pydantic --


class _Args(BaseModel):
    path: str
    count: int = Field(1, ge=1)


def test_a_pydantic_declaration_delegates_to_pydantic() -> None:
    """One definition produces the schema *and* the validator, which is the
    argument for preferring a model wherever the tool is written in Python.

    The violation is reformatted to this module's `location: message` shape, so a
    caller reading a list of them cannot tell which path produced it — which is
    what lets `ToolOutput` accept either declaration.
    """
    assert validate_json_schema_value(_Args, {"path": "a", "count": 2}) == []
    assert validate_json_schema_value(_Args, {"path": "a", "count": 0}) == [
        "count: Input should be greater than or equal to 1"
    ]
    (missing,) = validate_json_schema_value(_Args, {})
    assert missing.startswith("path: Field required")


def test_a_whole_value_of_the_wrong_shape_reads_as_root() -> None:
    """Pydantic reports an empty location for a non-dict input, and an empty
    location renders as nothing at all — so the message would have begun with a
    bare colon."""
    (violation,) = validate_json_schema_value(_Args, "not a mapping")

    assert violation.startswith("<root>: ")


def test_schema_of_answers_for_both_declaration_forms() -> None:
    """And the model's schema is built once: `model_json_schema()` is not
    memoized by pydantic and `ask_for_shape` asks for one per model request,
    where it was the largest single cost the structured path introduced."""
    raw = {"type": "object"}
    assert schema_of(raw) is raw, "a raw declaration is passed through, not copied"

    first = schema_of(_Args)
    assert first["properties"]["path"]["type"] == "string"
    assert schema_of(_Args) is first, "the schema is cached on the class"
