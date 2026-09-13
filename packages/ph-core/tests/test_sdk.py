"""P6-02 — the `tools:sdk` listing, in both languages it ships.

Under `mode: code` this text *is* the tool surface: the model is handed one
callable and this block, and everything it believes it can call it believes from
here. `test_code_mode` proves the section is assembled and mounted; what it does
not do is read the syntax, and the Python renderer being exercised there is why
the TypeScript one had never run at all.

That asymmetry is the risk the module's own docstring names — a model given
TypeScript signatures for a Python runtime writes code that cannot run, and the
failure looks like a model problem. A renderer nothing reads is one that can
drift into exactly that.

**No shipped profile can select the TypeScript one**, since the renderer is
chosen by the runtime's `language` and every runtime pH ships is Python. These
tests therefore pin a path that is registered and unreachable — deliberately, so
the runtime that makes it reachable does not inherit a renderer nothing ever ran.
`docs/dev-notes/typescript-code-runtime.md` records what that runtime needs.
"""

from __future__ import annotations

from typing import Any

from ph.seams.code_runtime import CodeBinding, CodeBindingNamespace
from ph.tools.sdk import code_only_rule, render_python_sdk, render_typescript_sdk


def _binding(name: str, description: str = "", **properties: Any) -> CodeBinding:  # noqa: ANN401
    """One binding whose parameters are spelled the way a tool schema spells them."""
    required = [key for key, definition in properties.items() if definition.pop("required", False)]
    return CodeBinding(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
    )


def _namespace(
    *bindings: CodeBinding,
    name: str = "tools",
    description: str = "",
) -> Any:  # noqa: ANN401
    return CodeBindingNamespace(name=name, bindings=bindings, description=description)


# ---------------------------------------------------------------- python --


def test_the_python_block_is_the_signature_a_program_would_write() -> None:
    """Name, order, types and optionality, in the runtime's own language.

    `= ...` on the optional argument is doing real work: without it the model
    reads every parameter as mandatory and supplies a value it was not asked for,
    which the tool then has to interpret.
    """
    block = render_python_sdk(
        [
            _namespace(
                _binding(
                    "read",
                    "Read a file, or a window of one.",
                    path={"type": "string", "required": True},
                    limit={"type": "integer"},
                ),
                description="the governed tool surface",
            )
        ]
    )

    assert block.splitlines() == [
        "# tools — the governed tool surface",
        "async def tools.read(path: str, limit: int = ...) -> Any: ...",
        '    """Read a file, or a window of one."""',
    ]


def test_every_schema_type_has_a_python_spelling() -> None:
    """A type table the model reads as annotations, so a miss is a wrong one.

    `Any` is the fallback rather than an omission: a parameter rendered with no
    annotation reads as untyped, which is a claim, where `Any` reads as "this
    renderer does not know", which is the truth.
    """
    block = render_python_sdk(
        [
            _namespace(
                _binding(
                    "everything",
                    s={"type": "string"},
                    n={"type": "number"},
                    i={"type": "integer"},
                    b={"type": "boolean"},
                    o={"type": "object"},
                    a={"type": "array"},
                    z={"type": "null"},
                    u={"type": "invented"},
                )
            )
        ]
    )

    assert (
        "async def tools.everything(s: str = ..., n: float = ..., i: int = ..., b: bool = ..., "
        "o: dict = ..., a: list = ..., z: None = ..., u: Any = ...) -> Any: ..." in block
    )


def test_a_nullable_type_renders_as_the_type_it_can_be() -> None:
    """`["string", "null"]` is how a schema spells optional, and it is the shape
    pydantic emits for `str | None` — so a renderer that took the first member
    would annotate half the harness's own tools as `None`."""
    block = render_python_sdk([_namespace(_binding("f", path={"type": ["null", "string"]}))])

    assert "path: str = ..." in block


def test_a_binding_with_no_parameters_and_no_summary_is_still_callable() -> None:
    """The empty signature has to render, not collapse: a namespace whose
    bindings all take nothing is exactly the shape of a status or list call."""
    block = render_python_sdk([_namespace(_binding("now"), name="clock")])

    assert block == "# clock\nasync def clock.now() -> Any: ..."


def test_only_the_first_line_of_a_description_reaches_the_listing() -> None:
    """A tool description is written for the *native* surface and runs to
    paragraphs; the SDK is a listing, and a listing that inlined all of it would
    spend the context window this mode exists to save."""
    block = render_python_sdk(
        [_namespace(_binding("read", "Read a file.\n\nPrefer this over shelling out to cat."))]
    )

    assert '    """Read a file."""' in block
    assert "shelling out" not in block


def test_namespaces_are_listed_in_the_order_they_were_given() -> None:
    """`tools` first is the code-mode contract — it is the one namespace that is
    never optional — and the renderer must not reorder what the caller composed."""
    block = render_python_sdk([_namespace(_binding("a")), _namespace(_binding("b"), name="rlm")])

    assert block.index("# tools") < block.index("# rlm")
    assert "\n\n# rlm" in block, "namespaces are separated by a blank line"


# ------------------------------------------------------------ typescript --


def test_the_typescript_block_declares_an_object_the_program_can_call() -> None:
    """The other renderer, and the one nothing had read.

    A different language means a different shape, not a translated one: bindings
    hang off a `declare const`, optionality is `?` on the name rather than a
    default, the doc comment goes *above* the member, and arguments arrive as one
    object rather than positionally.
    """
    block = render_typescript_sdk(
        [
            _namespace(
                _binding(
                    "read",
                    "Read a file, or a window of one.",
                    path={"type": "string", "required": True},
                    limit={"type": "integer"},
                ),
                description="the governed tool surface",
            )
        ]
    )

    assert block.splitlines() == [
        "// tools — the governed tool surface",
        "declare const tools: {",
        "  /** Read a file, or a window of one. */",
        "  read(args: { path: string, limit?: number }): Promise<unknown>",
        "}",
    ]


def test_every_schema_type_has_a_typescript_spelling() -> None:
    """`integer` and `number` collapse, because TypeScript has one; `array`
    widens to `unknown[]` rather than `any[]`, which is the same honesty `Any` is
    in Python — and `unknown` is the fallback for a type this table does not
    know."""
    block = render_typescript_sdk(
        [
            _namespace(
                _binding(
                    "everything",
                    s={"type": "string"},
                    n={"type": "number"},
                    i={"type": "integer"},
                    b={"type": "boolean"},
                    o={"type": "object"},
                    a={"type": "array"},
                    z={"type": "null"},
                    u={"type": "invented"},
                )
            )
        ]
    )

    assert (
        "everything(args: { s?: string, n?: number, i?: number, b?: boolean, o?: object, "
        "a?: unknown[], z?: null, u?: unknown }): Promise<unknown>" in block
    )


def test_a_typescript_binding_with_no_summary_carries_no_comment() -> None:
    """An empty `/** */` above a member is noise in a block whose whole purpose
    is to be read quickly."""
    block = render_typescript_sdk([_namespace(_binding("now"), name="clock")])

    assert block.splitlines() == [
        "// clock",
        "declare const clock: {",
        "  now(args: {  }): Promise<unknown>",
        "}",
    ]


def test_both_renderers_close_every_namespace_they_open() -> None:
    """The TypeScript one has a closing brace to lose, which is the failure mode
    Python's indentation-free listing does not have: an unbalanced block is
    invalid syntax the model would be asked to write against."""
    block = render_typescript_sdk(
        [_namespace(_binding("a")), _namespace(_binding("b"), name="rlm")]
    )

    assert block.count("declare const") == block.count("\n}") == 2
    assert block.endswith("}"), "the listing does not trail a blank line"


# ------------------------------------------------------------------ rule --


def test_the_code_only_rule_names_the_transport_it_was_given() -> None:
    """Profiles rename the transport — the RLM bundle presents it as `ipython` —
    so a rule quoting a fixed name would tell the model to call something that is
    not there (C6). It also states the C3 consequence, because a refusal ending
    the whole program is the fact that decides whether a model checks first.
    """
    rule = code_only_rule("ipython")

    assert "`ipython`" in rule and "run_code" not in rule
    assert "fails the whole program" in rule
