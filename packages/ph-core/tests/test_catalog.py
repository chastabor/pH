"""P6-02 — the config catalog, generated from the rows rather than written down.

A profile is two things: a list of rows, and a blob of config per row. `phern events`
made the first half enumerable years before the second, and the asymmetry showed
— the only way to learn what a row accepted was to open its module and read the
`Config` class, which is exactly the position a person is in when they are
already lost.

**The field prose is read from source on purpose, and the alternative is the
interesting part.** pydantic will collect attribute docstrings itself with
`use_attribute_docstrings`, and enabling that on `WireModel` would have been one
line. It is not enabled, because `WireModel` is also what model-facing **tool**
schemas are built from: these docstrings are internal rationale, often several
sentences of it, and switching the flag on would have spent a slice of every
request's context window explaining decisions no model can act on. So the catalog
does its own extraction, and the prose stays where it was written for.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field

from ph.cordis.catalog import _fields, config_catalog, field_docs, render_annotation
from ph.cordis.plugin import Affects
from ph.wire import WireModel

pytestmark = pytest.mark.anyio


class _Base(WireModel):
    inherited: int = 1
    """What a base class documents."""


class _Sample(_Base):
    plain: str = "x"
    """The prose under a field."""

    multi: bool = False
    """Prose that runs
    over two lines."""

    described: int = Field(default=2, description="An explicit description.")

    undocumented: float = 0.0

    required: str


def test_field_docs_reads_the_string_written_under_each_field() -> None:
    """The mechanism the catalog's prose column depends on."""
    docs = field_docs(_Sample)

    assert docs["plain"] == "The prose under a field."
    # Collapsed, because a table cell is one line and the source's wrapping is
    # an artefact of the editor rather than of the meaning.
    assert docs["multi"] == "Prose that runs over two lines."
    assert docs["inherited"] == "What a base class documents.", "a base class was not walked"
    assert "undocumented" not in docs
    assert "required" not in docs


def test_a_model_whose_source_cannot_be_read_contributes_no_prose() -> None:
    """A blank cell beats a traceback.

    A model built at runtime has no source for `inspect.getsource` to find. The
    catalog is consulted when something is already confusing, so one row whose
    prose is unavailable must not take the other eighty-seven down with it.
    """
    dynamic = type("_Dynamic", (WireModel,), {"__annotations__": {"field": int}, "field": 1})

    assert field_docs(dynamic) == {}


@pytest.mark.parametrize(
    ("annotation", "rendered"),
    [
        (str, "str"),
        (str | None, "str | null"),
        (list[str], "list[str]"),
        (dict[str, int], "dict[str, int]"),
        (Literal["a", "b"], "Literal[a, b]"),
        (list[str] | None, "list[str] | null"),
        # A tuple of any length. `str(...)` is `Ellipsis`, which nobody writes.
        (tuple[str, ...], "tuple[str, ...]"),
        (tuple[str | int, ...] | None, "tuple[str | int, ...] | null"),
    ],
)
def test_types_render_the_way_a_profile_would_write_them(annotation: object, rendered: str) -> None:
    """The audience is somebody about to type the value into YAML.

    `<class 'str'>` and `typing.Optional[str]` both make that person translate,
    and `null` rather than `None` because the file they are editing is YAML.
    """
    assert render_annotation(annotation) == rendered


def test_the_catalog_reports_a_row_that_cannot_be_imported_rather_than_dropping_it() -> None:
    """A row that vanished from the catalog is the least helpful possible answer.

    `phern doctor` prints a failing section in place for the same reason: the
    report is read *because* something is wrong, and silence about the broken
    row is indistinguishable from silence about a row that does not exist.
    """
    catalog = config_catalog(group="ph.plugins.nonexistent")
    assert catalog == [], "an unregistered group should be empty, not an error"

    real = {entry["name"]: entry for entry in config_catalog()}
    assert not [entry for entry in real.values() if "error" in entry]
    assert real["workspace-git-worktree"]["module"] == "ph.seams.workspace_git"


def test_every_row_reports_what_it_injects_and_what_it_configures() -> None:
    """The two facts `PluginSpec` already carries, which is why this cannot drift.

    A field added to a row's `Config` shows up here with nobody remembering to
    write it down — the property that makes a generated catalog worth having
    over a hand-kept table.
    """
    catalog = {entry["name"]: entry for entry in config_catalog()}

    schedule = catalog["schedule"]
    ((index,),) = (schedule["config"],)
    assert (index["name"], index["type"], index["default"]) == ("index", "bool", "True")
    assert index["required"] is False
    assert "survives a restart" in index["doc"]

    # A row that takes no config is listed with an empty option set rather than
    # omitted: "no options" and "no such row" are different answers.
    assert catalog["diagnostics"]["config"] == []
    assert catalog["workspace-git-worktree"]["injects"] == ["workspace", "subprocess"]


class _Leaf(WireModel):
    """What one leaf is."""

    depth: int = 0
    """How deep the leaf is."""


class _Tree(WireModel):
    """A model that holds itself."""

    children: list[_Tree] = Field(default_factory=list)
    leaf: _Leaf | None = None


class _Nesting(WireModel):
    one: _Leaf = Field(default_factory=_Leaf)
    many: list[_Leaf] = Field(default_factory=list)
    fixed: tuple[_Leaf, ...] = ()
    named: dict[str, _Leaf] = Field(default_factory=dict)
    """The leaves by name."""
    maybe: _Leaf | None = None
    either: _Leaf | _Tree | None = None
    tree: _Tree = Field(default_factory=_Tree)


def test_the_options_of_a_nested_model_are_listed_under_the_path_a_profile_writes() -> None:
    """An option whose type holds another model is followed by the options of
    that model. A list of models was one line before, and the catalog said
    nothing about what an entry accepts (`profiles` of `llm-openai-compatible`).

    Sabotage: stop the descent in `_fields` and only the seven top lines remain.
    """
    fields = {field["name"]: field for field in _fields(_Nesting)}

    assert list(fields) == [
        "one",
        "one.depth",
        "many",
        "many[].depth",
        "fixed",
        "fixed[].depth",
        "named",
        "named.<key>.depth",
        "maybe",
        "maybe.depth",
        "either",
        "either.depth",
        "either.children",
        "either.leaf",
        "either.leaf.depth",
        "tree",
        # A model that holds itself is listed one time.
        "tree.children",
        "tree.leaf",
        "tree.leaf.depth",
    ]
    assert fields["many[].depth"]["parent"] == "many[]"
    assert fields["many"]["parent"] is None
    assert fields["many[].depth"]["doc"] == "How deep the leaf is."
    # A field with no docstring borrows the docstring of the one model that its
    # type holds, through any container. Its own docstring is preferred, and a
    # type that holds two models borrows nothing.
    assert fields["many"]["doc"] == "What one leaf is."
    assert fields["named"]["doc"] == "The leaves by name."
    assert fields["either"]["doc"] == ""


def _needs_the_other_fields(data: dict[str, object]) -> int:
    return len(data)


class _Shaped(WireModel):
    on_mount: bool = True
    paths: list[str] = Field(default_factory=lambda: ["."])


class _Defaults(WireModel):
    plain: int = 3
    built: list[str] = Field(default_factory=lambda: ["KEY", "TOKEN"])
    empty: dict[str, int] = Field(default_factory=dict)
    shaped: _Shaped = Field(default_factory=_Shaped)
    several: tuple[_Shaped, ...] = (_Shaped(),)
    unbuildable: int = Field(default_factory=_needs_the_other_fields)
    required: str


def test_a_default_that_a_factory_builds_is_printed() -> None:
    """A field with `default_factory` showed a blank default, and a blank cell
    reads as "no default". The factory is called and its value is printed.

    A config model is printed as the mapping that a profile writes, with the
    keys of a profile (`onMount`). A factory that needs the other fields cannot
    be called here, and its cell stays blank. A required field has no default.

    Sabotage: report `None` for every field that has a `default_factory`, as
    `_fields` did, and `built`, `empty` and `shaped` are blank.
    """
    defaults = {field["name"]: field["default"] for field in _fields(_Defaults)}

    assert defaults["plain"] == "3"
    assert defaults["built"] == "['KEY', 'TOKEN']"
    assert defaults["empty"] == "{}"
    assert defaults["shaped"] == "{'onMount': True, 'paths': ['.']}"
    assert defaults["several"] == "({'onMount': True, 'paths': ['.']},)"
    assert defaults["unbuildable"] is None
    assert defaults["required"] is None
    # The options of the nested model keep their own defaults.
    assert defaults["shaped.onMount"] == "True"
    assert defaults["shaped.paths"] == "['.']"


def test_a_shipped_row_lists_the_options_of_its_nested_models() -> None:
    """The descent on a real row: a route of the `models` row holds a `Sampling`."""
    catalog = {entry["name"]: entry for entry in config_catalog()}
    names = [field["name"] for field in catalog["models"]["config"]]

    assert "models" in names
    assert "models.<key>.provider" in names
    assert "models.<key>.sampling.temperature" in names


# A row's kind decides which configuration may set it — the session profile, the
# TUI's, or the daemon's — so a family whose kind follows from its name is pinned
# by that name. A new `tui-*` row declared `environment` out of habit would put a
# screen layout into the session log's audit; a new `*-invariant` declared the
# same would let a person's profile switch off a check the host relies on.
_FAMILIES: tuple[tuple[str, str, Affects], ...] = (
    ("tui-", "", "presentation"),
    ("", "-invariant", "deployment"),
    ("session-persistence-", "", "deployment"),
    ("session-telemetry", "", "deployment"),
)


def test_every_shipped_row_says_which_configuration_owns_it() -> None:
    """The gate `affects` exists for: no shipped row is undeclared, and none of
    the named families is declared as the wrong kind.

    Undeclared or unknown is refused at the loader's door, so a row that forgot
    would show here as an error entry rather than as a missing key. The families are read
    off the names because that is how a person finds them — nobody reads
    `affects` to learn that `tui-status` is the TUI's.
    """
    catalog = config_catalog()

    assert not [entry["name"] for entry in catalog if "error" in entry]
    misfiled = {
        entry["name"]: entry["affects"]
        for entry in catalog
        for prefix, suffix, kind in _FAMILIES
        if entry["name"].startswith(prefix)
        and entry["name"].endswith(suffix)
        and entry["affects"] != kind
    }
    assert not misfiled, f"declared as the wrong kind for their family: {misfiled}"
    # Each family matched something, so a rename that emptied one is a failure
    # here and not a rule quietly checking nothing.
    for prefix, suffix, _ in _FAMILIES:
        assert any(
            entry["name"].startswith(prefix) and entry["name"].endswith(suffix) for entry in catalog
        ), f"no shipped row matches {prefix}*{suffix}"
