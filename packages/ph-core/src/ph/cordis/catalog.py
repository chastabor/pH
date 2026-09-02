"""The config catalog: what every registered row accepts, read off the rows (P6-02).

A profile is a list of rows and a blob of config per row, and until now the only
way to learn what a row accepted was to open its module and read the `Config`
class. That is the same failure `ph events` exists to fix one registry over: a
hand-kept table of options drifts from the code the moment somebody adds a field,
and the person it drifts under is the one editing a YAML file with no schema.

**Generated from `PluginSpec`, so it cannot drift.** The plugin decorator already
carries the two facts a catalog needs — `config_model` and `inject` — and the
entry-point group already enumerates every row a deployment could name, including
ones from third-party wheels. Nothing here is written down twice.

**Field prose is read from the source, not from pydantic.** These models write
their documentation as a string under the field, and pydantic can collect that
with `use_attribute_docstrings` — which is deliberately *not* enabled on
`WireModel`. Turning it on would put these strings into `model_json_schema()`,
and that is what a model-facing **tool** schema is built from; the prose here is
internal rationale ("outside the repository on purpose: a worktree inside `base`
would be walked by the agent's own `glob`") and spending a context window on it
would be a cost paid on every request to explain a decision no model can act on.
Reading it here keeps it where it was written for: a person asking what a row
takes.

@module ph.cordis.catalog
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
import typing
from itertools import pairwise
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .loader import ENTRY_POINT_GROUP, entry_point_targets, resolve_plugin
from .plugin import normalize_plugin

__all__ = ["config_catalog", "field_docs", "render_annotation"]


def field_docs(model: type[BaseModel]) -> dict[str, str]:
    """Field name → the docstring written beneath it, walking the MRO.

    Base classes first, so a subclass that redeclares a field documents it.
    A class whose source cannot be read — an interactively defined model, a
    zipped wheel — contributes nothing rather than raising: a catalog that
    refused to print because one row's prose was unavailable would be worse at
    its job than one with a blank cell.
    """
    docs: dict[str, str] = {}
    for klass in reversed(model.__mro__):
        if klass is BaseModel or not issubclass(klass, BaseModel):
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue
        (definition,) = (node for node in tree.body if isinstance(node, ast.ClassDef))
        body = definition.body
        for statement, following in pairwise(body):
            if not isinstance(statement, ast.AnnAssign) or not isinstance(
                statement.target, ast.Name
            ):
                continue
            if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                text = following.value.value
                if isinstance(text, str):
                    docs[statement.target.id] = " ".join(text.split())
    return docs


def render_annotation(annotation: Any) -> str:
    """A type as a person would write it, not as `repr` prints it.

    `typing.get_type_hints`-shaped output — `str | None`, `list[str]` — because
    the audience is somebody about to write the value in YAML, and
    `typing.Optional[str]` and `<class 'str'>` both make them translate.
    """
    if annotation is None or annotation is type(None):
        return "null"
    if isinstance(annotation, type):
        return annotation.__name__
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is None:
        return str(annotation).replace("typing.", "")
    if origin is typing.Union or origin is types.UnionType:
        # Both spellings reach here: `Optional[str]` carries `typing.Union`, and
        # `str | None` carries `types.UnionType`. A profile writes neither, so
        # both render as the `|` a person would read.
        return " | ".join(render_annotation(one) for one in args)
    rendered = ", ".join(render_annotation(one) for one in args)
    name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    return f"{name}[{rendered}]" if rendered else name


def _fields(model: type[BaseModel]) -> list[dict[str, Any]]:
    """One row per config field: the name a profile writes, and what it takes."""
    docs = field_docs(model)
    fields: list[dict[str, Any]] = []
    for name, field in model.model_fields.items():
        required = field.is_required()
        fields.append(
            {
                # The alias is what a profile actually writes — `wire_alias`
                # makes these camelCase — so it leads. A catalog printing the
                # Python name would document a key the loader rejects.
                "name": field.alias or name,
                "attribute": name,
                "type": render_annotation(field.annotation),
                "required": required,
                "default": None
                if required or field.default is PydanticUndefined
                else repr(field.default),
                "doc": field.description or docs.get(name, ""),
            }
        )
    return fields


def config_catalog(*, group: str = ENTRY_POINT_GROUP) -> list[dict[str, Any]]:
    """Every registered row, its injected services, and the config it accepts.

    In name order, one entry per row, whether or not it takes config — a row
    with no `Config` is a fact worth printing, since "this row has no options"
    and "I could not find this row's options" are the two answers a person is
    choosing between when they go looking.

    A row that cannot be imported is reported with its error rather than
    skipped, for the reason `ph doctor` reports a failing section in place: the
    catalog is consulted when something is already confusing, and a row that
    silently vanished from it is the least helpful possible response.
    """
    catalog: list[dict[str, Any]] = []
    for name, target in sorted(entry_point_targets(group).items()):
        entry: dict[str, Any] = {"name": name, "module": target.partition(":")[0]}
        try:
            spec = normalize_plugin(resolve_plugin(name))
        except Exception as error:
            catalog.append({**entry, "error": f"{type(error).__name__}: {error}", "config": []})
            continue
        entry["injects"] = list(spec.inject)
        entry["config"] = [] if spec.config_model is None else _fields(spec.config_model)
        catalog.append(entry)
    return catalog
