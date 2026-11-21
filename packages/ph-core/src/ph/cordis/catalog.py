"""The config catalog: what every registered row accepts, read from the rows (P6-02).

A profile is a list of rows with a blob of config for each row. Before this
catalog, the only way to learn what a row accepted was to open its module and
read the `Config` class. `phern events` exists to fix the same failure for a
different registry. A table of options kept by hand drifts from the code as soon
as somebody adds a field, and the one who suffers is the user who edits a YAML
file with no schema.

**The catalog is generated from `PluginSpec`, so it cannot drift.** The plugin
decorator already carries the facts that a catalog needs: `config_model`,
`inject`, and `affects`, which says which configuration the settings of the row
belong in. The entry-point group already enumerates every row that a deployment
could name, including rows from third-party wheels. Nothing here is written
down twice.

**Field prose is read from the source, not from pydantic.** These models write
their documentation as a string under the field. Pydantic can collect that
string with `use_attribute_docstrings`, which is deliberately *not* enabled on
`WireModel`. Enabling it would put these strings into `model_json_schema()`, and
a model-facing **tool** schema is built from that. The prose here is internal
rationale ("outside the repository on purpose: a worktree inside `base` would be
walked by the agent's own `glob`"). In a tool schema it would use part of the
context window on every request, to explain a decision that no model can act
on. Reading the prose here keeps it for the reader that it was written for: a
user who asks what a row takes.

@module ph.cordis.catalog
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
import typing
from collections.abc import Iterator, Mapping
from itertools import pairwise
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .loader import ENTRY_POINT_GROUP, entry_point_targets, resolve_plugin
from .plugin import normalize_plugin

__all__ = ["config_catalog", "field_docs", "render_annotation"]


def field_docs(model: type[BaseModel]) -> dict[str, str]:
    """Each field name mapped to the docstring written beneath it, over the whole MRO.

    Base classes are read first, so a subclass that declares a field again
    documents it. A class whose source cannot be read (a model defined
    interactively, a zipped wheel) contributes nothing and does not raise. A
    catalog that refused to print because the prose of one row was unavailable
    would do its job worse than a catalog with a blank cell.
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


def render_annotation(annotation: object) -> str:
    """A type as a person would write it, and not as `repr` prints it.

    The output has the shape of `typing.get_type_hints` text: `str | None`,
    `list[str]`. The reader is somebody who is about to write the value in YAML,
    and both `typing.Optional[str]` and `<class 'str'>` would make that reader
    translate.
    """
    if annotation is None or annotation is type(None):
        return "null"
    if annotation is Ellipsis:
        # `tuple[str, ...]`: a tuple of any length. `str(...)` is `Ellipsis`,
        # which nobody writes.
        return "..."
    if isinstance(annotation, type):
        return annotation.__name__
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is None:
        return str(annotation).replace("typing.", "")
    if origin is typing.Union or origin is types.UnionType:
        # Both spellings reach this branch: `Optional[str]` carries
        # `typing.Union`, and `str | None` carries `types.UnionType`. A profile
        # writes neither, so both render with the `|` that a person would read.
        return " | ".join(render_annotation(one) for one in args)
    rendered = ", ".join(render_annotation(one) for one in args)
    name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    return f"{name}[{rendered}]" if rendered else name


def _models_in(annotation: object, path: str = "") -> Iterator[tuple[str, type[BaseModel]]]:
    """Each config model inside the type of a field, with the path that a profile
    writes to reach an option of that model.

    The path follows the container: `` for the model itself or `Model | None`,
    `[]` for an entry of a list or a tuple, and `.<key>` for a value of a
    mapping. So `profiles: list[ProviderProfile]` gives `("[]", ProviderProfile)`,
    and the catalog names its options `profiles[].baseUrl`.
    """
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is None:
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            yield path, annotation
        return
    if origin is typing.Union or origin is types.UnionType:
        step = ""
    elif inspect.isclass(origin) and issubclass(origin, Mapping):
        step, args = ".<key>", args[1:]
    else:
        step = "[]"
    for one in args:
        yield from _models_in(one, path + step)


def _nested_doc(annotation: object) -> str:
    """The docstring of a nested config model, for a field that has none of its own.

    The line of an option whose type holds another model would otherwise say
    *nothing*. The obvious repair is a docstring on the field that restates the
    model, and that is the copy that this catalog exists to avoid. The claim of
    `phern config` is that it cannot drift from the code, and a hand-written
    paragraph beside the prose that it paraphrases is the first thing to drift.

    So a field with nothing of its own to say borrows the docstring of its
    model. A field that *does* have a docstring still wins, because some things
    are worth saying about a setting that are not true of its type. A type that
    holds two models borrows nothing, because neither docstring is about both.
    """
    models = [model for _, model in _models_in(annotation)]
    if len(models) != 1:
        return ""
    return " ".join((models[0].__doc__ or "").split())


def _default_of(field: FieldInfo) -> str | None:
    """The default of a config field as text, or `None` for a field that has none.

    A default that a factory builds (`Field(default_factory=list)`) is built
    here and printed. Without that, the catalog showed a blank cell for the
    option, and a blank cell reads as "no default". A factory that cannot be
    called with no arguments, or that raises, gives `None`: a blank cell is a
    worse answer than the value and a better one than no catalog.
    """
    if field.is_required():
        return None
    if field.default_factory is None:
        return None if field.default is PydanticUndefined else repr(_as_written(field.default))
    try:
        built = field.get_default(call_default_factory=True)
    except Exception:
        return None
    return repr(_as_written(built))


def _as_written(value: object) -> object:
    """A default in the form that a profile writes it. A config model becomes
    its mapping, with the keys that a profile uses (`onMount`, not `on_mount`).
    The `repr` of the model itself would name a Python class and Python
    attributes, and a profile contains neither."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, Mapping):
        return {key: _as_written(one) for key, one in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_as_written(one) for one in value)
    return value


def _fields(
    model: type[BaseModel], parent: str | None = None, within: tuple[type[BaseModel], ...] = ()
) -> list[dict[str, Any]]:
    """One row per config field: the name that a profile writes, and what the field takes.

    The rows descend. The options of a model inside the type of a field follow
    the row of that field, named by the path that a profile writes
    (`profiles[].baseUrl`, `models.<key>.sampling.temperature`). Each of them
    carries `parent`, the name of the option that holds it. Without the
    descent, an option such as `profiles: list[ProviderProfile]` was one line,
    and the catalog said nothing about what an entry accepts.

    `within` holds the models that enclose this one. A model that holds itself
    is listed one time and not without end.
    """
    docs = field_docs(model)
    fields: list[dict[str, Any]] = []
    for name, field in model.model_fields.items():
        required = field.is_required()
        # The alias is what a profile actually writes (`wire_alias` makes these
        # camelCase), so it is preferred. A catalog that printed the Python
        # name would document a key that the loader rejects.
        written = field.alias or name
        full = written if parent is None else f"{parent}.{written}"
        fields.append(
            {
                "name": full,
                "attribute": name,
                "parent": parent,
                "type": render_annotation(field.annotation),
                "required": required,
                "default": _default_of(field),
                "doc": field.description or docs.get(name) or _nested_doc(field.annotation),
            }
        )
        for path, nested in _models_in(field.annotation):
            if nested is not model and nested not in within:
                fields.extend(_fields(nested, f"{full}{path}", (*within, model)))
    return fields


def config_catalog(*, group: str = ENTRY_POINT_GROUP) -> list[dict[str, Any]]:
    """Every registered row: what it shapes, its injected services, and the config
    that it accepts.

    The entries are in name order, one per row, whether or not the row takes
    config. A row with no `Config` is a fact worth printing. "This row has no
    options" and "I could not find the options of this row" are the two answers
    that a user chooses between when they look.

    A row that cannot be imported is reported with its error and not skipped,
    for the reason that `phern doctor` reports a failing section in place. A user
    consults the catalog when something is already confusing, and a row that
    silently vanished from it would be the least helpful response.
    """
    catalog: list[dict[str, Any]] = []
    for name, target in sorted(entry_point_targets(group).items()):
        entry: dict[str, Any] = {"name": name, "module": target.partition(":")[0]}
        try:
            spec = normalize_plugin(resolve_plugin(name))
        except Exception as error:
            catalog.append({**entry, "error": f"{type(error).__name__}: {error}", "config": []})
            continue
        entry["affects"] = spec.affects
        entry["injects"] = list(spec.inject)
        entry["config"] = [] if spec.config_model is None else _fields(spec.config_model)
        catalog.append(entry)
    return catalog
