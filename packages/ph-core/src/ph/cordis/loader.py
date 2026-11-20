"""Rows in, plugin tree out.

A pH profile is a list of rows. Each row names one plugin module and carries
its config. A bundle contributes rows through a **patch**. A patch is either an
`insert:` of new rows or an entry that addresses one row by id. An entry by id
replaces the *whole* config of the row, disables the row, removes it, or sets
its `isolate:`. Layers apply in order, and the last write wins for each row.
`--dump-config` prints the composed result with the layer that each row came
from.

Two rules make this data and not code (D9, invariant I-8):

* The only interpolation is `${env:VAR:-default}`.
* YAML is parsed with a **safe** loader whose set of implicit resolvers is
  narrowed further. A `!!python/...`-style tag is therefore a load error and
  not a call. The `!!js` idiom of dsh is deliberately not ported.

@module ph.cordis.loader
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import NoReturn, assert_never

import yaml
from pydantic import ValidationError

from ..json import JsonObject, JsonValue, as_str
from .context import Context, ForkScope
from .errors import LoaderError
from .events import events
from .key import ServiceKey
from .plugin import CONFIGURED_IN, LOG_LAYER, Affects, LogLayer, normalize_plugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "Mount",
    "Profile",
    "ProfileDocument",
    "Row",
    "compose_rows",
    "entry_ids",
    "entry_point_targets",
    "import_plugin_modules",
    "interpolate",
    "load_profile_documents",
    "new_declarations",
    "resolve_entry_point",
    "resolve_plugin",
    "resolve_row",
    "safe_yaml_load",
    "sparse_entries",
]

log = logging.getLogger("ph.cordis")

ENTRY_POINT_GROUP = "ph.plugins"

_ENV_PATTERN = re.compile(
    r"\$\{env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>(?:[^{}]|\{[^{}]*\})*))?\}"
)
"""`${env:NAME}` or `${env:NAME:-default}`.

The character class of the default is **not** `[^}]*` (A10). A fallback in JSON
shape is an ordinary thing to write there, for example
`${env:PH_EXTRA:-{"tier":"none"}}`. The old class matched up to the *first*
`}`, so the row silently received `{"tier":"none"` and the text after it stayed
literal. Nobody writes a default with more than one level of balanced braces.
The pattern accepts one level, so it stays a regular expression and does not
become a parser.
"""
_PREDICATE_PATTERN = re.compile(r"^\$\{(?P<kind>platform|env):(?P<value>[^}]*)\}$")


events.declare(
    "profile/mounted",
    "serial",
    owner="ph.cordis",
    doc=(
        "A composed profile finished mounting. A listener that raises refuses the run; "
        "one that returns a value stops the chain, so a listener doing collection "
        "rather than refusal must return None."
    ),
)
"""The one moment at which a profile is whole and nothing has run yet.

The event has two uses, and the second arrived after the first (P4-13):

* A row may **refuse the deployment** that it is mounted in (E8,
  `containment.strict`).
* A row may **collect what the whole profile contains**: whether any subagent
  provider was mounted, or whether any skill was installed. Such a question has
  no final answer until this moment. `ctx.inject` cannot express it, because
  neither fact is a service key.

The two uses share one dispatch, so the rules of that dispatch apply to both.
`serial` stops at a return value that is not null, so a collector that returned
a value would silently skip the refusals registered after it. `serial` also
propagates exceptions, so a collector that raises stops the process. A collector
must therefore collect by side effect and return `None`. The refusals are then
the only listeners that can end a run.

The event is declared here and not in a seam, because the loader dispatches it.
A declaration in a module that the loader never imports has not happened by the
time the dispatch checks for it."""


class SafeRowLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` with every non-scalar implicit conversion removed.

    **This class deliberately does not use `CSafeLoader` of libyaml.**
    `CSafeLoader` is 8x faster on a real document, and it was tried. A subclass
    of it with the two customizations below makes `Resolver.resolve` return
    `None` for *every* node kind. Every tag then becomes undefined, and the
    first profile load fails with "could not determine a constructor for the tag
    None". The failure reproduces only when this module is imported while
    `coverage` is tracing: under the `--cov` gate of CI, at collection, and not
    in an ordinary run. Each ingredient works alone. A bare `CSafeLoader`
    subclass built under coverage parses, and so does one that carries the
    resolver rebuild. The interaction is therefore not understood. The speed is
    not worth a loader that fails only where it is measured.

    The base class already refuses `!!python/object`. This subclass also
    declines to convert timestamps and sexagesimals. A row value that looks
    like a date (`2026-08-26`) or like a time (`1:30`, which YAML 1.1 reads as
    the integer 90) stays the string that the author wrote. A config file is
    data, and a silent type change is the same class of surprise as evaluation.
    """


_UNRESOLVED = frozenset({"tag:yaml.org,2002:timestamp"})
"""Implicit tags this loader declines to apply. See `SafeRowLoader`."""

_NUMERIC = frozenset({"tag:yaml.org,2002:int", "tag:yaml.org,2002:float"})
"""Implicit tags that this loader applies only to a value with no colon.

PyYAML reads `1:30` as base 60, and gives the integer 90. The colon is in no
other form of a number, so a value that has one is not resolved as a number
and stays a string. See `SafeRowLoader`."""


def _kept(tag: str, regexp: re.Pattern[str]) -> re.Pattern[str]:
    """The pattern of one implicit resolver, as this loader applies it."""
    if tag not in _NUMERIC:
        return regexp
    return re.compile(f"(?!.*:){regexp.pattern}", regexp.flags)


# **The table is rebound on the subclass. It is never item-assigned into the
# inherited table.** `yaml_implicit_resolvers` is one dict that every loader
# defined by PyYAML shares. An earlier version of this code wrote
# `SafeRowLoader.yaml_implicit_resolvers[first] = ...`. That write reached the
# copy that `Resolver` holds, and took timestamps away from `yaml.safe_load`
# **process-wide**: for pH and for any library in the same interpreter.
# `add_implicit_resolver` of PyYAML copies before it writes for exactly this
# reason. This assignment is that copy, done one time.
SafeRowLoader.yaml_implicit_resolvers = {
    first: [(tag, _kept(tag, regexp)) for tag, regexp in resolvers if tag not in _UNRESOLVED]
    for first, resolvers in SafeRowLoader.yaml_implicit_resolvers.items()
}


def _reject_unknown_tag(loader: yaml.Loader, suffix: str, node: yaml.Node) -> NoReturn:
    raise LoaderError(
        f"pH config is data, not code: tag '!{suffix}' at line "
        f"{node.start_mark.line + 1} is not allowed"
    )


SafeRowLoader.add_multi_constructor("!", _reject_unknown_tag)
SafeRowLoader.add_multi_constructor("tag:", _reject_unknown_tag)


def safe_yaml_load(text: str, *, origin: str = "<string>") -> JsonValue:
    """Parse YAML with no code evaluation and no implicit date coercion.

    `SafeRowLoader` makes the result a JSON tree and not the `Any` of
    `yaml.load`. It has no custom tags and no implicit dates, so it can build
    only the things that `JsonValue` names. This function is the one YAML entry
    point, so the fact is stated here and not at each reader.
    """
    try:
        value: JsonValue = yaml.load(text, Loader=SafeRowLoader)
    except yaml.YAMLError as error:
        raise LoaderError(f"{origin}: {error}") from error
    return value


# --------------------------------------------------------------------- rows --


@dataclass(frozen=True, slots=True)
class Row:
    """One composed profile row."""

    id: str
    name: str
    config: JsonValue = None
    disabled: bool = False
    layer: str = ""
    """The profile document that contributed the current config of this row."""
    isolate: dict[str, JsonValue] | None = None
    """The row ids that this row wants private copies of. Each id maps to a config
    override or to `None`.

    This is the `isolate.fs` of dsh: an isolated realm for one service. Here it
    names **row ids** and not service keys, because a row does not declare what
    it provides (`provide` is a runtime call). So `fs` names the row whose
    `apply` provides `ctx.fs`, and the loader mounts a second copy of that row
    inside the realm of this row. The row id and the service key are sometimes
    the same (`fs` → `ctx.fs`, `tools` → `ctx.tools`). In `base.yaml` they often
    differ: `session` provides `ctx.sessions`, and `agent` provides
    `ctx.agents`. So this is a row id and not a service key. A row id is also
    the one of the two that can be checked at compose time."""

    def to_entry(self) -> dict[str, JsonValue]:
        """This row in the form that a profile document declares it, which is the
        form that the loader reads back.

        It is `to_dump` without the provenance. `layer` says where a row came
        from, and a document cannot say that about itself. A saved profile made
        of dumps would therefore be one that the loader refuses.
        """
        entry: dict[str, JsonValue] = {"id": self.id, "name": self.name}
        if self.config is not None:
            entry["config"] = self.config
        if self.disabled:
            entry["disabled"] = True
        if self.isolate is not None:
            # Always write the mapping. `isolate: [fs]` dumps as `{fs: null}`,
            # and `_as_isolate` reads that back to the same value. So there is
            # one dump shape.
            entry["isolate"] = dict(self.isolate)
        return entry

    def to_dump(self) -> dict[str, JsonValue]:
        """`to_entry` plus the layer that the row came from, for `--dump-config`."""
        return {**self.to_entry(), "layer": self.layer}


# ------------------------------------------------------------ interpolation --


def interpolate(value: object, env: Mapping[str, str] | None = None) -> object:
    """Expand `${env:VAR:-default}` through a value tree.

    A match of the whole string still gives a string, and no other type. pH does
    not guess numbers from the environment. A row whose meaning changes with an
    accidental `"0"` is exactly the failure that a typed config model exists to
    catch.
    """
    source = os.environ if env is None else env
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            resolved = source.get(name)
            if resolved is None:
                if default is None:
                    raise LoaderError(
                        f"config references ${{env:{name}}} but it is unset and "
                        "declares no :- default"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, list):
        return [interpolate(item, source) for item in value]
    if isinstance(value, dict):
        return {key: interpolate(item, source) for key, item in value.items()}
    return value


def evaluate_predicate(value: object, env: Mapping[str, str] | None = None) -> bool:
    """Resolve the `disabled:` field of a row.

    The field accepts a literal boolean or one of two closed predicates:
    `${platform:win32}` and `${env:VAR}` (true when the variable is set and not
    empty). Anything else is a config error, and not an expression to evaluate.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if not isinstance(value, str):
        raise LoaderError(f"disabled: must be a boolean or a predicate, got {value!r}")
    match = _PREDICATE_PATTERN.match(value.strip())
    if match is None:
        raise LoaderError(
            f'disabled: "{value}" is not a supported predicate; use a boolean, '
            "${platform:<name>} or ${env:VAR}"
        )
    kind, target = match.group("kind"), match.group("value")
    if kind == "platform":
        return sys.platform == target or (target == "win32" and os.name == "nt")
    source = os.environ if env is None else env
    return bool(source.get(target))


# ----------------------------------------------------------------- patching --


def _as_rows(entries: Iterable[JsonValue], layer: str) -> list[Row]:
    rows: list[Row] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise LoaderError(f"{layer}: a row must be a mapping, got {entry!r}")
        if "name" not in entry:
            raise LoaderError(f"{layer}: row {entry!r} has no name")
        unknown = set(entry) - {"id", "name", "config", "disabled", "isolate"}
        if unknown:
            raise LoaderError(f"{layer}: row {entry!r} has unknown keys {sorted(unknown)}")
        rows.append(
            Row(
                id=_id_of(entry),
                name=str(entry["name"]),
                config=entry.get("config"),
                disabled=evaluate_predicate(entry.get("disabled")),
                layer=layer,
                isolate=_as_isolate(entry.get("isolate"), layer),
            )
        )
    return rows


def _id_of(entry: Mapping[str, JsonValue]) -> str:
    """The id a row entry declares: its `id`, or its `name` when it gives none."""
    return str(entry.get("id") or entry.get("name") or "")


def _as_isolate(value: object, layer: str) -> dict[str, JsonValue] | None:
    """`isolate:` as a list of row ids, or as a mapping of row id to config override.

    The two spellings state one fact, and both normalize to the mapping. `[fs]`
    means "a private `fs` with the config of the `fs` row".
    `{fs: {root: /tmp/x}}` means "a private `fs` with a different root". The
    feature exists for the second case, because a private copy with an identical
    config is only a second instance.
    """
    if value is None:
        return None
    if isinstance(value, list):
        if not all(isinstance(one, str) for one in value):
            raise LoaderError(f"{layer}: isolate: must list row ids, got {value!r}")
        return dict.fromkeys(value)
    if isinstance(value, dict):
        if not all(isinstance(one, str) for one in value):
            raise LoaderError(f"{layer}: isolate: keys must be row ids, got {value!r}")
        return dict(value)
    raise LoaderError(f"{layer}: isolate: must be a list of row ids or a mapping, got {value!r}")


def _apply_patch(rows: list[Row], patch: JsonObject, layer: str) -> list[Row]:
    """Apply one patch entry to `rows` in place, and return the rows that it touched."""
    unknown = set(patch) - {"insert", "id", "config", "disabled", "remove", "isolate"}
    if unknown:
        raise LoaderError(f"{layer}: patch has unknown keys {sorted(unknown)}")
    if "insert" in patch:
        inserted = patch["insert"]
        if not isinstance(inserted, list):
            raise LoaderError(f"{layer}: insert: must be a list of rows")
        # There is no duplicate check here. `_check_unique_ids` runs over the
        # composed list and covers this case with a better message. It names
        # the layer that declared the id first, which a check local to the
        # insert cannot see.
        added = _as_rows(inserted, layer)
        rows.extend(added)
        return added
    row_id = patch.get("id")
    if not isinstance(row_id, str):
        raise LoaderError(f"{layer}: a patch must carry either insert: or id:")
    for index, row in enumerate(rows):
        if row.id != row_id:
            continue
        if patch.get("remove") is True:
            del rows[index]
            return [row]
        updated = row
        if "config" in patch:
            # A patch replaces the WHOLE config of the row and does not merge
            # into it. The effective value of a row is therefore always the
            # value of one layer, readable in one place. This is the rule of
            # dsh, kept deliberately.
            updated = replace(updated, config=patch["config"], layer=layer)
        if "disabled" in patch:
            updated = replace(updated, disabled=evaluate_predicate(patch["disabled"]), layer=layer)
        if "isolate" in patch:
            updated = replace(updated, isolate=_as_isolate(patch["isolate"], layer), layer=layer)
        rows[index] = updated
        return [updated]
    raise LoaderError(f'{layer}: no row with id "{row_id}" to patch')


def _check_unique_ids(rows: Sequence[Row]) -> None:
    """No two composed rows share an id (A7).

    `insert:` has refused a duplicate since it was written. A plain row list did
    not, and the same people write both halves of a profile. A duplicate is not
    a second copy of the row. It is a row that **cannot be addressed**.
    `_apply_patch` returns at the first match, so a `config:` from a later layer
    is applied to one of the two rows, and the other silently keeps the old
    value. `Mount.forks` is keyed by id, so only one is ever forked.

    The check runs over the composed list and not per document, because the
    collision that matters is between layers. A local profile that declares
    again a row that `ph-base` already layers is exactly how the collision
    happens, and neither document is wrong alone. `insert:` had its own copy of
    this refusal until the general one existed. That copy is gone. With two
    spellings of one rule, only the one that somebody edits stays right.
    """
    seen: dict[str, str] = {}
    for row in rows:
        first = seen.get(row.id)
        if first is not None:
            raise LoaderError(
                f'{row.layer}: row id "{row.id}" is already declared by {first}; '
                "address it by id to replace its config instead of declaring it twice"
            )
        seen[row.id] = row.layer


def _check_isolation(rows: Sequence[Row]) -> None:
    """Every `isolate:` names a row that exists, is enabled, and is not the row itself.

    The check runs when the layers are composed and not at mount, so that
    `--dump-config` refuses the same profile that `phern` would refuse. Suppose
    that a later layer removed a row that another row wants a private copy of.
    Without this check, the mount would fail after the user had read a dump
    that looked correct.
    """
    by_id = {row.id: row for row in rows}
    for row in rows:
        for source_id in row.isolate or ():
            source = by_id.get(source_id)
            if source is None:
                raise LoaderError(
                    f'{row.layer}: row "{row.id}" isolates "{source_id}", which is not a row'
                )
            if source.id == row.id:
                raise LoaderError(f'{row.layer}: row "{row.id}" cannot isolate itself')
            if source.disabled:
                raise LoaderError(
                    f'{row.layer}: row "{row.id}" isolates "{source_id}", which is disabled — '
                    "a private copy of a row that is off would be the only copy running"
                )


@dataclass(frozen=True, slots=True)
class ProfileDocument:
    """One parsed layer of a profile: where it came from, its entries, and what it may set.

    The provenance is a *name* and not a path, because not every layer has a
    file. A `--patch` on the command line is a document like any other. A name
    is also what `Row.layer` carries and what `--dump-config` prints.
    """

    layer: str
    entries: JsonValue
    sets: Affects | None = None
    """The one kind of row that this layer may touch, or `None` for any kind.

    `None` is for a layer that pH ships. A bundle defines rows of every kind,
    and the layers of the user are written against what the bundle defines. A
    layer that a user writes names its kind, because each kind has one owner
    (`CONFIGURED_IN`). A session profile that switched off persistence, or a
    daemon configuration that armed a tool, would be a setting kept where the
    configuration that owns it cannot see it. A restart that compares the
    environment of the session would then compare something that it does not
    own."""
    override: bool = False
    """True for a start option (`--patch`), and false for part of the profile that a
    session starts on. A start option is composed like any layer. It is left out
    of the saved base of a session (`ph.session_profile`), which is the named
    profile as it composes. A start option is how the session deviates from that
    profile, and S4 logs it as such."""


def _check_kind(document: ProfileDocument, touched: Sequence[Row]) -> None:
    """Refuse an entry whose rows are of a kind that its layer does not own. The rows
    are the ones that the entry added, patched or removed.

    The caller passes the rows that the one dispatcher in `compose_rows` touched,
    so the grammar of an entry is read one time. Resolving the plugin imports
    it, which the mount would do anyway. The check runs here so that
    `--dump-config` refuses the same layer that `phern` would refuse.
    """
    if document.sets is None:
        return
    for row in touched:
        kind = normalize_plugin(resolve_plugin(row.name)).affects
        if kind != document.sets:
            raise LoaderError(
                f'{document.layer}: row "{row.id}" is {kind}, and this layer sets '
                f"{document.sets} rows only; {kind} settings belong in {CONFIGURED_IN[kind]}"
            )


def compose_rows(documents: Sequence[ProfileDocument]) -> list[Row]:
    """Compose ordered profile documents into the final row list.

    Each entry of a document is either a plain row or a patch entry. Rows keep
    file order, except for the log layer (`_place_log_layer`). Patches address
    rows by id. When a document says what it `sets`, every entry is checked
    against the kind of the row that it touches.
    """
    rows: list[Row] = []
    for document in documents:
        layer, entries = document.layer, document.entries
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise LoaderError(f"{layer}: a profile document must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise LoaderError(f"{layer}: entry must be a mapping, got {entry!r}")
            if _is_patch(entry):
                touched = _apply_patch(rows, entry, layer)
            else:
                touched = _as_rows([entry], layer)
                rows.extend(touched)
            _check_kind(document, touched)
    _check_unique_ids(rows)
    _check_isolation(rows)
    return _place_log_layer(rows)


def _place_log_layer(rows: list[Row]) -> list[Row]:
    """Gather the log layer at the position of its first row, in `LOG_LAYER` order
    (`LogLayer` gives the reason), and keep the order of the other rows around it.

    Only an enabled row is gathered. A disabled row does not mount, so it has no
    place to keep, and resolving it could refuse a profile because of a row that
    is off (the rule of `resolve_row`). A row whose plugin does not resolve is
    not gathered either. It stays where it was written, and the mount is what
    refuses it.
    """
    rank = {
        row.id: LOG_LAYER.index(place)
        for row in rows
        if not row.disabled and (place := _log_layer_of(row)) is not None
    }
    if not rank:
        return rows
    first = next(index for index, row in enumerate(rows) if row.id in rank)
    layer = sorted((row for row in rows if row.id in rank), key=lambda row: rank[row.id])
    rest = [row for row in rows if row.id not in rank]
    return [*rest[:first], *layer, *rest[first:]]


def _log_layer_of(row: Row) -> LogLayer | None:
    """The place in the log layer that the plugin of `row` declares, or `None` when
    the plugin declares none or does not resolve."""
    try:
        return normalize_plugin(resolve_plugin(row.name)).log_layer
    except LoaderError:
        return None


def _warn_unflushed(rows: Iterable[Row]) -> None:
    """Warn when `rows` keep the log and none of them says when the log reaches disk.

    Without a `checkpoint` row, nothing flushes before a model request or a tool
    body. A crash then loses whatever no other write put on disk. The one write
    left at shutdown is the last write of the mount
    (`ph.persistence.write_on_unwind`), which runs after telemetry has gone. This
    is a warning and not a refusal, because the log is still written and the
    deployment may choose this trade.

    The check reads the rows that a mount activated, after the mount has
    settled. A checkpoint row that still waits for a service flushes nothing,
    and a mount that a row refused warns of nothing. The check is not in
    `compose_rows`, which runs again whenever the environment of a session is
    compared.
    """
    places = {row.id: _log_layer_of(row) for row in rows}
    stores = [row_id for row_id, place in places.items() if place == "storage"]
    if stores and "checkpoint" not in places.values():
        log.warning(
            "ph.cordis: %s keeps the session log, but no row flushes it (log_layer "
            '"checkpoint", such as session-checkpoint-policy): a crash loses what no '
            "other write took",
            ", ".join(stores),
        )


def resolve_row(row: Row) -> dict[str, JsonValue]:
    """One row as it would mount: its config passed through the model of its plugin,
    with defaults included and every field stated, including the fields that a
    document may leave out. The unset `sampling` of a route is the exception. It
    is left off, as it is everywhere that it is written (`Sampling.is_empty`).
    Absent and empty read back as one value, so a saved profile and the named
    profile still compare.

    The config is interpolated as a mount interpolates it, so a `${env:...}`
    setting reads as the value that it would run with. A row whose plugin takes
    no config says `config: null`, which is its whole state. A *disabled* row
    whose config its model refuses keeps the config as written. It does not
    mount, so nothing ever checked it, and a listing of a profile must not fail
    because of a row that is off.
    """
    spec = normalize_plugin(resolve_plugin(row.name))
    config: JsonValue
    try:
        model = spec.resolve_config(interpolate(row.config))
        config = None if model is None else model.model_dump(mode="json", by_alias=True)
    except (LoaderError, ValidationError):
        if not row.disabled:
            raise
        config = row.config
    entry = replace(row, config=config).to_entry()
    entry.setdefault("config", None)
    entry.setdefault("disabled", False)
    return entry


def _is_patch(entry: JsonObject) -> bool:
    """Whether an entry addresses rows that are already composed, and declares none."""
    return bool({"insert", "remove"} & set(entry)) or ("id" in entry and "name" not in entry)


def new_declarations(entry: JsonObject, held: Collection[str]) -> JsonObject | None:
    """`entry` without each row that it declares whose id is in `held`, or `None` when
    nothing of the entry is left.

    This is for an entry that may be given again to a session that already holds
    it: a start option at the second start of a session (`ph.session_profile`).
    An entry that patches a row by id declares no row and comes back unchanged.
    Composed twice, it says what it said the first time. A declaration composed
    twice is a row declared twice, which `_check_unique_ids` refuses. So the
    rows that `held` names are left out, with their config. A held row keeps the
    config that it is held with, and a different config given here is not
    applied (a setting is changed by id). A malformed entry comes back
    unchanged, for `compose_rows` to refuse."""
    if "insert" in entry:
        inserted = entry["insert"]
        if not isinstance(inserted, list):
            return entry
        new = [row for row in inserted if not (isinstance(row, dict) and _id_of(row) in held)]
        return {**entry, "insert": new} if new else None
    if _is_patch(entry) or _id_of(entry) not in held:
        return entry
    return None


def entry_ids(entry: JsonObject) -> list[str]:
    """The row ids that one profile entry adds, patches or removes. This is the
    grammar that `compose_rows` reads, stated for a caller that needs it without
    composing."""
    if "insert" in entry:
        inserted = entry["insert"]
        return [row.id for row in _as_rows(inserted if isinstance(inserted, list) else [], "entry")]
    if _is_patch(entry):
        row_id = as_str(entry.get("id"))
        return [row_id] if row_id else []
    return [row.id for row in _as_rows([entry], "entry")]


def sparse_entries(base: Sequence[Row], rows: Sequence[Row]) -> list[JsonObject]:
    """The fewest entries that, composed over `base`, give `rows` (session profiles, S2).

    This is what a saved named profile holds: only the rows that differ from the
    profile that it extends.

    * A row that `base` has and `rows` does not have is removed.
    * A row that `rows` adds is declared whole.
    * A row in both is patched by the fields that differ. A patch replaces the
      *whole* config of a row, so the config in the patch is the entire config
      of the row and never a delta of it.
    * A row whose plugin changed under the same id is removed and declared
      again, because no patch changes a `name`.

    `compose_rows([*base_documents, ProfileDocument(layer, sparse_entries(base, rows))])`
    gives back `rows`, in order, for every `rows` that a composition over `base`
    can produce. Added rows are placed at the end, which is where a later layer
    put them, and a log-layer row is placed again where its plugin says.
    """
    kept = {row.id for row in rows}
    before = {row.id: row for row in base}
    entries: list[JsonObject] = [
        {"id": row.id, "remove": True} for row in base if row.id not in kept
    ]
    for row in rows:
        was = before.get(row.id)
        if was is None or was.name != row.name:
            if was is not None:
                entries.append({"id": row.id, "remove": True})
            entries.append(row.to_entry())
            continue
        patch: dict[str, JsonValue] = {}
        if row.config != was.config:
            patch["config"] = row.config
        if row.disabled != was.disabled:
            patch["disabled"] = row.disabled
        if row.isolate != was.isolate:
            patch["isolate"] = dict(row.isolate) if row.isolate is not None else None
        if patch:
            entries.append({"id": row.id, **patch})
    return entries


def load_profile_documents(
    paths: Sequence[Path], *, sets: Affects | None = None
) -> list[ProfileDocument]:
    """Read and parse each layer. The path is the provenance that its rows carry.

    `sets` applies to every layer, for a caller that reads the files of a user.
    See `ProfileDocument.sets`.
    """
    return [
        ProfileDocument(
            str(path), safe_yaml_load(path.read_text(encoding="utf-8"), origin=str(path)), sets
        )
        for path in paths
    ]


# --------------------------------------------------------------- resolution --


@cache
def entry_point_targets(group: str) -> dict[str, str]:
    """The `{name: target}` of one entry-point group, scanned one time per process.

    Reading the metadata of every installed distribution is the expensive part
    of resolution, and the set cannot change while the process runs. The cache
    is per group, so a second group (`ph.bundles`) pays the same cost one time,
    and there is one place where a cache would ever have to be cleared.
    """
    return {entry.name: entry.value for entry in entry_points(group=group)}


def resolve_entry_point(
    group: str,
    name: str,
    *,
    default_attribute: str = "",
) -> object:
    """Import what `name` registers in `group`, or return `None` if nothing registers it.

    This is the mechanical half of resolution (look up, import, `getattr`), with
    no policy. A caller that wants an exception raises its own, and a caller
    that wants to offer an alternative gets `None`. `ph.bundles.resolve_bundle`
    is a thin wrapper over this function. `resolve_plugin` is **not**. It needs
    a dotted-path fallback, a `LoaderError` in place of `None`, and two
    attribute candidates in place of one, so it has its own copy of the
    sequence. A third group wanting any of those three should widen this rather
    than add a fourth.
    """
    target = _entry_point_targets(group).get(name)
    if target is None:
        return None
    module_path, _, attribute = target.partition(":")
    module = importlib.import_module(module_path)
    attribute = attribute or default_attribute
    return getattr(module, attribute) if attribute else module


def _entry_point_targets(group: str = ENTRY_POINT_GROUP) -> dict[str, str]:
    """The plugin group by default, so existing callers read unchanged."""
    return entry_point_targets(group)


def _state(fork: ForkScope) -> str:
    """The state of one fork, as `Mount.topology` prints it.

    This function only formats. `ForkScope.state` decides which of the six names
    applies. The ordering that was in the ladder here is now with it: `failed`
    comes before `waiting on`, because a row that raised may also be missing a
    key. Two of the six states take a phrase in place of the bare name, and that
    is all that is left here. `waiting on <key>` names what is missing.
    `unwound · waiting on <key>` says that the row was up and a provider swap
    took it down, which is the case that a running daemon is asked about.
    """
    injects = ", ".join(fork.injects) or "nothing"
    match fork.state:
        case "failed":
            return f"failed · {fork.failure} · injects {injects}"
        case "waiting" | "unwound" as state:
            lead = "unwound · " if state == "unwound" else ""
            return f"{lead}waiting on {', '.join(fork.waiting_on)} · injects {injects}"
        case "unmounted" | "active" | "activating" as state:
            return f"{state} · injects {injects}"
        case _ as unhandled:
            # The states above are named, and not captured by this case. A
            # seventh `ForkState` is therefore a build failure here, and not a
            # row that formats as its own name.
            assert_never(unhandled)


def import_plugin_modules() -> list[ModuleType]:
    """Import the module behind every registered plugin, in name order.

    Event declarations are in the modules that own them. A tool that wants the
    complete registry (`phern events`) therefore imports the plugin surface and
    does not keep a list by hand. The same call covers third-party wheels.
    """
    modules: list[ModuleType] = []
    for name in sorted(_entry_point_targets()):
        module_path = _entry_point_targets()[name].partition(":")[0]
        modules.append(importlib.import_module(module_path))
    return modules


def resolve_plugin(name: str) -> object:
    """Resolve the `name:` of a row to a plugin object.

    The name is looked up first in the `ph.plugins` entry-point group, which is
    the compatibility surface that a third-party wheel registers into. It is
    then tried as a dotted module path, and then as `module:attribute`.
    """
    target = _entry_point_targets().get(name, name)
    module_path, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise LoaderError(f'cannot resolve plugin "{name}": {error}') from error
    if attribute:
        try:
            return getattr(module, attribute)
        except AttributeError as error:
            raise LoaderError(
                f'plugin "{name}" resolved to {module_path} which has no "{attribute}"'
            ) from error
    for candidate in ("plugin", "apply"):
        found = getattr(module, candidate, None)
        if found is not None:
            return found
    return module


# ------------------------------------------------------------------ loader --


@dataclass(slots=True)
class Profile:
    """A composed profile: the rows, and the documents that they came from.

    A profile is composed **one time**: at `profile_or_exit` for a command, and
    at its fixture for a test. It is then mounted one time for each deployment.
    The daemon holds one profile and a `Mount` per root. Sharing is safe because
    `Row` is frozen and `interpolate` copies each config on its way into a fork.
    Two mounts of one profile therefore never hold the same config object.
    """

    documents: list[ProfileDocument] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    name: str = ""
    """The named composition that this is (`headless`, `rlm`), or `""` for one built
    from a path or assembled ad hoc.

    The name is deliberately opaque provenance. Cordis takes rows in and produces
    a plugin tree, and `ph.paths` says where the files of a named profile are on
    disk. The name lets a row that writes on behalf of the user tell the two
    cases apart, because an ad-hoc composition has no name to write under."""

    @classmethod
    def from_documents(cls, documents: Sequence[ProfileDocument], *, name: str = "") -> Profile:
        return cls(documents=list(documents), rows=compose_rows(documents), name=name)

    @classmethod
    def from_paths(cls, paths: Sequence[Path]) -> Profile:
        return cls.from_documents(load_profile_documents(paths))

    def enabled_rows(self) -> Iterator[Row]:
        return (row for row in self.rows if not row.disabled)

    def dump(self) -> list[dict[str, JsonValue]]:
        """The composed row list, for `--dump-config`."""
        return [row.to_dump() for row in self.rows]

    def rows_of(self, kinds: Collection[Affects]) -> list[Row]:
        """The rows whose plugin declares one of `kinds` (`affects`), in order."""
        return [
            row for row in self.rows if normalize_plugin(resolve_plugin(row.name)).affects in kinds
        ]

    def resolved(self, kinds: Collection[Affects] | None = None) -> list[dict[str, JsonValue]]:
        """Every row as it would mount (`resolve_row`), narrowed to the rows of `kinds`.
        `{"environment"}` gives the profile of a session. This is the "full" profile
        of Decision 8."""
        return [resolve_row(row) for row in (self.rows if kinds is None else self.rows_of(kinds))]

    async def mount(self, ctx: Context, *, project: Path | None = None) -> Mount:
        """Mount every enabled row onto `ctx`, settle the tree, and return the mount.

        Rows mount in the order composed, but nothing runs until `reconcile()`.
        Activation is driven by service availability, so a row that needs `llm`
        waits for the row that provides it, wherever that row is in the order.

        `project` is the directory that this mount works in. See `PROJECT_ROOT`,
        which this method provides beside `ctx.mount` for the same reason. When
        `project` is omitted, nothing is provided, and `fs-local` falls back to
        the directory of the process. That is right for a host that *is* in the
        project, and wrong for a daemon that holds roots in several projects.

        **The `Mount` is provided as `ctx.mount` before the first row.**
        `ph.seams` may import `ph.cordis` and never the reverse, so this is how
        the `topology` row reaches what only the mount knows. This is
        deliberately a widening, and it is named as one here: every row can now
        read the whole composition, disabled rows included. It grants no
        *capability* that a row lacked, because `ctx.plugin` and `ctx.scope` are
        public. The composition is the profile that the user wrote, and
        `--dump-config` prints it for anyone who can start the process.
        """
        mount = Mount(profile=self, root=ctx)
        ctx.provide(MOUNT, mount)
        if project is not None:
            ctx.provide(PROJECT_ROOT, project)
        forks = mount.forks
        by_id = {row.id: row for row in self.rows}
        for row in self.enabled_rows():
            if not row.isolate:
                forks[row.id] = ctx.plugin(
                    resolve_plugin(row.name), interpolate(row.config), row=row.id
                )
                continue
            # This is the `isolate.fs` of dsh. The scope of the row becomes an
            # isolation boundary and a provisioning realm of its own. `scope()`
            # does exactly that, and an agent gets the same kind of scope. A
            # second copy of each named row is mounted *inside* the realm. The
            # `provide` of that copy lands in the realm, so `ctx.fs` resolves to
            # the private instance for this row and for everything beneath it,
            # while every other row keeps the shared instance. Nothing is
            # redirected: `_provision` walks up from the realm and finds the
            # nearer provision first.
            #
            # The copies are settled before this row mounts, and then
            # **checked**. `has(key)` is true from the realm as soon as root
            # provides the key. The isolating row is therefore ready
            # immediately, against the shared service, and only registration
            # order would make the private copy win. A reconcile here fixes the
            # order for a copy that is ready. It does nothing for a copy that
            # is not ready. A copy whose own `inject` is unmet stays waiting,
            # the row activates against the instance of root, and the realm
            # silently falls through to exactly the service that it was meant
            # to replace. So a private copy that did not activate is a refusal
            # that names the key it lacks. What the copy needs must be provided
            # above the realm, by a row earlier in the profile.
            realm = ctx.scope(f"realm:{row.id}")
            privates: dict[str, ForkScope] = {}
            for source_id, override in row.isolate.items():
                source = by_id[source_id]
                config = source.config if override is None else override
                privates[source_id] = forks[f"{row.id}/{source_id}"] = realm.plugin(
                    resolve_plugin(source.name), interpolate(config), row=row.id
                )
            await ctx.reconcile()
            for source_id, private in privates.items():
                if not private.active:
                    missing = ", ".join(private.waiting_on)
                    raise LoaderError(
                        f'row "{row.id}" isolates "{source_id}", whose private copy is waiting '
                        f"on {missing}; a row mounted into a realm must have its dependencies "
                        "provided by rows above it, or the realm would fall through to the "
                        "shared service it exists to replace"
                    )
            # **This mount is transparent, and only this mount** (A9). Without
            # that, the activation scope of this row would inherit the
            # isolation of the realm, and `reaches` would answer "nobody" for
            # it. The row would get no dispatch, and no tool, prompt section,
            # fs screen or skill restriction that it registers would be
            # visible. It is a deployment row with a narrower view of one
            # service, so it gets the visibility of a deployment row.
            #
            # The private copies above are never transparent. A second `fs`
            # that answered every dispatch would handle each event a second
            # time, beside the instance that it exists to shadow, and would
            # register a second copy of whatever it registers.
            forks[row.id] = realm.plugin(
                resolve_plugin(row.name), interpolate(row.config), transparent=True, row=row.id
            )
        await ctx.reconcile()
        # This is the one moment at which a composed profile is whole and
        # nothing has run yet, so a row can refuse the deployment that it is
        # mounted in (E8). The dispatch is `serial` and not `emit`. A listener
        # that raises must stop the process, and an `emit` contains what its
        # listeners raise, which is exactly this refusal. A row cannot check
        # this in its own `apply`. A backend that it depends on may be layered
        # after it, so a verdict computed in `apply` would be wrong for exactly
        # the profile that orders its rows that way.
        await ctx.serial("profile/mounted")
        _warn_unflushed(row for row in self.enabled_rows() if forks[row.id].active)
        return mount


MOUNT: ServiceKey[Mount] = ServiceKey("mount")
"""The mount itself. It is provided before the first row, so that a seam can ask
the composition what it is (`ph.seams.topology`)."""

PROJECT_ROOT: ServiceKey[Path] = ServiceKey("project_root")
"""The directory that this mount works in: the directory that the header of the
session names (P5-14).

**It is declared and provided beside `MOUNT` because it is the same kind of
fact.** It is true of *this* mount and not of the profile, and a row needs it
while the row applies. `fs-local` reads it as its root, and `workspace-lifecycle`
then branches its worktrees from it. A profile setting could not carry it. One
daemon mounts one composition many times, one time per session, and those
sessions are in different repositories. Composing a profile per root would import
every plugin again to change one path.

The key was in `ph.keys`, and `ph_app.runtime` provided it: the key in the
package that reads it, and the provision in the package above. That split made
the fact reachable only through an undocumented protocol (provide it, *then*
mount). Every other caller of `Profile.mount` (the tests, the mounts that the
plugins make, a library embedder) had no way to say where it worked, and
silently took `Path.cwd()`. Now `Profile.mount`, which needs the fact, takes it
as an argument."""


@dataclass(slots=True)
class Mount:
    """What one `Profile` became on one context: the half that a `Profile` cannot know.

    The composition is `profile`. This object is what happened when the profile
    was mounted: the fork that each row became, on the root that it was mounted
    on. They are two objects because they have two lifetimes. A `Profile`
    outlives every mount of it, and a `Mount` ends with its root. The split also
    means that nothing composes twice: the daemon composes one time and mounts
    per root.
    """

    profile: Profile
    root: Context
    """The deployment context, for the realm walk of `topology`. It is held here so
    that a reader need not know to pass the *root* and not its own scope. A walk
    from the scope of a row finds no realms and silently reports "none"."""
    forks: dict[str, ForkScope] = field(default_factory=dict)
    """Every mounted plugin by row id. A private copy mounted into a realm has the
    key `"<isolating row>/<source row>"`, which is also the label that `topology`
    gives it."""
    reconfigured: set[str] = field(default_factory=set)
    """The ids of the rows that were applied again live on this mount. `topology`
    marks them.

    This is a set and not the configs. The only question asked of it is whether
    a row was reconfigured, and the config that each fork runs is already on the
    fork (`ForkScope.config`). A parallel dict was a second statement of one
    fact, and it retained every superseded config for the life of the mount."""

    async def reconfigure(self, row_id: str, config: object) -> ForkScope:
        """Apply one row again with a new config, on this mount, and touch nothing else.

        **This uses the shape of cordis for its purpose.** The `apply` of a row
        registers into seams, and every registration is an effect of the
        activation scope of the row (I2). Disposing the fork unwinds all of
        them, and mounting the row again with the new config makes all of them
        again. No other row is asked to do anything. A seam whose slot was
        released and refilled answers its next call with the new value, and an
        agent in the middle of a command notices only when its next command
        asks. That is how `/sandbox allow host` takes effect while the agent is
        still running.

        The change is per *mount* and not per profile. A daemon mounts one
        `Profile` one time per root and `Row` is frozen, so the new config is
        recorded here, and `topology` marks the row `reconfigured live`. Only
        `ph.session_profile` calls this method, from three functions:

        * `override`, which records the change in the log of the session before
          this method runs.
        * `clear_overrides`, which records the clear in the same way.
        * `opened`, which brings a mount to what that log already says (S4).

        `test_session_profile` checks that no other module calls it.

        The method refuses three kinds of row:

        * A row that is disabled. There is no fork to replace. Enable the row in
          the profile.
        * A row that isolates others. Its realm holds private copies that this
          method would orphan.
        * A row that others isolate. Their private copies would keep the old
          config while the shared instance moved.
        """
        row = next((one for one in self.profile.rows if one.id == row_id), None)
        if row is None:
            raise LoaderError(f'no row with id "{row_id}" to reconfigure')
        if row.disabled:
            raise LoaderError(
                f'row "{row_id}" is disabled by {row.layer}; enable it in the profile '
                "before reconfiguring it"
            )
        if row.isolate:
            raise LoaderError(f'row "{row_id}" isolates other rows and cannot be reconfigured live')
        copies = [key for key in self.forks if key.endswith(f"/{row_id}")]
        if copies:
            raise LoaderError(
                f'row "{row_id}" has private copies ({", ".join(copies)}) that would keep '
                "the old config; change the profile instead"
            )
        previous = self.forks.get(row_id)
        if previous is not None:
            await previous.dispose()
        fork = self.forks[row_id] = self.root.plugin(
            resolve_plugin(row.name), interpolate(config), row=row_id
        )
        self.reconfigured.add(row_id)
        await self.root.reconcile()
        return fork

    def inactive(self) -> list[str]:
        """The ids of the rows whose plugin is not active, whatever keeps it down.

        The method reads `state` and not `not fork.active`, because the two
        stopped meaning the same thing when `failed` joined the set. This list
        used to mean "an unmet `inject` key". A row whose `apply` raised is now
        in the same list for a different reason. A caller reads the reason from
        `_state`.
        """
        return [row_id for row_id, fork in self.forks.items() if fork.state != "active"]

    def topology(self) -> list[tuple[str, str]]:
        """What the mount *became*, row by row: the half that `Profile.dump()` cannot show.

        `dump()` is the composition before anything runs, and it claims nothing
        more. But a row that mounted and never activated (an unmet `inject`
        key) looks identical there to a row that runs, and a reader of the YAML
        cannot tell which of the two they have. The rule of dsh is that the dump
        must show what *is* running, because when structure comes from
        configuration the static code no longer says. `inactive()` had that
        answer, and nothing called it.

        There is one line per row: its state (`_state`: active, activating,
        failed, waiting on a key, unwound, unmounted), what it injects, and the
        layer that put it there. Disabled rows are listed too. "This was turned
        off by `rlm-stable.yaml`" is what a user who asks "why isn't X running"
        needs, and without that line a disabled row would look the same as an
        absent row. The isolated realms under the root come last. There are none
        at `phern doctor` time, because the scope of an agent is created when
        the agent runs. That absence is stated, and not left as a missing line.
        """
        lines: list[tuple[str, str]] = []
        for row in self.profile.rows:
            fork = self.forks.get(row.id)
            # Show the last two path components (`bundles/base.yaml`,
            # `ph_rlm/bundle.yaml`, `profiles/rlm-stable.yaml`). Every bundle
            # file is called `bundle.yaml`, so its directory is the name that
            # distinguishes it. The full absolute path would be a table column
            # that nobody can read.
            layer = "/".join(Path(row.layer).parts[-2:])
            if fork is None:
                lines.append((row.id, f"disabled · by {layer}"))
                continue
            provenance = f"from {layer}"
            if row.id in self.reconfigured:
                provenance += ", reconfigured live"
            lines.append((row.id, f"{_state(fork)} · {provenance}"))
            for source_id, override in (row.isolate or {}).items():
                private = self.forks[f"{row.id}/{source_id}"]
                how = "own config" if override is None else "overridden config"
                lines.append(
                    (
                        f"{row.id}/{source_id}",
                        f"{_state(private)} · private copy in realm:{row.id} · {how}",
                    )
                )
        realms = [node.path for node in self.root.descendants() if node.isolation is node]
        lines.append(
            (
                "isolated realms",
                ", ".join(realms) or "none — an agent's scope is created when it runs",
            )
        )
        return lines
