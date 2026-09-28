"""Rows in, plugin tree out.

A pH profile is a list of rows. Each row names one plugin module and carries
its config. A bundle contributes rows through a **patch** — either an
`insert:` of new rows or an id-addressed replacement of one row's *whole*
config. Layers apply in order, last write wins per row, and `--dump-config`
prints the composed result with the layer each row came from.

Two rules make this data rather than code (D9, invariant I-8):

* the only interpolation is `${env:VAR:-default}`;
* YAML is parsed with a **safe** loader whose implicit-resolver set is
  narrowed further, so a `!!python/...`-style tag is a load error rather than a
  call — dsh's `!!js` idiom is deliberately not ported.

@module ph.cordis.loader
"""

from __future__ import annotations

import importlib
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
from .plugin import CONFIGURED_IN, Affects, normalize_plugin

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
    "resolve_entry_point",
    "resolve_plugin",
    "resolve_row",
    "safe_yaml_load",
    "sparse_entries",
]

ENTRY_POINT_GROUP = "ph.plugins"

_ENV_PATTERN = re.compile(
    r"\$\{env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>(?:[^{}]|\{[^{}]*\})*))?\}"
)
"""`${env:NAME}` or `${env:NAME:-default}`.

The default is **not** `[^}]*` (A10). A JSON-shaped fallback is the ordinary
thing to want there — `${env:PH_EXTRA:-{"tier":"none"}}` — and under the old
class it matched up to the *first* `}`, so the row silently received
`{"tier":"none"` and whatever followed stayed literal. One level of balanced
braces is what a default is ever written with, and it keeps the pattern a
regular expression rather than a parser.
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
"""The one moment a profile is whole and nothing has run yet.

Two uses, and the second arrived after the first (P4-13). A row may **refuse the
deployment** it finds itself in (E8, `containment.strict`), and a row may
**collect what the whole profile turned out to contain** — whether any subagent
provider was mounted, whether any skill was installed — which is a question with
no final answer until this moment and which `ctx.inject` cannot express, since
neither is a service key.

The two share a dispatch, so the rules of the shared one apply to both: `serial`
bails on a non-null return, so a collector that returned a value would silently
skip the refusals registered after it, and it propagates exceptions, so a
collector that raises stops the process. Collect by side effect, return `None`,
and let the refusals be the only listeners that can end a run.

Declared here rather than in a seam because the loader is what dispatches it, and
a declaration in a module the loader never imports is one that has not happened
by the time the dispatch checks for it."""


class SafeRowLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` with every non-scalar implicit conversion removed.

    **Deliberately not libyaml's `CSafeLoader`**, which is 8x faster on a real
    document and was tried. Subclassed here, with the two customizations below,
    it makes `Resolver.resolve` answer `None` for *every* node kind — so every
    tag becomes undefined and the first profile load dies with "could not
    determine a constructor for the tag None". It reproduces only when this
    module is imported while `coverage` is tracing, which is to say: under CI's
    own `--cov` gate, at collection, and not in an ordinary run. Each ingredient
    is fine alone — a bare `CSafeLoader` subclass built under coverage parses, and
    so does one carrying the resolver rebuild — so the interaction is not
    understood and the speed is not worth shipping a loader that fails only where
    it is measured.

    The base already refuses `!!python/object`. This subclass additionally
    refuses timestamps and sexagesimals, so a row value that looks like a date
    stays the string the author wrote — a config file is data, and a silent
    type change is the same class of surprise as evaluation.
    """


_UNRESOLVED = frozenset({"tag:yaml.org,2002:timestamp"})
"""Implicit tags this loader declines to apply. See `SafeRowLoader`."""

# **Rebound on the subclass, never item-assigned into the inherited table.**
# `yaml_implicit_resolvers` is one dict shared by every loader PyYAML defines, so
# `SafeRowLoader.yaml_implicit_resolvers[first] = ...` — which is what this was —
# reached through to `Resolver`'s own copy and took timestamps away from
# `yaml.safe_load` **process-wide**, for pH and for any library sharing the
# interpreter. PyYAML's own `add_implicit_resolver` copies before it writes for
# exactly this reason; this is that copy, done once.
SafeRowLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag not in _UNRESOLVED]
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

    `SafeRowLoader` is what makes the result a JSON tree rather than `yaml.load`'s
    `Any` — no custom tags, no implicit dates, so the only things it can build are
    the ones `JsonValue` names. This is the one YAML entry point, so that is said
    here instead of at each reader.
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
    """Which profile document contributed this row's current config."""
    isolate: dict[str, JsonValue] | None = None
    """Row ids this row wants private copies of, each with a config override or `None`.

    dsh's `isolate.fs`: an isolated realm for one service. Here it is spelled
    against **row ids** rather than service keys, because a row does not declare
    what it provides — `provide` is a runtime call — so `fs` names the row whose
    `apply` provides `ctx.fs`, and the loader mounts a second copy of that row
    inside this row's realm. The two spellings sometimes coincide (`fs` →
    `ctx.fs`, `tools` → `ctx.tools`) and in `base.yaml` often do not — `session`
    provides `ctx.sessions`, `agent` provides `ctx.agents` — so this is a row id
    and not a service key, which is also the half that can be checked at compose
    time."""

    def to_entry(self) -> dict[str, JsonValue]:
        """This row as a profile document declares it — what the loader reads back.

        `to_dump` without its provenance: `layer` is where a row came from, which a
        document cannot say about itself, so a saved profile made of dumps would be
        one the loader refuses.
        """
        entry: dict[str, JsonValue] = {"id": self.id, "name": self.name}
        if self.config is not None:
            entry["config"] = self.config
        if self.disabled:
            entry["disabled"] = True
        if self.isolate is not None:
            # Always the mapping: `isolate: [fs]` dumps as `{fs: null}`, which
            # `_as_isolate` reads back to the same thing. One dump shape.
            entry["isolate"] = dict(self.isolate)
        return entry

    def to_dump(self) -> dict[str, JsonValue]:
        """`to_entry` and the layer it came from, for `--dump-config`."""
        return {**self.to_entry(), "layer": self.layer}


# ------------------------------------------------------------ interpolation --


def interpolate(value: object, env: Mapping[str, str] | None = None) -> object:
    """Expand `${env:VAR:-default}` through a value tree.

    A whole-string match keeps the environment value's own type only insofar as
    it is a string: pH does not guess numbers out of the environment, because a
    row whose meaning changes with an accidental `"0"` is exactly the failure
    a typed config model exists to catch.
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
    """Resolve a row's `disabled:` field.

    Accepts a literal boolean or one of two closed predicates:
    `${platform:win32}` and `${env:VAR}` (truthy when set and not empty).
    Anything else is a config error rather than an expression to evaluate.
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
                id=str(entry.get("id") or entry["name"]),
                name=str(entry["name"]),
                config=entry.get("config"),
                disabled=evaluate_predicate(entry.get("disabled")),
                layer=layer,
                isolate=_as_isolate(entry.get("isolate"), layer),
            )
        )
    return rows


def _as_isolate(value: object, layer: str) -> dict[str, JsonValue] | None:
    """`isolate:` as a list of row ids, or a mapping of row id to config override.

    Two spellings for one fact, and both normalize to the mapping: `[fs]` is
    "a private `fs` with the row's own config", `{fs: {root: /tmp/x}}` is "a
    private `fs` rooted somewhere else" — which is the case the feature exists
    for, since a private copy with identical config is a second instance and
    nothing more.
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
    """Apply one patch entry to `rows`, in place, and answer the rows it touched."""
    unknown = set(patch) - {"insert", "id", "config", "disabled", "remove", "isolate"}
    if unknown:
        raise LoaderError(f"{layer}: patch has unknown keys {sorted(unknown)}")
    if "insert" in patch:
        inserted = patch["insert"]
        if not isinstance(inserted, list):
            raise LoaderError(f"{layer}: insert: must be a list of rows")
        # No duplicate check here: `_check_unique_ids` runs over the composed
        # list and subsumes it, with a better message — it names the layer that
        # declared the id first, which an insert-local check cannot see.
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
            # A patch replaces the row's WHOLE config rather than merging into
            # it, so a row's effective value is always one layer's, readable in
            # one place (dsh's rule, kept deliberately).
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

    `insert:` has refused this since it was written; a plain row list did not,
    and the two halves of a profile are written by the same people. A duplicate
    is not a second copy of the row — it is a row that **cannot be addressed**:
    `_apply_patch` returns at the first match, so a later layer's `config:`
    lands on one of them and the other keeps the old value silently, and
    `Mount.forks` is keyed by id so only one is ever forked.

    Checked over the composed list rather than per document, because the
    collision that matters is between layers: a local profile re-declaring a row
    `ph-base` already layers is exactly how this is reached, and neither
    document is wrong on its own. `insert:` had its own copy of this refusal
    until the general one existed; it is gone, because two spellings of one rule
    means whichever somebody edits is the one that stays right.
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
    """Every `isolate:` names a row that exists, is enabled, and is not itself.

    Checked once the layers are composed rather than at mount, so `--dump-config`
    refuses the same profile `phern` would — a private copy of a row a later layer
    removed is a mount that fails after the person has read a dump that looked
    fine.
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

    The provenance is a *name*, not a path, because not every layer has a file — a
    `--patch` on the command line is a document like any other — and a name is what
    `Row.layer` carries and `--dump-config` prints.
    """

    layer: str
    entries: JsonValue
    sets: Affects | None = None
    """The one kind of row this layer may touch, or `None` for any.

    `None` for a layer pH ships: a bundle defines rows of every kind, and the
    person's layers are written against what it defines. A layer a person writes
    names its kind, because each kind has one owner (`CONFIGURED_IN`): a session
    profile that switched off persistence, or a daemon configuration that armed a
    tool, would be a setting kept where the configuration that owns it cannot see
    it — and a restart that compares the session's environment would compare
    something it does not own."""
    override: bool = False
    """A start option — `--patch` — rather than part of the profile a session starts
    on. Composed like any layer, and left out of a session's saved base
    (`ph.session_profile`), which is the named profile as it composes: a start
    option is what the session deviates from it by, and S4 logs it as such."""


def _check_kind(document: ProfileDocument, touched: Sequence[Row]) -> None:
    """Refuse an entry whose rows — added, patched or removed — are of a kind its
    layer does not own.

    Handed the rows the one dispatcher in `compose_rows` touched, so the grammar of
    an entry is read once. Resolving the plugin imports it, which the mount would do
    anyway; asked here so `--dump-config` refuses the same layer `phern` would.
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

    Each document is either a plain list of rows or a list of patch entries.
    Rows keep file order; patches address rows by id. A document that says what it
    `sets` has every entry checked against the kind of the row it touches.
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
    return rows


def resolve_row(row: Row) -> dict[str, JsonValue]:
    """One row as it would mount: its config through its plugin's model, defaults
    included, and every field stated — the ones a document may leave out too.

    Interpolated as a mount interpolates, so a `${env:...}` setting reads as the value
    it would run with. A row whose plugin takes no config says `config: null`, which
    is its whole state. A *disabled* row whose config its model refuses keeps the
    config as written: it does not mount, so nothing ever checked it, and a listing of
    a profile must not fail over a row that is off.
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
    """Whether an entry addresses rows already composed, rather than declaring one."""
    return bool({"insert", "remove"} & set(entry)) or ("id" in entry and "name" not in entry)


def entry_ids(entry: JsonObject) -> list[str]:
    """The row ids one profile entry adds, patches or removes — the grammar
    `compose_rows` reads, stated for a caller that needs it without composing."""
    if "insert" in entry:
        inserted = entry["insert"]
        return [row.id for row in _as_rows(inserted if isinstance(inserted, list) else [], "entry")]
    if _is_patch(entry):
        row_id = as_str(entry.get("id"))
        return [row_id] if row_id else []
    return [row.id for row in _as_rows([entry], "entry")]


def sparse_entries(base: Sequence[Row], rows: Sequence[Row]) -> list[JsonObject]:
    """The fewest entries that, composed over `base`, give `rows` (session profiles, S2).

    What a saved named profile holds: only the rows that differ from the profile
    it extends. A row `base` has and `rows` does not is removed; one `rows` adds
    is declared whole; one in both is patched by the fields that differ, and a
    patch replaces a row's *whole* config, so the config it carries is the row's
    entire config and never a delta of it. A row whose plugin changed under the
    same id is removed and declared again, since no patch changes a `name`.

    `compose_rows([*base_documents, (layer, sparse_entries(base, rows))])` gives
    back `rows`, in order, for every `rows` a composition over `base` can
    produce: added rows land at the end, which is where a later layer put them.
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
    """Read and parse each layer; the path is the provenance its rows carry.

    `sets` is every layer's, for a caller reading a person's files — see
    `ProfileDocument.sets`.
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
    """One entry-point group's `{name: target}`, scanned once per process.

    Reading every installed distribution's metadata is the expensive part of
    resolution, and the set cannot change while the process runs. Cached per
    group so a second group — `ph.bundles` — pays the same once and there is one
    place a cache would ever have to be cleared.
    """
    return {entry.name: entry.value for entry in entry_points(group=group)}


def resolve_entry_point(
    group: str,
    name: str,
    *,
    default_attribute: str = "",
) -> object:
    """Import what `name` registers in `group`, or `None` if nothing does.

    The mechanical half of resolution — look up, import, `getattr` — with no
    policy: a caller that wants an exception raises its own, and one that wants
    to offer an alternative gets `None`. `ph.bundles.resolve_bundle` is a thin
    wrapper over this. `resolve_plugin` is **not**: it needs a dotted-path
    fallback, a `LoaderError` rather than `None`, and two attribute candidates
    rather than one, so it carries its own copy of the dance — a third group
    wanting any of those three should widen this rather than add a fourth.
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
    """One fork's state, as `Mount.topology` prints it.

    Formatting only: which of the six names applies is `ForkScope.state`'s to
    answer, and the ordering that used to live in the ladder here — `failed`
    ahead of `waiting on`, because a row that raised may also be missing a key —
    lives with it. Two of the six take a phrase rather than the bare name, which
    is the whole of what is left: `waiting on <key>` names what is missing, and
    `unwound · waiting on <key>` says it was up and a provider swap took it down,
    which is the case a running daemon is asked about.
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
            # Named rather than captured, so a seventh `ForkState` is a build
            # failure here instead of a row that formats as its own name.
            assert_never(unhandled)


def import_plugin_modules() -> list[ModuleType]:
    """Import the module behind every registered plugin, in name order.

    Event declarations live in the modules that own them, so a tool that wants
    the complete registry — `phern events` — imports the plugin surface rather
    than a hand-kept list. Third-party wheels are covered by the same call.
    """
    modules: list[ModuleType] = []
    for name in sorted(_entry_point_targets()):
        module_path = _entry_point_targets()[name].partition(":")[0]
        modules.append(importlib.import_module(module_path))
    return modules


def resolve_plugin(name: str) -> object:
    """Resolve a row's `name:` to a plugin object.

    Looked up first in the `ph.plugins` entry-point group — the compatibility
    surface a third-party wheel registers into — then as a dotted module path,
    then as `module:attribute`.
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
    """A composed profile: the rows, and the documents they came from.

    Composed **once** — at `profile_or_exit` for a command, at its fixture for a
    test — and mounted as many times as there are deployments to mount it on; the
    daemon holds one of these and a `Mount` per root. Sharing is safe because `Row`
    is frozen and `interpolate` copies each config on its way into a fork, so two
    mounts of one profile never hold the same config object.
    """

    documents: list[ProfileDocument] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    name: str = ""
    """Which named composition this is — `headless`, `rlm` — and `""` for one built
    from a path or assembled ad hoc.

    Opaque provenance, deliberately: cordis takes rows in and produces a plugin
    tree, and where a named profile's files live on disk is `ph.paths`'s to say.
    What this enables is that a row writing on the person's behalf can tell the two
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
        """Every row as it would mount (`resolve_row`), narrowed to the rows of `kinds` —
        `{"environment"}` is a session's profile. Decision 8's "full" profile."""
        return [resolve_row(row) for row in (self.rows if kinds is None else self.rows_of(kinds))]

    async def mount(self, ctx: Context, *, project: Path | None = None) -> Mount:
        """Mount every enabled row onto `ctx`, settle the tree, and return the mount.

        Rows mount in file order, but nothing runs until `reconcile()`:
        activation is service-availability driven, so a row that needs `llm`
        waits for whichever row provides it regardless of where it sits.

        `project` is where this mount works — see `PROJECT_ROOT`, which it
        provides beside `ctx.mount` and for the same reason. Omitted, nothing is
        provided and `fs-local` falls back to the process's own directory, which
        is right for a host that *is* in the project and wrong for a daemon
        holding roots in several.

        **The `Mount` is provided as `ctx.mount` before the first row**: `ph.seams`
        may import `ph.cordis` and never the reverse, so this is how the
        `topology` row reaches what only the mount knows. Deliberately a widening,
        and worth naming as one — every row can now read the whole composition,
        disabled rows included. It grants no *capability* a row lacked, since
        `ctx.plugin` and `ctx.scope` are public, and the composition is the profile
        the person wrote, which `--dump-config` prints for anyone who can start the
        process.
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
            # dsh's `isolate.fs`. The row's own scope becomes an isolation
            # boundary and its own provisioning realm — `scope()` is exactly
            # that, and it is the same scope an agent gets — and a second copy of
            # each named row is mounted *inside* it. That copy's `provide` lands
            # in the realm, so `ctx.fs` resolves to the private instance for this
            # row and for everything beneath it, while every other row keeps the
            # shared one. Nothing is redirected: `_provision` walks up from the
            # realm and finds the nearer provision first.
            #
            # Settled before this row mounts, and then **checked**. `has(key)` is
            # true from the realm the moment root provides the key, so the
            # isolating row is ready immediately — against the shared service —
            # and only registration order would make the private copy win. A
            # reconcile here fixes the order for a copy that is ready, and does
            # nothing for one that is not: a copy whose own `inject` is unmet
            # stays waiting, the row activates against root's instance, and the
            # realm silently falls through to exactly what it was meant to
            # replace. So a private copy that did not activate is a refusal,
            # naming the key it lacks: what it needs has to be provided above the
            # realm, by a row earlier in the profile.
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
            # **Transparent, and only this mount** (A9). This row's activation
            # scope would otherwise inherit the realm's isolation, and `reaches`
            # answers "nobody" for it — no dispatch, and no tool, prompt
            # section, fs screen or skill restriction it registers is visible
            # either. It is a deployment row with a narrower view of one
            # service, so it gets a deployment row's visibility.
            #
            # Never the private copies above: a second `fs` that answered every
            # dispatch would double-handle events with the instance it exists to
            # shadow, and register a second copy of whatever it registers.
            forks[row.id] = realm.plugin(
                resolve_plugin(row.name), interpolate(row.config), transparent=True, row=row.id
            )
        await ctx.reconcile()
        # The one moment a composed profile is whole and nothing has run yet, so
        # a row can refuse the deployment it finds itself in (E8). `serial`
        # rather than `emit`: a listener that raises must stop the process, and
        # a contained emit would swallow exactly the refusal that matters. A row
        # cannot check this in its own `apply` — a backend it depends on may be
        # layered after it, so a verdict computed then would be wrong for
        # precisely the profile that orders things that way.
        await ctx.serial("profile/mounted")
        return mount


MOUNT: ServiceKey[Mount] = ServiceKey("mount")
"""The mount itself, provided before the first row so a seam can ask the
composition what it is (`ph.seams.topology`)."""

PROJECT_ROOT: ServiceKey[Path] = ServiceKey("project_root")
"""Where this mount works: the directory a session's own header names (P5-14).

**Declared and provided beside `MOUNT` because it is the same kind of fact** —
something true of *this* mount rather than of the profile, needed by a row while
it applies. `fs-local` reads it as its root, and `workspace-lifecycle` then
branches its worktrees from it. A profile setting could not carry it: one daemon
mounts one composition many times, once per session, and those sessions are in
different repositories; composing a profile per root would re-import every
plugin to change one path.

It lived in `ph.keys` and was provided by `ph_app.runtime` — the key in the
package that reads it, the provision in the package above. That split made the
fact reachable only through an undocumented protocol (provide it, *then* mount),
so every other caller of `Profile.mount` — the tests, the plugins' own mounts, a
library embedder — had no way to say where it worked and silently took
`Path.cwd()`. Now the door that needs it takes it."""


@dataclass(slots=True)
class Mount:
    """What one `Profile` became on one context — the half a `Profile` cannot know.

    The composition is `profile`; this is what happened when it was mounted: the
    fork each row became, on the root it was mounted on. Two objects because they
    are two lifetimes — a `Profile` outlives every mount of it, a `Mount` goes with
    its root — and so that nothing composes twice: the daemon composes once and
    mounts per root.
    """

    profile: Profile
    root: Context
    """The deployment context, for `topology`'s realm walk. Held here so a reader
    need not know to pass the *root* rather than its own scope — a walk from a
    row's scope finds no realms and reports "none", silently."""
    forks: dict[str, ForkScope] = field(default_factory=dict)
    """Every mounted plugin by row id. A private copy mounted into a realm is keyed
    `"<isolating row>/<source row>"`, which is also how `topology` labels it."""
    reconfigured: set[str] = field(default_factory=set)
    """Row ids re-applied live on this mount — what `topology` marks.

    A set, not the configs: the only question anyone asks of it is whether a row
    was, and the config each fork runs is already on the fork (`ForkScope.config`).
    Keeping a parallel dict was a second statement of one fact, and it retained
    every superseded config for the life of the mount."""

    async def reconfigure(self, row_id: str, config: object) -> ForkScope:
        """Re-apply one row with a new config, on this mount, and touch nothing else.

        **Cordis's own shape, used for what it is for.** A row's `apply` registers
        into seams, and every registration is an effect of the row's activation
        scope (I2): disposing the fork unwinds them all, and mounting the row again
        with the new config makes them all again. No other row is asked to do
        anything — a seam whose slot was released and refilled answers its next
        call with the new value, and an agent mid-command notices only when its
        next command asks. That is what lets `/sandbox allow host` take effect
        with the agent still running.

        Per *mount*, not per profile: a daemon mounts one `Profile` once per root
        and `Row` is frozen, so the new config is recorded here and `topology`
        marks the row `reconfigured live`. Called only from `ph.session_profile` —
        its `override`, which records the change in the session's log before this
        runs, and `opened`, which brings a mount to what that log already says (S4).
        `test_session_profile` holds every caller to it.

        Refused for a row that is disabled (there is no fork to replace; enable it
        in the profile), that isolates others (its realm holds private copies this
        would orphan), or that others isolate (their private copies would keep the
        old config while the shared instance moved).
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
        """Row ids whose plugin is not active, whatever is keeping it down.

        Through `state` rather than `not fork.active`, because the two stopped
        meaning the same thing when `failed` joined the set: this said "an unmet
        `inject` key" and a row whose `apply` raised now lands in the same list
        for a different reason. `_state` is where a caller reads which.
        """
        return [row_id for row_id, fork in self.forks.items() if fork.state != "active"]

    def topology(self) -> list[tuple[str, str]]:
        """What the mount *became*, row by row — the half `Profile.dump()` cannot show.

        `dump()` is the composition before anything runs, and it is honest about
        that. But a row that mounted and never activated — an unmet `inject` key
        — looks identical there to one that runs, and a reader of the YAML has no
        way to tell which they have. dsh's rule is that the dump has to show what
        *is* running, because once structure comes from configuration the static
        code no longer says. `inactive()` had that answer and nothing called it.

        One line per row: its state (`_state` — active, activating, waiting on a
        key, unwound, unmounted), what it injects, and which layer put it there.
        Disabled rows are listed too — "this was turned off by `rlm-stable.yaml`"
        is what a person asking "why isn't X running" needs, and omitting it would
        make a disabled row indistinguishable from an absent one. Then the
        isolated realms under the root: none at `phern doctor` time, since an agent's
        scope is created when it runs, and that absence is stated rather than left
        as a missing line.
        """
        lines: list[tuple[str, str]] = []
        for row in self.profile.rows:
            fork = self.forks.get(row.id)
            # The last two path components — `bundles/base.yaml`,
            # `ph_rlm/bundle.yaml`, `profiles/rlm-stable.yaml` — because every
            # bundle file is called `bundle.yaml` and its directory is the name
            # that distinguishes them, while the full absolute path is a table
            # column nobody can read.
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
