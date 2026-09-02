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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from .context import Context, ForkScope
from .errors import LoaderError
from .events import events

__all__ = [
    "ENTRY_POINT_GROUP",
    "Loader",
    "Row",
    "compose_rows",
    "entry_point_targets",
    "import_plugin_modules",
    "interpolate",
    "load_profile_documents",
    "resolve_entry_point",
    "resolve_plugin",
    "safe_yaml_load",
]

ENTRY_POINT_GROUP = "ph.plugins"

_ENV_PATTERN = re.compile(r"\$\{env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
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

    `SafeLoader` already refuses `!!python/object`. This subclass additionally
    refuses timestamps and sexagesimals, so a row value that looks like a date
    stays the string the author wrote — a config file is data, and a silent
    type change is the same class of surprise as evaluation.
    """


for _tag in ("tag:yaml.org,2002:timestamp",):
    for _first, _resolvers in list(SafeRowLoader.yaml_implicit_resolvers.items()):
        SafeRowLoader.yaml_implicit_resolvers[_first] = [
            (tag, regexp) for tag, regexp in _resolvers if tag != _tag
        ]


def _reject_unknown_tag(loader: yaml.Loader, suffix: str, node: yaml.Node) -> Any:
    raise LoaderError(
        f"pH config is data, not code: tag '!{suffix}' at line "
        f"{node.start_mark.line + 1} is not allowed"
    )


SafeRowLoader.add_multi_constructor("!", _reject_unknown_tag)
SafeRowLoader.add_multi_constructor("tag:", _reject_unknown_tag)


def safe_yaml_load(text: str, *, origin: str = "<string>") -> Any:
    """Parse YAML with no code evaluation and no implicit date coercion."""
    try:
        return yaml.load(text, Loader=SafeRowLoader)
    except yaml.YAMLError as error:
        raise LoaderError(f"{origin}: {error}") from error


# --------------------------------------------------------------------- rows --


@dataclass(frozen=True, slots=True)
class Row:
    """One composed profile row."""

    id: str
    name: str
    config: Any = None
    disabled: bool = False
    layer: str = ""
    """Which profile document contributed this row's current config."""
    isolate: dict[str, Any] | None = None
    """Row ids this row wants private copies of, each with a config override or `None`.

    dsh's `isolate.fs`: an isolated realm for one service. Here it is spelled
    against **row ids** rather than service keys, because a row does not declare
    what it provides — `provide` is a runtime call — so `fs` names the row whose
    `apply` provides `ctx.fs`, and the loader mounts a second copy of that row
    inside this row's realm. For the seams the two spellings coincide by
    convention (`fs` → `ctx.fs`, `tools` → `ctx.tools`); where they do not, the
    row id is the one that can be checked at compose time."""

    def to_dump(self) -> dict[str, Any]:
        dump: dict[str, Any] = {"id": self.id, "name": self.name}
        if self.config is not None:
            dump["config"] = self.config
        if self.disabled:
            dump["disabled"] = True
        if self.isolate is not None:
            # Always the mapping: `isolate: [fs]` dumps as `{fs: null}`, which
            # `_as_isolate` reads back to the same thing. One dump shape.
            dump["isolate"] = dict(self.isolate)
        dump["layer"] = self.layer
        return dump


# ------------------------------------------------------------ interpolation --


def interpolate(value: Any, env: Mapping[str, str] | None = None) -> Any:
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


def evaluate_predicate(value: Any, env: Mapping[str, str] | None = None) -> bool:
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


def _as_rows(entries: Iterable[Any], layer: str) -> list[Row]:
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


def _as_isolate(value: Any, layer: str) -> dict[str, Any] | None:
    """`isolate:` as a list of row ids, or a mapping of row id to config override.

    Two spellings for one fact, and both normalise to the mapping: `[fs]` is
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


def _apply_patch(rows: list[Row], patch: Mapping[str, Any], layer: str) -> list[Row]:
    """Apply one patch entry to the composed row list."""
    unknown = set(patch) - {"insert", "id", "config", "disabled", "remove", "isolate"}
    if unknown:
        raise LoaderError(f"{layer}: patch has unknown keys {sorted(unknown)}")
    if "insert" in patch:
        inserted = patch["insert"]
        if not isinstance(inserted, list):
            raise LoaderError(f"{layer}: insert: must be a list of rows")
        existing = {row.id for row in rows}
        for row in _as_rows(inserted, layer):
            if row.id in existing:
                raise LoaderError(
                    f'{layer}: insert would duplicate row id "{row.id}"; address it '
                    "by id to replace its config instead"
                )
            rows.append(row)
        return rows
    row_id = patch.get("id")
    if not isinstance(row_id, str):
        raise LoaderError(f"{layer}: a patch must carry either insert: or id:")
    for index, row in enumerate(rows):
        if row.id != row_id:
            continue
        if patch.get("remove") is True:
            del rows[index]
            return rows
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
        return rows
    raise LoaderError(f'{layer}: no row with id "{row_id}" to patch')


def _check_isolation(rows: Sequence[Row]) -> None:
    """Every `isolate:` names a row that exists, is enabled, and is not itself.

    Checked once the layers are composed rather than at mount, so `--dump-config`
    refuses the same profile `ph` would — a private copy of a row a later layer
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


def compose_rows(documents: Sequence[tuple[str, Any]]) -> list[Row]:
    """Compose ordered profile documents into the final row list.

    Each document is either a plain list of rows or a list of patch entries.
    Rows keep file order; patches address rows by id.
    """
    rows: list[Row] = []
    for layer, document in documents:
        if document is None:
            continue
        if not isinstance(document, list):
            raise LoaderError(f"{layer}: a profile document must be a list")
        for entry in document:
            if not isinstance(entry, dict):
                raise LoaderError(f"{layer}: entry must be a mapping, got {entry!r}")
            if {"insert", "remove"} & set(entry) or ("id" in entry and "name" not in entry):
                rows = _apply_patch(rows, entry, layer)
            else:
                rows.extend(_as_rows([entry], layer))
    _check_isolation(rows)
    return rows


ProfileLayer = Path | tuple[str, Any]
"""One layer of a profile: a YAML file, or a `(name, document)` already parsed.

The second form is what lets a layer come from somewhere other than a file
without every caller that mounts a profile learning a second type — a
`--patch` on the command line is a document with no path, and it composes like
any other layer, with its name as its provenance."""


def load_profile_documents(layers: Sequence[ProfileLayer]) -> list[tuple[str, Any]]:
    return [
        layer
        if isinstance(layer, tuple)
        else (str(layer), safe_yaml_load(layer.read_text(encoding="utf-8"), origin=str(layer)))
        for layer in layers
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


def resolve_entry_point(group: str, name: str, *, default_attribute: str = "") -> Any:
    """Import what `name` registers in `group`, or `None` if nothing does.

    The mechanical half of resolution — look up, import, `getattr` — with no
    policy: a caller that wants an exception raises its own, and one that wants
    to offer an alternative gets `None`. Both `resolve_plugin` and
    `ph.bundles.resolve_bundle` are thin wrappers over this, so a third group
    does not bring a third copy of the importlib dance.
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
    """`active · injects …` or `waiting on <the unmet key> · injects …`."""
    injects = ", ".join(fork.injects) or "nothing"
    if fork.active:
        return f"active · injects {injects}"
    return f"waiting on {', '.join(fork.waiting_on)} · injects {injects}"


def import_plugin_modules() -> list[ModuleType]:
    """Import the module behind every registered plugin, in name order.

    Event declarations live in the modules that own them, so a tool that wants
    the complete registry — `ph events` — imports the plugin surface rather
    than a hand-kept list. Third-party wheels are covered by the same call.
    """
    modules: list[ModuleType] = []
    for name in sorted(_entry_point_targets()):
        module_path = _entry_point_targets()[name].partition(":")[0]
        modules.append(importlib.import_module(module_path))
    return modules


def resolve_plugin(name: str) -> Any:
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
class Loader:
    """Composes a profile and mounts its rows onto a context."""

    documents: list[tuple[str, Any]] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    forks: dict[str, ForkScope] = field(default_factory=dict)
    """Every mounted plugin by row id. A private copy mounted into a realm is keyed
    `"<isolating row>/<source row>"`, which is also how `topology` labels it."""

    @classmethod
    def from_documents(cls, documents: Sequence[tuple[str, Any]]) -> Loader:
        return cls(documents=list(documents), rows=compose_rows(documents))

    @classmethod
    def from_paths(cls, paths: Sequence[ProfileLayer]) -> Loader:
        return cls.from_documents(load_profile_documents(paths))

    def enabled_rows(self) -> Iterator[Row]:
        return (row for row in self.rows if not row.disabled)

    def dump(self) -> list[dict[str, Any]]:
        """The composed row list, for `--dump-config`."""
        return [row.to_dump() for row in self.rows]

    async def mount(self, ctx: Context) -> None:
        """Mount every enabled row, then settle the tree.

        Rows mount in file order, but nothing runs until `reconcile()`:
        activation is service-availability driven, so a row that needs `llm`
        waits for whichever row provides it regardless of where it sits.
        """
        by_id = {row.id: row for row in self.rows}
        for row in self.enabled_rows():
            if not row.isolate:
                self.forks[row.id] = ctx.plugin(resolve_plugin(row.name), interpolate(row.config))
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
                privates[source_id] = self.forks[f"{row.id}/{source_id}"] = realm.plugin(
                    resolve_plugin(source.name), interpolate(config)
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
            self.forks[row.id] = realm.plugin(resolve_plugin(row.name), interpolate(row.config))
        await ctx.reconcile()
        # The one moment a composed profile is whole and nothing has run yet, so
        # a row can refuse the deployment it finds itself in (E8). `serial`
        # rather than `emit`: a listener that raises must stop the process, and
        # a contained emit would swallow exactly the refusal that matters. A row
        # cannot check this in its own `apply` — a backend it depends on may be
        # layered after it, so a verdict computed then would be wrong for
        # precisely the profile that orders things that way.
        await ctx.serial("profile/mounted")

    def inactive(self) -> list[str]:
        """Row ids whose plugin never activated — an unmet `inject` key."""
        return [row_id for row_id, fork in self.forks.items() if not fork.active]

    def topology(self, ctx: Context) -> list[tuple[str, str]]:
        """What the mount *became*, row by row — the half `dump()` cannot show.

        `dump()` is the composition before anything runs, and it is honest about
        that. But a row that mounted and never activated — an unmet `inject` key
        — looks identical there to one that runs, and a reader of the YAML has no
        way to tell which they have. dsh's rule is that the dump has to show what
        *is* running, because once structure comes from configuration the static
        code no longer says. `inactive()` had that answer and nothing called it.

        One line per row: whether it activated, what it injects, which key it is
        waiting on when it did not, and which layer put it there. Disabled rows
        are listed too — "this was turned off by `rlm-stable.yaml`" is what a
        person asking "why isn't X running" needs, and omitting it would make a
        disabled row indistinguishable from an absent one. Then the isolated
        realms reachable from `ctx`: none at `ph doctor` time, since an agent's
        scope is created when it runs, and that absence is stated rather than
        left as a missing line.
        """
        lines: list[tuple[str, str]] = []
        for row in self.rows:
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
            lines.append((row.id, f"{_state(fork)} · from {layer}"))
            for source_id, override in (row.isolate or {}).items():
                private = self.forks[f"{row.id}/{source_id}"]
                how = "own config" if override is None else "overridden config"
                lines.append(
                    (
                        f"{row.id}/{source_id}",
                        f"{_state(private)} · private copy in realm:{row.id} · {how}",
                    )
                )
        realms = [node.path for node in ctx.descendants() if node.isolation is node]
        lines.append(
            (
                "isolated realms",
                ", ".join(realms) or "none — an agent's scope is created when it runs",
            )
        )
        return lines
