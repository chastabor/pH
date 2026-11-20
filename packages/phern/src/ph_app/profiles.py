"""Profile resolution: which bundle documents compose a run.

A profile is an ordered list of YAML documents. The shipped documents are in
`ph.bundles` and in this package. A named profile of the user is a file at
`$PH_HOME/profiles/<name>.yaml`. It `extends` one shipped profile and holds only
the rows that differ (`ph_app.named_profiles`). A deployment can therefore
change a row by id and does not have to fork a bundle.

**There are three owners, and each one sets one kind of row** (decision 23).
Every row declares what its settings shape (`Affects`). Each layer that a user
writes may set one kind:

| layer | sets |
|---|---|
| The shipped documents of pH, `presentation.yaml` included | any |
| The `rows:` of `$PH_HOME/daemon.yaml`, and the flags of the daemon | `deployment` |
| The rows of `$PH_HOME/profiles/<name>.yaml`, its drop-ins, `--patch` | `environment` |

The user has no layer here that sets presentation rows. Those rows ship in
every profile, and the user hides a screen in the `tui.json` of the TUI.
`compose_rows` refuses a row that is set in the wrong layer, and the refusal
names the configuration that the row belongs in.

@module ph_app.profiles
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import typer
from pydantic import ValidationError

from ph.bundles import BASE, HEADLESS, resolve_bundle
from ph.cordis import (
    LoaderError,
    Profile,
    ProfileDocument,
    compose_rows,
    load_profile_documents,
    sparse_entries,
)
from ph.cordis.loader import safe_yaml_load
from ph.cordis.plugin import Affects
from ph.documents import decode_document
from ph.host import host_config_path, load_host_config
from ph.json import JsonObject
from ph.paths import PathRoots, resolve_roots, write_atomic
from ph.session_profile import (
    LoggedEnvironment,
    ProfileBase,
    ProfileChange,
    base_of,
    override_documents,
    profile_change,
    rebuilt,
    withdrawn_note,
)
from ph.text import count_of

from .console import detail, fail
from .named_profiles import NamedProfile, parse_named_profile, render_named_profile

__all__ = [
    "DEFAULT_PROFILE",
    "NAMED",
    "PRESENTATION",
    "PROFILES",
    "PROFILE_DIR",
    "Bundle",
    "ModelOption",
    "NamedProfileStore",
    "NoteDoor",
    "PatchOption",
    "ProfileOption",
    "ProfilePlan",
    "ProviderOption",
    "SaveRefused",
    "StartingProfile",
    "available_profiles",
    "base_documents",
    "compose_profile",
    "composed_like",
    "host_rows",
    "kept_note",
    "log_host",
    "named_version",
    "person_profiles",
    "profile_documents",
    "profile_file",
    "profile_name",
    "profile_or_exit",
    "profile_plan",
    "read_named_profile",
    "resolve_profile",
    "save_session",
    "session_profile",
    "sparse_text",
    "start_documents",
    "unfolded_profiles",
]

PROFILE_DIR = Path(__file__).parent / "profiles"

PRESENTATION = PROFILE_DIR / "presentation.yaml"
"""The presentation rows that this package ships. They are layered into every
named profile.

They are outside the `PROFILES` table on purpose. A screen is not part of the
environment that a profile names, so no entry in that table says whether the
profile has a screen. Every named profile mounts these rows, because the daemon
that holds a headless run is also the daemon that a TUI attaches to, and a
screen that nothing draws costs nothing. `base.yaml` gives the same reason for
`tui-screens`."""


@dataclass(frozen=True, slots=True)
class Bundle:
    """A layer that another distribution provides. It is resolved late.

    A profile is an ordered list of layers. P3-20 added one thing: one *kind* of
    layer is not a path that this package can name. A layer of that kind is
    found through the `ph.bundles` entry-point group, because `ph_app` must not
    *import* `ph_rlm`. The same rule lets the app read `subagent/*` events
    without importing the row that emits them.

    **The `phern` distribution depends on those wheels, and this module still
    may not name them.** Shipping them together decides what is present by
    default, and nothing more. Every bundle reaches this dataclass in the way
    that a third-party wheel would. So the third-party path is in real use, and
    is not only possible in theory.
    """

    name: str
    required: bool = True
    """Whether a profile that names this bundle is refused when the bundle is
    missing, or composes without it.

    **Required is the old behavior and stays the default.** `rlm` without the
    RLM bundle is not a degraded `rlm`. It is a profile whose documents patch
    rows that do not exist, so it is better not to offer it. `rlm-stable.yaml`
    shows this in one line: it arms `tool-todo`, which only the stabilize
    bundle mounts.

    Optional is for a layer that is *additive* and that every posture wants.
    Nothing outside the stabilize bundle addresses a stabilize row, so a profile
    without the bundle composes and never compacts. The flag does not say how
    much the bundle is wanted. It says whether anything else in the profile
    refers to the bundle.
    """


Layer: TypeAlias = "Path | Bundle"

STABILIZE: Bundle = Bundle("stabilize", required=False)
"""Context management. It is layered into every shipped profile.

**A session that silently grows past its window is a defect in any posture.**
It is not a feature of an interactive posture, so a profile does not opt into
this layer. `ph-base` mounts the compaction *seam* with no engine, and says that
a deployment that wants the plain harness should get it. That is still true of
the bundle documents. What changed is that every profile that this package
*offers* now layers the engine over them.

**The layer is optional and not required, and it stays optional although
`phern` now depends on the wheel.** `available_profiles` checks whether bundles
*resolve*, and not what the manifest asked for. So the question that this flag
answers is "what happens to `--profile tui` when the bundle is not there". The
answer must stay "it still composes, without compaction", for the two cases
that outlive any dependency list:

- Somebody uninstalled `ph-stabilize` from a `phern` environment.
- Somebody builds on `ph-core` with a front end of their own.

A *required* layer here would take a working profile away to add a feature. The
comment on `rlm-indexed` below refuses exactly that trade.

The cost is that the behavior of a profile now depends on what is installed,
which it never did before. For that reason this is one named layer and not a
habit. `phern doctor` reports what activated, and this is true of exactly one
bundle."""

TUI_LAYERS: tuple[Layer, ...] = (BASE, HEADLESS, STABILIZE, PROFILE_DIR / "tui.yaml")
"""The interactive posture: `headless` plus one document, `tui.yaml`, which
changes two rows. A user is present to answer the seams, so the workspace is
writable (see `tui.yaml`).

`STABILIZE` comes before the document of the profile, so that `tui.yaml` and any
layer after it can address a stabilize row by id. When a bundle is layered after
the document that patches it, its patches are silently undone."""

RLM_LAYERS: tuple[Layer, ...] = (*TUI_LAYERS, Bundle("rlm"))
"""The interactive posture plus the RLM bundle, because a user is present for
the approvals that the dispatches of Code Mode raise (P3-20).

The tuple has a name for the same reason as `TUI_LAYERS` above. `rlm-stable`
*is* "rlm plus stabilize". A second list of the same layers would let the two
drift apart while a comment still claimed that they could not."""

RLM_STABLE_LAYERS: tuple[Layer, ...] = (
    *RLM_LAYERS,
    Bundle("stabilize"),
    PROFILE_DIR / "rlm-stable.yaml",
)
"""Everything, with the gates on. The tuple has a name for the same reason as
the two above.

`rlm-indexed` *is* "rlm-stable plus the indexing bundles". The first draft of
`rlm-indexed` listed these three layers again by hand. That is exactly the drift
that `RLM_LAYERS` was introduced to stop: a row added to `rlm-stable` would
silently have stopped reaching a profile whose comment claimed to be built from
it."""


PROFILES: dict[str, tuple[Layer, ...]] = {
    # Every entry carries `STABILIZE`, `base` included. The bundle *documents*
    # still ship the plain harness: `ph-base` mounts the compaction seam with no
    # engine. This table says that no profile that pH offers by name has a
    # conversation that grows until the provider refuses it.
    "base": (BASE, STABILIZE),
    "headless": (BASE, HEADLESS, STABILIZE),
    "tui": TUI_LAYERS,
    # Real providers layer onto base. The fake adapter is deliberately absent, so
    # that a misconfigured key fails loudly and does not silently answer "ok".
    "deepseek": (BASE, STABILIZE, PROFILE_DIR / "deepseek.yaml"),
    # A server on localhost and not a hosted service. The differences are in the
    # document: the window is the window of one slot and not of the whole server,
    # and the profile claims none of the media that the hosted default claims.
    "llama": (BASE, STABILIZE, PROFILE_DIR / "llama.yaml"),
    "anthropic": (BASE, STABILIZE, PROFILE_DIR / "anthropic.yaml"),
    "google": (BASE, STABILIZE, PROFILE_DIR / "google.yaml"),
    "rlm": RLM_LAYERS,
    # Everything, with the gates on (P4-15). This is `rlm` plus `stabilize`, plus
    # the profile that turns on the two rows that those bundles ship disabled. A
    # bundle that armed them when it was layered would make "I want offload" mean
    # "and also a tool, and also a corpus".
    "rlm-stable": RLM_STABLE_LAYERS,
    # `rlm-stable` plus the two indexing bundles. The RLM asks a codebase and a
    # document corpus about themselves and does not read them (`code_graph`,
    # `text_search`). Under Code Mode both tools arrive as
    # `await tools.<name>(...)` with no work from either package. Every
    # registered tool is in the SDK listing, which is what C1 means.
    #
    # **This is its own profile, and not rows added to `rlm-stable`.** Whether a
    # profile can be composed is decided by whether its *bundles* resolve. If
    # these bundles were added to `rlm-stable`, that profile would be unavailable
    # on an install that lacks one of their distributions. That would take a
    # working profile away to add an optional feature. Here, an install that
    # lacks one of them is not offered this profile, and `resolve_profile` names
    # the package to install.
    #
    # `utcp` comes first. The `code-graph` row registers the manual of its
    # package with `utcp`, so the row that it waits on is in the profile that
    # asks for the tools.
    "rlm-indexed": (
        *RLM_STABLE_LAYERS,
        Bundle("utcp"),
        Bundle("code-graph"),
        Bundle("text-index"),
    ),
    # `rlm` plus a model that edits its own context (ph-clm): the context tools,
    # and the context as a file that a cell can edit. It is built on `rlm` and not
    # on `rlm-stable`, so that what is measured is the editing, with compaction
    # still the backstop.
    "rlm-clm": (*RLM_LAYERS, Bundle("clm")),
}


def _resolve_layer(layer: Layer) -> Path | None:
    """One layer as a path, or `None` when this install cannot provide it."""
    return resolve_bundle(layer.name) if isinstance(layer, Bundle) else layer


def _composed(layers: Sequence[Layer]) -> list[Layer]:
    """The layers as they will be composed: one entry for each bundle.

    **A bundle that is named twice is one layer, and it is required if either
    naming was required.** `rlm-stable` is the case. It declares stabilize
    *required*, because its document arms `tool-todo`, and composing without the
    bundle would patch a row that is not there. It also inherits the optional
    stabilize layer that every profile carries. The merge keeps the requirement
    and the first position. "First occurrence wins" alone would have separated
    the position of the bundle from whether it is required.

    `Path` layers are left exactly as written. An earlier form removed duplicate
    paths too, and that silently dropped a document that a profile had
    deliberately layered twice. A bundle is a named thing that can be asked for
    twice by accident, and a path is not.
    """
    required: dict[str, bool] = {}
    for layer in layers:
        if isinstance(layer, Bundle):
            required[layer.name] = required.get(layer.name, False) or layer.required
    composed: list[Layer] = []
    seen: set[str] = set()
    for layer in layers:
        if not isinstance(layer, Bundle):
            composed.append(layer)
        elif layer.name not in seen:
            seen.add(layer.name)
            composed.append(Bundle(layer.name, required=required[layer.name]))
    return composed


def _missing_required(layers: Sequence[Layer]) -> str | None:
    """The first required bundle that this install cannot provide, or `None` when
    it provides every one.

    **This is the one predicate**, for the reason that `profile_file` gives for
    itself. `available_profiles` and `resolve_profile` asked the same question
    in two ways, and the second form was an inverted double negative. Two
    predicates for one question is how a `--help` line and a command line come
    to disagree.
    """
    for layer in _composed(layers):
        if isinstance(layer, Bundle) and layer.required and _resolve_layer(layer) is None:
            return layer.name
    return None


def available_profiles() -> list[str]:
    """Every profile that this install can compose: the shipped profiles, and the
    named profiles of the user over them.

    The answer comes from the same resolution that `resolve_profile` performs,
    so a profile is never offered and then refused. Two predicates for one
    question is how a `--help` line and a command line come to disagree. A file
    of the user that cannot be read is not offered either. When the user names
    it, the refusal says why.
    """
    shipped = {name for name, layers in PROFILES.items() if _missing_required(layers) is None}
    named = {one.path.stem for one in person_profiles() if one.extends in shipped}
    return sorted(shipped | named)


def person_profiles() -> list[NamedProfile]:
    """Every named profile that the user wrote under `$PH_HOME/profiles/` and that
    can be read.

    A file that cannot be read is skipped here. It is refused by name when the
    user asks for it.
    """
    found: list[NamedProfile] = []
    directory = resolve_roots().profiles_dir()
    for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else []:
        try:
            found.append(read_named_profile(path, path.stem))
        except (LoaderError, OSError, ValueError):
            continue
    return found


def read_named_profile(path: Path, name: str) -> NamedProfile:
    """The named profile at `path`, run as `name`, against the shipped table."""
    return parse_named_profile(decode_document(path), path, name, shipped=PROFILES)


def unfolded_profiles() -> list[str]:
    """Profiles with a layer that `phern profiles fold` has not folded yet (item 0).

    Such a layer is a `<name>.d/` directory of drop-ins, or a file that is still
    in the list format from before S2. Both are read until they are folded, so
    nothing is dropped. `phern doctor` names them, so nothing is read silently
    either.
    """
    roots = resolve_roots()
    directory = roots.profiles_dir()
    if not directory.is_dir():
        return []
    names = {path.name.removesuffix(".d") for path in directory.glob("*.d") if path.is_dir()}
    names |= {one.path.stem for one in person_profiles() if one.legacy}
    return sorted(names)


def profile_file(name: str) -> Path | None:
    """The `.yaml` file that this `--profile` value names, or `None` when the value
    names a profile.

    **This is the one predicate**, because there were two and they disagreed.
    `available_profiles` in this module states the rule: "two predicates for one
    question is how a `--help` line and a command line come to disagree".
    `profile_name` had derived the answer again without the `.exists()` half. A
    mistyped path was therefore a file for one of them and a named profile for
    the other.
    """
    candidate = Path(name)
    return candidate if candidate.suffix in (".yaml", ".yml") and candidate.exists() else None


def resolve_profile(name: str) -> list[Path]:
    """The documents for `name`: the built-in layers first, then the layers of the
    user.

    A name that is a path is used directly. That makes a scenario test or a
    one-off deployment a single file and not an install step.

    This returns paths only, so it does not say what each layer may set.
    `profile_documents` is the function that reads the layers with that. Nothing
    in `ph_app` calls this function. The test suites of the workspace do: it is
    how a test mounts a shipped profile from its documents.
    """
    plan = _plan(name, resolve_roots())
    return [*plan.shipped, *([plan.named.path] if plan.named else []), *plan.dropins]


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """What composes a name: the layers of pH, the file of the user over them, and
    the drop-ins."""

    shipped: list[Path]
    named: NamedProfile | None = None
    dropins: list[Path] = field(default_factory=list)
    whole: ProfileDocument | None = None
    """A `--profile` path in the list format, already read. It is the whole
    composition. `shipped` names it only so that `resolve_profile` can list it."""


def profile_plan(name: str) -> ProfilePlan:
    """The layers that `name` composes from. This is the one answer to "file or
    named profile", and to which of the layers belongs to the user."""
    return _plan(name, resolve_roots())


def _plan(name: str, roots: PathRoots) -> ProfilePlan:
    """The layers for `name`. Whether it is a file or a named profile is decided
    one time, here.

    A path whose file is a list is a whole composition, deployment rows
    included. That is its purpose: one file is the scenario. So it is one layer
    of the kind that pH ships, with nothing over it. `presentation.yaml` is not
    over it either: a file that wants the trajectory screen inserts it. A path
    whose file has `extends:` and `rows:` is a named profile that is stored
    somewhere else, and it composes as a named profile.

    A name is a shipped profile, a named profile of the user over one, or both:
    the `tui.yaml` of the user over the shipped `tui`.
    """
    candidate = profile_file(name)
    if candidate is not None:
        raw = decode_document(candidate)
        if not isinstance(raw, Mapping):
            return ProfilePlan(shipped=[candidate], whole=ProfileDocument(str(candidate), raw))
        elsewhere = parse_named_profile(raw, candidate, None, shipped=PROFILES)
        return ProfilePlan(shipped=_shipped_layers(elsewhere.extends), named=elsewhere)
    overlay = roots.profile_overlay(name)
    named = read_named_profile(overlay, name) if overlay.is_file() else None
    extends = named.extends if named is not None else name
    return ProfilePlan(shipped=_shipped_layers(extends), named=named, dropins=_dropins(name, roots))


def _shipped_layers(name: str) -> list[Path]:
    """The layers that pH provides for the named profile `name`."""
    declared = PROFILES.get(name)
    if declared is None:
        raise ValueError(
            f'unknown profile "{name}"; available are '
            f"{', '.join(available_profiles())}, or pass a path to a .yaml"
        )
    missing = _missing_required(declared)
    if missing is not None:
        # The name of the package is the next step for the user.
        raise ValueError(
            f'profile "{name}" needs the "{missing}" bundle, which no installed '
            f"distribution provides; install ph-{missing} and try again"
        )
    layers: list[Path] = []
    for layer in _composed(declared):
        resolved = _resolve_layer(layer)
        if resolved is None:
            # An optional bundle that this install does not have.
            # `_missing_required` already refused every required one. This is
            # silent here and visible in `phern doctor`, which reports what
            # activated. A warning on every command would be a warning that
            # nobody reads, about a profile that composed exactly as this table
            # says it should.
            continue
        layers.append(resolved)
    layers.append(PRESENTATION)
    return layers


def _dropins(name: str, roots: PathRoots) -> list[Path]:
    """The drop-ins that `/sandbox` wrote over `name` before S4 made its changes
    overrides.

    They are returned in name order and compose after the file of the user, as
    they always did. They are read until `phern profiles fold` folds them into
    that file. Nothing writes a drop-in now.
    """
    directory = roots.profile_dropins(name)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix in (".yaml", ".yml") and path.is_file()
    )


def profile_name(profile: str) -> str | None:
    """The name that a `--profile` value names, or `None` for a path to a `.yaml`.

    This is what `Profile.name` records (`_composed_as`). A caller that writes
    under the name of a profile then knows where to write. It also knows *not*
    to write for a file profile, whose one document belongs to the user. The
    answer comes from `profile_file`, so it cannot disagree with what was
    resolved.
    """
    return None if profile_file(profile) is not None else profile


def _composed_as(documents: Sequence[ProfileDocument], profile: str) -> Profile:
    """`documents` composed as the profile that the `--profile` value `profile`
    names.

    This is the one place where `profile_name` meets `Profile.name`. The loader
    spells "no name" as a name that is left out, so a path composes with none.
    """
    name = profile_name(profile)
    if name is None:
        return Profile.from_documents(documents)
    return Profile.from_documents(documents, name=name)


DEFAULT_PROFILE = "headless"

ProfileOption: TypeAlias = Annotated[
    str, typer.Option("--profile", help="Profile name or path to a .yaml.")
]
"""Declared one time, so that two commands cannot come to disagree about what
`--profile` means, or about what an unknown profile costs. The second nearly
happened here.

**It is here and not in `cli.py`**, where it started. `cli.py` imports the
sub-apps, so a sub-app that wanted the alias had to import back into `cli.py`.
`phern workspaces gc` did exactly that: it reached for a private `_documents`
across the cycle. This module already owns what a profile *is*, and the flag
that names a profile belongs beside it. `console.py` was separated for the same
reason, and its docstring says so.
"""

PatchOption: TypeAlias = Annotated[
    list[str],
    typer.Option(
        "--patch",
        help="A profile patch as YAML — `{id: fs, config: {root: /tmp/x}}`, "
        "`{id: tool-todo, disabled: false}`, `{id: hitl, remove: true}`, or "
        "`{insert: [...]}`. Repeatable; applied last, as the `cli` layer.",
    ),
]
"""The third layer of dsh: bundle, profile, then a *patch from the command
line*. pH had that layer only as a file under `$PH_HOME/profiles/`. The grammar
is deliberately the grammar of a profile document, because a second spelling for
"change this row" is how the two come to accept different things.
`safe_yaml_load` parses the value, so the code-tag refusal that guards a file
also guards the flag."""


ProviderOption: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help="With --model, run on this exact route instead of one the profile lists.",
    ),
]
"""`--provider`, declared one time for `phern` and `phern daemon`. The reason is
the reason of `ProfileOption`: with two copies of one flag, two commands come to
mean different things by it. `ModelChoice.from_flags` reads it together with
`--model`."""

ModelOption: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="A model the profile lists, by key (`fast`); with --provider, a model name.",
    ),
]
"""`--model`, declared one time beside `--provider` for the same reason."""


CLI_LAYER = "cli"
"""The layer that a `--patch` composes under. `--dump-config` and the topology of
`phern doctor` print it as the provenance of the patch."""


def profile_documents(name: str) -> list[ProfileDocument]:
    """The layers of the profile as documents, each of which says what it may set.
    Every command composes through this function, and it raises what its parts
    raise.

    A caller uses this stage when it has a layer of its own to add before it
    composes: the `bench` document of the benchmark, or the overlay of a
    fixture. The kinds are in the table of the module docstring. The rows of
    `daemon.yaml` sit between the layers of pH and the layers of the user. They
    come after what they patch, and they are apart from the environment that a
    session profile names.
    """
    roots = resolve_roots()
    plan = _plan(name, roots)
    named = (
        [ProfileDocument(str(plan.named.path), plan.named.rows, sets="environment")]
        if plan.named is not None
        else []
    )
    shipped = [plan.whole] if plan.whole is not None else load_profile_documents(plan.shipped)
    return [
        *shipped,
        *_host_documents(roots),
        *named,
        *load_profile_documents(plan.dropins, sets="environment"),
    ]


def _host_documents(roots: PathRoots) -> list[ProfileDocument]:
    """The rows of `daemon.yaml`, as the deployment layer that every composition
    carries."""
    host = load_host_config(roots.home)
    if host.rows is None:
        return []
    return [ProfileDocument(str(host_config_path(roots.home)), host.rows, sets="deployment")]


def base_documents(extends: str) -> list[ProfileDocument]:
    """What a named profile over the shipped profile `extends` composes over.

    These are the layers of pH and the rows of the host, and none of the layers
    of the user. A saved profile is the difference from exactly this. The rows
    of the host are in both, so they cancel. What is left is the environment
    that the user chose.
    """
    return [*load_profile_documents(_shipped_layers(extends)), *_host_documents(resolve_roots())]


class SaveRefused(RuntimeError):
    """`/profile save` wrote nothing: the name, the file, or the round trip refused it."""


PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
"""What a saved profile may be called. The name is a file name under
`$PH_HOME/profiles` and a word that `--profile` takes. It has no separator, so
it cannot name a file anywhere else."""


def save_session(
    name: str, env: LoggedEnvironment, *, comment: str, replace: bool = False
) -> tuple[Path, list[JsonObject]]:
    """Write the environment of a session as the sparse named profile `name` (S7,
    decision 5).

    **The base is merged with the overrides**, as documents. The first documents
    are the shipped profile that the base extends. The layers of the user that
    the base was composed from come next, as they were recorded at that time.
    Every override comes after them. The documents are composed and not
    resolved. So what the user wrote, such as an `${env:...}` setting or a
    sparse config, is written as the user wrote it, and the defaults of pH are
    left to pH. Only the rows that differ are then written (`sparse_entries`),
    and the file is composed again. The file is kept only when it gives the
    environment that the documents give. Otherwise it is taken back, as
    `phern profiles fold` does. Returns the file and the entries that were
    written.

    :raises SaveRefused: for a name that is not a file name, a file that exists
        without `replace`, a base with no named profile to extend, or a round trip
        that does not agree.
    """
    if not PROFILE_NAME.match(name):
        raise SaveRefused(f'"{name}" is not a profile name: letters, digits, ".", "_" and "-"')
    base = env.base
    if base is None or not base.name:
        raise SaveRefused(
            "this session started on a profile file, not a named profile, so there is "
            "no shipped profile to save it over"
        )
    path = resolve_roots().profile_overlay(name)
    if path.exists() and not replace:
        raise SaveRefused(f"{path} exists; `--replace` writes over it")
    plan = profile_plan(base.name)
    extends = plan.named.extends if plan.named is not None else base.name
    shipped = base_documents(extends)
    wanted = [
        *shipped,
        *(
            ProfileDocument(source.layer, list(source.entries), sets="environment")
            for source in base.sources
        ),
        *override_documents(env.overrides),
    ]
    composed = Profile.from_documents(wanted, name=name)
    entries = sparse_entries(compose_rows(shipped), composed.rows)
    original = path.read_text(encoding="utf-8") if path.exists() else None
    write_atomic(path, render_named_profile(extends, entries, comment=comment))
    try:
        agrees = compose_profile(name).resolved({"environment"}) == composed.resolved(
            {"environment"}
        )
        failure = None if agrees else "composed again, it gives a different environment"
    except (LoaderError, OSError, ValueError, ValidationError) as error:
        failure = detail(error)
    if failure is not None:
        if original is None:
            path.unlink()
        else:
            write_atomic(path, original)
        raise SaveRefused(f"{name}: not saved — {failure}; {path} is as it was")
    return path, entries


def sparse_text(profile: Profile, *, extends: str, comment: str) -> str:
    """`profile` as the text of a sparse named profile over `extends`: only the rows
    that differ from it (`sparse_entries`)."""
    entries = sparse_entries(compose_rows(base_documents(extends)), profile.rows)
    return render_named_profile(extends, entries, comment=comment)


def start_documents(profile: Profile) -> list[ProfileDocument]:
    """The start options from the command line that `profile` was composed with:
    `--patch`, and the flags of the daemon. The rebuilt profile of a session
    takes them last too."""
    return [document for document in profile.documents if document.override]


def composed_like(name: str, profile: Profile) -> Profile:
    """The named profile `name` with the start options of `profile` after it, or
    `profile` itself when it is that profile. This raises and does not exit, for
    a caller that answers with a refusal of its own (the `session/new` of the
    daemon)."""
    if name == profile.name:
        return profile
    return compose_profile(name, then=start_documents(profile))


@dataclass(frozen=True, slots=True)
class StartingProfile:
    """What a session mounts, and how its named profile has moved since (S6)."""

    profile: Profile
    change: ProfileChange | None = None
    """The named profile as it composes now, compared with the version that the
    session starts on. It is `None` when the profile has not moved, or when
    there is nothing to compare: a new session, a `--profile` path, or a named
    profile that no longer composes."""
    adopting: ProfileBase | None = None
    """The version that this start applies, adopted since the base (S6). A start
    that the loader refuses takes this version back (`runtime.mount_session`)."""
    withdrawn: str | None = None
    """What this start took back, in the words of `withdrawn_note`, when the
    loader refused the version that the start was to apply. Otherwise `None`."""


def session_profile(requested: Profile, environment: LoggedEnvironment) -> StartingProfile:
    """The profile that a session mounts in (S5): the environment of its own log,
    when the log has one.

    A session whose log records a base is mounted from that log. Its profile is
    that base, or a version adopted since (S6), then its overrides in order,
    then the start options of `requested`. These go over the deployment and
    presentation rows of the profile that the base was recorded on, as that
    profile composes now. This is what makes the profile of a session its own.
    A daemon whose default moved, or a root that went idle and came back, still
    runs the session where it was. The profile that the base names is composed
    for its host rows, and to say how it has moved (`change`). When that
    profile no longer composes, because it was renamed or removed, the rows of
    `requested` serve in its place. The environment comes from the log in both
    cases.

    `environment` is what the log of the session says
    (`RecordedStart.environment`). It is empty for a new session, and for a
    session whose store its host cannot read before the mount. A session with
    no base, which is a new session or one from before S3, mounts `requested`
    and records its base from `requested` as it opens.
    """
    starts_on = environment.starts_on
    if environment.base is None or starts_on is None:
        return StartingProfile(requested)
    named: Profile | None = None
    if environment.base.name:
        try:
            named = composed_like(environment.base.name, requested)
        except (LoaderError, OSError, ValueError):
            named = None
    start = start_documents(requested)
    # An adoption that clears the overrides (`/profile use --clear`) mounts
    # without them. The start that applies the adoption logs the clears. A mount
    # that was composed with the overrides would run with settings that the log
    # is about to record as dropped.
    kept = () if environment.adopted_clears else environment.overrides
    profile = rebuilt(starts_on, named or requested, kept, then=start)
    change = profile_change(environment, base_of(named)) if named is not None else None
    return StartingProfile(profile, change, adopting=environment.adopted)


NoteDoor: TypeAlias = Literal["cli", "session"]
"""Where a user who reads `kept_note` acts on it: on the command line, or inside
the session (`/profile`). It decides which commands the note names."""


def kept_note(change: ProfileChange, session_id: str, *, door: NoteDoor = "cli") -> str:
    """One line for a session that is kept on its version of a named profile that
    moved (S6), or on the version that it had when a start took the new version
    back. This is the one wording. `cli` is for the one-shot path and the log of
    the daemon. `session` is for a terminal that attaches to the session, and
    the line is sent as `RootDescription.profile_note`."""
    name, count, yours = change.now.name, len(change.differences), change.yours
    if door == "cli":
        who, diff, take = (
            session_id,
            f"`phern profiles diff {name} --session {session_id}`",
            "`phern profiles adopt`",
        )
    else:
        who, diff, take = "this session", "/profile diff", f"/profile use {name}"
    if change.withdrawn:
        # The version was adopted, and taken back at the start that applied it.
        # Taking it again would only fail on it again, so the note offers what
        # changed.
        return f"{withdrawn_note(name, change.withdrawn)} {diff} lists what it changes."
    if yours == count:
        whose = "yours"
    elif yours:
        whose = f"{yours} yours, {count - yours} pH's"
    else:
        whose = change.ph_moved
    return (
        f"{name} has changed since {who} started ({count_of(count, 'setting')}, {whose}); "
        f"it runs on the version it started on. {diff} lists them; {take} takes them."
    )


def named_version(name: str) -> ProfileBase:
    """The named profile `name` as a session's base would record it now (S6)."""
    return base_of(compose_profile(name))


def log_host(name: str) -> Profile:
    """Only the host rows of the composition of `name`: what opening a stored log
    needs.

    These are the deployment rows (`affects`): the session store, where the
    store keeps its logs, and the rest of the host. No tool, workspace or model
    is mounted. So a log can be written to (`phern profiles adopt`) without
    starting what runs in it.
    """
    return host_rows(compose_profile(name))


def host_rows(profile: Profile) -> Profile:
    """The deployment rows of `profile` alone: the session store and the rest of
    the host, with nothing that runs. That is enough to write to a log without
    starting it."""
    rows = [row.to_entry() for row in profile.rows_of({"deployment"})]
    return Profile.from_documents([ProfileDocument("host", rows)], name=profile.name)


class NamedProfileStore:
    """`ctx.named_profiles`: the named profiles that this host composes (S7b). A
    profile that a parent assigns to its child is read from here as the
    narrowing that it is."""

    def compose(self, name: str) -> Profile:
        return compose_profile(name)


NAMED = NamedProfileStore()
"""The one store: it holds nothing, so every mount may share it."""


def compose_profile(name: str, *, then: Sequence[ProfileDocument] = ()) -> Profile:
    """The profile that `name` names, composed and ready to mount, with `then`
    (start options) after it.

    A test or a bench calls this, and so does a caller that answers a bad
    profile with a refusal of its own. A command that exits on a bad profile
    goes through `profile_or_exit`, which is this plus the exit code.
    """
    return _composed_as([*profile_documents(name), *then], name)


def profile_or_exit(
    profile: str, patches: Sequence[str] = (), *, deployment: Sequence[str] = ()
) -> Profile:
    """The composed profile, `--patch` entries included, or exit 2 with the reason.

    `patches` is the `--patch` of the user, which sets environment rows as a
    session profile does. `deployment` is a host flag written as the patch that
    it is, such as `phern daemon --max-concurrent-children`. It sets deployment
    rows as `daemon.yaml` does, and it comes after `daemon.yaml`. Both carry
    `cli` as their provenance.

    The refusal belongs to the command and not to the resolver.
    `profile_documents` raises a `ValueError` that already names the available
    profiles, and every caller wants that sentence on stderr under the same exit
    code.

    **Everything about the profile is refused here, one time, and nothing
    composes twice.** A command past this line holds a `Profile`. Every mode
    mounts that `Profile` and does not read or compose it again. So "no such
    profile", "not YAML" and a row id that does not exist are one refusal under
    one exit code wherever they are met. A bad row used to be exit 2 from
    `phern -p`, and "profile does not mount" under exit 1 from `phern doctor`.
    What stays with the mount belongs to the mount: a plugin that cannot be
    imported, an `isolate:` copy that never activates, a row that refuses the
    deployment.
    """
    try:
        documents = profile_documents(profile)
    except (ValueError, LoaderError, OSError) as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
    layered: tuple[tuple[Sequence[str], Affects], ...] = (
        (deployment, "deployment"),
        (patches, "environment"),
    )
    for texts, sets in layered:
        if texts:
            entries = [entry for text in texts for entry in _patch_entries(text)]
            documents.append(ProfileDocument(CLI_LAYER, entries, sets=sets, override=True))
    try:
        return _composed_as(documents, profile)
    except LoaderError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)


def _patch_entries(text: str) -> list[Any]:
    """One `--patch` value as the entries of a profile document.

    A mapping is one entry. A list is spliced in as several entries. There is
    deliberately no other shape check. `compose_rows` already decides between
    row and patch for each entry and refuses every malformed one, so a second
    checker here would be the grammar written twice. A first draft inferred
    `insert:` around a list whose entries all carried `name`. The loader
    already applies that rule to a bare row in any document. The draft returned
    the case that it did not infer as a nested list, which the loader could
    only refuse.
    """
    try:
        entry = safe_yaml_load(text, origin="--patch")
    except LoaderError as error:
        fail(f"[red]--patch {text!r}: {detail(error)}[/red]", code=2, cause=error)
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return [entry]
    fail(
        f"[red]--patch {text!r}: expected a mapping like `{{id: <row>, config: {{...}}}}`[/red]",
        code=2,
    )
