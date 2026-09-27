"""Profile resolution: which bundle documents compose a run.

A profile is an ordered list of YAML documents. The shipped ones live in
`ph.bundles` and this package; a person's named profile is a file under
`$PH_HOME/profiles/<name>.yaml` that `extends` one of them and holds only the rows
that differ (`ph_app.named_profiles`), so a deployment changes a row by id without
forking a bundle.

**Three owners, one kind each** (decision 23). Every row declares what its
settings shape (`Affects`), and each layer a person writes may set one kind:

| layer | sets |
|---|---|
| pH's shipped documents, `presentation.yaml` included | any |
| `$PH_HOME/daemon.yaml`'s `rows:`, and the daemon's own flags | `deployment` |
| `$PH_HOME/profiles/<name>.yaml`'s rows, its drop-ins, `--patch` | `environment` |

Presentation has no layer a person writes here: those rows ship in every
profile, and hiding a screen is the TUI's `tui.json`. A row set in the wrong
layer is refused by `compose_rows`, naming the configuration it belongs in.

@module ph_app.profiles
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypeAlias

import typer

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
from ph.paths import PathRoots, resolve_roots, write_atomic

from .console import detail, fail
from .named_profiles import NamedProfile, parse_named_profile, render_named_profile

__all__ = [
    "DEFAULT_PROFILE",
    "PRESENTATION",
    "PROFILES",
    "PROFILE_DIR",
    "Bundle",
    "ModelOption",
    "ProfileOption",
    "ProfilePlan",
    "ProviderOption",
    "available_profiles",
    "base_documents",
    "compose_profile",
    "person_profiles",
    "profile_documents",
    "profile_file",
    "profile_name",
    "profile_or_exit",
    "profile_plan",
    "read_named_profile",
    "resolve_profile",
    "save_named_profile",
    "sparse_text",
    "unfolded_profiles",
]

PROFILE_DIR = Path(__file__).parent / "profiles"

PRESENTATION = PROFILE_DIR / "presentation.yaml"
"""The presentation rows this package ships, layered into every named profile.

Outside the `PROFILES` table on purpose: a screen is not part of the environment
a profile names, so no entry there says whether it has one. Every named profile
mounts them, because the daemon that holds a headless run is also what a TUI
attaches to, and a screen nothing draws costs nothing (`base.yaml` says the same
of `tui-screens`)."""


@dataclass(frozen=True, slots=True)
class Bundle:
    """A layer another distribution provides, resolved late.

    A profile is an ordered list of layers; the only thing P3-20 added is that
    one *kind* of layer is not a path this package can name. It is discovered
    through the `ph.bundles` entry-point group — `ph_app` must not *import*
    `ph_rlm`, the same rule that lets the app read `subagent/*` events without
    importing the row that emits them.

    **The `phern` distribution depends on those wheels; this module still may
    not name them.** Shipping them together decides what is present by default
    and nothing more: every bundle reaches this dataclass the way a third-party
    wheel would, which is what keeps that path real rather than theoretical.
    """

    name: str
    required: bool = True
    """Whether a profile naming this bundle is refused without it, or composes
    without it.

    **Required is the old behavior and stays the default**: `rlm` without the
    RLM bundle is not a degraded `rlm`, it is a profile whose documents patch
    rows that do not exist, so it is better not offered. `rlm-stable.yaml` makes
    the point in one line — it arms `tool-todo`, which only the stabilize bundle
    mounts.

    Optional is for a layer that is *additive* and that every posture wants:
    nothing outside the stabilize bundle addresses a stabilize row, so a profile
    without it composes and simply never compacts. That is the difference this
    flag names — not "how much do we want it", but "does anything else here
    refer to it".
    """


Layer: TypeAlias = "Path | Bundle"

STABILIZE: Bundle = Bundle("stabilize", required=False)
"""Context management, layered into every shipped profile.

**A session that silently grows past its window is a defect in any posture**,
not a feature of an interactive one — so this is not a thing a profile opts
into. `ph-base` mounts the compaction *seam* with no engine and says a
deployment wanting the plain harness should get it; that remains true of the
bundle documents, and what changed is that every profile this package *offers*
now layers the engine over them.

**Optional rather than required, and it stays that way although `phern` now
depends on the wheel.** `available_profiles` gates on bundles *resolving*, not
on what the manifest asked for, so the question this flag answers is "what
happens to `--profile tui` when the bundle is not there" — and the answer has to
stay "it still composes, without compaction" for the two cases that outlive any
dependency list: somebody who uninstalled `ph-stabilize` from a `phern`
environment, and anybody building on `ph-core` with a front end of their own. A
*required* layer here would take a working profile away to add a feature, which
is precisely the trade `rlm-indexed`'s comment refuses further down.

The cost is that a profile's behavior now depends on what is installed, which
until now it never did. That is why it is one named layer rather than a habit:
`phern doctor` reports what actually activated, and there is exactly one bundle
this is true of."""

TUI_LAYERS: tuple[Layer, ...] = (BASE, HEADLESS, STABILIZE, PROFILE_DIR / "tui.yaml")
"""The interactive posture: `headless` plus one row. A person is present to
answer the seams, so the workspace is writable (see tui.yaml).

`STABILIZE` before the profile's own document, so `tui.yaml` — and any layer
after it — can address a stabilize row by id. A bundle layered after the
document that patches it is a bundle whose patches are silently undone."""

RLM_LAYERS: tuple[Layer, ...] = (*TUI_LAYERS, Bundle("rlm"))
"""The interactive posture plus the RLM bundle, because a person is present for
the approvals Code Mode's dispatches raise (P3-20).

Named for `TUI_LAYERS`' reason four lines up: `rlm-stable` *is* "rlm plus
stabilize", and re-listing the layers would let the two drift while a comment
went on claiming they could not."""

RLM_STABLE_LAYERS: tuple[Layer, ...] = (
    *RLM_LAYERS,
    Bundle("stabilize"),
    PROFILE_DIR / "rlm-stable.yaml",
)
"""Everything, with the gates on — and named for the reason the two above are.

`rlm-indexed` *is* "rlm-stable plus the indexing bundles", and the first draft
of it re-listed these three by hand, which is exactly the drift `RLM_LAYERS`
was introduced to stop: a row added to `rlm-stable` would silently have stopped
reaching a profile whose comment claimed to be built from it."""


PROFILES: dict[str, tuple[Layer, ...]] = {
    # Every entry carries `STABILIZE`, `base` included. The bundle *documents*
    # still ship the plain harness — `ph-base` mounts the compaction seam with
    # no engine — and what this table says is that no profile pH offers by name
    # is one whose conversation grows until the provider refuses it.
    "base": (BASE, STABILIZE),
    "headless": (BASE, HEADLESS, STABILIZE),
    "tui": TUI_LAYERS,
    # Real providers layer onto base; the fake adapter is deliberately absent so
    # a misconfigured key fails loudly instead of silently answering "ok".
    "deepseek": (BASE, STABILIZE, PROFILE_DIR / "deepseek.yaml"),
    # A server on localhost rather than a service, and the differences are in the
    # document: one slot's window rather than the whole server's, and none of the
    # media the hosted default claims.
    "llama": (BASE, STABILIZE, PROFILE_DIR / "llama.yaml"),
    "anthropic": (BASE, STABILIZE, PROFILE_DIR / "anthropic.yaml"),
    "google": (BASE, STABILIZE, PROFILE_DIR / "google.yaml"),
    "rlm": RLM_LAYERS,
    # Everything, with the gates on (P4-15). `rlm` plus `stabilize`, plus the
    # profile that turns on the two rows those bundles ship disabled — a bundle
    # that armed them on layering would make "I want offload" mean "and also a
    # tool, and also a corpus".
    "rlm-stable": RLM_STABLE_LAYERS,
    # `rlm-stable` plus the two indexing bundles: the RLM asks a codebase and a
    # document corpus about themselves instead of reading them (`code_graph`,
    # `text_search`). Under Code Mode both arrive as `await tools.<name>(...)`
    # with no work from either package — every registered tool is in the SDK
    # listing, which is what C1 means.
    #
    # **Its own profile rather than rows added to `rlm-stable`**, because
    # composability is decided by whether a profile's *bundles* resolve: adding
    # them there would make `rlm-stable` unavailable on an install without both
    # distributions, taking a working profile away to add an optional feature.
    # Here, an install missing one is simply not offered this profile, and
    # `resolve_profile` names the package to install.
    "rlm-indexed": (*RLM_STABLE_LAYERS, Bundle("code-graph"), Bundle("text-index")),
}


def _resolve_layer(layer: Layer) -> Path | None:
    """One layer as a path, or `None` when this install cannot provide it."""
    return resolve_bundle(layer.name) if isinstance(layer, Bundle) else layer


def _composed(layers: Sequence[Layer]) -> list[Layer]:
    """The layers as they will actually be composed: one entry per bundle.

    **A bundle named twice is one layer, required if either naming was.**
    `rlm-stable` is the case: it declares stabilize *required* — its document
    arms `tool-todo`, so composing without the bundle would patch a row that is
    not there — while inheriting the optional one every profile carries. Merging
    keeps the requirement and the first position, where "first occurrence wins"
    alone would have split one bundle's position from its requiredness.

    `Path` layers are left exactly as written. An earlier form deduplicated
    those too, which silently dropped a document a profile had deliberately
    layered twice — a bundle is a named thing that can be asked for twice by
    accident, and a path is not.
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


def _missing_required(layers: Sequence[Layer]) -> str:
    """The first required bundle this install cannot provide, or `""`.

    **The one predicate**, for the reason `profile_file` states about itself:
    `available_profiles` and `resolve_profile` were asking the same question two
    ways, and the second spelling was an inverted double negative. Two
    predicates for one question is how a `--help` line and a command line come
    to disagree.
    """
    for layer in _composed(layers):
        if isinstance(layer, Bundle) and layer.required and _resolve_layer(layer) is None:
            return layer.name
    return ""


def available_profiles() -> list[str]:
    """Every profile this install can actually compose: the shipped ones, and the
    person's named profiles over them.

    Answered by the same resolution `resolve_profile` performs, so a profile is
    never offered and then refused: two predicates for one question is how a
    `--help` line and a command line come to disagree. A person's file that does
    not read is not offered either; naming it gets the sentence for why.
    """
    shipped = {name for name, layers in PROFILES.items() if not _missing_required(layers)}
    return sorted(shipped | {one.name for one in person_profiles() if one.extends in shipped})


def person_profiles() -> list[NamedProfile]:
    """Every named profile a person wrote under `$PH_HOME/profiles/` that reads.

    A file that does not is skipped here, and refused by name when it is asked for.
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
    """Profiles with a layer `phern profiles fold` has not folded yet (item 0).

    A `<name>.d/` of drop-ins, or a file still in the list format before S2. Both
    are read until then, so nothing is dropped, and `phern doctor` names them so
    nothing is read silently either.
    """
    roots = resolve_roots()
    directory = roots.profiles_dir()
    if not directory.is_dir():
        return []
    names = {path.name.removesuffix(".d") for path in directory.glob("*.d") if path.is_dir()}
    names |= {one.name for one in person_profiles() if one.legacy}
    return sorted(names)


def profile_file(name: str) -> Path | None:
    """The `.yaml` this `--profile` value names, or `None` when it names a profile.

    **The one predicate**, because there were two and they disagreed: this module's
    own `available_profiles` states the rule — "two predicates for one question is
    how a `--help` line and a command line come to disagree" — and `profile_name`
    had re-derived it without the `.exists()` half, so a mistyped path resolved as a
    file here and as a named profile there.
    """
    candidate = Path(name)
    return candidate if candidate.suffix in (".yaml", ".yml") and candidate.exists() else None


def resolve_profile(name: str) -> list[Path]:
    """The documents for `name`, built-in layers first then the person's.

    A name that is a path is used directly, which is what makes a scenario test
    or a one-off deployment a single file rather than an install step.

    Paths only, so what each layer may set is not here: `profile_documents` is the
    door that reads them with it.
    """
    plan = _plan(name, resolve_roots())
    return [*plan.shipped, *([plan.named.path] if plan.named else []), *plan.dropins]


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """What composes a name: pH's layers, the person's file over them, and drop-ins."""

    shipped: list[Path]
    named: NamedProfile | None = None
    dropins: list[Path] = field(default_factory=list)
    whole: ProfileDocument | None = None
    """A `--profile` path in the list format, already read: it is the whole composition,
    and `shipped` names it only so `resolve_profile` can list it."""


def profile_plan(name: str) -> ProfilePlan:
    """The layers `name` composes from — the one answer to "file or named", and to
    which of them is the person's."""
    return _plan(name, resolve_roots())


def _plan(name: str, roots: PathRoots) -> ProfilePlan:
    """The layers for `name` — file or named, decided once.

    A path whose file is a list is a whole composition, deployment rows and all,
    which is the point of it: one file is the scenario. So it is one layer of pH's
    kind, with nothing over it — `presentation.yaml` included: a file that wants
    the trajectory screen inserts it. A path whose file is `extends:` and `rows:`
    is a named profile that lives somewhere else, and composes as one.

    A name is a shipped profile, the person's named profile over one, or both — a
    person's `tui.yaml` over the shipped `tui`.
    """
    candidate = profile_file(name)
    if candidate is not None:
        raw = decode_document(candidate)
        if not isinstance(raw, Mapping):
            return ProfilePlan(shipped=[candidate], whole=ProfileDocument(str(candidate), raw))
        elsewhere = parse_named_profile(raw, candidate, "", shipped=PROFILES)
        return ProfilePlan(shipped=_shipped_layers(elsewhere.extends), named=elsewhere)
    overlay = roots.profile_overlay(name)
    named = read_named_profile(overlay, name) if overlay.is_file() else None
    extends = named.extends if named is not None else name
    return ProfilePlan(shipped=_shipped_layers(extends), named=named, dropins=_dropins(name, roots))


def _shipped_layers(name: str) -> list[Path]:
    """The layers pH provides for the named profile `name`."""
    declared = PROFILES.get(name)
    if declared is None:
        raise ValueError(
            f'unknown profile "{name}"; available are '
            f"{', '.join(available_profiles())}, or pass a path to a .yaml"
        )
    missing = _missing_required(declared)
    if missing:
        # Naming the package is the person's next step.
        raise ValueError(
            f'profile "{name}" needs the "{missing}" bundle, which no installed '
            f"distribution provides; install ph-{missing} and try again"
        )
    layers: list[Path] = []
    for layer in _composed(declared):
        resolved = _resolve_layer(layer)
        if resolved is None:
            # An optional bundle this install does not have — `_missing_required`
            # already refused every required one. Silent here and visible in
            # `phern doctor`, which reports what activated: a warning on every
            # command would be a warning nobody reads, about a profile that
            # composed exactly as this table says it should.
            continue
        layers.append(resolved)
    layers.append(PRESENTATION)
    return layers


def _dropins(name: str, roots: PathRoots) -> list[Path]:
    """The drop-ins `/sandbox` wrote over `name` before S4 made its changes overrides.

    In name order, after the person's file, as they always composed; read until
    `phern profiles fold` folds them into that file. Nothing writes one now.
    """
    directory = roots.profile_dropins(name)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix in (".yaml", ".yml") and path.is_file()
    )


def profile_name(profile: str) -> str:
    """The name a `--profile` value names, or `""` for a path to a `.yaml`.

    What `Profile.name` records, so a caller writing under a profile's name knows where to —
    and knows *not to* for a file profile, whose one document is the person's own.
    Asked through `profile_file`, so it cannot disagree with what actually resolved.
    """
    return "" if profile_file(profile) is not None else profile


DEFAULT_PROFILE = "headless"

ProfileOption: TypeAlias = Annotated[
    str, typer.Option("--profile", help="Profile name or path to a .yaml.")
]
"""Declared once, so two commands cannot come to disagree about what `--profile`
means — or, as nearly happened here, about what an unknown one costs.

**Here rather than in `cli.py`**, which is where it started: `cli.py` imports the
sub-apps, so a sub-app that wanted the alias had to import back into it — and
`phern workspaces gc` did exactly that, reaching for a private `_documents` across
the cycle. This module already owns what a profile *is*; the flag that names one
belongs beside it. `console.py` was carved out for the same reason and states it.
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
"""dsh's third layer — bundle, profile, *patch from the command line* — which pH
had only as a file under `$PH_HOME/profiles/`. Same grammar as a profile
document, deliberately: a second spelling for "change this row" is how the two
come to accept different things. Parsed by `safe_yaml_load`, so the code-tag
refusal that guards a file guards the flag."""


ProviderOption: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help="With --model, run on this exact route instead of one the profile lists.",
    ),
]
"""`--provider`, declared once for `phern` and `phern daemon`, `ProfileOption`'s
reason: two copies of one flag are how two commands come to mean different
things by it. Read with `--model` by `ModelChoice.from_flags`."""

ModelOption: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="A model the profile lists, by key (`fast`); with --provider, a model name.",
    ),
]
"""`--model`, declared once beside `--provider` for the same reason."""


CLI_LAYER = "cli"
"""The layer a `--patch` composes under — what `--dump-config` and `phern doctor`'s
topology print as its provenance."""


def profile_documents(name: str) -> list[ProfileDocument]:
    """The profile's layers as documents, each saying what it may set — the door
    every command composes through, raising as its parts raise.

    The stage a caller wants when it has a layer of its own to add before
    composing — the benchmark's `bench` document, a fixture's overlay. The kinds
    are the module docstring's table, and `daemon.yaml`'s rows sit between pH's
    layers and the person's: after what they patch, and apart from the environment
    a session profile names.
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
    """`daemon.yaml`'s rows, as the deployment layer every composition carries."""
    host = load_host_config(roots.home)
    if host.rows is None:
        return []
    return [ProfileDocument(str(host_config_path(roots.home)), host.rows, sets="deployment")]


def base_documents(extends: str) -> list[ProfileDocument]:
    """What a named profile over the shipped `extends` composes over.

    pH's layers and the host's rows, and none of the person's: a saved profile is
    the difference from exactly this, so the host's rows — in both — cancel, and
    what is left is the environment the person chose.
    """
    return [*load_profile_documents(_shipped_layers(extends)), *_host_documents(resolve_roots())]


def save_named_profile(name: str, profile: Profile, *, extends: str, comment: str) -> Path:
    """Write `profile` as the sparse named profile `name` over `extends` (S2).

    Only the rows that differ (`sparse_entries`), so composing `name` gives back
    `profile`'s rows — the round trip decision 8 rests on. Written whole and
    atomically; the comment says who wrote it, since a person will read it.
    """
    path = resolve_roots().profile_overlay(name)
    write_atomic(path, sparse_text(profile, extends=extends, comment=comment))
    return path


def sparse_text(profile: Profile, *, extends: str, comment: str) -> str:
    """`profile` as the text of a sparse named profile over `extends`: only the rows
    that differ from it (`sparse_entries`)."""
    entries = sparse_entries(compose_rows(base_documents(extends)), profile.rows)
    return render_named_profile(extends, entries, comment=comment)


def compose_profile(name: str) -> Profile:
    """A shipped profile, composed and ready to mount — for a test or a bench.

    A command goes through `profile_or_exit`, which is this plus the exit code.
    """
    return Profile.from_documents(profile_documents(name), name=profile_name(name))


def profile_or_exit(
    profile: str, patches: Sequence[str] = (), *, deployment: Sequence[str] = ()
) -> Profile:
    """The profile composed, `--patch` entries included — or exit 2 saying why not.

    `patches` is a person's `--patch`, which sets environment rows as a session
    profile does. `deployment` is a host flag written as the patch it is —
    `phern daemon --max-concurrent-children` — and sets deployment rows as
    `daemon.yaml` does, after it. Both carry `cli` as their provenance.

    The refusal is the command's, not the resolver's: `resolve_profile` raises a
    `ValueError` that already names the available profiles, and every caller
    wants that sentence on stderr under the same exit code.

    **Everything about the profile is refused here, once, and nothing composes
    twice.** A command past this line holds a `Profile`; every mode mounts that
    rather than re-reading or re-composing, so "no such profile", "not YAML" and
    a row id that does not exist are one refusal under one exit code wherever
    they are met — a bad row used to be exit 2 from `phern -p` and "profile does not
    mount" under exit 1 from `phern doctor`. What stays with the mount is the
    mount's: a plugin that cannot be imported, an `isolate:` copy that never
    activates, a row refusing the deployment.
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
        return Profile.from_documents(documents, name=profile_name(profile))
    except LoaderError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)


def _patch_entries(text: str) -> list[Any]:
    """One `--patch` value as the entries of a profile document.

    A mapping is one entry; a list is spliced in as several. No shape check
    beyond that, deliberately: `compose_rows` already decides row-versus-patch
    per entry and refuses every malformed one, and a second checker here is the
    grammar written twice. A first draft inferred `insert:` around a list whose
    entries all carried `name` — a rule the loader already applies to a bare
    row in any document — and returned the un-inferred case as a nested list,
    which the loader could only refuse.
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
