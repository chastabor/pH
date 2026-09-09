"""Profile resolution: which bundle documents compose a run.

A profile is an ordered list of YAML documents. The shipped ones live in
`ph.bundles`; a user profile is a file under `$PH_HOME/profiles/<name>.yaml`
layered on top, so a deployment overrides a row by id without forking a bundle.

@module ph_app.profiles
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeAlias

import typer

from ph.bundles import BASE, HEADLESS, resolve_bundle
from ph.cordis import LoaderError, Profile, ProfileDocument, load_profile_documents
from ph.cordis.loader import safe_yaml_load
from ph.paths import resolve_roots

from .console import fail

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "PROFILE_DIR",
    "Bundle",
    "ProfileOption",
    "available_profiles",
    "compose_profile",
    "profile_documents",
    "profile_file",
    "profile_name",
    "profile_or_exit",
    "resolve_profile",
]

PROFILE_DIR = Path(__file__).parent / "profiles"


@dataclass(frozen=True, slots=True)
class Bundle:
    """A layer another distribution provides, resolved late.

    A profile is an ordered list of layers; the only thing P3-20 added is that
    one *kind* of layer is not a path this package can name. It is discovered
    through the `ph.bundles` entry-point group — ph-app must not depend on
    ph-rlm, the same rule that lets the app read `subagent/*` events without
    importing the row that emits them.
    """

    name: str


Layer: TypeAlias = "Path | Bundle"

TUI_LAYERS: tuple[Layer, ...] = (BASE, HEADLESS, PROFILE_DIR / "tui.yaml")
"""The interactive posture: `headless` plus one row. A person is present to
answer the seams, so the workspace is writable (see tui.yaml)."""

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
    "base": (BASE,),
    "headless": (BASE, HEADLESS),
    "tui": TUI_LAYERS,
    # Real providers layer onto base; the fake adapter is deliberately absent so
    # a misconfigured key fails loudly instead of silently answering "ok".
    "deepseek": (BASE, PROFILE_DIR / "deepseek.yaml"),
    # A server on localhost rather than a service, and the differences are in the
    # document: one slot's window rather than the whole server's, and none of the
    # media the hosted default claims.
    "llama": (BASE, PROFILE_DIR / "llama.yaml"),
    "anthropic": (BASE, PROFILE_DIR / "anthropic.yaml"),
    "google": (BASE, PROFILE_DIR / "google.yaml"),
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


def available_profiles() -> list[str]:
    """Every profile this install can actually compose.

    Answered by the same resolution `resolve_profile` performs, so a profile is
    never offered and then refused: two predicates for one question is how a
    `--help` line and a command line come to disagree.
    """
    return sorted(
        name
        for name, layers in PROFILES.items()
        if all(_resolve_layer(layer) is not None for layer in layers)
    )


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
    """The documents for `name`, built-in layers first then the user's overlay.

    A name that is a path is used directly, which is what makes a scenario test
    or a one-off deployment a single file rather than an install step.
    """
    candidate = profile_file(name)
    if candidate is not None:
        return [candidate]
    declared = PROFILES.get(name)
    if declared is None:
        raise ValueError(
            f'unknown profile "{name}"; available are '
            f"{', '.join(available_profiles())}, or pass a path to a .yaml"
        )
    layers: list[Path] = []
    for layer in declared:
        resolved = _resolve_layer(layer)
        if resolved is None:
            # Only a `Bundle` can fail to resolve, and naming the package is the
            # person's next step.
            assert isinstance(layer, Bundle)
            raise ValueError(
                f'profile "{name}" needs the "{layer.name}" bundle, which no installed '
                f"distribution provides; install ph-{layer.name} and try again"
            )
        layers.append(resolved)
    roots = resolve_roots()
    overlay = roots.profile_overlay(name)
    if overlay.exists():
        layers.append(overlay)
    # Then whatever pH wrote on the person's behalf — `/sandbox`'s allowlists — in
    # name order, after the overlay, so a change made from the TUI outlives the
    # process without touching the file the person edits. See
    # `PathRoots.profile_dropins`.
    dropins = roots.profile_dropins(name)
    if dropins.is_dir():
        layers.extend(
            sorted(
                path
                for path in dropins.iterdir()
                if path.suffix in (".yaml", ".yml") and path.is_file()
            )
        )
    return layers


def profile_name(profile: str) -> str:
    """The name a `--profile` value names, or `""` for a path to a `.yaml`.

    What `Profile.name` records, so a command that writes a drop-in knows where to —
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
`ph workspaces gc` did exactly that, reaching for a private `_documents` across
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


CLI_LAYER = "cli"
"""The layer a `--patch` composes under — what `--dump-config` and `ph doctor`'s
topology print as its provenance."""


def profile_documents(name: str) -> list[ProfileDocument]:
    """`resolve_profile`, read: the profile's layers as documents, raising as its parts raise.

    The stage a caller wants when it has a layer of its own to add before
    composing — the benchmark's `bench` document, a fixture's overlay.
    """
    return load_profile_documents(resolve_profile(name))


def compose_profile(name: str) -> Profile:
    """A shipped profile, composed and ready to mount — for a test or a bench.

    A command goes through `profile_or_exit`, which is this plus the exit code.
    """
    return Profile.from_documents(profile_documents(name), name=profile_name(name))


def profile_or_exit(profile: str, patches: Sequence[str] = ()) -> Profile:
    """The profile composed, `--patch` entries included — or exit 2 saying why not.

    The refusal is the command's, not the resolver's: `resolve_profile` raises a
    `ValueError` that already names the available profiles, and every caller
    wants that sentence on stderr under the same exit code.

    **Everything about the profile is refused here, once, and nothing composes
    twice.** A command past this line holds a `Profile`; every mode mounts that
    rather than re-reading or re-composing, so "no such profile", "not YAML" and
    a row id that does not exist are one refusal under one exit code wherever
    they are met — a bad row used to be exit 2 from `ph -p` and "profile does not
    mount" under exit 1 from `ph doctor`. What stays with the mount is the
    mount's: a plugin that cannot be imported, an `isolate:` copy that never
    activates, a row refusing the deployment.
    """
    try:
        documents = profile_documents(profile)
    except (ValueError, LoaderError, OSError) as error:
        fail(f"[red]{error}[/red]", code=2, cause=error)
    if patches:
        entries = [entry for text in patches for entry in _patch_entries(text)]
        documents.append((CLI_LAYER, entries))
    try:
        return Profile.from_documents(documents, name=profile_name(profile))
    except LoaderError as error:
        fail(f"[red]{error}[/red]", code=2, cause=error)


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
        fail(f"[red]--patch {text!r}: {error}[/red]", code=2, cause=error)
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return [entry]
    fail(
        f"[red]--patch {text!r}: expected a mapping like `{{id: <row>, config: {{...}}}}`[/red]",
        code=2,
    )
