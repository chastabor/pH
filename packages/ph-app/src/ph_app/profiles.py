"""Profile resolution: which bundle documents compose a run.

A profile is an ordered list of YAML documents. The shipped ones live in
`ph.bundles`; a user profile is a file under `$PH_HOME/profiles/<name>.yaml`
layered on top, so a deployment overrides a row by id without forking a bundle.

@module ph_app.profiles
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from ph.bundles import BASE, HEADLESS, resolve_bundle
from ph.paths import resolve_roots

__all__ = ["PROFILES", "PROFILE_DIR", "Bundle", "available_profiles", "resolve_profile"]

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

PROFILES: dict[str, tuple[Layer, ...]] = {
    "base": (BASE,),
    "headless": (BASE, HEADLESS),
    "tui": TUI_LAYERS,
    # Real providers layer onto base; the fake adapter is deliberately absent so
    # a misconfigured key fails loudly instead of silently answering "ok".
    "deepseek": (BASE, PROFILE_DIR / "deepseek.yaml"),
    "anthropic": (BASE, PROFILE_DIR / "anthropic.yaml"),
    # `rlm` is the interactive posture plus the RLM bundle, because a person is
    # present for the approvals Code Mode's dispatches raise (P3-20).
    "rlm": (*TUI_LAYERS, Bundle("rlm")),
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


def resolve_profile(name: str) -> list[Path]:
    """The documents for `name`, built-in layers first then the user's overlay.

    A name that is a path is used directly, which is what makes a scenario test
    or a one-off deployment a single file rather than an install step.
    """
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
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
    overlay = resolve_roots().profiles_dir() / f"{name}.yaml"
    if overlay.exists():
        layers.append(overlay)
    return layers
