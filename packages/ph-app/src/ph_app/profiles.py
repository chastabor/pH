"""Profile resolution: which bundle documents compose a run.

A profile is an ordered list of YAML documents. The shipped ones live in
`ph.bundles`; a user profile is a file under `$PH_HOME/profiles/<name>.yaml`
layered on top, so a deployment overrides a row by id without forking a bundle.

@module ph_app.profiles
"""

from __future__ import annotations

from pathlib import Path

from ph.bundles import BASE, HEADLESS
from ph.paths import resolve_roots

__all__ = ["BUILTIN_PROFILES", "PROFILE_DIR", "resolve_profile"]

PROFILE_DIR = Path(__file__).parent / "profiles"

BUILTIN_PROFILES: dict[str, tuple[Path, ...]] = {
    "base": (BASE,),
    "headless": (BASE, HEADLESS),
    # Real providers layer onto base; the fake adapter is deliberately absent so
    # a misconfigured key fails loudly instead of silently answering "ok".
    "deepseek": (BASE, PROFILE_DIR / "deepseek.yaml"),
    "anthropic": (BASE, PROFILE_DIR / "anthropic.yaml"),
}


def resolve_profile(name: str) -> list[Path]:
    """The documents for `name`, built-in layers first then the user's overlay.

    A name that is a path is used directly, which is what makes a scenario test
    or a one-off deployment a single file rather than an install step.
    """
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        return [candidate]
    layers = list(BUILTIN_PROFILES.get(name, ()))
    if not layers:
        raise ValueError(
            f'unknown profile "{name}"; built-ins are '
            f"{', '.join(sorted(BUILTIN_PROFILES))}, or pass a path to a .yaml"
        )
    overlay = resolve_roots().profiles_dir() / f"{name}.yaml"
    if overlay.exists():
        layers.append(overlay)
    return layers
