"""Shipped profile bundles. A bundle is YAML rows, addressed by id.

`BASE` and `HEADLESS` ship with ph-core. Every other bundle is **discovered**,
not imported. A bundle is the contribution of a distribution to the profile
surface, and `ph_app` must be able to compose the `rlm` profile without
importing `ph_rlm`. The same rule lets the app read `subagent/*` events without
importing the bundle that emits them.

A distribution therefore registers its bundle in the `ph.bundles` entry-point
group. The entry point names a module attribute that holds the path:

    [project.entry-points."ph.bundles"]
    rlm = "ph_rlm:BUNDLE"

@module ph.bundles
"""

from __future__ import annotations

from pathlib import Path

from ..cordis.loader import entry_point_targets, resolve_entry_point

BUNDLE_DIR = Path(__file__).parent

BASE = BUNDLE_DIR / "base.yaml"
HEADLESS = BUNDLE_DIR / "headless.yaml"

BUNDLE_ENTRY_POINT_GROUP = "ph.bundles"

__all__ = [
    "BASE",
    "BUNDLE_DIR",
    "BUNDLE_ENTRY_POINT_GROUP",
    "HEADLESS",
    "installed_bundles",
    "resolve_bundle",
]


def installed_bundles() -> list[str]:
    """Every bundle name that *registers*, whether or not it resolves.

    A caller that offers a profile should ask `resolve_bundle`. This function
    is for a diagnostic that wants to say "registered but broken".
    """
    return sorted(entry_point_targets(BUNDLE_ENTRY_POINT_GROUP))


def resolve_bundle(name: str) -> Path | None:
    """The bundle path that `name` registers, or `None` if it does not resolve.

    The result is `None` and not an exception, both for a missing registration
    *and* for a registration whose module does not import. A profile that names
    a bundle that this install cannot deliver is a configuration answer, and
    the caller has the context to word it. A diagnostic that wants to tell the
    two cases apart can use `installed_bundles()`.
    """
    try:
        path = resolve_entry_point(BUNDLE_ENTRY_POINT_GROUP, name, default_attribute="BUNDLE")
    except (ImportError, AttributeError):
        return None
    return path if isinstance(path, Path) and path.exists() else None
