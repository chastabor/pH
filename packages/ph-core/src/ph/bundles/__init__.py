"""Shipped profile bundles. YAML rows, addressed by id.

`BASE` and `HEADLESS` ship with ph-core. Everything else is **discovered**, not
imported: a bundle is a distribution's contribution to the profile surface, and
`ph-app` must be able to compose the `rlm` profile without depending on
`ph-rlm` — the same rule that keeps the app reading `subagent/*` events without
importing the bundle that emits them.

So a distribution registers its bundle in the `ph.bundles` entry-point group,
pointing at a module attribute holding the path:

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

    `resolve_bundle` is the question a caller offering a profile should ask —
    this one is for a diagnostic that wants to say "registered but broken".
    """
    return sorted(entry_point_targets(BUNDLE_ENTRY_POINT_GROUP))


def resolve_bundle(name: str) -> Path | None:
    """The bundle path `name` registers, or `None` if it does not resolve.

    `None` rather than an exception — for a missing registration *and* for a
    registration whose module will not import: a profile that names a bundle
    this install cannot deliver is a configuration answer, and the caller has
    the context to word it. The distinction between the two is available from
    `installed_bundles()` when a diagnostic wants it.
    """
    try:
        path = resolve_entry_point(BUNDLE_ENTRY_POINT_GROUP, name, default_attribute="BUNDLE")
    except (ImportError, AttributeError):
        return None
    return path if isinstance(path, Path) and path.exists() else None
