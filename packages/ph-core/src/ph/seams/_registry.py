"""The one "release only if still mine" rule for seam registrations.

Every seam lets a plugin claim a slot (the sandbox provider, the code runtime) or
a key in a table (a command, a skill, a renderer) and hand back a disposer.
Written by hand six times, the release step drifted: some copies checked
identity before removing, some did not. A disposer that removes whatever
*currently* occupies the slot would tear down a successor registered after its
own owner was replaced — so identity is checked here, once.

@module ph.seams._registry
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, Disposer

__all__ = ["claim_key", "claim_slot"]


def claim_key(
    owner: Context, table: dict[str, Any], key: str, value: Any, *, label: str
) -> Disposer:
    """Put `value` under `key`; the disposer removes it only while it is still there."""
    if key in table:
        raise ValueError(f"{label}: {key!r} is already registered")
    table[key] = value

    def release() -> None:
        if table.get(key) is value:
            del table[key]

    return owner.add_disposer(release, label=f"{label}({key})")


def claim_slot(owner: Context, holder: Any, attribute: str, value: Any, *, label: str) -> Disposer:
    """Set `holder.<attribute>`; the disposer clears it only while it still holds `value`."""
    if getattr(holder, attribute) is not None:
        raise RuntimeError(f"{label}: a provider is already registered")
    setattr(holder, attribute, value)

    def release() -> None:
        if getattr(holder, attribute) is value:
            setattr(holder, attribute, None)

    return owner.add_disposer(release, label=label)
