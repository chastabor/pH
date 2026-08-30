"""Who owns a seam registration, and the one "release only if still mine" rule.

Every seam lets a plugin claim a slot (the sandbox provider, the code runtime) or
a key in a table (a command, a skill, a renderer) and hand back a disposer.
Written by hand six times, the release step drifted: some copies checked
identity before removing, some did not. A disposer that removes whatever
*currently* occupies the slot would tear down a successor registered after its
own owner was replaced — so identity is checked here, once.

**Who owns a registration is `Context.owner_for`, not here** (P6-12). It was
briefly this module's, and it does not belong: this module is about *tables* —
release only what is still mine — while ownership is about `Context` lifetimes
and touches nothing else. Two of its three consumers (`ph.tools.registry`,
`ph.system_prompt.assembly`) are not seams at all, so keeping it here made them
reach into a sibling package's underscore module and left a latent import cycle
one re-export away.

@module ph.seams._registry
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, Disposer

__all__ = ["claim_entry", "claim_key", "claim_slot"]


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


def claim_entry(owner: Context, entries: list[Any], value: Any, *, label: str) -> Disposer:
    """Append `value`; the disposer removes **that object**, not one equal to it.

    `list.remove` compares with `==`, which is identity for the closures and
    objects the other contribution lists in this package hold — so those got the
    right answer by accident. A registry of *values* does not: two rows
    contributing an equal entry would have one disposer take the other's, which
    is this module's whole complaint one container over.
    """
    entries.append(value)

    def release() -> None:
        for index, held in enumerate(entries):
            if held is value:
                del entries[index]
                return

    return owner.add_disposer(release, label=label)


def claim_slot(owner: Context, holder: Any, attribute: str, value: Any, *, label: str) -> Disposer:
    """Set `holder.<attribute>`; the disposer clears it only while it still holds `value`."""
    if getattr(holder, attribute) is not None:
        raise RuntimeError(f"{label}: a provider is already registered")
    setattr(holder, attribute, value)

    def release() -> None:
        if getattr(holder, attribute) is value:
            setattr(holder, attribute, None)

    return owner.add_disposer(release, label=label)
