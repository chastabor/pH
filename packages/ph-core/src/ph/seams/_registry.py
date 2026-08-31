"""Who owns a seam registration, and the one "release only if still mine" rule.

Every seam lets a plugin claim a slot (the sandbox provider, the code runtime) or
a key in a table (a command, a skill, a renderer) and hand back a disposer.
Written by hand six times, the release step drifted: some copies checked
identity before removing, some did not. A disposer that removes whatever
*currently* occupies the slot would tear down a successor registered after its
own owner was replaced — so identity is checked here, once.

**A `Running` may be passed wherever a `Context` may** (P6-29). Both spell
`add_disposer(release, label=)`, and the pair's releases on *either* of its two
scopes — which is what a registry whose container is keyed by the visibility
scope needs, and what one keyed only by the owner does not. Opting in is passing
`by` rather than `by.owner`; nothing else changes.

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

from ..cordis import Context, Disposer, Running

__all__ = ["claim_entry", "claim_key", "claim_slot"]


def claim_key[T](
    owner: Context | Running, table: dict[str, T], key: str, value: T, *, label: str
) -> Disposer:
    """Put `value` under `key`; the disposer removes it only while it is still there.

    Parameterised so the value and the table it lands in have to agree. `Any`
    here was the last hop of a chain that is otherwise checked end to end: a
    registry threads its element type through its own bucket and its `_register`,
    and then handed both to a helper that would take either from anywhere. Free
    at every call site, because `T` is inferred from the table.
    """
    if key in table:
        raise ValueError(f"{label}: {key!r} is already registered")
    table[key] = value

    def release() -> None:
        if table.get(key) is value:
            del table[key]

    return owner.add_disposer(release, label=f"{label}({key})")


def claim_entry[T](owner: Context | Running, entries: list[T], value: T, *, label: str) -> Disposer:
    """Append `value`; the disposer removes **that object**, not one equal to it.

    Parameterised for the reason `claim_key` above is: the type a registry took
    care to thread through its bucket stopped being checked at this boundary.

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


def claim_slot(by: Running, holder: Any, attribute: str, value: Any, *, label: str) -> Disposer:
    """Set `holder.<attribute>`; the disposer clears it only while it still holds `value`.

    **Takes the pair rather than the owner** (P6-29), and holds it in
    `<attribute>_by` for exactly as long as the slot itself. A provider is a body
    the seam invokes *later* — every one of the five in this tree is — so to
    enter the right binding then it has to have kept who registered it and where
    that landed. Set and cleared with the slot in one claim, because the
    alternative was five seams writing the same four lines around this call, and
    five chances for the record to outlive the value it describes.

    **The name is derived, not a parameter**, and that is the difference between
    a rule and a reminder. All five sites spelled `f"{attribute}_by"`, so the
    argument carried no information — and being optional, it made "forgot to
    record" a silent state: `running(None)` binds nothing, and the P6-30 gate
    classifies the *registration method*, which is present either way. Derived,
    a sixth slot that omits the field fails at registration instead, because
    every one of these holders is `slots=True` and `setattr` for an undeclared
    name raises there and then.

    That does not move the ownership decision here, which the module docstring
    above is right to refuse: the caller still decides, through
    `Context.running_for`, exactly as it decided through `owner_for` before.
    This stores what it was told.
    """
    if getattr(holder, attribute) is not None:
        raise RuntimeError(f"{label}: a provider is already registered")
    record = f"{attribute}_by"
    setattr(holder, attribute, value)
    setattr(holder, record, by)

    def release() -> None:
        if getattr(holder, attribute) is value:
            setattr(holder, attribute, None)
            setattr(holder, record, None)

    return by.owner.add_disposer(release, label=label)
