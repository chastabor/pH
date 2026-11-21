"""`sandbox-allow`: what a confined command may reach beyond its workspace.

This row states the allowances and does nothing else. `Allowances` is the value
of the seam: the directories that are writable beyond the workspace and its
scratch directory, and the network posture with its host list.
`SandboxSeam.effective` merges the value into every policy. So this row
registers the value and describes it, and it enforces nothing itself. Because
of that split, the row can be **applied again while an agent runs**. To unmount
the row releases one slot, and to mount it again fills the slot. No provider is
swapped, no probe runs again and no proxy restarts, and the new statement
bounds the next confined command. `/sandbox` applies the row again, as an
override of the session (`ph.session_profile.override`), and the record that it
writes makes the change survive a restart.

**The row ships enabled, with the defaults of the seam.** No network and no
extra directory is the closed answer. It is the wrong default for a harness
whose agents install packages and read documentation, because somebody turns
off an allowlist that nobody can work inside. `DEFAULT_HOSTS` is the default
list, and the user can trim it.

@module ph.seams.sandbox_allow
"""

from __future__ import annotations

from pathlib import Path

from ..cordis import ChildLimit, ChildReach, Context, NarrowingRefused, plugin
from ..keys import SANDBOX
from .diagnostics import Diagnostic, contribute
from .sandbox import Allowances, SandboxSeam, allowance_path, held_under

__all__ = ["apply", "describe", "describe_paths", "narrows"]


def narrows(_mounted: Allowances, asked: Allowances, reach: ChildReach) -> ChildLimit:
    """The `sandbox-allow` row of a child may bind fewer of the writable
    directories of its parent, or directories inside them, and never others
    (S7b, item 3).

    The check is against the reach of the parent as the seam holds it now
    (`writable_for`). That is the reach of the deployment after any
    `/sandbox allow path`, less the narrowing of the parent itself when the
    parent is also a child.

    **The network of a child is exactly the network of its parent.** Confined
    commands reach the network through one egress proxy. The proxy knows the
    agent but has no list per agent, so a child cannot yet hold fewer hosts than
    its parent. A row that asks for a different network, wider or narrower, is
    refused. It is not silently given the network of its parent, because nobody
    could read from the admission of a child a ceiling that the child did not
    get.
    """
    seam = reach.ctx.require(SANDBOX)
    held = seam.writable_for(reach.agent)
    paths = tuple(dict.fromkeys(allowance_path(one) for one in asked.paths))
    beyond = [one for one in paths if not held_under(one, held)]
    if beyond:
        raise NarrowingRefused(
            f"its sandbox-allow lets a child write {', '.join(beyond)}, which its parent's does not"
        )
    parent, wanted = seam.allowed.network, asked.network
    if wanted.mode != parent.mode:
        differs = f"{wanted.mode} network where its parent has {parent.mode}"
    elif wanted.mode == "allowlist" and set(wanted.hosts) != set(parent.hosts):
        hosts = sorted(set(wanted.hosts) ^ set(parent.hosts))
        differs = f"other hosts than its parent's ({', '.join(hosts)})"
    else:
        return ChildLimit(writable_paths=None if set(paths) == set(held) else paths)
    raise NarrowingRefused(
        f"its sandbox-allow gives a child {differs}; a child's network is its parent's, "
        "since one egress proxy serves both"
    )


@plugin(
    "sandbox-allow",
    affects="environment",
    inject=[SANDBOX],
    config=Allowances,
    narrows=narrows,
)
async def apply(ctx: Context, config: Allowances) -> None:
    """Register the allowances of the deployment and describe them.

    `Allowances` **is** the row config, and there is no `Config` subclass. An
    empty subclass meant that `apply` had to rebuild a plain value field by
    field, so that a later `==` in `/sandbox` would not compare types as well as
    fields. The equality check of one command had become a rule that the
    registration of this row had to remember.
    """
    ctx.require(SANDBOX).register_allowances(config)
    contribute(
        ctx,
        Diagnostic(
            id="sandbox-allow",
            title="Sandbox allowances",
            read=lambda: describe(ctx.require(SANDBOX)),
            order=16,
        ),
    )


def describe(seam: SandboxSeam) -> list[tuple[str, str]]:
    """What `phern doctor` prints, and what `/sandbox` prints, from the same
    function.

    The value is read live and not at mount. The row can be applied again, and
    the egress bridge that decides whether `allowlist` has any effect belongs to
    another row.
    """
    allowances = seam.allowances
    if allowances is None:
        return [("allowances", "none registered — confined commands reach nothing extra")]
    rows = [("network", seam.network_posture())]
    if allowances.network.mode == "allowlist":
        rows.append(("hosts", ", ".join(allowances.network.hosts) or "none — nothing is reachable"))
    rows.append(("writable beyond the workspace", describe_paths(allowances.paths) or "none"))
    return rows


def describe_paths(paths: list[str]) -> str:
    """The allowed directories, each marked when it does not exist and so cannot
    be bound.

    `SandboxSeam.allowed_paths` skips a missing directory, so that `bwrap` does
    not refuse every command over it. This function reports that omission, so
    that nobody has to discover it from a refused write.
    """
    described = []
    for entry in paths:
        path = Path(entry).expanduser()
        described.append(entry if path.is_dir() else f"{entry} (missing — not bound)")
    return ", ".join(described)
