"""`sandbox-allow` — what a confined command may reach beyond its workspace.

The row that says it, and only says it. `Allowances` is the seam's value —
directories writable beyond the workspace and its scratch, and the network posture
with its host list — and `SandboxSeam.effective` merges it into every policy, so
this row registers the value and describes it and enforces nothing itself. That
split is what makes the row **re-applicable while an agent runs**: unmounting it
releases one slot and mounting it again fills the slot, with no provider swapped,
no probe rerun and no proxy restarted, and the next confined command is bounded by
the new statement. `/sandbox` is what does that re-apply (`Mount.reconfigure`), and
the profile drop-in it writes is what makes the change survive a restart.

**Ships enabled, with the seam's defaults.** No network and no extra directory is
the closed answer and the wrong default for a harness whose agents install
packages and read documentation: an allowlist nobody can live inside is one
somebody turns off. `DEFAULT_HOSTS` is that list, and it is the user's to trim.

@module ph.seams.sandbox_allow
"""

from __future__ import annotations

from pathlib import Path

from ..cordis import Context, plugin
from ..keys import SANDBOX
from .diagnostics import Diagnostic, contribute
from .sandbox import Allowances, SandboxSeam

__all__ = ["apply", "describe", "describe_paths"]


@plugin("sandbox-allow", inject=[SANDBOX], config=Allowances)
async def apply(ctx: Context, config: Allowances) -> None:
    """Register the deployment's allowances and say what they are.

    `Allowances` **is** the row config — there is no `Config` subclass, because an
    empty one meant `apply` had to rebuild a plain value field by field so that a
    later `==` in `/sandbox` would not compare types as well as fields. One
    command's equality check had become a rule this row's registration had to
    remember.
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
    """What `ph doctor` prints — and what `/sandbox` prints, from the same function.

    Read live rather than at mount: the row can be re-applied, and the egress
    bridge that decides whether `allowlist` means anything is another row's.
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
    """The allowed directories, each marked when it is not there to bind.

    `SandboxSeam.allowed_paths` skips a missing directory rather than letting
    `bwrap` refuse every command over it; this is where that omission is said out
    loud rather than left for someone to discover from a write that was refused.
    """
    described = []
    for entry in paths:
        path = Path(entry).expanduser()
        described.append(entry if path.is_dir() else f"{entry} (missing — not bound)")
    return ", ".join(described)
