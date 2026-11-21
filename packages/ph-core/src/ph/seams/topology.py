"""The Topology section of `phern doctor`: what the mount *became* (the dump rule of dsh).

`--dump-config` prints the composition: the rows as written, before anything
runs. It claims nothing more. This section is the other half: which rows
activated, what each row injects, which key a row is still waiting on, which
layer disabled a row, and which isolated realms exist.

The section is contributed through `ctx.diagnostics` like every other section.
That gives it a declared order, keeps a raise in the reader inside this section
alone, and lets `phern agents doctor` carry it unchanged. It is a row and not
something that `Profile.mount` files itself, because `ph.seams` may import
`ph.cordis` and never the reverse. That is also why the mount is a service
(`ctx.mount`).

Not enforced (§5 rule 6): a profile that mounts no `diagnostics` row has no
Topology section either. That is the trade that `contribute` makes: a hard
`inject` on the seam would make the report a precondition for the thing that
it reports on. `phern doctor` states the absence in place of the seam.

@module ph.seams.topology
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..keys import MOUNT
from .diagnostics import ORDER_SELF_ASSESSMENT, Diagnostic, contribute

__all__ = ["apply"]


@plugin("topology", affects="deployment", inject=[MOUNT])
async def apply(ctx: Context, config: None) -> None:
    """Offer the account that the mount gives of itself as a section."""
    contribute(
        ctx,
        Diagnostic(
            id="topology",
            title="Topology",
            read=ctx.require(MOUNT).topology,
            # After every reading and before the self-assessment. The
            # self-assessment asks for exactly this order: a reader meets the
            # account that the deployment gives of itself first.
            order=ORDER_SELF_ASSESSMENT - 10,
        ),
    )
