"""`ph doctor`'s Topology section: what the mount *became* (dsh's dump rule).

`--dump-config` prints the composition — the rows as written, before anything
runs — and is honest about that. This is the other half: which rows activated,
what each injects, which key one is still waiting on, which layer disabled one,
and which isolated realms exist.

Contributed through `ctx.diagnostics` like every other section, which is what
gives it a declared order, makes a raise in the reader drop this section alone,
and lets `ph agents doctor` carry it unchanged. A row rather than something
`Profile.mount` files itself, because `ph.seams` may import `ph.cordis` and never
the reverse — which is also why the mount is a service (`ctx.mount`).

Not enforced (§5 rule 6): a profile that mounts no `diagnostics` row has no
Topology section either. That is `contribute`'s trade — a hard `inject` on the
seam would make the report a precondition for the thing being reported on — and
`ph doctor` says so in the seam's place.

@module ph.seams.topology
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from .diagnostics import ORDER_SELF_ASSESSMENT, Diagnostic, contribute

__all__ = ["apply"]


@plugin("topology", inject=["mount"])
async def apply(ctx: Context, _config: Any) -> None:
    """Offer the mount's account of itself as a section."""
    contribute(
        ctx,
        Diagnostic(
            id="topology",
            title="Topology",
            read=ctx.mount.topology,
            # After every reading and before the self-assessment, which asks
            # for exactly this: meet the deployment's account of itself first.
            order=ORDER_SELF_ASSESSMENT - 10,
        ),
    )
