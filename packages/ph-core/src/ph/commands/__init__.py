"""Human slash commands: things a person asks the harness to do directly.

`ph/seams/` is for a seam or a provider of one. A command is neither — it
publishes no service and registers no provider, it *consumes* several — and the
shape's existing siblings (`ph_stabilize.compact_command`, `ph_rlm.harness`'s
`/refine`) already live outside `seams/` in their own packages. This is where
ph-core's own commands go, so `/workspaces` and P4-09's `/revert` share a home
rather than each making the same choice differently.

@module ph.commands
"""

from __future__ import annotations

__all__: list[str] = []
