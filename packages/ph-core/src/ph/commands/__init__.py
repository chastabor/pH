"""Human slash commands: actions that a person asks the harness for directly.

`ph/seams/` is for a seam or a provider of a seam. A command is neither. It
publishes no service and registers no provider. It *consumes* several services.
The existing commands of the same shape (`ph_stabilize.compact_command`, and
`/refine` in `ph_rlm.harness`) already live outside `seams/` in their own
packages. The commands of `ph-core` go here: `/autonomous`, `/revert` (P4-09),
`/sandbox` and `/workspaces`. They share one package, so that each does not make
the same choice differently.

@module ph.commands
"""

from __future__ import annotations

__all__: list[str] = []
