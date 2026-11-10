"""`ph-code-graph` — `code_index` and `code_graph` from a command line.

For trying a query by hand, and what this package's UTCP manual calls. The tools
are the ones the `code-graph-engine` row registers (`ph_code_graph.engine`);
`ph.toolcli` mounts that row in this process and runs them through the registry.
The `code-graph` row a session mounts reaches them by running this command.

```bash
ph-code-graph call code_index --paths packages/ph-core --format text
ph-code-graph call code_graph --mode callers --query claim_slot --format text
ph-code-graph status
ph-code-graph manual
```

@module ph_code_graph.cli
"""

from __future__ import annotations

from importlib.metadata import version
from typing import NoReturn

from ph.toolcli import ToolCli
from ph.toolcli import main as run_cli

from ._config import GRAMMAR_CACHE_ENV, ROW

__all__ = ["CLI", "main"]

CLI = ToolCli(
    program="ph-code-graph",
    manual="ph_code_graph",
    summary="Index a code tree and ask about its structure: definitions, callers, impact.",
    version=version("ph-code-graph"),
    # The engine row, under the session row's id: the config a profile writes for
    # `code-graph` is then the config this command reads for it.
    rows=({"id": ROW, "name": "code-graph-engine"},),
    command=ROW,
    inherit_env=(GRAMMAR_CACHE_ENV,),
    module="ph_code_graph",
)


def main() -> NoReturn:
    run_cli(CLI)
