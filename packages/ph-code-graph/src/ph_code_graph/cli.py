"""`ph-code-graph`: `code_index` and `code_graph` from a command line.

The command is for trying a query by hand, and it is what the UTCP manual of this
package calls. The tools are the ones that the `code-graph-engine` row registers
(`ph_code_graph.engine`). `ph.toolcli` mounts that row in this process and runs
the tools through the registry. The `code-graph` row that a session mounts
reaches the tools by running this command.

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
    # The engine row, under the id of the session row. The config that a profile
    # writes for `code-graph` is then the config that this command reads.
    rows=({"id": ROW, "name": "code-graph-engine"},),
    command=ROW,
    inherit_env=(GRAMMAR_CACHE_ENV,),
    module="ph_code_graph",
)


def main() -> NoReturn:
    run_cli(CLI)
