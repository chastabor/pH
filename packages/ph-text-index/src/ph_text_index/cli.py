"""`ph-text-index`: `text_index` and `text_search` from a command line.

The command is for trying a query by hand, and it is what the UTCP manual of this
package calls. The tools are the ones that the `text-index-engine` row registers
(`ph_text_index.engine`). `ph.toolcli` mounts that row and an embedder row in
this process, and runs the tools through the registry. The rows that a session
mounts reach the tools by running this command.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --format text
ph-text-index status
ph-text-index manual
```

The embedder is the local `sentence-transformers` model unless a config names
another provider row (`ToolCli.providers`). The settings of the embedder (the
model and its prefixes) belong to that row. For a person, they are in
`$PH_HOME/tools/ph-text-index.yaml`. For an agent, the session row passes them
across.

@module ph_text_index.cli
"""

from __future__ import annotations

from importlib.metadata import version
from typing import NoReturn

from ph.toolcli import ToolCli
from ph.toolcli import main as run_cli

from ._config import ENDPOINT_ROW, LOCAL_ROW, ROW

__all__ = ["CLI", "MODEL_ENV", "main"]

MODEL_ENV: tuple[str, ...] = (
    "HF_ENDPOINT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)
"""The variables that an operator sets to say how a local model is fetched and
run: a mirror, no network, the number of threads, and the device. `HF_TOKEN` is
not here. The scrubbed environment of the deployment drops a credential,
whatever a template names."""

CLI = ToolCli(
    program="ph-text-index",
    manual="ph_text_index",
    summary="Index documents by meaning and search them: passages back, with their lines.",
    version=version("ph-text-index"),
    # The engine row, under the id of the session row. The config that a profile
    # writes for `text-index` is then the config that this command reads.
    rows=({"id": ROW, "name": "text-index-engine"},),
    # The rows that make a vector. The command mounts one of them: the first,
    # unless a config names another.
    providers=(
        {"id": LOCAL_ROW, "name": "text-index-local-engine"},
        {"id": ENDPOINT_ROW, "name": "text-index-endpoint-engine"},
    ),
    command=ROW,
    inherit_env=MODEL_ENV,
    module="ph_text_index",
)


def main() -> NoReturn:
    run_cli(CLI)
