"""`ph-text-index` — `text_index` and `text_search` from a command line.

For trying a query by hand, and what this package's UTCP manual calls. The tools
are the ones the `text-index-engine` row registers (`ph_text_index.engine`);
`ph.toolcli` mounts that row and an embedder's in this process and runs them
through the registry. The rows a session mounts reach them by running this
command.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --format text
ph-text-index status
ph-text-index manual
```

The embedder is the local `sentence-transformers` model unless a config names
another provider row (`ToolCli.providers`); its settings — the model, its
prefixes — are that row's, in `$PH_HOME/tools/ph-text-index.yaml` for a person
and handed across by the session's row for an agent.

@module ph_text_index.cli
"""

from __future__ import annotations

from importlib.metadata import version
from typing import NoReturn

from ph.toolcli import ToolCli
from ph.toolcli import main as run_cli

from ._config import ADAPTER_ROW, ENDPOINT_ROW, LOCAL_ROW, ROW

__all__ = ["CLI", "MODEL_ENV", "main"]

MODEL_ENV: tuple[str, ...] = (
    "HF_ENDPOINT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)
"""What an operator sets to say how a local model is fetched and run: a mirror,
no network, how many threads, which device. Not `HF_TOKEN` — the deployment's
scrubbed environment drops a credential whatever a template names."""

CLI = ToolCli(
    program="ph-text-index",
    manual="ph_text_index",
    summary="Index documents by meaning and search them: passages back, with their lines.",
    version=version("ph-text-index"),
    # The engine row, under the session row's id: the config a profile writes for
    # `text-index` is then the config this command reads for it.
    rows=({"id": ROW, "name": "text-index-engine"},),
    # What makes a vector. One of these, and the first unless a config says.
    providers=(
        {"id": LOCAL_ROW, "name": "text-index-local-engine"},
        {"id": ENDPOINT_ROW, "name": "text-index-endpoint-engine"},
        {"id": ADAPTER_ROW, "name": "text-index-adapter-engine"},
    ),
    command=ROW,
    inherit_env=MODEL_ENV,
    module="ph_text_index",
)


def main() -> NoReturn:
    run_cli(CLI)
