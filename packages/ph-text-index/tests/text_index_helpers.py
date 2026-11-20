"""What the `ph-text-index` suites share on the side of the session: an agent, a
tree, and the row that names the stub embedder.

This is a `<package>_helpers.py` on `pythonpath`, like `code_graph_helpers`, and
not a `conftest.py`. The tests of `ph-rlm` import their conftest as a top-level
module. A second file of that name anywhere in `testpaths` shadows it on
`sys.path`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from text_index_stub import (
    DOCUMENT,
    STUB_CLI,
    STUB_ROW,
    HashingEmbedder,
    StubConfig,
)

from ph.cordis import Context, plugin
from ph.testing import llama_refusal, new_agent
from ph_text_index.row import TEXT_INDEX, Provider
from ph_utcp import UTCP

__all__ = [
    "KEY_ENV",
    "ROWS",
    "Endpoint",
    "command_environment",
    "docs_tree",
    "embeddings_server",
    "long_document",
    "new_agent",
    "prose",
    "stub",
]

ROWS: tuple[dict[str, Any], ...] = (
    {"id": "utcp", "name": "utcp"},
    {"id": "text-index", "name": "text-index"},
    {"id": STUB_ROW, "name": "text_index_helpers:stub"},
)
"""What the `utcp` and `text-index` bundles add to a profile, in their order. The
row of the stub embedder is where the bundle has `text-index-local`."""


@plugin(STUB_ROW, affects="environment", inject=[TEXT_INDEX, UTCP], config=StubConfig)
async def stub(ctx: Context, config: StubConfig) -> None:
    """Name the stub embedder, as `text-index-local` names a model. The name says
    which row of which command makes the vectors, and gives the config of that
    row."""
    await ctx.require(TEXT_INDEX).register(
        Provider(row=STUB_ROW, config=config, cli=STUB_CLI, label=f"stub-hashing:{config.dim}"),
        scope=ctx,
    )


@pytest.fixture(autouse=True)
def command_environment(tmp_path: Path) -> Iterator[Path]:
    """What the `ph-text-index` processes of a test need, for the length of the
    test.

    **`PYTHONPATH`**, so that `python -m text_index_stub` finds this directory.
    The stub command is on the `pythonpath` of the suite. pytest gives that path
    to its own process and to no child process.

    **`$PH_CACHE` beside the workspace, and a short `$PH_RUNTIME` for the test**,
    for the reasons that `code_graph_helpers.cache_beside_the_tree` gives. A
    grant that overlaps the workspace is refused, and the address of a socket is
    at most 108 bytes.

    The fixture is autouse in each module that imports it, after
    `_isolated_home`, so these values win.
    """
    directory = tmp_path.parent / f"{tmp_path.name}-cache"
    here = str(Path(__file__).parent)
    inherited = os.environ.get("PYTHONPATH")
    runtime = Path(tempfile.mkdtemp(prefix="ph-run-", dir="/tmp"))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("PH_CACHE", str(directory))
            patch.setenv("PH_RUNTIME", str(runtime))
            patch.setenv("PYTHONPATH", f"{here}{os.pathsep}{inherited}" if inherited else here)
            yield directory
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


KEY_ENV = "FX_EMBEDDINGS_API_KEY"
"""A name that the scrubbed environment of the deployment drops, because it
contains `KEY`."""


@dataclass(slots=True)
class Endpoint:
    """The fixture server: where it listens, and what it has been asked."""

    url: str
    asked: list[dict[str, Any]] = field(default_factory=list)
    keys: list[str | None] = field(default_factory=list)
    status: int = 200
    window: int | None = None
    """How many tokens `/props` says that the server reads, as a llama.cpp server
    answers. `None` is a server with nothing to ask, which answers 404."""
    reads: int | None = None
    """How many tokens an input may have before the server refuses it, in the
    sentence that a llama.cpp server uses. A token is a word here."""
    took: list[int] = field(default_factory=list)
    """The longest input, in words, of each embeddings request it answered."""
    props: int = 0
    """How many times it was asked what it reads."""
    refused: int = 0
    """How many embeddings requests it refused as too long."""


@pytest.fixture(name="endpoint")
def embeddings_server() -> Iterator[Endpoint]:
    """`endpoint`, to a test: a server that speaks the OpenAI-compatible
    embeddings request, with the hashing embedder of the suite behind it. It
    doubles every vector on the way out, because not every server normalizes.
    It keeps what it was asked. That is how a test sees the key that a request
    carried, and the key is never printed."""
    embedder = HashingEmbedder(dim=96)
    state = Endpoint(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.asked.append({"path": self.path, **body})
            state.keys.append(self.headers.get("Authorization"))
            if state.status != 200:
                self._answer(state.status, {"error": {"message": "the model is loading"}})
                return
            longest = max((len(text.split()) for text in body["input"]), default=0)
            if state.reads is not None and longest > state.reads:
                state.refused += 1
                said = llama_refusal(longest, state.reads)
                self._answer(500, {"error": {"code": 500, "message": said}})
                return
            state.took.append(longest)
            vectors = embedder.encode(body["input"], query=False) * 2.0
            # Out of order, as the API allows: `index` is what places a vector.
            data = [
                {"index": index, "embedding": row.tolist()}
                for index, row in reversed(list(enumerate(vectors)))
            ]
            usage = {"prompt_tokens": sum(len(text.split()) for text in body["input"])}
            self._answer(200, {"data": data, "model": body.get("model", ""), "usage": usage})

        def do_GET(self) -> None:
            if self.path == "/props" and state.window is not None:
                state.props += 1
                said = {"default_generation_settings": {"n_ctx": state.window}, "total_slots": 4}
                self._answer(200, said)
            else:
                self._answer(404, {"error": {"message": "not found"}})

        def _answer(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def prose(paragraphs: int) -> str:
    """`paragraphs` paragraphs of three lines, with six words in each line. Each
    paragraph is 18 words and about 125 characters, so a reader can work out the
    sizes in a test."""
    return "\n\n".join(
        "\n".join(" ".join(f"w{p}l{line}n{n}" for n in range(6)) for line in range(3))
        for p in range(paragraphs)
    )


def long_document(root: Path, paragraphs: int = 12) -> Path:
    """`root/docs/long.md`, which holds `paragraphs` paragraphs of `prose`. In
    total it is longer than a small model reads in one input."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "long.md").write_text(prose(paragraphs), encoding="utf-8")
    return root


def docs_tree(root: Path, documents: int = 2) -> Path:
    """`root/docs`: `w.md`, the workspaces document, and `documents - 1` others,
    each about something else."""
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "w.md").write_text(DOCUMENT, encoding="utf-8")
    for n in range(documents - 1):
        (docs / f"billing{n}.md").write_text(
            f"# Billing {n}\n\nInvoices are issued monthly, and number {n} was late.\n",
            encoding="utf-8",
        )
    return root
