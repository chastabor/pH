"""A deterministic embedder, and a `ph-text-index` command that mounts it.

## Why a stub embedder rather than the real model

The shipped provider is `sentence-transformers`, which downloads weights on
first use. A suite that loaded it would be a suite that fails in an airgapped
CI, takes a minute to prove that paging works, and tests the model rather than
this package. So the suites use a deterministic hashing embedder and exercise
**the real turbovec index** through it.

`HashingEmbedder` is a bag of words over `blake2b` buckets, so similar text
really does produce similar vectors and the retrieval assertions are about
ranking rather than about a fixture. `blake2b` and not `hash()`, which is salted
per process — a stub whose vectors changed between runs would make a reopened
index look corrupt, and here the index is built by one process and searched by
the next.

## Why it is also a command

The tools run in the package's command, so a suite that proves the session's
rows needs a command whose embedder is this one. `python -m text_index_stub` is
`ph-text-index` with one difference: its embedder row is `engine` below. It is
the package's own engine row, tools, envelope and manual otherwise
(`ph.toolcli`).

Apart from `text_index_helpers`, which the session's side of a test imports:
this module is what the command's process loads on every call, and it must not
pay for `ph-utcp` to do it.

@module text_index_stub
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ph.cordis import Context, plugin
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED
from ph.seams._registry import contribute_item
from ph.toolcli import ToolCli, main
from ph.tools.errors import HarnessError
from ph.wire import WireModel
from ph_text_index._config import ROW
from ph_text_index._embed import Limit, Vectors, estimate_tokens
from ph_text_index.engine import TEXT_INDEX

__all__ = [
    "DOCUMENT",
    "STUB_CLI",
    "STUB_ROW",
    "BoundedEmbedder",
    "HashingEmbedder",
    "StubConfig",
    "engine",
]

DOCUMENT = """# Workspaces

The workspace seam decides where an agent's writes land.

A containment tier is what a deployment asks for: advisory keeps the agent in
the person's own checkout, and a worktree branches a fresh one.

## Sandboxing

Confinement is a different question, answered by the sandbox backend — bwrap on
Linux and Seatbelt on macOS.
"""

STUB_ROW = "text-index-stub"
"""The stub embedder's row id, in a session and in the command alike."""


@dataclass(slots=True)
class HashingEmbedder:
    """A deterministic bag-of-words embedder. See the module docstring."""

    dim: int = 96
    calls: int = 0
    delay: float = 0.0
    """Seconds each `encode` takes: what a model's forward pass costs, for a test
    that needs a run to still be going when it looks."""

    @property
    def name(self) -> str:
        return f"stub-hashing:{self.dim}"

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            for token in "".join(
                character if character.isalnum() else " " for character in text.lower()
            ).split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                rows[index, int.from_bytes(digest, "big") % self.dim] += 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(rows / norms, dtype=np.float32)


@dataclass(slots=True)
class BoundedEmbedder(HashingEmbedder):
    """A hashing embedder that reads only so many tokens, where a token is a
    word — so a test can count them by eye.

    Three separate facts, because the code under test treats them separately:
    what a profile *set* (`max_tokens`), what the model *reports* when asked
    (`reads`), and what it *does* with an input that is too long
    (`refuses_over`: refuse it, in a llama.cpp's words). `asked` counts the
    times the limit was asked for, which of a real model means loading it.
    """

    max_tokens: int | None = None
    reads: int | None = None
    refuses_over: int | None = None
    vague: bool = False
    """Refuse without numbers, as a provider whose prose names none."""
    estimates: bool = False
    """Measure by the bytes, as an embedder with no tokenizer does, and not by
    the words it really counts."""
    prefix: str = ""
    """What an estimate counts in front of each passage, as an endpoint's
    `document_prefix` is."""
    asked: int = 0
    refused: int = 0
    """How many calls were turned away as too long."""
    took: list[int] = field(default_factory=list)
    """The longest input, in words, of each call that was answered."""

    def limit(self) -> Limit | None:
        self.asked += 1
        if self.max_tokens is not None:
            return Limit(self.max_tokens, "configured")
        return None if self.reads is None else Limit(self.reads, "model")

    def measure(self, texts: Sequence[str], bytes_per_token: float) -> list[int]:
        if self.estimates:
            return [estimate_tokens(f"{self.prefix}{text}", bytes_per_token) for text in texts]
        return [len(text.split()) for text in texts]

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        longest = max((len(text.split()) for text in texts), default=0)
        if self.refuses_over is not None and longest > self.refuses_over:
            self.refused += 1
            raise HarnessError(
                "that input is too long"
                if self.vague
                else llama_refusal(longest, self.refuses_over),
                CONTEXT_WINDOW_EXCEEDED,
            )
        self.took.append(longest)
        return HashingEmbedder.encode(self, texts, query=query)


def llama_refusal(sent: int, reads: int) -> str:
    """What a llama.cpp says when an input is longer than its batch: both numbers,
    as `ph.llm.adapter.too_long_facts` reads them."""
    return (
        f"input ({sent} tokens) is too large to process. increase the "
        f"physical batch size (current batch size: {reads})"
    )


class StubConfig(WireModel):
    """Row config for the stub embedder: what stands in for "which model"."""

    dim: int = 96
    delay: float = 0.0
    """`HashingEmbedder.delay`."""
    record: str = ""
    """A file to add a line to each time the embedder is made — each time "the
    model is loaded" — so a test can count the processes that loaded one."""


@plugin("text-index-stub-engine", affects="environment", inject=[TEXT_INDEX], config=StubConfig)
async def engine(ctx: Context, config: StubConfig) -> None:
    """Register the hashing embedder, as `text-index-local-engine` registers a model."""
    if config.record:
        with Path(config.record).open("a", encoding="utf-8") as record:
            record.write(f"{os.getpid()}\n")
    contribute_item(
        ctx,
        TEXT_INDEX,
        HashingEmbedder(dim=config.dim, delay=config.delay),
        label="embedder(stub)",
    )


STUB_CLI = ToolCli(
    program="ph-text-index",
    manual="ph_text_index",
    summary="The text index, with a deterministic embedder, for a test.",
    version="0.0.0",
    rows=({"id": ROW, "name": "text-index-engine"},),
    providers=({"id": STUB_ROW, "name": "text_index_stub:engine"},),
    command=ROW,
    # This module is on the suite's `pythonpath`, not installed.
    inherit_env=("PYTHONPATH",),
    module="text_index_stub",
)

if __name__ == "__main__":
    main(STUB_CLI)
