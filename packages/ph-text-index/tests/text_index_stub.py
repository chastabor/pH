"""A deterministic embedder, and a `ph-text-index` command that mounts it.

## Why a stub embedder and not the real model

The shipped provider is `sentence-transformers`, which downloads weights on
first use. A suite that loaded it would fail in an airgapped CI. It would take a
minute to prove that paging works, and it would test the model and not this
package. The suites therefore use a deterministic hashing embedder and exercise
**the real turbovec index** through it.

`HashingEmbedder` is a bag of words over `blake2b` buckets. Similar text
therefore really produces similar vectors, and the retrieval assertions are
about ranking and not about a fixture. It uses `blake2b` and not `hash()`,
which is salted for each process. A stub whose vectors changed between runs
would make a reopened index look corrupt. Here one process builds the index and
the next process searches it.

## Why it is also a command

The tools run in the command of the package. A suite that proves the rows of
the session therefore needs a command whose embedder is this one.
`python -m text_index_stub` is `ph-text-index` with one difference: its
embedder row is `engine` below. Everything else is the package's own: the
engine row, the tools, the envelope, and the manual (`ph.toolcli`).

This module is separate from `text_index_helpers`, which the session side of a
test imports. The process of the command loads this module on every call, and
it must not pay for `ph-utcp` to do that.

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
from ph.testing import llama_refusal
from ph.toolcli import ToolCli, main
from ph.tools.errors import HarnessError
from ph.wire import WireModel
from ph_text_index._config import ROW, Limit
from ph_text_index._embed import Vectors, estimate_tokens, weigh_by_bytes
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
    """How many seconds each `encode` takes. This stands for the cost of the
    forward pass of a model. A test uses it when a run must still be working at
    the time that the test looks."""

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
    """A hashing embedder that reads only a set number of tokens. A token is a
    word here, so a reader can count the tokens in a test.

    There are three separate facts, because the code under test treats them
    separately:

    * `max_tokens` is what a profile *set*.
    * `reads` is what the model *reports* when it is asked.
    * `refuses_over` is what the model *does* with an input that is too long. It
      refuses the input, in the words of a llama.cpp server.

    `asked` counts how many times the limit was asked for. For a real model,
    asking means loading the model.
    """

    max_tokens: int | None = None
    reads: int | None = None
    refuses_over: int | None = None
    vague: bool = False
    """Refuse without numbers, like a provider whose refusal names no number."""
    estimates: bool = False
    """Measure by bytes, as an embedder with no tokenizer does, and not by the
    words that this embedder really counts."""
    prefix: str = ""
    """The text that an estimate counts in front of each passage, like the
    `document_prefix` of an endpoint."""
    breaks: int = 0
    """How many tokens a blank line between two paragraphs costs when a passage
    is counted whole. Weighing the paragraphs one at a time does not count them.
    The count that a tokenizer gives for a whole passage can be more than the
    sum of the counts of its parts."""
    asked: int = 0
    refused: int = 0
    """How many calls were turned away as too long."""
    took: list[int] = field(default_factory=list)
    """The longest input, in words, of each call that was answered."""

    def limit(self) -> Limit | None:
        self.asked += 1
        if self.max_tokens is not None:
            return Limit(tokens=self.max_tokens, source="configured")
        return None if self.reads is None else Limit(tokens=self.reads, source="model")

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        if self.estimates:
            return [estimate_tokens(f"{self.prefix}{text}", bytes_per_token) for text in texts]
        return [self._whole(text) for text in texts]

    def _whole(self, text: str) -> int:
        """The count of `text` as one input: its words plus its paragraph breaks."""
        return len(text.split()) + self.breaks * text.count("\n\n")

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        if self.estimates:
            return weigh_by_bytes(texts, bytes_per_token)
        return [len(text.split()) for text in texts]

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        longest = max((self._whole(text) for text in texts), default=0)
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


class StubConfig(WireModel):
    """Row config for the stub embedder: what stands in for "which model"."""

    dim: int = 96
    delay: float = 0.0
    """`HashingEmbedder.delay`."""
    record: str = ""
    """A file that gets one more line each time the embedder is made, which
    stands for each time "the model is loaded". A test can then count the
    processes that loaded one."""


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
