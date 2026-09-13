"""What turns a passage into a vector, and the local model that does it here.

**A protocol, and then one provider** — the shape `ctx.subprocess` and
`ctx.code_runtime` already have in this tree. `text-index` mounts the seam and
the tools; `text-index-local` mounts a `sentence-transformers` model. The split
earns itself three times over:

* a deployment with an embeddings *endpoint* — llama.cpp's `/v1/embeddings`, a
  provider's API — writes its own row and keeps the tools, rather than forking
  them;
* the tests exercise the real turbovec index against a deterministic stub, so
  the suite does not download a model or install torch to prove that chunking
  and paging work;
* the row that pulls in a two-gigabyte dependency is one a profile can leave
  out, which is the only way `ph-text-index` can be installed by someone who
  wanted the seam and brought their own vectors.

`name` is not decoration: it is written into the index's sidecar and compared on
every open. A vector from one model is meaningless to another, so an index whose
embedder changed under it must refuse rather than return neighbours computed in
a space nothing shares.

@module ph_text_index._embed
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Embedder", "LocalWeights", "SentenceTransformerEmbedder"]

log = logging.getLogger("ph_text_index.embed")


@runtime_checkable
class Embedder(Protocol):
    """Text in, one row of floats out."""

    @property
    def name(self) -> str:
        """A stable identity for the vector space this produces.

        Stored in the index and compared on open, so it has to change whenever
        the space does — the model, and anything else that moves a vector.
        """

    def encode(self, texts: Sequence[str], *, query: bool) -> Any:  # noqa: ANN401
        """`(len(texts), dim)` float32, L2-normalised.

        **Blocking**; the seam calls it in a worker thread. Normalised because
        turbovec scores an inner product, and an inner product over normalised
        vectors is cosine similarity — without that, a long passage outranks a
        relevant one for having a bigger norm.

        `query` distinguishes the two sides of an asymmetric model, which want
        different instructions for a question than for a passage. A symmetric
        model ignores it.
        """


@runtime_checkable
class LocalWeights(Protocol):
    """An embedder with weights to fetch and load. **Optional capability.**

    Its own Protocol rather than a `getattr` probe, which is the rule this tree
    states three times — `ReclaimingProvider`, `ExportingProvider`,
    `RehydratableProvider` — and argues in the first of them: *"an optional
    capability as its own Protocol, never a `getattr` probe … a probe would
    report a provider whose method is misnamed as one that cannot"*. The failure
    was live: an embedder whose `ready` was misspelled reported "not loaded"
    forever, and a misspelled `load` made `/text-index install` answer "needs no
    download".

    An endpoint-backed embedder implements none of this and is simply not an
    instance, which is the whole point of a second Protocol.
    """

    cache_folder: str

    def ready(self) -> bool:
        """Whether the model is loaded in this process."""

    def load(self) -> int:
        """Fetch and load it now, and return its dimension."""


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """A local `sentence-transformers` model, loaded on first use.

    **Not at mount.** The first load downloads weights, and a harness that
    stalled for a few hundred megabytes before it could answer `ph doctor` would
    be paying for a capability the session may never call. The refusal that
    belongs at mount is the one about the *package* being absent, which is
    cheap; the model itself is acquired when something asks for a vector.
    """

    model_name: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32
    cache_folder: str = ""
    """Where the weights land. Empty leaves it to `sentence-transformers`.

    The row passes `$PH_CACHE/models`, and that is the point: left alone,
    `sentence-transformers` writes to `$HF_HOME` or `~/.cache/huggingface` —
    **outside all three of pH's roots**, so a gigabyte of weights would sit
    somewhere `ph doctor` never mentions and deleting `$PH_CACHE` would not
    reclaim. Rebuildable and large is exactly the lifecycle `$PH_CACHE` names
    (Q1), and it is where the runtime venv lives for the same reason.

    Through the constructor argument rather than by setting `HF_HOME`: a row
    that mutated the process environment would change it for every child
    `ctx.subprocess` spawns as well, and this is a decision about one library.
    """
    trust_remote_code: bool = False
    """Whether to let this model bring its own modelling code.

    **Off by default, and it is a trust decision rather than a compatibility
    flag.** `True` downloads Python from the model's repository and executes it
    in this process — for `nomic-ai/nomic-embed-text-v1.5` from a *second* repo
    at that (`nomic-bert-2048`, via `auto_map`). A deployment that wants such a
    model says so; nothing infers it from a load failure, because "retry with
    arbitrary code execution enabled" is not a fallback.

    Deliberately **not** part of `name`: it changes what is allowed to load, not
    where a vector lands, so an index does not become invalid because an
    operator granted or revoked it.
    """
    _model: Any = field(default=None, init=False)

    @property
    def name(self) -> str:
        """The model, plus the prefixes — all three move the vector space.

        The prefixes are in the identity because they are part of the function:
        an `e5` index built with `passage: ` and searched without it is a
        different space, and the failure is quiet — slightly wrong neighbours,
        which is worse than an error.
        """
        return f"sentence-transformers:{self.model_name}:{self.query_prefix}|{self.document_prefix}"

    def _load(self) -> Any:  # noqa: ANN401
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("ph_text_index: loading %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_folder=self.cache_folder or None,
            )
        return self._model

    def load(self) -> int:
        """Fetch and load the model now, and return its dimension.

        **The whole point is that it loads rather than downloads.** Fetching the
        weights is not the same check: `nomic-embed-text-v1.5` downloads its
        config and its remote modelling code successfully and then fails at
        import with

            ImportError: This modeling file requires the following packages
            that were not found in your environment: einops

        — a missing transitive dependency that only the model's own code knows
        about, and that no amount of pre-downloading would have surfaced. So the
        provisioning path calls this, and whatever it raises is what the person
        reads.
        """
        model = self._load()
        # `get_sentence_embedding_dimension` is deprecated in favour of
        # `get_embedding_dimension`; both spellings exist across the versions
        # this package allows, so ask for the new one and fall back rather than
        # pinning a floor for a method name.
        measure = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(measure())

    def ready(self) -> bool:
        """Whether this model is loaded in *this process*.

        Deliberately not "is it in the cache": a cached model that will not
        import — the `einops` case — is not ready, and a diagnostic that said it
        was would be answering an easier question than the one being asked.
        """
        return self._model is not None

    def encode(self, texts: Sequence[str], *, query: bool) -> Any:  # noqa: ANN401
        import numpy as np

        prefix = self.query_prefix if query else self.document_prefix
        rows = self._load().encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(rows, dtype=np.float32)
