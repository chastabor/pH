"""What turns a passage into a vector: the embedder protocol, and the two embedders here.

**A protocol, and then the providers.** `ctx.subprocess` and `ctx.code_runtime`
already have this shape in this tree. `text-index` mounts the seam and the tools,
and one embedder row names what makes the vectors. This module holds both
embedders that this package ships:

* `SentenceTransformerEmbedder` is a `sentence-transformers` model on this
  machine. `text-index-local` names it.
* `EndpointEmbedder` is an OpenAI-compatible `/v1/embeddings` endpoint: a local
  server such as llama.cpp, or the API of a provider. `text-index-endpoint`
  names it.

The split between the seam and the embedder has three benefits:

* A deployment with an embeddings *endpoint* mounts `text-index-endpoint` and
  keeps the tools. A deployment with another source of vectors writes its own
  row. Neither forks the tools.
* The tests exercise the real turbovec index against a deterministic stub. The
  suite therefore does not download a model or install torch to prove that
  chunking and paging work.
* A profile can leave out the row that pulls in a two-gigabyte dependency. That
  is the only way that someone who wanted the seam and brought their own vectors
  can install `ph-text-index`.

`name` is not decoration. It is written into the sidecar of the index and
compared on every open. A vector from one model is meaningless to another model.
An index whose embedder changed must therefore refuse, and not return neighbors
that were computed in a space that nothing shares.

@module ph_text_index._embed
"""

from __future__ import annotations

import logging
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from ph.characters import capped_text
from ph.json import JsonValue, as_int, as_obj, as_seq, dumps, loads
from ph.llm.adapter import input_too_long
from ph.llm.tokens import tokens_of
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, MISSING_CREDENTIAL
from ph.tools.errors import HarnessError

from ._config import Facts, Limit

__all__ = [
    "ENDPOINT_FAILED",
    "ENDPOINT_UNREACHABLE",
    "Embedder",
    "EndpointEmbedder",
    "LocalWeights",
    "Metered",
    "SentenceTransformerEmbedder",
    "Served",
    "TokenBounded",
    "Vectors",
    "estimate_tokens",
    "weigh_by_bytes",
]

ENDPOINT_UNREACHABLE = "TEXT_INDEX_ENDPOINT_UNREACHABLE"
ENDPOINT_FAILED = "TEXT_INDEX_ENDPOINT_FAILED"

log = logging.getLogger("ph_text_index.embed")


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Vectors: TypeAlias = "NDArray[np.float32]"
"""One L2-normalized float32 row for each text: `(len(texts), dim)`.

This is the shape that the whole chain moves. `Embedder.encode` produces it,
`TextIndexSeam.embed` carries it across the hop to the worker thread, and
`TextIndex.add`, `_maybe_calibrate` and `search` consume it. The shape has a name
and is not repeated, so that the stub embedder that a test registers is held to
what the store indexes.

numpy is imported under `TYPE_CHECKING` only. It is loaded lazily inside the
functions that need it, and an annotation must not be the reason that it loads at
import time.
"""


def estimate_tokens(text: str, bytes_per_token: float) -> int:
    """About how many tokens `text` is, for an embedder that has no tokenizer to
    ask. This is its bytes divided by `bytes_per_token`, plus two for the markers
    that a model adds. `bytes_per_token` is `BYTES_PER_TOKEN`, or the density of
    the corpus after a refusal showed it."""
    return math.ceil(tokens_of(text, bytes_per_token)) + 2


def weigh_by_bytes(texts: Sequence[str], bytes_per_token: float) -> list[float]:
    """About how many tokens each text adds to a passage, when there is no
    tokenizer to ask. This is its bytes divided by `bytes_per_token`, not
    rounded. The weights of the parts of a passage therefore add up to the weight
    of the passage, and `estimate_tokens` of the whole is never over what the
    weights said."""
    return [tokens_of(text, bytes_per_token) for text in texts]


@runtime_checkable
class TokenBounded(Protocol):
    """An embedder that can report how long an input it reads. **Optional
    capability**, with its own Protocol for the same reason as `LocalWeights`.

    The size of a passage concerns the embedder, and is not only a number in a
    profile, because a model reads a limited number of tokens and no more. Past
    that number, a local model drops the rest without a warning: the end of the
    passage is indexed and can never be found. A server refuses the request.
    `limit` gives the bound, `weigh` gives the weights that a passage is packed
    by, and `measure` checks what is sent against the bound.
    """

    max_tokens: int | None
    """The setting, when a profile wrote one. It is read without loading anything,
    so a run can tell that the setting changed before it needs the model."""

    def limit(self) -> Limit | None:
        """The bound: the setting, or else what the model or its server reports.
        `None` when nothing gives one. The call may load a model or ask a
        server."""

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        """How many tokens each of `texts` is as it is sent, as a document or as a
        question, with the prefix and the markers included. The tokens are
        counted where there is a tokenizer. Otherwise they are estimated at
        `bytes_per_token`, which is the density of the corpus as far as it is
        known."""

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        """How many tokens each of `texts` adds to a passage that it is part of:
        its own tokens, with no prefix and no markers. Passages are packed by
        this weight (`_chunk.Weight`), so it is called for a paragraph and for a
        line."""


@runtime_checkable
class Served(Protocol):
    """An embedder behind a server. **Optional capability.**

    The server decides what such an embedder reads, and a server can be moved or
    reconfigured. What the server reported is therefore kept with the address
    that was asked (`EmbedderRecord`), and another address is asked again."""

    def asked_of(self) -> str:
        """The address `limit` asks."""

    def reported(self) -> Facts:
        """What the server reported when `limit` last asked it, unchanged. Empty
        when it was not asked or did not answer."""


@runtime_checkable
class Metered(Protocol):
    """An embedder that counts what it spends. **Optional capability.**"""

    def spent(self) -> Mapping[str, int]:
        """What it has spent since it was created, by unit: the `requests` that it
        made, the `texts` that it embedded, and the `tokens` that its provider
        reported, if the provider reports any. Each is counted after it
        succeeds."""


@runtime_checkable
class Embedder(Protocol):
    """Text in, one row of floats out."""

    @property
    def name(self) -> str:
        """A stable identity for the vector space that this embedder produces.

        It is stored in the index and compared on open. It must therefore change
        whenever the space changes: with the model, and with anything else that
        moves a vector.
        """

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        """`(len(texts), dim)` float32, L2-normalized.

        **This blocks**, so the seam calls it in a worker thread. The rows are
        normalized because turbovec scores an inner product, and an inner product
        over normalized vectors is cosine similarity. Without normalization, a
        long passage outranks a relevant passage because it has a bigger norm.

        `query` distinguishes the two sides of an asymmetric model, which needs
        different instructions for a question than for a passage. A symmetric
        model ignores it.
        """


@runtime_checkable
class LocalWeights(Protocol):
    """An embedder with weights to fetch and load. **Optional capability.**

    This is its own Protocol and not a `getattr` probe. This tree states that
    rule three times (`ReclaimingProvider`, `ExportingProvider`,
    `RehydratableProvider`) and argues it in the first of them: *"an optional
    capability as its own Protocol, never a `getattr` probe … a probe would
    report a provider whose method is misnamed as one that cannot"*. The failure
    really happened. An embedder whose `ready` was misspelled reported "not
    loaded" permanently, and a misspelled `load` made `/text-index install`
    answer "needs no download".

    An endpoint-backed embedder implements none of this and is simply not an
    instance. That is the purpose of a second Protocol.
    """

    cache_folder: str

    def ready(self) -> bool:
        """Whether the model is loaded in this process."""

    def cached(self) -> bool:
        """Whether its weights are on disk, so a load fetches nothing."""

    def load(self) -> int:
        """Fetch and load it now, and return its dimension."""


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """A local `sentence-transformers` model, loaded on first use.

    **It is not loaded at mount.** The first load downloads weights. A harness
    that waited for a few hundred megabytes before it could answer
    `phern doctor` would pay for a capability that the session may never call.
    The refusal that belongs at mount is the one for an absent *package*, which
    is cheap to check. The model itself is acquired when something asks for a
    vector.
    """

    model_name: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32
    cache_folder: str = ""
    """Where the weights are stored. An empty value leaves the choice to
    `sentence-transformers`.

    The row passes `$PH_CACHE/models`, on purpose. By default,
    `sentence-transformers` writes to `$HF_HOME` or `~/.cache/huggingface`, which
    is **outside all three roots of pH**. A gigabyte of weights would then be in
    a place that `phern doctor` never mentions and that deleting `$PH_CACHE` does
    not reclaim. `$PH_CACHE` is the root for data that is large and can be
    rebuilt (Q1), and the runtime venv is there for the same reason.

    The value goes through the constructor argument and not through `HF_HOME`. A
    row that changed the process environment would also change it for every
    child that `ctx.subprocess` spawns, and this is a decision about one library.
    """
    max_tokens: int | None = None
    """A bound on an input, below the bound of the model itself. See `limit`."""
    trust_remote_code: bool = False
    """Whether this model may bring its own modeling code.

    **It is off by default, and it is a trust decision, not a compatibility
    flag.** `True` downloads Python from the repository of the model and executes
    it in this process. For `nomic-ai/nomic-embed-text-v1.5` the code even comes
    from a *second* repo (`nomic-bert-2048`, through `auto_map`). A deployment
    that wants such a model says so. Nothing infers it from a load failure,
    because "retry with arbitrary code execution enabled" is not a fallback.

    It is deliberately **not** part of `name`. It changes what is allowed to
    load, not where a vector lands. An index therefore does not become invalid
    because an operator granted or revoked it.
    """
    _model: Any = field(default=None, init=False)
    _spent: Counter[str] = field(default_factory=Counter, init=False)

    @property
    def name(self) -> str:
        """The model, plus the prefixes. All three move the vector space.

        The prefixes are in the identity because they are part of the function.
        An `e5` index that is built with `passage: ` and searched without it is a
        different space. The failure is quiet: slightly wrong neighbors, which is
        worse than an error.
        """
        return f"sentence-transformers:{self.model_name}:{self.query_prefix}|{self.document_prefix}"

    def _load(self) -> Any:  # noqa: ANN401
        if self._model is None:
            # Not a declared dependency: the row checks for it before this runs.
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            log.info("ph_text_index: loading %s", self.model_name)

            def build(*, offline: bool) -> Any:  # noqa: ANN401
                return SentenceTransformer(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    cache_folder=self.cache_folder or None,
                    local_files_only=offline,
                )

            # **Weights that are already on disk are loaded without asking the
            # hub.** By default, the library checks the repository for a newer
            # revision on every load, so a model that is entirely cached still
            # needs the network. A confined command reaches only the hosts that
            # its deployment allows. There, the check was answered with
            # `403 Forbidden`, and a search failed over an index that needed
            # nothing fetched. The offline load also means that a load cannot
            # quietly become a download.
            #
            # An `OSError` from the offline load means a cache that is not whole:
            # an interrupted download leaves the directory behind. The hub can
            # repair it, so that one case is tried again with the network. Any
            # other error is a failure of the model itself and is raised
            # unchanged.
            if self.cached():
                try:
                    self._model = build(offline=True)
                except OSError:
                    log.warning(
                        "ph_text_index: %s is cached but did not load from disk; asking the hub",
                        self.model_name,
                        exc_info=True,
                    )
                    self._model = build(offline=False)
            else:
                self._model = build(offline=False)
        return self._model

    def load(self) -> int:
        """Fetch and load the model now, and return its dimension.

        **The purpose is that it loads, and does not only download.** A fetch of
        the weights is not the same check. `nomic-embed-text-v1.5` downloads its
        config and its remote modeling code successfully, and then fails at
        import with

            ImportError: This modeling file requires the following packages
            that were not found in your environment: einops

        That is a missing transitive dependency that only the code of the model
        knows about, and no amount of downloading in advance would have shown it.
        The provisioning path therefore calls this method, and the user reads
        whatever it raises.
        """
        model = self._load()
        # `get_sentence_embedding_dimension` is deprecated in favor of
        # `get_embedding_dimension`. Both names exist across the versions that
        # this package allows. Ask for the new name and fall back to the old one,
        # so that a method name does not force a minimum version.
        measure = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(measure())

    def ready(self) -> bool:
        """Whether this model is loaded in *this process*.

        This is deliberately not "is it in the cache". A cached model that does
        not import (the `einops` case) is not ready. A diagnostic that reported
        it as ready would answer an easier question than the one that was asked.
        """
        return self._model is not None

    def limit(self) -> Limit | None:
        """The `max_seq_length` of the model itself, or the setting when it is
        lower.

        The length of the model is the truth. `sentence-transformers` silently
        truncates every input to it, so a setting above it would only bring the
        silent truncation back. Measured on the `docs/seams` of this repository
        with the default model, which reads 256 tokens: 144 of 233 passages that
        were cut at 1,200 characters were longer, and the last eleventh of the
        text was never embedded."""
        own = getattr(self._load(), "max_seq_length", None)
        if isinstance(own, int) and own > 0:
            if self.max_tokens is not None and self.max_tokens < own:
                return Limit(tokens=self.max_tokens, source="configured")
            return Limit(tokens=own, source="model")
        return (
            None if self.max_tokens is None else Limit(tokens=self.max_tokens, source="configured")
        )

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        """Counted by the tokenizer of the model itself, with the prefix and the
        markers that it adds. "Fits" then means what the model means by it."""
        prefix = self.query_prefix if query else self.document_prefix
        counted = self._counted([f"{prefix}{text}" for text in texts], markers=True)
        return _estimated(prefix, texts, bytes_per_token) if counted is None else counted

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        """Counted by the tokenizer of the model itself: the tokens of the text
        alone."""
        counted = self._counted(texts, markers=False)
        return weigh_by_bytes(texts, bytes_per_token) if counted is None else counted

    def _counted(self, texts: Sequence[str], *, markers: bool) -> list[int] | None:
        """How many tokens the tokenizer of the model makes of each text, or
        `None` for a model that has no tokenizer to ask."""
        tokenizer = getattr(self._load(), "tokenizer", None)
        if tokenizer is None:
            return None
        if not texts:
            return []
        encoded = tokenizer(
            list(texts),
            add_special_tokens=markers,
            truncation=False,
            # Otherwise it logs a warning for every text that is over the length
            # of the model, which is the thing that is being measured.
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def cached(self) -> bool:
        """Whether the weights are already under `cache_folder`.

        This is the question that a command that runs one time for each call can
        answer. `ready` is about this process, and a process that is asked for
        its status has loaded nothing. The answer is read from the layout of the
        hub cache itself: one directory for each repository,
        `models--<owner>--<name>`. A model that is named by a path is on disk
        because the path exists. The answer says that a load will not download,
        not that it will succeed. `load` is the only check of that, for the
        reason that its docstring gives.
        """
        if Path(self.model_name).expanduser().exists():
            return True
        repository = (
            self.model_name
            if "/" in self.model_name
            else f"sentence-transformers/{self.model_name}"
        )
        folder = Path(self.cache_folder) if self.cache_folder else None
        return folder is not None and (folder / f"models--{repository.replace('/', '--')}").is_dir()

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        prefix = self.query_prefix if query else self.document_prefix
        rows = self._load().encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._spent.update(requests=1, texts=len(texts))
        return np.ascontiguousarray(rows, dtype=np.float32)

    def spent(self) -> Mapping[str, int]:
        """Calls of the model and texts embedded. A local model reports no
        tokens and costs none."""
        return self._spent


def _estimated(prefix: str, texts: Sequence[str], bytes_per_token: float) -> list[int]:
    """`estimate_tokens` of each text as it is sent, with its prefix in front."""
    return [estimate_tokens(f"{prefix}{text}", bytes_per_token) for text in texts]


def _batched(
    texts: Sequence[str],
    prefix: str,
    batch_size: int,
    ask: Callable[[list[str]], Sequence[Sequence[float]]],
) -> Vectors:
    """`texts`, each with `prefix` in front, embedded by `ask` one batch at a
    time, as one float32 row for each text.

    The rows are normalized here, and the server is not trusted to do it. The
    index scores by inner product, and not every server returns unit vectors."""
    # Heavy, and needed only where an index is built or searched.
    import numpy as np  # noqa: PLC0415

    size = max(batch_size, 1)
    rows: list[Any] = []
    for start in range(0, len(texts), size):
        rows.extend(ask([f"{prefix}{text}" for text in texts[start : start + size]]))
    vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _embeddings_body(
    model: str, inputs: Sequence[str], dimensions: int | None
) -> dict[str, JsonValue]:
    """The OpenAI `/embeddings` request. `model` is left out when it is empty,
    because a server that serves one model needs no name. `dimensions` is left
    out when it was not asked for."""
    body: dict[str, JsonValue] = {"input": list(inputs)}
    if model:
        body["model"] = model
    if dimensions is not None:
        body["dimensions"] = dimensions
    return body


def _vectors_of(reply: JsonValue, expected: int) -> tuple[tuple[float, ...], ...]:
    """The vectors of an OpenAI `/embeddings` answer, in the order of the inputs.
    They are placed by `index`, because the wire allows them to come back in any
    order.

    An answer whose count of embeddings differs from the count of inputs means
    that the server answered in another shape. It is refused with a
    `ValueError`. Matched by position, it would store each passage under the
    vector of a neighbor."""
    data = sorted(
        (as_obj(one) for one in as_seq(as_obj(reply).get("data"))),
        key=lambda one: as_int(one.get("index")),
    )
    if len(data) != expected:
        raise ValueError(
            f"the embeddings API answered {expected} inputs with {len(data)} embeddings"
        )
    return tuple(_vector(one.get("embedding")) for one in data)


def _vector(values: object) -> tuple[float, ...]:
    """The JSON array of numbers from a server, as one vector.

    A value in the array that is not a number means that the server answered in
    another shape. It is refused and not coerced: `float("nan")` from a string
    would be a vector that scores against nothing, with no report anywhere."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("the embeddings API returned an embedding that is not a list")
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("the embeddings API returned a value that is not a number")
        vector.append(float(value))
    return tuple(vector)


_PROPS = ("total_slots", "model_path", "model_alias")
"""The top-level fields of the `/props` of a llama.cpp server that are kept with
its window. They are enough to tell later which server and model answered, and
none of them is long."""


@dataclass(slots=True, kw_only=True)
class EndpointEmbedder:
    """An embedder that is somebody else's process: the OpenAI-compatible
    `/v1/embeddings` request, to a local server or a hosted API.

    There is nothing to load and nothing to download, so this is not a
    `LocalWeights`. There is no tokenizer here, so the tokens of a passage are
    estimated. Each failure is reported as a message and not as a traceback, with
    a code that a caller can route on. These are the failures:

    * The variable of the key is empty (`MISSING_CREDENTIAL`).
    * An input is too long (`CONTEXT_WINDOW_EXCEEDED`).
    * The endpoint cannot be reached.
    * The endpoint answered with something that is not embeddings.

    The first two are failures that an adapter also meets, and they use the codes
    of the adapter.

    This uses `urllib` because the work is one POST, and the standard library
    honors the proxy variables that a sandbox sets for a confined command. That
    is the only way that such a command reaches a host that its deployment
    allows.
    """

    base_url: str
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 60.0
    query_prefix: str = ""
    document_prefix: str = ""
    dimensions: int | None = None
    batch_size: int = 32
    max_tokens: int | None = None
    _props: Facts = field(default_factory=dict, init=False, repr=False)
    """What `/props` last answered, reduced to `n_ctx` and the fields in `_PROPS`."""
    _spent: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    """Requests, texts, and the total of the tokens in the `usage` of the server.
    A unit that the server never reported is absent."""

    def limit(self) -> Limit | None:
        """The setting, else what the server reports."""
        if self.max_tokens is not None:
            return Limit(tokens=self.max_tokens, source="configured")
        return self._reported()

    def asked_of(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1").rstrip("/")

    def spent(self) -> Mapping[str, int]:
        """Requests answered, texts embedded, and the tokens that the `usage` of
        the server reported, if it reported any."""
        return self._spent

    def reported(self) -> Facts:
        return dict(self._props)

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        prefix = self.query_prefix if query else self.document_prefix
        return _estimated(prefix, texts, bytes_per_token)

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        return weigh_by_bytes(texts, bytes_per_token)

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        prefix = self.query_prefix if query else self.document_prefix
        return _batched(texts, prefix, self.batch_size, self._ask)

    def _reported(self) -> Limit | None:
        """What the server reports that it reads, if it reports that.

        llama.cpp answers `/props` with its window. A hosted API has nothing to
        ask, so it needs the setting. Without the setting, a passage that the API
        refuses is cut smaller and sent again. That costs a request, which is why
        the setting is worth writing."""
        try:
            request = urllib.request.Request(
                f"{self.asked_of()}/props", headers=self._headers(body=False)
            )
            with urllib.request.urlopen(request, timeout=5.0) as response:
                answer = as_obj(loads(response.read()))
        except (OSError, ValueError, HarnessError):
            # Not a llama.cpp server, or it does not answer, or the key is
            # missing. Nothing reports a limit.
            self._props = {}
            return None
        window = as_obj(answer.get("default_generation_settings")).get("n_ctx")
        said: dict[str, JsonValue] = {"n_ctx": window}
        said |= {name: answer.get(name) for name in _PROPS}
        self._props = {
            name: value for name, value in said.items() if isinstance(value, (str, int, float))
        }
        return (
            Limit(tokens=window, source="server")
            if isinstance(window, int) and window > 0
            else None
        )

    def _headers(self, *, body: bool = True) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} if body else {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise HarnessError(
                    f"the embeddings endpoint's key is missing: ${self.api_key_env} is not "
                    "set where the command runs. Set it in the environment pH, or the "
                    "harness that runs this command, is started from; the command is "
                    "handed it by name.",
                    MISSING_CREDENTIAL,
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @property
    def name(self) -> str:
        """The model, both prefixes and a requested width: the things that move
        the vector space. **Not the address.** The same model served from another
        address is the same space, and moving a server must not cost an index. A
        hosted model that changes under an unchanged name cannot be detected
        except by its width, which the index checks on every add."""
        width = "" if self.dimensions is None else f":{self.dimensions}"
        return f"endpoint:{self.model}{width}:{self.query_prefix}|{self.document_prefix}"

    def _ask(self, inputs: list[str]) -> tuple[tuple[float, ...], ...]:
        """The vectors of one batch. `inputs` already have their prefix in front."""
        headers = self._headers()
        body = _embeddings_body(self.model, inputs, self.dimensions)
        address = f"{self.base_url.rstrip('/')}/embeddings"
        request = urllib.request.Request(
            address, data=dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                answer = loads(response.read())
        except urllib.error.HTTPError as error:
            said = capped_text(error.read(600), 600).strip()
            raise HarnessError(
                f"the embeddings endpoint {address} answered {error.code}: {said or error.reason}",
                CONTEXT_WINDOW_EXCEEDED if input_too_long(said) else ENDPOINT_FAILED,
            ) from error
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise HarnessError(
                f"the embeddings endpoint {address} could not be reached: {reason}. For a "
                "confined agent its host has to be among the hosts the deployment allows "
                "(`sandbox-allow`).",
                ENDPOINT_UNREACHABLE,
            ) from error
        except ValueError as error:
            raise HarnessError(
                f"the embeddings endpoint {address} did not answer in JSON: {error}",
                ENDPOINT_FAILED,
            ) from error
        try:
            vectors = _vectors_of(answer, len(inputs))
        except ValueError as error:
            raise HarnessError(f"{address}: {error}", ENDPOINT_FAILED) from error
        self._spent.update(requests=1, texts=len(inputs))
        usage = as_obj(as_obj(answer).get("usage"))
        used = usage.get("total_tokens", usage.get("prompt_tokens"))
        if isinstance(used, int):
            self._spent["tokens"] += used
        return vectors
