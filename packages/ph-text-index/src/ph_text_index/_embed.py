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
embedder changed under it must refuse rather than return neighbors computed in
a space nothing shares.

@module ph_text_index._embed
"""

from __future__ import annotations

import base64
import logging
import math
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from ph.framing import FramingError, ask
from ph.json import JsonValue, as_int, as_obj, as_str, dumps, loads
from ph.llm.adapter import LlmError, input_too_long, openai_embeddings_body, openai_vectors
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, MISSING_CREDENTIAL
from ph.toolcli import read_envelope
from ph.tools.errors import HarnessError

from ._config import Limit

__all__ = [
    "BYTES_PER_TOKEN",
    "ENDPOINT_FAILED",
    "ENDPOINT_UNREACHABLE",
    "SESSION_GONE",
    "Embedder",
    "EndpointEmbedder",
    "LocalWeights",
    "SentenceTransformerEmbedder",
    "SessionEmbedder",
    "TokenBounded",
    "Vectors",
    "estimate_tokens",
    "weigh_by_bytes",
]

BYTES_PER_TOKEN = 3
"""What `estimate_tokens` assumes, on the careful side. English prose runs near
four characters a token and code nearer three; a script that is three bytes a
character is about one token a character. So bytes over three is seldom under."""

ENDPOINT_UNREACHABLE = "TEXT_INDEX_ENDPOINT_UNREACHABLE"
ENDPOINT_FAILED = "TEXT_INDEX_ENDPOINT_FAILED"
SESSION_GONE = "TEXT_INDEX_SESSION_GONE"

log = logging.getLogger("ph_text_index.embed")


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Vectors: TypeAlias = "NDArray[np.float32]"
"""One L2-normalized float32 row per text — `(len(texts), dim)`.

The shape the whole chain moves: `Embedder.encode` produces it, `TextIndexSeam.embed`
carries it across the worker-thread hop, and `TextIndex.add`/`_maybe_calibrate`/`search`
consume it. Named rather than repeated so the stub embedder a test registers is held
to what the store indexes.

Imported under `TYPE_CHECKING` alone: numpy is loaded lazily inside the functions
that need it, and an annotation must not be what drags it into import time.
"""


def estimate_tokens(text: str, bytes_per_token: float) -> int:
    """About how many tokens `text` is, for an embedder with no tokenizer to ask:
    its bytes over `bytes_per_token` — `BYTES_PER_TOKEN`, or the corpus's density
    once a refusal said it — and two for the markers a model adds."""
    return math.ceil(weigh_by_bytes([text], bytes_per_token)[0]) + 2


def weigh_by_bytes(texts: Sequence[str], bytes_per_token: float) -> list[float]:
    """About how many tokens each text adds to a passage, where there is no
    tokenizer to ask: its bytes over `bytes_per_token`, unrounded — so the
    weights of a passage's parts add up to the passage's, and `estimate_tokens`
    of the whole is never over what they said."""
    return [len(text.encode("utf-8")) / bytes_per_token for text in texts]


@runtime_checkable
class TokenBounded(Protocol):
    """An embedder that can say how long an input it reads. **Optional
    capability**, its own Protocol for `LocalWeights`' reason.

    What makes a passage's size the embedder's business rather than a number in
    a profile: a model reads so many tokens and no more. Past that a local model
    drops the rest without a word — the tail of the passage is indexed and can
    never be found — and a server refuses the request. `limit` is where the
    bound comes from, `weigh` is what a passage is packed by and `measure` is
    how what is sent is checked against it.
    """

    max_tokens: int | None
    """The setting, when a profile wrote one. Read without loading anything, so
    a run can tell that the setting changed before it needs the model."""

    def limit(self) -> Limit | None:
        """The bound, or `None` when nothing says: the setting, else what the
        model or its server reports. May load a model or ask a server."""

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        """How many tokens each of `texts` is as it is sent — as a document, or
        as a question — prefix and markers included: counted where there is a
        tokenizer, estimated at `bytes_per_token` — the corpus's density, as far
        as it is known — where not."""

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        """How many tokens each of `texts` adds to a passage it is part of: its
        own, with no prefix and no markers. What passages are packed by
        (`_chunk.Weight`), so it is asked of a paragraph and of a line."""


@runtime_checkable
class Embedder(Protocol):
    """Text in, one row of floats out."""

    @property
    def name(self) -> str:
        """A stable identity for the vector space this produces.

        Stored in the index and compared on open, so it has to change whenever
        the space does — the model, and anything else that moves a vector.
        """

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        """`(len(texts), dim)` float32, L2-normalized.

        **Blocking**; the seam calls it in a worker thread. Normalized because
        turbovec scores an inner product, and an inner product over normalized
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

    def cached(self) -> bool:
        """Whether its weights are on disk, so a load fetches nothing."""

    def load(self) -> int:
        """Fetch and load it now, and return its dimension."""


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """A local `sentence-transformers` model, loaded on first use.

    **Not at mount.** The first load downloads weights, and a harness that
    stalled for a few hundred megabytes before it could answer `phern doctor` would
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
    somewhere `phern doctor` never mentions and deleting `$PH_CACHE` would not
    reclaim. Rebuildable and large is exactly the lifecycle `$PH_CACHE` names
    (Q1), and it is where the runtime venv lives for the same reason.

    Through the constructor argument rather than by setting `HF_HOME`: a row
    that mutated the process environment would change it for every child
    `ctx.subprocess` spawns as well, and this is a decision about one library.
    """
    max_tokens: int | None = None
    """A bound on an input, under the model's own. See `limit`."""
    trust_remote_code: bool = False
    """Whether to let this model bring its own modeling code.

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
        different space, and the failure is quiet — slightly wrong neighbors,
        which is worse than an error.
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

            # **Weights already on disk are loaded without asking the hub.** Left
            # to itself the library checks the repository for a newer revision on
            # every load, so a model that is entirely cached still needs the
            # network — and a confined command has only the hosts its deployment
            # allows, which answered that check with `403 Forbidden` and failed a
            # search over an index that needed nothing fetched. It also means a
            # load cannot quietly become a download.
            #
            # An `OSError` there is a cache that is not whole — an interrupted
            # download leaves the directory — and the hub can repair it, so that
            # one case is tried again with the network. Anything else is the
            # model's own failure and is raised as it is.
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

        **The whole point is that it loads rather than downloads.** Fetching the
        weights is not the same check: `nomic-embed-text-v1.5` downloads its
        config and its remote modeling code successfully and then fails at
        import with

            ImportError: This modeling file requires the following packages
            that were not found in your environment: einops

        — a missing transitive dependency that only the model's own code knows
        about, and that no amount of pre-downloading would have surfaced. So the
        provisioning path calls this, and whatever it raises is what the person
        reads.
        """
        model = self._load()
        # `get_sentence_embedding_dimension` is deprecated in favor of
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

    def limit(self) -> Limit | None:
        """The model's own `max_seq_length`, or the setting when it is lower.

        The model's is the truth: `sentence-transformers` truncates every input
        to it, silently, so a setting above it would only put the silence back.
        Measured on this repository's `docs/seams` with the default model, which
        reads 256 tokens: 144 of 233 passages cut at 1,200 characters were
        longer, and the last eleventh of the text was never embedded."""
        own = getattr(self._load(), "max_seq_length", None)
        if isinstance(own, int) and own > 0:
            if self.max_tokens is not None and self.max_tokens < own:
                return Limit(self.max_tokens, "configured")
            return Limit(own, "model")
        return None if self.max_tokens is None else Limit(self.max_tokens, "configured")

    def measure(
        self, texts: Sequence[str], bytes_per_token: float, *, query: bool = False
    ) -> list[int]:
        """Counted by the model's own tokenizer, with the prefix and the markers
        it adds — so "fits" means what the model means by it."""
        prefix = self.query_prefix if query else self.document_prefix
        counted = self._counted([f"{prefix}{text}" for text in texts], markers=True)
        return _estimated(prefix, texts, bytes_per_token) if counted is None else counted

    def weigh(self, texts: Sequence[str], bytes_per_token: float) -> Sequence[float]:
        """Counted by the model's own tokenizer: the text's own tokens."""
        counted = self._counted(texts, markers=False)
        return weigh_by_bytes(texts, bytes_per_token) if counted is None else counted

    def _counted(self, texts: Sequence[str], *, markers: bool) -> list[int] | None:
        """How many tokens the model's tokenizer makes of each text, or `None`
        for a model that has none to ask."""
        tokenizer = getattr(self._load(), "tokenizer", None)
        if tokenizer is None:
            return None
        if not texts:
            return []
        encoded = tokenizer(
            list(texts),
            add_special_tokens=markers,
            truncation=False,
            # Or it logs a warning for every text over the model's length, which
            # is the thing being measured.
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def cached(self) -> bool:
        """Whether the weights are under `cache_folder` already.

        The question a command that runs once per call can answer: `ready` is
        about this process, and a process asked for its status has loaded
        nothing. Read from the hub cache's own layout — a directory per
        repository, `models--<owner>--<name>` — and a model named by a path is
        on disk by being one. It says a load will not download, not that it will
        succeed: `load` is the only check of that, for the reason it gives.
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
        return np.ascontiguousarray(rows, dtype=np.float32)


def _estimated(prefix: str, texts: Sequence[str], bytes_per_token: float) -> list[int]:
    """`estimate_tokens` of each text as it is sent: behind its prefix."""
    return [estimate_tokens(f"{prefix}{text}", bytes_per_token) for text in texts]


def _batched(
    texts: Sequence[str],
    prefix: str,
    batch_size: int,
    ask: Callable[[list[str]], Sequence[Sequence[float]] | Vectors],
) -> Vectors:
    """`texts`, behind `prefix`, embedded by `ask` a batch at a time, as one
    float32 row each.

    Normalized here rather than trusted: the index scores by inner product, and
    not every server returns unit vectors."""
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


@dataclass(slots=True, kw_only=True)
class _Remote(ABC):
    """What the two embedders that are another process share: no tokenizer
    here, so a passage is estimated; texts sent a batch at a time behind their
    prefix; and a limit that is the setting, else what whoever embeds reports.
    Each says only how it asks (`_ask`, `_reported`) and what it is called."""

    query_prefix: str = ""
    document_prefix: str = ""
    dimensions: int | None = None
    batch_size: int = 32
    max_tokens: int | None = None

    def limit(self) -> Limit | None:
        if self.max_tokens is not None:
            return Limit(self.max_tokens, "configured")
        return self._reported()

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

    def _space(self, kind: str, model: str) -> str:
        """A `name`: the model, both prefixes and an asked-for width."""
        width = "" if self.dimensions is None else f":{self.dimensions}"
        return f"{kind}:{model}{width}:{self.query_prefix}|{self.document_prefix}"

    @abstractmethod
    def _reported(self) -> Limit | None:
        """The limit whoever embeds reports, or `None` when it says nothing."""

    @abstractmethod
    def _ask(self, inputs: list[str]) -> Sequence[Sequence[float]] | Vectors:
        """One batch's vectors, `inputs` already behind their prefix."""


@dataclass(slots=True, kw_only=True)
class EndpointEmbedder(_Remote):
    """An embedder that is somebody else's process: the OpenAI-compatible
    `/v1/embeddings` request, to a local server or a hosted API.

    Nothing to load and nothing to download, so it is not a `LocalWeights`. What
    can go wrong is said rather than a traceback, each with a code a caller can
    route on — and the two an adapter would also meet are the adapter's codes:
    the key's variable is empty (`MISSING_CREDENTIAL`), an input is too long
    (`CONTEXT_WINDOW_EXCEEDED`); and the endpoint cannot be reached, or answered
    with something that is not embeddings.

    `urllib`, because this is one POST and the standard library honors the proxy
    variables a sandbox sets for a confined command — which is the only way such
    a command reaches a host its deployment allows.
    """

    base_url: str
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 60.0

    def _reported(self) -> Limit | None:
        """What the server says it reads, where it says.

        llama.cpp answers `/props` with its window. A hosted API has nothing to
        ask, so it needs the setting — and without one a passage the API refuses
        is cut smaller and sent again, which costs a request and is why the
        setting is worth writing."""
        root = self.base_url.rstrip("/").removesuffix("/v1").rstrip("/")
        try:
            request = urllib.request.Request(f"{root}/props", headers=self._headers(body=False))
            with urllib.request.urlopen(request, timeout=5.0) as response:
                answer = loads(response.read())
        except (OSError, ValueError, HarnessError):
            # Not a llama.cpp, or not answering, or no key: nothing says.
            return None
        window = as_obj(as_obj(answer).get("default_generation_settings")).get("n_ctx")
        return Limit(window, "server") if isinstance(window, int) and window > 0 else None

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
        """The model, both prefixes and an asked-for width — what moves the
        vector space. **Not the address**: the same model served somewhere else
        is the same space, and moving a server must not cost an index. A hosted
        model that changes under an unchanged name cannot be told apart beyond
        its width, which the index checks on every add."""
        return self._space("endpoint", self.model)

    def _ask(self, inputs: list[str]) -> tuple[tuple[float, ...], ...]:
        headers = self._headers()
        body = openai_embeddings_body(self.model, inputs, self.dimensions)
        address = f"{self.base_url.rstrip('/')}/embeddings"
        request = urllib.request.Request(
            address, data=dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                answer = loads(response.read())
        except urllib.error.HTTPError as error:
            said = error.read(600).decode("utf-8", errors="replace").strip()
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
            return openai_vectors(answer, len(inputs))
        except LlmError as error:
            raise HarnessError(f"{address}: {error}", ENDPOINT_FAILED) from error


@dataclass(slots=True, kw_only=True)
class SessionEmbedder(_Remote):
    """An embedder that is the pH session this command is run for.

    The command sends its texts down a unix socket and the session embeds them
    through one of its own model routes. What that buys is where the credential
    is: in the harness, resolved by the adapter at its edge, and never in this
    process — which runs inside an agent's sandbox, where everything it holds is
    something the agent's own tools could read.

    One request a connection, a frame each way (`ph.framing`), answered in
    `ph.toolcli`'s envelope. A failure carries the adapter's own code and
    sentence, so a missing key reads here as it would on a model call: by the
    variable's name. Vectors travel as the bytes of a float32 array, base64 —
    a quarter of the size of a JSON list of them, and nothing to parse.
    """

    where: str
    """The session's socket."""
    route: str
    """The session's provider and model, `provider/model`."""
    agent: str | None = None
    """The agent this command works for (`ph.toolcli.AGENT_ENV`), told to the
    session with each request so what the route spends is in that agent's log.
    `None` where nobody said, and then the spend is in no log."""

    @property
    def name(self) -> str:
        """The route, both prefixes and an asked-for width: what moves the vector
        space, as `EndpointEmbedder.name` says it."""
        return self._space("adapter", self.route)

    def _reported(self) -> Limit | None:
        """The window the session's route reports.

        A route's window is its server's, which for a llama.cpp is the model
        being served. For a hosted provider it is the chat model's and far above
        an embedding model's, so it bounds nothing there: write the setting."""
        try:
            window = as_obj(self._request({"limit": True})).get("limit")
        except HarnessError:
            return None
        return Limit(window, "route") if isinstance(window, int) and window > 0 else None

    def _request(self, asked: dict[str, Any]) -> JsonValue:
        """One request of the session, and the value it answered with."""
        try:
            answer = ask(self.where, asked)
        except (OSError, FramingError) as error:
            raise HarnessError(
                f"the session that embeds for this index is not answering at {self.where} "
                f"({error}); it has ended, or this command was not started by it",
                SESSION_GONE,
            ) from error
        try:
            value, _ = read_envelope(answer)
        except ValueError as error:
            raise HarnessError(
                f"the session answered with something that is not an envelope: {error}",
                ENDPOINT_FAILED,
            ) from error
        return value

    def _ask(self, inputs: list[str]) -> Vectors:
        """The session's answer, read back: the bytes of one float32 array, as
        `row._Embedding` packs them."""
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        answered = as_obj(self._request({"inputs": inputs, "agent": self.agent}))
        packed = base64.b64decode(as_str(answered.get("vectors")))
        rows = np.frombuffer(packed, dtype=np.float32)
        width = as_int(answered.get("dim"))
        if width <= 0 or rows.size != width * len(inputs):
            raise HarnessError(
                f"the session answered {len(inputs)} texts with {rows.size} numbers of "
                f"{width} a vector",
                ENDPOINT_FAILED,
            )
        return rows.reshape(len(inputs), width)
