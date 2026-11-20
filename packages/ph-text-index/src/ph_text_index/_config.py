"""The configs of the `text-index` rows, in a module apart from the rows that read them.

Three pairs of rows use these three models. The `text-index`,
`text-index-local` and `text-index-endpoint` rows, which a session mounts,
validate the configs and pass them to the command. The `text-index-engine`,
`text-index-local-engine` and `text-index-endpoint-engine` rows, which the
command mounts, index and embed by them. The models are in their own module so
that the session rows can import them without importing the engine, and with the
engine turbovec and numpy.

The module also holds the facts that both processes need about where the engine
keeps things: the row ids that the configs travel under, the directory name of
the index in the scratch directory of an agent, and the shared path that the
embedder writes. The session row grants that shared path to the confined
command. Each fact has one derivation, so a grant cannot name a directory that
the engine never writes.

@module ph_text_index._config
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from ph.paths import default_cache_path, resolve_roots
from ph.seams.fs import IndexScope
from ph.wire import EnvName, WireModel

__all__ = [
    "ENDPOINT_ROW",
    "LOCAL_ROW",
    "MISSING_MODEL",
    "ROW",
    "STATE",
    "Config",
    "EndpointConfig",
    "Limit",
    "LocalConfig",
    "Scope",
    "index_base",
    "weights_cache",
]

ROW = "text-index"
"""The id of the session row. The command mounts the engine under the same id, so
one config serves both. It is also the name of the provisioning command that both
offer."""

LOCAL_ROW = "text-index-local"
"""The row id of the local embedder, in a session and in the command."""

MISSING_MODEL = (
    "the text-index-local row needs `sentence-transformers`, which is an extra: "
    "install `ph-text-index[local]` (or `phern[local]` for the whole harness), "
    "or configure another embedder row in its place"
)
"""Why the local embedder refused, and the two ways to fix it. A session reports
this text at mount, where a person reads it. The command reports it for a person
at a shell.

`sentence-transformers` is an extra (`ph-text-index[local]`) and not a
dependency, on purpose. The model is most of a gigabyte, and the seam that it
fills has other providers. A deployment that embeds through an *endpoint* must
be able to install this package and never resolve torch. The text names the
extra exactly as a person must type it."""

ENDPOINT_ROW = "text-index-endpoint"
"""The row id of the endpoint embedder, in a session and in the command."""

STATE = "text-index"
"""The directory, in the scratch directory of the calling agent, that holds the
indexes of a session: one for each embedder beneath it."""


class Limit(WireModel):
    """The longest input that an embedder reads, in its own tokens, and the source
    of that number.

    The class is here and not with the embedders because a session reports a
    limit too: a setting, in `phern doctor`. The process of a session never
    imports the module that holds the embedders. The class is a wire model
    because a limit is also kept in a file (`EmbedderRecord`)."""

    tokens: int
    source: Literal["configured", "model", "server", "refused"]
    """`refused` is a limit that was learned from a refusal: the embedder refused
    an input as too long, and the refusal said how long an input it takes."""

    def said(self) -> str:
        whose = {
            "configured": "configured",
            "model": "the model's",
            "server": "the server's",
            "refused": "what its refusal said",
        }[self.source]
        return f"{self.tokens} tokens ({whose})"


Facts = dict[str, str | int | float]
"""What the server of an embedder reported about itself, reduced to plain values."""


class Learned(WireModel):
    """What refusals showed about an embedder, in addition to what it reported
    when asked."""

    tokens: int | None = None
    """The limit that a refusal named, if one did. It is lower than the reported
    limit. A llama.cpp server reports its window and refuses an input that is
    past its batch."""
    density: float | None = None
    """The number of bytes of this corpus that make one token, if a refusal showed
    it. It is lower than the careful estimate, which happens for a corpus of
    paths, code and tables."""


class EmbedderRecord(WireModel):
    """What an embedder reported about itself when an index run asked it.

    A record is one entry of the embedder log of the index
    (`TextIndex.embedder`). The log is kept in the sidecar. It is saved with the
    passages that the record was used to cut, under the same writer lock.

    An index run that embeds asks the embedder. An answer that differs from the
    newest entry is added as a new entry. The answer differs when a server was
    moved or given another window, or when a refusal showed something new. The
    newest entry is the one in use. The earlier entries show what changed, and
    when. A search reads the newest entry and asks nothing. For an endpoint, that
    saves one request on every search, because each search is its own command."""

    asked_of: str | None = None
    """The address that was asked, for an embedder behind a server. `None` for an
    embedder that is a model in the process of the command."""
    asked_at_ms: int
    """When this answer was first given, in milliseconds since the epoch."""
    reported: Facts = Field(default_factory=dict)
    """What the embedder reported, unchanged: the `/props` of a llama.cpp server,
    reduced to the fields that are read here."""
    limit: Limit | None = None
    """The longest input that it reads, from the setting or from what it
    reported."""
    learned: Learned = Field(default_factory=Learned)

    def effective(self) -> Limit | None:
        """The limit to pack to: the limit from a refusal if there is one,
        otherwise the reported limit."""
        if self.learned.tokens is not None:
            return Limit(tokens=self.learned.tokens, source="refused")
        return self.limit


class Scope(IndexScope):
    """`IndexScope`, with prose as the default `include`, plus whether a session
    starts to index it without a request."""

    include: list[str] = Field(default_factory=lambda: ["**/*.md"])
    on_mount: bool = True
    """Start to index this scope in the background when the agent of a root
    session takes its first step. The index is then in progress before a call
    asks for it. The footer shows the progress, and the agent is told when the
    run finishes.

    `false` leaves indexing to `text_index` calls."""


class Config(WireModel):
    """The row config for `text-index`."""

    index: Scope = Field(default_factory=Scope)
    """What `text_index` looks at when the call does not say: the trees, the files
    in them, and the files to leave out (`ph.seams.fs.IndexScope`).

    `include` is Markdown by default because the prose of a repository is in
    Markdown. Also, a default of `**/*` would index a `node_modules` on the first
    call that somebody made without thought."""
    max_chars: int = 1_200
    """The largest passage, in characters. This is roughly 300 tokens.

    The value is large enough to hold an argument, and small enough that a hit is
    mostly the relevant part. The cost of a large chunk is not storage. The cost
    is that everything else in the chunk dilutes the sentence that matches."""
    overlap_chars: int = 200
    """How much of the end of a passage is repeated at the start of the next
    passage.

    See `_chunk`. The overlap allows for a paragraph whose meaning depends on the
    paragraph before it."""
    bit_width: Literal[2, 4] = 4
    """The quantization width of turbovec: 4, or 2 for half the memory and less
    recall.

    The type is a closed set, so the **profile** refuses a third value. Otherwise
    the row would mount without complaint, and `IdMapIndex` would meet an 8 at
    the first index. This is the refuse-at-mount rule of the cookbook. The
    docstring already said what the type now says."""
    calibrate: bool = True
    """Whether to fit the TQ+ calibration of turbovec when a first batch is large
    enough to sample fairly. See `_store`: the calibration is committed in one
    situation only, on purpose."""
    max_bytes: int = 2 * 1024 * 1024
    """The largest document that is indexed. A larger document is skipped and
    reported, and is not an error. A call over a directory must not fail because
    it found a minified bundle. A caller that never learns which files were
    skipped cannot tell a quiet corpus from a quiet failure."""
    save_every_s: float = 30.0
    """The time between two saves of what an indexing run has embedded so far.

    A save rewrites the whole sidecar, so a run does not save after every
    document. A run also does not save only at the end, as it did before: a run
    that was stopped at the ninth minute of ten kept nothing, and the next run
    embedded all of it again. A run that is stopped loses at most this much
    work."""
    wait_s: float = 5.0
    """How long a `text_index` call waits for a run before it answers
    "indexing". In a session, it waits for the run that the session started for
    it in the background. In the command, it waits for another run that holds the
    index.

    The value is long enough that a run with little to embed (a few documents, or
    a second run over a corpus) answers with its counts. It limits the wait for a
    run that nobody should wait through: minutes of embedding."""
    max_files: int = 5_000
    """The number of files that one `text_index` call considers.

    This is config and not a constant hidden in the walk, where it was before.
    There it had already drifted from the `max_files` of the sibling package. Two
    rows that do the same walk then disagreed about the bound, and nobody had
    chosen that."""


class LocalConfig(WireModel):
    """The row config for `text-index-local`."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """The `sentence-transformers` model. It is downloaded on first use, then
    cached.

    The default is small, symmetric and 384-dimensional. It needs no prefixes, it
    is about 90 MB, and it runs on a CPU fast enough that indexing a
    documentation tree takes minutes and not hours. A deployment that wants
    recall more than speed names a bigger model. It pays for that twice, in load
    time and in a fresh index, because vectors do not transfer between models.

    **Switching is safe to try.** The index directory is keyed by a digest of the
    identity of the embedder: the model name *and* both prefixes. A new model
    therefore gets its own index, and a switch back finds the old index intact.
    The README has a worked `nomic-embed-text-v1.5` configuration."""
    query_prefix: str = ""
    """The text that an asymmetric model needs in front of a question.

    `e5` needs `query: `, and `nomic-embed-text` needs `search_query: `. The
    prefix is part of the identity of the embedder. An index that is built with a
    prefix and searched without one is a different space, and the failure is
    quiet: slightly wrong neighbors and no error."""
    document_prefix: str = ""
    """The text that it needs in front of a passage (`e5`: `passage: `, nomic:
    `search_document: `)."""
    trust_remote_code: bool = False
    """Whether the model may bring its own modeling code. See the embedder.

    **This is a trust decision, not a compatibility flag.** `true` executes
    Python that was downloaded from the repository of the model, in the process
    of the command. `nomic-embed-text-v1.5` requires it, and so does every other
    model whose `config.json` has an `auto_map`. It is off by default, so that a
    profile grants it explicitly and a load failure does not suggest it."""
    batch_size: int = 32
    """How many passages are embedded in one forward pass."""
    max_tokens: int | None = None
    """The longest passage to embed, in the tokens of the model, for when that
    must be less than the model reads. When this is unset, the length of the
    model itself is the bound, and that is nearly always right. The default model
    reads 256 tokens, and passages are cut to fit that length, so that their ends
    are not dropped unread."""
    cache: str = ""
    """Where the weights are stored. The default is `$PH_CACHE/models`.

    The directory is under a pH root on purpose. Without this setting,
    `sentence-transformers` puts the weights in `$HF_HOME` or
    `~/.cache/huggingface`. `phern doctor` does not mention those directories,
    and `rm -rf $PH_CACHE` does not reclaim them. `$PH_CACHE` is the root for
    data that is large and can be rebuilt (Q1), and the runtime venv is there for
    the same reason.

    Every agent shares the weights, while each agent has its own index. Weights
    are the content of the model, keyed by its name, and say nothing about a
    repository."""
    preload: bool = False
    """Load the model at **mount** and not on the first call.

    It is off by default because a person at a TUI prefers that the harness
    starts now, and that the model is paid for when it is used. Turn it on for an
    **unattended** run: a daemon, a scheduled tick, or `phern -p` in CI. There,
    the alternative is to find out in the middle of a turn that the weights
    cannot be fetched, and nobody is present to type `/text-index install`.

    When the load fails, it refuses the mount (`MountRefusal`). This is the rule
    of the cookbook: refuse at mount, not at first use. At first use the agent is
    already running, and "refuse to start" has already been disobeyed. The
    session row asks the command to load the model one time, at its own mount,
    and refuses when the command refuses."""


class EndpointConfig(WireModel):
    """The row config for `text-index-endpoint`: an embedder that is somebody
    else's process, reached with the OpenAI-compatible `/v1/embeddings` request.

    One row serves a local server (llama.cpp, Ollama, vLLM) and a hosted API. No
    model is loaded here, so a deployment that embeds this way installs
    `ph-text-index` without the `[local]` extra and never resolves torch.

    **A remote endpoint receives the text of every indexed document and of every
    question.** A deployment that writes this row states that it accepts that.
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    """The API root. Requests go to `/embeddings` beneath it."""
    model: str = ""
    """The model that the endpoint is asked for. Empty for a server that serves
    one model."""
    api_key_env: Annotated[str, EnvName()] = ""
    """The **name** of the environment variable that holds the key, or empty for
    a server that needs no key.

    This is a name and never a value: nothing in a profile, a log or
    `$PH_TOOL_CONFIG` is a credential. The session row has `ph-utcp` pass that
    one variable to the command, from the environment of the harness, each time
    it starts a command. A command that finds the variable empty reports it by
    name."""
    query_prefix: str = ""
    """The text that an asymmetric model needs in front of a question
    (`search_query: `)."""
    document_prefix: str = ""
    """The text that it needs in front of a passage (`search_document: `)."""
    dimensions: int | None = None
    """Ask the endpoint for vectors of this width, if it can shorten them. The
    width is part of the identity of the embedder: another width is another
    index."""
    batch_size: int = 32
    """How many passages are sent in one request."""
    max_tokens: int | None = None
    """The longest passage that the endpoint reads, in its tokens: 8191 for the
    embedding models of OpenAI. When this is unset, a llama.cpp server is asked
    for its window (`/props`). An API that has nothing to ask has no bound that
    is known in advance, and a passage that it refuses is cut smaller and sent
    again."""
    timeout_s: float = 60.0
    """How long one request may take."""


def index_base() -> Path:
    """The directory of the per-workspace indexes, for a caller that names no
    state directory."""
    return default_cache_path(None, "text-index")


def weights_cache(config: LocalConfig) -> Path:
    """The directory for the weights of the local model: the `cache` of the row,
    or `$PH_CACHE/models`."""
    return Path(config.cache).expanduser() if config.cache else resolve_roots().cache / "models"
