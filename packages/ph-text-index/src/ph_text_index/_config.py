"""The `text-index` rows' configs, apart from the rows that read them.

Two pairs of rows take these two models. The `text-index` and `text-index-local`
rows a session mounts validate them and hand them across to the command; the
`text-index-engine` and `text-index-local-engine` rows the command mounts index
and embed by them. In a module of its own so the session's rows can import the
models without importing the engine — and with it turbovec and numpy.

So are the facts both processes need about where the engine keeps things: the
row ids the configs travel under, the directory name the index takes in an
agent's scratch, and the shared path the embedder writes, which the session's
row grants to the confined command. One derivation each, so a grant cannot name
a directory the engine never writes.

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
"""The session row's id, the id the command mounts the engine under — so one
config serves both — and the name of the provisioning command both offer."""

LOCAL_ROW = "text-index-local"
"""The local embedder's row id, in a session and in the command alike."""

MISSING_MODEL = (
    "the text-index-local row needs `sentence-transformers`, which is an extra: "
    "install `ph-text-index[local]` (or `phern[local]` for the whole harness), "
    "or configure another embedder row in its place"
)
"""Why the local embedder refused, and the two ways out — said at a session's
mount, where a person reads it, and by the command for a person at a shell.

The extra rather than a dependency is the point (`ph-text-index[local]`): the
model is most of a gigabyte and the seam it fills has other providers, so a
deployment that embeds through an *endpoint* should be able to install this
package and never resolve torch. It names the extra exactly as a person would
have to type it."""

ENDPOINT_ROW = "text-index-endpoint"
"""The endpoint embedder's row id, in a session and in the command alike."""

STATE = "text-index"
"""The directory, in the calling agent's scratch, a session's indexes are kept
in: one per embedder beneath it."""


class Limit(WireModel):
    """How long an input an embedder reads, in its own tokens, and who said so.

    Here and not with the embedders, because a session says it too — a setting,
    in `phern doctor` — and a session's process never imports the module the
    embedders are in. A wire model, because it is kept in a file too
    (`EmbedderRecord`)."""

    tokens: int
    source: Literal["configured", "model", "server", "refused"]
    """`refused` is one learned the hard way: the embedder turned an input away
    as too long, and said in doing so how long an input it takes."""

    def said(self) -> str:
        whose = {
            "configured": "configured",
            "model": "the model's",
            "server": "the server's",
            "refused": "what its refusal said",
        }[self.source]
        return f"{self.tokens} tokens ({whose})"


Facts = dict[str, str | int | float]
"""What an embedder's server said of itself, down to plain values."""


class Learned(WireModel):
    """What refusals taught about an embedder, beyond what it said when asked."""

    tokens: int | None = None
    """The limit a refusal named, where one did: lower than the one reported.
    A llama.cpp reports its window and refuses past its batch."""
    density: float | None = None
    """How many bytes of this corpus make a token, where a refusal said: lower
    than the careful guess, for a corpus of paths, code and tables."""


class EmbedderRecord(WireModel):
    """What an embedder said of itself when an index run asked it: one entry of
    the index's embedder log (`TextIndex.embedder`), kept in the sidecar and
    saved with the passages it was used to cut, under the same writer lock.

    An index run that embeds asks, and an answer that differs from the newest
    entry — a server moved or given another window, or a refusal that taught
    something — is added as a new entry. The newest is the one used; the ones
    before it say what changed, and when. A search reads the newest and asks
    nothing: for an endpoint that is a request saved on every search, since
    each is a command of its own."""

    asked_of: str | None = None
    """The address that was asked, for an embedder behind a server; `None` for
    one that is a model in the command's own process."""
    asked_at_ms: int
    """When this answer was first given, in milliseconds since the epoch."""
    reported: Facts = Field(default_factory=dict)
    """What it said, as it said it: a llama.cpp's `/props`, down to the fields
    read here."""
    limit: Limit | None = None
    """How long an input it reads, by the setting or by what it reported."""
    learned: Learned = Field(default_factory=Learned)

    def effective(self) -> Limit | None:
        """The limit to pack to: what a refusal taught, over what was reported."""
        if self.learned.tokens is not None:
            return Limit(tokens=self.learned.tokens, source="refused")
        return self.limit


class Scope(IndexScope):
    """`IndexScope`, with prose as what is taken when nobody says, and whether a
    session starts on it unasked."""

    include: list[str] = Field(default_factory=lambda: ["**/*.md"])
    on_mount: bool = True
    """Start indexing this scope when a root session's agent takes its first
    step, in the background, so the index is being built before anything asks
    for it. The footer says how far it has got and the agent is told when it
    finishes.

    `false` leaves indexing to `text_index` calls."""


class Config(WireModel):
    """Row config for `text-index`."""

    index: Scope = Field(default_factory=Scope)
    """What `text_index` looks at when the call does not say: which trees, which
    files in them, which to leave out (`ph.seams.fs.IndexScope`).

    `include` is Markdown by default because that is what a repository's prose is
    in, and because a default of `**/*` would index a `node_modules` on the first
    call somebody made without thinking."""
    max_chars: int = 1_200
    """The largest passage, in characters. Roughly 300 tokens.

    Big enough to hold an argument, small enough that a hit is mostly the
    relevant part — the cost of a large chunk is not storage but that the
    matching sentence is diluted by everything else in it."""
    overlap_chars: int = 200
    """How much of a passage's tail is repeated at the head of the next.

    See `_chunk`: the concession to a paragraph whose meaning depends on the one
    before it."""
    bit_width: Literal[2, 4] = 4
    """turbovec's quantization width: 4, or 2 for half the memory and less recall.

    A closed set, so the **profile** refuses a third value rather than the row
    mounting happily and `IdMapIndex` meeting an 8 at the first index — which is
    the cookbook's refuse-at-mount rule, and the docstring already said what the
    type now says."""
    calibrate: bool = True
    """Whether to fit turbovec's TQ+ calibration when a first batch is big enough
    to sample fairly. See `_store` — it is committed in one situation on purpose."""
    max_bytes: int = 2 * 1024 * 1024
    """The largest document indexed. One past it is skipped and reported, not an
    error: a call over a directory should not fail because it found a minified
    bundle, and a caller that never learns which files were skipped cannot tell a
    quiet corpus from a quiet failure."""
    save_every_s: float = 30.0
    """How long an indexing run goes between saves of what it has embedded so far.

    A save rewrites the sidecar whole, so not after every document; and not only
    at the end, which is what it was: a run stopped at the ninth minute of ten
    kept nothing, and the next one embedded all of it again. A run that is
    stopped loses at most this much."""
    wait_s: float = 5.0
    """How long a `text_index` call waits on a run before it answers "indexing":
    on the run a session started for it in the background, and, in the command,
    on another run that holds the index.

    Long enough that a run with little to embed — a few documents, a second run
    over a corpus — answers with its counts. What it bounds is the run nobody
    should sit through: minutes of embedding."""
    max_files: int = 5_000
    """How many files one `text_index` call will consider.

    Config rather than a constant hidden in the walk, which is where it was —
    and where it had already drifted from the sibling package's `max_files`, so
    two rows doing the same walk disagreed about the bound for no reason anybody
    chose."""


class LocalConfig(WireModel):
    """Row config for `text-index-local`."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """The `sentence-transformers` model. Downloaded on first use, then cached.

    A small, symmetric, 384-dimensional default: it needs no prefixes, it is
    about 90 MB, and it runs on a CPU at a speed that makes indexing a
    documentation tree a coffee break rather than an afternoon. A deployment
    wanting recall over speed names a bigger one — and pays for it twice, in
    load time and in a fresh index, because vectors do not transfer between
    models.

    **Switching is safe to try.** The index directory is keyed by a digest of
    the embedder's identity — the model name *and* both prefixes — so a new
    model gets its own index and switching back finds the old one intact. The
    README has a worked `nomic-embed-text-v1.5` configuration."""
    query_prefix: str = ""
    """What an asymmetric model wants in front of a question.

    `e5` wants `query: `; `nomic-embed-text` wants `search_query: `. Part of the
    embedder's identity, because an index built with a prefix and searched
    without one is a different space and the failure is quiet — slightly wrong
    neighbors rather than an error."""
    document_prefix: str = ""
    """What it wants in front of a passage (`e5`: `passage: `, nomic:
    `search_document: `)."""
    trust_remote_code: bool = False
    """Whether the model may bring its own modeling code — see the embedder.

    **A trust decision, not a compatibility flag**: `true` executes Python
    downloaded from the model's repository in the command's process. Required by
    `nomic-embed-text-v1.5` and every other model whose `config.json` carries an
    `auto_map`, and off by default so that granting it is something a profile
    says rather than something a load failure suggests."""
    batch_size: int = 32
    """How many passages are embedded per forward pass."""
    max_tokens: int | None = None
    """The longest passage to embed, in the model's tokens, when that should be
    less than the model reads. Left unset the model's own length is the bound,
    and that is nearly always right: the default model reads 256 tokens, and
    passages are cut to fit it rather than have their tails dropped unread."""
    cache: str = ""
    """Where the weights land. Defaults to `$PH_CACHE/models`.

    Under a pH root on purpose: left to `sentence-transformers` the weights go
    to `$HF_HOME` or `~/.cache/huggingface`, which no `phern doctor` mentions and
    no `rm -rf $PH_CACHE` reclaims. Rebuildable and large is the lifecycle
    `$PH_CACHE` names (Q1), and the runtime venv is there for the same reason.

    Shared by every agent, where an index is each agent's own: weights are the
    model's content, keyed by its name, and say nothing of a repository."""
    preload: bool = False
    """Load the model at **mount** rather than on the first call.

    Off by default because a person at a TUI would rather the harness start now
    and pay for the model when they use it. On for an **unattended** run — a
    daemon, a scheduled tick, `phern -p` in CI — where the alternative is finding
    out mid-turn that the weights cannot be fetched, and where nobody is present
    to type `/text-index install`.

    When it fails it refuses the mount (`MountRefusal`), which is the cookbook's
    own rule: refuse at mount, not at first use, because by then the agent is
    running and "refuse to start" has already been disobeyed. A session's row
    asks the command to load the model once, at its own mount, and refuses on
    the command's refusal."""


class EndpointConfig(WireModel):
    """Row config for `text-index-endpoint`: an embedder that is somebody else's
    process, reached over the OpenAI-compatible `/v1/embeddings` request.

    One row for a local server — llama.cpp, Ollama, vLLM — and for a hosted API.
    No model is loaded here, so a deployment that embeds this way installs
    `ph-text-index` without the `[local]` extra and never resolves torch.

    **A remote endpoint is sent the text of every document indexed, and of every
    question asked.** Writing this row is how a deployment says it means that.
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    """The API root; `/embeddings` is asked beneath it."""
    model: str = ""
    """The model the endpoint is asked for. Empty for a server that serves one."""
    api_key_env: Annotated[str, EnvName()] = ""
    """The **name** of the environment variable that holds the key, or empty for
    a server that wants none.

    A name and never a value: nothing in a profile, a log or `$PH_TOOL_CONFIG`
    is a credential. The session's row has `ph-utcp` hand that one variable to
    the command from the harness's own environment each time it starts one, and
    a command that finds it empty says so by name."""
    query_prefix: str = ""
    """What an asymmetric model wants in front of a question (`search_query: `)."""
    document_prefix: str = ""
    """What it wants in front of a passage (`search_document: `)."""
    dimensions: int | None = None
    """Ask the endpoint for vectors of this width, where it can shorten them.
    Part of the embedder's identity: another width is another index."""
    batch_size: int = 32
    """How many passages are sent in one request."""
    max_tokens: int | None = None
    """The longest passage the endpoint reads, in its tokens: 8191 for OpenAI's
    embedding models. Left unset a llama.cpp server is asked for its window
    (`/props`); an API with nothing to ask has no bound known ahead of time, and
    a passage it refuses is cut smaller and sent again."""
    timeout_s: float = 60.0
    """How long one request may take."""


def index_base() -> Path:
    """Where the per-workspace indexes live, for a caller that names no state
    directory."""
    return default_cache_path(None, "text-index")


def weights_cache(config: LocalConfig) -> Path:
    """Where the local model's weights land: the row's `cache`, or `$PH_CACHE/models`."""
    return Path(config.cache).expanduser() if config.cache else resolve_roots().cache / "models"
