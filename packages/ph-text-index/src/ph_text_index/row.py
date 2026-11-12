"""`text-index` and `text-index-local` — the rows a session mounts: they register
the package's manual.

Two tools. `text_index` cuts documents into passages, embeds them and puts them
in an index on disk. `text_search` answers a question with the passages
themselves plus the `path:start-end` each came from.

## The rows a session mounts register a manual

The tools run in the `ph-text-index` command, not in the session's process. These
rows add the package's UTCP manual to `ctx.utcp`, and `ph-utcp` registers what
the manual names — under the tools' own names, with the values, the rendered
text and the error codes they have always had. The bodies are in
`ph_text_index.engine`, which the command mounts.

What that arrangement buys:

* **Any UTCP harness can use the same tools**, by registering the same manual
  (`ph-text-index manual --registration`).
* **The tool sees what its agent sees.** The command is started by `ctx.shell`
  for the agent that called it: in that agent's workspace, under its sandbox. A
  child working in its own worktree indexes that worktree.
* **The harness process never loads the model.** `sentence-transformers`, torch
  and turbovec live in the command's process — several hundred megabytes a
  session that never searches does not hold.

What it costs is stated in `ph-utcp`'s README and worth repeating here:
`permissions-fs` governs calls through `ctx.fs`, and a command is not one. These
tools' reads are bounded by the sandbox, as `bash`'s are.

## Why the manual waits for an embedder

`text-index` publishes a seam, `ctx.text_index`; an embedder's row — here
`text-index-local` — claims it with a `Provider`: which row of which command
makes the vectors, and that row's config. The manual is registered **by the
claim**, on the provider's own scope. So a profile that mounts `text-index` and
no embedder advertises nothing, and unmounting the embedder takes the tools and
the skill with it — the rule the engine keeps for the same reason (its module
docstring), one process up.

It is the claim that carries the config across, too. The command reads both
rows' settings from `$PH_TOOL_CONFIG`, and being named there is how it knows
which embedder row to mount (`ph.toolcli.ToolCli.providers`). So choosing the
model, or another embedder altogether, is done where it always was: in the
session's profile.

## What these rows still do themselves

The things a manual has no word for. They name where an index goes: a directory
of the calling agent's scratch (`Manual.state`), which is writable under every
sandbox mode — so an agent that may not touch a tree can still index it, and
cannot write the tree doing so — and is the same directory for the same agent
after a restart, so a resumed session embeds only what changed. They grant the
command what agents share: the model's weights and its log, both under
`$PH_CACHE`. And they offer the skill, the `/text-index` command and the `phern
doctor` section, as when the tools were registered here.

## Its own module

Not the package's `__init__`, which `python -m ph_text_index` imports before it
runs a single call: these rows import `ph-utcp` and with it the UTCP library, and
every tool call would pay to load what only the session uses.

@module ph_text_index.row
"""

from __future__ import annotations

import base64
import importlib.util
import logging
import secrets
import shlex
from array import array
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel

from ph.cordis import Context, Disposer, MountRefusal, Running, ServiceKey, plugin
from ph.framing import answer_one
from ph.json import as_seq, as_str
from ph.keys import AGENTS, COMMANDS, FS, LLM, SHELL
from ph.llm.adapter import EmbeddingRequest, LlmError
from ph.seams._registry import claim_slot, contribute_item
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import offer_skills as offer_package_skills
from ph.session import Session
from ph.toolcli import (
    ToolCli,
    config_env,
    failure_envelope,
    manual_call_template,
    value_envelope,
)
from ph_utcp import UTCP, Host, Manual, ManualState, ToolTraits, ToolView, command_said

from ._config import (
    ADAPTER_ROW,
    EMBEDDER_ENV,
    ENDPOINT_ROW,
    LOCAL_ROW,
    MISSING_MODEL,
    ROW,
    STATE,
    AdapterConfig,
    Config,
    EndpointConfig,
    Limit,
    LocalConfig,
    weights_cache,
)
from .cli import CLI

__all__ = [
    "TEXT_INDEX",
    "AdapterConfig",
    "Config",
    "EndpointConfig",
    "LocalConfig",
    "Provider",
    "TextIndexRow",
    "adapter",
    "apply",
    "endpoint",
    "local",
]

log = logging.getLogger("ph_text_index")

PROVISION_TIMEOUT_MS = 30 * 60 * 1000
"""How long `/text-index install` may run: a model's download, and its first load."""

INDEX_TIMEOUT_MS = 30 * 60 * 1000
"""How long one `text_index` call may run. Through the host it answers within
seconds and the run goes on in the background; this is for the call that finds
no host and does the work itself — this repository's `docs/` is 20 seconds
under MiniLM and three minutes under nomic."""

SEARCH_TIMEOUT_MS = 5 * 60 * 1000
"""How long one `text_search` call may run: the first one a host answers waits
for the model to load, and a cold load is tens of seconds."""


@dataclass(frozen=True, slots=True)
class Provider:
    """An embedder, as a session knows one: which row of which command makes the
    vectors, and what the session has to grant it."""

    row: str
    """The id the command mounts the embedder's row under, and the key its config
    travels under in `$PH_TOOL_CONFIG`."""
    config: BaseModel
    cli: ToolCli
    """The command that mounts that row. The package's own, for a row the package
    ships."""
    label: str
    """What `phern doctor` calls it: the model."""
    writable: tuple[str, ...] = ()
    """Shared paths it writes outside the workspace: a model's weights."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Variables set on the command: where to ask a session for vectors."""
    max_tokens: int | None = None
    """The limit a profile set on the embedder: how long an input it reads."""
    reads: Callable[[], Limit | None] | None = None
    """How long an input the embedder reads with no setting, where a session can
    say without starting the command: a route's window."""
    report: Callable[[], list[tuple[str, str]]] | None = None
    """More rows for `phern doctor`, read live: what the session has embedded."""

    def limit(self) -> Limit | None:
        """How long an input the embedder reads, as far as a session knows."""
        if self.max_tokens is not None:
            return Limit(self.max_tokens, "configured")
        return None if self.reads is None else self.reads()


TEXT_INDEX: ServiceKey[TextIndexRow] = ServiceKey("text_index")
"""The session's text index: the door an embedder's row claims."""


@dataclass(slots=True)
class TextIndexRow:
    """The service published as `ctx.text_index` in a session."""

    ctx: Context
    config: Config
    provider: Provider | None = None
    provider_by: Running | None = None
    manual: Manual | None = None
    """The manual registered for the provider there is."""
    state: ManualState | None = None
    """What `ph-utcp` made of it."""

    async def register(self, provider: Provider, *, scope: Context | None = None) -> ManualState:
        """Claim the embedder slot, and register the manual it makes possible.

        On the *provider's* scope, so unmounting the embedder's row takes the
        tools, the skill and the claim with it. A second claim is refused: one
        index is one vector space.
        """
        owner = scope if scope is not None else self.ctx
        release = claim_slot(
            self.ctx.running_for(scope), self, "provider", provider, label="text_index_embedder"
        )
        own = manual(self.config, provider)

        def claimed() -> Disposer:
            def unclaim() -> None:
                release()
                if self.manual is own:
                    self.manual = self.state = None

            return unclaim

        await owner.effect(claimed, label=f"text-index-embedder({provider.row})")
        state = await self.ctx.require(UTCP).add(own, scope=owner)
        self.manual, self.state = own, state
        if state.available:
            # The skill follows the tools. Offered without them, its catalog line
            # would ride every prompt naming two tools the session does not have.
            offer_skills(owner)
        return state

    def report(self) -> list[tuple[str, str]]:
        """`phern doctor`'s section: what these rows know without starting the
        command. Counts and whether the model loads are the command's to say."""
        provider = self.provider
        if provider is None or self.state is None:
            return [("embedder", "none registered — text_index is not offered")]
        limit = provider.limit()
        rows = [
            ("tools", self.state.said()),
            ("command", shlex.join(provider.cli.invocation())),
            ("embedder", provider.label),
            (
                "limit",
                # A model's own length, or a server's, is the command's to ask:
                # of a local model it means loading it.
                limit.said()
                if limit is not None
                else f"asked of the embedder when it embeds; `/{ROW} status` shows it",
            ),
            ("index", f"each agent's own, in its scratch ({STATE}/)"),
        ]
        rows.extend(("shared", path) for path in provider.writable)
        if provider.report is not None:
            rows.extend(provider.report())
        rows.append(("details", f"/{ROW} status"))
        return rows


def manual(config: Config, provider: Provider) -> Manual:
    """This package's manual for one embedder, and what `ph-utcp` needs to run it."""
    cli = provider.cli
    handed, passed = config_env({ROW: config, provider.row: provider.config})
    return Manual(
        template=manual_call_template(cli),
        # The only source of these two names, so they keep them: the skill, the
        # prompts and every program a model has written say `text_search`.
        names="bare",
        envelope=True,
        # The vectors, the sidecar and the lock file that orders their writers:
        # each agent's own, in its scratch.
        state=STATE,
        writable=[
            *provider.writable,
            # The command logs to a file, because under UTCP its stderr is discarded.
            str(cli.log_file().parent),
        ],
        # Both rows' configs, each under its row id — which is also how the
        # command learns which embedder row to mount — and the embedder's own
        # variables; and by name, what the configs mark (an endpoint's key).
        env={**handed, **provider.env},
        pass_env=passed,
        # One process that stays up for each tree: the model is loaded once, and
        # an indexing run goes on past the call that started it.
        host=Host(
            command=[*cli.invocation(), "serve"],
            label="text index",
            warm=["text_index"],
            eager=config.index.on_mount,
        ),
        tools={
            "text_index": ToolTraits(
                # Two of these in one batch would both load, both mutate and both
                # save. The engine's lock makes the outcome correct rather than
                # racy, and this keeps the scheduler from queueing them against
                # each other in the first place (B6).
                concurrency_safe=False,
                self_limits=True,
                # The passages are a payload this call delivered to the index,
                # not something the model refers back to.
                arguments_disposable=True,
                timeout_ms=INDEX_TIMEOUT_MS,
                view=ToolView(card="search", title="Index text", argument="paths"),
            ),
            "text_search": ToolTraits(
                concurrency_safe=True,
                self_limits=True,
                timeout_ms=SEARCH_TIMEOUT_MS,
                view=ToolView(card="search", title="Search text", argument="query"),
            ),
        },
    )


def offer_skills(ctx: Context) -> None:
    """Install this package's `SKILL.md` files (`ph.seams.skills.offer_skills`).

    **Called from the provider's claim, not from the `text-index` row's `apply`.**
    The tools appear only once an embedder exists, and the skill has to follow
    them: with the seam mounted and no provider — a configuration this package
    explicitly supports — the catalog line for `text-search` rode every prompt
    while `text_index`/`text_search` were absent.
    """
    offer_package_skills(ctx, Path(__file__).parent / "skills", source="ph-text-index")


@plugin(ROW, affects="environment", inject=[UTCP, FS, SHELL], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam an embedder's row claims, and offer what goes with the tools."""
    seam = TextIndexRow(ctx=ctx, config=config)
    ctx.provide(TEXT_INDEX, seam)
    utcp = ctx.require(UTCP)

    async def provision(verb: Provisioning, invocation: CommandContext) -> str:
        """`/text-index install|status` — the command's own answer, asked of it.

        A **command** rather than a tool, per the seam's own rule: a person asks
        the harness to provision, and it costs no model turn. Unconfined, as it
        was when the model was fetched in this process: a person asked, and
        `install` needs the network.

        About the agent that typed it: the index its `text_index` builds, so
        `status` counts what its `text_search` would answer from
        (`UtcpSeam.run_command`).
        """
        provider, own = seam.provider, seam.manual
        if provider is None or own is None:
            return (
                "No embedder is registered, so there is nothing to install — "
                "mount `text-index-local` or a provider row of your own."
            )
        result = await utcp.run_command(
            own,
            (*provider.cli.invocation(), verb),
            invocation.agent,
            timeout_ms=PROVISION_TIMEOUT_MS,
        )
        return command_said(provider.cli.program, verb, result)

    contribute_item(
        ctx,
        COMMANDS,
        install_or_status(
            ROW,
            "Download and load the embedding model, or report whether it is ready.",
            provision,
        ),
        label="text-index command",
    )
    contribute(ctx, Diagnostic(id=ROW, title="Text index", order=61, read=seam.report))


@plugin(LOCAL_ROW, affects="environment", inject=[TEXT_INDEX, UTCP, FS, SHELL], config=LocalConfig)
async def local(ctx: Context, config: LocalConfig) -> None:
    """Name a local `sentence-transformers` model as the embedder."""
    # The package, not the weights. Refusing here is the difference between "this
    # deployment cannot embed" — which an operator can fix — and a manual whose
    # discovery fails with the same sentence several layers down. `find_spec`
    # loads nothing: the model's own imports stay in the command.
    if importlib.util.find_spec("sentence_transformers") is None:
        raise MountRefusal(MISSING_MODEL)

    seam = ctx.require(TEXT_INDEX)
    provider = Provider(
        row=LOCAL_ROW,
        config=config,
        cli=CLI,
        label=config.model,
        max_tokens=config.max_tokens,
        # Shared by every agent: weights are the model's content, keyed by its
        # name. A confined command reads them here, and `huggingface_hub` keeps
        # its locks beside them.
        writable=(str(weights_cache(config)),),
    )

    if config.preload:
        # At mount, and refusing the mount if it fails — the cookbook's rule, and
        # the reason this option exists: an unattended run has nobody to type
        # `/text-index install`. The command's `install`, which fetches and loads
        # the model and exits failing when it cannot.
        result = await ctx.require(UTCP).run_command(
            manual(seam.config, provider),
            (*CLI.invocation(), "install"),
            None,
            timeout_ms=PROVISION_TIMEOUT_MS,
        )
        if result.exit_code != 0:
            said = command_said(CLI.program, "install", result)
            raise MountRefusal(f"text-index-local was asked to preload and could not: {said}")

    await seam.register(provider, scope=ctx)


@plugin(
    ENDPOINT_ROW, affects="environment", inject=[TEXT_INDEX, UTCP, FS, SHELL], config=EndpointConfig
)
async def endpoint(ctx: Context, config: EndpointConfig) -> None:
    """Name an OpenAI-compatible embeddings endpoint as the embedder.

    Nothing of the model is on this machine, so nothing is granted and nothing is
    checked here. The key, when the endpoint wants one, is named and not held:
    `ph-utcp` reads that one variable from the harness's environment each time
    it starts the command. For a confined agent the endpoint's host has to be
    among the hosts the deployment allows (`sandbox-allow`); the command says so
    when it cannot reach it."""
    where = config.base_url.rstrip("/")
    await ctx.require(TEXT_INDEX).register(
        Provider(
            row=ENDPOINT_ROW,
            config=config,
            cli=CLI,
            label=f"{config.model} at {where}" if config.model else where,
            max_tokens=config.max_tokens,
        ),
        scope=ctx,
    )


@dataclass(slots=True)
class _Embedding:
    """The session's side of `text-index-adapter`: a socket a command sends its
    texts to, and the model route that turns them into vectors.

    **This is where the credential is, and the only place.** The command runs in
    an agent's sandbox, so nothing it holds is out of that agent's reach. The
    route's key is resolved by the adapter, in this process, for the length of
    one request; what crosses the socket is text in and numbers out.

    Anything that can reach the socket can spend the route: it is granted to the
    commands this session's rows start, as the hosts' sockets are.

    **What it spends is in the log of the agent it was spent for.** A request
    names the agent its command works for (`ph.toolcli.AGENT_ENV`) — whose index
    it is, which for a child that shares its parent's is the parent — and the
    route's door records the spend there (`LlmRuntime.embed`), where a budget
    and a child's account read it. The name comes from inside a sandbox, so it
    decides which log of this session's family a record lands in and nothing
    else; a request that names nobody here is embedded and counted on this row
    alone.
    """

    ctx: Context
    config: AdapterConfig
    path: Path
    requests: int = 0
    texts: int = 0
    tokens: int = 0
    failed: str = ""
    """The last failure, for `phern doctor`: a key that is not set says so here
    as well as on the call that met it."""
    _serving: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    """What `close` cancels. The row's own stop, because a listener never ends by
    itself and a teardown that waited for it would wait out its whole drain."""

    async def serve(self) -> None:
        with self._serving:
            async with await anyio.create_unix_listener(self.path) as listener:
                await listener.serve(partial(answer_one, respond=self._embedded))

    def close(self) -> None:
        self._serving.cancel()
        self.path.unlink(missing_ok=True)

    async def _embedded(self, asked: dict[str, Any]) -> dict[str, object]:
        """One request's answer, in `ph.toolcli`'s envelope."""
        config = self.config
        if asked.get("limit"):
            # How long an input the route reads, as far as the route says: its
            # window. The command holds a passage to it.
            window = self.window()
            return value_envelope({"limit": None if window is None else window.tokens})
        inputs = tuple(as_str(one) for one in as_seq(asked.get("inputs")))
        try:
            result = await self.ctx.require(LLM).embed(
                EmbeddingRequest(
                    provider=config.provider,
                    model=config.model,
                    inputs=inputs,
                    dimensions=config.dimensions,
                ),
                session=self._session_of(asked.get("agent")),
            )
        except LlmError as error:
            # The adapter's own sentence and code: a key that is not set is said
            # by the variable's name, as it is on a model call.
            self.failed = f"{error.code}: {error}"
            return failure_envelope(error.code, str(error))
        except Exception as error:
            log.warning("ph_text_index: the session could not embed", exc_info=True)
            self.failed = f"{type(error).__name__}: {error}"
            return failure_envelope("EMBEDDING_FAILED", self.failed)
        self.requests += 1
        self.texts += len(inputs)
        self.tokens += result.input_tokens or 0
        self.failed = ""
        # The bytes of a float32 array, base64: a quarter the size of a JSON list
        # of the same numbers, and nothing for either end to parse. Packed here and
        # read by `SessionEmbedder._ask`, which this process never imports.
        packed = array("f", chain.from_iterable(result.vectors)).tobytes()
        width = len(result.vectors[0]) if result.vectors else 0
        return value_envelope({"vectors": base64.b64encode(packed).decode("ascii"), "dim": width})

    def window(self) -> Limit | None:
        """The route's window, when the route says one."""
        config = self.config
        window = self.ctx.require(LLM).resolve_model(config.provider, config.model).context_window
        return None if window is None or window <= 0 else Limit(window, "route")

    def _session_of(self, agent_id: object) -> Session | None:
        """The log an embedding for `agent_id` is recorded in: that agent's, when
        it is one of this session's."""
        registry = self.ctx.get(AGENTS)
        if registry is None or not isinstance(agent_id, str):
            return None
        driver = registry.get(agent_id)
        return None if driver is None else driver.session

    def report(self) -> list[tuple[str, str]]:
        spent = f", {self.tokens} tokens" if self.tokens else ""
        rows = [("embedded", f"{self.texts} texts in {self.requests} requests{spent}")]
        if self.failed:
            rows.append(("last failure", self.failed))
        return rows


@plugin(
    ADAPTER_ROW,
    affects="environment",
    inject=[TEXT_INDEX, UTCP, FS, SHELL, LLM],
    config=AdapterConfig,
)
async def adapter(ctx: Context, config: AdapterConfig) -> None:
    """Name one of the session's own model routes as the embedder.

    The command asks this row for vectors over a socket and this row asks the
    route's adapter, so the route's credential never leaves the harness
    (`_Embedding`). A route whose adapter cannot embed is refused here, by name:
    a profile that names one is corrected by whoever wrote it, now, and not by an
    agent meeting a failed call."""
    llm = ctx.require(LLM)
    if config.provider in llm.list_providers() and not llm.embeds(config.provider):
        raise MountRefusal(
            f'{ADAPTER_ROW} names provider "{config.provider}", whose adapter does not '
            "embed. Name a provider whose adapter does — the OpenAI-compatible and Google "
            f"adapters do — or use `{ENDPOINT_ROW}` or `{LOCAL_ROW}`."
        )
    directory = ctx.require(UTCP).socket_dir
    if directory is None:
        raise MountRefusal(
            f"{ADAPTER_ROW} needs $PH_RUNTIME for the socket its command asks on, and "
            "there is no usable runtime directory"
        )
    service = _Embedding(
        ctx=ctx, config=config, path=directory / f"embed-{secrets.token_hex(6)}.sock"
    )
    # `leave`, with the stop registered beside it: the listener serves until it
    # is told to stop, so it is not work a teardown should wait to see finish.
    ctx.detach(service.serve(), label="text-index embeddings", at_dispose="leave")
    ctx.add_disposer(service.close, label="text-index embeddings")
    await ctx.require(TEXT_INDEX).register(
        Provider(
            row=ADAPTER_ROW,
            config=config,
            cli=CLI,
            label=f"{config.model} through {config.provider}",
            max_tokens=config.max_tokens,
            reads=service.window,
            # The socket the command asks on: a confined command connects there.
            writable=(str(directory),),
            env={EMBEDDER_ENV: str(service.path)},
            report=service.report,
        ),
        scope=ctx,
    )
