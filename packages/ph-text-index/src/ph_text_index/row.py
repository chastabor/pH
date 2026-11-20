"""`text-index`, `text-index-local`, and `text-index-endpoint`: the rows that a
session mounts. They register the manual of the package.

The package has two tools. `text_index` cuts documents into passages, embeds
them, and puts them in an index on disk. `text_search` answers a question with
the passages themselves, and with the `path:start-end` that each passage came
from.

## The rows that a session mounts register a manual

The tools run in the `ph-text-index` command, not in the process of the session.
These rows add the UTCP manual of the package to `ctx.utcp`. `ph-utcp` then
registers what the manual names, under the tools' own names, with the same
values, rendered text, and error codes as before. The tool bodies are in
`ph_text_index.engine`, which the command mounts.

This arrangement has three benefits:

* **Any UTCP harness can use the same tools**, by registering the same manual
  (`ph-text-index manual --registration`).
* **The tool sees what its agent sees.** `ctx.shell` starts the command for the
  agent that called it, in the workspace of that agent and under its sandbox. A
  child agent that works in its own worktree indexes that worktree.
* **The harness process never loads the model.** `sentence-transformers`, torch,
  and turbovec are in the process of the command. They take several hundred
  megabytes, which a session that never searches does not hold.

The cost is stated in the README of `ph-utcp`, and it is repeated here.
`permissions-fs` governs calls through `ctx.fs`, and a command is not such a
call. The sandbox limits the reads of these tools, as it limits the reads of
`bash`.

## Why the manual waits for an embedder

`text-index` publishes a seam, `ctx.text_index`. An embedder row, here
`text-index-local` or `text-index-endpoint`, claims the seam with a `Provider`.
A `Provider` says which row of which command makes the vectors, and gives the
config of that row. The manual is registered **by the claim**, on the scope of
the provider. A profile that mounts `text-index` and no embedder therefore
advertises nothing, and unmounting the embedder removes the tools and the skill
with it. This is the rule that the engine keeps for the same reason (see its
module docstring), applied one process up.

The claim also carries the config across. The command reads the settings of
both rows from `$PH_TOOL_CONFIG`. The command learns which embedder row to mount
from the row that is named there (`ph.toolcli.ToolCli.providers`). The model, or
a different embedder, is therefore chosen where it always was: in the profile of
the session.

## What these rows still do themselves

The rows do the things that a manual cannot express:

* They name where an index goes: a directory in the scratch directory of the
  calling agent (`Manual.state`). That directory is writable in every sandbox
  mode. An agent that may not change a tree can therefore still index it, and
  cannot write the tree by doing so. It is the same directory for the same agent
  after a restart, so a resumed session embeds only what changed.
* They grant the command the paths that agents share: the weights of the model
  and the log of the command, both under `$PH_CACHE`.
* They offer the skill, the `/text-index` command, and the `phern doctor`
  section, as they did when the tools were registered here.

## Why the rows have their own module

The rows are not in the `__init__` of the package, which `python -P -m
ph_text_index` imports before it runs a call. These rows import `ph-utcp`, and
with it the UTCP library. Every tool call would then pay to load code that only
the session uses.

@module ph_text_index.row
"""

from __future__ import annotations

import importlib.util
import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from ph.cordis import Context, Disposer, MountRefusal, Running, ServiceKey, plugin
from ph.keys import COMMANDS, FS, SHELL
from ph.seams._registry import claim_slot, contribute_item
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import offer_skills as offer_package_skills
from ph.toolcli import ToolCli, config_env, manual_call_template
from ph_utcp import UTCP, Background, Manual, ManualState, ToolTraits, ToolView, command_said

from ._config import (
    ENDPOINT_ROW,
    LOCAL_ROW,
    MISSING_MODEL,
    ROW,
    STATE,
    Config,
    EndpointConfig,
    Limit,
    LocalConfig,
    weights_cache,
)
from .cli import CLI

__all__ = [
    "TEXT_INDEX",
    "Config",
    "EndpointConfig",
    "LocalConfig",
    "Provider",
    "TextIndexRow",
    "apply",
    "endpoint",
    "local",
]

PROVISION_TIMEOUT_MS = 30 * 60 * 1000
"""How long `/text-index install` may run: the download of a model, and its first
load."""

INDEX_TIMEOUT_MS = 30 * 60 * 1000
"""How long one `text_index` call may run. In a session, the call answers within
`wait_s` and the run continues in the background. This limit is for the call
that does the work itself. The `docs/` of this repository takes 20 seconds under
MiniLM and three minutes under nomic."""

SEARCH_TIMEOUT_MS = 5 * 60 * 1000
"""How long one `text_search` call may run. With a local model, each call loads
the model, and a cold load takes tens of seconds."""


@dataclass(frozen=True, slots=True)
class Provider:
    """An embedder, as a session knows it: which row of which command makes the
    vectors, and what the session must grant it."""

    row: str
    """The id that the command mounts the embedder row under. It is also the key
    that the config of that row travels under in `$PH_TOOL_CONFIG`."""
    config: BaseModel
    cli: ToolCli
    """The command that mounts that row. For a row that this package ships, it is
    the command of this package."""
    label: str
    """What `phern doctor` calls it: the model."""
    writable: tuple[str, ...] = ()
    """The shared paths that it writes outside the workspace: the weights of a
    model."""
    max_tokens: int | None = None
    """The limit a profile set on the embedder: how long an input it reads."""

    def limit(self) -> Limit | None:
        """The longest input that the embedder reads, as far as a session knows.
        That is the setting. When there is no setting, only the command can say."""
        return (
            None if self.max_tokens is None else Limit(tokens=self.max_tokens, source="configured")
        )


TEXT_INDEX: ServiceKey[TextIndexRow] = ServiceKey("text_index")
"""The text index of the session: the seam that an embedder row claims."""


@dataclass(slots=True)
class TextIndexRow:
    """The service published as `ctx.text_index` in a session."""

    ctx: Context
    config: Config
    provider: Provider | None = None
    provider_by: Running | None = None
    manual: Manual | None = None
    """The manual that is registered for the current provider."""
    state: ManualState | None = None
    """The result of adding that manual to `ph-utcp`."""

    async def register(self, provider: Provider, *, scope: Context | None = None) -> ManualState:
        """Claim the embedder slot, and register the manual that the claim makes
        possible.

        Both are on the scope of the *provider*, so unmounting the embedder row
        removes the tools, the skill, and the claim with it. A second claim is
        refused, because one index is one vector space.
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
            # The skill follows the tools. If it were offered without them, its
            # catalog line would be in every prompt and would name two tools
            # that the session does not have.
            offer_skills(owner)
        return state

    def report(self) -> list[tuple[str, str]]:
        """The section for `phern doctor`: what these rows know without starting
        the command. The command itself reports counts and whether the model
        loads."""
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
                # Only the command asks for the length that a model or a server
                # reads. For a local model, asking means loading it.
                limit.said()
                if limit is not None
                else f"asked of the embedder when it embeds; `/{ROW} status` shows it",
            ),
            ("index", f"each agent's own, in its scratch ({STATE}/)"),
        ]
        rows.extend(("shared", path) for path in provider.writable)
        rows.append(("details", f"/{ROW} status"))
        return rows


def manual(config: Config, provider: Provider) -> Manual:
    """Build the manual of this package for one embedder, with what `ph-utcp`
    needs to know to run it."""
    cli = provider.cli
    handed, passed = config_env({ROW: config, provider.row: provider.config})
    return Manual(
        template=manual_call_template(cli),
        # This manual is the only source of these two tool names, so the tools
        # keep them. The skill, the prompts, and every program that a model has
        # written say `text_search`.
        names="bare",
        envelope=True,
        # The vectors, the sidecar, and the lock file that orders their writers.
        # Each agent has its own, in its scratch directory.
        state=STATE,
        writable=[
            *provider.writable,
            # The command logs to a file, because under UTCP its stderr is discarded.
            str(cli.log_file().parent),
        ],
        # The configs of both rows, each under its row id. The row id is also how
        # the command learns which embedder row to mount. `pass_env` passes by
        # name the variables that the configs mark (the key of an endpoint).
        env=handed,
        pass_env=passed,
        # An indexing run continues after the call that started it, and reports
        # on the stderr of the command. One run starts before the first step of
        # a root agent.
        background=Background(
            command=[*cli.invocation(), "call"],
            label="text index",
            tools=["text_index"],
            eager=config.index.on_mount,
            wait_s=config.wait_s,
        ),
        tools={
            "text_index": ToolTraits(
                # Two of these calls in one batch would both load, both change,
                # and both save. The lock of the engine makes the outcome
                # correct, and this setting keeps the scheduler from queueing
                # the two calls against each other (B6).
                concurrency_safe=False,
                self_limits=True,
                # The passages are a payload that this call delivered to the
                # index. The model does not refer back to them.
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

    **The claim of the provider calls this, not the `apply` of the `text-index`
    row.** The tools appear only when an embedder exists, and the skill must
    follow them. With the seam mounted and no provider, which is a configuration
    that this package explicitly supports, the catalog line for `text-search`
    was in every prompt while `text_index`/`text_search` were absent.
    """
    offer_package_skills(ctx, Path(__file__).parent / "skills", source="ph-text-index")


@plugin(ROW, affects="environment", inject=[UTCP, FS, SHELL], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam that an embedder row claims, and offer what goes with the
    tools."""
    seam = TextIndexRow(ctx=ctx, config=config)
    ctx.provide(TEXT_INDEX, seam)
    utcp = ctx.require(UTCP)

    async def provision(verb: Provisioning, invocation: CommandContext) -> str:
        """`/text-index install|status`: ask the command and return its answer.

        This is a **command** and not a tool, by the rule of the seam: a person
        asks the harness to provision, and that costs no model turn. It is not
        confined, as it was not when the model was fetched in this process. A
        person asked, and `install` needs the network.

        The answer is about the agent that typed the command. It is about the
        index that the `text_index` of that agent builds, so `status` counts
        what the `text_search` of that agent would answer from
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
    # This checks for the package, not the weights. A refusal here says "this
    # deployment cannot embed", which an operator can fix. Without it, the
    # discovery of the manual would fail with the same sentence several layers
    # down. `find_spec` loads nothing, so the imports of the model stay in the
    # command.
    if importlib.util.find_spec("sentence_transformers") is None:
        raise MountRefusal(MISSING_MODEL)

    seam = ctx.require(TEXT_INDEX)
    provider = Provider(
        row=LOCAL_ROW,
        config=config,
        cli=CLI,
        label=config.model,
        max_tokens=config.max_tokens,
        # Every agent shares this path. Weights are content of the model, keyed
        # by its name. A confined command reads them here, and `huggingface_hub`
        # keeps its locks beside them.
        writable=(str(weights_cache(config)),),
    )

    if config.preload:
        # This runs at mount, and refuses the mount if it fails. That is the rule
        # of the cookbook, and the reason that this option exists: an unattended
        # run has nobody to type `/text-index install`. It runs the `install` of
        # the command, which fetches and loads the model, and exits with a
        # failure when it cannot.
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

    No part of the model is on this machine, so this row grants nothing and
    checks nothing. When the endpoint needs a key, the key is named and not
    held. `ph-utcp` reads that one variable from the environment of the harness
    each time it starts the command. For a confined agent, the host of the
    endpoint must be among the hosts that the deployment allows
    (`sandbox-allow`). The command says so when it cannot reach the host.
    `ph-utcp` reports whether the scrub of the deployment drops the name of the
    key, as it does for every variable that a manual passes (`UtcpSeam.add`)."""
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
