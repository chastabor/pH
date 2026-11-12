"""`code-graph` — the row a session mounts: it registers the package's manual.

Two tools. `code_index` walks a tree, parses it with tree-sitter and records what
each file defines and references. `code_graph` answers six questions over that
index — and answers every one of them with `path:start-end`, so a hit is a
pointer the agent can hand straight to `read`.

## The row a session mounts registers a manual

The tools run in the `ph-code-graph` command, not in the session's process. This
row adds the package's UTCP manual to `ctx.utcp`, and `ph-utcp` registers what
the manual names — under the tools' own names, with the values, the rendered
text and the error codes they have always had. The bodies are in
`ph_code_graph.engine`, which the command mounts.

What that arrangement buys:

* **Any UTCP harness can use the same tools**, by registering the same manual
  (`ph-code-graph manual --registration`).
* **The tool sees what its agent sees.** The command is started by `ctx.shell`
  for the agent that called it: in that agent's workspace, under its sandbox. A
  child working in its own worktree indexes that worktree.
* **The harness process never loads the parser.** tree-sitter and its grammars
  live in the command's process, and so does a crash in one.

What it costs is stated in `ph-utcp`'s README and worth repeating here:
`permissions-fs` governs calls through `ctx.fs`, and a command is not one. These
tools' reads are bounded by the sandbox, as `bash`'s are.

## What this row still does itself

The things a manual has no word for. It names where the index goes: a directory
of the calling agent's scratch (`Manual.state`), which is writable under every
sandbox mode — so an agent that may not touch a tree can still index it, and
cannot write the tree doing so — and is the same directory for the same agent
after a restart, so a resumed session re-parses only what changed. It grants the
command what agents share, the grammar cache and its log, both under `$PH_CACHE`.
It hands its own config across in `$PH_TOOL_CONFIG`. And it offers the skill, the
`/code-graph` command and the `phern doctor` section, as it did when the tools
were registered here.

If the manual cannot be discovered the row stands down rather than refusing the
mount: no tools, no skill, and a `phern doctor` line saying why.

## Its own module

Not the package's `__init__`, which `python -m ph_code_graph` imports before it
runs a single call: this row imports `ph-utcp` and with it the UTCP library, and
every tool call would pay to load what only the session uses.

@module ph_code_graph.row
"""

from __future__ import annotations

import shlex
from pathlib import Path

from ph.cordis import Context, plugin
from ph.keys import COMMANDS, FS, SHELL
from ph.seams._registry import contribute_item
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import offer_skills as offer_package_skills
from ph.toolcli import config_env, manual_call_template
from ph_utcp import UTCP, Host, Manual, ToolTraits, ToolView, command_said

from ._config import ROW, STATE, Config, grammar_cache
from .cli import CLI

__all__ = ["Config", "apply", "manual"]

PROVISION_TIMEOUT_MS = 10 * 60 * 1000
"""How long `/code-graph install` may run. Grammars are fetched from GitHub, so
this is a download's budget and not a query's."""


def manual(config: Config) -> Manual:
    """This package's manual, and what `ph-utcp` needs to know to run it."""
    env, pass_env = config_env({ROW: config})
    return Manual(
        template=manual_call_template(CLI),
        # The only source of these two names, so they keep them: the skill, the
        # prompts and every program a model has written say `code_graph`.
        names="bare",
        envelope=True,
        # The database and the lock file that orders its writers: each agent's
        # own, in its scratch.
        state=STATE,
        writable=[
            # The pack keeps every grammar it uses in a writable directory,
            # fetched there the first time, and fails hard without one. Shared:
            # grammars are the package's content, keyed by its version, and say
            # nothing of a repository.
            str(grammar_cache()),
            # The command logs to a file, because under UTCP its stderr is
            # discarded.
            str(CLI.log_file().parent),
        ],
        env=env,
        pass_env=pass_env,
        # One process that stays up for each tree: a call does not pay to start
        # an interpreter, and a long indexing run goes on past the call that
        # started it.
        host=Host(
            command=[*CLI.invocation(), "serve"],
            label="code graph",
            warm=["code_index"],
            eager=config.index.on_mount,
        ),
        tools={
            "code_index": ToolTraits(
                # One index, one writer at a time: the lock makes two runs
                # correct, and this keeps the scheduler from queueing them
                # against each other in the first place (B6).
                concurrency_safe=False,
                self_limits=True,
                view=ToolView(card="search", title="Index code", argument="paths"),
            ),
            "code_graph": ToolTraits(
                concurrency_safe=True,
                self_limits=True,
                view=ToolView(card="search", title="Code graph", argument="query"),
            ),
        },
    )


def offer_skills(ctx: Context) -> None:
    """Install this package's `SKILL.md` files (`ph.seams.skills.offer_skills`)."""
    offer_package_skills(ctx, Path(__file__).parent / "skills", source="ph-code-graph")


@plugin(ROW, affects="environment", inject=[UTCP, FS, SHELL], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the manual, and offer what goes with its tools."""
    utcp = ctx.require(UTCP)
    own = manual(config)
    state = await utcp.add(own, scope=ctx)
    if state.available:
        # The skill follows the tools. Offered without them, its catalog line
        # would ride every prompt naming two tools the session does not have.
        offer_skills(ctx)

    async def provision(verb: Provisioning, invocation: CommandContext) -> str:
        """`/code-graph install|status` — the command's own answer, asked of it.

        A **command** rather than a tool, per the seam's own rule: a person asks
        the harness to provision, and it costs no model turn. Unconfined, as it
        was when the grammars were fetched in this process: a person asked, and
        `install` needs the network.

        About the agent that typed it: the index its `code_index` builds, so
        `status` counts what its `code_graph` would answer from
        (`UtcpSeam.run_command`).
        """
        result = await utcp.run_command(
            own, (*CLI.invocation(), verb), invocation.agent, timeout_ms=PROVISION_TIMEOUT_MS
        )
        return command_said(CLI.program, verb, result)

    contribute_item(
        ctx,
        COMMANDS,
        install_or_status(
            CLI.command,
            "Make the tree-sitter grammars ready, or report whether they are.",
            provision,
        ),
        label="code-graph command",
    )

    def report() -> list[tuple[str, str]]:
        """`phern doctor`'s section: what this row knows without starting the
        command. Counts and grammar readiness are the command's to say."""
        return [
            ("tools", state.said()),
            ("command", shlex.join(CLI.invocation())),
            ("index", f"each agent's own, in its scratch ({STATE}/)"),
            ("grammars", str(grammar_cache())),
            ("details", f"/{CLI.command} status"),
        ]

    contribute(ctx, Diagnostic(id=ROW, title="Code graph", order=60, read=report))
