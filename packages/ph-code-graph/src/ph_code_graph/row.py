"""`code-graph`: the row that a session mounts. It registers the manual of the package.

The package has two tools. `code_index` walks a tree, parses it with tree-sitter,
and records what each file defines and references. `code_graph` answers six
questions over that index. Every answer is a `path:start-end`, so the agent can
pass a hit directly to `read`.

## The row that a session mounts registers a manual

The tools run in the `ph-code-graph` command, not in the process of the session.
This row adds the UTCP manual of the package to `ctx.utcp`. `ph-utcp` then
registers what the manual names, under the tools' own names, with the same
values, rendered text, and error codes as before. The tool bodies are in
`ph_code_graph.engine`, which the command mounts.

This arrangement has three benefits:

* **Any UTCP harness can use the same tools**, by registering the same manual
  (`ph-code-graph manual --registration`).
* **The tool sees what its agent sees.** `ctx.shell` starts the command for the
  agent that called it, in the workspace of that agent and under its sandbox. A
  child agent that works in its own worktree indexes that worktree.
* **The harness process never loads the parser.** tree-sitter and its grammars
  are in the process of the command, and a crash in them stays there.

The cost is stated in the README of `ph-utcp`, and it is repeated here.
`permissions-fs` governs calls through `ctx.fs`, and a command is not such a
call. The sandbox limits the reads of these tools, as it limits the reads of
`bash`.

## What this row still does itself

The row does the things that a manual cannot express:

* It names where the index goes: a directory in the scratch directory of the
  calling agent (`Manual.state`). That directory is writable in every sandbox
  mode. An agent that may not change a tree can therefore still index it, and
  cannot write the tree by doing so. It is the same directory for the same agent
  after a restart, so a resumed session parses again only what changed.
* It grants the command the paths that agents share: the grammar cache and the
  log of the command, both under `$PH_CACHE`.
* It passes its own config to the command in `$PH_TOOL_CONFIG`.
* It offers the skill, the `/code-graph` command, and the `phern doctor`
  section, as it did when the tools were registered here.

If the manual cannot be discovered, the row stands down and does not refuse the
mount. There are then no tools and no skill, and a `phern doctor` line gives the
reason.

## Why the row has its own module

The row is not in the `__init__` of the package, which `python -P -m
ph_code_graph` imports before it runs a call. This row imports `ph-utcp`, and
with it the UTCP library. Every tool call would then pay to load code that only
the session uses.

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
from ph_utcp import UTCP, Background, Manual, ToolTraits, ToolView, command_said

from ._config import ROW, STATE, Config, grammar_cache
from .cli import CLI

__all__ = ["Config", "apply", "manual"]

PROVISION_TIMEOUT_MS = 10 * 60 * 1000
"""How long `/code-graph install` may run. The grammars are fetched from GitHub,
so this is the time limit for a download and not for a query."""


def manual(config: Config) -> Manual:
    """Build the manual of this package, with what `ph-utcp` needs to know to run it."""
    env, pass_env = config_env({ROW: config})
    return Manual(
        template=manual_call_template(CLI),
        # This manual is the only source of these two tool names, so the tools
        # keep them. The skill, the prompts, and every program that a model has
        # written say `code_graph`.
        names="bare",
        envelope=True,
        # The database and the lock file that orders its writers. Each agent has
        # its own, in its scratch directory.
        state=STATE,
        writable=[
            # The pack keeps every grammar that it uses in a writable directory.
            # It fetches each grammar there the first time, and fails hard
            # without the directory. The directory is shared: grammars are
            # content of the package, keyed by its version, and they say nothing
            # about a repository.
            str(grammar_cache()),
            # The command logs to a file, because under UTCP its stderr is
            # discarded.
            str(CLI.log_file().parent),
        ],
        env=env,
        pass_env=pass_env,
        # An indexing run continues after the call that started it, and reports
        # on the stderr of the command. One run starts before the first step of
        # a root agent.
        background=Background(
            command=[*CLI.invocation(), "call"],
            label="code graph",
            tools=["code_index"],
            eager=config.index.on_mount,
            wait_s=config.wait_s,
        ),
        tools={
            "code_index": ToolTraits(
                # One index has one writer at a time. The lock makes two runs
                # correct. This setting keeps the scheduler from queueing them
                # against each other at all (B6).
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
        # The skill follows the tools. If it were offered without them, its
        # catalog line would be in every prompt and would name two tools that
        # the session does not have.
        offer_skills(ctx)

    async def provision(verb: Provisioning, invocation: CommandContext) -> str:
        """`/code-graph install|status`: ask the command and return its answer.

        This is a **command** and not a tool, by the rule of the seam: a person
        asks the harness to provision, and that costs no model turn. It is not
        confined, as it was not when the grammars were fetched in this process.
        A person asked, and `install` needs the network.

        The answer is about the agent that typed the command. It is about the
        index that the `code_index` of that agent builds, so `status` counts what
        the `code_graph` of that agent would answer from
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
        """The section for `phern doctor`: what this row knows without starting
        the command. The command itself reports counts and grammar readiness."""
        return [
            ("tools", state.said()),
            ("command", shlex.join(CLI.invocation())),
            ("index", f"each agent's own, in its scratch ({STATE}/)"),
            ("grammars", str(grammar_cache())),
            ("details", f"/{CLI.command} status"),
        ]

    contribute(ctx, Diagnostic(id=ROW, title="Code graph", order=60, read=report))
