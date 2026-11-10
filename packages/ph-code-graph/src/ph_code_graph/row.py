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

The things a manual has no word for. It grants the command the paths it writes
outside the workspace — the index and the grammar cache, both under `$PH_CACHE` —
which holds in a read-only session too, so an agent that may not touch a tree can
still index it. It hands its own config across in `$PH_TOOL_CONFIG`. And it
offers the skill, the `/code-graph` command and the `phern doctor` section, as it
did when the tools were registered here.

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
from ph.seams.commands import Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import offer_skills as offer_package_skills
from ph.toolcli import config_env, manual_call_template
from ph_utcp import UTCP, Manual, ManualState, ToolTraits, ToolView

from ._config import ROW, Config, grammar_cache, index_base, index_path
from .cli import CLI

__all__ = ["Config", "apply", "manual"]

PROVISION_TIMEOUT_MS = 10 * 60 * 1000
"""How long `/code-graph install` may run. A grammar outside the wheel's bundled
set is fetched from GitHub, so this is a download's budget and not a query's."""


def manual(config: Config) -> Manual:
    """This package's manual, and what `ph-utcp` needs to know to run it."""
    writable = [
        # One directory per workspace beneath it, each with its database and the
        # lock file that orders its writers.
        index_base(),
        # The pack materializes even its bundled grammars into a writable
        # directory on first use, and fails hard without one.
        grammar_cache(),
        # The command logs to a file, because under UTCP its stderr is discarded.
        CLI.log_file().parent,
    ]
    if config.path:
        writable.append(index_path(config, Path()).parent)
    return Manual(
        template=manual_call_template(CLI),
        # The only source of these two names, so they keep them: the skill, the
        # prompts and every program a model has written say `code_graph`.
        names="bare",
        envelope=True,
        writable=[str(path) for path in writable],
        env=config_env({ROW: config}),
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
    state = await ctx.require(UTCP).add(manual(config), scope=ctx)
    if state.available:
        # The skill follows the tools. Offered without them, its catalog line
        # would ride every prompt naming two tools the session does not have.
        offer_skills(ctx)

    async def provision(verb: Provisioning) -> str:
        """`/code-graph install|status` — the command's own answer, asked of it.

        A **command** rather than a tool, per the seam's own rule: a person asks
        the harness to provision, and it costs no model turn. Run for no agent,
        so in the session's own root and unconfined, as it was when the grammars
        were fetched in this process: a person asked, and `install` needs the
        network.
        """
        root = ctx.require(FS).root_for(None)
        result = await ctx.require(SHELL).run_argv(
            (*CLI.invocation(), verb, "--root", str(root)),
            cwd=root,
            timeout_ms=PROVISION_TIMEOUT_MS,
            extra_env=config_env({ROW: config}),
        )
        said = result.stdout.strip() or result.stderr.strip()
        if result.exit_code != 0:
            return f"{CLI.program} {verb} failed (exit {result.exit_code}): {said}"
        return said

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
        index = index_path(config, ctx.require(FS).root_for(None))
        return [
            ("tools", _said(state)),
            ("command", shlex.join(CLI.invocation())),
            ("index", str(index)),
            ("state", "built" if index.exists() else "not built — run code_index"),
            ("details", f"/{CLI.command} status"),
        ]

    contribute(ctx, Diagnostic(id=ROW, title="Code graph", order=60, read=report))


def _said(state: ManualState) -> str:
    if state.available and not state.problems:
        return f"{', '.join(state.tools)} (through UTCP)"
    reasons = "; ".join(state.problems) or "the manual named no tools"
    return f"{', '.join(state.tools) or 'unavailable'} — {reasons}"
