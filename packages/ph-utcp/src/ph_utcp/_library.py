"""The one module that imports `utcp`, and the pH door its commands run through.

## Imported behind a fence

`utcp` and every protocol module call `logging.basicConfig(level=INFO)` when they
are imported. In a process whose root logger has no handler yet — a test, a
`phern -p` that has not configured logging — that installs a stderr handler at
INFO for the whole process, on behalf of a library a row happened to mount. So
the imports below are bracketed: the root logger's handlers and level are put
back as they were, and the library's own loggers are set to WARNING, because its
INFO line per call ("Executing shell script…") is narration a harness log does
not want.

Everything else in this package takes the library's names from here, so there is
no second place the fence could be forgotten.

## A command runs where a pH command runs

The library's `cli` protocol starts `/bin/bash -c <script>` itself, with
`asyncio`, in the client's own working directory. Used unchanged inside pH that
would be a shell command outside every row that bounds one: not the agent's
workspace, not the sandbox, not the orphan journal, not the call's cancellation.
`PhCliProtocol` keeps the library's script building — the argument substitution
is where its security fixes live — and replaces only how the script is started:
through `ctx.shell.run_argv`, for the agent the call is for.

Which agent travels in a `ContextVar`, because the library's `call_tool` has no
parameter for it: `caller` is the client, and one client serves every agent of a
mount. `calling()` sets it for the span of one call; outside any such span the
protocol behaves exactly as the library's own, so a plain `UtcpClient` in the
same process — a test, an embedding application — is not affected by this row
having been imported.

## Failing closed

The override rests on two private methods of `utcp_cli` and on winning the
library's registry, so each is checked rather than assumed. `install` refuses
when either method is missing from the class it overrides, and registers a
*factory* — which the library resolves ahead of any shared instance of the same
type. And a `Call` records that its command went through `run_argv`: a call that
comes back without having done so ran outside the sandbox, and `require_ran`
says so instead of handing its answer to a model.

@module ph_utcp._library
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ph.agent.types import AgentHandle
from ph.cancel import Canceled, Cancellation
from ph.cordis import MountRefusal
from ph.seams.shell import ShellService
from ph.tools.errors import HarnessError

_root = logging.getLogger()
_handlers, _level = list(_root.handlers), _root.level
try:
    from utcp.data.call_template import CallTemplateSerializer  # type: ignore[import-untyped]
    from utcp.data.tool import Tool  # type: ignore[import-untyped]
    from utcp.plugins.discovery import (  # type: ignore[import-untyped]
        register_communication_protocol_factory,
    )
    from utcp.plugins.plugin_loader import (  # type: ignore[import-untyped]
        ensure_plugins_initialized,
    )
    from utcp.utcp_client import UtcpClient  # type: ignore[import-untyped]
    from utcp_cli.cli_communication_protocol import (  # type: ignore[import-untyped]
        CliCommunicationProtocol,
    )
finally:
    _root.handlers[:] = _handlers
    _root.setLevel(_level)
for _name in ("utcp", "utcp_cli"):
    logging.getLogger(_name).setLevel(logging.WARNING)

__all__ = [
    "UTCP_TIMEOUT",
    "UTCP_UNCONFINED",
    "Call",
    "CallTemplateSerializer",
    "PhCliProtocol",
    "Tool",
    "UtcpClient",
    "calling",
    "install",
    "require_ran",
]

UTCP_TIMEOUT = "UTCP_TIMEOUT"
UTCP_UNCONFINED = "UTCP_UNCONFINED"

_OVERRIDDEN = ("_prepare_environment", "_execute_shell_script")
"""The `CliCommunicationProtocol` methods `PhCliProtocol` replaces."""


@dataclass(slots=True)
class Call:
    """Who one UTCP call is for, how pH runs its command, and what it did."""

    shell: ShellService
    root: Path
    """The agent's workspace, which a template's relative `working_dir` is under."""
    timeout_ms: int
    """pH's budget for the command, in place of the library's fixed one."""
    env: Mapping[str, str]
    """This manual's own variables, set over everything the template says."""
    writable: tuple[str, ...] = ()
    """Paths outside the workspace this manual's commands may write."""
    agent: AgentHandle | None = None
    signal: Cancellation | None = None
    inherited: tuple[str, ...] = ()
    """The names the template inherits, as the library read them. Selected by
    `run_argv` from the deployment's scrubbed environment rather than from
    `os.environ`, so a manual cannot name its way to a variable pH drops."""
    ran: bool = False
    """Whether a command of this call went through `run_argv`."""


_CALL: ContextVar[Call | None] = ContextVar("ph_utcp_call", default=None)


@contextmanager
def calling(call: Call) -> Iterator[None]:
    """Run the body's UTCP calls as `call`. One task, one call: the variable is
    read in the same task that set it, a few frames down the library."""
    token = _CALL.set(call)
    try:
        yield
    finally:
        _CALL.reset(token)


def require_ran(call: Call, what: str) -> None:
    """Refuse an answer whose command did not go through `run_argv`.

    After the fact, because nothing earlier can see it: the library chose another
    door — a protocol registered over this one, a method renamed under it — and
    ran the command itself, unconfined. The answer is withheld and the reason is
    the error, so the model and the log both see it."""
    if not call.ran:
        raise HarnessError(
            f"{what} was run by the UTCP library itself, outside pH's sandbox; its "
            "answer is withheld. The `cli` protocol pH installs was bypassed — check "
            "the installed utcp-cli against ph-utcp's pin.",
            UTCP_UNCONFINED,
        )


class PhCliProtocol(CliCommunicationProtocol):  # type: ignore[misc]
    """The library's `cli` protocol, with its commands started by `ctx.shell`."""

    def _prepare_environment(self, provider: Any) -> dict[str, str]:  # noqa: ANN401
        call = _CALL.get()
        if call is None:
            inherited: dict[str, str] = super()._prepare_environment(provider)
            return inherited
        # The names go to `run_argv`, which selects them from the scrubbed
        # environment below the workspace's redirection; only the template's own
        # values travel with the script.
        call.inherited = (
            self._default_inherited_keys()
            if provider.inherit_env_vars is None
            else tuple(provider.inherit_env_vars)
        )
        return dict(provider.env_vars or {})

    async def _execute_shell_script(
        self,
        script: str,
        env: dict[str, str],
        timeout: float = 60.0,
        working_dir: str | None = None,
    ) -> tuple[str, str, int]:
        call = _CALL.get()
        if call is None:
            outcome: tuple[str, str, int] = await super()._execute_shell_script(
                script, env, timeout, working_dir
            )
            return outcome
        call.ran = True
        result = await call.shell.run_argv(
            # Not the platform's login shell: `bash -lc` reads the person's
            # profile, and what a profile prints would be parsed as the answer.
            ("/bin/bash", "-c", script),
            cwd=None if working_dir is None else call.root / working_dir,
            agent=call.agent,
            # pH's budget, not the library's fixed 60 and 120 seconds: how long a
            # tool may take is the row's to say.
            timeout_ms=call.timeout_ms,
            signal=call.signal,
            keep_env=call.inherited,
            # The template's own values and the call's arguments (`env`), then the
            # manual's: what the caller supplies wins over what is inherited.
            extra_env={**env, **call.env},
            writable_extra=call.writable,
        )
        if result.canceled:
            raise Canceled("the UTCP command was canceled")
        if result.timed_out:
            raise HarnessError(
                f"the UTCP command ran past {call.timeout_ms / 1000:g}s and was stopped",
                UTCP_TIMEOUT,
            )
        return result.stdout, result.stderr, result.exit_code


_INSTALLED = False


def install() -> None:
    """Make `PhCliProtocol` the process's `cli` protocol. Idempotent.

    Registered as a factory, the library's per-client door, which its resolver
    takes ahead of a shared instance of the same type — so the library's own
    `cli` registration, an instance, cannot win over it. Process-wide because the
    registry is; safe to leave in place for the life of the process, for what the
    protocol does outside `calling()`.

    Refuses when a method it overrides is not on the class any more: the
    override would then be dead code, and every command would run unconfined.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    missing = [
        name for name in _OVERRIDDEN if not callable(getattr(CliCommunicationProtocol, name, None))
    ]
    if missing:
        raise MountRefusal(
            f"the installed utcp-cli has no {', '.join(missing)}, which ph-utcp overrides "
            "to run commands through pH's sandbox; install the utcp-cli version ph-utcp pins"
        )
    # First, so the library's own registrations have happened and the one below
    # replaces any `cli` factory among them rather than being replaced by it.
    ensure_plugins_initialized()
    register_communication_protocol_factory("cli", PhCliProtocol, override=True)
    _INSTALLED = True
