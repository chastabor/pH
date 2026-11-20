"""The only module that imports `utcp`. It makes the library run its commands through pH.

## The imports are guarded

`utcp` and each of its protocol modules call `logging.basicConfig(level=INFO)`
at import. If the root logger has no handler yet, that call installs a stderr
handler at INFO for the whole process. This happens in a test, and in a
`phern -p` run that has not configured logging. The process would get that
handler only because a row mounted a library. The imports below are therefore
guarded. After the imports, this module restores the handlers and the level of
the root logger. It also sets the library's own loggers to WARNING. The library
logs one INFO line for each call ("Executing shell script…"), and a harness log
does not need that line.

Every other module in this package takes the library's names from this module.
No second import site exists where the guard could be forgotten.

## A command runs where a pH command runs

The library's `cli` protocol starts `/bin/bash -c <script>` itself, with
`asyncio`, in the working directory of the client. Inside pH, that command would
run outside every row that limits a shell command. It would not run in the
agent's workspace or under the sandbox. The orphan journal would not record it,
and canceling the call would not stop it. `PhCliProtocol` keeps the way the
library builds the script, because the argument substitution holds the library's
security fixes. It replaces only the way the script is started. The script
starts through `ctx.shell.run_argv`, for the agent that the call is for.

The library's `call_tool` has no parameter for the agent. Its `caller` is the
client, and one client serves every agent of a mount. A `ContextVar` therefore
carries the agent. `calling()` sets the variable for the duration of one call.
Outside `calling()`, the protocol behaves exactly as the library's own protocol.
A plain `UtcpClient` in the same process, such as one in a test or in an
embedding application, is not affected when this row is imported.

## Failing closed

The override depends on two private methods of `utcp_cli`, and on taking
precedence in the library's registry. The code checks both and does not assume
them. `install` refuses to proceed when either method is missing from the class
that it overrides. `install` registers a *factory*, which the library resolves
before a shared instance of the same type. A `Call` records whether its command
went through `run_argv`. If a call returns and its command did not go through
`run_argv`, the command ran outside the sandbox. `require_ran` then raises an
error and does not give the answer to a model.

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

log = logging.getLogger("ph_utcp")

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
"""The `CliCommunicationProtocol` methods that `PhCliProtocol` replaces."""


@dataclass(slots=True)
class Call:
    """The context of one UTCP call: the agent that the call is for, how pH runs
    its command, and what the command did."""

    shell: ShellService
    root: Path
    """The tree that the call works on. This is the workspace of the agent, or
    the tree of the agent whose state it shares. The command runs here. A
    relative `working_dir` in a template is resolved under this directory."""
    timeout_ms: int
    """The time limit that pH sets for the command. It replaces the library's
    fixed limit."""
    env: Mapping[str, str]
    """The variables of this manual. They override every value from the template."""
    writable: tuple[str, ...] = ()
    """Paths outside the workspace that the commands of this manual may write."""
    agent: AgentHandle | None = None
    signal: Cancellation | None = None
    inherited: tuple[str, ...] = ()
    """The names of the variables that the template inherits, as the library read
    them. `run_argv` selects these variables from the scrubbed environment of the
    deployment, not from `os.environ`. A manual therefore cannot get a variable
    that pH removes by naming it."""
    ran: bool = False
    """Whether a command of this call went through `run_argv`."""


_CALL: ContextVar[Call | None] = ContextVar("ph_utcp_call", default=None)


@contextmanager
def calling(call: Call) -> Iterator[None]:
    """Run the UTCP calls in the body as `call`. One task makes one call. The
    library reads the variable in the same task that set it, a few frames deeper."""
    token = _CALL.set(call)
    try:
        yield
    finally:
        _CALL.reset(token)


def require_ran(call: Call, what: str) -> None:
    """Refuse an answer whose command did not go through `run_argv`.

    This check runs after the call, because nothing earlier can detect the
    problem. The library took a different path and ran the command itself,
    outside the sandbox. Two possible causes are a protocol registered over this
    one, and an overridden method that the library renamed. This function
    withholds the answer and raises an error that gives the reason. The model
    and the log both see that error."""
    if not call.ran:
        raise HarnessError(
            f"{what} was run by the UTCP library itself, outside pH's sandbox; its "
            "answer is withheld. The `cli` protocol pH installs was bypassed — check "
            "the installed utcp-cli against ph-utcp's pin.",
            UTCP_UNCONFINED,
        )


class PhCliProtocol(CliCommunicationProtocol):  # type: ignore[misc]
    """The library's `cli` protocol, changed so that `ctx.shell` starts its commands."""

    def _log_error(self, message: str) -> None:
        """Log the library's errors, except one message that is not an error.

        The library logs "Missing argument" at ERROR for each placeholder that the
        call gave no argument for. A model leaves out optional fields on every
        call, so the library logs this message on every call. The command reads
        the substituted placeholder as an absent value (`ph.toolcli`). Inside a pH
        call, the message is therefore logged at DEBUG."""
        if _CALL.get() is not None and message.startswith("Missing argument"):
            log.debug("ph_utcp: %s", message)
            return
        super()._log_error(message)

    def _prepare_environment(self, provider: Any) -> dict[str, str]:  # noqa: ANN401
        call = _CALL.get()
        if call is None:
            inherited: dict[str, str] = super()._prepare_environment(provider)
            return inherited
        # Pass only the names to `run_argv`. It selects those variables from the
        # scrubbed environment, below the workspace's redirection. Only the
        # template's own values travel with the script.
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
            # Do not use the platform's login shell. `bash -lc` reads the user's
            # profile, and the output of the profile would be parsed as the answer.
            ("/bin/bash", "-c", script),
            # Run in the tree that the call works on (`UtcpSeam._tree`). The
            # working directory is how a `ph.toolcli` command learns its root.
            cwd=call.root if working_dir is None else call.root / working_dir,
            agent=call.agent,
            # Use the time limit from pH, not the library's fixed 60 and 120
            # seconds. The row decides how long a tool may run.
            timeout_ms=call.timeout_ms,
            signal=call.signal,
            keep_env=call.inherited,
            # First the template's own values and the call's arguments (`env`),
            # then the manual's values. A value that the caller supplies overrides
            # an inherited value.
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
    """Make `PhCliProtocol` the `cli` protocol of the process. Safe to call again.

    This function registers a factory, which is how the library creates a
    protocol for each client. The library's resolver uses a factory before a
    shared instance of the same type. The library registers its own `cli`
    protocol as an instance, so that registration cannot take precedence. The
    registration is process-wide because the registry is process-wide. It is safe
    to leave in place for the life of the process, because outside `calling()`
    the protocol behaves as the library's own.

    This function refuses to proceed when a method that it overrides is no longer
    on the class. The override would then be dead code, and every command would
    run outside the sandbox.
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
    # Initialize the library's plugins first. The library's own registrations
    # have then happened, and the registration below replaces any `cli` factory
    # among them instead of being replaced by one.
    ensure_plugins_initialized()
    register_communication_protocol_factory("cli", PhCliProtocol, override=True)
    _INSTALLED = True
