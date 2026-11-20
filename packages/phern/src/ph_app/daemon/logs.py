"""The log file of the daemon: the one place for its warnings after it is detached.

`launch._detach` starts the daemon with stdout on the null device and stderr on a
file that the launcher reads only until the socket answers. A Textual screen is
therefore not overwritten. Before this module existed, everything that the daemon
reported after that point was lost: a root that gave up its retry ladder, a wake
that it could not make, an invariant that drifted, a socket check that failed.
The last-resort handler of Python writes warnings to stderr only while no handler
is configured, and nothing in pH configured one. `phern daemon` therefore
installs this configuration before it serves.

There are two handlers, and each has one job:

- A rotating file under `$PH_HOME/logs` holds `INFO` and above. It is bounded, so
  a daemon that runs for months does not fill a disk.
- A stderr handler at `WARNING` keeps a foreground `phern daemon` printing what
  it printed before this module existed. The last-resort handler stops printing
  as soon as a file handler is attached. A detached daemon does not get the
  stderr handler (`to_stderr=False`): its stderr is the file of the launcher,
  which does not rotate, and the log already holds every warning.

Both handlers are attached to the root logger, so the `ph.*` logger of a seam and
the logger of the daemon itself are both recorded. Those two are the loggers that
are raised to `INFO` (`LOUD`).

The configuration is idempotent by path. A second configuration of the same file
adds nothing, so a host that runs `serve` twice in one process (a test, an
embedded supervisor) does not write every line twice.

@module ph_app.daemon.logs
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = ["LOG_BYTES", "LOG_FILES", "LOUD", "configure_daemon_logging", "release_daemon_logging"]

LOG_BYTES = 5 * 1024 * 1024
"""The maximum size of one log file before it rotates."""

LOG_FILES = 3
"""How many rotated files are kept beside the live one."""


class _DaemonLog(RotatingFileHandler):
    """The file handler that this module owns. Its type tells it apart from any
    other handler."""


class _DaemonStderr(logging.Handler):
    """Write warnings to whatever `sys.stderr` is *when they are emitted*.

    This is not a `StreamHandler` that holds the stream it was built with. A host
    can swap `sys.stderr` after this handler is configured (a CLI test runner, a
    redirection). A `StreamHandler` would then write through a stream that the
    host has since closed. When stderr is closed or missing, this handler drops
    the line and does not raise out of `logging`.
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = sys.stderr
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            stream.write(self.format(record) + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass


LOUD = ("ph", "ph_app")
"""The loggers that are raised to `INFO` for the file. The root logger keeps its
`WARNING` level. A library that logs every request at `INFO` (httpx does) is
therefore not written for each call, and the rotation keeps the warnings that
this file exists for."""


def _owned(handler: logging.Handler, path: Path) -> bool:
    """Whether `handler` is one that `configure_daemon_logging(path)` attached."""
    return isinstance(handler, _DaemonStderr) or (
        isinstance(handler, _DaemonLog) and Path(handler.baseFilename) == path
    )


def configure_daemon_logging(
    path: Path, *, level: int = logging.INFO, to_stderr: bool = True
) -> None:
    """Send the logging of the process to `path`, and warnings to stderr when `to_stderr`.

    The function creates the directory. It lowers the loggers of pH to `level`
    where their level is higher or unset, and never raises them. Every other
    logger keeps the level of the root logger.
    """
    root = logging.getLogger()
    if any(isinstance(handler, _DaemonLog) and _owned(handler, path) for handler in root.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    to_file = _DaemonLog(path, maxBytes=LOG_BYTES, backupCount=LOG_FILES, encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(to_file)
    if to_stderr and not any(isinstance(handler, _DaemonStderr) for handler in root.handlers):
        echo = _DaemonStderr()
        echo.setLevel(logging.WARNING)
        echo.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        root.addHandler(echo)
    for name in LOUD:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)


def release_daemon_logging(path: Path) -> None:
    """Detach and close what `configure_daemon_logging(path)` attached.

    The command calls this at its own exit. A host that runs the command
    in-process (a test runner that invokes `phern daemon`) is then not left with
    a handler on an unlinked file for the rest of its life. Logger levels stay
    as they are.
    """
    root = logging.getLogger()
    for handler in [one for one in root.handlers if _owned(one, path)]:
        root.removeHandler(handler)
        handler.close()
