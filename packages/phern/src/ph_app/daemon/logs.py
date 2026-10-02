"""The daemon's log file: the one place its warnings land once it is detached.

`launch._detach` starts the daemon with stdout and stderr on the null device, so a
Textual screen is not written over. Everything the daemon says after that went
with them: a root that gave up its retry ladder, a wake it could not make, an
invariant that drifted, the socket check failing. Python's last-resort handler
writes warnings to stderr only while no handler is configured, and nothing in pH
configured one. So `phern daemon` installs this before it serves.

Two handlers, each with one job. A rotating file under `$PH_HOME/logs` holds
`INFO` and up, bounded so a daemon left running for months does not fill a disk.
A stderr handler at `WARNING` keeps a foreground `phern daemon` saying what it
said before this module existed, which the last-resort handler stops doing the
moment a file handler is attached. Both are attached to the root logger, so a
seam's `ph.*` logger and the daemon's own are both heard; those two are the ones
raised to `INFO` (`LOUD`).

Idempotent by path: a second configuration of the same file adds nothing, so a
host that runs `serve` twice in one process (a test, an embedded supervisor)
does not write every line twice.

@module ph_app.daemon.logs
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = ["LOG_BYTES", "LOG_FILES", "LOUD", "configure_daemon_logging", "release_daemon_logging"]

LOG_BYTES = 5 * 1024 * 1024
"""One log file's ceiling before it rotates."""

LOG_FILES = 3
"""How many rotated files are kept beside the live one."""


class _DaemonLog(RotatingFileHandler):
    """The file handler this module owns, told apart from any other by type."""


class _DaemonStderr(logging.Handler):
    """Warnings to whatever `sys.stderr` is *when they are emitted*.

    Not a `StreamHandler` holding the stream it was built with: a host that swaps
    `sys.stderr` after this is configured — a CLI test runner, a redirection — would
    otherwise be written to through a stream it has since closed. A closed or
    missing stderr drops the line rather than raising out of `logging`.
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
"""The loggers raised to `INFO` for the file. The root logger keeps its `WARNING`, so
a library that logs every request at `INFO` (httpx does) is not written per call,
and the rotation keeps the warnings this file exists for."""


def _owned(handler: logging.Handler, path: Path) -> bool:
    """Whether `handler` is one `configure_daemon_logging(path)` attached."""
    return isinstance(handler, _DaemonStderr) or (
        isinstance(handler, _DaemonLog) and Path(handler.baseFilename) == path
    )


def configure_daemon_logging(path: Path, *, level: int = logging.INFO) -> None:
    """Send the process's logging to `path`, and warnings to stderr.

    The directory is made. pH's own loggers are lowered to `level` where higher or
    unset, and never raised; every other logger keeps the root's level.
    """
    root = logging.getLogger()
    if any(isinstance(handler, _DaemonLog) and _owned(handler, path) for handler in root.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    to_file = _DaemonLog(path, maxBytes=LOG_BYTES, backupCount=LOG_FILES, encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(to_file)
    if not any(isinstance(handler, _DaemonStderr) for handler in root.handlers):
        to_stderr = _DaemonStderr()
        to_stderr.setLevel(logging.WARNING)
        to_stderr.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        root.addHandler(to_stderr)
    for name in LOUD:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)


def release_daemon_logging(path: Path) -> None:
    """Detach and close what `configure_daemon_logging(path)` attached.

    For the command's own exit, so a host that runs it in-process — a test runner
    invoking `phern daemon` — is not left with a handler on an unlinked file for
    the rest of its life. Logger levels are left where they are.
    """
    root = logging.getLogger()
    for handler in [one for one in root.handlers if _owned(one, path)]:
        root.removeHandler(handler)
        handler.close()
