"""The daemon's log file (`ph_app.daemon.logs`).

A detached daemon's stderr is the null device, and until this module nothing in
pH configured a logging handler, so every warning the daemon raised after start
— a root giving up, a wake it could not make, an invariant drifting — went
nowhere. `test_daemon_launch` holds the end-to-end half (a spawned daemon writes
the file); this holds the handler's own rules.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from ph_app.daemon import logs
from ph_app.daemon.logs import configure_daemon_logging, release_daemon_logging


@pytest.fixture
def root_logger() -> Iterator[logging.Logger]:
    """The root logger as it was, put back after: handlers and levels are process
    state, and `LOUD`'s loggers are lowered too."""
    root = logging.getLogger()
    loud = [logging.getLogger(name) for name in logs.LOUD]
    before = (list(root.handlers), [one.level for one in loud])
    yield root
    root.handlers[:] = before[0]
    for one, level in zip(loud, before[1], strict=True):
        one.setLevel(level)


def test_a_warning_lands_in_the_file(tmp_path: Path, root_logger: logging.Logger) -> None:
    path = tmp_path / "logs" / "daemon.log"
    configure_daemon_logging(path)
    logging.getLogger("ph_app.daemon").warning("ph_app.daemon: root %s failed after %d", "r", 3)
    logging.getLogger("ph.seams.schedule").info("ph.seams.schedule: a seam speaks too")
    logging.getLogger("httpx").info("HTTP Request: POST https://example.invalid")
    written = path.read_text(encoding="utf-8")
    assert "WARNING [ph_app.daemon] ph_app.daemon: root r failed after 3" in written
    assert "INFO [ph.seams.schedule]" in written, "INFO and up from pH's own loggers"
    assert "HTTP Request" not in written, "and a library's INFO is not written per call"


def test_configuring_twice_adds_nothing(tmp_path: Path, root_logger: logging.Logger) -> None:
    """A host that runs `serve` twice in one process must not write every line twice.

    Sabotage: drop the by-path check, and the second call adds a second file handler.
    """
    path = tmp_path / "logs" / "daemon.log"
    configure_daemon_logging(path)
    added = len(root_logger.handlers)
    configure_daemon_logging(path)
    assert len(root_logger.handlers) == added


def test_the_file_is_bounded(tmp_path: Path, root_logger: logging.Logger) -> None:
    """A daemon left up for months must not fill a disk: the file rotates."""
    configure_daemon_logging(tmp_path / "daemon.log")
    # By type, not `FileHandler`: pytest's own log-file handler is one too.
    handler = next(h for h in root_logger.handlers if isinstance(h, RotatingFileHandler))
    assert handler.maxBytes == logs.LOG_BYTES
    assert handler.backupCount == logs.LOG_FILES


def test_warnings_reach_the_stderr_of_the_moment(
    tmp_path: Path, root_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    """A foreground `phern daemon` keeps saying its warnings on stderr, as the
    last-resort handler did before a file handler silenced it — and on whatever
    `sys.stderr` is when the line is emitted, not the one at configuration: a
    host that swaps or closes the stream after configuring must not be written
    to through the old one.

    Sabotage: make `_DaemonStderr` a `StreamHandler(sys.stderr)`, and the line
    goes to the stream capsys has since replaced.
    """
    configure_daemon_logging(tmp_path / "daemon.log")
    logging.getLogger("ph_app.daemon").warning("ph_app.daemon: said on stderr")
    logging.getLogger("ph_app.daemon").info("ph_app.daemon: not said on stderr")
    err = capsys.readouterr().err
    assert "WARNING [ph_app.daemon] ph_app.daemon: said on stderr" in err
    assert "not said" not in err


def test_release_takes_back_what_configure_attached(
    tmp_path: Path, root_logger: logging.Logger
) -> None:
    """The command's exit, for a host that ran it in-process: no handler is left on
    an unlinked file for the rest of that process's life."""
    before = list(root_logger.handlers)
    path = tmp_path / "daemon.log"
    configure_daemon_logging(path)
    assert len(root_logger.handlers) == len(before) + 2
    release_daemon_logging(path)
    assert root_logger.handlers == before
