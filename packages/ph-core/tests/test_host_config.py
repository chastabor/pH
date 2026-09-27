"""`$PH_HOME/daemon.yaml` — the host's configuration, and the directories it moves.

Decision 23 gives each kind of row one owner, and gives the host's owner the
locations the other two are kept in. What is pinned here is the half `ph-core`
owns: the file is read whole or refused whole, and once it moves a directory,
every reader of `PathRoots` agrees — the persistence rows' default and the
daemon's cold browse are the same call, which is what a row's own `root` could
never promise (`Supervisor.sessions_directory` says why).

How its `rows:` compose is `ph_app.profiles`' and is tested beside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ph.cordis import LoaderError
from ph.host import HostConfig, host_config_path, load_host_config
from ph.paths import canonical, resolve_roots
from ph.testing import write_host_config


def test_no_file_is_the_empty_configuration(tmp_path: Path) -> None:
    load_host_config.cache_clear()

    assert load_host_config(tmp_path) == HostConfig()
    roots = resolve_roots()
    assert roots.sessions_dir() == roots.home / "sessions"
    assert roots.profiles_dir() == roots.home / "profiles"


def test_the_paths_block_moves_both_directories(tmp_path: Path) -> None:
    """Relative under `$PH_HOME`, absolute as written, `~` expanded — and the
    overlay and drop-ins follow the profiles directory, because they are named
    from it rather than from the home."""
    elsewhere = tmp_path / "logs"
    write_host_config(f"paths:\n  sessions: {elsewhere}\n  profiles: mine\n")

    roots = resolve_roots()

    assert roots.sessions_dir() == canonical(elsewhere)
    assert roots.profiles_dir() == canonical(tmp_path / "mine")
    assert roots.profile_overlay("tui") == canonical(tmp_path / "mine") / "tui.yaml"
    assert roots.profile_dropins("tui") == canonical(tmp_path / "mine") / "tui.d"
    described = dict(roots.describe())
    assert described["sessions"].endswith("(daemon.yaml)")
    assert described["profiles"].endswith("(daemon.yaml)")


def test_the_rows_are_kept_raw_for_the_profile_grammar(tmp_path: Path) -> None:
    """Not parsed here: `compose_rows` owns that grammar, so this reader has no
    opinion about a row it would only be able to half-check."""
    write_host_config("rows:\n  - id: jobs\n    config: {concurrency: {subagent: 2}}\n")

    assert load_host_config(tmp_path).rows == [
        {"id": "jobs", "config": {"concurrency": {"subagent": 2}}}
    ]


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("- id: jobs\n", "expected a mapping"),
        ("telemetry: {}\n", "unknown keys ['telemetry']"),
        ("paths:\n  logs: /tmp/x\n", "paths:"),
        ("paths: [sessions]\n", "paths:"),
    ],
)
def test_a_malformed_file_is_refused_whole(tmp_path: Path, text: str, said: str) -> None:
    """Whole, not in part: a paths block that failed to parse and fell back to
    the default would have a daemon writing logs where nothing looks for them.
    Refused through `resolve_roots` too, since that is how every reader meets it.
    """
    write_host_config(text)

    with pytest.raises(LoaderError, match=f"daemon.yaml: .*{said[:12]}"):
        load_host_config(tmp_path)
    with pytest.raises(LoaderError):
        resolve_roots()


def test_it_is_read_once_per_process(tmp_path: Path) -> None:
    """An edit takes effect at the next start. A daemon that re-read it while
    running would put two roots' logs in two directories with nobody having
    restarted anything."""
    write_host_config("paths:\n  sessions: first\n")
    before = resolve_roots().sessions_dir()

    host_config_path(tmp_path).write_text("paths:\n  sessions: second\n", encoding="utf-8")

    assert resolve_roots().sessions_dir() == before == canonical(tmp_path / "first")
