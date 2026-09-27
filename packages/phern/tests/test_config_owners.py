"""Three owners, one kind each — the layers a person writes, composed (decision 23).

Every row declares what its settings shape (`Affects`), and each configuration a
person writes may set one kind: a session profile the environment, the daemon's
`daemon.yaml` the deployment, the TUI's `tui.json` the presentation. This file
pins the composition half: that each layer reaches the rows it owns, in the order
`ph_app.profiles` states, and that a row set in the wrong one is refused naming
the right one — before anything mounts.

The shipped layers are exempt, because a bundle defines rows of every kind and
the person's layers are written against it; `test_catalog.py` pins that every
shipped row declares its kind, and which.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ph.paths import resolve_roots
from ph.testing import write_host_config
from ph_app.cli import app
from ph_app.profiles import available_profiles, profile_or_exit

runner = CliRunner()


def _overlay(name: str, text: str) -> Path:
    path = resolve_roots().profile_overlay(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _dump(*args: str) -> dict[str, Any]:
    result = runner.invoke(app, ["--dump-config", *args])
    assert result.exit_code == 0, result.output
    return {row["id"]: row for row in yaml.safe_load(result.stdout)}


# ---------------------------------------------------------------- the host --


def test_daemon_yaml_sets_the_host_s_rows(tmp_path: Path) -> None:
    """Its rows compose after pH's layers — so they can patch what those define —
    and the dump names the file as their provenance, which is how a person finds
    out why a bound is what it is."""
    path = write_host_config("rows:\n  - id: jobs\n    config: {concurrency: {subagent: 2}}\n")

    jobs = _dump("--profile", "headless")["jobs"]

    assert jobs["config"] == {"concurrency": {"subagent": 2}}
    assert jobs["layer"] == str(path)


def test_the_daemon_s_own_flag_wins_overwrite_host_config(tmp_path: Path) -> None:
    """`--max-concurrent-children` is a host flag written as the patch it is, so
    it sets deployment rows too, after the file — a flag typed at start is the
    later decision."""
    write_host_config("rows:\n  - id: jobs\n    config: {concurrency: {subagent: 2}}\n")

    rows = profile_or_exit(
        "headless", deployment=["{id: jobs, config: {concurrency: {subagent: 5}}}"]
    ).dump()

    jobs = next(row for row in rows if row["id"] == "jobs")
    assert jobs["config"] == {"concurrency": {"subagent": 5}} and jobs["layer"] == "cli"


def test_daemon_yaml_may_not_set_the_environment(tmp_path: Path) -> None:
    """The other direction of the same rule: a tool armed from the host's file
    would be part of every session's environment without being in any session's
    profile, so a restart comparing that profile would miss it."""
    write_host_config("rows:\n  - id: tool-bash\n    disabled: true\n")

    result = runner.invoke(app, ["--dump-config", "--profile", "headless"])

    assert result.exit_code == 2, result.output
    assert 'row "tool-bash" is environment, and this layer sets deployment rows only' in (
        result.output
    )
    assert "$PH_HOME/profiles/<name>.yaml" in result.output


# ------------------------------------------------------------ the session --


def test_a_session_profile_sets_the_environment(tmp_path: Path) -> None:
    _overlay("headless", "- id: tool-bash\n  disabled: true\n")

    assert _dump("--profile", "headless")["tool-bash"]["disabled"] is True


@pytest.mark.parametrize("where", ["overlay", "drop-in"])
def test_a_session_profile_may_not_set_the_host(tmp_path: Path, where: str) -> None:
    """The person's overlay and pH's drop-ins beside it are one owner's layers,
    and both are refused a host row by name — the drop-in included, although
    only `/sandbox` writes one today, because the rule is about the layer and
    not about who happened to write it."""
    patch = "- id: session-persistence\n  config: {root: /tmp/elsewhere}\n"
    if where == "overlay":
        written = _overlay("headless", patch)
    else:
        dropins = resolve_roots().profile_dropins("headless")
        dropins.mkdir(parents=True)
        written = dropins / "moved.yaml"
        written.write_text(patch, encoding="utf-8")

    result = runner.invoke(app, ["--dump-config", "--profile", "headless"])

    assert result.exit_code == 2, result.output
    assert f'{written}: row "session-persistence" is deployment' in result.output
    assert "daemon.yaml" in result.output


# ------------------------------------------------------------ presentation --


def test_every_named_profile_mounts_the_screens_pH_ships() -> None:
    """Presentation is not part of the environment a profile names, so no named
    profile decides whether it has a screen: all of them do, headless included,
    because the daemon holding a headless run is also what a TUI attaches to.
    Hiding one is `tui.json`'s (`test_tui_screens.py`)."""
    for name in available_profiles():
        rows = {row.id for row in profile_or_exit(name).enabled_rows()}
        assert "tui-screen-trajectory" in rows, f"{name} ships without the trajectory screen"


def test_a_session_profile_may_not_remove_a_screen(tmp_path: Path) -> None:
    """The refusal names `tui.json`, which is where the person's intent — not
    seeing the screen — can actually be said."""
    _overlay("tui", "- id: tui-screen-trajectory\n  remove: true\n")

    result = runner.invoke(app, ["--dump-config", "--profile", "tui"])

    assert result.exit_code == 2, result.output
    assert 'row "tui-screen-trajectory" is presentation' in result.output
    assert "tui.json" in result.output
