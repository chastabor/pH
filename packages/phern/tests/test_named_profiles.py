"""Named profiles — `extends`, sparse rows, the round trip, and folding the old layers (S2).

A person's named profile is one file: the shipped profile it extends, and only the
rows that differ. The gate is decision 8's premise — a profile saved from a
composition composes back into the same rows, holding nothing that does not
differ — and item 0's: folding a profile's drop-ins and its old list-format file
into that one file changes nothing it composes, or changes nothing at all.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ph.cordis import LoaderError, Row
from ph.json import JsonValue
from ph.keys import NAMED_PROFILES
from ph.paths import resolve_roots
from ph.testing import write_host_config, write_profile
from ph_app.cli import app
from ph_app.profiles import (
    available_profiles,
    compose_profile,
    profile_or_exit,
    save_named_profile,
    unfolded_profiles,
)
from ph_app.profiles_cli import FoldRefused, fold_profile
from ph_app.runtime import mounted

runner = CliRunner()

TODAY = date(2026, 9, 26)


def _dropin(name: str, text: str) -> Path:
    directory = resolve_roots().profile_dropins(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sandbox.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _entries(rows: list[Row]) -> list[dict[str, JsonValue]]:
    return [row.to_entry() for row in rows]


ALLOW = "- id: sandbox-allow\n  config:\n    network: {mode: allowlist, hosts: [pypi.org]}\n"


# ------------------------------------------------------------- the format --


def test_a_named_profile_extends_a_shipped_one_with_only_what_differs() -> None:
    """A name of the person's own: offered beside the shipped ones, composed as the
    profile it extends and then its rows — which run."""
    write_profile("work", "extends: tui\nrows:\n  - id: tool-bash\n    disabled: true\n")

    work = {row.id: row for row in compose_profile("work").rows}
    tui = {row.id: row for row in compose_profile("tui").rows}

    assert "work" in available_profiles()
    assert work["tool-bash"].disabled and not tui["tool-bash"].disabled
    assert set(work) == set(tui), "it adds nothing the rows did not"
    assert runner.invoke(app, ["-p", "hi", "--profile", "work"]).exit_code == 0


@pytest.mark.parametrize(
    ("name", "text", "said"),
    [
        ("work", "rows: []\n", "`extends` names the shipped profile"),
        ("work", "extends: work\n", 'extends "work", which is not a shipped profile'),
        ("tui", "extends: headless\n", 'layers over it, so it extends "tui"'),
        ("work", "extends: tui\nmodel: fast\n", "unknown keys ['model']"),
        ("work", "- id: tool-bash\n  disabled: true\n", 'there is no shipped "work"'),
    ],
)
def test_a_named_profile_that_cannot_say_what_it_extends_is_refused(
    name: str, text: str, said: str
) -> None:
    """One level, and only over a shipped profile: a chain of person files would make
    "what does this profile default to" a question about files a person may not be
    looking at. Named after a shipped profile, it can only layer over that one."""
    write_profile(name, text)

    with pytest.raises(LoaderError, match=re.escape(said)):
        compose_profile(name)
    assert name not in available_profiles() or name == "tui", "offered and then refused"


def test_a_named_profile_sets_the_environment_only() -> None:
    """The S1 rule holds inside the new format: its rows are a session profile's."""
    write_profile("work", "extends: headless\nrows:\n  - id: session-persistence\n    config: {}\n")

    with pytest.raises(LoaderError, match="is deployment, and this layer sets environment"):
        compose_profile("work")


def test_a_profile_file_in_the_named_format_extends_too(tmp_path: Path) -> None:
    """`--profile ./x.yaml` in this format is a named profile that lives elsewhere;
    a list is still a whole composition, which is what scenario files are."""
    path = tmp_path / "elsewhere.yaml"
    path.write_text("extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n")

    rows = {row.id: row for row in compose_profile(str(path)).rows}

    assert rows["tool-bash"].disabled and "session-persistence" in rows


# --------------------------------------------------------- the round trip --


def test_a_saved_profile_composes_back_into_the_rows_it_was_saved_from() -> None:
    """The gate. A composition — a shipped profile with a person's patches — saved as a
    named profile, then composed by that name, gives back the same rows, and the
    saved file holds only the rows that differ."""
    composed = profile_or_exit(
        "tui",
        [
            "{id: tool-bash, disabled: true}",
            "{id: models, config: {default: fast, models: {fast: "
            "{provider: fake, model: fake-2}}}}",
        ],
    )

    path = save_named_profile("saved", composed, extends="tui", comment="Saved by a test.")
    reloaded = compose_profile("saved")

    assert _entries(reloaded.rows) == _entries(composed.rows)
    saved = yaml.safe_load(path.read_text())
    assert saved["extends"] == "tui"
    assert {entry["id"] for entry in saved["rows"]} == {"tool-bash", "models"}
    assert path.read_text().startswith("# Saved by a test.\n")


def test_a_host_row_is_in_neither_side_of_a_saved_profile() -> None:
    """`daemon.yaml`'s rows are in the composition being saved and in the base it is
    saved against, so they cancel: a saved profile is the environment a person
    chose, and a deployment row in it would be refused on reload."""
    write_host_config("rows:\n  - id: jobs\n    config: {concurrency: {subagent: 2}}\n")
    composed = profile_or_exit("headless", ["{id: tool-bash, disabled: true}"])

    path = save_named_profile("saved", composed, extends="headless", comment="")

    assert [entry["id"] for entry in yaml.safe_load(path.read_text())["rows"]] == ["tool-bash"]
    assert _entries(compose_profile("saved").rows) == _entries(composed.rows)


# ------------------------------------------------------------------ fold --


def test_folding_keeps_what_composes_and_the_comments_a_person_wrote() -> None:
    """A file in the old list format with a drop-in over it becomes one named file:
    the same rows compose, the person's comments are where they were, the drop-in's
    header comes with its row, and the directory is moved aside where nothing reads
    it. The doctor stops naming it."""
    write_profile("tui", "# my own tweaks\n- id: tool-bash\n  disabled: true   # no shell\n")
    _dropin("tui", "# Written by /sandbox.\n" + ALLOW)
    before = _entries(compose_profile("tui").rows)
    assert unfolded_profiles() == ["tui"]

    said = fold_profile("tui", today=TODAY)

    text = resolve_roots().profile_overlay("tui").read_text()
    assert _entries(compose_profile("tui").rows) == before
    assert text.startswith("extends: tui\nrows:\n")
    assert "# my own tweaks" in text and "# no shell" in text and "# Written by /sandbox." in text
    assert not resolve_roots().profile_dropins("tui").exists()
    assert (resolve_roots().profiles_dir() / "tui.d.folded-2026-09-26" / "sandbox.yaml").is_file()
    assert unfolded_profiles() == [] and "folded 1 drop-in" in said
    assert fold_profile("tui", today=TODAY) == "tui: nothing to fold"


def test_folding_with_no_file_writes_one_sparse() -> None:
    _dropin("headless", ALLOW)
    before = _entries(compose_profile("headless").rows)

    fold_profile("headless", today=TODAY)

    saved = yaml.safe_load(resolve_roots().profile_overlay("headless").read_text())
    assert saved["extends"] == "headless" and [one["id"] for one in saved["rows"]] == [
        "sandbox-allow"
    ]
    assert _entries(compose_profile("headless").rows) == before


def test_folding_appends_to_a_named_file_s_own_rows() -> None:
    write_profile("tui", "extends: tui  # mine\nrows:\n  - id: tool-bash\n    disabled: true\n")
    _dropin("tui", ALLOW)
    before = _entries(compose_profile("tui").rows)

    fold_profile("tui", today=TODAY)

    assert _entries(compose_profile("tui").rows) == before
    assert "extends: tui  # mine" in resolve_roots().profile_overlay("tui").read_text()


def test_a_fold_whose_file_no_longer_reads_changes_nothing() -> None:
    """Text appended to YAML a person wrote is a best effort — a flow list takes no
    appended item. Composing after is what makes that safe: the file and the
    drop-ins are put back, and the refusal names the file."""
    original = "extends: tui\nrows: [{id: tool-bash, disabled: true}]\n"
    write_profile("tui", original)
    dropin = _dropin("tui", ALLOW)
    before = _entries(compose_profile("tui").rows)

    with pytest.raises(FoldRefused, match="not folded"):
        fold_profile("tui", today=TODAY)

    assert resolve_roots().profile_overlay("tui").read_text() == original
    assert dropin.is_file(), "the drop-in was put back where it is read"
    assert _entries(compose_profile("tui").rows) == before


def test_a_fold_that_reads_back_differently_is_refused_by_the_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison itself, not the parse: a text edit that yields a valid file
    missing the drop-in's rows — the silent case — is caught by composing before and
    after, and named. Sabotage: skip `_moved`, and the drop-in is moved aside with
    its row never written anywhere."""
    original = "extends: tui\nrows:\n  - id: tool-bash\n    disabled: true\n"
    write_profile("tui", original)
    dropin = _dropin("tui", ALLOW)
    monkeypatch.setattr("ph_app.profiles_cli.append_rows", lambda text, *_args: text)

    with pytest.raises(FoldRefused, match="it would change sandbox-allow"):
        fold_profile("tui", today=TODAY)

    assert resolve_roots().profile_overlay("tui").read_text() == original
    assert dropin.is_file()


def test_the_doctor_names_a_profile_that_is_not_folded() -> None:
    _dropin("headless", ALLOW)

    result = runner.invoke(app, ["doctor", "--profile", "headless"], env={"COLUMNS": "200"})

    assert "not folded: headless" in result.output


# ------------------------------------------------------------------ show --


def test_show_prints_what_the_file_sets_and_full_prints_everything() -> None:
    """Sparse by default — what the person chose — and with `--full` every setting a
    session runs with, a row the file never names included, through its model."""
    write_profile("work", "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n")

    sparse = runner.invoke(app, ["profiles", "show", "work"])
    full = runner.invoke(app, ["profiles", "show", "work", "--full"])
    shipped = runner.invoke(app, ["profiles", "show", "headless"])

    assert sparse.exit_code == 0, sparse.output
    assert yaml.safe_load(sparse.stdout) == {
        "extends": "headless",
        "rows": [{"id": "tool-bash", "disabled": True}],
    }
    assert full.exit_code == 0, full.output
    rows = {row["id"]: row for row in yaml.safe_load(full.stdout)}
    assert rows["tool-bash"]["disabled"] is True
    assert rows["models"]["config"]["models"]["main"]["model"] == "fake-1", (
        "a default it never wrote"
    )
    assert "session-persistence" not in rows, "a session's profile, not the host's"
    assert "no file of yours" in shipped.stdout


@pytest.mark.anyio
async def test_every_mount_can_compose_a_named_profile_for_a_child() -> None:
    """A profile a parent assigns its child is composed by the host (S7b): `mounted`
    provides the store, so the seam can read a person's own named profile as the
    narrowing it is. Sabotage: drop the `provide` in `runtime.mounted`, and a spawn
    naming a profile is refused for want of one."""
    write_profile("reviewer", "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n")

    async with mounted(compose_profile("headless")) as ctx:
        reviewer = ctx.require(NAMED_PROFILES).compose("reviewer")

    rows = {row.id: row for row in reviewer.rows}
    assert rows["tool-bash"].disabled and reviewer.name == "reviewer"
