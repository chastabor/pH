"""P0-17 — the command line.

Gate: *a one-shot Q&A against the fake adapter writes an inspectable JSONL.*

"Inspectable" means readable by dsh tooling, not just by pH: the log is the
trace (§8), so the first thing the CLI has to earn is a file someone else can
read without a converter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ph.testing import stored_log
from ph_app.cli import app
from ph_app.profiles import resolve_profile

runner = CliRunner()

ReapedHost = Callable[..., Path]
"""The repo-root `reaped_host` fixture, spelled where it is read — structurally
rather than by `from conftest import …`, which resolves to this package's own
conftest rather than to the root one the fixture lives in."""


def test_dump_config_shows_the_composed_rows() -> None:
    result = runner.invoke(app, ["--dump-config", "--profile", "headless"])
    assert result.exit_code == 0, result.output
    rows = yaml.safe_load(result.stdout)
    ids = [row["id"] for row in rows]
    assert ids[0] == "llm"
    assert "llm-fake" in ids
    # Every row names the layer it came from, so a surprising value is
    # traceable to the file that set it.
    assert all(row["layer"].endswith(".yaml") for row in rows)
    fake = next(row for row in rows if row["id"] == "llm-fake")
    assert fake["config"]["providers"] == ["fake"]
    assert fake["layer"].endswith("headless.yaml")


def test_machine_readable_output_stays_parseable_under_force_color(monkeypatch: Any) -> None:
    """`--dump-config` and `ph events --json` are documents, not prose.

    Rich decides colour from the environment, and `FORCE_COLOR` is set by CI
    images and by plenty of shells — so both commands were emitting ANSI escapes
    into their own machine-readable output, and `yaml.safe_load` refused it with
    "unacceptable character #x001b". A person piping `ph --dump-config` into a
    parser got that, not a diagnosis.
    """
    monkeypatch.setenv("FORCE_COLOR", "3")
    dumped = runner.invoke(app, ["--dump-config", "--profile", "headless"])
    assert dumped.exit_code == 0, dumped.output
    assert "\x1b" not in dumped.stdout
    assert yaml.safe_load(dumped.stdout)

    matrix = runner.invoke(app, ["events", "--json"])
    assert matrix.exit_code == 0, matrix.output
    assert "\x1b" not in matrix.stdout
    assert json.loads(matrix.stdout)


def test_unknown_profile_is_refused() -> None:
    result = runner.invoke(app, ["--dump-config", "--profile", "nonesuch"])
    assert result.exit_code == 2
    assert "unknown profile" in result.output


def test_doctor_prints_three_roots(roots: Path) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    for name in ("PH_HOME", "PH_CACHE", "PH_RUNTIME"):
        assert name in result.stdout


def test_doctor_says_whether_the_daemon_socket_survives_logout(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """P5-11's static half, printed with the roots and before any mount.

    It needs no profile, and a profile that refuses to start is exactly when a
    person wants it — so it sits above the section that can fail rather than
    inside it. Printed on a good host too: rule 6 says to state what is not
    enforced next to where it would be assumed, and "`ph daemon` runs until I
    stop it" is assumed by everyone who never sees a warning.
    """
    reaped_host()
    markers = tmp_path / "linger"

    reaped = runner.invoke(app, ["doctor"])
    assert reaped.exit_code == 0, reaped.output
    assert "daemon socket lifetime" in reaped.stdout
    assert "logind removes" in reaped.stdout
    assert "loginctl enable-linger someone" in reaped.stdout

    (markers / "someone").touch()
    lingering_host = runner.invoke(app, ["doctor"])
    assert lingering_host.exit_code == 0, lingering_host.output
    assert "daemon socket lifetime" in lingering_host.stdout
    assert "loginctl" not in lingering_host.stdout, "nothing left to advise"


def test_starting_a_daemon_that_will_not_outlive_logout_says_so_first(
    tmp_path: Path, monkeypatch: Any, reaped_host: ReapedHost
) -> None:
    """The row's own wording: *names `enable-linger` when a daemon is configured
    without it* — and this is the moment it is being configured.

    Said here as well as in `ph doctor` because the two have different readers.
    Doctor is run by somebody already debugging; this line is read by somebody
    who is not, ten seconds before they close the terminal it was printed in.

    The bind is made to fail on `AF_UNIX`'s 107-byte path limit so the command
    returns instead of blocking — the same refusal `serve` names explicitly, and
    the only way to observe a startup notice from a process whose next act is to
    run forever.
    """
    reaped_host()
    # Still inside `$XDG_RUNTIME_DIR`, so still reaped — just deep enough that
    # the bind fails on the 107-byte limit instead of blocking forever.
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "xdg" / ("d" * 90) / ("e" * 90)))

    result = runner.invoke(app, ["daemon", "--profile", "headless"])

    assert result.exit_code == 1, result.output
    assert "does not survive logout" in result.output
    assert "loginctl enable-linger someone" in result.output
    # Before the bind, not after it: a daemon that failed to start for an
    # unrelated reason still told the person what would have happened if it had.
    assert result.output.index("enable-linger") < result.output.index("cannot listen")


def test_doctor_mounts_the_profile_and_prints_the_tier_table(roots: Path) -> None:
    """P4-12's own gate. Doctor answered from `resolve_roots()` alone until this
    row, so it could say where the log would go and nothing about what the
    process would be — and every question worth running it for is a row's."""

    result = runner.invoke(app, ["doctor", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "tier (effective)" in result.stdout
    # §4.8's third column, which is the one a tier name cannot be trusted to
    # convey on its own (E1).
    assert "does NOT bound" in result.stdout


def test_doctor_reports_the_live_topology_not_only_the_composition(roots: Path) -> None:
    """dsh's rule, which pH had only half of: the dump must show what *is* running.

    `--dump-config` prints the composed rows before anything runs, and says so.
    But a row that mounted and never activated — an unmet `inject` key — looks
    identical there to one that runs. `Loader.inactive()` knew the difference
    and nothing called it, so a reader of the YAML had no way to learn which
    they had. `doctor` now ends with the loader's own account of the mount: per
    row, whether it activated, on what, and from which layer; then the isolated
    realms, which are none at doctor time and are said to be none rather than
    left as a missing line.
    """
    result = runner.invoke(app, ["doctor", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "Topology" in result.stdout
    assert "active · injects" in result.stdout
    # Provenance is the last two path components, because every bundle file is
    # `bundle.yaml` and its directory is the name that tells them apart.
    assert "bundles/base.yaml" in result.stdout
    assert "an agent's scope is created when it runs" in result.stdout


def test_a_row_contributes_a_reading_without_ph_app_importing_it(
    tmp_path: Path, roots: Path
) -> None:
    """The other half of the gate: `permissions-fs` lives in ph-stabilize, which
    this package must never import (P3-20's rule, and the reason the reading is
    a seam rather than four `ctx.<name>` lookups doctor would have to know).

    With no rules configured, deliberately: a deployment that wrote nothing has
    a *wider* reach than one that wrote a deny list, and E9's sentence is most
    worth printing exactly there.
    """
    profile = tmp_path / "reach.yaml"
    profile.write_text(
        yaml.safe_dump(
            [
                {"id": "diagnostics", "name": "diagnostics"},
                {"id": "fs", "name": "fs-local", "config": {"root": str(tmp_path)}},
                {"id": "permissions-fs", "name": "permissions-fs"},
            ]
        )
    )

    result = runner.invoke(app, ["doctor", "--profile", str(profile)])

    assert result.exit_code == 0, result.output
    assert "File permissions" in result.stdout
    assert "not covered" in result.stdout


def test_doctor_reports_a_profile_that_refuses_to_start(tmp_path: Path, roots: Path) -> None:
    """E8's refusal reaches the person as a sentence, not a traceback: doctor is
    what someone runs *because* the process will not start, and the exit code
    still says it failed."""
    profile = tmp_path / "strict.yaml"
    profile.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "containment",
                    "name": "containment",
                    "config": {"tier": "sandbox", "strict": True},
                }
            ]
        )
    )

    result = runner.invoke(app, ["doctor", "--profile", str(profile)])

    assert result.exit_code == 1
    assert "no sandbox backend is mounted" in result.output


def test_doctor_refuses_an_unknown_profile_with_the_same_code(roots: Path) -> None:
    """Exit 2, as `--dump-config` gives, not the exit 1 a mount failure gives.

    Worth pinning because `typer.Exit` subclasses `RuntimeError`: resolved
    inside doctor's broad mount-failure catch, an unknown profile came out as
    "does not mount" under the wrong code, having already printed the right
    sentence.
    """

    result = runner.invoke(app, ["doctor", "--profile", "nonesuch"])

    assert result.exit_code == 2
    assert "unknown profile" in result.output
    assert "does not mount" not in result.output


def test_a_patch_from_the_command_line_composes_as_the_last_layer() -> None:
    """dsh's third layer — bundle, profile, *patch from the CLI* — which pH had
    only as a file.

    Same grammar as a profile document, on purpose: a second spelling for "change
    this row" is how a flag and a file come to accept different things. The
    provenance is what proves it composed rather than being applied some other
    way — `layer: cli`, printed beside the change, in the same dump every other
    layer appears in.
    """
    result = runner.invoke(
        app,
        ["--dump-config", "--profile", "headless", "--patch", "{id: llm-fake, disabled: true}"],
    )
    assert result.exit_code == 0, result.output
    rows = {row["id"]: row for row in yaml.safe_load(result.stdout)}
    assert rows["llm-fake"]["disabled"] is True
    assert rows["llm-fake"]["layer"] == "cli"
    # Untouched rows keep their own provenance: the cli layer is one more
    # document, not a rewrite of the composition.
    assert rows["llm"]["layer"].endswith("base.yaml")


def test_two_patches_compose_in_order_and_reach_the_live_topology() -> None:
    """Repeatable, and visible where it matters — in what the mount *became*.

    `isolate:` and `disabled:` are both patch verbs, so both are reachable from
    the flag; and `doctor` names `cli` as the layer that flipped a row, which is
    the answer to "why isn't X running" when the reason was typed a moment ago.
    """
    result = runner.invoke(
        app,
        [
            "doctor",
            "--profile",
            "headless",
            "--patch",
            "{id: skills-invariant, disabled: true}",
            "--patch",
            "{id: tools-invariant, disabled: true}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skills-invariant" in result.stdout and "disabled · by cli" in result.stdout
    assert "skill-reach-cache" not in result.stdout, "a disabled row's invariant was still reported"


def test_a_malformed_patch_is_refused_with_the_command_s_exit_code() -> None:
    """The refusal is the command's, under the same exit code an unknown profile gets.

    Three shapes a person can type that cannot mean anything: not YAML, a scalar
    where a mapping is needed, and — the one that matters — a code tag. The
    argument that a `!!js`-style value in a profile is refused at parse time is
    worth nothing if the command line is a way around it.
    """
    for bad, expect in (
        ("{id: llm-fake, disabled: [", "--patch"),
        ("just-a-word", "expected a mapping"),
        ("{id: llm-fake, config: !!python/object:os.system {}}", "--patch"),
        # The loader's own refusals, not the flag's: a row that does not exist
        # and an unknown key. A first draft checked two shapes in the CLI and let
        # these reach the loader uncaught, so `--dump-config` printed a traceback
        # for a typo in a row id.
        ("{id: nope, disabled: true}", 'no row with id "nope"'),
        ("{id: llm-fake, bogus: 1}", "unknown keys"),
    ):
        # Both commands, because they used to disagree: `doctor` composed inside
        # its broad mount-failure catch and reported the same bad row as
        # "profile does not mount" under exit 1. `profile_or_exit` composes
        # now, so the grammar is one refusal wherever it is met.
        for command in (["--dump-config"], ["doctor"]):
            result = runner.invoke(app, [*command, "--profile", "headless", "--patch", bad])
            assert result.exit_code == 2, (command, bad, result.output)
            assert expect in result.output, (command, bad, result.output)
            assert "does not mount" not in result.output, (command, bad)


def test_a_profile_that_will_not_parse_is_refused_before_anything_mounts(
    tmp_path: Path,
) -> None:
    """Every command, exit 2, the parser's sentence — and never "does not mount".

    `profile_or_exit` resolves *and reads*, so a mode is handed a profile or
    nothing. While it handed over paths, "that file is not YAML" surfaced at
    whichever compose point the mode reached: exit 2 for `ph -p`, which maps
    `LoaderError`, and — one step from the truth — `doctor`'s broad catch
    reporting a profile that *parsed* nowhere as one that refused to *start*.

    All three commands, because the point is that the refusal is the boundary's
    rather than each command's: a fourth mode inherits it without writing
    anything.
    """
    profile = tmp_path / "broken.yaml"
    profile.write_text("- id: fs\n  name: [\n", encoding="utf-8")

    for argv in (["doctor"], ["--dump-config"], ["--print", "hello"]):
        result = runner.invoke(app, [*argv, "--profile", str(profile)])

        assert result.exit_code == 2, (argv, result.output)
        assert "broken.yaml" in result.output, (argv, result.output)
        assert "does not mount" not in result.output, (argv, result.output)


def test_doctor_says_when_no_row_can_report(tmp_path: Path) -> None:
    """Rule 6, in the seam's place.

    Every section `doctor` prints after the mount arrives through
    `ctx.diagnostics` — Topology included, since it became a row. A profile that
    mounts no `diagnostics` row has nothing to report *through*, and a report
    that was simply empty would read as "nothing wrong", the one thing it cannot
    mean. The hand-appended Topology used to print regardless; this is what
    replaced that accident of construction with a sentence.
    """
    profile = tmp_path / "bare.yaml"
    profile.write_text(
        yaml.safe_dump([{"id": "fs", "name": "fs-local", "config": {"root": str(tmp_path)}}]),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", "--profile", str(profile)])

    assert result.exit_code == 0, result.output
    assert "`diagnostics` row" in result.output
    assert "Topology" not in result.output


def test_events_matrix_is_generated_from_the_registry() -> None:
    result = runner.invoke(app, ["events", "--json"])
    assert result.exit_code == 0, result.output
    matrix = json.loads(result.stdout)
    by_name = {row["name"]: row for row in matrix}
    assert by_name["agent/pre-step"]["mode"] == "waterfall"
    assert by_name["agent/pre-step"]["payload"] == "PreStepRequest"
    assert by_name["session/flush"]["mode"] == "parallel"
    assert by_name["session/event"]["mode"] == "emit"


def test_the_matrix_names_consumers_not_only_producers() -> None:
    """The half a producer/consumer matrix is named for, and it was empty.

    Two defects, one behind the other. The rendered table had no consumers
    column at all — `matrix()` carried the field, so `--json` looked complete.
    And the field itself was *always* empty: `note_consumer` guards on
    `if module`, `Context._module` was only ever set by `Context.scope`, and
    nothing in the mount path set it — so every scope inherited the root's empty
    string and no listener was ever recorded, in any profile. `ForkScope` now
    stamps the plugin's own `apply.__module__`.

    Consumers are also why this command mounts. A `declare` runs at import, so
    producers are knowable without one; `ctx.on` runs when a row *activates*, so
    consumers are a property of the profile and not of the code.

    `llm/stream` is asserted because it is the one every deployment listens to
    for a reason worth noticing — the I3 invariant prepends itself there — so an
    empty answer here means the mechanism is broken rather than the profile
    being quiet.
    """
    result = runner.invoke(app, ["events", "--json"])
    assert result.exit_code == 0, result.output
    by_name = {row["name"]: row for row in json.loads(result.stdout)}

    assert "ph.agent_loop.invariant" in by_name["llm/stream"]["consumers"], (
        "no listener was recorded, so the consumer half of the matrix is dead"
    )
    assert by_name["llm/stream"]["producer"] != "", "a producer went missing"
    assert "consumers" in runner.invoke(app, ["events", "--type", "llm"]).output


def test_events_refuses_an_unknown_profile_rather_than_answering_without_one() -> None:
    """The failure mode a broad `except` introduced, caught once and pinned here.

    Mounting is what fills the consumer half, and a profile that will not mount
    is reported rather than fatal — the declarations are still worth printing.
    But `profile_or_exit` reports an *unknown* profile by raising
    `typer.Exit`, which is an `Exception`, so guarding the resolve turned "no
    such profile" into a complete-looking matrix and exit 0: the answer that
    looks most like success, for the input most likely to be a typo. The resolve
    now happens outside the guard, and `doctor`'s exit code is the one to match.
    """
    result = runner.invoke(app, ["events", "--profile", "nosuchprofile"])
    assert result.exit_code == 2
    assert "unknown profile" in result.output


def test_the_config_catalog_is_generated_from_each_row_s_own_model() -> None:
    """P6-02's other half: a profile is rows *and* their config, and only the
    rows were enumerable.

    Generated from `PluginSpec.config_model`, so a field added to a row appears
    here without anybody remembering to write it down — the argument `ph events`
    makes about declarations, applied to configuration. The assertions below name
    a real row's real field, so a catalog that stopped reading the models would
    fail rather than print an empty shell.
    """
    result = runner.invoke(app, ["config", "--json"])
    assert result.exit_code == 0, result.output
    catalog = json.loads(result.stdout)
    by_name = {entry["name"]: entry for entry in catalog}

    assert not [entry for entry in catalog if "error" in entry], "a row failed to resolve"
    worktree = by_name["workspace-git-worktree"]
    assert worktree["injects"] == ["workspace", "subprocess"]
    ((option,),) = (worktree["config"],)
    assert (option["name"], option["type"], option["default"]) == ("root", "str | null", "None")
    assert "outside the repository on purpose" in option["doc"], (
        "the field's own prose was not read off the source"
    )
    # A row with no `Config` is a fact, not an absence: it is listed with an
    # empty option set rather than omitted, so "no options" and "not found"
    # stay distinguishable.
    assert by_name["diagnostics"]["config"] == []


def test_the_catalog_refuses_an_unknown_row_rather_than_printing_nothing() -> None:
    """An empty table answers "no such row" and "that row has no options"
    identically, and the person typing it meant one of them."""
    result = runner.invoke(app, ["config", "--row", "workspace-git-worktree", "--row", "nope"])
    assert result.exit_code == 2
    assert "nope" in result.output and "workspace-git-worktree" not in result.output


def test_print_mode_answers_and_writes_a_readable_log(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    result = runner.invoke(app, ["-p", "what is a session log?", "--session", "demo"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.stdout

    path = stored_log(tmp_path / "sessions", "demo")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["type"] == "session/header"
    assert records[0]["header"]["version"] == 0

    events = records[1:]
    # dsh's envelope, byte-for-byte: `{type, seq, time, data}` plus the optional
    # camelCase surface fields (D2, Q2).
    assert [event["seq"] for event in events] == list(range(len(events)))
    for event in events:
        assert set(event) <= {
            "type",
            "seq",
            "time",
            "data",
            "ignorable",
            "sourceEventSeqs",
            "surfaceOp",
        }
    user = next(e for e in events if e["type"] == "user/message")
    assert user["surfaceOp"] == "append"
    assert user["data"]["content"][0]["text"] == "what is a session log?"

    # And the harness can read its own log back into the same conversation.
    from ph.persistence.jsonl import read_session
    from ph.session import Session

    header, restored = read_session(path)
    assert header.id == "demo"
    session = Session("demo", seed=restored, header=header)
    assert [m.role for m in session.derive_messages()] == ["user", "assistant"]


def test_each_mode_is_reachable_from_the_command_line(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path))

    text = runner.invoke(app, ["-p", "hello"])
    assert text.exit_code == 0, text.output
    assert "ok" in text.stdout

    transcript = runner.invoke(app, ["-p", "hello", "--mode", "transcript"])
    assert transcript.exit_code == 0, transcript.output
    assert "you: hello" in transcript.stdout
    assert "pH: ok" in transcript.stdout

    stream = runner.invoke(app, ["-p", "hello", "--mode", "json"])
    assert stream.exit_code == 0, stream.output
    lines = [json.loads(line) for line in stream.stdout.splitlines() if line.startswith("{")]
    assert lines[0]["type"] == "session/header"
    # The log's own envelopes, not a rendering (I-7).
    assert any(event.get("type") == "turn/end" for event in lines)


def test_an_unknown_mode_is_refused() -> None:
    result = runner.invoke(app, ["-p", "hi", "--mode", "nonsense"])
    assert result.exit_code != 0


def test_bare_invocation_prints_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_profiles_resolve_to_bundle_documents() -> None:
    documents = resolve_profile("headless")
    assert [path.name for path in documents] == ["base.yaml", "headless.yaml"]
    with pytest.raises(ValueError):
        resolve_profile("nope")


def test_an_unknown_profile_names_what_is_available() -> None:
    """The error is a person's next step, so it lists what this install can
    actually compose — a bundle profile whose distribution is missing is not
    offered rather than offered and then failing (P3-20)."""
    result = runner.invoke(app, ["--dump-config", "--profile", "nonesuch"])
    assert result.exit_code != 0
    # The literal built-ins, not `available_profiles()` — an expectation built
    # from the function under test passes even when both are empty.
    for name in ("base", "headless", "tui", "deepseek", "anthropic"):
        assert name in result.output


def test_the_tui_profile_layers_onto_headless() -> None:
    documents = resolve_profile("tui")
    assert [path.name for path in documents] == ["base.yaml", "headless.yaml", "tui.yaml"]


def test_the_tui_profile_makes_the_workspace_writable() -> None:
    """The first of the rows that differ, and the reason a `tui` profile exists.

    `read-only` is right unattended — nothing can answer an approval prompt. In
    the TUI a person is present, so the workspace is writable and everything
    outside it still asks.
    """
    result = runner.invoke(app, ["--dump-config", "--profile", "tui"])
    assert result.exit_code == 0, result.output
    rows = yaml.safe_load(result.stdout)
    sandbox = next(row for row in rows if row["id"] == "sandbox")
    assert sandbox["config"]["defaultMode"] == "workspace-write"
    assert sandbox["layer"].endswith("tui.yaml")

    headless = runner.invoke(app, ["--dump-config", "--profile", "headless"])
    rows = yaml.safe_load(headless.stdout)
    unattended = next(row for row in rows if row["id"] == "sandbox")
    assert unattended["config"]["defaultMode"] == "read-only"


def test_only_a_profile_with_a_screen_offers_the_model_ask_user() -> None:
    """The same fact as above from the other side: somebody is there to *ask*.

    Read off `--dump-config` rather than off a mounted context, because the claim
    is about what a *deployment* composes: `ph -p --profile headless` must never
    put a question tool in the prompt, and the only thing standing between it and
    one is which layer last spoke about this row. The layer is asserted too — a
    row that came out enabled because somebody deleted it from `base.yaml` would
    satisfy the flag and lose the disarmed default everywhere else.

    Sabotage: drop the patch from `tui.yaml`, and an interactive session can no
    longer ask; drop `disabled: true` from `base.yaml`, and every headless run
    starts paying for a tool whose only possible answer is "nobody is there".
    """
    headless = yaml.safe_load(runner.invoke(app, ["--dump-config", "--profile", "headless"]).stdout)
    tui = yaml.safe_load(runner.invoke(app, ["--dump-config", "--profile", "tui"]).stdout)

    unattended = next(row for row in headless if row["id"] == "tool-ask-user")
    assert unattended["disabled"] is True
    assert unattended["layer"].endswith("base.yaml")
    attended = next(row for row in tui if row["id"] == "tool-ask-user")
    assert not attended.get("disabled")
    assert attended["layer"].endswith("tui.yaml"), "the profile with a modal is what arms it"


def test_tui_is_an_accepted_mode() -> None:
    """`--mode tui` is offered and takes no `--print`.

    Only the wiring is asserted here — the TUI itself is covered by the pilot
    and snapshot tests, which drive it without a terminal.
    """
    from ph_app.cli import _MODES

    help_text = runner.invoke(app, ["--help"]).output
    assert "tui" in help_text
    # The one-shot table is for modes that answer a prompt and exit; the TUI
    # has its own entry point because the prompt *is* the interface.
    assert "tui" not in _MODES


def test_passivate_after_accepts_minutes_or_off() -> None:
    """`--passivate-after` is minutes or `"off"`, and nothing else (P5-05).

    Refused rather than defaulted when it is neither: a typo in a duration is a
    daemon that silently keeps every root it ever started, and this is the one
    process where that goes unnoticed for a week.
    """
    import typer

    from ph_app.cli import _passivation

    assert _passivation("90") == 90 * 60.0
    assert _passivation("0.5") == 30.0
    assert _passivation("off") is None
    assert _passivation(" OFF ") is None

    for bad in ("ninety", "", "-5", "0"):
        with pytest.raises(typer.BadParameter):
            _passivation(bad)


def test_every_public_command_is_registered_and_nothing_private_leaked() -> None:
    """The command table is a fact about the module, not about its source order.

    P5-05 inserted a module-level helper directly beneath an `@app.command()`
    decorator, which silently rebound it: `ph daemon` stopped existing (exit 2)
    and a `-passivation` command appeared in `--help`. The whole suite stayed
    green, because nothing anywhere asserted that a command is registered — the
    CLI's commands were only ever invoked through `CliRunner`, which is a
    different question from whether they are reachable.

    Named explicitly rather than counted: a list that grows when a command is
    added is a list that notices when one disappears.
    """
    from ph_app.cli import app

    registered = {command.name or command.callback.__name__ for command in app.registered_commands}
    assert {"daemon", "doctor"} <= registered, f"a command stopped being registered: {registered}"
    private = {name for name in registered if name.startswith("_")}
    assert not private, f"a helper was captured by an @app.command() decorator: {private}"


# ------------------------------------------------ P6-33: `ph events --type` --


def test_events_filters_to_a_namespace_and_drills_into_one() -> None:
    """The bus half of the selector, through the command that prints it."""
    listed = runner.invoke(app, ["events", "--type", "tools", "--json"])
    assert listed.exit_code == 0, listed.output
    names = [row["name"] for row in json.loads(listed.stdout)]
    assert names and all(name.startswith("tools/") for name in names)

    one = runner.invoke(app, ["events", "--type", "tools/execute", "--json"])
    assert [row["name"] for row in json.loads(one.stdout)] == ["tools/execute"]


def test_events_does_not_answer_with_the_session_logs_near_miss() -> None:
    """`tool` is a *session-log* namespace; this registry holds `tools/*`. The two
    differ by one letter, which is exactly what a substring filter gets wrong —
    so the honest answer is "nothing here", not four `tool/*` rows."""
    result = runner.invoke(app, ["events", "--type", "tool"])
    assert result.exit_code == 2
    assert "no declared event matches" in result.output


def test_events_refuses_the_other_vocabulary_rather_than_answering_emptily() -> None:
    """A person asking a bus registry for log types has the wrong surface, and an
    empty table would let them conclude the namespace is quiet."""
    result = runner.invoke(app, ["events", "--type", "log:workspace"])
    assert result.exit_code == 2
    assert "does not serve" in result.output


def test_events_unfiltered_is_unchanged() -> None:
    """No `--type` is no filter — the default must not have become a query."""
    everything = json.loads(runner.invoke(app, ["events", "--json"]).stdout)
    assert len(everything) > 20
