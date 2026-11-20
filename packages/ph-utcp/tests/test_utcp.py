"""`utcp`: the tools of a manual are pH tools, and their commands run where pH's run.

Gates:

* A session calls a tool of a manual, and the command runs in the workspace of
  the agent.
* Under a sandbox backend, the command cannot write outside that workspace and
  the paths that its manual names.
* A manual whose discovery fails leaves the mount up, and the failure is
  reported in three places.

## Why the provider is written here

The row works with any manual. It is therefore tested against a provider that is
only a manual: a script that prints a manual and answers its tools. The script
reports what it *sees*: its working directory, the variables it was given, and
whether a write succeeded. Those are the claims of this row. A tool that only
worked would prove none of them.

## Why the manual of the fixture is hand-written JSON

Hand-written JSON is what a provider in any language would publish. A manual
built with `ph.toolcli` would test the row against only the one manual shape
that pH itself writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT
from ph.keys import SANDBOX, SYSTEM_PROMPT, TOOLS, WORKSPACE
from ph.llm.types import text_of
from ph.seams.workspace import scratch_of
from ph.testing import MountProfile, new_agent, report_section, run_tool, session_of
from ph.tools.definition import text_content
from ph_utcp import UTCP, Manual

pytestmark = pytest.mark.anyio

PROVIDER = r"""
import json, os, sys, tempfile, time

def arg(name):
    for one in sys.argv[3:]:
        if one.startswith(f"--{name}="):
            value = one.split("=", 1)[1]
            return None if value.startswith("MISSING_ARG_") else value
    return None

ME = [sys.executable, os.path.abspath(__file__)]

def template(tool, *names):
    flags = " ".join(f"--{n}=UTCP_ARG_{n}_UTCP_END" for n in names)
    return {
        "call_template_type": "cli",
        "commands": [{"command": f"{ME[0]} {ME[1]} call {tool} {flags}"}],
        "inherit_env_vars": ["PATH", "FX_VISIBLE", "FX_API_KEY"],
        "env_vars": {"FX_FROM_TEMPLATE": "template"},
    }

MANUAL = {
    "utcp_version": "1.2.0",
    "manual_version": "1.0.0",
    "tools": [
        {
            "name": "echo",
            "description": "Say what was said, and where from.",
            "inputs": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "times": {"type": "integer", "minimum": 1},
                },
                "required": ["text"],
            },
            "outputs": {
                "type": "object",
                "properties": {"said": {"type": "string"}, "cwd": {"type": "string"}},
                "required": ["said", "cwd"],
            },
            "tool_call_template": template("echo", "text", "times"),
        },
        {
            "name": "write",
            "description": "Write one file.",
            "inputs": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "outputs": {"type": "object", "properties": {"wrote": {"type": "string"}}},
            "tool_call_template": template("write", "path"),
        },
        {
            "name": "wrapped",
            "description": "Answer in pH's envelope.",
            "inputs": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
            },
            "outputs": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "value": {"$ref": "#/$defs/Count"},
                    "text": {"type": "string"},
                },
                "$defs": {
                    "Count": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                    }
                },
            },
            "tool_call_template": template("wrapped", "mode"),
        },
        {
            "name": "keep",
            "description": "Write one file of working state.",
            "inputs": {"type": "object", "properties": {}},
            "outputs": {"type": "object", "properties": {"kept": {"type": "string"}}},
            "tool_call_template": template("keep"),
        },
        {
            "name": "slow",
            "description": "Take a while.",
            "inputs": {"type": "object", "properties": {}},
            "outputs": {"type": "object"},
            "tool_call_template": template("slow"),
        },
    ],
}

verb = sys.argv[1]
if verb == "manual":
    # A test that names a record file learns what each run of `manual` was given.
    record = os.environ.get("FX_RECORD")
    if record:
        with open(record, "a") as handle:
            handle.write(os.environ.get("FX_API_KEY", "-") + "\n")
    print(json.dumps(MANUAL))
elif sys.argv[2] == "echo":
    times = int(arg("times") or 1)
    print(json.dumps({
        "said": " ".join([arg("text")] * times),
        "cwd": os.getcwd(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("FX_")},
        "home": os.environ.get("HOME"),
        "state": os.environ.get("PH_TOOL_STATE"),
    }))
elif sys.argv[2] == "write":
    with open(arg("path"), "w") as handle:
        handle.write("written")
    print(json.dumps({"wrote": arg("path")}))
elif sys.argv[2] == "wrapped":
    if arg("mode") == "fail":
        print(json.dumps({"ok": False, "error": {"code": "FX_REFUSED", "message": "no"}}))
    elif arg("mode") == "bare":
        print(json.dumps({"count": 3}))
    else:
        print(json.dumps({"ok": True, "value": {"count": 3}, "text": "three of them"}))
elif sys.argv[2] == "keep":
    state = os.environ["PH_TOOL_STATE"]
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, "kept.txt"), "w") as handle:
        handle.write("kept")
    with tempfile.NamedTemporaryFile() as scratch:
        made = scratch.name
    print(json.dumps({"kept": os.path.join(state, "kept.txt"), "temporary": made}))
elif sys.argv[2] == "slow":
    time.sleep(30)
    print("{}")
"""


def _provider(tmp_path: Path) -> Path:
    """Write the provider script outside the workspace. A tool is installed in
    some other place, not in the tree that it works on."""
    home = tmp_path.parent / f"{tmp_path.name}-provider"
    home.mkdir(exist_ok=True)
    script = home / "provider.py"
    script.write_text(PROVIDER, encoding="utf-8")
    return script


def _manual(tmp_path: Path, name: str = "fx", **settings: object) -> dict[str, Any]:
    return {
        "template": {
            "name": name,
            "call_template_type": "cli",
            "commands": [{"command": f"{sys.executable} {_provider(tmp_path)} manual"}],
        },
        **settings,
    }


def _row(*manuals: dict[str, Any], **config: object) -> dict[str, Any]:
    return {"id": "utcp", "name": "utcp", "config": {"manuals": list(manuals), **config}}


# ------------------------------------------------------------------ the tool --


async def test_a_manuals_tools_are_registered_with_the_manuals_own_schemas(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_row(_manual(tmp_path)))

    schemas = {one.name: one for one in ctx.require(TOOLS).schemas(scope=DEPLOYMENT)}

    assert {"fx_echo", "fx_write", "fx_wrapped", "fx_slow"} <= set(schemas)
    assert schemas["fx_echo"].description == "Say what was said, and where from."
    assert schemas["fx_echo"].parameters["required"] == ["text"]
    assert schemas["fx_echo"].parameters["properties"]["times"]["minimum"] == 1
    assert ctx.require(UTCP).states["fx"].tools == (
        "fx_echo",
        "fx_write",
        "fx_wrapped",
        "fx_keep",
        "fx_slow",
    )


async def test_a_call_runs_in_the_agents_workspace(mount: MountProfile, tmp_path: Path) -> None:
    """**The gate.** The UTCP library would start the command in the working
    directory of the harness. A tool that indexes "the project" would then index
    pH's own directory."""
    ctx = await mount(_row(_manual(tmp_path)))

    result = await run_tool(ctx, "fx_echo", {"text": "hi", "times": 2}, agent=new_agent(ctx))

    assert not result.is_error, text_of(result.content)
    assert result.value["said"] == "hi hi"
    assert Path(result.value["cwd"]) == tmp_path


async def test_arguments_are_held_to_the_manuals_schema_before_anything_runs(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A raw-schema tool is responsible for its own validation. Without
    validation, a value of the wrong type reaches the shell as text, and the tool
    answers with whatever it makes of that text."""
    ctx = await mount(_row(_manual(tmp_path)))
    agent = new_agent(ctx)

    wrong = await run_tool(ctx, "fx_echo", {"text": "hi", "times": "many"}, agent=agent)
    missing = await run_tool(ctx, "fx_echo", {}, agent=agent)

    for result in (wrong, missing):
        assert result.is_error
        assert result.error is not None and result.error.info is not None
        assert result.error.info["code"] == "UTCP_BAD_ARGUMENTS"


async def test_the_command_sees_the_scrubbed_environment_narrowed_by_its_template(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template names the variables that it inherits. It inherits them from the
    environment that the *deployment* gives a child process. Naming a credential
    therefore does not get one."""
    monkeypatch.setenv("FX_VISIBLE", "seen")
    monkeypatch.setenv("FX_UNNAMED", "not asked for")
    monkeypatch.setenv("FX_API_KEY", "a credential")
    ctx = await mount(_row(_manual(tmp_path, env={"FX_FROM_ROW": "row"})))

    result = await run_tool(ctx, "fx_echo", {"text": "x"}, agent=new_agent(ctx))

    assert result.value["env"] == {
        "FX_VISIBLE": "seen",
        "FX_FROM_TEMPLATE": "template",
        "FX_FROM_ROW": "row",
    }
    assert result.value["home"] is None, "`HOME` is not in this template's list"


async def test_a_passed_credential_reaches_every_command_of_its_manual(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pass_env` hands a variable that the scrub drops to the commands of one
    manual: its discovery, each tool call, and a command that a person runs. The
    first and the last run unconfined, and the docs of `pass_env` say so."""
    monkeypatch.setenv("FX_API_KEY", "a credential")
    record = tmp_path.parent / f"{tmp_path.name}-record.txt"
    manual = _manual(tmp_path, env={"FX_RECORD": str(record)}, pass_env=["FX_API_KEY"])
    ctx = await mount(_row(manual))
    agent = new_agent(ctx)

    called = await run_tool(ctx, "fx_echo", {"text": "x"}, agent=agent)
    ran = await ctx.require(UTCP).run_command(
        Manual.model_validate(manual),
        [sys.executable, str(_provider(tmp_path)), "manual"],
        agent,
        timeout_ms=20_000,
    )

    assert called.value["env"]["FX_API_KEY"] == "a credential", "the tool call"
    assert ran.exit_code == 0, ran
    assert record.read_text(encoding="utf-8").splitlines() == ["a credential"] * 2, (
        "discovery, then the command of a person"
    )


async def test_an_answer_outside_the_declared_shape_is_the_error_it_usually_is(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The `cli` protocol ignores exit codes and returns stderr as the result. A
    failed tool that declares an object therefore answers with a sentence. The
    model needs that sentence."""
    ctx = await mount(_row(_manual(tmp_path)))

    result = await run_tool(
        ctx, "fx_write", {"path": str(tmp_path / "no" / "such" / "dir" / "f")}, agent=new_agent(ctx)
    )

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "UTCP_BAD_ANSWER"
    assert "No such file or directory" in result.error.message


async def test_a_command_that_runs_past_its_budget_is_stopped_and_said(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_row(_manual(tmp_path, tools={"slow": {"timeout_ms": 500}})))

    result = await run_tool(ctx, "fx_slow", {}, agent=new_agent(ctx))

    assert result.is_error
    assert "0.5s" in text_of(result.content) or "budget" in text_of(result.content)


# ------------------------------------------------------------- the envelope --


async def test_an_enveloped_tool_reads_as_it_would_registered_here(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The value is the value of the envelope, the content is the tool's own
    rendering, and the tool has its bare name."""
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {}, agent=new_agent(ctx))

    assert not result.is_error, text_of(result.content)
    assert result.value == {"count": 3}
    assert text_of(result.content) == "three of them"


async def test_a_post_execute_row_edits_the_tools_own_text_and_its_edit_stands(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The text of the envelope is the content from the dispatch onward. A row
    that adds to the content, such as the receipt of ph-clm or the preview of
    offload, therefore works on what the model reads. When the text was applied
    after those rows instead, it replaced whatever they wrote."""
    from ph.tools import Accept

    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))
    seen: list[str] = []

    async def receipt(execution: Any, result: Any, next_: Any) -> Any:  # noqa: ANN401
        seen.append(text_of(result.content))
        return Accept(content=[*result.content, *text_content("[receipt]")])

    ctx.on("tools/post-execute", receipt)
    result = await run_tool(ctx, "wrapped", {}, agent=new_agent(ctx))

    assert seen == ["three of them"]
    assert text_of(result.content) == "three of them\n[receipt]"


async def test_an_envelopes_failure_keeps_its_code(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {"mode": "fail"}, agent=new_agent(ctx))

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "FX_REFUSED"
    assert result.error.message == "no"


async def test_a_tool_that_promised_an_envelope_and_sent_none_is_refused(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`envelope` is stated, not detected. A broken promise is therefore an error,
    and not a value that happens to have no `ok` key."""
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {"mode": "bare"}, agent=new_agent(ctx))

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "UTCP_BAD_ANSWER"


# -------------------------------------------------------------- degradation --


async def test_a_manual_that_cannot_be_discovered_leaves_the_mount_up(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** A provider that is down is a missing capability. The manual
    that answered still works. The manual that did not answer is named in the
    state, in `phern doctor`, and in the prompt."""
    broken = {
        "template": {
            "name": "gone",
            "call_template_type": "cli",
            "commands": [{"command": "no-such-utcp-provider-anywhere manual"}],
        }
    }
    ctx = await mount(_row(broken, _manual(tmp_path)))
    seam = ctx.require(UTCP)

    assert not seam.states["gone"].available
    assert seam.states["fx"].available
    assert "unavailable" in report_section(ctx, "UTCP tools")["gone"]
    assert "5 tools" in report_section(ctx, "UTCP tools")["fx"]

    agent = new_agent(ctx)
    result = await run_tool(ctx, "fx_echo", {"text": "still here"}, agent=agent)
    assert result.value["said"] == "still here"

    prompt = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    said = "\n".join(text for _, text in prompt.sections)
    assert "`gone`" in said
    assert "`fx`" not in said


async def test_nothing_is_said_to_the_model_when_every_manual_answered(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_row(_manual(tmp_path)))

    assert ctx.require(UTCP).unavailable() == ""


async def test_a_protocol_the_deployment_did_not_allow_is_not_run(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`cli` runs through the pH shell service and its sandbox. Every other
    protocol runs inside the harness process. A deployment allows that with a
    line in its config."""
    remote = {"template": {"name": "remote", "call_template_type": "http", "url": "http://x"}}
    ctx = await mount(_row(remote))

    state = ctx.require(UTCP).states["remote"]

    assert not state.available
    assert '"http" protocol' in state.problems[0]


async def test_another_row_can_add_a_manual_and_takes_it_back_when_it_unmounts(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A package that publishes its own manual adds it from its own row. The
    manual belongs to that row. Its tools and its state are removed with the
    scope of the row, and the name can then be added again."""
    ctx = await mount(_row())
    seam = ctx.require(UTCP)
    row = ctx.scope("a-row-with-a-manual")

    first = await seam.add(Manual.model_validate(_manual(tmp_path)), scope=row)
    assert first.available
    assert "fx_echo" in ctx.require(TOOLS).names(scope=row)

    # The name is already held. Nothing is discovered twice, and the add returns
    # the existing state.
    assert await seam.add(Manual.model_validate(_manual(tmp_path)), scope=row) is first

    await row.dispose()
    assert "fx" not in seam.states
    assert "fx_echo" not in ctx.require(TOOLS).names(scope=DEPLOYMENT)

    again = await seam.add(Manual.model_validate(_manual(tmp_path)))
    assert again.available


async def test_a_failed_add_takes_nothing_from_the_add_that_replaced_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A name whose add gave no tools can be added again, from any scope. The name
    then belongs to the second add only. Before the fix, the teardown of the
    first add stayed on its scope. When that scope ended, the teardown removed
    the client's record of the manual. The tools stayed listed, and every call of
    a tool failed."""
    ctx = await mount(_row())
    seam = ctx.require(UTCP)
    row = ctx.scope("a-row-whose-manual-fails")
    refused = _manual(tmp_path)
    refused["template"]["call_template_type"] = "http"

    first = await seam.add(Manual.model_validate(refused), scope=row)
    assert not first.available
    second = await seam.add(Manual.model_validate(_manual(tmp_path)))
    assert second.available
    assert not [one.label for one in row._effects if one.label == "utcp-manual(fx)"]

    await row.dispose()

    assert seam.states["fx"] is second
    result = await run_tool(ctx, "fx_echo", {"text": "hi", "times": 1}, agent=new_agent(ctx))
    assert not result.is_error, text_of(result.content)


async def test_a_forgotten_manual_is_warmed_again_when_it_comes_back(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Forgetting a manual cancels the runs that it started, so the marks that
    record those runs are removed too. A manual that is added again starts its
    eager run again. The take-back that would have removed the mark when its
    agent ended is dropped from the agent (Holdings rule 3). It is not left to
    run against a mark that no longer exists.

    Sabotage: pop the mark in `forget` without `drop()`, and its take-back stays.
    """
    ctx = await mount(_row())
    seam = ctx.require(UTCP)
    row = ctx.scope("a-row-with-a-manual")
    await seam.add(Manual.model_validate(_manual(tmp_path)), scope=row)
    agent = new_agent(ctx)
    seam._warm("fx", agent)
    seam._warm("another", agent)

    await row.dispose()

    assert set(seam.warmed) == {("another", agent.id)}
    taking_back = [one for one in agent.ctx._effects if one.label == "utcp-warmed"]
    assert len(taking_back) == 1, "the forgotten manual's take-back stayed on its agent"


# ----------------------------------------------------------------- confined --


@pytest.mark.parametrize(
    ("mode", "workspace_is_writable"), [("workspace-write", True), ("read-only", False)]
)
async def test_under_a_sandbox_a_command_writes_only_what_the_session_and_its_manual_allow(
    mount: MountProfile, tmp_path: Path, mode: str, workspace_is_writable: bool
) -> None:
    """**The gate.** A UTCP tool has the same limits as `bash`. It can write the
    workspace when the sandbox mode of the session allows that, and it can never
    write a path that nobody named.

    **A read-only session protects the workspace, not the working state of a
    tool.** The `state` directory of the manual is in the scratch directory of
    the agent, and `writable` names what agents share. Both stay writable in a
    read-only session, because the sandbox reads the extra paths of a policy in
    either mode. This lets a tool build an index for an agent that may not change
    the tree that the tool indexes.

    No grant makes the tree writable again. A grant inside the workspace is
    refused. The sandbox mode of the session then decides for that path, as it
    does for every other file in the tree.
    """
    granted = tmp_path.parent / f"{tmp_path.name}-granted"
    refused = tmp_path.parent / f"{tmp_path.name}-refused"
    inside_the_tree = tmp_path / "granted-inside"
    for directory in (granted, refused, inside_the_tree):
        directory.mkdir()
    ctx = await mount(
        {"id": "sandbox-local", "disabled": False},
        {"id": "sandbox", "config": {"defaultMode": mode}},
        _row(_manual(tmp_path, state="fx", writable=[str(granted), str(inside_the_tree)])),
    )
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    agent = new_agent(ctx)
    # Do what the lifecycle row does when a loop starts an agent. Without a
    # workspace there is no boundary to enforce, and `ctx.shell` confines nothing.
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=tmp_path
    )

    inside = await run_tool(ctx, "fx_write", {"path": str(tmp_path / "in.txt")}, agent=agent)
    named = await run_tool(ctx, "fx_write", {"path": str(granted / "in.txt")}, agent=agent)
    outside = await run_tool(ctx, "fx_write", {"path": str(refused / "in.txt")}, agent=agent)
    reopened = await run_tool(
        ctx, "fx_write", {"path": str(inside_the_tree / "in.txt")}, agent=agent
    )
    kept = await run_tool(ctx, "fx_keep", {}, agent=agent)

    assert inside.is_error is not workspace_is_writable, text_of(inside.content)
    assert (tmp_path / "in.txt").exists() is workspace_is_writable
    assert not named.is_error, text_of(named.content)
    assert (granted / "in.txt").exists()
    assert outside.is_error
    assert not (refused / "in.txt").exists()
    assert reopened.is_error is not workspace_is_writable, text_of(reopened.content)
    assert not kept.is_error, text_of(kept.content)
    scratch = scratch_of(ctx, session_of(agent).id, agent)
    assert Path(kept.value["kept"]).parent == scratch / "fx"
    assert Path(kept.value["kept"]).exists()
    # Its temporary files go in the scratch directory too. In a read-only session
    # they have no other place: `/tmp` is not writable, and this workspace has no
    # redirection of its own.
    assert Path(kept.value["temporary"]).parent == scratch


async def test_a_manuals_state_is_a_directory_of_the_calling_agents_scratch(
    mount: MountProfile, tmp_path: Path
) -> None:
    """There is one manual, and the calls of each agent get that agent's own
    directory. Two agents therefore never share state. The same agent gets the
    same directory again. "The same agent" means the same session id and agent
    id, which is what a restart gives it. A manual that names no state gets no
    directory."""
    ctx = await mount(_row(_manual(tmp_path, state="fx-state"), _manual(tmp_path, name="bare")))
    one, other = new_agent(ctx), new_agent(ctx)

    async def told(tool: str, agent: Any) -> str | None:  # noqa: ANN401
        answer = await run_tool(ctx, tool, {"text": "x"}, agent=agent)
        state: str | None = answer.value["state"]
        return state

    assert await told("fx_echo", one) == str(scratch_of(ctx, session_of(one).id, one) / "fx-state")
    assert await told("fx_echo", one) == await told("fx_echo", one)
    assert await told("fx_echo", other) != await told("fx_echo", one)
    assert await told("bare_echo", one) is None


async def test_a_grant_that_overlaps_the_workspace_is_refused_and_said(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A grant can be inside the tree, be the tree itself, or be a directory that
    contains the tree. Each would make the repository writable in a session that
    the user made read-only. The manual keeps its tools. The grant is left out,
    and `phern doctor` reports it."""
    apart = tmp_path.parent / f"{tmp_path.name}-apart"
    grants = [str(tmp_path / "sub"), str(tmp_path), str(tmp_path.parent), str(apart)]
    ctx = await mount(_row(_manual(tmp_path, writable=grants)))

    said = report_section(ctx, "UTCP tools")["fx"]

    assert said.startswith("5 tools")
    for refused in grants[:3]:
        assert f"{refused} cannot be granted as writable: it overlaps the workspace" in said
    assert str(apart) not in said
    assert apart.is_dir() and not (tmp_path / "sub").exists()


async def test_an_argument_left_out_is_not_logged_as_an_error(
    mount: MountProfile, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The library logs each placeholder that has no argument at ERROR. A model
    leaves out optional fields on every call, so the library logs this on every
    call."""
    ctx = await mount(_row(_manual(tmp_path)))

    with caplog.at_level("WARNING"):
        answer = await run_tool(ctx, "fx_echo", {"text": "x"}, agent=new_agent(ctx))

    assert not answer.is_error
    assert [one.getMessage() for one in caplog.records] == []


# ------------------------------------------------------------- the library --


def test_importing_the_row_leaves_the_process_logging_alone() -> None:
    """`utcp` calls `logging.basicConfig(level=INFO)` at import. In a process with
    no handler yet, that call installs a stderr handler for everything, only
    because a row mounted a library. The test uses a fresh interpreter, because
    this process already has pytest's handlers."""
    probe = (
        "import logging, ph_utcp; root = logging.getLogger(); "
        "print(len(root.handlers), root.level, logging.getLogger('utcp_cli').level)"
    )

    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True
    )

    assert done.stdout.split() == ["0", str(30), str(30)]
    assert done.stderr == ""


async def test_a_plain_client_in_the_same_process_is_not_affected(tmp_path: Path) -> None:
    """The protocol that this row installs is process-wide, because the library's
    registry is process-wide. Outside a call of this row, it behaves as the
    library's own protocol."""
    from ph_utcp._library import UtcpClient, install

    install()
    template = json.loads(json.dumps(_manual(tmp_path)["template"]))
    client = await UtcpClient.create(config={"manual_call_templates": [template]})

    answer = await client.call_tool("fx.echo", {"text": "plain"})

    assert answer["said"] == "plain"
    assert Path(answer["cwd"]) == Path.cwd()
