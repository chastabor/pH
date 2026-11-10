"""`utcp` — a manual's tools are pH tools, and their commands run where pH's do.

Gates: *a session calls a manual's tool and the command runs in the agent's
workspace; under a sandbox backend it cannot write outside that workspace and the
paths its manual names; a manual whose discovery fails leaves the mount up, and
says so in three places.*

## Why a provider written here

The row is generic over manuals, so it is tested against one that is nothing but
a manual: a script that prints a manual and answers three tools. It reports what
it *sees* — its working directory, the variables it was given, whether a write
landed — because those are this row's claims, and a tool that merely worked
would prove none of them.

## Why the fixture's manual is hand-written JSON

It is what a provider in any language would publish. Building it with
`ph.toolcli` would test the row against the one manual shape pH itself writes.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT, Context
from ph.keys import AGENTS, SANDBOX, SESSIONS, SYSTEM_PROMPT, TOOLS, WORKSPACE
from ph.llm.types import text_of
from ph.testing import FAKE_OPTIONS, MountProfile, report_section, run_tool
from ph_utcp import UTCP, Manual

pytestmark = pytest.mark.anyio

PROVIDER = r"""
import json, os, sys, time

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
    print(json.dumps(MANUAL))
elif sys.argv[2] == "echo":
    times = int(arg("times") or 1)
    print(json.dumps({
        "said": " ".join([arg("text")] * times),
        "cwd": os.getcwd(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("FX_")},
        "home": os.environ.get("HOME"),
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
elif sys.argv[2] == "slow":
    time.sleep(30)
    print("{}")
"""

_SEQ = itertools.count()


def _provider(tmp_path: Path) -> Path:
    """The provider script, outside the workspace: a tool is installed somewhere,
    and it is not in the tree it works on."""
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


def _agent(ctx: Context) -> Any:  # noqa: ANN401
    return ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create(f"utcp-{next(_SEQ)}"), FAKE_OPTIONS
    )


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
    assert ctx.require(UTCP).states["fx"].tools == ("fx_echo", "fx_write", "fx_wrapped", "fx_slow")


async def test_a_call_runs_in_the_agents_workspace(mount: MountProfile, tmp_path: Path) -> None:
    """**The gate.** The UTCP library would start the command in the harness's own
    working directory; a tool that indexes "the project" would index pH's."""
    ctx = await mount(_row(_manual(tmp_path)))

    result = await run_tool(ctx, "fx_echo", {"text": "hi", "times": 2}, agent=_agent(ctx))

    assert not result.is_error, text_of(result.content)
    assert result.value["said"] == "hi hi"
    assert Path(result.value["cwd"]) == tmp_path


async def test_arguments_are_held_to_the_manuals_schema_before_anything_runs(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A raw-schema tool owns its validation. Without it a wrong type reaches the
    shell as text and the tool answers whatever it makes of it."""
    ctx = await mount(_row(_manual(tmp_path)))
    agent = _agent(ctx)

    wrong = await run_tool(ctx, "fx_echo", {"text": "hi", "times": "many"}, agent=agent)
    missing = await run_tool(ctx, "fx_echo", {}, agent=agent)

    for result in (wrong, missing):
        assert result.is_error
        assert result.error is not None and result.error.info is not None
        assert result.error.info["code"] == "UTCP_BAD_ARGUMENTS"


async def test_the_command_sees_the_scrubbed_environment_narrowed_by_its_template(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template names what it inherits, and it inherits from what the
    *deployment* gives a child — so naming a credential does not fetch one."""
    monkeypatch.setenv("FX_VISIBLE", "seen")
    monkeypatch.setenv("FX_UNNAMED", "not asked for")
    monkeypatch.setenv("FX_API_KEY", "a credential")
    ctx = await mount(_row(_manual(tmp_path, env={"FX_FROM_ROW": "row"})))

    result = await run_tool(ctx, "fx_echo", {"text": "x"}, agent=_agent(ctx))

    assert result.value["env"] == {
        "FX_VISIBLE": "seen",
        "FX_FROM_TEMPLATE": "template",
        "FX_FROM_ROW": "row",
    }
    assert result.value["home"] is None, "`HOME` is not in this template's list"


async def test_an_answer_outside_the_declared_shape_is_the_error_it_usually_is(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The `cli` protocol ignores exit codes and returns stderr as the result, so a
    failed tool that declares an object answers with a sentence. The model needs
    the sentence."""
    ctx = await mount(_row(_manual(tmp_path)))

    result = await run_tool(
        ctx, "fx_write", {"path": str(tmp_path / "no" / "such" / "dir" / "f")}, agent=_agent(ctx)
    )

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "UTCP_BAD_ANSWER"
    assert "No such file or directory" in result.error.message


async def test_a_command_that_runs_past_its_budget_is_stopped_and_said(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_row(_manual(tmp_path, tools={"slow": {"timeout_ms": 500}})))

    result = await run_tool(ctx, "fx_slow", {}, agent=_agent(ctx))

    assert result.is_error
    assert "0.5s" in text_of(result.content) or "budget" in text_of(result.content)


# ------------------------------------------------------------- the envelope --


async def test_an_enveloped_tool_reads_as_it_would_registered_here(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The value is the envelope's value, the content is the tool's own rendering,
    and the tool takes its bare name."""
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {}, agent=_agent(ctx))

    assert not result.is_error, text_of(result.content)
    assert result.value == {"count": 3}
    assert text_of(result.content) == "three of them"


async def test_an_envelopes_failure_keeps_its_code(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {"mode": "fail"}, agent=_agent(ctx))

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "FX_REFUSED"
    assert result.error.message == "no"


async def test_a_tool_that_promised_an_envelope_and_sent_none_is_refused(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`envelope` is stated rather than detected, so a broken promise is an error
    and not a value that happens to lack an `ok`."""
    ctx = await mount(_row(_manual(tmp_path, envelope=True, names="bare")))

    result = await run_tool(ctx, "wrapped", {"mode": "bare"}, agent=_agent(ctx))

    assert result.is_error
    assert result.error is not None and result.error.info is not None
    assert result.error.info["code"] == "UTCP_BAD_ANSWER"


# -------------------------------------------------------------- degradation --


async def test_a_manual_that_cannot_be_discovered_leaves_the_mount_up(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** A provider that is down is a missing capability. The manual
    that answered still works, and the one that did not is named in the state,
    in `phern doctor` and in the prompt."""
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
    assert "4 tools" in report_section(ctx, "UTCP tools")["fx"]

    agent = _agent(ctx)
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
    """`cli` runs through pH's doors; anything else runs inside the harness
    process. Allowing that is a line a deployment writes."""
    remote = {"template": {"name": "remote", "call_template_type": "http", "url": "http://x"}}
    ctx = await mount(_row(remote))

    state = ctx.require(UTCP).states["remote"]

    assert not state.available
    assert '"http" protocol' in state.problems[0]


async def test_another_row_can_add_a_manual_and_takes_it_back_when_it_unmounts(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A package that publishes its own manual adds it from a row of its own. The
    manual is that row's: its tools and its state leave with the row's scope, and
    the name can then be added again."""
    ctx = await mount(_row())
    seam = ctx.require(UTCP)
    row = ctx.scope("a-row-with-a-manual")

    first = await seam.add(Manual.model_validate(_manual(tmp_path)), scope=row)
    assert first.available
    assert "fx_echo" in ctx.require(TOOLS).names(scope=row)

    # Held already: nothing is discovered twice, and the state is the one there is.
    assert await seam.add(Manual.model_validate(_manual(tmp_path)), scope=row) is first

    await row.dispose()
    assert "fx" not in seam.states
    assert "fx_echo" not in ctx.require(TOOLS).names(scope=DEPLOYMENT)

    again = await seam.add(Manual.model_validate(_manual(tmp_path)))
    assert again.available


# ----------------------------------------------------------------- confined --


@pytest.mark.parametrize(
    ("mode", "workspace_is_writable"), [("workspace-write", True), ("read-only", False)]
)
async def test_under_a_sandbox_a_command_writes_only_what_the_session_and_its_manual_allow(
    mount: MountProfile, tmp_path: Path, mode: str, workspace_is_writable: bool
) -> None:
    """**The gate.** A UTCP tool is bounded as `bash` is: the workspace when the
    session's posture allows writing it, and never a path nobody named.

    `writable` is how a tool that keeps state outside the workspace is let do so,
    and it holds in a read-only session too — the sandbox reads the extras under
    either mode. That is what lets an index under `$PH_CACHE` be built for an
    agent that may not touch the tree it is indexing.
    """
    granted = tmp_path.parent / f"{tmp_path.name}-granted"
    refused = tmp_path.parent / f"{tmp_path.name}-refused"
    granted.mkdir()
    refused.mkdir()
    ctx = await mount(
        {"id": "sandbox-local", "disabled": False},
        {"id": "sandbox", "config": {"defaultMode": mode}},
        _row(_manual(tmp_path, writable=[str(granted)])),
    )
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    agent = _agent(ctx)
    # What the lifecycle row does when a loop starts an agent. Without a workspace
    # there is no boundary to enforce, and `ctx.shell` confines nothing.
    await ctx.require(WORKSPACE).acquire(
        session_id=agent.session.id, agent_id=agent.id, base=tmp_path
    )

    inside = await run_tool(ctx, "fx_write", {"path": str(tmp_path / "in.txt")}, agent=agent)
    named = await run_tool(ctx, "fx_write", {"path": str(granted / "in.txt")}, agent=agent)
    outside = await run_tool(ctx, "fx_write", {"path": str(refused / "in.txt")}, agent=agent)

    assert inside.is_error is not workspace_is_writable, text_of(inside.content)
    assert (tmp_path / "in.txt").exists() is workspace_is_writable
    assert not named.is_error, text_of(named.content)
    assert (granted / "in.txt").exists()
    assert outside.is_error
    assert not (refused / "in.txt").exists()


# ------------------------------------------------------------- the library --


def test_importing_the_row_leaves_the_process_logging_alone() -> None:
    """`utcp` calls `logging.basicConfig(level=INFO)` on import. In a process with
    no handler yet that installs one on stderr, for everything, on behalf of a
    library a row happened to mount. A fresh interpreter, because this one has
    handlers of pytest's."""
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
    """The protocol this row installs is process-wide, because the library's
    registry is. Outside one of this row's calls it is the library's own."""
    from ph_utcp._library import UtcpClient, install

    install()
    template = json.loads(json.dumps(_manual(tmp_path)["template"]))
    client = await UtcpClient.create(config={"manual_call_templates": [template]})

    answer = await client.call_tool("fx.echo", {"text": "plain"})

    assert answer["said"] == "plain"
    assert Path(answer["cwd"]) == Path.cwd()
