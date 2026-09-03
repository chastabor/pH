"""P7-10 — `!!<command>`: the person's own shell, logged and never sent.

Ported from tau with one narrowing: tau ships a pair — `!` splices the output
into the model's context, `!!` does not — and pH takes only the quiet one.

**The filter is the type, not a rule somebody remembered.** `_surface_op_of`
makes surface membership a property of the event type: an ineligible type may not
carry a `surfaceOp`, so `derive_messages` cannot see it. "Log it as a
`tool/result` and filter it out of the context" is not expressible — `SurfaceError`
refuses it — and would be wrong anyway, since a tool result with no `tool_use`
block to pair with is an orphan several providers reject.

**Everyone sees it, though.** The private-composer rule is about text you have
not sent; pressing enter on `!!ls` is an act in the session. So there is one
rendering path and not two, and `TuiState` stays entirely event-derived — which
is what lets the browser and the terminal share one fold.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import anyio
import pytest
from daemon_helpers import running, until

from ph.bundles import BASE, HEADLESS
from ph.cordis import Profile, load_profile_documents
from ph_app.protocol import DaemonError
from ph_app.shell import shell_body

pytestmark = pytest.mark.anyio


async def _run(client: Any, root: Any, command: str) -> dict[str, Any]:
    reply = await client.call("session/shell", sessionId=root.id, command=command)
    await until(
        lambda: root.session.latest("shell/result") is not None, what="the command to finish"
    )
    return dict(reply)


async def test_a_shell_command_is_logged_and_never_reaches_the_model(tmp_path: Any) -> None:
    """The claim the row exists for, asserted against the model's actual view.

    `derive_messages` is byte-identical across the command, so no amount of
    shell output can spend a token or change what the model is asked next.

    Sabotage: make `shell/result` surface-eligible and teach `derive_event_message`
    about it — the two message lists stop matching.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("quiet")
        client = await daemon.client()
        before = [one.to_wire() for one in root.session.derive_messages()]

        await _run(client, root, "echo hello-from-the-person")

        assert [one.to_wire() for one in root.session.derive_messages()] == before
        types = [one.type for one in root.session.events]
        assert types == ["shell/command", "shell/result"], "logged, in full, in order"


async def test_the_command_is_logged_before_it_runs(tmp_path: Any) -> None:
    """§5 rule 2, and the reason the two events are two.

    A `!!` that hangs — or that takes the daemon down with it — must still say in
    the log what was started, because that is exactly the command worth knowing
    about. So the ordering is observed **while the command is still running**:
    the child blocks on a file this test controls, and the log is read from
    outside it. Asserting the two events' relative order instead would pass for
    a daemon that appended both on completion, which is the sabotage.

    Sabotage: append both after `shell.run`, and the wait below times out.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("ordered")
        client = await daemon.client()
        release = tmp_path / "release"

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(
                partial(
                    client.call,
                    "session/shell",
                    sessionId=root.id,
                    command=f"while [ ! -f {release} ]; do sleep 0.01; done; echo released",
                )
            )
            await until(
                lambda: root.session.latest("shell/command") is not None,
                what="the command to be logged while it is still running",
            )

            assert root.session.latest("shell/result") is None, "it has not finished yet"
            release.touch()

        result = root.session.latest("shell/result")
        assert result is not None and "released" in result.data["stdout"]


async def test_the_output_and_the_exit_code_reach_the_log(tmp_path: Any) -> None:
    """What a person ran it to see. Both streams, one field, as a terminal shows."""
    async with running(tmp_path) as daemon:
        root = await daemon.root("noisy")
        client = await daemon.client()

        reply = await _run(client, root, "echo out; echo err >&2; exit 3")

        assert reply["exitCode"] == 3 and reply["ok"] is False
        result = root.session.latest("shell/result")
        assert result is not None
        # Apart on the log, because joining them is a presentation choice: a
        # front end that wants stderr in red can make one, and `shell_body` is
        # the shared default for one that wants a single column.
        assert result.data["stdout"].strip() == "out"
        assert result.data["stderr"].strip() == "err"
        assert result.data["ok"] is False and result.data["truncated"] is False
        assert result.data["cwd"], "the seam reports where the child actually ran"
        assert "[stderr]" in shell_body(result.data) and "[exit 3]" in shell_body(result.data)


async def test_a_shell_command_from_one_ui_appears_in_every_attached_one(
    tmp_path: Any,
) -> None:
    """One act, one rendering path — the multiplex rule applied to `!!`.

    The person who typed it reads it back off the same event as everybody else,
    which is what keeps `TuiState` event-derived and the two front ends sharing
    one fold.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []
        watcher = await daemon.client(
            on_notify=lambda method, params: (
                seen.append(params) if method == "session.event" else None
            )
        )
        root = await daemon.root("shared")
        await watcher.call("session/attach", sessionId=root.id)
        typist = await daemon.client()

        await _run(typist, root, "echo seen-by-both")

        await until(
            lambda: any(one["event"]["type"] == "shell/result" for one in seen),
            what="the result to reach the watcher",
        )
        relayed = [one["event"]["type"] for one in seen]
        assert relayed == ["shell/command", "shell/result"]


async def test_a_shell_child_never_inherits_a_credential(tmp_path: Any) -> None:
    """`!!env` prints no harness secret, and the mechanism is one, not two.

    `ctx.shell` passes `env=scrub_env(...)`, which drops every name matching
    `KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL` from the inherited environment —
    and pH reads a provider key through `ctx.credentials` rather than the child's
    environment, so it is not there to print. Scrubbing the *output* would be a
    second mechanism for a hazard the first one already closed.

    Pinned from the path `!!` actually takes, because that is the claim: not that
    `scrub_env` works, but that this route uses it.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("scrubbed")
        client = await daemon.client()

        await _run(client, root, "env")

        output = root.session.latest("shell/result").data["stdout"]  # type: ignore[union-attr]
        assert "ANTHROPIC_API_KEY" not in output
        assert "SECRET" not in output and "PASSWORD" not in output


async def test_an_empty_command_is_refused_rather_than_run(tmp_path: Any) -> None:
    """A bare `!!` is a typo, and running the shell with nothing is not what it meant."""
    async with running(tmp_path) as daemon:
        root = await daemon.root("empty")
        client = await daemon.client()

        with pytest.raises(DaemonError):
            await client.call("session/shell", sessionId=root.id, command="   ")

        assert [one.type for one in root.session.events] == []


async def test_a_result_cites_the_command_it_settles(tmp_path: Any) -> None:
    """The pair is joined by the log, not by "the one in flight".

    Every other pair in the fold correlates through an id in the event data —
    `tool/result`'s `callId`, `question/answered`'s `askId` — and a shell result
    does the same with `commandSeq`. An adapter field holding the in-flight card
    would be wrong here for a reason this session actively has: two attached UIs
    can each be running a command, and the second result would settle the first
    one's card or be dropped.

    Sabotage: drop `commandSeq` and settle "the most recent unsettled card".
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("cited")
        client = await daemon.client()

        await _run(client, root, "echo one")

        command = root.session.latest("shell/command")
        result = root.session.latest("shell/result")
        assert command is not None and result is not None
        assert result.data["commandSeq"] == command.seq


async def test_a_deployment_with_no_shell_refuses_before_claiming_the_key(
    tmp_path: Any,
) -> None:
    """The seam check belongs in `prepare`, which is what orders it before `once`.

    It used to run inside `act` — after the idempotence key was recorded — so a
    `!!` sent to a deployment with no shell burnt the retry the client still
    needed. That is exactly what the two halves of a `Mutation` are for.

    Sabotage: move `shell_of` back into `_act_shell`, and the second call comes
    back `repeated` instead of refusing again.
    """
    bare = Profile.from_documents(
        [*load_profile_documents([BASE, HEADLESS]), ("test", [{"id": "shell", "remove": True}])]
    )
    async with running(tmp_path, profile=bare) as daemon:
        root = await daemon.root("shell-less")
        client = await daemon.client()
        keyed = {"sessionId": root.id, "command": "echo hi", "clientId": "c", "commandId": "1"}

        for _ in range(2):
            with pytest.raises(DaemonError) as refused:
                await client.call("session/shell", **keyed)
            assert refused.value.reason == "seam_absent"

        assert [one.type for one in root.session.events] == [], "nothing was recorded"
