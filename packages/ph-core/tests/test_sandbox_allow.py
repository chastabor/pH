"""P6-38 — `Allowances`: what a confined command may reach, said once and merged once.

The seam-level half of the network and directory allowlists. `sandbox-allow`
registers a value; `SandboxSeam.effective` merges it into every policy in
`confine`; the proxy asks `permits` per connection; and a refusal is a
`sandbox/denied` record in the agent's session. Everything here is asserted
against the seam with a stub backend, because what is being pinned is the *merge*
and the *record* — the kernel half is `test_sandbox_local.py`'s and the proxy's
is `test_sandbox_egress.py`'s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.seams.sandbox import (
    DEFAULT_HOSTS,
    DENIED,
    NOTHING,
    Allowances,
    Denial,
    DenialReader,
    Egress,
    NetworkAllowance,
    SandboxPolicy,
    SandboxSeam,
    host_allowed,
)
from ph.seams.sandbox_local import Bubblewrap, Seatbelt
from ph.testing import StubSandboxProvider, report_section

pytestmark = pytest.mark.anyio


def _allow(**config: Any) -> dict[str, Any]:
    return {"id": "sandbox-allow", "config": config}


def _seam(**kwargs: Any) -> SandboxSeam:
    seam = SandboxSeam(ctx=Context())
    seam.register_provider(StubSandboxProvider())
    if kwargs:
        seam.register_allowances(Allowances(**kwargs))
    return seam


# ------------------------------------------------------------- host_allowed --


def test_a_bare_host_is_exact_and_a_wildcard_is_its_subdomains_only() -> None:
    """Two spellings, neither implying the other; the apex has to be listed."""
    hosts = ["example.com", "*.pypi.org"]
    assert host_allowed(hosts, "example.com", 443)
    assert host_allowed(hosts, "EXAMPLE.COM.", 80), "case and a trailing dot are DNS's, not ours"
    assert not host_allowed(hosts, "www.example.com", 443), "a bare host does not cover children"
    assert host_allowed(hosts, "files.pypi.org", 443)
    assert host_allowed(hosts, "a.b.pypi.org", 443)
    assert not host_allowed(hosts, "pypi.org", 443), "a wildcard does not cover its apex"
    assert not host_allowed(hosts, "notpypi.org", 443)


def test_a_port_on_an_entry_pins_the_port() -> None:
    assert host_allowed(["example.com:443"], "example.com", 443)
    assert not host_allowed(["example.com:443"], "example.com", 22)
    assert host_allowed(["example.com"], "example.com", 22), "no port means any port"


def test_the_shipped_list_covers_the_apex_and_the_family_it_names() -> None:
    assert host_allowed(DEFAULT_HOSTS, "github.com", 443)
    assert host_allowed(DEFAULT_HOSTS, "raw.githubusercontent.com", 443)
    assert host_allowed(DEFAULT_HOSTS, "docs.rs", 443)
    assert not host_allowed(DEFAULT_HOSTS, "example.com", 443)


# --------------------------------------------------------------- effective --


def test_without_a_row_the_seam_fails_closed() -> None:
    """No `sandbox-allow` is `NOTHING`: no extra directory, no network, no door."""
    seam = _seam()
    seam.register_egress(Egress(socket="/s", port=1))
    effective = seam.effective(SandboxPolicy(mode="workspace-write", workspace_root="/w"))
    assert effective.network is False
    assert effective.egress is None
    assert effective.writable_extra is None
    assert not seam.permits("github.com", 443)
    assert NOTHING.network.mode == "off"


def test_allowed_directories_join_the_writable_set_when_they_exist(tmp_path: Path) -> None:
    """A missing directory is skipped — bwrap refuses to start over an absent bind
    source — and the skip is the closed direction."""
    present = tmp_path / "cache"
    present.mkdir()
    seam = _seam(paths=[str(present), str(tmp_path / "absent")])
    effective = seam.effective(SandboxPolicy(mode="workspace-write", workspace_root="/w"))
    assert effective.writable_extra == [str(present)]
    assert seam.allowed_paths() == (present,)


def test_read_only_takes_no_extra_directories(tmp_path: Path) -> None:
    """The session said nothing is writable; a deployment's cache is not an exception."""
    seam = _seam(paths=[str(tmp_path)])
    assert seam.effective(SandboxPolicy(mode="read-only")).writable_extra is None


def test_the_callers_own_extras_are_kept_and_not_duplicated(tmp_path: Path) -> None:
    seam = _seam(paths=[str(tmp_path)])
    policy = SandboxPolicy(
        mode="workspace-write", workspace_root="/w", writable_extra=[str(tmp_path)]
    )
    assert seam.effective(policy).writable_extra == [str(tmp_path)]


@pytest.mark.parametrize(
    ("mode", "refused", "network", "door"),
    [
        ("off", False, False, False),
        ("off", True, False, False),
        ("allowlist", False, False, True),
        ("allowlist", True, False, False),
        ("full", False, True, False),
        ("full", True, False, False),
    ],
)
def test_the_deployment_decides_and_a_caller_can_only_refuse(
    mode: str, refused: bool, network: bool, door: bool
) -> None:
    """§6.5 on the network axis: the deployment decides, and the only thing a caller
    can do is want less. The door opens for `allowlist` with a bridge, and never
    when the caller refused.

    There is deliberately no "ask for it" case to parametrize — a request can only
    narrow, which is why `SandboxPolicy` has `refuse_network` and a resolved
    `network` rather than one tri-state field whose third state took the same branch
    as its second.
    """
    seam = _seam(network=NetworkAllowance(mode=mode, hosts=["x"]))  # type: ignore[arg-type]
    seam.register_egress(Egress(socket="/px.sock", port=3128))
    effective = seam.effective(
        SandboxPolicy(mode="workspace-write", workspace_root="/w", refuse_network=refused),
        agent="a1",
    )
    assert effective.network is network
    assert (effective.egress is not None) is door
    if door:
        assert effective.egress is not None
        assert (effective.egress.socket, effective.egress.port, effective.egress.agent) == (
            "/px.sock",
            3128,
            "a1",
        )


def test_an_unmerged_policy_is_closed_at_the_backend() -> None:
    """`network` defaults to `False`, so a policy handed straight to a backend —
    as the backends' own tests do — has no network rather than the host's."""
    assert SandboxPolicy(mode="workspace-write", workspace_root="/w").network is False


def test_allowlist_without_a_bridge_is_no_network_and_says_so() -> None:
    seam = _seam(network=NetworkAllowance(mode="allowlist"))
    effective = seam.effective(SandboxPolicy(mode="workspace-write", workspace_root="/w"))
    assert effective.network is False and effective.egress is None
    assert "no egress bridge" in seam.network_posture()


def test_danger_full_access_means_the_network_too() -> None:
    seam = _seam(network=NetworkAllowance(mode="off"))
    assert seam.effective(SandboxPolicy(mode="danger-full-access")).network is True


def test_confine_stamps_the_effective_policy_on_what_it_returns() -> None:
    """A caller reading the result back learns what was bounded, not what it asked."""
    seam = _seam(network=NetworkAllowance(mode="full"))
    confined = seam.confine(("true",), SandboxPolicy(mode="workspace-write", workspace_root="/w"))
    assert confined.policy is not None
    assert confined.policy.network is True


def test_permits_reads_the_allowances_in_force_now() -> None:
    """Re-registering is how the row changes, and the proxy asks per connection."""
    seam = _seam(network=NetworkAllowance(mode="allowlist", hosts=["a.example"]))
    assert seam.permits("a.example", 443) and not seam.permits("b.example", 443)
    release = seam.allowances_by.owner.add_disposer(lambda: None) if seam.allowances_by else None
    assert release is not None
    # The row unloading: its slot is released, and the seam is closed again.
    seam.allowances = None
    assert not seam.permits("a.example", 443)
    seam.register_allowances(
        Allowances(network=NetworkAllowance(mode="allowlist", hosts=["b.example"]))
    )
    assert seam.permits("b.example", 443) and not seam.permits("a.example", 443)


# ------------------------------------------------------------------ denials --


def test_the_backend_reads_its_own_kernels_words_and_the_seam_asks_it() -> None:
    """The signatures are `Bubblewrap`'s, because they are facts about its platform;
    the seam delegates and a caller never learns which backend answered."""
    backend = Bubblewrap()
    output = "one\nsh: 1: cannot create /etc/x: Read-only file system\nthree\n"
    found = backend.read_denial(output, network=False)
    assert found is not None and found.kind == "filesystem" and found.via == "output"
    assert found.evidence == "sh: 1: cannot create /etc/x: Read-only file system"

    unreachable = "OSError: [Errno 101] Network is unreachable"
    assert backend.read_denial(unreachable, network=False) is not None
    assert backend.read_denial(unreachable, network=True) is None, "an outage is not ours"
    assert backend.read_denial("all good\n", network=False) is None

    # The earliest refusal in the output wins, whichever signature it matches.
    both = "Network is unreachable\nRead-only file system\n"
    first = backend.read_denial(both, network=False)
    assert first is not None and first.kind == "network"

    seam = SandboxSeam(ctx=Context())
    assert seam.read_denial(output, network=False) is None, "no backend, no reading"
    seam.register_provider(backend)
    assert seam.read_denial(output, network=False) == found
    # A backend that has not been measured reads nothing rather than guessing.
    assert not isinstance(Seatbelt(), DenialReader)


def test_a_denial_carries_the_sentence_with_the_way_out() -> None:
    proxy = Denial(kind="network", via="proxy", host="example.com", port=443).record("a1")
    assert proxy["message"] == (
        "Sandbox blocked network access to example.com:443. "
        "Allow it with /sandbox allow host example.com."
    )
    assert proxy["agent"] == "a1"
    written = Denial(kind="filesystem", via="output", evidence="e").record(None)
    assert "reported" in written["message"] and "/sandbox allow path" in written["message"]
    assert "agent" not in written


async def test_a_denial_lands_in_the_agents_session(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="f"))

    ctx.sandbox.record_denial(Denial(kind="network", via="proxy", host="h", port=1), agent=agent.id)

    (event,) = [one for one in session.events if one.type == DENIED]
    assert event.data["host"] == "h" and event.data["agent"] == agent.id
    assert event.ignorable, "accounting, not conversation"


async def test_a_denial_with_no_agent_to_charge_is_logged_not_lost(
    mount: Any, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = await mount()
    with caplog.at_level("WARNING", logger="ph.seams.sandbox"):
        ctx.sandbox.record_denial(Denial(kind="network", via="proxy", host="h", port=1), agent=None)
        ctx.sandbox.record_denial(
            Denial(kind="network", via="proxy", host="h", port=1), agent="nobody"
        )
    assert sum("no session to record it in" in record.message for record in caplog.records) == 2


# ---------------------------------------------------------------- the row --


async def test_the_row_registers_the_shipped_defaults_and_describes_them(mount: Any) -> None:
    ctx = await mount()
    assert ctx.sandbox.allowances == Allowances(paths=[], network=NetworkAllowance())
    rows = report_section(ctx, "Sandbox allowances")
    assert rows["network"].startswith("allowlist")
    assert "github.com" in rows["hosts"]
    assert rows["writable beyond the workspace"] == "none"


async def test_a_profile_replaces_the_whole_statement(mount: Any, tmp_path: Path) -> None:
    """The loader's patch rule: `config:` is one layer's, wholly."""
    ctx = await mount(
        _allow(paths=[str(tmp_path), "~/definitely-not-here-ph"], network={"mode": "off"})
    )
    assert ctx.sandbox.allowances == Allowances(
        paths=[str(tmp_path), "~/definitely-not-here-ph"],
        network=NetworkAllowance(mode="off", hosts=list(DEFAULT_HOSTS)),
    )
    rows = report_section(ctx, "Sandbox allowances")
    assert rows["network"] == "off — confined commands have no network"
    assert "(missing — not bound)" in rows["writable beyond the workspace"]
    assert str(tmp_path) in rows["writable beyond the workspace"]


async def test_unmounting_the_row_closes_the_seam_again(mount: Any) -> None:
    ctx = await mount()
    assert ctx.sandbox.allowances is not None
    await ctx.mount.forks["sandbox-allow"].dispose()
    assert ctx.sandbox.allowances is None
    assert not ctx.sandbox.permits("github.com", 443)


# ------------------------------------------------------------------ shell --


async def test_the_shell_records_what_the_kernel_refused_from_the_commands_words(
    mount: Any, tmp_path: Path
) -> None:
    """`ctx.shell` runs confined, reads the output, and appends the record for the
    agent it ran for. A stub backend, so what is pinned is the reading: the
    command is not actually confined and prints the kernel's sentence itself."""
    ctx = await mount()
    # `Bubblewrap` because the *reading* is its own (`DenialReader`); it confines
    # nothing here, since the command prints the kernel's sentence itself.
    ctx.sandbox.register_provider(Bubblewrap())
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="f"))
    # The lifecycle row acquires at the agent's first step; the shell confines only
    # an agent that has a workspace, so acquire it by hand as the ladder tests do.
    await ctx.workspace.acquire(session_id=session.id, agent_id=agent.id, base=tmp_path)

    result = await ctx.shell.run(
        "echo 'sh: 1: cannot create /etc/x: Read-only file system' >&2; exit 1", agent=agent
    )

    assert result.confined_by == "bwrap"
    (event,) = [one for one in session.events if one.type == DENIED]
    assert event.data["kind"] == "filesystem" and event.data["via"] == "output"
    assert event.data["agent"] == agent.id


async def test_the_shell_does_not_blame_the_sandbox_for_an_outage_under_full_network(
    mount: Any, tmp_path: Path
) -> None:
    ctx = await mount(_allow(network={"mode": "full"}))
    ctx.sandbox.register_provider(Bubblewrap())
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="f"))
    await ctx.workspace.acquire(session_id=session.id, agent_id=agent.id, base=tmp_path)

    await ctx.shell.run("echo 'Network is unreachable' >&2", agent=agent)

    assert not [one for one in session.events if one.type == DENIED]
