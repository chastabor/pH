"""P6-38 — `/sandbox`: the posture in force, changed without a restart.

An edit is an override of the session (session profiles, S4): recorded in the
session's log first, then applied — `Mount.reconfigure` re-applies `sandbox-allow`
with the new config, releasing and refilling one slot on the seam. Both halves are
asserted, and so is the seam between them: the agent's next command is bounded by
the new statement while nothing else in the mount was touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.json import as_obj, as_seq
from ph.keys import AGENTS, COMMANDS, MOUNT, SANDBOX, SESSIONS, TUI_STATUS
from ph.paths import resolve_roots
from ph.seams.sandbox import DEFAULT_HOSTS, Denial
from ph.seams.tui_status import StatusReading
from ph.session import Session
from ph.session_profile import overrides
from ph.testing import MountProfile, not_none

pytestmark = pytest.mark.anyio


async def _run(ctx: Context, line: str, session: Session | None = None) -> str:
    """Dispatch `line` in a session — the one the daemon always passes — so an edit
    has a log to be recorded in."""
    sessions = ctx.require(SESSIONS)
    here = session or sessions.get("sandboxed") or sessions.create("sandboxed")
    shown = await ctx.require(COMMANDS).dispatch(line, session=here)
    assert isinstance(shown, str)
    return shown


async def test_show_says_what_is_in_force(mount: MountProfile) -> None:
    ctx = await mount()
    shown = await _run(ctx, "/sandbox")
    assert "confinement: none" in shown, "the fixture turns the backend off"
    assert "network:" in shown
    assert "github.com" in shown
    assert "writable beyond the workspace: none" in shown
    assert "usage:" in shown


async def test_allow_host_is_live_before_it_is_kept(mount: MountProfile) -> None:
    """Applied, then kept — in that order, and the seam sees it at once."""
    ctx = await mount()
    assert not ctx.require(SANDBOX).permits("example.com", 443)

    shown = await _run(ctx, "/sandbox allow host example.com")

    assert "example.com is now reachable" in shown
    assert ctx.require(SANDBOX).permits("example.com", 443)
    assert ctx.require(SANDBOX).allowances is not None
    assert not_none(not_none(ctx.require(SANDBOX).allowances).network).hosts == [
        *DEFAULT_HOSTS,
        "example.com",
    ]
    assert "Kept in this session's log" in shown
    assert dict(ctx.require(MOUNT).topology())["sandbox-allow"].endswith(", reconfigured live")


async def test_the_change_is_kept_as_an_override_in_the_session_s_log(mount: MountProfile) -> None:
    """Kept where the session's other facts are, rather than beside the profile where
    every session on it would share it — and no drop-in is written any more."""
    ctx = await mount()
    ctx.require(MOUNT).profile.name = "headless"
    session = ctx.require(SESSIONS).create("kept")

    await _run(ctx, "/sandbox allow host example.com", session)
    await _run(ctx, "/sandbox revoke host example.com", session)

    changes = overrides(session)
    assert [change.command for change in changes] == [
        "/sandbox allow host example.com",
        "/sandbox revoke host example.com",
    ]
    last = changes[-1].entry
    assert last["id"] == "sandbox-allow"
    hosts = as_seq(as_obj(as_obj(last["config"])["network"])["hosts"])
    assert "example.com" not in hosts, "the whole config, each time"
    assert not resolve_roots().profile_dropins("headless").exists()


async def test_a_host_is_normalized_the_same_way_in_both_directions(mount: MountProfile) -> None:
    """Allow and revoke run the argument through one validator, so what a person
    typed means the same thing to both.

    They did not: `allow host` lowercased through `_valid_host` while `revoke host`
    compared the raw argument, so allowing `GitHub.com` stored `github.com` and
    revoking `GitHub.com` answered that it was never on the list.
    """
    ctx = await mount()
    assert "example.com is now reachable" in await _run(ctx, "/sandbox allow host Example.COM")
    assert ctx.require(SANDBOX).permits("example.com", 443)

    assert "example.com is no longer reachable" in await _run(
        ctx, "/sandbox revoke host Example.COM"
    )
    assert not ctx.require(SANDBOX).permits("example.com", 443)


async def test_allow_and_revoke_path(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount()
    cache = tmp_path / "cache"
    cache.mkdir()

    shown = await _run(ctx, f"/sandbox allow path {cache}")
    assert "is now writable beyond the workspace" in shown
    assert ctx.require(SANDBOX).allowed_paths() == (cache,)

    shown = await _run(ctx, f"/sandbox revoke path {cache}")
    assert "is no longer writable beyond the workspace" in shown
    assert ctx.require(SANDBOX).allowed_paths() == ()


async def test_network_mode_is_switched_by_name(mount: MountProfile) -> None:
    ctx = await mount()
    assert "network is now off" in await _run(ctx, "/sandbox network off")
    assert not ctx.require(SANDBOX).permits("github.com", 443)
    assert "network is now full" in await _run(ctx, "/sandbox network full")
    assert ctx.require(SANDBOX).permits("anything.example", 22)
    assert "network was already full" in await _run(ctx, "/sandbox network full")
    assert (await _run(ctx, "/sandbox network sideways")).startswith("usage:")


async def test_refusals_name_what_was_wrong(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount()
    assert "is not a host" in await _run(ctx, "/sandbox allow host https://example.com/x")
    assert "refusing to allow `/`" in await _run(ctx, "/sandbox allow path /")
    assert "must be an absolute path" in await _run(ctx, "/sandbox allow path relative/dir")
    assert "is not a directory" in await _run(ctx, f"/sandbox allow path {tmp_path / 'absent'}")
    assert ctx.require(SANDBOX).allowances is not None
    assert not_none(ctx.require(SANDBOX).allowances).paths == [], (
        "nothing was applied along the way"
    )


async def test_an_unchanged_edit_says_so_and_touches_nothing(mount: MountProfile) -> None:
    ctx = await mount()
    ctx.require(MOUNT).profile.name = "headless"
    shown = await _run(ctx, "/sandbox allow host github.com")
    assert shown == "github.com was already on the allowlist"
    assert not (resolve_roots().profile_dropins("headless") / "sandbox.yaml").exists()
    assert "sandbox-allow" not in ctx.require(MOUNT).reconfigured


async def test_a_profile_without_the_row_is_told_so(mount: MountProfile) -> None:
    ctx = await mount({"id": "sandbox-allow", "remove": True})
    shown = await _run(ctx, "/sandbox allow host example.com")
    assert "mounts no sandbox-allow row" in shown


def _refusals(ctx: Context, session: Session) -> Any:  # noqa: ANN401
    """This row's reading, by id — `sandbox` is the refusal count.

    The mode is `sandbox-mode`, contributed by the seam's own row: two facts
    about one seam, which is why each has to be named rather than taken as
    "the reading".
    """
    return next(
        (one for one in ctx.require(TUI_STATUS).readings(session) if one.id == "sandbox"), None
    )


async def test_the_footer_counts_refusals_and_show_lists_them(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, AgentOptions(provider="fake", model="f"))
    assert _refusals(ctx, session) is None, "nothing refused, nothing said"

    ctx.require(SANDBOX).record_denial(
        Denial(kind="network", via="proxy", host="h", port=443), agent=agent.id
    )
    ctx.require(SANDBOX).record_denial(
        Denial(kind="filesystem", via="output", evidence="e"), agent=agent.id
    )

    assert _refusals(ctx, session) == StatusReading(
        id="sandbox", text="sandbox: 2 refusals", level="warning"
    )
    shown = await _run(ctx, "/sandbox", session=session)
    assert "denied this session: 2" in shown
    assert "Sandbox blocked network access to h:443" in shown
