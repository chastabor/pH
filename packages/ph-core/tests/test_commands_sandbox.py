"""P6-38 — `/sandbox`: the posture in force, changed without a restart.

An edit is applied first — `Mount.reconfigure` re-applies `sandbox-allow` with the
new config, releasing and refilling one slot on the seam — and kept second, as a
drop-in under `$PH_HOME/profiles/<name>.d/`. Both halves are asserted, and so is
the seam between them: the agent's next command is bounded by the new statement
while nothing else in the mount was touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ph.agent.types import AgentOptions
from ph.keys import AGENTS, COMMANDS, MOUNT, SANDBOX, SESSIONS, TUI_STATUS
from ph.paths import resolve_roots
from ph.seams.sandbox import DEFAULT_HOSTS, Denial
from ph.seams.tui_status import StatusReading
from ph.testing import MountProfile, not_none

pytestmark = pytest.mark.anyio


async def _run(ctx: Any, line: str, session: Any = None) -> str:  # noqa: ANN401
    shown = await ctx.require(COMMANDS).dispatch(line, session=session)
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
    assert "Not saved" in shown, "a test composition has no profile name to save under"
    assert dict(ctx.require(MOUNT).topology())["sandbox-allow"].endswith(", reconfigured live")


async def test_the_change_is_kept_as_a_drop_in_under_the_profile(mount: MountProfile) -> None:
    """The file pH owns, beside the one the person edits."""
    ctx = await mount()
    # A composition a person would run is named; the fixture's is not, so name it.
    ctx.require(MOUNT).profile.name = "headless"

    shown = await _run(ctx, "/sandbox allow host example.com")

    path = resolve_roots().profile_dropins("headless") / "sandbox.yaml"
    assert f"Saved to {path}" in shown
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Written by /sandbox.")
    (patch,) = yaml.safe_load(text)
    assert patch["id"] == "sandbox-allow"
    assert patch["config"]["network"]["hosts"][-1] == "example.com"
    assert patch["config"]["paths"] == []

    await _run(ctx, "/sandbox revoke host example.com")
    (patch,) = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "example.com" not in patch["config"]["network"]["hosts"], "rewritten, not appended"


async def test_a_host_is_normalised_the_same_way_in_both_directions(mount: MountProfile) -> None:
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


def _refusals(ctx: Any, session: Any) -> Any:  # noqa: ANN401
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
