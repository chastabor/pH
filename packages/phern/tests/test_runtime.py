"""`ph_app.runtime.mounted` — the teardown every mode shares.

The gate is *a mount unwinds even when the thing that asked for it stops
waiting*. Six front ends and the daemon reach the same `finally` here, so a
cancellation it does not survive is not one mode's bug: it is every lease,
worktree and child process in the process, left behind at once.

The composition is deliberately empty. What is under test is the order of two
calls and the protection each carries, and a profile with rows in it would only
add ways for this to fail for another reason.
"""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

import anyio
import pytest

from ph.cordis import Profile
from ph.cordis import context as context_module
from ph.testing import log_event
from ph_app.runtime import mounted

pytestmark = pytest.mark.anyio

EMPTY = Profile.from_documents([])


async def test_a_canceled_mount_still_disposes() -> None:
    """The unwind is not the drain's to lose.

    `Context.dispose` raises its own shield before its first await, so it
    survives being *entered* under a pending cancellation. This `finally` calls
    `drain` first, and until that shielded itself a Ctrl-C in print mode — or the
    daemon canceling a handler when its client's socket closed — raised out of
    the drain and the dispose line never ran at all.

    The detached listener is what makes the drain await: with the background set
    already empty the loop has no checkpoint, and the bug cannot be reached. A
    lease stands in for the artifact because that is the one this cost in
    practice — the daemon takes it through `open_session`, and a mount that
    skipped its disposal answered `session_already_active` for that id until the
    process was restarted.
    """
    released: list[str] = []
    settled: list[str] = []

    async def listener() -> None:
        await anyio.sleep(0.05)
        settled.append("listener")

    with anyio.move_on_after(0.02):
        async with mounted(EMPTY) as ctx:
            await ctx.effect(lambda: partial(released.append, "lease"), label="lease")
            ctx.detach(listener(), label="a listener that outlives the turn")
            await anyio.sleep(30)

    assert settled == ["listener"], "the drain still waited for what it is there to wait for"
    assert released == ["lease"], "and the unwind behind it ran"
    assert not ctx.active


async def test_a_listener_that_never_settles_does_not_hold_the_unwind() -> None:
    """The deadline, which is the price of the shield.

    Shielding the drain without bounding it would trade a skipped unwind for a
    shutdown that never finishes, which is the trade `daemon/server.py` refuses
    in so many words. The budget is `DRAIN_SECONDS`, derived from the unwind's
    own so the two cannot drift apart.
    """
    released: list[str] = []

    async def never() -> None:
        await anyio.sleep(30)

    started = anyio.current_time()
    with patch.object(context_module, "DRAIN_SECONDS", 0.02):
        async with mounted(EMPTY) as ctx:
            await ctx.effect(lambda: partial(released.append, "lease"), label="lease")
            ctx.detach(never(), label="a listener that will not settle")
    spent = anyio.current_time() - started

    assert released == ["lease"], "the unwind ran rather than waiting forever"
    assert spent < 1.0, f"the drain spent more than its budget: {spent:.3f}s"


async def test_opening_a_session_makes_its_teardown_durable() -> None:
    """F2, at the one door every host opens a session through.

    `open_session` takes the lease through the store's `claim`, which registers
    the mount's last write beside it: after every agent scope has unwound, before
    the lease is given back. The shipped `headless` profile, because the defect lives
    in the order its rows unwind — an empty composition cannot reach it.
    """
    from ph.keys import AGENTS, SESSIONS
    from ph.paths import resolve_roots
    from ph.persistence import read_session
    from ph.seams.workspace import DISPOSED
    from ph.testing import stored_log, workspace_disposed
    from ph_app.profiles import profile_or_exit
    from ph_app.runtime import open_session

    async with mounted(profile_or_exit("headless")) as ctx:
        session = await open_session(ctx, "teardown")
        agent = ctx.require(AGENTS).create(session)
        agent.ctx.add_disposer(
            lambda: log_event(session, *workspace_disposed(agent.id)), label="release"
        )
        await ctx.require(SESSIONS).flush(session)

    _header, events = read_session(stored_log(resolve_roots().sessions_dir(), "teardown"))
    assert events[-1].type == DISPOSED
