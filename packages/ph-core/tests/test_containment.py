"""P4-11 — choosing a rung, and refusing to pretend one is there (E1, E8).

Everything below the selector was built over five rows and none of it is *in
force* until something chooses. Two claims carry this file.

**Choosing is per acquire, and the interesting default is not uniform.** A person
running `rlm` works in the directory they opened, so the root agent stays where
they are; its *children* are the fan-out §4.8 opens with, so they get worktrees.
One knob would have forced the wrong answer on one of them.

**`strict` refuses to start, and `partial` is a refusal.** An operator who sets
it is saying "I do not want to run at all unless confinement is real" — so a
backend that bounds *some* of it is not a downgrade to accept quietly, because a
downgrade nobody notices is indistinguishable from the thing they were trying to
prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams.containment import ContainmentUnavailableError
from ph.testing import StubSandboxProvider, acquire_for_role

pytestmark = pytest.mark.anyio


def _row(**config: Any) -> dict[str, Any]:
    return {"id": "containment", "config": config}


# ------------------------------------------------------------------ choosing --


async def test_mounting_the_row_chooses_nothing(mount: Any, tmp_path: Path) -> None:
    """`None` is no opinion, and that is not the same as `advisory`.

    The row is in `ph-base`, so it is mounted in every profile — and mounting it
    must not opt a deployment out of a provider it deliberately layered. Only a
    profile that *names* `advisory` is saying "this agent stays put".
    """
    ctx = await mount()

    workspace = await acquire_for_role(ctx, tmp_path)

    assert ctx.containment.for_role(child=False) is None
    assert workspace.kind == "worktree", "mounting the selector disabled the tier"


async def test_advisory_is_a_choice_and_declines_the_provider(mount: Any, tmp_path: Path) -> None:
    """The person's own checkout, said out loud.

    A registered provider is *not* consulted — which is the whole difference
    between "nobody chose" and "this agent works where the person is".
    """
    ctx = await mount(_row(tier="advisory"))

    workspace = await acquire_for_role(ctx, tmp_path)

    assert workspace.kind == "shared"
    assert workspace.root == tmp_path


async def test_a_parent_and_its_children_sit_on_different_rungs(mount: Any, tmp_path: Path) -> None:
    """The shipped `rlm` posture, and the reason there are two knobs.

    The root agent stays in the directory the person opened; the children that
    would otherwise write it concurrently get their own trees. One knob would
    have had to be wrong for one of them.
    """
    ctx = await mount(_row(tier="advisory", child_tier="worktree"))

    root = await acquire_for_role(ctx, tmp_path)
    child = await acquire_for_role(ctx, tmp_path, child=True)

    assert root.kind == "shared"
    assert child.kind == "worktree"
    assert child.root != root.root


async def test_child_tier_follows_the_tier_when_unset(mount: Any, tmp_path: Path) -> None:
    """A profile that names one rung means it for everyone; the second knob is
    for the deployment that wants them to differ, not a thing every profile has
    to restate."""
    ctx = await mount(_row(tier="worktree"))

    assert ctx.containment.for_role(child=False) == "worktree"
    assert ctx.containment.for_role(child=True) == "worktree"


# -------------------------------------------------------------------- strict --


async def test_strict_refuses_when_no_backend_is_mounted(mount: Any) -> None:
    """E8's gate, and it fires at mount because that is what "refuse to start"
    means. The refusal names the backend *and* the other way out.

    A refusal that named only the problem would leave an operator guessing
    whether the answer is to install something or to change a setting — and one
    of those is a decision they may be entitled to make.
    """
    with pytest.raises(ContainmentUnavailableError) as refused:
        await mount(_row(tier="sandbox", strict=True))

    assert "no sandbox backend is mounted" in str(refused.value)
    assert "sandbox-local" in str(refused.value)
    assert "containment.strict" in str(refused.value)


async def test_strict_refuses_a_tier_that_enforces_nothing(mount: Any) -> None:
    """`worktree` bounds relative writes and nothing else (§4.8), so asking for
    strictness while configuring it is asking for a guarantee no rung below
    `sandbox` can give."""
    with pytest.raises(ContainmentUnavailableError) as refused:
        await mount(_row(tier="worktree", strict=True))

    assert "enforces nothing" in str(refused.value)


async def _strict_with(ctx: Any, backend: StubSandboxProvider | None) -> None:
    """Ask the question a backend mounted *after* this row would pose.

    Set on the live service rather than in config because that is the ordering
    the check exists for: a profile may layer its backend after the row that
    asked for confinement, and a verdict taken at either row's `apply` would be
    wrong for one of them.
    """
    if backend is not None:
        ctx.sandbox.register_provider(backend)
    ctx.containment.strict = True
    ctx.containment.verify()


async def test_strict_refuses_a_partial_backend(mount: Any) -> None:
    """The one E8 states explicitly: **`partial` is a refusal, not a
    downgrade.** A boundary that holds for some of it is the shape an operator
    would trust and should not."""
    ctx = await mount(_row(tier="sandbox"))

    with pytest.raises(ContainmentUnavailableError) as refused:
        await _strict_with(ctx, StubSandboxProvider(enforcement="partial"))

    assert "partial" in str(refused.value)
    assert "rather than a downgrade" in str(refused.value)


async def test_strict_is_satisfied_by_a_full_backend(mount: Any) -> None:
    ctx = await mount(_row(tier="sandbox"))

    await _strict_with(ctx, StubSandboxProvider(enforcement="full"))

    assert ctx.sandbox.enforcement == "full"


async def test_without_strict_a_missing_backend_is_not_a_refusal(mount: Any) -> None:
    """Configuring `sandbox` on a host that cannot provide it runs — and reports
    honestly that nothing is enforced. `strict` is the flag for a deployment
    that would rather not start than find that out later."""
    ctx = await mount(_row(tier="sandbox"))

    ctx.containment.verify()

    assert ctx.sandbox.enforcement is None


# ------------------------------------------------------------------- startup --


async def test_a_profile_that_cannot_honour_strict_does_not_start(mount: Any) -> None:
    """ "Refuse to start" has to mean the process, not the first unconfined call.

    Through the loader's own `profile/mounted` hook, so the refusal covers every
    way a profile is composed — the app's five modes, an embedder, and this
    suite — rather than one `if` in whichever caller happens to start it. That
    is also what lets a ph-core row test its own central claim without reaching
    up into `ph-app`.
    """
    with pytest.raises(ContainmentUnavailableError):
        await mount(_row(tier="sandbox", strict=True))
