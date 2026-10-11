"""ph-clm Phase 3 — `rlm-clm`: the RLM, editing its own context.

**A repo-level scenario, not a package test**, for `test_rlm_stable_profile.py`'s
reason: it composes `ph-app`'s profile over `ph-rlm` and `ph-clm`, and ph-clm depends
on neither of the others.

What a composition can get wrong: that both of ph-clm's rows activate beside the RLM's
(an unmet `inject` key mounts a row that never runs), that the model is told about the
file, and that the file is written where the agent's own tools can reach it.
"""

from __future__ import annotations

import pytest

from ph.cordis import DEPLOYMENT, Profile
from ph.keys import AGENTS, LLM_FAKE, SESSIONS, TOOLS
from ph.testing import FAKE_OPTIONS, MountProfile
from ph_app.profiles import available_profiles, resolve_profile
from ph_clm import BUNDLE
from ph_clm.keys import CLM, CLM_MIRROR

pytestmark = pytest.mark.anyio

PROFILE = "rlm-clm"


def test_the_profile_is_offered_by_name() -> None:
    assert PROFILE in available_profiles()
    assert BUNDLE in resolve_profile(PROFILE)


def test_it_composes_the_rlm_and_the_context_rows() -> None:
    rows = {row.id for row in Profile.from_paths(resolve_profile(PROFILE)).rows}

    assert {"code-runtime-python", "clm-context", "clm-mirror"} <= rows


async def test_the_profile_boots_runs_a_turn_and_writes_the_file(mount: MountProfile) -> None:
    ctx = await mount(profile=resolve_profile(PROFILE))
    session = ctx.require(SESSIONS).create("clm")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await agent.prompt("hello")

    assert ctx.get(CLM) is not None and ctx.get(CLM_MIRROR) is not None, "a row never activated"
    assert ctx.require(TOOLS).get("context_tombstone", scope=DEPLOYMENT) is not None
    path = ctx.require(CLM_MIRROR).path(session, agent)
    assert path.read_text(encoding="utf-8").startswith(f"[[LIVE_CONTEXT session={session.id} ")
    first = next(one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request)
    assert str(path) in (first.system or ""), "the prompt does not name the file"
