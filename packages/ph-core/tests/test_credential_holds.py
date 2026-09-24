"""T5 — nothing resumes without its credentials, and nothing stores one.

A route names the credential its adapter resolves at the edge, by name
(`ResolvedModel.credential`); `missing_credential` asks the credential seam whether
this deployment can supply it, without a value ever leaving the seam (I-3). A
session or child whose route's name is missing is held, and the hold is a pair of
journal records — `credential/needed`, then `credential/supplied` — written through
`record_wait`, which asks again on every open and appends nothing when nothing
changed. The subagent seam and the daemon build on these; `tests/test_mass_restart.py`
holds the whole of it after a real `SIGKILL`.
"""

from __future__ import annotations

import pytest

from ph.cordis import Context
from ph.keys import CREDENTIALS, INTENTS, LLM_FAKE, SESSIONS
from ph.llm.adapter import ResolvedModel
from ph.seams.credentials import credential_waits, missing_credential, record_wait
from ph.session import IntentError, Session
from ph.session.kinds import CREDENTIAL_WAIT, SESSION_HOLDER, SHELL_COMMAND
from ph.testing import MountProfile

pytestmark = pytest.mark.anyio

KEY = "PH_T5_PROBE_KEY"


@pytest.fixture(autouse=True)
def _no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY, raising=False)


async def _keyed(mount: MountProfile) -> Context:
    """A deployment whose fake route names `KEY`, which nothing supplies yet."""
    ctx = await mount()
    ctx.require(LLM_FAKE).route = ResolvedModel(credential=KEY)
    return ctx


def _types(session: Session) -> list[str]:
    return [event.type for event in session.events if event.type.startswith("credential/")]


async def test_a_route_names_its_credential_and_the_seam_says_whether_it_is_here(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name comes from the mounted adapter; the answer from the credential seam."""
    ctx = await _keyed(mount)

    assert missing_credential(ctx, "fake", "fake-1") == KEY
    monkeypatch.setenv(KEY, "from-the-environment")
    assert missing_credential(ctx, "fake", "fake-1") is None, "the environment supplies it"
    monkeypatch.delenv(KEY)
    ctx.require(CREDENTIALS).provide_value(KEY, "handed-over")
    assert missing_credential(ctx, "fake", "fake-1") is None, "so does a value handed over"


async def test_a_route_with_no_credential_or_no_adapter_is_missing_nothing(
    mount: MountProfile,
) -> None:
    """A route with no adapter fails for that reason, loudly, at its first request —
    calling it a missing key would send a person looking for the wrong thing."""
    ctx = await mount()

    assert missing_credential(ctx, "fake", "fake-1") is None, "the fake route needs none"
    assert missing_credential(ctx, "nobody", "m") is None, "no adapter to ask"


async def test_a_hold_is_recorded_once_however_often_it_is_asked(mount: MountProfile) -> None:
    """Every open asks again — a second restart before the key arrives included — and
    the log must not grow a record per ask."""
    ctx = await _keyed(mount)
    session = ctx.require(SESSIONS).create("held")

    for _ in range(3):
        await record_wait(ctx, session, "run-1", KEY)

    assert _types(session) == ["credential/needed"]
    assert credential_waits(session.events) == {"run-1": KEY}
    (needed,) = [event for event in session.events if event.type == "credential/needed"]
    assert dict(needed.data) == {"key": f"run-1:{KEY}", "holder": "run-1", "name": KEY}, (
        "the name and who waits, and never a value"
    )


async def test_a_hold_is_released_by_name_and_can_be_needed_again(mount: MountProfile) -> None:
    ctx = await _keyed(mount)
    session = ctx.require(SESSIONS).create("released")
    await record_wait(ctx, session, SESSION_HOLDER, KEY)

    await record_wait(ctx, session, SESSION_HOLDER, None)
    assert _types(session) == ["credential/needed", "credential/supplied"]
    assert credential_waits(session.events) == {}

    await record_wait(ctx, session, SESSION_HOLDER, KEY)
    assert _types(session)[-1] == "credential/needed", "a new wait, not the old one answered"
    assert credential_waits(session.events) == {SESSION_HOLDER: KEY}


async def test_a_route_that_stops_naming_a_credential_releases_its_old_hold(
    mount: MountProfile,
) -> None:
    """A profile that renamed the variable is asked about the new name: the old hold
    is settled rather than left waiting for a name nothing will ever supply."""
    ctx = await _keyed(mount)
    session = ctx.require(SESSIONS).create("renamed")
    await record_wait(ctx, session, "run-1", "PH_OLD_NAME")

    await record_wait(ctx, session, "run-1", KEY)

    assert credential_waits(session.events) == {"run-1": KEY}


async def test_holds_of_two_holders_are_their_own(mount: MountProfile) -> None:
    ctx = await _keyed(mount)
    session = ctx.require(SESSIONS).create("two")
    await record_wait(ctx, session, "run-1", KEY)
    await record_wait(ctx, session, "run-2", KEY)

    await record_wait(ctx, session, "run-1", None)

    assert credential_waits(session.events) == {"run-2": KEY}


async def test_claims_are_handed_back_only_for_a_kind_its_owner_settles(
    mount: MountProfile,
) -> None:
    """Any other kind's orphans are repair's, and its live intents the opener's —
    claiming one back from the log would let a second writer settle it."""
    ctx = await mount()
    journal = ctx.require(INTENTS)
    session = ctx.require(SESSIONS).create("claims")

    assert journal.held(session, CREDENTIAL_WAIT) == (), "nothing open"
    with pytest.raises(IntentError, match="shell/command"):
        journal.held(session, SHELL_COMMAND)
