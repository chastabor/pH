"""The harness bridge, with no terminal in the loop.

`frontend.py` claims to be testable against a `ModalHost` stub. This is the
test that makes the claim true: a plain object answers the seams, and the log
records the decisions exactly as it does under the real app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tui_helpers import StubHost

from ph.seams.user_questions import UserQuestion
from ph.testing import StubAgent, stored_log
from ph_app.profiles import compose_profile
from ph_app.tui.frontend import open_harness

pytestmark = pytest.mark.anyio


async def test_the_frontend_answers_both_seams_through_a_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    host = StubHost()
    front = await open_harness(
        compose_profile("headless"),
        host=host,
        provider="fake",
        model="fake-1",
        session_id="stub",
    )
    try:
        outcome = await front.ctx.approval.request(
            agent=StubAgent(ctx=front.ctx, session=front.session), tool_name="write", call_id="c1"
        )
        assert outcome == "allowed-once"
        assert [request.tool_name for request in host.approvals] == ["write"]
        types = [event.type for event in front.session.events]
        assert types.count("approval/asked") == 1
        assert types.count("approval/decided") == 1

        assert await front.ctx.user_questions.ask(UserQuestion(question="port?")) == "42"

        await front.submit("hello")
        assert front.state.status == "idle"
        assert any(item.role == "assistant" for item in front.state.items)
        # Every appended event reached the host as a redraw request.
        assert host.redraws >= len(front.session.events)
    finally:
        await front.close()


async def test_close_flushes_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disposal does not flush; `close()` must, or a session is lost on exit."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    front = await open_harness(
        compose_profile("headless"),
        host=StubHost(),
        provider="fake",
        model="fake-1",
        session_id="s",
    )
    await front.submit("hello")
    await front.close()
    store = stored_log(tmp_path / "sessions", "s")
    assert store.exists()
    assert "turn/end" in store.read_text()
