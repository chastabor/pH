"""P4-02 — `input-offload`: a pasted blob relocated, not lost (G3).

The row's gate is the threshold, but the claim worth testing is the *split*:
after an offload the model reads a preview while the log still holds what the
person actually sent. Those are two projections of one append-only log — the
same mechanism compaction uses — and a test that checked only the model's side
would pass for the design this one was chosen over, where the harness rewrites
the message before logging it and the record quietly attributes its own words
to the human.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, blob, break_spill

from ph.llm.types import text_of
from ph.session import Session, SurfaceIntent
from ph.session.events import SurfaceReplace
from ph.session.known_event_types import (
    IGNORABLE_SESSION_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
)
from ph.testing import FAKE_OPTIONS, user_payload
from ph_stabilize.input_offload import (
    HUMAN_TOKEN_LIMIT_BEFORE_EVICT,
    TOO_LARGE_HUMAN_MSG,
    Config,
    _pending,
)
from ph_stabilize.offload import NUM_CHARS_PER_TOKEN

pytestmark = pytest.mark.anyio
THRESHOLD = NUM_CHARS_PER_TOKEN * HUMAN_TOKEN_LIMIT_BEFORE_EVICT
"""200 000, derived from the constants so the gate moves with the policy."""

TOO_LARGE = TOO_LARGE_HUMAN_MSG.partition(" and")[0]
"""The replacement's opening words, from the constant rather than retyped."""


async def _prompt(ctx: Any, session: Session, text: str) -> Any:
    """Run one real turn on the fake adapter with `text` as the human message."""
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    await agent.prompt(text)
    return agent


def _model_text(session: Session) -> str:
    """What the model was sent — the derived surface, not the raw log."""
    return "\n".join(text_of(message.content) for message in session.derive_messages())


def _human_text(session: Session) -> str:
    """What a person scrolling the transcript reads."""
    return "\n".join(text_of(message.content) for message in session.transcript())


# ------------------------------------------------------------- the threshold --


async def test_a_paste_at_the_threshold_is_left_alone(mount: Any) -> None:
    """200 000 characters is admitted — the limit is what is still allowed."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("at-limit")
    original = blob(THRESHOLD)

    await _prompt(ctx, session, original)

    assert not [e for e in session.events if e.type == "offload/input-spilled"]
    assert original in _model_text(session)


async def test_one_character_over_is_offloaded(mount: Any) -> None:
    """200 001 is not. The row's gate, and the reason the comparison is `>`."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("over-limit")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    (spilled,) = [e for e in session.events if e.type == "offload/input-spilled"]
    assert Path(spilled.data["locator"]).is_file()


# ---------------------------------------------------- the split, which is (c) --


async def test_the_model_reads_a_preview_and_the_log_keeps_what_was_typed(
    mount: Any,
) -> None:
    """The whole design, in one assertion pair.

    A substitution on the *surface*, not an edit to the log — so the record
    never attributes the harness's preview to the person, and the person can
    still scroll back to what they pasted.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("split")
    original = blob(THRESHOLD + 1)

    await _prompt(ctx, session, original)

    model = _model_text(session)
    assert TOO_LARGE in model
    assert original not in model, "the model was sent the blob after all"

    assert original in _human_text(session), "the log lost what the person actually wrote"
    assert TOO_LARGE not in _human_text(session)


async def test_the_original_is_recoverable_from_the_path_the_model_was_given(
    mount: Any,
) -> None:
    """A relocation, not a deletion: the path must hold the text."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("recoverable")
    original = blob(THRESHOLD + 1)

    await _prompt(ctx, session, original)
    (spilled,) = [e for e in session.events if e.type == "offload/input-spilled"]

    assert spilled.data["locator"] in _model_text(session)
    assert Path(spilled.data["locator"]).read_text(encoding="utf-8") == original


async def test_the_preview_is_a_plugins_notice_not_the_persons_words(
    mount: Any,
) -> None:
    """Attribution. The replacement is the harness speaking, and says so.

    Marking it `user` would be the same false record the rejected design made,
    one layer down — and the trajectory view reads exactly this field to say
    who produced a row.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("attribution")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    replacement = next(
        e for e in session.events if e.type == "user/message" and e.surface_op != "append"
    )
    source = replacement.data["source"]
    assert source["kind"] == "plugin"
    assert source["plugin"] == "input-offload"
    assert source["form"] == "notice"


# ------------------------------------------------------------------ idempotent --


def test_a_replacement_is_never_offloaded_again() -> None:
    """Idempotence, asked of the predicate directly.

    End-to-end could not express this, and a test that tried passed while the
    check was deleted. After an offload the next thing appended is usually the
    *next* human message, so `latest("user/message")` is that one and the check
    is never consulted. It is consulted on a second **step** of the same turn —
    a tool call, then another request — where the replacement is still the
    newest human message. Rather than script a tool round-trip to reach that
    state, the predicate is put in it directly, which is the only form that
    fails when the check is removed.

    The threshold is lowered so the preview is itself "oversized": at the
    shipped 200 000 a ~11 KB preview could never re-trip it, and a deployment
    that tightens the limit is exactly who this protects — without the check
    every step would append another replacement, without end.
    """
    config = Config(token_limit=100)
    session = Session("idempotent")
    session.append("turn/start", {"turn": 1})
    original = session.append("user/message", user_payload("p" * 2_001), SurfaceIntent("append"))
    assert _pending(session, config) is not None, "the paste should have been offloaded"

    session.append(
        "user/message",
        user_payload("still long " * 100),
        SurfaceIntent(
            surface_op=SurfaceReplace(start=original.seq, end=original.seq),
            source_event_seqs=(original.seq,),
        ),
    )

    assert _pending(session, config) is None, "the replacement was offloaded again"


# ------------------------------------------------------------------ fail open --


async def test_a_spill_that_fails_keeps_the_message(
    mount: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offload that cannot store the content must not be why it is lost."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("no-disk")
    break_spill(monkeypatch)
    original = blob(THRESHOLD + 1)

    await _prompt(ctx, session, original)

    # The turn *completed*. Without this the test could not tell fail-open from
    # the listener taking the whole request down — `agent.prompt` reports a
    # failed turn rather than raising, so "no spill event, original still
    # derived" is equally true of a crash. A mutation that re-raised survived
    # this file until the assertion below was added.
    assert [e for e in session.events if e.type == "assistant/message"], "the turn did not run"
    assert not [e for e in session.events if e.type == "offload/input-spilled"]
    assert original in _model_text(session)


def test_the_event_type_is_in_the_vocabulary() -> None:
    """The proof a producer outside ph-core owes through its own bundle."""
    assert "offload/input-spilled" in KNOWN_SESSION_EVENT_TYPES
    assert "offload/input-spilled" in IGNORABLE_SESSION_EVENT_TYPES
