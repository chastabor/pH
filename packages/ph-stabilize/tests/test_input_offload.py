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

import pytest
from stabilize_helpers import PROFILE, blob, break_spill

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.json import as_obj
from ph.keys import AGENTS, SESSIONS
from ph.llm.types import PluginSource, create_user_message, text_of
from ph.session import Session, SurfaceIntent
from ph.session.events import SurfaceReplace
from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES
from ph.system_prompt.assembly import CONTEXT_PLUGIN
from ph.testing import FAKE_OPTIONS, MountProfile, log_event, user_payload
from ph_stabilize.input_offload import (
    HUMAN_TOKEN_LIMIT_BEFORE_EVICT,
    TOO_LARGE_HUMAN_MSG,
    UPSTREAM_TOO_LARGE_HUMAN_MSG,
    Config,
    _pending,
)
from ph_stabilize.offload import NUM_CHARS_PER_TOKEN

pytestmark = pytest.mark.anyio
THRESHOLD = NUM_CHARS_PER_TOKEN * HUMAN_TOKEN_LIMIT_BEFORE_EVICT
"""200 000, derived from the constants so the gate moves with the policy."""

TOO_LARGE = TOO_LARGE_HUMAN_MSG.partition(" and")[0]
"""The replacement's opening words, from the constant rather than retyped."""


async def _prompt(ctx: Context, session: Session, text: str) -> AgentDriver:
    """Run one real turn on the fake adapter with `text` as the human message."""
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await agent.prompt(text)
    return agent


def _model_text(session: Session) -> str:
    """What the model was sent — the derived surface, not the raw log."""
    return "\n".join(text_of(message.content) for message in session.derive_messages())


def _human_text(session: Session) -> str:
    """What a person scrolling the transcript reads."""
    return "\n".join(text_of(message.content) for message in session.transcript())


# ------------------------------------------------------------- the threshold --


async def test_a_paste_at_the_threshold_is_left_alone(mount: MountProfile) -> None:
    """200 000 characters is admitted — the limit is what is still allowed."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("at-limit")
    original = blob(THRESHOLD)

    await _prompt(ctx, session, original)

    assert not [e for e in session.events if e.type == "offload/input-spilled"]
    assert original in _model_text(session)


async def test_one_character_over_is_offloaded(mount: MountProfile) -> None:
    """200 001 is not. The row's gate, and the reason the comparison is `>`."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("over-limit")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    (spilled,) = [e for e in session.events if e.type == "offload/input-spilled"]
    assert Path(str(spilled.data["locator"])).is_file()


# ---------------------------------------------------- the split, which is (c) --


async def test_the_model_reads_a_preview_and_the_log_keeps_what_was_typed(
    mount: MountProfile,
) -> None:
    """The whole design, in one assertion pair.

    A substitution on the *surface*, not an edit to the log — so the record
    never attributes the harness's preview to the person, and the person can
    still scroll back to what they pasted.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("split")
    original = blob(THRESHOLD + 1)

    await _prompt(ctx, session, original)

    model = _model_text(session)
    assert TOO_LARGE in model
    assert original not in model, "the model was sent the blob after all"

    assert original in _human_text(session), "the log lost what the person actually wrote"
    assert TOO_LARGE not in _human_text(session)


async def test_the_original_is_recoverable_from_the_path_the_model_was_given(
    mount: MountProfile,
) -> None:
    """A relocation, not a deletion: the path must hold the text."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("recoverable")
    original = blob(THRESHOLD + 1)

    await _prompt(ctx, session, original)
    (spilled,) = [e for e in session.events if e.type == "offload/input-spilled"]

    assert str(spilled.data["locator"]) in _model_text(session)
    assert Path(str(spilled.data["locator"])).read_text(encoding="utf-8") == original


async def test_the_preview_is_a_plugins_notice_not_the_persons_words(
    mount: MountProfile,
) -> None:
    """Attribution. The replacement is the harness speaking, and says so.

    Marking it `user` would be the same false record the rejected design made,
    one layer down — and the trajectory view reads exactly this field to say
    who produced a row.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("attribution")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    replacement = next(
        e for e in session.events if e.type == "user/message" and e.surface_op != "append"
    )
    source = replacement.data["source"]
    assert as_obj(source)["kind"] == "plugin"
    assert as_obj(source)["plugin"] == "input-offload"
    assert as_obj(source)["form"] == "notice"


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
    log_event(session, "turn/start", {"turn": 1})
    original = log_event(
        session, "user/message", user_payload("p" * 2_001), SurfaceIntent("append")
    )
    assert _pending(session, config) is not None, "the paste should have been offloaded"

    log_event(
        session,
        "user/message",
        user_payload("still long " * 100),
        SurfaceIntent(
            surface_op=SurfaceReplace(replaces=(original.seq,)),
            source_event_seqs=(original.seq,),
        ),
    )

    assert _pending(session, config) is None, "the replacement was offloaded again"


def test_the_persons_paste_is_offloaded_not_the_harness_context_after_it() -> None:
    """C12 — the newest *human* message, not the newest `user/message`.

    A step whose context changed appends the harness's snapshot after the
    person's words, in one batch. Taking the newest let the paste through whole
    and offloaded the snapshot instead — the harness's own context, which the
    loop then re-sent every step, because the model no longer saw it.

    Sabotage: take `session.latest("user/message")` again and the snapshot is
    the one chosen.
    """
    config = Config(token_limit=100)
    session = Session("batch")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    paste = log_event(session, "user/message", user_payload("p" * 2_001), SurfaceIntent("append"))
    log_event(
        session,
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "context " * 300}],
            source=PluginSource(plugin=CONTEXT_PLUGIN, form="snapshot", sections=[]),
        ).to_wire(),
        SurfaceIntent("append"),
    )

    pending = _pending(session, config)

    assert pending is not None and pending[0].seq == paste.seq


# ------------------------------------------------------------------ fail open --


async def test_a_spill_that_fails_keeps_the_message(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offload that cannot store the content must not be why it is lost."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("no-disk")
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


def test_the_event_type_is_ignorable() -> None:
    """For `offload/spilled`'s reason: the paste's replacement is what the model
    reads. (That it is *known* is `test_log_writers.py`'s.)"""
    assert "offload/input-spilled" in IGNORABLE_SESSION_EVENT_TYPES


async def test_a_spilled_paste_names_the_same_tools_its_sibling_does(
    mount: MountProfile,
) -> None:
    """The half the first correction missed.

    Both blocks are upstream's and both name `read_file`, which pH does not
    register — but only the tool-result side was corrected. A spilled paste and a
    spilled result are the same kind of file in the same store, and the model
    cannot tell which row wrote the path it was handed, so a sentence true of one
    and absent from the other is worse than either.

    Sabotage: append `SPILL_TOOLS_HINT` here without formatting it and the model
    reads a literal `{searchers}`; drop the localization and it reads `read_file`.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("named")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    said = _model_text(session)
    assert TOO_LARGE in said, "the paste was not offloaded"
    assert "read_file" not in said, "the person's paste was answered with a tool pH lacks"
    assert "read tool" in said
    assert "`grep`" in said and "`glob`" in said
    assert "{searchers}" not in said and "{reader}" not in said, "an unformatted template"
    # The tracked literal is untouched, which is the whole reason it is separate.
    assert "read_file" in UPSTREAM_TOO_LARGE_HUMAN_MSG
