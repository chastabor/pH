"""P4-02, `input-offload`: a pasted blob is relocated, not lost (G3).

The gate of the row is the threshold, but the claim that is worth a test is the
*split*: after an offload, the model reads a preview and the log still holds
what the user sent. Those are two projections of one append-only log, which is
the mechanism that compaction also uses. A test that checked only the side of
the model would pass for the design that this one was chosen over. In that
design, the harness rewrites the message before it logs it, and the record
attributes the words of the harness to the human with no sign of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from stabilize_helpers import PROFILE, blob, break_spill

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.json import as_int, as_obj, as_seq
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
"""The opening words of the replacement, taken from the constant and not retyped."""


async def _prompt(ctx: Context, session: Session, text: str) -> AgentDriver:
    """Run one real turn on the fake adapter with `text` as the human message."""
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await agent.prompt(text)
    return agent


def _model_text(session: Session) -> str:
    """What the model was sent: the derived surface, not the raw log."""
    return "\n".join(text_of(message.content) for message in session.derive_messages())


def _human_text(session: Session) -> str:
    """What a user reads when they scroll the transcript."""
    return "\n".join(text_of(message.content) for message in session.transcript())


# ------------------------------------------------------------- the threshold --


async def test_a_paste_at_the_threshold_is_left_alone(mount: MountProfile) -> None:
    """200 000 characters are admitted. The limit is the size that is still allowed."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("at-limit")
    original = blob(THRESHOLD)

    await _prompt(ctx, session, original)

    assert not [e for e in session.events if e.type == "offload/input-spilled"]
    assert original in _model_text(session)


async def test_one_character_over_is_offloaded(mount: MountProfile) -> None:
    """200 001 characters are not admitted. This is the gate of the row, and the
    reason why the comparison is `>`."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("over-limit")

    await _prompt(ctx, session, blob(THRESHOLD + 1))

    (spilled,) = [e for e in session.events if e.type == "offload/input-spilled"]
    assert Path(str(spilled.data["locator"])).is_file()
    # The accounting and the replacement it describes, in one batch (S14).
    [revised] = [as_int(one) for one in as_seq(spilled.data["revises"])]
    preview = session.events[revised]
    assert preview.type == "user/message" and preview.surface_op is not None
    assert session.records_of(revised) == (spilled,)
    size = as_obj(spilled.data["size"])
    assert (size["unit"], size["before"]) == ("bytes", spilled.data["bytes"])
    assert 0 < as_int(size["after"]) < as_int(size["before"])
    assert spilled.batch is not None and preview.batch == spilled.batch


# ---------------------------------------------------- the split, which is (c) --


async def test_the_model_reads_a_preview_and_the_log_keeps_what_was_typed(
    mount: MountProfile,
) -> None:
    """The whole design, in one pair of assertions.

    The substitution is on the *surface* and is not an edit to the log. The
    record therefore never attributes the preview of the harness to the user,
    and the user can still scroll back to what they pasted.
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
    """Attribution. The replacement is the harness speaking, and it says so.

    To mark it `user` would be the same false record that the rejected design
    made, one layer down. The trajectory view reads exactly this field to say
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

    An end-to-end test could not express this. One that tried still passed after
    the check was deleted. After an offload, the next message appended is
    usually the *next* human message, so the newest `user/message` is that one
    and the check is never consulted. The check is consulted when a request is
    built again in the same **step**, for example on a retry. The replacement is
    then still the newest `user/message` of the step. The test does not script a
    retry to reach that state. It puts the predicate in that state directly,
    which is the only form that fails when the check is removed.

    The threshold is lowered so that the preview is itself "oversized". At the
    shipped 200 000, a preview of about 11 KB could never exceed it again. A
    deployment that tightens the limit is exactly the one that this protects:
    without the check, every such request would append another replacement,
    without end.
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
    """C12: the newest *human* message, not the newest `user/message`.

    When the context of a step changed, the step appends the snapshot of the
    harness after the words of the user, in one batch. When the row took the
    newest message, the paste went through whole and the snapshot was offloaded
    in its place. The snapshot is the context of the harness itself. The model
    no longer saw it, so the loop sent it again at every step.

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

    # The turn *completed*. Without this assertion, the test could not tell
    # fail-open from a listener that takes the whole request down.
    # `agent.prompt` reports a failed turn and does not raise, so "no spill
    # event, original still derived" is equally true of a crash. A mutation that
    # re-raised survived this file until the assertion below was added.
    assert [e for e in session.events if e.type == "assistant/message"], "the turn did not run"
    assert not [e for e in session.events if e.type == "offload/input-spilled"]
    assert original in _model_text(session)


def test_the_event_type_is_ignorable() -> None:
    """Ignorable for the same reason as `offload/spilled`: the replacement of the
    paste is what the model reads. (`test_log_writers.py` checks that the type
    is *known*.)"""
    assert "offload/input-spilled" in IGNORABLE_SESSION_EVENT_TYPES


async def test_a_spilled_paste_names_the_same_tools_its_sibling_does(
    mount: MountProfile,
) -> None:
    """The half that the first correction missed.

    Both blocks come from upstream and both name `read_file`, which pH does not
    register. Only the tool-result side was corrected. A spilled paste and a
    spilled result are the same kind of file in the same store, and the model
    cannot tell which row wrote the path that it was given. A sentence that is
    true of one and absent from the other is therefore worse than either
    alternative.

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
    # The tracked literal is untouched. That is the reason why it is separate.
    assert "read_file" in UPSTREAM_TOO_LARGE_HUMAN_MSG
