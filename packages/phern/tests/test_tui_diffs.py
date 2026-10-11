"""`ph_app.tui.diffs`: the projection behind `/diff`, held without a terminal.

Every surface replacement is one entry, whoever wrote it, classified from the event
alone. Its before is what it replaced, its records are the ones in its batch, shown
field by field, and its recorded size is a record's own before/after pair. Building the
list renders no text. The screen's half is in `test_tui_screens.py`.
"""

from __future__ import annotations

import pytest
from tui_helpers import revisable, rewritten, stand_in

from ph.llm.types import ContextForm
from ph.session import Session, SessionBatch, SessionEvent, substitute
from ph.session.known_event_types import declare_log_type, is_known
from ph.testing import isolated_log_types, log_event
from ph_app.tui import diffs
from ph_app.tui.diffs import DiffEntry, Size, build_diffs, revisions, unified

OUTPUT = "first line\nsecond line"


def substitute_text(
    log: Session | SessionBatch,
    shadowed: tuple[int, ...],
    text: str,
    *,
    form: ContextForm = "compaction",
) -> SessionEvent:
    """`text` from the probe plugin in place of `shadowed`, its summary saying so."""
    return substitute(log, shadowed, stand_in(text, form=form, summary="probe said"))


# ------------------------------------------------------------------ entries --


def test_a_log_with_no_replacement_has_no_changes() -> None:
    session, *_ = revisable(OUTPUT)

    assert build_diffs(session) == []
    assert build_diffs(Session("empty")) == []


def test_a_substitution_is_one_entry_shown_on_its_own_row() -> None:
    session, asked, replied, _ = revisable(OUTPUT)

    replacement = substitute_text(session, (asked.seq, replied.seq), "the gist")

    [entry] = build_diffs(session)
    before, after = entry.texts()
    assert (entry.kind, entry.seq) == ("revision", replacement.seq)
    assert entry.reveal == replacement.seq, "a substitution has a transcript row of its own"
    assert entry.producer == "probe" and entry.summary == "probe said"
    assert before.startswith("[user]\nread a.md\n\n[assistant]\nreading a.md")
    assert after == "[user]\nthe gist"
    assert {"+the gist", "-read a.md"} <= set(unified(before, after))


def test_an_offloaded_paste_is_a_substitution_and_a_summary_a_stand_in() -> None:
    """`is_stand_in` is the log's own claim, so a relocated paste is never called
    condensed conversation."""
    session, asked, replied, _ = revisable(OUTPUT)

    substitute_text(session, (asked.seq,), "[pasted text moved]", form="notice")
    substitute_text(session, (replied.seq,), "the gist")

    assert [entry.title for entry in build_diffs(session)] == ["substitution", "stand-in"]


def test_an_in_place_rewrite_shows_on_the_row_it_rewrote() -> None:
    """A rewrite draws no row of its own, so going to it lands on the original."""
    session, _, _, result = revisable(OUTPUT)

    landed = rewritten(session, result, "(cut)")

    [entry] = build_diffs(session)
    assert (entry.title, entry.seq, entry.reveal) == ("rewrite", landed.seq, result.seq)
    assert entry.summary == "tool/result rewritten in place"
    assert unified(*entry.texts())[2:] == (
        "@@ -1,4 +1,3 @@",
        " [user]",
        " [result c1]",
        "-first line",
        "-second line",
        "+(cut)",
    )


def test_a_revision_of_a_revision_is_diffed_against_what_it_replaced() -> None:
    """Each edit once: the second entry's before is the first one's after, not the
    conversation as first written."""
    session, _, replied, result = revisable(OUTPUT)
    landed = rewritten(session, result, "(cut)")

    replacement = substitute_text(session, (replied.seq, landed.seq), "the gist")

    first, second = build_diffs(session)
    before, _ = second.texts()
    assert second.seq == replacement.seq
    assert "(cut)" in before and "second line" not in before
    assert first.texts()[1] in before
    assert f"seq {landed.seq} was itself a revision" in "\n".join(second.facts)


def test_building_the_list_renders_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary's before can be most of the conversation: the screen renders the
    row it shows, and the list must not render every row to be built."""
    rendered: list[int] = []

    def counting(session: Session, shadowed: tuple[int, ...], seq: int) -> tuple[str, str]:
        rendered.append(seq)
        return "", ""

    monkeypatch.setattr(diffs, "_texts", counting)
    session, asked, replied, _ = revisable(OUTPUT)
    substitute_text(session, (asked.seq,), "one")
    substitute_text(session, (replied.seq,), "two")

    first, _ = build_diffs(session)
    assert rendered == []
    first.texts()
    assert rendered == [first.seq]


# ------------------------------------------------------------------ records --


def test_a_record_in_the_batch_is_shown_field_by_field_and_its_pair_is_the_size() -> None:
    """Through a mirror that does not know the record's type, as the TUI holds a
    log whose bundle it may not import. The pair is a convention, not a name."""
    with isolated_log_types():
        declare_log_type("probe/revised", owner="probe", ignorable=True, audit_only=True)
        session, asked, _, _ = revisable(OUTPUT)
        with session.batch() as batch:
            substitute_text(batch, (asked.seq,), "q")
            log_event(
                batch,
                "probe/revised",
                {"verb": "replace", "tokensBefore": 120, "tokensAfter": 8, "contextBefore": 900},
            )
    assert not is_known("probe/revised")
    mirror = Session(session.id)
    for event in session.events:
        mirror.admit(SessionEvent.from_wire(event.to_wire()))

    [entry] = build_diffs(mirror)

    [record] = entry.records
    assert (record.type, record.shared) == ("probe/revised", False)
    assert record.fields == (
        ("verb", "replace"),
        ("tokensBefore", "120"),
        ("tokensAfter", "8"),
        ("contextBefore", "900"),
    )
    assert entry.recorded == Size(120, 8, "tokens")


def test_without_a_recorded_pair_there_is_no_recorded_size() -> None:
    """The screen counts characters for the row it shows instead."""
    session, asked, _, _ = revisable(OUTPUT)
    with session.batch() as batch:
        log_event(batch, "compaction/summarized", {"trigger": "manual", "shadowedTokens": 40})
        substitute_text(batch, (asked.seq,), "q")

    [entry] = build_diffs(session)

    assert entry.records[0].type == "compaction/summarized"
    assert entry.recorded is None


def test_several_edits_in_one_batch_each_keep_their_own_record() -> None:
    """One record per replacement pairs up in order; a record for the whole batch is
    shown on each and sizes none."""
    session, asked, replied, _ = revisable(OUTPUT)
    with session.batch() as batch:
        substitute_text(batch, (asked.seq,), "one")
        log_event(batch, "compaction/summarized", {"tokensBefore": 10, "tokensAfter": 1})
        substitute_text(batch, (replied.seq,), "two")
        log_event(batch, "compaction/summarized", {"tokensBefore": 20, "tokensAfter": 2})
        log_event(batch, "compaction/declined", {"tokensBefore": 99, "tokensAfter": 9})

    first, second = build_diffs(session)

    assert first.recorded == Size(10, 1, "tokens")
    assert second.recorded == Size(20, 2, "tokens")
    assert [(one.type, one.shared) for one in second.records] == [
        ("compaction/summarized", False),
        ("compaction/declined", True),
    ]


# ------------------------------------------------------------------ sources --


def test_a_new_kind_is_a_source_function_and_the_entries_merge_by_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What "meant to grow more kinds" costs: one function in `SOURCES`."""
    session, asked, _, _ = revisable(OUTPUT)
    substitute_text(session, (asked.seq,), "q")

    def another(log: Session) -> list[DiffEntry]:
        return [
            DiffEntry(
                kind="revision",
                seq=1,
                reveal=1,
                title="other",
                summary="",
                producer="",
                texts=lambda: ("a", "b"),
            )
        ]

    monkeypatch.setattr(diffs, "SOURCES", (revisions, another))

    assert [entry.title for entry in build_diffs(session)] == ["other", "stand-in"]
