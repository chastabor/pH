"""`ph_app.tui.diffs`: the projection behind `/diff`, tested without a terminal.

Every surface replacement is one entry, whoever wrote it. The entry is
classified from the event alone. It is in the group of the node that it led to,
as git groups commits by file. The before of an entry is what the entry
replaced. Its records are the records that its producer wrote through the
revision door (`Session.records_of`), shown field by field. Its recorded size
is the `size` that the door wrote. The whole history of a group is the node as
first written against the node now. Building the list renders no text. The half
that belongs to the screen is in `test_tui_screens.py`.
"""

from __future__ import annotations

import pytest
from tui_helpers import revisable, rewritten, stand_in

from ph.json import JsonObject
from ph.llm.types import ContextForm, ModelSource, create_user_message
from ph.session import (
    RevisionRecord,
    RevisionSize,
    Session,
    SessionBatch,
    SessionEvent,
    editable_message,
    rewrite_many,
    substitute,
    unified_diff,
)
from ph.session.known_event_types import declare_log_type, is_known
from ph.testing import isolated_log_types, log_event, revision_record
from ph_app.revisions import Revision, read_revision, revision_rows
from ph_app.tui import diffs
from ph_app.tui.diffs import DiffEntry, DiffGroup, build_diffs, revisions

OUTPUT = "first line\nsecond line"

LABELS = ("before", "after")


def substitute_text(
    log: Session | SessionBatch,
    shadowed: tuple[int, ...],
    text: str,
    *,
    form: ContextForm = "compaction",
    record: RevisionRecord | None = None,
) -> SessionEvent:
    """Put `text` from the probe plugin in place of `shadowed`. Its summary says so."""
    why = record or revision_record("compaction/summarized")
    return substitute(log, shadowed, stand_in(text, form=form, summary="probe said"), why)


def only(groups: list[DiffGroup]) -> DiffEntry:
    """The one change in the one group."""
    [group] = groups
    [entry] = group.entries
    return entry


# ------------------------------------------------------------------ entries --


def test_a_log_with_no_replacement_has_no_changes() -> None:
    session, *_ = revisable(OUTPUT)

    assert build_diffs(session) == []
    assert build_diffs(Session("empty")) == []


def test_a_substitution_is_one_entry_shown_on_its_own_row() -> None:
    session, asked, replied, _ = revisable(OUTPUT)

    replacement = substitute_text(session, (asked.seq, replied.seq), "the gist")

    [group] = build_diffs(session)
    entry = only([group])
    before, after = entry.texts()
    assert (group.kind, entry.seq) == ("revision", replacement.seq)
    assert group.reveal == replacement.seq, "a substitution has a transcript row of its own"
    assert entry.producer == "probe" and entry.summary == "probe said"
    assert before.startswith("[user]\nread a.md\n\n[assistant]\nreading a.md")
    assert after == "[user]\nthe gist"
    assert {"+the gist", "-read a.md"} <= set(unified_diff(before, after, LABELS))


def test_an_offloaded_paste_is_a_substitution_and_a_summary_a_stand_in() -> None:
    """`is_stand_in` is a claim of the log itself, so a relocated paste is never
    called condensed conversation."""
    session, asked, replied, _ = revisable(OUTPUT)

    substitute_text(session, (asked.seq,), "[pasted text moved]", form="notice")
    substitute_text(session, (replied.seq,), "the gist")

    assert [group.latest.title for group in build_diffs(session)] == ["substitution", "stand-in"]


def test_an_in_place_rewrite_shows_on_the_row_it_rewrote() -> None:
    """A rewrite draws no row of its own, so going to it lands on the original."""
    session, _, _, result = revisable(OUTPUT)

    landed = rewritten(session, result, "(cut)")

    [group] = build_diffs(session)
    entry = only([group])
    assert (entry.title, entry.seq) == ("rewrite", landed.seq)
    assert (group.seq, group.reveal) == (landed.seq, result.seq)
    assert entry.summary == "tool/result rewritten in place"
    assert unified_diff(*entry.texts(), LABELS)[2:] == (
        "@@ -1,4 +1,3 @@",
        " [user]",
        " [result c1]",
        "-first line",
        "-second line",
        "+(cut)",
    )


def test_a_revision_of_a_revision_is_diffed_against_what_it_replaced() -> None:
    """Each edit is shown one time. The before of the second entry is the after
    of the first, and not the conversation as first written. Both entries are in
    the group of the node as it is now."""
    session, _, replied, result = revisable(OUTPUT)
    landed = rewritten(session, result, "(cut)")

    replacement = substitute_text(session, (replied.seq, landed.seq), "the gist")

    [group] = build_diffs(session)
    first, second = group.entries
    before, _ = second.texts()
    assert (group.seq, second.seq) == (replacement.seq, replacement.seq)
    assert "(cut)" in before and "second line" not in before
    assert first.texts()[1] in before
    assert f"seq {landed.seq} was itself a revision" in "\n".join(second.facts)


def test_a_summary_over_two_edited_sections_holds_both_edits_and_its_whole_history() -> None:
    """The gate of H3, as `git log -p -- file` and `git diff A..B -- file` show a
    file. A summary over two nodes that had each been edited is one group. The
    group holds both edits and the summary itself, oldest first. Its whole
    history is the conversation as first written against the summary. Sabotage:
    group by the change rather than by the node it led to, or diff the whole
    against the last edit's before."""
    session, asked, replied, result = revisable(OUTPUT)
    cut = rewritten(session, result, "(cut)")
    revised = substitute_text(session, (asked.seq,), "the ask, revised", form="recall")

    summary = substitute_text(session, (revised.seq, replied.seq, cut.seq), "the gist")

    [group] = build_diffs(session)
    assert group.seq == summary.seq
    assert [entry.seq for entry in group.entries] == [cut.seq, revised.seq, summary.seq]
    before, after = group.whole()
    assert after == "[user]\nthe gist"
    assert "read a.md" in before and "first line\nsecond line" in before
    assert "(cut)" not in before and "revised" not in before, "the whole began mid-history"


def test_building_the_list_renders_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The before of a summary can be most of the conversation. The screen
    renders the row that it shows, and building the list must not render every
    row."""
    rendered: list[str] = []

    def edit(_revision: Revision) -> tuple[str, str]:
        rendered.append("edit")
        return "", ""

    def whole(_session: Session, _node: int) -> tuple[str, str]:
        rendered.append("whole")
        return "", ""

    monkeypatch.setattr(Revision, "texts", edit)
    monkeypatch.setattr(diffs, "_whole", whole)
    session, asked, replied, _ = revisable(OUTPUT)
    substitute_text(session, (asked.seq,), "one")
    substitute_text(session, (replied.seq,), "two")

    first, _ = build_diffs(session)
    assert rendered == []
    first.latest.texts()
    first.whole()
    assert rendered == ["edit", "whole"]


# ------------------------------------------------------------------ records --


def test_a_record_is_shown_field_by_field_and_its_size_is_the_entry_s() -> None:
    """The record is read through a mirror that does not know its type, as the
    TUI holds a log whose bundle it may not import. The size is the size that
    the door wrote, whatever the type. Sabotage: read sizes off field names
    again."""
    with isolated_log_types():
        declare_log_type("probe/revised", owner="probe", ignorable=True, audit_only=True)
        session, asked, _, _ = revisable(OUTPUT)
        fields: JsonObject = {"verb": "replace", "tokensBefore": 7, "contextBefore": 900}
        why = revision_record("probe/revised", fields, RevisionSize("tokens", 120, 8))
        substitute_text(session, (asked.seq,), "q", record=why)
    assert not is_known("probe/revised")
    mirror = Session(session.id)
    for event in session.events:
        mirror.admit(SessionEvent.from_wire(event.to_wire()))

    entry = only(build_diffs(mirror))

    [record] = entry.records
    assert (record.type, record.revises) == ("probe/revised", (entry.seq,))
    assert record.fields == (
        ("verb", "replace"),
        ("tokensBefore", "7"),
        ("contextBefore", "900"),
    ), "the door's own stamps are listed as the producer's fields"
    assert entry.recorded == RevisionSize("tokens", 120, 8)


def test_without_a_recorded_size_there_is_none() -> None:
    """The screen counts characters for the row it shows instead."""
    session, asked, _, _ = revisable(OUTPUT)
    summarized = revision_record("compaction/summarized", {"trigger": "manual"})
    substitute_text(session, (asked.seq,), "q", record=summarized)

    entry = only(build_diffs(session))

    assert entry.records[0].type == "compaction/summarized"
    assert entry.recorded is None


def test_several_edits_in_one_batch_each_keep_their_own_record() -> None:
    """Each edit shows the record that was written for it. A record that only
    shares its batch belongs to no edit. Sabotage: pair records by where they
    sit in the batch again."""
    session, asked, replied, _ = revisable(OUTPUT)
    with session.batch() as batch:
        substitute_text(
            batch,
            (asked.seq,),
            "one",
            record=revision_record("compaction/summarized", {}, RevisionSize("tokens", 10, 1)),
        )
        substitute_text(
            batch,
            (replied.seq,),
            "two",
            record=revision_record("compaction/summarized", {}, RevisionSize("tokens", 20, 2)),
        )
        log_event(batch, "compaction/declined", {"trigger": "manual"})

    first, second = (group.latest for group in build_diffs(session))

    assert first.recorded == RevisionSize("tokens", 10, 1)
    assert second.recorded == RevisionSize("tokens", 20, 2)
    assert [one.type for one in second.records] == ["compaction/summarized"]


def test_one_record_for_several_rewrites_is_shown_on_each_and_sizes_none() -> None:
    """Compaction writes one record for every argument that it elided. Each
    rewrite shows that record, and the record says that it explains all of them.
    The numbers of the record are the size of none of the rewrites alone."""
    session, _, replied, result = revisable(OUTPUT)
    reply = editable_message(replied)
    reply["content"][0] = {"type": "text", "text": "reading"}
    output = editable_message(result)
    output["content"][0]["content"] = [{"type": "text", "text": "(cut)"}]
    elided = revision_record("compaction/args-truncated", {}, RevisionSize("tokens", 90, 9))

    rewrite_many(session, [(replied, reply), (result, output)], elided)

    first, second = (group.latest for group in build_diffs(session))
    assert first.records == second.records
    assert first.records[0].revises == (first.seq, second.seq)
    assert first.recorded is None and second.recorded is None


def test_a_change_whose_message_names_no_plugin_has_no_producer() -> None:
    """The producer of a revision is the plugin that its message names, or `None`.

    A substitution by a plugin names that plugin. A message whose source is a
    model names a model and no plugin, and a rewritten tool result names no
    plugin either. The row for training then says `null`. `/diff` names the
    types of the records beside such a change, and nothing when there are none.

    Sabotage: answer the name for every kind of source in `read_revision`, and
    the model is reported as the producer.
    """
    session, asked, _, result = revisable(OUTPUT)
    summarized = substitute_text(session, (asked.seq,), "q")
    clipped = rewritten(session, result, "(cut)")
    other, asked_again, _, _ = revisable(OUTPUT)
    by_model = substitute(
        other,
        (asked_again.seq,),
        create_user_message(
            content=[{"type": "text", "text": "q"}],
            source=ModelSource(provider="fake", model="fake-1"),
        ),
        revision_record("compaction/summarized"),
    )

    assert read_revision(session, summarized).producer == "probe"
    assert read_revision(session, clipped).producer is None
    assert read_revision(other, by_model).producer is None
    rows = {row["seq"]: row["producer"] for row in revision_rows(session)}
    assert rows == {summarized.seq: "probe", clipped.seq: None}
    assert [row["producer"] for row in revision_rows(other)] == [None]

    by_seq = {group.latest.seq: group.latest for group in build_diffs(session)}
    assert by_seq[clipped.seq].producer == "offload/spilled"
    assert diffs._beside(()) is None
    unnamed = DiffEntry(seq=1, title="other", summary="", producer=None, texts=lambda: ("a", "b"))
    group = DiffGroup(kind="revision", seq=1, reveal=1, entries=(unnamed,), whole=unnamed.texts)
    assert group.producers == ""


# ------------------------------------------------------------------ sources --


def test_a_new_kind_is_a_source_function_and_the_groups_merge_by_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost of "meant to grow more kinds" is one function in `SOURCES`."""
    session, asked, _, _ = revisable(OUTPUT)
    substitute_text(session, (asked.seq,), "q")
    entry = DiffEntry(
        seq=1,
        title="other",
        summary="",
        producer=None,
        texts=lambda: ("a", "b"),
    )

    def another(log: Session) -> list[DiffGroup]:
        return [DiffGroup(kind="revision", seq=1, reveal=1, entries=(entry,), whole=entry.texts)]

    monkeypatch.setattr(diffs, "SOURCES", (revisions, another))

    assert [group.latest.title for group in build_diffs(session)] == ["other", "stand-in"]
