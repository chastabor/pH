"""`text-index`: chunking that keeps its place, and an index that survives a reopen.

Gates:

* Without an embedder, a call says what is missing.
* A reopened index answers the same question.
* The line span of a chunk names the lines that the chunk really came from.

These tests mount the **engine** rows, as the `ph-text-index` command mounts
them. `test_text_index_row.py` covers the rows that a session mounts, and
`test_text_index_cli.py` covers the command itself.

## Why a stub embedder and not the real model

The shipped provider is `sentence-transformers`, which downloads weights on
first use. A suite that loaded it would fail in an airgapped CI. It would take a
minute to prove that paging works, and it would test the model and not this
row. These tests therefore register a deterministic hashing embedder and
exercise **the real turbovec index** through it. The index is where the
interesting behavior is: quantized scores, an allowlist that raises when it is
empty, incremental `sync`, and a sidecar that must agree with a binary file.

`HashingEmbedder` is a bag of words over `blake2b` buckets. Similar text
therefore really produces similar vectors, and the retrieval assertions below
are about ranking and not about a fixture. It uses `blake2b` and not `hash()`,
which is salted for each process. A stub whose vectors changed between runs
would make a reopened index look corrupt.

## Why the line span is asserted against the file

A chunk that does not know where it came from is the failure that this row
exists to avoid. The failure is silent: the passage still reads correctly, the
score is still fine, and the next `read` of the model lands in another place.
The assertions therefore compare the span against the lines of the document
itself, and not against an expected number.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import anyio
import pytest
from text_index_helpers import new_agent as _agent
from text_index_helpers import prose
from text_index_stub import DOCUMENT, BoundedEmbedder, HashingEmbedder

from ph.characters import cut_before
from ph.cordis import DEPLOYMENT, Context
from ph.indexable import paragraphs, triage
from ph.keys import COMMANDS, FS, SYSTEM_PROMPT, TOOLS
from ph.llm.types import text_of
from ph.testing import EMOJI_FAMILY, MountProfile, raising, report_section, run_tool
from ph.testing.git import git, git_repo
from ph.testing.jj import jj_repo
from ph.toolcli import STATE_ENV, reporting_to
from ph.workspace_caches import WORKSPACE_FILE, digest
from ph_text_index._chunk import Chunk, Weight, chunk_paragraphs, chunk_text, head_of
from ph_text_index._config import index_base
from ph_text_index._embed import Vectors, weigh_by_bytes
from ph_text_index._store import IndexMismatch, TextIndex
from ph_text_index.engine import TEXT_INDEX, Config, TextIndexSeam

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "text-index", "name": "text-index-engine"}
"""The engine row: the tools themselves, as the `ph-text-index` command mounts them.

The row is mounted under the id of the session row, as the command does it. A
config that is written for `text-index` is then read by the row that indexes."""


def _opened(ctx: Context) -> TextIndex:
    """The one index that the agents of a test opened. They share a workspace."""
    [store] = ctx.require(TEXT_INDEX)._indexes.values()
    return store


async def _mounted(
    mount: MountProfile,
    tmp_path: Path,
    **config: object,
) -> tuple[Any, HashingEmbedder]:
    """Mount the engine row, then claim a stub embedder in the way that an embedder
    row claims one.

    The engine row registers both tools at mount. The claim fills the embedder
    slot, which the tools need when they are called.
    """
    ctx = await mount({**ROW, "config": config})
    embedder = HashingEmbedder()
    ctx.require(TEXT_INDEX).register(embedder)
    return ctx, embedder


# ----------------------------------------------------------------- chunking ----


def test_a_chunk_knows_the_lines_it_came_from() -> None:
    """Asserted against the document, not against a number. See the module docstring."""
    lines = DOCUMENT.splitlines()
    chunks = chunk_text(DOCUMENT, max_chars=200, overlap_chars=0)

    assert chunks
    for chunk in chunks:
        span = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph in span, f"{paragraph!r} is not within its own line span"


def test_cuts_land_between_paragraphs() -> None:
    """The reason why this is not a character stride: no chunk starts in the
    middle of a sentence."""
    chunks = chunk_text(DOCUMENT, max_chars=200, overlap_chars=0)

    assert len(chunks) > 1, "the fixture did not exercise a cut"
    for chunk in chunks:
        assert chunk.text == chunk.text.strip()
        assert not chunk.text.startswith(("and ", "the "))


def test_the_overlap_carries_the_previous_tail_forward() -> None:
    chunks = chunk_text(DOCUMENT, max_chars=200, overlap_chars=150)

    assert len(chunks) > 1
    # Some later chunk begins with something an earlier one also held.
    assert any(
        chunks[index].text.split("\n\n")[0] in chunks[index - 1].text
        for index in range(1, len(chunks))
    )


def test_no_chunk_exceeds_the_bound_however_large_the_overlap() -> None:
    """X4: the carry-over was added on top of the limit and not inside it.

    After a flush, the next chunk was seeded with the blocks from `_tail`, and
    the block that caused the flush was appended without a second check. The
    input limit of an embedder is a hard limit. What goes over it is truncated
    at the far end. The index then holds a chunk whose tail no search can ever
    match, and nothing reports it.

    **The paragraphs are deliberately uneven.** With blocks of one size, the
    overshoot is a couple of characters, because a tail plus one block is
    roughly a chunk by construction. A fixture built that way passes against the
    broken code, and that is how this nearly shipped untested. A
    small-small-large chunk carries a tail that is almost the whole limit, and
    the large block behind it is then added on top: 324 characters against a
    bound of 200.
    """
    small, large = "s" * 20, "L" * 150
    text = "\n\n".join([small, small, large, large, small, large])

    for overlap in (0, 180):
        chunks = chunk_text(text, max_chars=200, overlap_chars=overlap)
        assert chunks, overlap
        widest = max(len(chunk.text) for chunk in chunks)
        assert widest <= 200, f"overlap={overlap} produced a {widest}-character chunk"

    # The overlap still carries something where there is room for it. The fix is
    # therefore a trim, and the feature is not switched off.
    even = "\n\n".join(f"para {number} " + "word " * 10 for number in range(20))
    assert len(chunk_text(even, max_chars=200, overlap_chars=150)) > len(
        chunk_text(even, max_chars=200, overlap_chars=0)
    )


def test_the_overlap_never_repeats_a_whole_chunk() -> None:
    """The guard in `_tail`: a carry-over of everything would stop the walk from
    advancing."""
    text = "\n\n".join(f"paragraph {number} " + "word " * 30 for number in range(6))

    chunks = chunk_text(text, max_chars=200, overlap_chars=10_000)

    assert len({chunk.text for chunk in chunks}) == len(chunks)
    assert chunks[-1].end_line == len(text.splitlines())


def test_a_paragraph_bigger_than_a_chunk_is_split_on_line_boundaries() -> None:
    """A code block or a table. The span has to stay exact through the split."""
    text = "\n".join(f"line {number} " + "x" * 60 for number in range(20))

    chunks = chunk_text(text, max_chars=200, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    # Contiguous and complete: every line lands in exactly one chunk.
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 20
    for earlier, later in pairwise(chunks):
        assert later.start_line == earlier.end_line + 1


def test_a_single_line_too_long_to_be_a_chunk_is_cut_anyway() -> None:
    """X5: the bound was on the lines packed together, and not on the chunk.

    `_split_block` cut only on line boundaries. A line longer than `max_chars`
    had no cut to make and went out whole. Such lines are common: a minified
    bundle, a base64 blob, and a long CSV row are each one line of thousands of
    characters. Every one of them reached the embedder over its input limit and
    was silently truncated there. That is the failure of X4, by the route that
    X4 did not cover, and the docstring of the chunker promised that it could
    not happen.

    The pieces all claim the one line that they came from. That is the cost, and
    it is asserted here so that nobody has to discover it. The span is
    approximate for exactly the lines whose line number was never a useful
    pointer.

    Sabotage: drop the `bounds.past(len(line), own)` arm in `_split_block` and
    the first case returns one 782-character chunk against a bound of 100.
    """
    # A real long line: a data row that someone may search by name. X6 drops the
    # *machine* long lines, such as a minified bundle or an encoded blob. This
    # cut is for what remains, so the fixture must be something worth keeping.
    row = ",".join(f"Some Name {number},Austin TX,{number * 7}" for number in range(30))
    chunks = chunk_text(row, max_chars=100, overlap_chars=10)

    assert len(chunks) == len(row) // 100 + 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == row, "characters were lost in the cut"
    assert {(chunk.start_line, chunk.end_line) for chunk in chunks} == {(1, 1)}

    # It is the *last* resort. A paragraph whose lines each fit is still cut
    # between its lines, so the ordinary case keeps its exact span.
    ordinary = "\n".join(f"line {number} " + "x" * 60 for number in range(20))
    assert all(
        chunk.start_line != chunk.end_line or len(chunk.text) <= 80
        for chunk in chunk_text(ordinary, max_chars=200, overlap_chars=0)
    )

    # A long line among short lines does not push its neighbors over the bound.
    mixed = "short\n" + row + "\nshort again"
    assert all(
        len(chunk.text) <= 100 for chunk in chunk_text(mixed, max_chars=100, overlap_chars=10)
    )


def test_an_empty_document_yields_nothing() -> None:
    assert chunk_text("", max_chars=100, overlap_chars=0) == []
    assert chunk_text("\n\n   \n", max_chars=100, overlap_chars=0) == []


def _in_words(most: float) -> Weight:
    """A weight that counts words, so that a reader can add them up."""
    return Weight(of=lambda texts: [len(text.split()) for text in texts], most=most)


def test_a_chunk_keeps_to_a_weight_as_well_as_to_its_characters() -> None:
    """A profile sizes a passage in characters, and an embedder does not count
    characters. The packer therefore takes a second bound in the unit of the
    embedder, and every cut keeps to both bounds. When a dense document was
    sized in characters alone and converted, it came out over the limit, and all
    of it was packed again smaller.

    Here the bound is ten words for each chunk, with room in characters for many
    more:

    * Paragraphs are packed while they fit. A blank line between two paragraphs
      counts as one.
    * A paragraph that is heavier than a chunk is cut between its lines.
    * A line that is heavier than a chunk is cut inside itself.
    * Every span still names real lines.
    """
    paragraph = "one two three four"
    heavy = "\n".join(["a b c d e f"] * 3)
    line = " ".join(f"w{n}" for n in range(25))
    text = "\n\n".join([paragraph, paragraph, paragraph, heavy, line])
    weight = _in_words(10)

    chunks = chunk_paragraphs(paragraphs(text), max_chars=1_000, overlap_chars=0, weight=weight)

    assert weight.bound, "the weight ended a chunk and did not say so"
    counted = [len(chunk.text.split()) for chunk in chunks]
    assert counted[:5] == [8, 4, 6, 6, 6]
    # The weight decides where inside the line a cut lands. A word that is cut in
    # two counts twice, so the pieces are held to the bound and not to a count.
    assert len(counted) == 8 and all(count <= 10 for count in counted[5:])
    lines = text.splitlines()
    for chunk in chunks[:5]:
        assert chunk.text.splitlines() == lines[chunk.start_line - 1 : chunk.end_line]
    # The pieces of the one long line all claim that line, as they do when it is
    # cut by characters. They are never joined again with a blank line between.
    assert {(chunk.start_line, chunk.end_line) for chunk in chunks[5:]} == {(11, 11)}
    assert "".join(chunk.text for chunk in chunks[5:]) == line, "characters were lost in the cut"

    # A weight that never binds changes nothing, and reports that it did not bind.
    roomy = _in_words(10_000)
    assert chunk_paragraphs(
        paragraphs(text), max_chars=60, overlap_chars=20, weight=roomy
    ) == chunk_paragraphs(paragraphs(text), max_chars=60, overlap_chars=20)
    assert not roomy.bound


def test_a_long_line_is_weighed_about_once_however_many_pieces_it_makes() -> None:
    """Each piece that is cut off a line is weighed. The weight of what is left
    of the line is estimated from that piece, until the line is nearly done.
    When the rest of the line was weighed again at every cut, it cost one pass
    of a tokenizer for each piece: 18 s for one 100,000-character line of CJK,
    where `max_chars` let the line through whole."""
    weighed = 0

    def counted(texts: Sequence[str]) -> list[float]:
        nonlocal weighed
        weighed += sum(len(text) for text in texts)
        return [len(text.split()) for text in texts]

    line = " ".join(f"w{n}" for n in range(20_000))

    chunks = chunk_paragraphs(
        paragraphs(line), max_chars=len(line), overlap_chars=0, weight=Weight(counted, 50)
    )

    assert len(chunks) > 300
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == line, "characters were lost in the cut"
    assert weighed < 6 * len(line), f"weighed {weighed / len(line):.0f} times the line"


def test_what_is_carried_into_a_chunk_is_at_most_a_quarter_of_its_weight() -> None:
    """The overlap is also a budget in the unit of the weight. The budget is what
    is left after the block that starts the chunk, and never more than a quarter
    of a chunk. A bound that leaves room for a few sentences therefore does not
    spend that room on the last sentence."""
    text = "\n\n".join(["one two", "three four five six seven eight"] * 4)

    chunks = chunk_paragraphs(
        paragraphs(text), max_chars=1_000, overlap_chars=500, weight=_in_words(12)
    )

    assert all(len(chunk.text.split()) <= 12 for chunk in chunks)
    # Two words are carried, because a quarter of twelve is three. Six words are
    # never carried.
    assert [chunk.text.split()[0] for chunk in chunks[1:]] == ["one"] * (len(chunks) - 1)
    assert len(chunks) > 2


def test_the_head_of_a_text_is_as_much_of_its_start_as_the_weight_allows() -> None:
    """This is for a question, which should stay as nearly whole as possible. The
    head is all of the question when it fits. Otherwise it is the start of the
    question, cut where the weight runs out and then back to the end of a word.
    A text with no word boundary to cut at is cut anyway."""
    question = "which\n\n" + " ".join(f"word{n}" for n in range(40))

    assert head_of(question, _in_words(100)) == question
    head = head_of(question, _in_words(10))
    assert question.startswith(head) and question[len(head)].isspace()
    assert 8 <= len(head.split()) <= 10, "packed as paragraphs, the head was the first word"
    by_characters = Weight(of=lambda texts: [len(text) for text in texts], most=10)
    assert head_of("x" * 100, by_characters) == "x" * 10


# -------------------------------------------------------------------- store ----


def _store(root: Path, **kwargs: Any) -> TextIndex:  # noqa: ANN401
    store = TextIndex(root=root, model="stub-hashing:96", **kwargs)
    store.open()
    return store


def _indexed(store: TextIndex, embedder: HashingEmbedder, path: str, text: str) -> int:
    chunks = chunk_text(text, max_chars=200, overlap_chars=0)
    return store.add(path, chunks, embedder.encode([one.text for one in chunks], query=False))


def test_a_reopened_index_answers_the_same_question(tmp_path: Path) -> None:
    """The whole persistence claim: both halves come back, and agree."""
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    store.save()
    query = embedder.encode(["which containment tier does a deployment ask for"], query=True)
    before = store.search(query, 3)

    reopened = _store(tmp_path / "ix")

    assert reopened.stats()["chunks"] == store.stats()["chunks"]
    assert reopened.documents() == ["docs/w.md"]
    after = reopened.search(query, 3)
    assert [record.id for _, record in after] == [record.id for _, record in before]
    assert [record.text for _, record in after] == [record.text for _, record in before]


def test_retrieval_ranks_the_passage_that_shares_the_query_words_first(
    tmp_path: Path,
) -> None:
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)

    hits = store.search(
        embedder.encode(["bwrap Seatbelt confinement sandbox backend"], query=True), 3
    )

    assert hits, "nothing came back"
    assert "Seatbelt" in hits[0][1].text


def test_forgetting_a_document_removes_its_passages_from_both_halves(
    tmp_path: Path,
) -> None:
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    _indexed(store, embedder, "docs/other.md", "Something else entirely, about billing.")
    store.save()

    removed = store.forget("docs/w.md")
    store.save()

    assert removed > 0
    assert _store(tmp_path / "ix").documents() == ["docs/other.md"]


def test_a_reindexed_document_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """What makes "run it again after an edit" correct rather than cumulative."""
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    first = store.stats()["chunks"]

    store.forget("docs/w.md")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)

    assert store.stats()["chunks"] == first


def test_an_index_built_by_another_embedder_is_refused(tmp_path: Path) -> None:
    """A vector from one model means nothing to another, so this must not open."""
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    store.save()

    other = TextIndex(root=tmp_path / "ix", model="some-other-model")

    with pytest.raises(IndexMismatch, match="means nothing to the other"):
        other.open()


def test_a_sidecar_chunk_with_no_vector_is_dropped_on_open(tmp_path: Path) -> None:
    """The case of a crash between the two writes. It is reconciled toward the
    half that can be retrieved."""
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    store.save()
    sidecar = tmp_path / "ix" / "chunks.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["records"].append(
        {"id": 9_999, "path": "docs/ghost.md", "start_line": 1, "end_line": 2, "text": "ghost"}
    )
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    reopened = _store(tmp_path / "ix")

    assert "docs/ghost.md" not in reopened.documents()


def test_a_vector_the_sidecar_never_named_is_removed_on_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the crash between the two writes: the vectors were
    synced, and the sidecar that names them was never written.

    The ids of those vectors start at the `next_id` of the sidecar, which is
    where the next add allocates. turbovec refuses an id that it already holds.
    Every add after the crash therefore raised, until someone deleted the
    directory.

    Sabotage: skip `_remove_strays` in `open`, and the add raises.
    """
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    store.save()
    _indexed(store, embedder, "docs/lost.md", "A passage whose sidecar never landed.")

    monkeypatch.setattr(
        "ph_text_index._store.write_atomic", raising(OSError("the process died here"))
    )
    with pytest.raises(OSError, match="died"):
        store.save()
    monkeypatch.undo()

    reopened = _store(tmp_path / "ix")
    assert reopened.documents() == ["docs/w.md"]
    assert _indexed(reopened, embedder, "docs/next.md", "Billing, invoiced monthly.") > 0
    reopened.save()
    assert _store(tmp_path / "ix").documents() == ["docs/next.md", "docs/w.md"]


def _seam(embedder: HashingEmbedder) -> TextIndexSeam:
    """The service on its own, as a second root or process would hold it."""
    seam = TextIndexSeam(ctx=Context(), config=Config())
    seam.provider = embedder
    return seam


async def _index_one(seam: TextIndexSeam, workspace: Path, path: str, text: str) -> None:
    async with seam.writing(workspace) as store:
        assert store is not None
        _indexed(store, HashingEmbedder(), path, text)
        store.save()


async def test_two_writers_sharing_a_directory_both_keep_their_documents(
    tmp_path: Path,
) -> None:
    """Every root and every process on a workspace shares the directory of that
    workspace. Each one held its own copy and saved it whole. The second save
    therefore described only its own documents, and the vectors of the first
    writer were left with no text.

    Sabotage: take the copy in `writing` without asking whether it is stale, and
    the first writer's document is gone.
    """
    embedder, tree = HashingEmbedder(), tmp_path / "tree"
    first, second = _seam(embedder), _seam(embedder)
    await second.index(tree)  # a copy loaded before the other writer's change

    await _index_one(first, tree, "docs/w.md", DOCUMENT)
    await _index_one(second, tree, "docs/billing.md", "Invoices are issued monthly.")

    assert _store(first.root(tree)).documents() == ["docs/billing.md", "docs/w.md"]
    assert (await first.index(tree)).documents() == ["docs/billing.md", "docs/w.md"]


async def test_a_second_writer_waits_for_the_first(tmp_path: Path) -> None:
    """One change at a time across seams, and not only within one seam. A reload
    before the change does not help a writer whose copy changes under it during
    the change.

    Sabotage: take no file lock in `writing`, and the second writer enters while
    the first holds it.
    """
    embedder, tree = HashingEmbedder(), tmp_path / "tree"
    first, second = _seam(embedder), _seam(embedder)
    entered = anyio.Event()

    async def wait_to_write() -> None:
        async with second.writing(tree):
            entered.set()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            async with first.writing(tree):
                group.start_soon(wait_to_write)
                await anyio.sleep(0.3)
                assert not entered.is_set(), "two writers held one directory"
            await entered.wait()


async def test_a_search_reads_what_a_writer_has_saved_and_changes_no_file(
    tmp_path: Path,
) -> None:
    """A search never waits for a run, so it reads while a writer holds the lock.
    It reads what that writer has saved. Between the two writes of a save, the
    vectors are ahead of the sidecar. A reader then answers for the documents
    that the sidecar names and leaves the file as it is. The vectors that it
    cannot name belong to the writer, which is about to name them.

    Sabotage: open as a writer does (`readonly=False`), and the reader removes
    the vectors the writer synced; the writer's sidecar then names passages the
    index no longer holds.
    """
    embedder, tree = HashingEmbedder(), tmp_path / "tree"
    reader, writer = _seam(embedder), _seam(embedder)

    async with writer.writing(tree) as store:
        assert store is not None
        _indexed(store, embedder, "docs/a.md", DOCUMENT)
        store.save()
        assert (await reader.index(tree)).documents() == ["docs/a.md"]

        _indexed(store, embedder, "docs/w.md", DOCUMENT)
        assert store._index is not None
        store._index.sync(str(store.index_path))
        synced = store.index_path.read_bytes()
        assert (await reader.index(tree)).documents() == ["docs/a.md"]
        assert store.index_path.read_bytes() == synced, "a reader changed the writer's file"

        store.save()
        assert (await reader.index(tree)).documents() == ["docs/a.md", "docs/w.md"]

    again = writer._store(writer.root(tree))
    again.open()
    assert sorted(again.documents()) == ["docs/a.md", "docs/w.md"]


async def test_each_workspace_has_its_own_index(tmp_path: Path) -> None:
    """What an index holds belongs to one tree: documents named relative to that
    tree, one version-control token, and a sweep that drops what that tree no
    longer has. When two workspaces used one directory, each deleted the
    documents of the other in turn.

    Sabotage: key `root` by the embedder alone, and one workspace's document is in
    the other's index.
    """
    seam = _seam(HashingEmbedder())
    here, there = tmp_path / "here", tmp_path / "there"

    await _index_one(seam, here, "docs/w.md", DOCUMENT)

    assert (await seam.index(there)).documents() == []
    assert (await seam.index(here)).documents() == ["docs/w.md"]
    assert seam.root(here).parent.parent == seam.root(there).parent.parent == index_base()


async def test_a_change_records_the_workspace_its_index_serves(tmp_path: Path) -> None:
    """The record that lets a sweep remove the index of a worktree that is gone.
    The directory names the workspace, which a digest cannot do
    (`ph.workspace_caches`).

    Sabotage: drop the `use` call from `writing`, and nothing is recorded.
    """
    seam = _seam(HashingEmbedder())
    here = tmp_path / "here"

    await _index_one(seam, here, "docs/billing.md", "Invoices are issued monthly.")

    recorded = seam.root(here).parent / WORKSPACE_FILE
    assert recorded.read_text(encoding="utf-8") == str(here)


async def test_an_index_goes_where_its_caller_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`$PH_TOOL_STATE`: a session names a directory in the scratch directory of
    the calling agent, and a person names one with `--state`. The index is then
    in that directory, under a directory for the embedder, so a second model
    still gets a second index. The index is not recorded for the sweep. The
    sweep removes what nobody owns, and this directory has an owner."""
    state = tmp_path / "scratch" / "text-index"
    monkeypatch.setenv(STATE_ENV, str(state))
    seam, here = _seam(HashingEmbedder()), tmp_path / "here"

    await _index_one(seam, here, "docs/w.md", DOCUMENT)

    assert seam.root(here) == state / digest(seam.embedder.name)
    assert seam.root(here) == seam.root(tmp_path / "there"), "the caller keys it, not the root"
    assert (await seam.index(here)).documents() == ["docs/w.md"]
    assert not (state / WORKSPACE_FILE).exists()
    assert not index_base().exists()


def test_a_filter_that_matches_nothing_returns_nothing_rather_than_raising(
    tmp_path: Path,
) -> None:
    """turbovec raises on an empty allowlist, and it is right to. It cannot tell
    "nothing allowed" from "no filter". This layer gives the empty answer."""
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)

    assert store.search(embedder.encode(["anything"], query=True), 3, allowed=[]) == []


def test_a_path_filter_restricts_the_result_to_that_subtree(tmp_path: Path) -> None:
    embedder = HashingEmbedder()
    store = _store(tmp_path / "ix")
    _indexed(store, embedder, "docs/w.md", DOCUMENT)
    _indexed(store, embedder, "notes/w.md", DOCUMENT)

    hits = store.search(
        embedder.encode(["containment tier"], query=True),
        10,
        allowed=store.ids_under(["notes"]),
    )

    assert hits
    assert {record.path for _, record in hits} == {"notes/w.md"}


def test_searching_an_empty_index_is_empty_and_not_an_error(tmp_path: Path) -> None:
    store = _store(tmp_path / "ix")

    assert store.search(HashingEmbedder().encode(["anything"], query=True), 5) == []
    assert store.stats()["calibration"] == "empty"


# ---------------------------------------------------------------- the whole row ----


async def test_without_an_embedder_a_call_says_so(mount: MountProfile, tmp_path: Path) -> None:
    """The command always mounts one embedder row, so only a test builds this
    profile. The tools say what is missing and do not fail in some other way.
    The profile of a session with no embedder offers no tools at all. That rule
    belongs to the session rows (`test_text_index_row.py`)."""
    (tmp_path / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx = await mount(ROW)

    refused = await run_tool(ctx, "text_index", {"paths": ["w.md"]}, agent=_agent(ctx))

    assert refused.error is not None and refused.error.info is not None
    assert refused.error.info["code"] == "TEXT_INDEX_FAILED"
    assert "no embedder is registered" in refused.error.message


async def test_an_embedder_claimed_anywhere_brings_both_tools(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx, _ = await _mounted(mount, tmp_path)

    assert ctx.require(TOOLS).get("text_index", scope=DEPLOYMENT) is not None
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None


async def test_the_rows_scope_is_what_a_bare_call_indexes(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A profile points the row at a tree and says what to leave out. `text_index`
    with no arguments indexes exactly that: its `paths` and its `include`,
    without its `exclude`."""
    for tree in ("docs", "notes"):
        (tmp_path / tree / "drafts").mkdir(parents=True)
        (tmp_path / tree / "guide.md").write_text(DOCUMENT, encoding="utf-8")
        (tmp_path / tree / "guide.txt").write_text(DOCUMENT, encoding="utf-8")
        (tmp_path / tree / "drafts" / "wip.md").write_text(DOCUMENT, encoding="utf-8")
    scope = {"paths": ["docs"], "include": ["**/*.md", "**/*.txt"], "exclude": ["**/drafts/**"]}
    ctx, _ = await _mounted(mount, tmp_path, index=scope)

    indexed = await run_tool(ctx, "text_index", {}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert sorted(indexed.value["documents"]) == ["docs/guide.md", "docs/guide.txt"]


async def test_prose_is_what_a_row_takes_when_nobody_says(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Markdown by default, and a call's `exclude` narrows it."""
    (tmp_path / "docs").mkdir()
    for name in ("a.md", "b.md", "c.py"):
        (tmp_path / "docs" / name).write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path)

    indexed = await run_tool(ctx, "text_index", {"exclude": ["**/b.md"]}, agent=_agent(ctx))

    assert indexed.value["documents"] == ["docs/a.md"]


async def test_indexing_a_directory_then_searching_finds_the_passage(
    mount: MountProfile, tmp_path: Path
) -> None:
    """End to end through both tools and the real index."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "workspaces.md").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "docs" / "billing.md").write_text(
        "# Billing\n\nInvoices are issued monthly and paid in arrears.\n", encoding="utf-8"
    )
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["total_documents"] == 2
    assert indexed.value["chunks_added"] > 2
    assert sorted(indexed.value["documents"]) == ["docs/billing.md", "docs/workspaces.md"]

    found = await run_tool(
        ctx,
        "text_search",
        {"query": "which sandbox backend confines an agent on Linux", "k": 2},
        agent=agent,
    )

    assert not found.is_error, text_of(found.content)
    hits = found.value["hits"]
    assert hits, text_of(found.content)
    assert hits[0]["path"] == "docs/workspaces.md"
    assert "bwrap" in hits[0]["text"]
    # The pointer that the row exists to return.
    assert 1 <= hits[0]["start_line"] <= hits[0]["end_line"] <= len(DOCUMENT.splitlines())
    assert f":{hits[0]['start_line']}-{hits[0]['end_line']}" in text_of(found.content)


async def test_a_hits_line_span_names_the_lines_in_the_real_file(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The span must survive the relative naming of `glob` and the rejoin."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "workspaces.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    found = await run_tool(
        ctx, "text_search", {"query": "containment tier advisory worktree"}, agent=agent
    )

    hit = found.value["hits"][0]
    lines = (tmp_path / hit["path"]).read_text(encoding="utf-8").splitlines()
    span = "\n".join(lines[hit["start_line"] - 1 : hit["end_line"]])
    assert hit["text"].split("\n\n")[0] in span


async def test_a_paths_filter_narrows_the_search(mount: MountProfile, tmp_path: Path) -> None:
    for directory in ("docs", "notes"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["docs", "notes"]}, agent=agent)

    found = await run_tool(
        ctx,
        "text_search",
        {"query": "containment tier", "paths": ["notes"], "k": 10},
        agent=agent,
    )

    assert {hit["path"] for hit in found.value["hits"]} == {"notes/w.md"}
    assert found.value["searched"] < _opened(ctx).stats()["chunks"]


async def test_the_spellings_a_model_writes_for_a_path_all_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The filter is matched against stored names, so it must be a stored name
    (X2).

    `index_tool` stores documents under the names that `fs.collect` returns,
    which are the workspace-relative names of `fs.named()`. `paths` were matched
    exactly as the model typed them. `["./docs"]` and an absolute path therefore
    matched nothing, and the reply said "among 0 indexed passages". That is not
    an error, and it is not an empty result from a real filter. It is a wrong
    answer that looks like an empty index.

    `["."]` deserves its own case. It is how a model says "everywhere". No
    document is stored under that name, so it excluded everything, which is the
    opposite of what it asks for.
    """
    for directory in ("docs", "notes"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["docs", "notes"]}, agent=agent)
    everything = int(_opened(ctx).stats()["chunks"])

    async def search(paths: list[str]) -> Any:  # noqa: ANN401
        found = await run_tool(
            ctx, "text_search", {"query": "containment tier", "paths": paths, "k": 10}, agent=agent
        )
        return found.value

    dotted = await search(["./notes"])
    assert {hit["path"] for hit in dotted["hits"]} == {"notes/w.md"}

    absolute = await search([str(tmp_path / "notes")])
    assert {hit["path"] for hit in absolute["hits"]} == {"notes/w.md"}

    # `.` is the workspace, which means every document and not none of them.
    here = await search(["."])
    assert here["searched"] == everything
    assert {hit["path"] for hit in here["hits"]} == {"docs/w.md", "notes/w.md"}


async def test_forget_removes_a_document_through_the_tool(
    mount: MountProfile, tmp_path: Path
) -> None:
    (tmp_path / "a.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["a.md"]}, agent=agent)

    removed = await run_tool(ctx, "text_index", {"paths": ["a.md"], "forget": True}, agent=agent)

    assert not removed.is_error, text_of(removed.content)
    assert removed.value["chunks_removed"] > 0
    assert removed.value["total_chunks"] == 0
    assert "Removed" in text_of(removed.content)


async def test_a_document_past_the_size_limit_is_reported_and_not_an_error(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A call over a directory must not fail because it found a bundle. The
    caller must also learn what was skipped, or a quiet corpus looks like a
    quiet failure."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "small.md").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "docs" / "huge.md").write_text("x" * 5_000, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_bytes=1_000)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["documents"] == ["docs/small.md"]
    assert [one["path"] for one in indexed.value["skipped"]] == ["docs/huge.md"]
    assert "skipped docs/huge.md" in text_of(indexed.content)


async def test_a_search_over_an_unindexed_corpus_says_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The empty result a model would otherwise read as "no such thing"."""
    ctx, _ = await _mounted(mount, tmp_path)

    found = await run_tool(ctx, "text_search", {"query": "anything at all"}, agent=_agent(ctx))

    assert not found.is_error
    assert found.value["hits"] == []
    assert "Index the documents first" in text_of(found.content)


async def test_indexing_reads_through_the_fs_seam(mount: MountProfile, tmp_path: Path) -> None:
    """The claim that makes this a tool and not an exfiltration primitive (I-9).

    A screen that is registered on `ctx.fs` decides what the walk may show. A
    document that the screen prunes must never reach the index. That holds
    because the row reads through the seam. It would plainly not hold if the row
    walked the tree itself.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "public.md").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "docs" / "secret.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    ctx.require(FS).screen(
        lambda path, name, agent, is_dir: "skip" if name == "secret.md" else "yield",
        scope=ctx,
    )

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert indexed.value["documents"] == ["docs/public.md"]


async def test_the_row_reports_itself_to_doctor(mount: MountProfile, tmp_path: Path) -> None:
    """What the `status` of the command prints. Before anything is indexed, it
    says so. Afterwards it gives the counts. It reads them from disk, because a
    command that is asked for its status has opened nothing."""
    (tmp_path / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, embedder = await _mounted(mount, tmp_path)

    before = report_section(ctx, "Text index")
    await run_tool(ctx, "text_index", {"paths": ["w.md"]}, agent=_agent(ctx))
    ctx.require(TEXT_INDEX)._indexes.clear()
    after = report_section(ctx, "Text index")

    assert before["embedder"] == embedder.name
    assert before["state"] == "not built — run text_index"
    assert after["documents"] == "1"


async def test_doctor_says_when_no_embedder_is_registered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The one thing that an operator must be told when the tools are simply absent."""
    ctx = await mount(ROW)

    assert "none registered" in report_section(ctx, "Text index")["embedder"]


# ------------------------------------------------------------------ Code Mode ----


async def test_code_mode_hands_the_model_both_tools_through_the_sdk(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Under Code Mode, "can the RLM use this" is "are these in the SDK listing".

    The profile is composed from the rows of Code Mode itself, and not by
    mounting `rlm`, because the claim is about any Code Mode deployment. The
    embedder is claimed by hand here, for the reason that `_mounted` does it:
    the real provider row would load a model to prove a fact about a prompt.
    """
    ctx = await mount(
        {"id": "tools", "config": {"mode": "code"}},
        {"id": "tools-code-mode", "name": "tools-code-mode"},
        {"id": "code-runtime-stub", "name": "code-runtime-stub"},
        ROW,
    )
    ctx.require(TEXT_INDEX).register(HashingEmbedder())
    agent = _agent(ctx)

    view = ctx.require(TOOLS).view(scope=agent.ctx)
    assert view.mode == "code"
    assert {"text_index", "text_search"} <= set(view.visible)

    prompt = await ctx.require(SYSTEM_PROMPT).assemble(agent=agent, scope=agent.ctx)
    sdk = dict(prompt.sections)["tools:sdk"]
    assert "async def tools.text_index(" in sdk
    assert "async def tools.text_search(" in sdk
    assert "Search the text index by meaning" in sdk


def test_a_model_that_brings_its_own_code_is_reachable_by_config() -> None:
    """`nomic-embed-text-v1.5`, and every other model with an `auto_map`.

    Its `config.json` maps `AutoModel` to `nomic-ai/nomic-bert-2048`, so
    `sentence-transformers` will not load it without `trust_remote_code=True`.
    Without this setting the model is simply unreachable, however the row is
    configured. The setting is off by default, because granting it executes
    Python that was downloaded from a model repository, in this process.

    The embedder is constructed, not loaded. The assertion is that the
    configuration reaches the embedder. Nobody should wait for a test that
    downloads half a gigabyte to prove that a keyword argument was passed.
    """
    from ph_text_index._embed import SentenceTransformerEmbedder

    default = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert default.trust_remote_code is False

    nomic = SentenceTransformerEmbedder(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        trust_remote_code=True,
    )
    assert nomic.trust_remote_code is True


def test_the_prefixes_are_in_the_identity_but_the_trust_flag_is_not() -> None:
    """Two different kinds of setting, and the index directory must tell them apart.

    A prefix changes where a vector lands. The `search_document: ` index of
    nomic is a different space from the unprefixed index of the same model. A
    prefix therefore belongs in the identity, and a switch gets a fresh index
    and not wrong neighbors. `trust_remote_code` changes what may *load*, so an
    index must not be invalidated because an operator granted or revoked it.
    """
    from ph_text_index._embed import SentenceTransformerEmbedder

    plain = SentenceTransformerEmbedder(model_name="m")
    prefixed = SentenceTransformerEmbedder(
        model_name="m", query_prefix="search_query: ", document_prefix="search_document: "
    )
    trusted = SentenceTransformerEmbedder(model_name="m", trust_remote_code=True)

    assert plain.name != prefixed.name, "a prefix must move the index"
    assert plain.name == trusted.name, "a trust grant must not move the index"


async def test_switching_the_model_gets_its_own_index_directory(
    mount: MountProfile, tmp_path: Path
) -> None:
    """What makes it safe to try a bigger model: no collision, and no need to
    embed again with the old model to switch back.

    The slot is set directly and not through `register`. The claim is about how
    `root()` keys the directory, and `register` would bring in the whole
    lifecycle of the provider to make it. Registering twice without disposing
    the first is misuse, and `claim_slot` refuses it. The first draft of this
    test did exactly that. It then reported a tool-name collision as if it were
    a finding about the index path.
    """
    ctx = await mount({**ROW})
    seam = ctx.require(TEXT_INDEX)

    seam.provider = HashingEmbedder(dim=96)
    small = seam.root(tmp_path)
    seam.provider = HashingEmbedder(dim=384)
    large = seam.root(tmp_path)

    assert small != large, "two embedders shared one index directory"
    assert small.parent == large.parent, "both still under the row's cache root"


# --------------------------------------------------- provisioning, and skills ----


@dataclass(slots=True)
class LoadableEmbedder(HashingEmbedder):
    """A stub that has a model to load, so that the provisioning paths are
    exercised.

    `HashingEmbedder` deliberately has neither `load` nor `ready`. It has the
    shape of an *endpoint*, with nothing to download, so it cannot test the
    branch that matters here. This stub counts loads and can be told to fail,
    which covers everything that `provision` and `preload` do.
    """

    fails: str = ""
    loads: int = 0
    loaded: bool = False
    cache_folder: str = "/nowhere"
    """`LocalWeights` declares this member, and `isinstance` checks it.

    The first draft of this double omitted it. The Protocol then reported the
    double as an endpoint-backed embedder with nothing to download. That is
    exactly the misnamed-member case that a `getattr` probe could not have
    caught."""

    @property
    def name(self) -> str:
        return f"stub-loadable:{self.dim}"

    def load(self) -> int:
        self.loads += 1
        if self.fails:
            raise RuntimeError(self.fails)
        self.loaded = True
        return self.dim

    def ready(self) -> bool:
        return self.loaded

    def cached(self) -> bool:
        return self.loaded


@dataclass(slots=True)
class UnfetchedEmbedder(LoadableEmbedder):
    """A model that is not on disk and cannot be reached. A confined command gets
    this when the hub is not among the hosts that its deployment allows."""

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        raise RuntimeError("403 Forbidden")


async def test_the_command_reports_and_then_provisions(mount: MountProfile, tmp_path: Path) -> None:
    """A person types this command. It costs no model turn, by the rule of the
    seam.

    `status` before `install` is the purpose of the pair. "Can I download it" is
    a question to answer *before* an agent is in the middle of a turn. That is
    the reason why this is not left to the first `text_index` call.
    """
    ctx = await mount(ROW)
    embedder = LoadableEmbedder()
    ctx.require(TEXT_INDEX).register(embedder)
    agent = _agent(ctx)

    before = await ctx.require(COMMANDS).dispatch(
        "/text-index status", agent=agent, scope=agent.ctx
    )
    assert before is not None and "not downloaded" in before
    assert embedder.loads == 0, "status must not download anything"

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)
    assert said is not None and "is ready (96-dimensional)" in said
    assert embedder.loads == 1

    after = await ctx.require(COMMANDS).dispatch("/text-index status", agent=agent, scope=agent.ctx)
    assert after is not None and "loaded" in after and "not downloaded" not in after


async def test_a_failed_install_says_what_upstream_said(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The `einops` case, and why the message is passed through unchanged.**

    `nomic-embed-text-v1.5` downloads its weights and its remote modeling code
    successfully. It then fails at import with "requires the following packages
    that were not found in your environment: einops". That is a transitive
    dependency that only the code of the model knows about. The sentence is the
    useful half, so the command reports it and does not reword it. It is raised
    under a code of its own, so the command exits with a failure. A session that
    asked for the model at its mount (`preload`) refuses on that failure, with
    this sentence.
    """
    from ph.tools.errors import HarnessError

    ctx = await mount(ROW)
    ctx.require(TEXT_INDEX).register(
        LoadableEmbedder(fails="requires the following packages ...: einops")
    )
    agent = _agent(ctx)

    with pytest.raises(HarnessError) as refused:
        await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert refused.value.code == "TEXT_INDEX_MODEL_FAILED"
    assert "did not load" in str(refused.value)
    assert "einops" in str(refused.value), "the actionable half of the message was dropped"


async def test_a_model_that_cannot_be_fetched_says_how_a_person_fetches_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A tool call fetches a model that is not on disk. A call made for a confined
    agent often cannot fetch it, because the network of that agent is the
    allowlist of the deployment. The upstream sentence is kept and the fix is
    named, under a code of its own. The fix is `/text-index install`, which is a
    command that a person runs, and it runs outside the sandbox."""
    (tmp_path / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx = await mount(ROW)
    ctx.require(TEXT_INDEX).register(UnfetchedEmbedder())

    refused = await run_tool(ctx, "text_index", {"paths": ["w.md"]}, agent=_agent(ctx))

    assert refused.is_error
    assert refused.error is not None and refused.error.info is not None
    assert refused.error.info["code"] == "TEXT_INDEX_MODEL_MISSING"
    assert "403 Forbidden" in refused.error.message
    assert "/text-index install" in refused.error.message


async def test_an_embedder_with_nothing_to_download_says_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An endpoint-backed embedder has no weights, and must not be made to pretend."""
    ctx = await mount(ROW)
    ctx.require(TEXT_INDEX).register(HashingEmbedder())
    agent = _agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert said is not None and "needs no download" in said


async def test_the_command_is_absent_rather_than_broken_without_a_provider(
    mount: MountProfile, tmp_path: Path
) -> None:
    """It is still registered, because a person may type it to find out *why*
    nothing works."""
    ctx = await mount(ROW)
    agent = _agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert said is not None and "No embedder is registered" in said


async def test_the_command_does_not_preload_whatever_its_config_says(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`preload` belongs to the session. The session asks this command to
    `install` one time, at its own mount, and refuses on the answer
    (`test_text_index_row.py`). When the command read `preload` as well, it
    loaded the model to answer `status`. For a person whose config file set it,
    the command loaded the model on every call. The command therefore mounts and
    offers its tools, and the model is loaded where it is needed: by `install`,
    or by the first embed.

    The load is patched on the class and is not pointed at a false model id. A
    404 from the hub would make this a network test that proves a local branch.
    """
    from ph.tools.errors import HarnessError
    from ph_text_index._embed import SentenceTransformerEmbedder

    def explode(self: object) -> int:
        raise RuntimeError("no network")

    monkeypatch.setattr(SentenceTransformerEmbedder, "load", explode)

    ctx = await mount(
        ROW,
        {
            "id": "text-index-local",
            "name": "text-index-local-engine",
            "config": {"preload": True},
        },
    )
    agent = _agent(ctx)

    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None
    with pytest.raises(HarnessError, match="no network"):
        await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)


async def test_the_weights_land_under_the_cache_root(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not `~/.cache/huggingface`, which no pH root covers.

    If `sentence-transformers` chose the place, a gigabyte of weights would be
    in a directory that `phern doctor` never mentions and that
    `rm -rf $PH_CACHE` would not reclaim.
    """
    from ph_text_index._config import LocalConfig, weights_cache
    from ph_text_index._embed import SentenceTransformerEmbedder

    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    from ph.paths import resolve_roots

    default = LocalConfig()
    assert default.cache == "", "the row decides, so the field stays empty"
    expected = str(resolve_roots().cache / "models")
    assert str(weights_cache(default)) == expected
    assert weights_cache(LocalConfig(cache=str(tmp_path / "w"))) == tmp_path / "w"

    built = SentenceTransformerEmbedder(model_name="m", cache_folder=expected)
    assert built.cache_folder == expected
    assert str(tmp_path) in built.cache_folder


# ------------------------------------------------- the version-control filter ----


@pytest.mark.needs_git
async def test_git_stops_an_unchanged_document_being_re_embedded(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**This saves the expensive work.** This loop had no digest short-circuit
    of its own. Before the filter, an unchanged document was therefore read
    again, cut again, and embedded again on every call. That was measured at
    20 s of MiniLM for `docs/`, or 185 s under nomic.

    The count is taken on the embedder, because the embedder is the cost. A test
    that watched the passage counts would pass against an implementation that
    embedded again and then wrote the same vectors back.
    """
    ctx, embedder = await _mounted(mount, tmp_path)
    root = await git_repo(ctx, tmp_path)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "w.md").write_text(DOCUMENT, encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add docs")
    agent = _agent(ctx)

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    assert first.value["chunks_added"] > 0, text_of(first.content)
    after_first = embedder.calls

    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert again.value["unchanged"] == 1
    assert again.value["chunks_added"] == 0
    assert embedder.calls == after_first, "the document was re-embedded anyway"
    assert "not re-embedded" in text_of(again.content)


@pytest.mark.needs_git
async def test_an_edited_document_is_re_embedded(mount: MountProfile, tmp_path: Path) -> None:
    """The safe direction. It must also hold for an *uncommitted* edit. That is
    the job of `status`, because the index of git still holds the old blob id."""
    ctx, embedder = await _mounted(mount, tmp_path)
    root = await git_repo(ctx, tmp_path)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "w.md").write_text(DOCUMENT, encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add docs")
    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    before = embedder.calls

    (root / "docs" / "w.md").write_text(
        DOCUMENT + "\n\nA new paragraph about invoices.\n", encoding="utf-8"
    )
    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert again.value["unchanged"] == 0
    assert again.value["chunks_added"] > 0
    assert embedder.calls > before, "an edited document must be re-embedded"


@pytest.mark.needs_jj
async def test_jj_stops_an_unchanged_document_being_re_embedded(
    mount: MountProfile, tmp_path: Path
) -> None:
    """No commit is needed. jj snapshots the working copy on any command."""
    ctx, embedder = await _mounted(mount, tmp_path)
    root = await jj_repo(ctx, tmp_path)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "w.md").write_text(DOCUMENT, encoding="utf-8")

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    assert first.value["chunks_added"] > 0, text_of(first.content)
    after_first = embedder.calls

    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert again.value["unchanged"] == 1
    assert embedder.calls == after_first


@pytest.mark.needs_jj
async def test_a_document_the_index_never_held_is_not_vouched_for(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The guard in front of `vouches_for`, and why it cannot be skipped.**

    jj answers with no per-file ids. `vouches_for` therefore proves only that a
    path has not *changed* since the token of the caller. That is equally true
    of a document that was never indexed at all. Index one file, then widen the
    call to the directory. Without `store.holds`, every other document is
    "unchanged" on the evidence of a token that was stored before anything
    looked at those documents, and they stay out of the index permanently. The
    failure is silent and permanent, and that combination is why it has a test
    of its own.
    """
    ctx, embedder = await _mounted(mount, tmp_path)
    root = await jj_repo(ctx, tmp_path)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "first.md").write_text(DOCUMENT, encoding="utf-8")
    (root / "docs" / "later.md").write_text(DOCUMENT, encoding="utf-8")

    # Only one of the two, so a token is stored while `later.md` is unindexed.
    narrow = await run_tool(ctx, "text_index", {"paths": ["docs/first.md"]}, agent=_agent(ctx))
    assert narrow.value["documents"] == ["docs/first.md"], text_of(narrow.content)

    wide = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert wide.value["documents"] == ["docs/later.md"], "the unindexed document must be read"
    assert wide.value["unchanged"] == 1, "and the one already held is still skipped"
    assert wide.value["chunks_added"] > 0
    assert embedder.calls > 0

    # And it is really searchable, not merely counted.
    found = await run_tool(
        ctx, "text_search", {"query": "containment tier advisory worktree"}, agent=_agent(ctx)
    )
    assert any(hit["path"] == "docs/later.md" for hit in found.value["hits"]), text_of(
        found.content
    )


async def test_without_version_control_a_document_is_known_by_its_own_digest(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A tree with no git and no jj, such as a tarball or a notes directory,
    embedded every document again on every call, because only the version
    control could say "unchanged". The text of each document now says it. The
    document is read, found to be what was indexed, and not embedded again. An
    edit is still an edit.

    Sabotage: drop the digest comparison in `index_run`, and the second run
    embeds the document it already holds.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, embedder = await _mounted(mount, tmp_path)

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    embedded = embedder.calls
    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    assert first.value["chunks_added"] > 0
    assert (again.value["unchanged"], again.value["chunks_added"]) == (1, 0)
    assert embedder.calls == embedded, "an unchanged document was embedded again"

    (tmp_path / "docs" / "w.md").write_text(DOCUMENT + "\nA new paragraph.\n", encoding="utf-8")
    edited = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    assert edited.value["documents"] == ["docs/w.md"]
    assert embedder.calls > embedded


async def test_a_change_to_how_documents_are_cut_cuts_them_again(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The digest covers the cut as well as the text. Otherwise, a profile that
    halves the passage size would be told that every document is unchanged, and
    would continue to search passages of the old size."""
    (tmp_path / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path)
    await run_tool(ctx, "text_index", {"paths": ["w.md"]}, agent=_agent(ctx))
    await ctx.drain()
    await ctx.dispose()

    recut, _ = await _mounted(mount, tmp_path, max_chars=120, overlap_chars=0)
    again = await run_tool(recut, "text_index", {"paths": ["w.md"]}, agent=_agent(recut))

    assert again.value["documents"] == ["w.md"]
    assert again.value["unchanged"] == 0


async def test_a_run_that_is_stopped_keeps_what_it_embedded(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A run used to save one time, at the end. A run that was stopped at the
    ninth minute of ten therefore kept nothing, and the next run embedded all of
    it again. What a stopped run embedded is complete: the passages and the
    vectors of each document are together. It is therefore saved on the way out.
    The next run finds those documents by their digests and embeds only the
    others.

    Sabotage: drop the save in `index_run`'s `except`, and the second run embeds
    all three.
    """
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.md").write_text(f"# {name}\n\nAbout {name}.\n", encoding="utf-8")
    ctx = await mount(ROW)
    embedder = FailingAfter(limit=2)
    ctx.require(TEXT_INDEX).register(embedder)

    stopped = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))
    embedder.limit = 100
    resumed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert stopped.is_error and "out of budget" in text_of(stopped.content)
    assert resumed.value["unchanged"] == 2
    assert resumed.value["documents"] == ["c.md"]
    assert resumed.value["total_documents"] == 3


@dataclass(slots=True)
class FailingAfter(HashingEmbedder):
    """An embedder that fails after it has been asked `limit` times. It gives a
    run that is stopped part of the way through, without a process to kill."""

    limit: int = 2

    def encode(self, texts: Sequence[str], *, query: bool) -> Vectors:
        if self.calls >= self.limit:
            raise RuntimeError("out of budget")
        return HashingEmbedder.encode(self, texts, query=query)


async def test_a_deleted_document_leaves_the_index_on_the_next_sweep(
    mount: MountProfile, tmp_path: Path
) -> None:
    """X3: the walk never compared itself with what the index already held.

    A document that was removed from the tree kept its passages until someone
    ran `forget` by hand, so search continued to answer with text that is not
    there. This is worse here than for the code graph. A passage is *quoted back
    to the model* as evidence, so a stale passage is not a broken pointer. It is
    a false claim about what the project says.

    Only a full sweep removes documents. That is the second half of the test: a
    narrowed walk must not empty the index, or
    `text_index paths=["docs/one.md"]` would delete everything else.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "workspaces.md").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "docs" / "billing.md").write_text(
        "# Billing\n\nInvoices are issued monthly and paid in arrears.\n", encoding="utf-8"
    )
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["."]}, agent=agent)

    (tmp_path / "docs" / "billing.md").unlink()
    swept = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=agent)

    assert not swept.is_error, text_of(swept.content)
    assert swept.value["total_documents"] == 1, "the deleted document kept its passages"

    found = await run_tool(ctx, "text_search", {"query": "invoices arrears", "k": 3}, agent=agent)
    assert all("billing" not in one["path"] for one in found.value["hits"]), (
        "search quoted a passage from a document that is gone"
    )


async def test_a_spilled_result_can_be_indexed_when_it_is_asked_for(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A large result is a file, and searching it is the useful thing to do.

    `tool-result-offload` writes an oversized result to `$PH_HOME/spill` and
    gives the model the path. That path is outside the workspace, so it is not
    in the default walk. That is deliberate. Embedding every spilled result of
    every session, unasked, costs minutes of model time for output that nobody
    has searched.

    It must work when it *is* asked for, and this test checks that it does. An
    absolute path resolves through `collect` like any other path. The request
    can therefore be made with the tools that already exist, and needs no new
    setting. The `glob` is explicit because spilled blobs are named by digest
    and have no extension, so the `**/*.md` default of the row would match none
    of them.
    """
    spill = tmp_path / "spill" / "large_tool_results"
    spill.mkdir(parents=True)
    (spill / "abc123-call-7").write_text(DOCUMENT, encoding="utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": [str(spill)], "glob": "*"}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["total_documents"] == 1

    found = await run_tool(
        ctx,
        "text_search",
        {"query": "which sandbox backend confines an agent on Linux", "k": 2},
        agent=agent,
    )
    assert not found.is_error, text_of(found.content)
    assert found.value["hits"], "a spilled result was indexed but cannot be found"


async def test_a_workspace_sweep_does_not_forget_an_indexed_spill(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The sweep of X3 and this must agree, and the existence check is why they
    do.

    A later `text_index .` walks the workspace, and a spill is not in the
    workspace. The first version of X3 compared the index with the *walk*. With
    that version, the next ordinary index call would have forgotten every
    spilled document, and the searchable copy that the model has of a large
    result would have disappeared for a reason that nothing reported. The two
    work together because the sweep asks the filesystem.
    """
    spill = tmp_path / "spill"
    spill.mkdir()
    (spill / "result.txt").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("# Note\n\nSomething else entirely.\n", "utf-8")
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)

    await run_tool(ctx, "text_index", {"paths": [str(spill)], "glob": "*"}, agent=agent)
    swept = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=agent)

    assert swept.value["chunks_removed"] == 0, "an ordinary walk forgot the indexed spill"
    assert swept.value["total_documents"] == 2


async def test_a_minified_bundle_is_skipped_and_a_doc_with_a_blob_keeps_itsprose(
    mount: MountProfile, tmp_path: Path
) -> None:
    """X6: junk text gives junk results, so the junk is kept out of the index.

    `skip_reason` is a `stat` and deliberately reads nothing, so a minified
    bundle under the size bound passes it. X5 then made things *worse* in one
    narrow way. Before X5, an over-long line became one oversized chunk that the
    embedder truncated. After X5, the same bundle becomes hundreds of
    well-formed passages of junk. Each one is a vector that matches nothing that
    a person will ask, and it costs an embedder call to make.

    Both halves are asserted, because the second half is the useful behavior.
    The bundle is refused outright. A document that only *contains* an encoded
    blob keeps every paragraph around the blob. Refusing that second document
    would be the easy rule and the wrong one.

    The skip is reported and is not silent, for the reason that `skip_reason`
    gives: a caller that never learns what was passed over cannot tell a quiet
    corpus from a broken one.

    Sabotage: drop the `triage` reason check in `index_run` and the bundle is
    indexed; keep every block in `triage` and the blob is too.
    """
    import base64
    import textwrap

    blob = "\n".join(textwrap.wrap(base64.b64encode(b"\x00" * 900).decode(), 76))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "bundle.md").write_text(
        ".a{margin:0;padding:0}.b{display:flex;align-items:center}" * 40, encoding="utf-8"
    )
    (tmp_path / "docs" / "guide.md").write_text(
        "# Workspaces\n\nEach agent is given its own worktree so two agents editing "
        "the same repository never see each other's half-finished changes.\n\n"
        f"![diagram](data:image/png;base64,\n{blob})\n\n"
        "The reclaim path waits for the tree to be released before deleting it.\n",
        encoding="utf-8",
    )
    ctx, _ = await _mounted(mount, tmp_path, max_chars=200, overlap_chars=0)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["documents"] == ["docs/guide.md"], "the bundle was indexed"
    skipped = {row["path"]: row["reason"] for row in indexed.value["skipped"]}
    assert "docs/bundle.md" in skipped, "the bundle was passed over without saying so"
    assert "prose" in skipped["docs/bundle.md"]

    # The document that only held a blob kept the prose around it, with none of
    # the blob. This is asserted on the chunks and not on a search result,
    # because a ranked answer can hide an unwanted passage behind a better one.
    # The question is what is *in* the index, not what came back first.
    # The chunks are made from what `index_run` gives the packer: the blocks that
    # `triage` kept. `chunk_text` alone has no policy, by design, so an assertion
    # on it would test the packer and not what reaches the index.
    guide = (tmp_path / "docs" / "guide.md").read_text(encoding="utf-8")
    passages = chunk_paragraphs(triage(guide).kept, max_chars=200, overlap_chars=0)
    assert any("worktree" in passage.text for passage in passages), "the prose was dropped too"
    assert not any("AAAA" in passage.text for passage in passages), (
        "the blob was chunked into passages nobody will ever search for"
    )

    found = await run_tool(
        ctx,
        "text_search",
        {"query": "why does each agent get its own worktree", "k": 2},
        agent=agent,
    )
    assert not found.is_error, text_of(found.content)
    assert "worktree" in text_of(found.content)


# ------------------------------------------- a passage fits its embedder ----


async def _bounded(mount: MountProfile, **settings: Any) -> tuple[Any, BoundedEmbedder]:  # noqa: ANN401
    ctx = await mount(ROW)
    embedder = BoundedEmbedder(**settings)
    ctx.require(TEXT_INDEX).register(embedder)
    return ctx, embedder


async def test_passages_are_packed_to_what_the_embedder_reads(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The failure that this fixes was silent.** A model reads a fixed number of
    tokens and drops the rest. A passage cut at `max_chars`, for a model that
    reads less, therefore had a tail that was indexed and could not be found.
    With the default model, that was the last eleventh of the documentation of
    this repository.

    The embedder states the limit, and passages are packed by what the embedder
    counts. Here a model that reads 50 tokens gets two paragraphs of 18 tokens
    in each passage, where `max_chars` would have packed nine paragraphs. It
    never gets an input that it would have cut. The spans still name real lines.
    """
    (tmp_path / "long.md").write_text(prose(12), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50)
    agent = _agent(ctx)

    report = io.StringIO()
    with reporting_to(report):
        indexed = await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=agent)
    found = await run_tool(ctx, "text_search", {"query": "w7l1n3", "k": 1}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert max(embedder.took) <= 50, "an input longer than the model reads was embedded"
    assert indexed.value["limit"] == "50 tokens (the model's)"
    assert indexed.value["passage_tokens"] == 50
    assert indexed.value["cut_smaller"] == []
    assert indexed.value["chunks_added"] == 6
    assert embedder.took[0] == 36, "a passage was not as long as the model reads"
    assert "Passages are held to 50 tokens of text" in text_of(indexed.content)
    # A reader of the note about a run that nobody waited for is told the same.
    last = json.loads(report.getvalue().splitlines()[-1])
    assert last["state"] == "done" and last["note"].splitlines() == [
        "Indexed 1 document: 6 passages added, 0 removed.",
        "Passages are held to 50 tokens of text, to fit the embedder's limit of "
        "50 tokens (the model's).",
        "The index now holds 6 passages from 1 document.",
    ]
    [hit] = found.value["hits"]
    lines = (tmp_path / "long.md").read_text(encoding="utf-8").splitlines()
    assert hit["text"].splitlines() == lines[hit["start_line"] - 1 : hit["end_line"]]
    assert "w7l1n3" in hit["text"]


async def test_dense_text_is_packed_by_its_tokens_at_the_first_cut(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder does not count characters. Code, a table, and a script that is
    one token for each character have as many characters as prose and several
    times the tokens. When such a document was packed by characters, it came out
    over the limit, was measured, and was packed again smaller. All of it was
    packed smaller, to fit its densest passage, and the result named it. When
    the document is packed by the count of the embedder itself, each passage is
    as long as the limit allows the first time. The prose beside the document is
    not made smaller because of it."""
    # 24 one-letter words in each line. A line has as many characters as a line
    # of prose, and four times the tokens.
    line = " ".join("abcdefghijklmnopqrstuvwx")
    dense = "\n\n".join("\n".join([line] * 3) for _ in range(8))
    (tmp_path / "dense.md").write_text(dense, encoding="utf-8")
    (tmp_path / "prose.md").write_text(prose(6), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50)

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert sorted(indexed.value["documents"]) == ["dense.md", "prose.md"]
    assert max(embedder.took) <= 50
    assert indexed.value["cut_smaller"] == [], "a document was packed a second time"
    assert embedder.calls == 2, "a document was sent more than once"
    # A passage is two lines of the dense document, or two paragraphs of prose.
    assert sorted(embedder.took) == [36, 48]


async def test_a_mixed_document_is_cut_small_only_where_it_is_dense(
    mount: MountProfile, tmp_path: Path
) -> None:
    """One table in a page of prose. When the page was packed again as a whole,
    its prose was cut to the size that its table needed. Each passage is now
    weighed separately, so the prose keeps passages as long as the model reads."""
    line = " ".join("abcdefghijklmnopqrstuvwx")
    table = "\n".join([line] * 3)
    (tmp_path / "page.md").write_text(f"{prose(4)}\n\n{table}\n\n{prose(4)}", encoding="utf-8")
    ctx, _ = await _bounded(mount, reads=50)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["page.md"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    seam = ctx.require(TEXT_INDEX)
    store = await seam.index(ctx.require(FS).root_for(agent))
    words = [len(record.text.split()) for record in store._records.values()]
    assert max(words) <= 50
    # Two lines of the table are one passage, and the third line goes with the
    # paragraph after it. The prose around them is still two paragraphs in each
    # passage.
    assert sorted(words) == [18, 36, 36, 36, 42, 48]


async def test_text_in_a_script_of_several_bytes_a_character_fits_at_the_first_cut(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder with no tokenizer estimates by bytes. Text in a script of three
    bytes for each character has three times as many bytes as characters. When
    such a passage was sized in characters, it came out three times over the
    limit, and the document was packed again. When it is weighed by its bytes,
    it fits at the first cut."""
    paragraph = "\n".join(" ".join("\u6587\u5b57\u5217" for _ in range(6)) for _ in range(3))
    (tmp_path / "cjk.md").write_text("\n\n".join([paragraph] * 12), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50, estimates=True)
    agent = _agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["cjk.md"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["cut_smaller"] == []
    assert embedder.calls == 1
    assert indexed.value["limit"] == "50 tokens (the model's)"
    seam = ctx.require(TEXT_INDEX)
    store = await seam.index(ctx.require(FS).root_for(agent))
    assert max(embedder.measure([one.text for one in store._records.values()], 3)) <= 50


async def test_what_is_sent_is_measured_and_a_document_over_the_limit_is_packed_again(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The packer adds up the weights of the paragraphs of a passage, and an
    embedder counts the whole passage. The whole can be more than its parts.
    Here every blank line costs twenty tokens that no paragraph weighed. The
    passages are therefore measured as they are sent. A document with a passage
    over the limit is packed again with less room, until its longest passage
    fits. The result names the document."""
    (tmp_path / "long.md").write_text(prose(12), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50, breaks=20)

    indexed = await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert max(embedder.took) <= 50
    assert embedder.refused == 0
    assert indexed.value["cut_smaller"] == ["long.md"]
    assert "1 document needed smaller passages still to fit: long.md" in text_of(indexed.content)


async def test_a_passage_the_embedder_refuses_is_cut_smaller_and_sent_again(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder that cannot say how long an input it reads, such as a hosted
    API with nothing to ask, shows its limit only by refusing an input. A
    refusal used to end the run. Now the document is packed at half the size and
    sent again until it is accepted, and nothing longer than the embedder reads
    was ever embedded.

    **The text of the refusal is also read.** A llama.cpp server names both
    numbers: how long the input was, and how long an input it accepts. The
    document is therefore packed to the limit that the refusal named, and is not
    halved. The rest of the run is packed to that limit too. The whole corpus
    costs one refusal. Otherwise, a llama.cpp server that reports a window, but
    does not accept an embedding input as long as that window, would refuse
    every document.
    """
    (tmp_path / "a.md").write_text(prose(20), encoding="utf-8")
    (tmp_path / "b.md").write_text(prose(20), encoding="utf-8")
    ctx, embedder = await _bounded(mount, refuses_over=40)

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert embedder.took and max(embedder.took) <= 40
    assert embedder.refused == 1, "the refusal's own numbers were not used"
    assert indexed.value["cut_smaller"] == ["a.md"]
    assert indexed.value["limit"] == "40 tokens (what its refusal said)"
    assert indexed.value["passage_tokens"] == 40
    assert indexed.value["total_documents"] == 2
    # The refused document is packed like the one after it: to the limit named.
    assert indexed.value["total_chunks"] == 20


async def test_a_refusal_that_counts_its_input_says_how_dense_the_corpus_is(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder with no tokenizer measures by bytes, at three bytes for each
    token, and technical text is denser than that. A dense passage therefore
    passes the estimate and is refused, and the refusal says how many tokens the
    passage really was. That number gives the density of the corpus itself. It
    is exact where the estimate was not. The documents after it are packed by
    that density. There is one refusal, where before every dense document was
    refused in its turn."""
    line = " ".join("abcdefghijklmnopqrstuvwx")
    dense = "\n\n".join("\n".join([line] * 3) for _ in range(6))
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.md").write_text(dense, encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=40, refuses_over=40, estimates=True)

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert max(embedder.took) <= 40
    assert embedder.refused == 1, "the second dense document was refused as well"
    assert indexed.value["cut_smaller"] == ["a.md"]
    assert ctx.require(TEXT_INDEX).density < 2, "the density a refusal stated was not used"


async def test_a_passage_packed_to_the_limit_fits_with_its_prefix_and_markers(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder counts the prefix of the passage and the markers of the model,
    as well as the text of the passage. When the text was packed to the full
    limit, every full passage measured a few tokens over the limit, and every
    document was packed again and reported as needing smaller passages: 14 of 30
    on a live server. The room that the text of a passage is packed to therefore
    leaves out the prefix and the markers. Here the prefix is 17 bytes and there
    are two markers. That is 8 tokens at three bytes for each token, out of 50,
    which leaves 42."""
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.md").write_text(prose(12), encoding="utf-8")
    ctx, _ = await _bounded(mount, reads=50, estimates=True, prefix="search_document: ")

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["passage_tokens"] == 42
    assert indexed.value["cut_smaller"] == [], "a passage packed to fit was packed again"
    assert indexed.value["total_documents"] == 3


async def test_a_refusal_with_no_numbers_halves_and_teaches_nothing(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A provider whose refusal names no number. Each document is halved until it
    is accepted, and nothing is concluded for the next document. The size at
    which one document was accepted says as much about that document as about
    the embedder. Packing every later document to that size would shrink the
    corpus to fit its densest page. The result names both documents. That is the
    signal to set `max_tokens`."""
    (tmp_path / "a.md").write_text(prose(20), encoding="utf-8")
    (tmp_path / "b.md").write_text(prose(20), encoding="utf-8")
    ctx, embedder = await _bounded(mount, refuses_over=40, vague=True)

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert max(embedder.took) <= 40
    assert indexed.value["cut_smaller"] == ["a.md", "b.md"]
    assert indexed.value["limit"] == "" and indexed.value["passage_tokens"] is None
    assert "2 documents needed smaller passages still to fit" in text_of(indexed.content)


async def test_a_document_that_cannot_be_made_to_fit_is_skipped_and_the_run_goes_on(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The floor. A passage is not cut below a phrase to satisfy an embedder. A
    document that the embedder refuses even then is reported with the words of
    the embedder, as a file that is too large to index is reported. The rest of
    the corpus is still indexed."""
    (tmp_path / "long.md").write_text(prose(4), encoding="utf-8")
    (tmp_path / "short.md").write_text("two words\n", encoding="utf-8")
    ctx, _ = await _bounded(mount, refuses_over=3)

    indexed = await run_tool(ctx, "text_index", {"paths": ["."]}, agent=_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["documents"] == ["short.md"]
    [skipped] = indexed.value["skipped"]
    assert skipped["path"] == "long.md"
    assert "refuses as too long even in passages of 16 tokens" in skipped["reason"]
    assert "too large to process" in skipped["reason"]


async def test_a_limit_a_profile_sets_recuts_and_an_unchanged_run_asks_for_none(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Two facts about when the limit is learned. A *setting* is part of how a
    document is cut, so a change to it cuts every document again. What a model
    reports is learned by asking the model, and for a local model that means
    loading it. A run that finds every document unchanged therefore does not
    ask."""
    (tmp_path / "long.md").write_text(prose(12), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50)
    agent = _agent(ctx)

    await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=agent)
    asked = embedder.asked
    again = await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=agent)
    assert (again.value["unchanged"], again.value["chunks_added"]) == (1, 0)
    assert embedder.asked == asked, "an unchanged run asked the embedder for its limit"

    embedder.max_tokens = 30
    seam = ctx.require(TEXT_INDEX)
    seam.known, seam._kept = None, None
    recut = await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=agent)
    assert recut.value["documents"] == ["long.md"]
    assert recut.value["limit"] == "30 tokens (configured)"
    assert embedder.took[-1] <= 30


async def test_doctor_says_how_long_an_input_the_embedder_reads(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A person who asks for status is told the limit, and the embedder is not
    asked for it. For a local model, asking would mean loading the model. The
    status therefore reports what is known:

    * Nothing, before a run has embedded.
    * What this process learned.
    * In a process that has asked nobody, what the last run to embed wrote beside
      the index.
    """
    (tmp_path / "long.md").write_text(prose(4), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50)
    seam = ctx.require(TEXT_INDEX)
    before = report_section(ctx, "Text index")["limit"]

    await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=_agent(ctx))
    learned = report_section(ctx, "Text index")["limit"]
    # The state of a command that is asked for its status: nothing open, nobody asked.
    seam.known, seam._kept = None, None
    seam._indexes.clear()
    asked = embedder.asked
    stored = report_section(ctx, "Text index")["limit"]

    assert before == "not known until a run embeds"
    assert learned == "50 tokens (the model's)"
    assert stored == "50 tokens (the model's), as the last run to embed asked"
    assert embedder.asked == asked, "status asked the embedder what it reads"


async def test_doctor_says_a_limit_a_profile_set_before_anything_is_indexed(
    mount: MountProfile,
) -> None:
    """A setting is reported with nobody asked and nothing embedded."""
    ctx, embedder = await _bounded(mount, max_tokens=30)

    assert report_section(ctx, "Text index")["limit"] == "30 tokens (configured)"
    assert embedder.asked == 0


# ------------------------------------------------ a question fits it as well ----


async def test_a_question_longer_than_the_embedder_reads_is_searched_by_its_head(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A question is an input, like a passage. A local model silently dropped the
    part of a question past what the model reads. A long question was therefore
    searched by its beginning, and nobody was told. A question is now measured
    in the way that a passage is, and cut to as much of its head as fits, in
    whole words. The answer says how much of the question was searched. A
    question that fits is sent as it was typed."""
    (tmp_path / "long.md").write_text(prose(12), encoding="utf-8")
    ctx, embedder = await _bounded(mount, reads=50)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["long.md"]}, agent=agent)
    head = "\n".join(" ".join(f"w7l{line}n{n}" for n in range(6)) for line in range(3))
    question = f"{head}\n\n{' '.join(f'padding{n}' for n in range(60))}"

    long = await run_tool(ctx, "text_search", {"query": question, "k": 1}, agent=agent)
    short = await run_tool(ctx, "text_search", {"query": "w7l1n3", "k": 1}, agent=agent)

    assert not long.is_error, text_of(long.content)
    sent, typed = embedder.took[-2:]
    assert 45 <= sent <= 50, "the question was not cut to as much as the model reads"
    assert typed == 1
    searched = long.value["query_chars"]
    assert len(head) < searched < len(question)
    assert question[searched].isspace(), "the head ends inside a word"
    assert long.value["query"] == question
    assert f"only its first {searched} characters were searched" in text_of(long.content)
    assert question not in text_of(long.content), "a page of question was repeated back"
    assert "w7l1n3" in long.value["hits"][0]["text"]
    assert short.value["query_chars"] is None
    assert "characters were searched" not in text_of(short.content)


async def test_a_question_the_embedder_refuses_is_cut_and_asked_again(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An embedder that says how long an input it reads only by refusing one. A
    question that it refused as too long used to fail the search. The question
    is now cut to the limit that the refusal names and asked again, and the
    search says so. What the refusal taught is kept, so the next long question
    is cut before it is sent."""
    (tmp_path / "a.md").write_text("two words\n", encoding="utf-8")
    ctx, embedder = await _bounded(mount, refuses_over=40)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["a.md"]}, agent=agent)
    question = " ".join(f"word{n}" for n in range(100))

    found = await run_tool(ctx, "text_search", {"query": question}, agent=agent)
    again = await run_tool(ctx, "text_search", {"query": question}, agent=agent)

    assert not found.is_error, text_of(found.content)
    assert embedder.refused == 1
    assert embedder.took[-1] <= 40
    assert 0 < found.value["query_chars"] < len(question)
    assert question.startswith(question[: found.value["query_chars"]])
    assert again.value["query_chars"] == found.value["query_chars"]
    assert "only its first" in text_of(found.content)


async def test_a_question_that_cannot_be_made_to_fit_fails_the_search_in_the_embedders_words(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The floor, as for a passage. A question is not cut below a phrase. If the
    embedder refuses the question even then, the search fails, in the words of
    the embedder."""
    (tmp_path / "a.md").write_text("two words\n", encoding="utf-8")
    ctx, _ = await _bounded(mount, refuses_over=3)
    agent = _agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["a.md"]}, agent=agent)

    refused = await run_tool(ctx, "text_search", {"query": " ".join(["word"] * 60)}, agent=agent)

    assert refused.is_error
    assert "too large to process" in text_of(refused.content)


def test_the_local_model_says_what_it_reads_and_counts_with_its_own_tokenizer() -> None:
    """`sentence-transformers` silently truncates every input to the
    `max_seq_length` of the model. That value is therefore the bound, and a
    setting can only lower it. A passage is counted in the way that the model
    counts it: with its prefix and its markers. The model is a fake, because the
    fact under test is which attributes of the model are read."""
    from ph_text_index._config import Limit
    from ph_text_index._embed import SentenceTransformerEmbedder, TokenBounded

    class Model:
        max_seq_length = 8

        @staticmethod
        def tokenizer(
            texts: list[str], *, add_special_tokens: bool, **_settings: object
        ) -> dict[str, list[list[int]]]:
            markers = [0] if add_special_tokens else []
            return {
                "input_ids": [[*markers, *range(len(text.split())), *markers] for text in texts]
            }

    embedder = SentenceTransformerEmbedder(model_name="m", document_prefix="passage: ")
    embedder._model = Model()

    assert isinstance(embedder, TokenBounded)
    assert embedder.limit() == Limit(tokens=8, source="model")
    assert embedder.measure(["one two three"], 3) == [6], "the prefix and both markers count"
    assert embedder.measure(["one two three"], 3, query=True) == [5], "a question has no prefix"
    assert embedder.weigh(["one two three"], 3) == [3], "a part of a passage is its own tokens"
    embedder.max_tokens = 5
    assert embedder.limit() == Limit(tokens=5, source="configured")
    embedder.max_tokens = 500
    assert embedder.limit() == Limit(tokens=8, source="model"), (
        "a setting above the model's put the silence back"
    )


def test_a_long_line_is_cut_between_characters_never_through_one() -> None:
    """The last-resort cut on a single long line (X5), and the cut by weight after
    it, land between characters as a reader sees them (`ph.characters`). A
    family emoji, a flag, a skin tone, and an accent that is written as two code
    points each stay whole in one chunk. When the line was cut by index, half of
    those cut points fell inside a character. An accent was indexed apart from
    its letter, and a flag was split into two letters that nobody searches for.

    Sabotage: slice the line by index in `_split_line` again and a family is
    split between two chunks.
    """
    line = " ".join(
        ["cafe\u0301", EMOJI_FAMILY, "\U0001f1fa\U0001f1f8", "\U0001f44d\U0001f3fd"] * 40
    )

    def cut_between(chunks: list[Chunk]) -> bool:
        """Whether every chunk starts and ends where `ph.characters` says that a
        cut may fall. The suite of ph-core holds `ph.characters` to the
        segmentation of Unicode itself."""
        at, ends = 0, []
        for chunk in chunks:
            at = line.index(chunk.text, at) + len(chunk.text)
            ends.append(at)
        return all(cut_before(line, end) == end for end in ends)

    by_length = chunk_text(line, max_chars=37, overlap_chars=0)
    assert "".join(chunk.text for chunk in by_length) == line
    assert cut_between(by_length) and all(len(chunk.text) <= 37 for chunk in by_length)

    by_bytes = Weight(of=lambda texts: weigh_by_bytes(texts, 3), most=20)
    by_weight = chunk_paragraphs(
        paragraphs(line), max_chars=len(line), overlap_chars=0, weight=by_bytes
    )
    assert cut_between(by_weight)
    first = head_of(EMOJI_FAMILY * 30, by_bytes)
    assert first and not first.replace(EMOJI_FAMILY, "")


async def test_a_guard_nobody_holds_or_waits_for_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock of an index is removed with its last holder and waiter.

    `_guards` made one lock for each index directory and kept it for the life of
    the mount: one for each worktree that an agent ever searched.

    Sabotage: keep the lock in a plain table again, and it is still there.
    """
    monkeypatch.setattr(TextIndexSeam, "root", lambda self, workspace: workspace / "index")
    seam = TextIndexSeam(ctx=Context(), config=Config())
    guard = seam.guard(tmp_path)
    async with guard:
        assert len(seam._guards) == 1
    async with guard:
        pass

    assert len(seam._guards) == 0
