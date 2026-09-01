"""The context loader (P3-17): a corpus reached through bindings, not a variable.

The row's gates, one test each: *queries produce dispatch records and are
individually offloadable; the corpus rehydrates after restart.*

The first is the whole argument for this shape. A corpus in the kernel namespace
queried as `context.search(...)` yields no `call` frame — the results reach the
model as merged cell stdout, capped but never offloaded, with nothing to attribute
a query to. Registered tools reached as `await tools.context_search(...)` yield
one governed dispatch per query, which is what lets `tools/post-execute` reshape
one oversized result and leave its siblings alone (C5).

## Why `_joined` is built once and sliced

`chunk` slices it through the offsets `Document` already keeps, so paging a large
corpus is arithmetic per call. The first version rebuilt — and, for lines, re-split
— the whole corpus on **every page of a walk that exists to visit all of it**.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from runtime_helpers import dispatch_names, run_ipython_cell, settled_dispatches

from ph.system_prompt import render_prompt
from ph.tools import Accept
from ph_rlm.context_loader import LOADED, Corpus, Document, render_manifest

pytestmark = pytest.mark.anyio

Loaded = Callable[..., Any]


def row(sources: list[dict[str, Any]], **config: Any) -> dict[str, Any]:
    return {
        "id": "rlm-context-loader",
        "name": "rlm-context-loader",
        "config": {"sources": sources, **config},
    }


@pytest.fixture
def loaded(mounted_runtime: Any, tmp_path: Path) -> Loaded:
    """`await loaded(sources=…)` → `(ctx, session, agent)` on the real runtime.

    The real kernel, because the load-bearing claim is about what a *cell* sees
    when it calls one of these tools.
    """

    async def build(
        sources: list[dict[str, Any]] | None = None,
        *,
        session_id: str = "context",
        **config: Any,
    ) -> tuple[Any, Any, Any]:
        if sources is None:
            (tmp_path / "notes.md").write_text(
                "# Notes\nthe deploy key lives in vault\n"
                "a second line about deploys\na third line about deploys\nunrelated\n"
            )
            sources = [{"kind": "path", "value": str(tmp_path / "notes.md")}]
        return await mounted_runtime(
            session_id=session_id, presentation=True, extra_rows=[row(sources, **config)]
        )

    return build


# ---------------------------------------------------------------- the unit --


def test_a_document_indexes_its_lines_without_splitting_it() -> None:
    """Offsets, not a tuple of lines: a corpus worth this row would otherwise
    hold every line as its own string for the life of the process."""
    document = Document.of("a.md", "first\nsecond\nthird\n")

    assert document.line_count == 3
    assert document.line(2) == "second"
    assert document.lines(2, 3) == "second\nthird"
    # Clamped rather than raising: a match on line 1 asks for line 0's context.
    assert document.lines(0, 1) == "first"
    assert document.lines(3, 99) == "third"
    assert document.lines(9, 12) == ""


def test_a_document_without_a_trailing_newline_still_ends() -> None:
    document = Document.of("a.md", "only")
    assert (document.line_count, document.line(1)) == (1, "only")


def _corpus(**documents: str) -> Corpus:
    return Corpus(
        name="notes",
        documents=tuple(Document.of(name, text) for name, text in documents.items()),
        sources=tuple(documents),
        digest="d",
    )


def test_search_returns_lines_with_context_not_documents() -> None:
    corpus = _corpus(a="alpha\nbeta\ngamma\n", b="delta\nbeta again\n")
    found = corpus.search("beta", limit=10, context_lines=1, regex=False)

    assert [(one.document, one.line) for one in found.matches] == [("a", 2), ("b", 2)]
    assert found.matches[0].text == "alpha\nbeta\ngamma"
    assert found.truncated is False


def test_search_says_when_it_held_matches_back() -> None:
    """The model can narrow a query; it cannot know to unless it is told.

    At most `limit` matches are ever built — the eleventh hit here sets the flag
    and stops the scan rather than being constructed and discarded.
    """
    corpus = _corpus(a="hit\n" * 10)
    found = corpus.search("hit", limit=3, context_lines=0, regex=False)

    assert len(found.matches) == 3
    assert found.truncated is True


def test_a_regex_keeps_its_line_anchors() -> None:
    """The scan runs over whole documents; `re.MULTILINE` is what keeps a model's
    `^`/`$` meaning what it meant when matching was per line."""
    corpus = _corpus(a="prefix beta\nbeta starts here\n")
    found = corpus.search("^beta", limit=10, context_lines=0, regex=True)
    assert [(one.line, one.text) for one in found.matches] == [(2, "beta starts here")]


def test_a_bad_regex_is_the_callers_mistake() -> None:
    with pytest.raises(ValueError, match="not a valid regular expression"):
        _corpus(a="x").search("(unclosed", limit=1, context_lines=0, regex=True)


def test_chunking_pages_the_whole_corpus_deterministically() -> None:
    """Paging, not one big read: the model walks the corpus as a sequence of
    governed dispatches it can stop, and every line is in exactly one chunk."""
    corpus = _corpus(a="1\n2\n3\n4\n", b="5\n6\n")
    first = corpus.chunk(by="lines", size=4, index=0)
    pages = [corpus.chunk(by="lines", size=4, index=n).text for n in range(first.chunks)]

    assert first.index == 0
    walked = "\n".join(pages)
    assert all(str(number) in walked for number in range(1, 7))
    assert walked.count("# a") == 1 and walked.count("# b") == 1
    # An index past the end is clamped, so paging cannot fail into an error.
    assert corpus.chunk(by="lines", size=4, index=99).index == first.chunks - 1
    assert corpus.chunk(by="bytes", size=8, index=0).text == "# a\n1\n2\n"


# --------------------------------------------------------------- the gate --


async def test_each_query_is_its_own_governed_dispatch(loaded: Loaded) -> None:
    """The plan's gate, and the reason access is a binding.

    Two queries in one cell are two dispatch pairs — per-query provenance a
    namespace variable could not produce, because a bare `context.search(...)`
    never leaves the kernel.
    """
    ctx, session, agent = await loaded()
    result = await run_ipython_cell(
        ctx,
        "a = await tools.context_search(query='deploy key')\n"
        "b = await tools.context_search(query='unrelated')\n"
        "(len(a['matches']), len(b['matches']))",
        agent=agent,
        session=session,
    )

    assert result.is_error is False
    assert result.value["value"] == [1, 1]
    assert dispatch_names(session) == ["context_search", "context_search"]
    # Two settled pairs, so a later reader can attribute each answer to its query.
    assert len(settled_dispatches(session)) == 2


async def test_one_oversized_query_is_offloaded_without_its_siblings(loaded: Loaded) -> None:
    """C5, on the seam the Phase 4 spill row attaches to.

    The property that matters is per-dispatch reshaping: the big query's result is
    replaced in what the *program* receives, and the small one passes through.
    A corpus read as merged cell stdout could offer neither.
    """
    ctx, session, agent = await loaded()

    async def offload(execution: Any, result: Any, next_: Any) -> Any:
        matches = (result.value or {}).get("matches") or []
        if execution.name != "context_search" or len(matches) <= 1:
            return await next_(execution, result)
        return Accept(
            value={"matches": [{"document": "[spilled]", "line": 0, "text": ""}]}, has_value=True
        )

    ctx.on("tools/post-execute", offload)

    result = await run_ipython_cell(
        ctx,
        "big = await tools.context_search(query='line about deploys', limit=50)\n"
        "small = await tools.context_search(query='vault')\n"
        "([m['document'] for m in big['matches']], [m['document'] for m in small['matches']])",
        agent=agent,
        session=session,
    )

    assert result.is_error is False
    big, small = result.value["value"]
    assert big == ["[spilled]"], "the oversized result was not reshaped"
    assert small != ["[spilled]"], "its sibling was reshaped with it"


async def test_the_corpus_rehydrates_and_says_when_it_changed(
    mounted_runtime: Any, tmp_path: Path
) -> None:
    """The plan's other gate. A recipe, not a snapshot (D17/Q4): what is durable
    is `{loader, sources, digest}`, and a later session re-resolves it."""
    source = tmp_path / "notes.md"
    source.write_text("original content\n")
    sources = [{"kind": "path", "value": str(source)}]

    ctx, session, agent = await mounted_runtime(session_id="c1", extra_rows=[row(sources)])
    assert await _assemble(ctx, agent) != ""
    (record,) = [event for event in session.events if event.type == LOADED]
    first_digest = str(record.data["digest"])
    assert record.data["loader"] == "rlm-context-loader"
    assert list(record.data["sources"]) == [str(source)]
    assert record.data["note"] == "", "a first load has nothing to report"

    # Re-resolving unchanged sources gives the same digest, so nothing new is
    # recorded and nothing is said — the silent case, and the common one.
    ctx2, _session2, _agent2 = await mounted_runtime(session_id="c2", extra_rows=[row(sources)])
    assert ctx2.context_corpus.corpus.digest == first_digest
    assert ctx2.context_corpus.announce(session) == ""
    assert len([event for event in session.events if event.type == LOADED]) == 1

    # Now the source moves under a session that was already told about it. A
    # snapshot would have restored the old bytes; a recipe re-resolves and says so.
    source.write_text("rewritten content, quite different\n")
    ctx3, _s3, _a3 = await mounted_runtime(session_id="c3", extra_rows=[row(sources)])
    assert ctx3.context_corpus.corpus.digest != first_digest
    note = ctx3.context_corpus.announce(session)
    assert "was rebuilt from changed sources" in note
    assert session.events[-1].data["note"] == note
    # And announcing again is a memo hit: same note, no third record.
    assert ctx3.context_corpus.announce(session) == note
    assert len([event for event in session.events if event.type == LOADED]) == 2


async def test_an_unreadable_source_is_reported_not_hidden(
    mounted_runtime: Any, tmp_path: Path
) -> None:
    ctx, session, agent = await mounted_runtime(
        session_id="missing",
        extra_rows=[
            row(
                [
                    {"kind": "text", "value": "a pasted blob", "name": "pasted"},
                    {"kind": "path", "value": str(tmp_path / "gone.md")},
                ]
            )
        ],
    )
    assert ctx.context_corpus.corpus.missing == (str(tmp_path / "gone.md"),)

    prompt = await _assemble(ctx, agent)
    assert "could not be read" in prompt
    assert "unavailable" in ctx.context_corpus.announce(session)


# ------------------------------------------------------------- the prompt --


async def _assemble(ctx: Any, agent: Any) -> str:
    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)
    return render_prompt(assembly)


async def test_the_prompt_carries_metadata_and_no_content(loaded: Loaded) -> None:
    """The point of the row: a corpus large enough to load is large enough that
    putting any of it in the prompt defeats the purpose."""
    ctx, _session, agent = await loaded()
    prompt = await _assemble(ctx, agent)

    assert "A corpus named `context` is loaded" in prompt
    assert "notes.md` — 5 lines" in prompt
    assert "await tools.context_search(query=...)" in prompt
    assert "the deploy key lives in vault" not in prompt, "the corpus reached the prompt"


async def test_the_corpus_is_in_the_cached_prefix(loaded: Loaded) -> None:
    """A `section`, not a `context()`: resolved once at mount, so the text is
    fixed for the session and belongs in the stable prefix (A12)."""
    ctx, _session, agent = await loaded()
    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)

    assert "# Loaded context" in render_prompt(assembly)


def test_a_manifest_names_every_document() -> None:
    text = render_manifest(_corpus(a="1\n", b="2\n"), note="something changed")
    assert "`a` — 1 lines" in text and "`b` — 1 lines" in text
    assert "something changed" in text


# ------------------------------------------------------- standing down --


async def test_no_sources_means_no_row(mounted_runtime: Any) -> None:
    """Off by default is the shipped state; a row with nothing to load must not
    advertise three tools that would answer nothing."""
    ctx, _session, agent = await mounted_runtime(session_id="empty", extra_rows=[row([])])

    assert ctx.get("context_corpus") is None
    assert "context_search" not in ctx.tools.names(scope=agent.ctx)
    assert "# Loaded context" not in await _assemble(ctx, agent)


async def test_a_corpus_under_the_threshold_stands_down(
    mounted_runtime: Any, tmp_path: Path
) -> None:
    """Q4's threshold: a corpus small enough to read with `tools.read` should be
    read with `tools.read` — governed, logged and offloadable already."""
    small = tmp_path / "small.md"
    small.write_text("a few characters\n")
    ctx, _session, agent = await mounted_runtime(
        session_id="small",
        extra_rows=[row([{"kind": "path", "value": str(small)}], minChars=200_000)],
    )

    assert ctx.get("context_corpus") is None
    assert "context_search" not in ctx.tools.names(scope=agent.ctx)
