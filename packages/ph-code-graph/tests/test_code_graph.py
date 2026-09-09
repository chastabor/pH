"""`code-graph` — line spans that are true, edges that point the right way.

Gates: *every span slices the source it claims; a call inside a closure belongs
to the closure; a second pass over an unchanged tree parses nothing.*

## Why spans are checked against the file

Two coordinate systems meet in `_extract`: `ProcessConfig` spans are 0-based,
tree-sitter points are 0-based, and every line pH shows a model is 1-based. An
off-by-one survives every plausible test — the symbol is right, the file is
right, the number looks fine — and shows up as a model that reads the wrong
function and stops trusting the tool. So the assertions here slice the real
source at the span and check the definition is actually there, rather than
comparing against a number someone wrote down.

## Why the enclosing rule gets its own test

`callers` is only as good as the attribution of a reference to the symbol
containing it, and the tempting implementation — first definition whose range
covers the line — is wrong in exactly the case this codebase is full of: a
closure inside a function. `_registry.py`'s `claim_key` spans 40-75 and its
inner `release` spans 69-73, so a call on line 70 has two candidates. Taking the
outer one would credit every closure's calls to whatever contained it, and no
count would look obviously wrong.

## Why the real parser and the real database

Neither is stubbed. The parser is the thing under test — a language pack that
changed its span base or its symbol kinds must fail here rather than in a
session — and the index is 200 lines of SQL whose two interesting queries (FTS5
and a recursive CTE) are exactly the ones a fake would not exercise.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ph.llm.types import text_of
from ph.testing import FAKE_OPTIONS, report_section, run_tool
from ph.testing.git import git, git_repo
from ph.testing.jj import jj_repo
from ph_code_graph._extract import (
    INHERITS,
    Definition,
    detect_language,
    extract,
    indexable,
    owners,
)

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "code-graph", "name": "code-graph"}

MODULE = '''\
"""A module."""

from helpers import shared


def outer(value):
    """Does the outer thing."""

    def inner():
        return shared(value)

    return inner()


class Widget:
    """A widget."""

    def render(self):
        return outer(1)


async def apply(ctx, config):
    def enter():
        return shared(2)

    return enter
'''
"""Deliberately the shapes that break naive extractors: a closure inside a
function, a method, and a module-level `async def` with a nested `def` (the
shape that crashed the PyPI `codegraph` outright)."""

OTHER = '''\
def shared(value):
    """The thing everyone calls."""
    return value * 2
'''


@pytest.fixture(scope="module", autouse=True)
def _grammar_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One writable grammar cache for this module, outside every test's `$PH_CACHE`.

    `tree-sitter-language-pack` materialises even its *bundled* grammars into a
    writable directory on first use — the wheel ships them as an archive, not as
    loadable libraries — and fails hard when it cannot create one. So every test
    that parses needs it, including the ones that call `extract` directly and
    mount no row.

    **Module-scoped, and not the row's own path.** The suite's `_isolated_home`
    pins `$PH_CACHE` per test, so leaving the row to choose would re-materialise
    the grammars for each — the same work tens of times, for isolation nobody
    wanted. Grammars are read-only content keyed by the pack's version.

    **In this module rather than a `conftest.py`**, which is not a style choice:
    `ph-rlm`'s tests do `from conftest import BINDINGS_ROW`, importing their
    conftest as a top-level module, and a second file of that name anywhere in
    `testpaths` shadows it on `sys.path` — adding one here broke ten of their
    modules at collection. A module-scoped fixture needs no new module name.

    Set through the environment variable rather than `use_cache`, which is that
    function's operator-override path — so the run exercises the branch where an
    operator's own spelling wins.
    """
    directory = tmp_path_factory.mktemp("tree-sitter-grammars")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", str(directory))
        yield directory


_SEQ = itertools.count()


def _agent(ctx: Any) -> Any:
    """A fresh agent, on a session id nothing else has taken.

    Counted rather than fixed: a test that needs two agents (an index call and a
    query call) would otherwise collide on the session id, which fails as
    `SESSION_ALREADY_EXISTS` several frames from the cause.
    """
    return ctx.agents.create(ctx.sessions.create(f"cg-{next(_SEQ)}"), FAKE_OPTIONS)


def _tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "mod.py").write_text(MODULE, encoding="utf-8")
    (root / "pkg" / "helpers.py").write_text(OTHER, encoding="utf-8")


async def _indexed(mount: Any, tmp_path: Path, **config: Any) -> Any:
    settings = {"path": str(tmp_path / "graph.db"), **config}
    ctx = await mount({**ROW, "config": settings})
    return ctx


# ---------------------------------------------------------------- extraction ----


def test_every_span_slices_the_definition_it_claims() -> None:
    """The off-by-one that would otherwise be invisible. See the docstring."""
    lines = MODULE.splitlines()
    found = extract("pkg/mod.py", MODULE, "python")

    assert found.definitions, "nothing was extracted"
    for one in found.definitions:
        head = lines[one.start_line - 1]
        assert one.name in head, f"{one.name} claims line {one.start_line}, which reads {head!r}"
        assert 1 <= one.start_line <= one.end_line <= len(lines)


def test_the_shapes_that_break_naive_extractors_all_come_out() -> None:
    found = extract("pkg/mod.py", MODULE, "python")
    by_name = {one.name: one for one in found.definitions}

    # A module-level `async def` containing a nested `def` — the shape the PyPI
    # `codegraph` crashed on outright.
    assert "apply" in by_name and "enter" in by_name
    # A closure inside a plain function, and a method on a class.
    assert "outer" in by_name and "inner" in by_name
    assert "Widget" in by_name and "render" in by_name
    assert by_name["Widget"].kind == "class"
    assert by_name["outer"].kind == "function"


def test_docstrings_ride_along() -> None:
    """What makes a `search` hit readable without a second call."""
    found = extract("pkg/mod.py", MODULE, "python")
    docs = {one.name: (one.doc or "") for one in found.definitions}

    assert "outer thing" in docs["outer"]
    assert "widget" in docs["Widget"].lower()


def test_calls_are_recorded_as_references() -> None:
    found = extract("pkg/mod.py", MODULE, "python")

    called = {one.name for one in found.references}
    assert {"shared", "outer", "inner"} <= called
    assert all(one.kind == "call" for one in found.references)


def test_imports_are_recorded() -> None:
    found = extract("pkg/mod.py", MODULE, "python")

    assert any("helpers" in one for one in found.imports)


def test_a_reference_belongs_to_the_tightest_enclosing_definition() -> None:
    """The attribution rule `callers` rests on. See the module docstring."""
    definitions = (
        Definition("claim_key", "function", 40, 75),
        Definition("release", "function", 69, 73),
    )

    table = owners(definitions)

    assert table[70].name == "release", "a closure's call belongs to the closure"
    assert table[45].name == "claim_key"
    assert 5 not in table, "module level is not inside anything"


def test_the_owner_table_is_built_once_for_the_whole_file() -> None:
    """Why it is a table and not a search: it was O(definitions x references).

    A generated file with 1 500 definitions and 3 000 references spent 58 ms in
    the per-reference scan — as much as its entire parse — and `max_bytes`
    admits such files.
    """
    definitions = tuple(Definition(f"f{n}", "function", n * 10 + 1, n * 10 + 9) for n in range(200))

    table = owners(definitions)

    assert table[1].name == "f0"
    assert table[1991].name == "f199"
    assert 10 not in table, "the gap between two definitions belongs to neither"


def test_the_typescript_query_inherits_javascript() -> None:
    """A `.ts` file matched against the typescript query alone finds no calls."""
    assert INHERITS["typescript"] == ("javascript",)

    code = "class C { run(): number { return helper(1); } }\nfunction helper(n: number){return n}\n"
    found = extract("a.ts", code, "typescript")

    names = {one.name for one in found.definitions}
    assert {"C", "run", "helper"} <= names, f"only got {names}"
    assert any(one.name == "helper" for one in found.references), "the call was lost"


@pytest.mark.parametrize(
    ("language", "code", "wanted"),
    [
        ("rust", "fn helper() -> i32 { 1 }\nfn main() { helper(); }\n", "helper"),
        ("go", "package m\nfunc helper() int { return 1 }\nfunc main() { helper() }\n", "helper"),
        ("java", "class A { int helper(){return 1;} void run(){ helper(); } }\n", "helper"),
        # Parenthesised on purpose: a *bare* Ruby send (`helper` with no
        # parens) parses as an identifier, not a `call`, so the language's tags
        # query cannot see it. That is a real limitation of this approach and
        # the README says so rather than this fixture hiding it.
        ("ruby", "def helper\n 1\nend\ndef run\n helper()\nend\n", "helper"),
    ],
)
def test_other_languages_yield_definitions_and_calls(language: str, code: str, wanted: str) -> None:
    """The claim that this is not a Python-only tool."""
    found = extract(f"a.{language}", code, language)

    assert any(one.name == wanted for one in found.definitions), f"{language}: no definition"
    assert found.references, f"{language}: no references"


def test_language_detection_reads_the_extension() -> None:
    assert detect_language("a/b/x.py") == "python"
    assert detect_language("a/x.rs") == "rust"
    assert detect_language("a/x.tsx") == "tsx"
    assert detect_language("a/CHANGELOG") is None


# ------------------------------------------------------------------- the row ----


async def test_indexing_then_asking_answers_with_a_readable_pointer(
    mount: Any, tmp_path: Path
) -> None:
    """End to end: the whole reason the package exists."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert not built.is_error, text_of(built.content)
    assert built.value["indexed"] == 2
    assert built.value["total_symbols"] > 5
    assert built.value["languages"] == ["python"]

    found = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    hit = found.value["symbols"][0]
    assert hit["path"] == "pkg/helpers.py"
    # The pointer, verified against the real file rather than a number.
    lines = (tmp_path / hit["path"]).read_text().splitlines()
    assert "def shared" in lines[hit["start_line"] - 1]
    assert f"{hit['path']}:{hit['start_line']}" in text_of(found.content)


async def test_callers_names_the_calling_symbol_and_the_calling_line(
    mount: Any, tmp_path: Path
) -> None:
    """Not the definition's line — the line to open."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    edges = found.value["edges"]
    assert edges, text_of(found.content)
    callers = {one["name"] for one in edges}
    # `inner` and `enter` call it; `outer` and `apply` merely contain them.
    assert {"inner", "enter"} <= callers, f"got {callers}"
    assert "outer" not in callers, "a closure's call was credited to its parent"
    for one in edges:
        line = (tmp_path / one["path"]).read_text().splitlines()[one["ref_line"] - 1]
        assert "shared(" in line, f"ref_line {one['ref_line']} reads {line!r}"


async def test_callees_reports_what_a_symbol_reaches(mount: Any, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callees", "query": "render"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    assert "outer" in {one["name"] for one in found.value["edges"]}


async def test_impact_walks_transitively_ring_by_ring(mount: Any, tmp_path: Path) -> None:
    """The recursive CTE — the query `pyturso` refuses. `shared` <- inner <- outer."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx,
        "code_graph",
        {"mode": "impact", "query": "shared", "distance": 3},
        agent=agent,
    )

    assert not found.is_error, text_of(found.content)
    rings = {
        ring["distance"]: {one["name"] for one in ring["symbols"]} for ring in found.value["rings"]
    }
    assert rings, text_of(found.content)
    assert {"inner", "enter"} <= rings[1]
    # `inner` is called by `outer`, so a second hop must reach it — this is what
    # a single-hop implementation would silently miss.
    assert 2 in rings and "outer" in rings[2], f"rings were {rings}"


async def test_a_symbol_reached_twice_is_reported_at_the_nearer_distance(
    mount: Any, tmp_path: Path
) -> None:
    """`MIN(depth)` in the CTE — what makes the rings a budget, not a multiset."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(
        "def leaf():\n    return 1\n\n\ndef mid():\n    return leaf()\n\n\n"
        "def top():\n    return leaf() + mid()\n",
        encoding="utf-8",
    )
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "impact", "query": "leaf", "distance": 3}, agent=agent
    )

    rings = {r["distance"]: {one["name"] for one in r["symbols"]} for r in found.value["rings"]}
    assert "top" in rings[1], "top calls leaf directly"
    assert all("top" not in names for depth, names in rings.items() if depth > 1)


async def test_search_ranks_by_name_and_docstring(mount: Any, tmp_path: Path) -> None:
    """FTS5 — the other query `pyturso` cannot serve."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "search", "query": "everyone calls"}, agent=agent
    )

    assert not found.is_error, text_of(found.content)
    assert "shared" in {one["name"] for one in found.value["symbols"]}


async def test_a_models_punctuation_does_not_become_fts_syntax(mount: Any, tmp_path: Path) -> None:
    """`read-before-edit` is an FTS5 syntax error and `a:b` is a column filter.

    A model writing prose meant neither, and a tool that raised on the phrasing
    would teach it to stop using the mode.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    for query in ("read-before-edit", 'the "widget" thing', "shared: value", "NEAR outer", "*"):
        found = await run_tool(ctx, "code_graph", {"mode": "search", "query": query}, agent=agent)
        assert not found.is_error, f"{query!r} raised: {text_of(found.content)}"


async def test_entities_lists_the_biggest_definitions_first(mount: Any, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "entities", "path": "pkg", "limit": 4}, agent=agent
    )

    assert not found.is_error, text_of(found.content)
    spans = [one["lines"] for one in found.value["symbols"]]
    assert spans == sorted(spans, reverse=True)
    assert found.value["truncated"], "paging did not report there was more"
    assert "offset=4" in text_of(found.content)


async def test_an_ambiguous_name_says_it_is_ambiguous(mount: Any, tmp_path: Path) -> None:
    """The honest ceiling of a name-based graph, surfaced rather than hidden."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def register():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def register():\n    return 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "c.py").write_text(
        "from a import register\n\n\ndef go():\n    return register()\n", encoding="utf-8"
    )
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "register"}, agent=agent)

    assert found.value["definitions"] == 2
    assert "defined in 2 places" in text_of(found.content)
    assert "mode=define" in text_of(found.content)


# ------------------------------------------------------------- incrementality ----


async def test_a_second_pass_over_an_unchanged_tree_parses_nothing(
    mount: Any, tmp_path: Path
) -> None:
    """Content-keyed, so re-running after a turn is cheap and correct."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    second = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert first.value["indexed"] == 2 and first.value["unchanged"] == 0
    assert second.value["indexed"] == 0, "an unchanged tree was re-parsed"
    assert second.value["unchanged"] == 2
    assert second.value["total_symbols"] == first.value["total_symbols"]


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path `FsService.read` opens from here on.

    Both change-filter tests want the same thing — *the file was never opened* is
    the whole claim — and each had written the patch out, nine lines apiece. The
    returned list is live: assert against it after the call under test.
    """
    from ph.seams.fs import FsService

    reads: list[str] = []
    original = FsService.read

    async def counted(self: Any, path: Any, **kwargs: Any) -> Any:
        reads.append(str(path))
        return await original(self, path, **kwargs)

    monkeypatch.setattr(FsService, "read", counted)
    return reads


async def test_a_touched_but_unmodified_file_stays_unchanged(mount: Any, tmp_path: Path) -> None:
    """An mtime is not content. A rebase moves every mtime in the tree."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    target = tmp_path / "pkg" / "mod.py"
    target.touch()
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert again.value["indexed"] == 0


async def test_an_edit_is_picked_up_and_replaces_the_old_rows(mount: Any, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    (tmp_path / "pkg" / "helpers.py").write_text(
        "def renamed(value):\n    return value\n", encoding="utf-8"
    )
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert again.value["indexed"] == 1
    gone = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert gone.value["symbols"] == [], "the replaced file's old symbols survived"
    here = await run_tool(ctx, "code_graph", {"mode": "define", "query": "renamed"}, agent=agent)
    assert here.value["symbols"], "the new symbol was not indexed"


async def test_forget_removes_a_path_from_the_index(mount: Any, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    dropped = await run_tool(
        ctx, "code_index", {"paths": ["pkg/helpers.py"], "forget": True}, agent=agent
    )

    assert dropped.value["removed"] == 1
    assert dropped.value["total_files"] == 1
    found = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert found.value["symbols"] == []


# --------------------------------------------------------------- the edges ----


async def test_non_code_and_oversized_files_are_skipped_not_fatal(
    mount: Any, tmp_path: Path
) -> None:
    """A tree walk must not fail because it found a bundle or a README."""
    _tree(tmp_path)
    (tmp_path / "pkg" / "README.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "pkg" / "huge.py").write_text("x = 1\n" * 5_000, encoding="utf-8")
    ctx = await _indexed(mount, tmp_path, max_bytes=1_000)
    agent = _agent(ctx)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert not built.is_error, text_of(built.content)
    assert built.value["indexed"] == 2
    assert [one["path"] for one in built.value["skipped"]] == ["pkg/huge.py"]
    # Markdown is not skipped-with-a-reason, it is simply not indexable —
    # reporting every prose file would make the skip list the whole repository.
    assert all("README" not in one["path"] for one in built.value["skipped"])
    assert not indexable("markdown"), "the predicate this rests on"


async def test_a_query_before_any_index_says_what_to_do(mount: Any, tmp_path: Path) -> None:
    ctx = await _indexed(mount, tmp_path)

    found = await run_tool(
        ctx, "code_graph", {"mode": "search", "query": "anything"}, agent=_agent(ctx)
    )

    assert found.is_error
    assert "run `code_index` first" in text_of(found.content)


async def test_a_mode_that_needs_a_query_says_so(mount: Any, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers"}, agent=agent)

    assert found.is_error
    assert "needs `query`" in text_of(found.content)


async def test_indexing_reads_through_the_fs_seam(mount: Any, tmp_path: Path) -> None:
    """The claim that makes this a tool and not a tree-reading primitive (I-9).

    A screen registered on `ctx.fs` decides what a walk may show, so a file it
    refuses must never reach the index — which holds only because the row globs
    and reads through the seam.
    """
    _tree(tmp_path)
    (tmp_path / "pkg" / "secret.py").write_text("def hidden():\n    return 1\n", encoding="utf-8")
    ctx = await _indexed(mount, tmp_path)
    ctx.fs.screen(
        lambda path, name, agent, is_dir: "skip" if name == "secret.py" else "yield",
        scope=ctx,
    )

    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))
    found = await run_tool(
        ctx, "code_graph", {"mode": "define", "query": "hidden"}, agent=_agent(ctx)
    )

    assert found.value["symbols"] == [], "a screened file reached the index"


async def test_the_row_reports_itself_to_doctor(mount: Any, tmp_path: Path) -> None:
    ctx = await _indexed(mount, tmp_path)

    rows = report_section(ctx, "Code graph")
    assert rows["state"] == "not built — run code_index"

    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))
    rows = report_section(ctx, "Code graph")
    assert rows["files"] == "0", "nothing was indexed in this test"


async def test_a_callees_reference_line_names_its_own_file(mount: Any, tmp_path: Path) -> None:
    """`path` and `ref_path` are different files for `callees`, and both matter.

    The symbol is the callee's *definition*; the reference is a line in the
    caller. Rendering the line against the definition's path printed
    `helpers.py:19` for a line that lives in `mod.py` — a pointer to nothing,
    and one that looks entirely plausible.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callees", "query": "inner"}, agent=agent)

    edges = [one for one in found.value["edges"] if one["name"] == "shared"]
    assert edges, text_of(found.content)
    edge = edges[0]
    assert edge["path"] == "pkg/helpers.py", "the definition is in helpers"
    assert edge["ref_path"] == "pkg/mod.py", "the call is in mod"
    # And the line really is the call, in the file the result names.
    line = (tmp_path / edge["ref_path"]).read_text().splitlines()[edge["ref_line"] - 1]
    assert "shared(" in line, f"{edge['ref_path']}:{edge['ref_line']} reads {line!r}"
    assert f"{edge['ref_path']}:{edge['ref_line']}" in text_of(found.content)


def test_doc_comments_attach_only_when_directly_above() -> None:
    """The positional rule, and the off-by-one that broke it in both directions.

    Rust's `///` comment node includes its trailing newline, so trusting
    `span.end_line` missed the definition one line below *and* matched one two
    lines below. Counting the lines the text actually spans is exact.
    """
    attached = extract("a.rs", "/// Doc for helper.\nfn helper() -> i32 { 1 }\n", "rust")
    assert {one.name: one.doc for one in attached.definitions}["helper"] == "Doc for helper."

    gapped = extract("a.rs", "/// Floating.\n\nfn helper() -> i32 { 1 }\n", "rust")
    assert {one.name: one.doc for one in gapped.definitions}["helper"] is None


def test_a_run_of_doc_comments_is_kept_whole() -> None:
    """Rust and Go write a paragraph as consecutive single-line comments.

    Attaching only the last would index the closing sentence of every doc
    comment in those languages and silently drop the rest.
    """
    found = extract(
        "a.go",
        "package m\n// Helper does a thing.\n// More detail.\nfunc Helper() int { return 1 }\n",
        "go",
    )

    doc = {one.name: one.doc for one in found.definitions}["Helper"]
    assert doc is not None
    assert "does a thing" in doc and "More detail" in doc


def test_prose_is_stored_without_its_delimiters() -> None:
    """Quotes in the FTS index put punctuation next to every search term."""
    found = extract("a.py", 'def f():\n    """Some prose."""\n    return 1\n', "python")

    assert {one.name: one.doc for one in found.definitions}["f"] == "Some prose."


def test_c_has_definitions_but_no_call_references() -> None:
    """A per-language limitation, pinned so it is a known shape and not a surprise.

    C's tags query has `@definition.*` captures and no `@reference.*` ones at
    all, so `search`, `define` and `entities` work for C while `callers` and
    `callees` are empty. Better stated in a test than discovered in a session.
    """
    found = extract("a.c", "int helper(void){return 1;}\nint main(void){return helper();}\n", "c")

    assert {one.name for one in found.definitions} >= {"helper", "main"}
    assert found.references == (), "if C gained reference captures, use them"


async def test_an_unwritable_grammar_cache_refuses_with_a_sentence(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment fact an operator can fix, not a traceback.

    The grammars are materialised into `$PH_CACHE` on first use, so a read-only
    one is fatal — and pH's contract is that a row which cannot honour its
    configuration raises `MountRefusal`, which every command that mounts a
    profile turns into a sentence and an exit code. Measured inside a sandbox
    that made `~/.cache` read-only, where it arrived as fourteen frames of
    `pathlib` instead.
    """
    from ph.cordis import MountRefusal

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.delenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", raising=False)
    monkeypatch.setenv("PH_CACHE", str(blocked / "cache"))

    try:
        with pytest.raises(MountRefusal, match="grammar cache"):
            await mount(ROW)
    finally:
        blocked.chmod(0o700)


# ---------------------------------------------------- the bundle, and the RLM ----
#
# `ph-code-graph` exists so an agent can ask a codebase about itself, and the
# agent it was added for writes Python rather than tool calls. Under Code Mode
# the model is handed one callable and a generated SDK listing, and *every*
# registered tool is in that listing — so these tools need no Code Mode work of
# their own. What needs pinning is that they are actually there: a bundle that
# resolves to nothing, or a row that mounts and registers nothing, both look
# fine from outside and leave the model with no way to ask.


def test_the_bundle_is_discoverable_without_importing_it() -> None:
    """The entry point is the whole coupling between a profile and this package.

    `ph-app` must be able to compose a profile that layers this bundle without
    depending on this distribution, so the profile discovers it through the
    `ph.bundles` group. A typo in that entry point would leave the profile
    failing for a user and nothing else in the suite would notice.
    """
    from ph.bundles import installed_bundles, resolve_bundle
    from ph_code_graph import BUNDLE

    assert "code-graph" in installed_bundles()
    assert resolve_bundle("code-graph") == BUNDLE


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in someone's session."""
    from ph.cordis import Profile
    from ph.cordis.loader import resolve_plugin
    from ph_code_graph import BUNDLE

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        assert resolve_plugin(row["name"]) is not None, row["name"]


def test_the_bundles_row_is_enabled() -> None:
    """Unlike `tool-todo`, this one is on: layering it has one meaning.

    A row that stood down would make the profile that layers this bundle a no-op
    with a comment explaining why.
    """
    from ph.cordis import Profile
    from ph_code_graph import BUNDLE

    # `to_dump` omits `disabled` when it is false, so `enabled_rows()` is the
    # question rather than a key that may not be there.
    assert [row.id for row in Profile.from_paths([BUNDLE]).enabled_rows()] == ["code-graph"]


async def test_the_rlm_indexed_profile_layers_this_bundle() -> None:
    """The wiring a person can type: `ph --profile rlm-indexed`.

    Asserted here rather than only in `ph-app`, because the *reason* the profile
    exists is this package — and `available_profiles()` gating on bundle
    resolution is what makes an install without this distribution see no
    `rlm-indexed` rather than one that fails at mount.
    """
    from ph_app.profiles import available_profiles, resolve_profile
    from ph_code_graph import BUNDLE

    assert "rlm-indexed" in available_profiles()
    assert BUNDLE in resolve_profile("rlm-indexed")


async def test_code_mode_hands_the_model_both_tools_through_the_sdk(
    mount: Any, tmp_path: Path
) -> None:
    """The claim the whole package rests on for its intended caller.

    Under `tools.mode: code` the model gets one callable and an SDK listing, so
    "can the RLM use this" is exactly "are these in the listing". Composed from
    Code Mode's own rows rather than by mounting the `rlm` profile, because the
    claim is about *any* Code Mode deployment and this way the test does not
    need `ph-rlm` installed to make it.
    """
    ctx = await mount(
        {"id": "tools", "config": {"mode": "code"}},
        {"id": "tools-code-mode", "name": "tools-code-mode"},
        {"id": "code-runtime-stub", "name": "code-runtime-stub"},
        {**ROW, "config": {"path": str(tmp_path / "graph.db")}},
    )
    agent = _agent(ctx)

    view = ctx.tools.view(scope=agent.ctx)
    assert view.mode == "code"
    assert view.schemas == (), "Code Mode offers one callable, not schemas"
    assert {"code_index", "code_graph"} <= set(view.visible)

    prompt = await ctx.system_prompt.assemble(agent=agent, scope=agent.ctx)
    sdk = dict(prompt.sections)["tools:sdk"]
    # The exact spelling the model will write, not merely the name somewhere.
    assert "async def tools.code_index(" in sdk
    assert "async def tools.code_graph(" in sdk
    # And the description it reads to decide whether to reach for them.
    assert "Ask about a codebase's structure" in sdk


async def test_every_row_in_the_rlm_indexed_profile_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five documents from four distributions, and nothing inactive.

    A row that mounts and never activates — an unmet `inject` key — looks
    identical in `--dump-config` to one that runs, so this is the only place the
    difference is visible. It matters more for this profile than for the others
    because it is the only one composed from four distributions' bundles: a key
    one of them provides and another waits on is exactly what would fail here
    and nowhere else.

    Asserted from this package because the profile exists for it. `ph-app` owns
    the `PROFILES` entry and cannot test it — nothing there depends on the two
    indexing distributions, which is the point of discovering them as bundles.
    """
    from ph.cordis import Context, Profile, load_profile_documents
    from ph_app.profiles import resolve_profile

    # `PH_HOME`/`PH_CACHE` are not pinned here: the suite's autouse
    # `_isolated_home` already does it, and a second writer of one rule is how
    # the first comes to be believed and bypassed.
    documents = load_profile_documents(resolve_profile("rlm-indexed"))
    documents.append(
        (
            "test-overlay",
            [
                {"id": "fs", "config": {"root": str(tmp_path)}},
                # The host probe and the managed venv are the two rows that
                # reach outside the process; neither is what this asserts.
                {"id": "sandbox-local", "disabled": True},
                {"id": "code-runtime-python", "disabled": True},
                {"id": "code-runtime-stub", "name": "code-runtime-stub"},
            ],
        )
    )
    ctx = Context()
    try:
        mount = await Profile.from_documents(documents, name="rlm-indexed").mount(ctx)
        # **Exactly these two, and no others.** `rlm-skills-python` and
        # `rlm-kernel-snapshot` both `inject=["python_runtime"]`, which only the
        # real `code-runtime-python` provides — so swapping in the stub above is
        # what makes them inactive, and saying so keeps the assertion able to
        # catch a *third* row that stops activating. `<=` rather than `==`
        # because a build without those rows must not fail here either.
        assert set(mount.inactive()) <= {"rlm-skills-python", "rlm-kernel-snapshot"}, (
            mount.inactive()
        )
        for row in ("code-graph", "text-index", "text-index-local"):
            assert row not in mount.inactive(), f"{row} mounted and never activated"
        # And the point of the profile: the RLM's own surface has all four.
        from ph.cordis import DEPLOYMENT

        visible = set(ctx.tools.view(scope=DEPLOYMENT).visible)
        assert {"code_index", "code_graph", "text_index", "text_search"} <= visible
        assert {"read", "grep", "glob"} <= visible, "the tools these point at"
    finally:
        await ctx.drain()
        await ctx.dispose()


# --------------------------------------------------- provisioning, and the skill ----


async def test_the_command_reports_grammar_readiness(mount: Any, tmp_path: Path) -> None:
    """A person asks before an agent does. Costs no model turn — the seam's rule.

    It matters less here than for `text-index`, because 26 grammars are inside
    the wheel and only the long tail fetches — but "is this ready" should have
    one answer per plugin, asked the same way.
    """
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})
    agent = _agent(ctx)

    said = await ctx.commands.dispatch("/code-graph status", agent=agent, scope=agent.ctx)

    assert "ready" in said
    # The path it names is the one `use_cache` chose, not `~/.cache`.
    assert "tree-sitter" in said
    installed = await ctx.commands.dispatch("/code-graph install", agent=agent, scope=agent.ctx)
    assert "grammars ready" in installed


async def test_a_name_that_is_not_an_indexable_language_is_filtered_out(
    mount: Any, tmp_path: Path
) -> None:
    """Not "missing" — not applicable, which is a different sentence.

    `/code-graph` used to report a language ready when the pack had a *parser*
    for it, so `bash` and `sql` counted as ready and then handed the extractor a
    language with no tags query. Filtering the list through `indexable` first
    means the count only ever covers languages that can produce an edge.
    """
    ctx = await mount(
        {
            **ROW,
            "config": {
                "path": str(tmp_path / "graph.db"),
                "languages": ["python", "sql", "not-a-language"],
            },
        }
    )
    agent = _agent(ctx)

    said = await ctx.commands.dispatch("/code-graph install", agent=agent, scope=agent.ctx)

    # One of the three can yield an edge, so one is what the count reports.
    assert "1 grammar" in said, said
    assert "not-a-language" not in said
    assert not indexable("sql"), "the premise: sql has a parser and no tags query"


async def test_a_grammar_that_is_not_on_disk_is_named(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported by name, because "19 of 20" sends a person looking.

    `local` is patched rather than pointed at a genuinely unfetched language:
    which languages a host has cached is not something a test can pin, and the
    branch under test is the reporting, not the pack's download.

    Patched **where it is defined**, not on the package that re-exported it:
    `readiness` resolves `local` in `_extract`'s own namespace, so a patch on the
    re-export bound nothing and the test passed by reporting the host's real
    grammars. Which is what it did when the partition moved next to `local`.
    """
    import ph_code_graph
    from ph_code_graph import _extract

    monkeypatch.setattr(_extract, "local", lambda language: False)
    monkeypatch.setattr(ph_code_graph, "ensure", lambda languages: ([], list(languages)))
    ctx = await mount(
        {**ROW, "config": {"path": str(tmp_path / "graph.db"), "languages": ["python"]}}
    )
    agent = _agent(ctx)

    status = await ctx.commands.dispatch("/code-graph status", agent=agent, scope=agent.ctx)
    assert "0 of 1 ready" in status
    assert "missing python" in status

    said = await ctx.commands.dispatch("/code-graph install", agent=agent, scope=agent.ctx)
    assert "could not fetch python" in said
    assert "needs network the first time" in said


async def test_the_grammars_are_cached_under_a_ph_root(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not `$XDG_CACHE_HOME`, which no pH root covers and no `ph doctor` names.

    The pack unpacks even its *bundled* grammars into this directory on first
    use, so where it points is not housekeeping — it is whether the row works
    in a container with a read-only `HOME`.
    """
    monkeypatch.delenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", raising=False)
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))

    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})

    assert ctx.code_graph.grammars == tmp_path / "cache" / "tree-sitter"
    assert ctx.code_graph.grammars.is_dir()
    rows = report_section(ctx, "Code graph")
    assert rows["grammars"] == str(tmp_path / "cache" / "tree-sitter")


async def test_the_skill_arrives_with_the_plugin(mount: Any, tmp_path: Path) -> None:
    """What the RLM reads before it reaches for these tools.

    Registered by the row, so the catalog entry exists exactly when the tools
    do — a procedure advertised for tools a deployment does not have is worse
    than no procedure. `skills-progressive` ships an empty `paths` precisely so
    that a skill is something a distribution installs on purpose (I7), and this
    is that deliberate act.
    """
    from ph.cordis import DEPLOYMENT

    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})

    assert "code-graph" in {one.name for one in ctx.skills.list(scope=DEPLOYMENT)}
    skill = ctx.skills.get("code-graph", DEPLOYMENT)
    assert skill is not None
    assert {"code_index", "code_graph"} <= set(skill.allowed_tools)
    # G9: a line in every prompt, a page on disk.
    assert len(skill.description) < 300
    body = ctx.skills.body("code-graph", DEPLOYMENT)
    assert body is not None and len(body) > 1_500
    assert 'code_graph(mode="impact"' in body


async def test_the_skill_goes_away_with_the_row(mount: Any) -> None:
    """A catalog entry for tools nobody has is the failure this must not have."""
    from ph.cordis import DEPLOYMENT

    ctx = await mount()

    assert "code-graph" not in {one.name for one in ctx.skills.list(scope=DEPLOYMENT)}


def test_the_grammar_cache_release_restores_the_base_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pack.configure` is process-global, so it is taken as an effect (§4.9, I2).

    Without a release the library kept pointing at an unmounted row's
    `$PH_CACHE` — a per-test `tmp_path` this suite then deleted. Harmless while
    every mount in a process computes the same path, and exactly the kind of
    global that is wrong depth until two deployments in one process disagree.

    Asserted against the function rather than through a mount and a dispose,
    because the thing under test *is* a process-wide global: a test that read it
    before and after a mount would be reading whatever the previous test left,
    and would pass or fail on collection order.

    This is also the test that caught the first version of the fix, which
    restored `pack.cache_dir()` — the resolved leaf — and so re-appended the
    pack's own suffix on every unwind.
    """
    import tree_sitter_language_pack as pack

    from ph_code_graph._extract import cache_release, use_cache

    # The env var is `use_cache`'s operator override and short-circuits it.
    monkeypatch.delenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", raising=False)
    first, second = tmp_path / "one", tmp_path / "two"

    use_cache(first)
    assert str(first) in pack.cache_dir()

    release = cache_release(second)
    assert str(second) in pack.cache_dir(), "the effect did not take"

    release()

    settled = pack.cache_dir()
    assert str(first) in settled, "the base it replaced was not restored"
    assert str(second) not in settled
    # And the leaf is not nested: restoring a *resolved* path would have made
    # `<base>/tree-sitter-language-pack/<v>/libs/tree-sitter-language-pack/...`.
    assert settled.count("tree-sitter-language-pack") == 1


# ------------------------------------------------- the version-control filter ----


@pytest.mark.needs_git
async def test_a_reindex_under_git_never_opens_an_unchanged_file(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the filter: proved unchanged, so never read.

    Counted at `ctx.fs.read`, because "unchanged" was already reported by the
    content hash before the filter existed — the *only* observable difference is
    whether the file was opened, and a test that watched the counts alone would
    pass against both implementations.
    """
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})
    root = await git_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add pkg")
    agent = _agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    assert first.value["indexed"] == 1, text_of(first.content)

    reads = _counting_reads(monkeypatch)

    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))

    assert again.value["unchanged"] == 1
    assert again.value["indexed"] == 0
    assert reads == [], f"git vouched for the file and it was read anyway: {reads}"


@pytest.mark.needs_git
async def test_an_edited_file_is_still_read_and_reindexed(mount: Any, tmp_path: Path) -> None:
    """The safe direction. git's index still holds the old blob id for it, so
    this is `status` doing its job — see `ph.seams.changes`."""
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})
    root = await git_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add pkg")
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))

    (root / "pkg" / "a.py").write_text("def renamed():\n    return 2\n", encoding="utf-8")
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))

    assert again.value["indexed"] == 1, "an uncommitted edit must be picked up"
    found = await run_tool(
        ctx, "code_graph", {"mode": "define", "query": "renamed"}, agent=_agent(ctx)
    )
    assert found.value["symbols"], "the new symbol is in the graph"


@pytest.mark.needs_jj
async def test_a_reindex_under_jj_never_opens_an_unchanged_file(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jj needs no commit — it snapshots the working copy, so the token moves.

    The interesting half is that the *first* run stores a usable token even
    though it proved nothing, which is what makes the second run cheap.
    """
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "graph.db")}})
    root = await jj_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))
    assert first.value["indexed"] == 1, text_of(first.content)

    reads = _counting_reads(monkeypatch)

    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))

    assert again.value["unchanged"] == 1
    assert reads == [], f"jj vouched for the file and it was read anyway: {reads}"


async def test_a_tree_with_no_version_control_still_indexes(mount: Any, tmp_path: Path) -> None:
    """The filter is an optimisation, so losing it must cost nothing but speed.

    This is the property that makes it safe to put in front of an indexer at
    all: a caller that ignored `TreeState` entirely would still be correct.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = _agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=_agent(ctx))

    assert first.value["indexed"] == 2, "the fixture's two modules"
    # Still reported unchanged — by the content hash, which never went away.
    assert again.value["indexed"] == 0 and again.value["unchanged"] == 2
