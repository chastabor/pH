"""`code-graph`: line spans that are correct, and edges that point the right way.

Gates:

* Every span slices the source that it claims.
* A call inside a closure belongs to the closure.
* A second pass over an unchanged tree parses nothing.

## Why spans are checked against the file

Two coordinate systems meet in `_extract`. `ProcessConfig` spans are 0-based,
tree-sitter points are 0-based, and every line that pH shows a model is 1-based.
An off-by-one error passes every plausible test: the symbol is correct, the file
is correct, and the number looks fine. It shows as a model that reads the wrong
function and stops trusting the tool. The assertions here therefore slice the
real source at the span and check that the definition is really there. They do
not compare against a number that someone wrote down.

## Why the enclosing rule has its own test

`callers` is only as good as the attribution of a reference to the symbol that
contains it. The obvious implementation takes the first definition whose range
covers the line. That is wrong in exactly the case that this codebase is full
of: a closure inside a function. In `_registry.py`, `claim_key` spans lines 40-75
and its inner `release` spans lines 69-73, so a call on line 70 has two
candidates. With the outer one, the calls of every closure would be credited to
whatever contains the closure, and no count would look wrong.

## Why the real parser and the real database

Neither is stubbed. The parser is the thing under test. A language pack that
changed its span base or its symbol kinds must fail here and not in a session.
The index is 200 lines of SQL, and its two interesting queries (FTS5 and a
recursive CTE) are exactly the queries that a fake would not exercise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from code_graph_helpers import (
    SHARED,
    grammar_cache,  # noqa: F401 - autouse, by being here
    new_agent,
)
from tree_sitter import Query, QueryCursor

from ph.cordis import Context
from ph.json import as_str
from ph.keys import COMMANDS, FS, TOOLS
from ph.llm.types import text_of
from ph.testing import MountProfile, report_section, run_tool
from ph.testing.git import git, git_repo
from ph.testing.jj import jj_repo
from ph.workspace_caches import WORKSPACE_FILE
from ph_code_graph._extract import (
    INHERITS,
    Definition,
    _pack,
    detect_language,
    extract,
    indexable,
    owners,
)
from ph_code_graph.engine import CODE_GRAPH

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "code-graph", "name": "code-graph-engine"}
"""The engine row: the tools themselves, as the `ph-code-graph` command mounts them.

The row is mounted under the id of the session row, as the command does it. A
config that is written for `code-graph` is then read by the row that indexes.
`test_code_graph_row.py` covers the row that a session mounts."""

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
"""Deliberately the shapes that break simple extractors: a closure inside a
function, a method, and a module-level `async def` with a nested `def`. The last
shape crashed the PyPI `codegraph` outright."""

OTHER = SHARED


def _tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "mod.py").write_text(MODULE, encoding="utf-8")
    (root / "pkg" / "helpers.py").write_text(OTHER, encoding="utf-8")


async def _indexed(mount: MountProfile, tmp_path: Path, **config: object) -> Context:
    return await mount({**ROW, "config": config})


# ---------------------------------------------------------------- extraction ----


def test_every_span_slices_the_definition_it_claims() -> None:
    """The off-by-one error that would be invisible without this test. See the
    module docstring."""
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

    # A module-level `async def` that contains a nested `def`. The PyPI
    # `codegraph` crashed outright on this shape.
    assert "apply" in by_name and "enter" in by_name
    # A closure inside a plain function, and a method on a class.
    assert "outer" in by_name and "inner" in by_name
    assert "Widget" in by_name and "render" in by_name
    assert by_name["Widget"].kind == "class"
    assert by_name["outer"].kind == "function"


def test_docstrings_ride_along() -> None:
    """Docstrings make a `search` hit readable without a second call."""
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
    """The attribution rule that `callers` depends on. See the module docstring."""
    definitions = (
        Definition("claim_key", "function", 40, 75),
        Definition("release", "function", 69, 73),
    )

    table = owners(definitions)

    assert table[70].name == "release", "a closure's call belongs to the closure"
    assert table[45].name == "claim_key"
    assert 5 not in table, "module level is not inside anything"


def test_the_owner_table_is_built_once_for_the_whole_file() -> None:
    """Why this is a table and not a search: the search was O(definitions x
    references).

    A generated file with 1 500 definitions and 3 000 references used 58 ms in
    the scan for each reference, as much as its whole parse. `max_bytes` admits
    such files.
    """
    definitions = tuple(Definition(f"f{n}", "function", n * 10 + 1, n * 10 + 9) for n in range(200))

    table = owners(definitions)

    assert table[1].name == "f0"
    assert table[1991].name == "f199"
    assert 10 not in table, "the gap between two definitions belongs to neither"


def test_a_module_level_constant_is_a_definition() -> None:
    """The python query of the pack nests a constant under `expression_statement`,
    and its grammar puts an assignment directly under `module`. No module-level
    constant reached the index. `SUPPLEMENTS` adds the pattern that fits the tree.
    An assignment inside a function or a class is not a module constant."""
    code = (
        "PRICES = {'tea': 3}\n"
        "ENTRIES: list[object] = []\n"
        "A = B = 2\n"
        "box.lid = 1\n"
        "class Shop:\n"
        "    OPEN = True\n"
        "def total():\n"
        "    tax = 1\n"
        "    return tax\n"
    )

    found = extract("shop.py", code, "python")

    constants = {(one.name, one.start_line) for one in found.definitions if one.kind == "constant"}
    assert constants == {("PRICES", 1), ("ENTRIES", 2), ("A", 3), ("B", 3)}, constants

    # The canary. When the query of the pack finds a module constant by itself,
    # the supplement is not needed any more, and this fails so that it is removed.
    pack = _pack()
    own = Query(pack.get_language("python"), pack.get_tags_query("python"))
    tree = pack.get_parser("python").parse(code.encode("utf-8"))
    named = {
        node.text
        for _pattern, capture in QueryCursor(own).matches(tree.root_node)
        if "definition.constant" in capture
        for node in capture["name"]
    }
    assert b"PRICES" not in named, "the pack finds module constants now: remove the supplement"


def test_the_typescript_query_inherits_javascript() -> None:
    """A `.ts` file that is matched against the typescript query alone finds no calls."""
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
        # The parentheses are deliberate. A *bare* Ruby send (`helper` with no
        # parentheses) parses as an identifier and not as a `call`, so the tags
        # query of the language cannot see it. That is a real limitation of this
        # approach. The README states it, and this fixture does not hide it.
        ("ruby", "def helper\n 1\nend\ndef run\n helper()\nend\n", "helper"),
    ],
)
def test_other_languages_yield_definitions_and_calls(language: str, code: str, wanted: str) -> None:
    """Evidence that this is not a tool for Python only."""
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
    mount: MountProfile, tmp_path: Path
) -> None:
    """End to end: the reason why the package exists."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert not built.is_error, text_of(built.content)
    assert built.value["indexed"] == 2
    assert built.value["total_symbols"] > 5
    assert built.value["languages"] == ["python"]

    found = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    hit = found.value["symbols"][0]
    assert hit["path"] == "pkg/helpers.py"
    # The pointer is verified against the real file, not against a number.
    lines = (tmp_path / hit["path"]).read_text().splitlines()
    assert "def shared" in lines[hit["start_line"] - 1]
    assert f"{hit['path']}:{hit['start_line']}" in text_of(found.content)


async def test_an_index_records_the_workspace_its_graph_serves(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The record that lets a sweep remove the graph of a worktree that is gone.
    The directory names the workspace, which a digest cannot do
    (`ph.workspace_caches`).

    Sabotage: drop the `use` call from `code_index`, and nothing is recorded.
    """
    _tree(tmp_path)
    ctx = await mount(ROW)
    root = ctx.require(FS).root_for(None)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert not built.is_error, text_of(built.content)
    recorded = ctx.require(CODE_GRAPH).store_for(root).path.parent / WORKSPACE_FILE
    assert recorded.read_text(encoding="utf-8") == str(root)


async def test_callers_names_the_calling_symbol_and_the_calling_line(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The line is not the line of the definition. It is the line to open."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    edges = found.value["edges"]
    assert edges, text_of(found.content)
    callers = {one["name"] for one in edges}
    # `inner` and `enter` call it. `outer` and `apply` only contain them.
    assert {"inner", "enter"} <= callers, f"got {callers}"
    assert "outer" not in callers, "a closure's call was credited to its parent"
    for one in edges:
        line = (tmp_path / one["path"]).read_text().splitlines()[one["ref_line"] - 1]
        assert "shared(" in line, f"ref_line {one['ref_line']} reads {line!r}"


async def test_callees_reports_what_a_symbol_reaches(mount: MountProfile, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callees", "query": "render"}, agent=agent)

    assert not found.is_error, text_of(found.content)
    assert "outer" in {one["name"] for one in found.value["edges"]}


async def test_impact_walks_transitively_ring_by_ring(mount: MountProfile, tmp_path: Path) -> None:
    """The recursive CTE, which `pyturso` refuses. `shared` <- inner <- outer."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
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
    # `outer` calls `inner`, so a second hop must reach it. An implementation
    # with one hop would miss this silently.
    assert 2 in rings and "outer" in rings[2], f"rings were {rings}"


async def test_a_symbol_reached_twice_is_reported_at_the_nearer_distance(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`MIN(depth)` in the CTE makes the rings a budget and not a multiset."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(
        "def leaf():\n    return 1\n\n\ndef mid():\n    return leaf()\n\n\n"
        "def top():\n    return leaf() + mid()\n",
        encoding="utf-8",
    )
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "impact", "query": "leaf", "distance": 3}, agent=agent
    )

    rings = {r["distance"]: {one["name"] for one in r["symbols"]} for r in found.value["rings"]}
    assert "top" in rings[1], "top calls leaf directly"
    assert all("top" not in names for depth, names in rings.items() if depth > 1)


async def test_search_ranks_by_name_and_docstring(mount: MountProfile, tmp_path: Path) -> None:
    """FTS5, the other query that `pyturso` cannot serve."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "search", "query": "everyone calls"}, agent=agent
    )

    assert not found.is_error, text_of(found.content)
    assert "shared" in {one["name"] for one in found.value["symbols"]}


async def test_a_models_punctuation_does_not_become_fts_syntax(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`read-before-edit` is an FTS5 syntax error, and `a:b` is a column filter.

    A model that writes prose means neither. A tool that raised on the wording
    would teach the model to stop using the mode.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    for query in ("read-before-edit", 'the "widget" thing', "shared: value", "NEAR outer", "*"):
        found = await run_tool(ctx, "code_graph", {"mode": "search", "query": query}, agent=agent)
        assert not found.is_error, f"{query!r} raised: {text_of(found.content)}"


async def test_entities_lists_the_biggest_definitions_first(
    mount: MountProfile, tmp_path: Path
) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(
        ctx, "code_graph", {"mode": "entities", "path": "pkg", "limit": 4}, agent=agent
    )

    assert not found.is_error, text_of(found.content)
    spans = [one["lines"] for one in found.value["symbols"]]
    assert spans == sorted(spans, reverse=True)
    assert found.value["truncated"], "paging did not report there was more"
    assert "offset=4" in text_of(found.content)


async def test_an_ambiguous_name_says_it_is_ambiguous(mount: MountProfile, tmp_path: Path) -> None:
    """The real limit of a name-based graph is shown and not hidden."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def register():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def register():\n    return 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "c.py").write_text(
        "from a import register\n\n\ndef go():\n    return register()\n", encoding="utf-8"
    )
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "register"}, agent=agent)

    assert found.value["definitions"] == 2
    assert "defined in 2 places" in text_of(found.content)
    assert "mode=define" in text_of(found.content)


# ------------------------------------------------------------- incrementality ----


async def test_a_second_pass_over_an_unchanged_tree_parses_nothing(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The index is keyed by content, so a second run after a turn is cheap and
    correct."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    second = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert first.value["indexed"] == 2 and first.value["unchanged"] == 0
    assert second.value["indexed"] == 0, "an unchanged tree was re-parsed"
    assert second.value["unchanged"] == 2
    assert second.value["total_symbols"] == first.value["total_symbols"]


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path that `FsService.read` opens from here on.

    Both change-filter tests need the same thing. Their whole claim is that *the
    file was never opened*, and each had written the patch out in nine lines. The
    returned list is live, so assert against it after the call under test.
    """
    from ph.seams.fs import FsService

    reads: list[str] = []
    original = FsService.read

    async def counted(self: Any, path: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        reads.append(str(path))
        return await original(self, path, **kwargs)

    monkeypatch.setattr(FsService, "read", counted)
    return reads


async def test_a_touched_but_unmodified_file_stays_unchanged(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An mtime is not content. A rebase changes every mtime in the tree."""
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    target = tmp_path / "pkg" / "mod.py"
    target.touch()
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert again.value["indexed"] == 0


async def test_an_edit_is_picked_up_and_replaces_the_old_rows(
    mount: MountProfile, tmp_path: Path
) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
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


async def test_a_renamed_symbol_is_not_found_under_its_old_name(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The full-text index must also forget, and it did not (X1).

    `symbols_fts` is contentless. A contentless FTS5 table cannot report which
    postings belong to a row. The `'delete'` command needs the column values that
    the row was *inserted* with, and it was given three empty strings. SQLite
    accepts that silently and removes nothing. Every posting for every symbol
    that was ever re-indexed therefore stayed in the index.

    Stale postings would only be wasteful if the rowids were retired, but they
    are not. `symbols.id` has no `AUTOINCREMENT`, so the next insert reuses the
    id of the deleted symbol, and the old posting then points at a *different*
    symbol. `search` joins the FTS rowid back to `symbols`. A search for the old
    name therefore returns the new symbol, which is worse than a miss, because
    it reads like an answer.

    The sibling test above already covers `define`, and it passes in both cases.
    `define` queries `symbols` directly and never uses the FTS table. `search` is
    the mode that uses it, which is why the defect survived.
    """
    # **The test needs exactly one file.** `symbols.id` is reused only when the
    # deleted row held the highest id. With a second file indexed after it, the
    # stale posting points at an id that nothing has taken. The join in `search`
    # then drops it, and the defect is hidden behind a miss.
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg/helpers.py"]}, agent=agent)

    (tmp_path / "pkg" / "helpers.py").write_text(
        "def gamma(value):\n    return value\n", encoding="utf-8"
    )
    await run_tool(ctx, "code_index", {"paths": ["pkg/helpers.py"]}, agent=agent)

    stale = await run_tool(ctx, "code_graph", {"mode": "search", "query": "shared"}, agent=agent)
    assert stale.value["symbols"] == [], "the old name still has postings in the index"
    found = await run_tool(ctx, "code_graph", {"mode": "search", "query": "gamma"}, agent=agent)
    assert [one["name"] for one in found.value["symbols"]] == ["gamma"]


async def test_forget_removes_a_path_from_the_index(mount: MountProfile, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
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
    mount: MountProfile, tmp_path: Path
) -> None:
    """A tree walk must not fail because it found a bundle or a README."""
    _tree(tmp_path)
    (tmp_path / "pkg" / "README.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "pkg" / "huge.py").write_text("x = 1\n" * 5_000, encoding="utf-8")
    ctx = await _indexed(mount, tmp_path, max_bytes=1_000)
    agent = new_agent(ctx)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert not built.is_error, text_of(built.content)
    assert built.value["indexed"] == 2
    assert [one["path"] for one in built.value["skipped"]] == ["pkg/huge.py"]
    # Markdown is not skipped with a reason. It is simply not indexable. A report
    # of every prose file would make the skip list as long as the repository.
    assert all("README" not in one["path"] for one in built.value["skipped"])
    assert not indexable("markdown"), "the predicate this rests on"


async def test_a_query_before_any_index_says_what_to_do(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await _indexed(mount, tmp_path)

    found = await run_tool(
        ctx, "code_graph", {"mode": "search", "query": "anything"}, agent=new_agent(ctx)
    )

    assert found.is_error
    assert "run `code_index` first" in text_of(found.content)


async def test_the_rows_scope_is_what_a_bare_call_indexes(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A profile points the row at a package, and `code_index` with no arguments
    indexes that package: its `paths`, without its `exclude`."""
    for package in ("ours", "theirs"):
        (tmp_path / package / "tests").mkdir(parents=True)
        (tmp_path / package / "mod.py").write_text(OTHER, encoding="utf-8")
        (tmp_path / package / "tests" / "test_mod.py").write_text(OTHER, encoding="utf-8")
    ctx = await _indexed(mount, tmp_path, index={"paths": ["ours"], "exclude": ["**/tests/**"]})
    agent = new_agent(ctx)

    built = await run_tool(ctx, "code_index", {}, agent=agent)

    assert built.value["indexed"] == 1
    found = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert [one["path"] for one in found.value["symbols"]] == ["ours/mod.py"]


async def test_a_call_can_narrow_the_rows_scope_and_cannot_lift_its_exclusions(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The `exclude` of a call is added to the `exclude` of the row. If it
    replaced it, any call could undo the exclusion of a deployment."""
    (tmp_path / "pkg" / "vendor").mkdir(parents=True)
    for name in ("a.py", "b.py", "vendor/dep.py"):
        (tmp_path / "pkg" / name).write_text(OTHER, encoding="utf-8")
    ctx = await _indexed(mount, tmp_path, index={"exclude": ["**/vendor/**"]})

    built = await run_tool(
        ctx, "code_index", {"paths": ["pkg"], "exclude": ["**/b.py"]}, agent=new_agent(ctx)
    )

    assert built.value["indexed"] == 1, "a.py: not b.py, which the call left out, nor vendor/"


async def test_a_mode_that_needs_a_query_says_so(mount: MountProfile, tmp_path: Path) -> None:
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callers"}, agent=agent)

    assert found.is_error
    assert "needs `query`" in text_of(found.content)


async def test_indexing_reads_through_the_fs_seam(mount: MountProfile, tmp_path: Path) -> None:
    """The claim that makes this a tool and not a primitive that reads the tree
    (I-9).

    A screen that is registered on `ctx.fs` decides what a walk may show. A file
    that the screen refuses must never reach the index. That holds only because
    the row globs and reads through the seam.
    """
    _tree(tmp_path)
    (tmp_path / "pkg" / "secret.py").write_text("def hidden():\n    return 1\n", encoding="utf-8")
    ctx = await _indexed(mount, tmp_path)
    ctx.require(FS).screen(
        lambda path, name, agent, is_dir: "skip" if name == "secret.py" else "yield",
        scope=ctx,
    )

    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))
    found = await run_tool(
        ctx, "code_graph", {"mode": "define", "query": "hidden"}, agent=new_agent(ctx)
    )

    assert found.value["symbols"] == [], "a screened file reached the index"


async def test_the_row_reports_itself_to_doctor(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await _indexed(mount, tmp_path)

    rows = report_section(ctx, "Code graph")
    assert rows["state"] == "not built — run code_index"

    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))
    rows = report_section(ctx, "Code graph")
    assert rows["files"] == "0", "nothing was indexed in this test"


async def test_a_callees_reference_line_names_its_own_file(
    mount: MountProfile, tmp_path: Path
) -> None:
    """For `callees`, `path` and `ref_path` are different files, and both matter.

    The symbol is the *definition* of the callee, and the reference is a line in
    the caller. When the line was rendered against the path of the definition,
    the output was `helpers.py:19` for a line that is in `mod.py`. That points to
    nothing, and it looks entirely plausible.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    found = await run_tool(ctx, "code_graph", {"mode": "callees", "query": "inner"}, agent=agent)

    edges = [one for one in found.value["edges"] if one["name"] == "shared"]
    assert edges, text_of(found.content)
    edge = edges[0]
    assert edge["path"] == "pkg/helpers.py", "the definition is in helpers"
    assert edge["ref_path"] == "pkg/mod.py", "the call is in mod"
    # The line really is the call, in the file that the result names.
    line = (tmp_path / edge["ref_path"]).read_text().splitlines()[edge["ref_line"] - 1]
    assert "shared(" in line, f"{edge['ref_path']}:{edge['ref_line']} reads {line!r}"
    assert f"{edge['ref_path']}:{edge['ref_line']}" in text_of(found.content)


def test_doc_comments_attach_only_when_directly_above() -> None:
    """The positional rule, and the off-by-one error that broke it in both
    directions.

    The `///` comment node of Rust includes its trailing newline. With
    `span.end_line`, the rule missed the definition one line below *and* matched
    one two lines below. A count of the lines that the text spans is exact.
    """
    attached = extract("a.rs", "/// Doc for helper.\nfn helper() -> i32 { 1 }\n", "rust")
    assert {one.name: one.doc for one in attached.definitions}["helper"] == "Doc for helper."

    gapped = extract("a.rs", "/// Floating.\n\nfn helper() -> i32 { 1 }\n", "rust")
    assert {one.name: one.doc for one in gapped.definitions}["helper"] is None


def test_a_run_of_doc_comments_is_kept_whole() -> None:
    """Rust and Go write a paragraph as consecutive single-line comments.

    If only the last comment were attached, the index would hold the closing
    sentence of every doc comment in those languages and silently drop the rest.
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
    """A limitation of one language, tested so that it is a known shape and not a
    surprise.

    The tags query of C has `@definition.*` captures and no `@reference.*`
    captures. `search`, `define`, and `entities` therefore work for C, and
    `callers` and `callees` are empty. It is better to state this in a test than
    to find it in a session.
    """
    found = extract("a.c", "int helper(void){return 1;}\nint main(void){return helper();}\n", "c")

    assert {one.name for one in found.definitions} >= {"helper", "main"}
    assert found.references == (), "if C gained reference captures, use them"


async def test_an_unwritable_grammar_cache_refuses_with_a_sentence(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fact of the deployment that an operator can fix, not a traceback.

    The grammars are materialized into `$PH_CACHE` on first use, so a read-only
    cache is fatal. The contract of pH is that a row that cannot honor its
    configuration raises `MountRefusal`. Every command that mounts a profile
    turns that into a sentence and an exit code. This was measured inside a
    sandbox that made `~/.cache` read-only, where the failure arrived as fourteen
    frames of `pathlib`.
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
# `ph-code-graph` exists so that an agent can ask a codebase about itself. The
# agent that it was added for writes Python and not tool calls. In Code Mode the
# model gets one callable and a generated SDK listing, and *every* registered
# tool is in that listing. These tools therefore need no Code Mode work of their
# own. What needs a test is that the tools are really there. A bundle that
# resolves to nothing, and a row that mounts and registers nothing, both look
# fine from outside and leave the model with no way to ask.


def test_the_bundle_is_discoverable_without_importing_it() -> None:
    """The entry point is the whole coupling between a profile and this package.

    `phern` must be able to compose a profile that layers this bundle without
    importing this package. The profile therefore discovers the bundle through
    the `ph.bundles` group. With a typo in that entry point, the profile
    would fail for a user, and nothing else in the suite would notice.
    """
    from ph.bundles import installed_bundles, resolve_bundle
    from ph_code_graph import BUNDLE

    assert "code-graph" in installed_bundles()
    assert resolve_bundle("code-graph") == BUNDLE


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in the session of a user."""
    from ph.cordis import Profile
    from ph.cordis.loader import resolve_plugin
    from ph_code_graph import BUNDLE

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        name = as_str(row["name"])
        assert resolve_plugin(name) is not None, name


def test_the_bundles_row_is_enabled() -> None:
    """Unlike `tool-todo`, this row is on, because layering the bundle has one
    meaning.

    A row that stood down would make the profile that layers this bundle do
    nothing.
    """
    from ph.cordis import Profile
    from ph_code_graph import BUNDLE

    # `to_dump` omits `disabled` when it is false. The test therefore asks
    # `enabled_rows()` and does not read a key that may be absent.
    assert [row.id for row in Profile.from_paths([BUNDLE]).enabled_rows()] == ["code-graph"]


async def test_the_rlm_indexed_profile_layers_this_bundle() -> None:
    """The wiring that a person can type: `phern --profile rlm-indexed`.

    This is asserted here and not only in `phern`, because this package is the
    *reason* why the profile exists. `available_profiles()` checks that bundles
    resolve. An install without this distribution therefore sees no
    `rlm-indexed`, and does not see one that fails at mount.
    """
    from ph_app.profiles import available_profiles, resolve_profile
    from ph_code_graph import BUNDLE

    assert "rlm-indexed" in available_profiles()
    assert BUNDLE in resolve_profile("rlm-indexed")


async def test_every_row_in_the_rlm_indexed_profile_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five documents from four distributions, and nothing inactive.

    A row that mounts and never activates, because an `inject` key is not met,
    looks the same in `--dump-config` as a row that runs. This test is the only
    place where the difference shows. It matters more for this profile than for
    the others, because this is the only profile that is composed from the
    bundles of four distributions. A key that one of them provides and another
    waits on would fail here and in no other place.

    This is asserted from this package because the profile exists for it.
    `phern` owns the `PROFILES` entry and cannot test it. `ph_app` imports
    neither of the two indexing packages, which is the purpose of discovering
    them as bundles.
    """
    from ph.cordis import Context, Profile, ProfileDocument, load_profile_documents
    from ph_app.profiles import resolve_profile

    # `PH_HOME` and `PH_CACHE` are not set here. The autouse `_isolated_home`
    # fixture of the suite already sets them. A second writer of one rule leads
    # people to trust the first and then bypass it.
    documents = load_profile_documents(resolve_profile("rlm-indexed"))
    documents.append(
        ProfileDocument(
            "test-overlay",
            [
                {"id": "fs", "config": {"root": str(tmp_path)}},
                # The host probe and the managed venv are the two rows that reach
                # outside the process. This test asserts neither.
                {"id": "sandbox-local", "disabled": True},
                {"id": "code-runtime-python", "disabled": True},
                {"id": "code-runtime-stub", "name": "code-runtime-stub"},
            ],
        )
    )
    ctx = Context()
    try:
        mount = await Profile.from_documents(documents, name="rlm-indexed").mount(ctx)
        # **Exactly these two rows, and no others.** `rlm-skills-python` and
        # `rlm-kernel-snapshot` both have `inject=["python_runtime"]`, which only
        # the real `code-runtime-python` provides. The stub above therefore makes
        # them inactive. Stating that keeps the assertion able to catch a *third*
        # row that stops activating. The comparison is `<=` and not `==`, because
        # a build without those rows must not fail here.
        assert set(mount.inactive()) <= {"rlm-skills-python", "rlm-kernel-snapshot"}, (
            mount.inactive()
        )
        for row in ("code-graph", "text-index", "text-index-local"):
            assert row not in mount.inactive(), f"{row} mounted and never activated"
        # The purpose of the profile: the RLM's own surface has all four tools.
        from ph.cordis import DEPLOYMENT

        visible = set(ctx.require(TOOLS).view(scope=DEPLOYMENT).visible)
        assert {"code_index", "code_graph", "text_index", "text_search"} <= visible
        assert {"read", "grep", "glob"} <= visible, "the tools these point at"
    finally:
        await ctx.drain()
        await ctx.dispose()


# --------------------------------------------------- provisioning, and the skill ----


async def test_the_command_reports_grammar_readiness(mount: MountProfile, tmp_path: Path) -> None:
    """A person asks before an agent does. It costs no model turn, by the rule of
    the seam.

    This matters less here than for `text-index`, because 26 grammars are inside
    the wheel and only the remaining ones are fetched. But "is this ready" must
    have one answer for each plugin, asked in the same way.
    """
    ctx = await mount(ROW)
    agent = new_agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/code-graph status", agent=agent, scope=agent.ctx)

    assert said is not None and "ready" in said
    # The path that it names is the one that `use_cache` chose, not `~/.cache`.
    assert said is not None and "tree-sitter" in said
    installed = await ctx.require(COMMANDS).dispatch(
        "/code-graph install", agent=agent, scope=agent.ctx
    )
    assert installed is not None and "grammars ready" in installed


async def test_a_fetch_that_fails_is_the_last_the_process_tries(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confined command cannot reach GitHub. When it asked one time for each
    language that had no grammar, it made one refused connection for each. Each
    was a `sandbox/denied` in the log of the agent, on every call that a host
    answered. Now the first fetch that fails is the last one that the process
    tries, across calls. Every such file is still skipped, with the reason that
    names the command. A grammar on disk that does not load is not a refused
    fetch, and stops nothing."""
    from ph_code_graph import _extract

    _tree(tmp_path)
    (tmp_path / "pkg" / "web.js").write_text("function go() { return 1 }\n", encoding="utf-8")
    (tmp_path / "pkg" / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    on_disk = {"javascript"}
    asked: list[str] = []

    def unloadable(language: str) -> bool:
        asked.append(language)
        return False

    monkeypatch.setattr(_extract, "local", lambda language: language in on_disk)
    monkeypatch.setattr(_extract, "parseable", unloadable)
    monkeypatch.setattr(_extract, "_fetch_refused", False)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)

    built = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    assert not built.is_error and not again.is_error, text_of(built.content)
    fetched = [one for one in asked if one not in on_disk]
    assert len(fetched) == 1, f"asked GitHub for {fetched}"
    assert asked.count("javascript") == 2, "what is on disk is loaded at every call"
    assert built.value["indexed"] == 0
    assert sorted(one["path"] for one in built.value["skipped"]) == [
        "pkg/helpers.py",
        "pkg/main.go",
        "pkg/mod.py",
        "pkg/web.js",
    ]
    assert all("/code-graph install" in one["reason"] for one in built.value["skipped"])


async def test_a_name_that_is_not_an_indexable_language_is_filtered_out(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Not "missing" but not applicable, which is a different sentence.

    `/code-graph` used to report a language as ready when the pack had a *parser*
    for it. `bash` and `sql` then counted as ready, and the extractor got a
    language with no tags query. The list is now filtered through `indexable`
    first, so the count covers only languages that can produce an edge.
    """
    ctx = await mount(
        {
            **ROW,
            "config": {
                "languages": ["python", "sql", "not-a-language"],
            },
        }
    )
    agent = new_agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/code-graph install", agent=agent, scope=agent.ctx)

    # One of the three can yield an edge, so the count reports one.
    assert said is not None and "1 grammar" in said, said
    assert "not-a-language" not in said
    assert not indexable("sql"), "the premise: sql has a parser and no tags query"


async def test_a_grammar_that_is_not_on_disk_is_named(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grammar is reported by name, because "19 of 20" makes a person search
    for the missing one.

    `local` is patched. The test does not use a language that is really not
    fetched, because a test cannot control which languages a host has cached.
    The branch under test is the reporting, not the download of the pack.

    The patch is applied **where `local` is defined**, not on the package that
    re-exported it. `readiness` resolves `local` in the namespace of `_extract`.
    A patch on the re-export bound nothing, and the test passed by reporting the
    real grammars of the host. That happened when the partition moved next to
    `local`.
    """
    from ph_code_graph import _extract, engine

    monkeypatch.setattr(_extract, "local", lambda language: False)
    monkeypatch.setattr(engine, "ensure", lambda languages: ([], list(languages)))
    ctx = await mount({**ROW, "config": {"languages": ["python"]}})
    agent = new_agent(ctx)

    status = await ctx.require(COMMANDS).dispatch(
        "/code-graph status", agent=agent, scope=agent.ctx
    )
    assert status is not None and "0 of 1 ready" in status
    assert status is not None and "missing python" in status

    said = await ctx.require(COMMANDS).dispatch("/code-graph install", agent=agent, scope=agent.ctx)
    assert said is not None and "could not fetch python" in said
    assert said is not None and "needs network the first time" in said


@pytest.mark.parametrize("fetchable", [True, False])
def test_install_fetches_what_is_missing_in_one_batch_and_one_at_a_time_if_it_fails(
    monkeypatch: pytest.MonkeyPatch, fetchable: bool
) -> None:
    """`ensure` asks the `prefetch` of the pack for the grammars that are not on
    disk. `prefetch` fetches all of them, where the `download()` of pack 1.16.2
    fetched none of those that it calls bundled. One name that the pack cannot
    fetch fails the whole batch. Each name that is still missing is then
    requested alone, and the one that cannot be obtained is named.

    A pack with one method stands in for the real pack, because a test cannot
    control which grammars a host has cached.
    """
    from ph_code_graph import _extract

    on_disk = {"rust"}
    batches: list[list[str]] = []
    alone: list[str] = []

    class Pack:
        def prefetch(self, languages: list[str]) -> None:
            batches.append(languages)
            if not fetchable:
                raise RuntimeError("Download error: kotlin is not in the manifest")
            on_disk.update(languages)

    def materialized(language: str) -> bool:
        alone.append(language)
        if language != "kotlin":
            on_disk.add(language)
        return language in on_disk

    monkeypatch.setattr(_extract, "_pack", Pack)
    monkeypatch.setattr(_extract, "indexable", lambda language: language != "sql")
    monkeypatch.setattr(_extract, "local", lambda language: language in on_disk)
    monkeypatch.setattr(_extract, "parseable", materialized)

    ready, unavailable = _extract.ensure(["python", "rust", "kotlin", "sql"])

    assert batches == [["python", "kotlin"]], "rust is on disk, and sql has no tags query"
    if fetchable:
        assert (alone, ready, unavailable) == ([], ["python", "rust", "kotlin"], ["sql"])
    else:
        assert alone == ["python", "kotlin"]
        assert (ready, unavailable) == (["python", "rust"], ["kotlin", "sql"])


def test_a_note_about_a_run_nobody_waited_for_says_what_it_skipped_and_why() -> None:
    """The note about the end of a background run says what its result would have
    said. For a graph, that means the skips. A grammar that could not be fetched
    leaves every file skipped, and "finished: 34 files" alone reads as a built
    graph."""
    from ph.text import BRIEFLY
    from ph_code_graph.engine import SHOWN, _index_summary

    reason = "no python parser available — /code-graph install"
    value = {
        "indexed": 0,
        "unchanged": 0,
        "symbols": 0,
        "skipped": [{"path": f"src/m{n}.py", "reason": reason} for n in range(BRIEFLY + 3)],
        "total_symbols": 0,
        "total_refs": 0,
        "total_files": 0,
        "languages": [],
    }

    brief = _index_summary(value, shown=BRIEFLY).splitlines()
    whole = _index_summary(value).splitlines()

    assert brief[0] == "Indexed 0 files (0 already current), 0 symbols."
    assert brief[1] == f"  skipped src/m0.py: {reason}"
    assert brief[-2] == "  [and 3 more skipped]"
    assert brief[-1] == "The graph holds 0 symbols and 0 references across 0 files (no languages)."
    assert len(whole) == 2 + BRIEFLY + 3 <= 2 + SHOWN, "a caller that waited is shown every one"


async def test_the_grammars_are_cached_under_a_ph_root(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not under `$XDG_CACHE_HOME`, which no pH root covers and no `phern doctor`
    names.

    The pack unpacks even its *bundled* grammars into this directory on first
    use. Where the cache points therefore decides whether the row works in a
    container with a read-only `HOME`.
    """
    monkeypatch.delenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", raising=False)
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))

    ctx = await mount(ROW)

    assert ctx.require(CODE_GRAPH).grammars == tmp_path / "cache" / "tree-sitter"
    assert ctx.require(CODE_GRAPH).grammars.is_dir()
    rows = report_section(ctx, "Code graph")
    assert rows["grammars"] == str(tmp_path / "cache" / "tree-sitter")


def test_the_grammar_cache_release_restores_the_base_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pack.configure` is process-global, so it is taken as an effect (§4.9, I2).

    Without a release, the library kept pointing at the `$PH_CACHE` of an
    unmounted row, which was a per-test `tmp_path` that this suite then deleted.
    That is harmless while every mount in a process computes the same path. It
    is still the kind of global that is at the wrong depth until two deployments
    in one process disagree.

    The test asserts against the function and not through a mount and a dispose,
    because the thing under test *is* a process-wide global. A test that read it
    before and after a mount would read whatever the previous test left, and
    would pass or fail by collection order.

    This test also caught the first version of the fix. That version restored
    `pack.cache_dir()`, the resolved leaf, and so appended the suffix of the
    pack again on every unwind.
    """
    import tree_sitter_language_pack as pack

    from ph_code_graph._extract import cache_release, use_cache

    # The environment variable is the operator override of `use_cache`, and it
    # short-circuits the function.
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
    # The leaf is also not nested. A restore of a *resolved* path would have made
    # `<base>/tree-sitter-language-pack/<v>/libs/tree-sitter-language-pack/...`.
    assert settled.count("tree-sitter-language-pack") == 1


# ------------------------------------------------- the version-control filter ----


@pytest.mark.needs_git
async def test_a_reindex_under_git_never_opens_an_unchanged_file(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The purpose of the filter: a file that is proved unchanged is never read.

    The count is taken at `ctx.fs.read`. The content hash already reported
    "unchanged" before the filter existed. The *only* difference that can be
    observed is whether the file was opened. A test that watched only the counts
    would pass against both implementations.
    """
    ctx = await mount(ROW)
    root = await git_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add pkg")
    agent = new_agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    assert first.value["indexed"] == 1, text_of(first.content)

    reads = _counting_reads(monkeypatch)

    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert again.value["unchanged"] == 1
    assert again.value["indexed"] == 0
    assert reads == [], f"git vouched for the file and it was read anyway: {reads}"


@pytest.mark.needs_git
async def test_an_edited_file_is_still_read_and_reindexed(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The safe direction. The index of git still holds the old blob id for the
    file, so this is `status` doing its job. See `ph.seams.changes`."""
    ctx = await mount(ROW)
    root = await git_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add pkg")
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    (root / "pkg" / "a.py").write_text("def renamed():\n    return 2\n", encoding="utf-8")
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert again.value["indexed"] == 1, "an uncommitted edit must be picked up"
    found = await run_tool(
        ctx, "code_graph", {"mode": "define", "query": "renamed"}, agent=new_agent(ctx)
    )
    assert found.value["symbols"], "the new symbol is in the graph"


@pytest.mark.needs_jj
async def test_a_reindex_under_jj_never_opens_an_unchanged_file(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jj needs no commit. It snapshots the working copy, so the token changes.

    The interesting part is that the *first* run stores a usable token although
    it proved nothing. That is what makes the second run cheap.
    """
    ctx = await mount(ROW)
    root = await jj_repo(ctx, tmp_path)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))
    assert first.value["indexed"] == 1, text_of(first.content)

    reads = _counting_reads(monkeypatch)

    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert again.value["unchanged"] == 1
    assert reads == [], f"jj vouched for the file and it was read anyway: {reads}"


async def test_a_tree_with_no_version_control_still_indexes(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The filter is an optimization, so its loss must cost only speed.

    This property makes it safe to put the filter in front of an indexer. A
    caller that ignored `TreeState` entirely would still be correct.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert first.value["indexed"] == 2, "the fixture's two modules"
    # Still reported as unchanged, by the content hash, which was never removed.
    assert again.value["indexed"] == 0 and again.value["unchanged"] == 2


async def test_a_deleted_file_leaves_the_index_on_the_next_sweep(
    mount: MountProfile, tmp_path: Path
) -> None:
    """X3: the walk never compared itself with what the index already held.

    A file that was removed from the tree kept its symbols until someone ran
    `forget` by hand. `code_graph` then continued to name definitions at paths
    that did not exist. A *rename* showed the symbol twice, one time under each
    name, and nothing said which one was real. A pointer into a file that does
    not exist is the one answer that this tool must not give, because the next
    step of the model is to open that file.

    Only a full sweep removes files. That is the second half of the test: a
    narrowed walk must not empty the index, or `code_index paths=["src/x.py"]`
    would delete everything else.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)

    (tmp_path / "pkg" / "helpers.py").unlink()
    swept = await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)

    assert not swept.is_error, text_of(swept.content)
    assert swept.value["removed"] == 1, "the deleted file kept its symbols"
    found = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert found.value["symbols"] == [], "a definition still points at a file that is gone"


async def test_a_walk_that_did_not_reach_a_file_does_not_forget_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The safeguard for X3, and the reason why the sweep asks the filesystem.

    The first version of this fix compared the index with the *walk*. It decided
    that a walk was a full sweep by matching the arguments (`paths == ["."]`).
    Two things make that wrong, and both empty the index and do not only trim
    it. `fs.collect` stops at `max_files` and reports nothing, so on a tree past
    the cap the walk is a prefix of the scope. The `permissions-fs` screen prunes
    whole directories, so a rule that hides a subtree makes those files
    "absent". A call to `exists()` handles all three cases (truncated, pruned,
    and narrowed), because the question is "is this file still there", and that
    question never depended on the walk.
    """
    _tree(tmp_path)
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)

    narrowed = await run_tool(ctx, "code_index", {"paths": ["pkg/helpers.py"]}, agent=agent)

    assert narrowed.value["removed"] == 0
    still = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert still.value["symbols"], "a narrowed walk emptied the index"

    # A walk that `max_files` truncated *looks* like a full sweep. The arguments
    # are the defaults of the row, and `paths` is still only a subset.
    capped = await _indexed(mount, tmp_path / "capped", max_files=1)
    narrow_agent = new_agent(capped)
    await run_tool(capped, "code_index", {"paths": ["."]}, agent=narrow_agent)
    swept = await run_tool(capped, "code_index", {"paths": ["."]}, agent=narrow_agent)

    assert swept.value["removed"] == 0, "a truncated walk forgot the files it never reached"


async def test_a_forgotten_symbols_docstring_stops_being_searchable(
    mount: MountProfile, tmp_path: Path
) -> None:
    """N5: the FTS delete must carry the values that the insert used.

    The `'delete'` command of FTS5 derives the postings to remove from the values
    that it is given. A delete that passes different values removes only the
    postings that those values imply. Measured against SQLite 3.45: a delete with
    `""`, for a symbol that was inserted with real docstring text, removes the
    **name** posting and leaves every **doc** posting behind.

    A stale posting is worse than a miss. `symbols.id` has no `AUTOINCREMENT`, so
    a later insert takes the id of the dead row, and the old docstring then
    answers with a different symbol. The test stages exactly that, because
    `search` joins the rowid back to `symbols`, and a posting whose symbol is
    simply gone is invisible.

    The test goes through `code_index` and `code_graph`, not against
    `_fts_values`. The first version of the fix shipped with the helper written,
    exported to a unit test, and never called. That test passed while both live
    expressions stayed inline.

    Sabotage: pass `""` as the doc at either call site and `zarquon` answers
    with the symbol that took the id.
    """
    _tree(tmp_path)
    (tmp_path / "pkg" / "old.py").write_text(
        'def retired():\n    """A zarquon of the third kind."""\n    return 1\n', encoding="utf-8"
    )
    ctx = await _indexed(mount, tmp_path)
    agent = new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    seen = await run_tool(ctx, "code_graph", {"mode": "search", "query": "zarquon"}, agent=agent)
    assert [one["name"] for one in seen.value["symbols"]] == ["retired"], "never indexed"

    # Forget it, then index a *new* symbol that takes the id of the dead row.
    (tmp_path / "pkg" / "old.py").unlink()
    await run_tool(ctx, "code_index", {"paths": ["pkg/old.py"], "forget": True}, agent=agent)
    (tmp_path / "pkg" / "new.py").write_text("def arrived():\n    return 2\n", encoding="utf-8")
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)

    gone = await run_tool(ctx, "code_graph", {"mode": "search", "query": "zarquon"}, agent=agent)

    assert gone.value["symbols"] == [], (
        "a docstring posting outlived its symbol and answered for the one that took its id"
    )


def test_two_mounts_released_out_of_order_leave_the_one_still_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each release takes back only its own claim on the grammar cache (Holdings
    rule 4).

    The release used to restore whatever was set before its claim. With two
    mounts in one process, and the first mount released first, the release reset
    the cache under the second mount while it was still mounted. The release of
    the second mount then restored the directory of the first, which was gone
    with its row by then.

    Sabotage: restore the previous base in `cache_release` again, and the second
    mount's cache is lost when the first is released.
    """
    import tree_sitter_language_pack as pack

    from ph_code_graph._extract import cache_release, use_cache

    monkeypatch.delenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", raising=False)
    base, first, second = tmp_path / "base", tmp_path / "one", tmp_path / "two"
    use_cache(base)

    release_first = cache_release(first)
    release_second = cache_release(second)
    release_first()
    assert str(second) in pack.cache_dir(), "the mount still up lost its cache"

    release_second()
    assert str(base) in pack.cache_dir(), "the base was not restored once both had gone"
