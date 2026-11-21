"""One file in, definitions and references out, with the lines that they are on.

The module makes two passes over `tree-sitter-language-pack`. Each pass answers
something that the other cannot, and both are cheap:

* **`process()`** is the intelligence layer of the pack. It gives definitions
  with kinds, the imports of the file, and its docstrings and comments. It does
  not give *references*: `SymbolInfo` says that a function exists, never who
  calls it. It also does not give a symbol its own prose. `SymbolInfo.doc` is
  always `None`. See `_documented`.
* **The tags query** is the `tags.scm` of tree-sitter for the language. GitHub
  builds its code navigation on the same data. It gives `@definition.*` and,
  most important, `@reference.call`. That is the edge that this package exists
  to record.

Measured over `packages/ph-core/src/ph`: 136 files and 1.58 MiB take 219 ms
through `process()` and 108 ms through the tags query. Both passes together index
the core of this repository in about a third of a second. The second parse gives
more value than it costs.

## The two coordinate systems, reconciled here

`ProcessConfig` spans are **0-based**, and tree-sitter points are 0-based too.
Every line number that pH shows to a model is **1-based**, because that is what
`read` takes and what an editor shows. Both are normalized one time, on the way
out of this module. Nothing downstream has to remember which source a number came
from. An error here gives a pointer that is off by one line. A model does not
report that kind of error. It stops trusting the tool.

## Query inheritance

The grammar of `tree-sitter-typescript` extends the javascript grammar, and
upstream splits the tags queries in the same way. The typescript query covers
only what typescript adds (interfaces, signatures, abstract classes). `class`,
`function`, and `call` are in the *javascript* query. A `.ts` file that is
matched against the typescript query alone yields two tags and no calls
(measured). `INHERITS` records that fact.

## Supplements

A query of the pack can miss a definition that its grammar has. The python query
writes a module-level constant as `(module (expression_statement (assignment
…)))`. The python grammar of the pack puts an assignment directly under `module`,
so that pattern matched only the inner half of a chained assignment (`B` of
`A = B = 2`), and no other module-level constant reached the index (measured with
tree-sitter 0.26.0 and pack 1.16.2). `SUPPLEMENTS` adds a pattern that matches the
tree, joined to the query of the pack so that a file is walked one time. When a
later grammar makes the pattern impossible, the joined query does not compile,
and the query of the pack is used alone, with a warning. That costs the
supplement, not every file of the language.

@module ph_code_graph._extract
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

from ._config import GRAMMAR_CACHE_ENV

__all__ = [
    "INHERITS",
    "SUPPLEMENTS",
    "Definition",
    "Extraction",
    "Reference",
    "cache_release",
    "detect_language",
    "ensure",
    "extract",
    "indexable",
    "local",
    "obtained",
    "owners",
    "parseable",
    "readiness",
    "use_cache",
]

log = logging.getLogger("ph_code_graph.extract")

_configured: Path | None = None
"""The cache base that this process last gave to the pack, or `None` for the
default of the pack.

The value is mirrored here because the pack does not report it. `cache_dir()`
returns the resolved leaf that it derives, and that leaf cannot be passed back
in. The value is module state because it mirrors a module-level global in
another library. A record in any narrower place would be a second answer to a
process-wide question.
"""

_fetch_refused = False
"""Whether a grammar fetch has failed in this process. After a failure,
`obtained` tries no more fetches. The value is process-wide for the same reason
as `_configured`, and `_reset` clears it."""

_claims: list[_CacheClaim] = []
"""The claim of each mount on the cache of the pack, oldest first
(`cache_release`). The newest claim is in force. The last release restores the
base from before the first claim."""

_base: Path | None = None
"""The cache base before the first claim that is still held was made."""

INHERITS: dict[str, tuple[str, ...]] = {
    "typescript": ("javascript",),
    "tsx": ("javascript",),
}
"""The languages whose own tags query is only part of the query. See the module
docstring."""

SUPPLEMENTS: dict[str, str] = {
    "python": "(module (assignment left: (identifier) @name) @definition.constant)",
}
"""Patterns for a definition that the query of the pack misses, by language. See
the module docstring."""

REFERENCE_KINDS = ("call",)
"""The `@reference.*` captures that become edges.

Only `call`, deliberately. The tags queries also emit `reference.type`,
`reference.class`, `reference.implementation`, and more. Each is a real
relationship, but they answer a different question from "what runs what". In one
edge table with calls, they would make `callers` return the places that only
*mention* a type. They are dropped, not stored and filtered, so that the index
does not carry rows that nothing reads.
"""


@dataclass(frozen=True, slots=True)
class Definition:
    """A symbol that this file defines."""

    name: str
    kind: str
    """`function`, `class`, `method`, `interface`, `module`, `constant`: the name
    that the tags query of the language uses, without the `definition.` prefix."""
    start_line: int
    """1-based, inclusive."""
    end_line: int
    """1-based, inclusive."""
    doc: str | None = None


@dataclass(frozen=True, slots=True)
class Reference:
    """A name that this file uses, and the line where it uses it."""

    name: str
    kind: str
    line: int
    """1-based."""


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything that one file yielded."""

    language: str
    definitions: tuple[Definition, ...]
    references: tuple[Reference, ...]
    imports: tuple[str, ...]
    lines: int


def _pack() -> Any:  # noqa: ANN401
    # Not a declared dependency: the row checks for it before this runs.
    import tree_sitter_language_pack  # noqa: PLC0415

    return tree_sitter_language_pack


def cache_release(directory: Path) -> Callable[[], None]:
    """Point the grammar cache of the pack at `directory`, and return the function
    that restores it.

    **This is an `enter` that returns its own release**, which is the shape that
    `ctx.effect` takes. The change therefore unwinds with the row that made it
    (§4.9, I2) and does not outlive it. `pack.configure` is process-global, and
    the last writer wins. Without the release, the library kept pointing at the
    `$PH_CACHE` of an unmounted row. This showed in the test suite, where that
    path is a per-test `tmp_path` that the pack still held after teardown.

    In production today this is harmless, because every mount in a process
    computes the same path. It was still at the wrong depth, and that is the
    argument for the rule: nobody notices a global that is taken without a
    release until two deployments in one process disagree.
    """
    # **Restore the base that this module set, not `cache_dir()`.** `cache_dir()`
    # reports the *resolved* leaf, `<base>/tree-sitter-language-pack/<version>/libs`.
    # If that is passed back to `PackConfig(cache_dir=...)`, the pack appends its
    # suffix a second time, and the restore nests the path and does not restore
    # it. The test for this function caught that, which is the argument for the
    # test.
    #
    # **Each release takes back its own claim and no other** (Holdings rule 4).
    # An earlier version restored whatever was set before the claim. Two mounts
    # that were released out of order then reset the cache under the mount that
    # was still up, and the last release restored a directory that was already
    # gone with its row. Now the newest claim that is still held is in force.
    # With no claim left, the base from before the first claim is restored.
    # The claim is recorded after the pack accepted the directory (Holdings rule
    # 1). When the pack refuses a directory, such as an unwritable `$PH_CACHE`,
    # this function raises and returns no release. A claim that was recorded
    # first then stayed on top of the stack permanently.
    global _base
    previous = _configured
    use_cache(directory)
    if not _claims:
        _base = previous
    claim = _CacheClaim(directory)
    _claims.append(claim)

    def restore() -> None:
        # Best effort: a failure is logged and not raised. A disposer that raises
        # is logged and the unwind continues anyway, so a loud failure here would
        # only hide whatever else the scope was releasing.
        if claim not in _claims:
            return
        _claims.remove(claim)
        wanted = _claims[-1].directory if _claims else _base
        try:
            if wanted is None:
                _reset()
            elif wanted != _configured:
                use_cache(wanted)
        except Exception:  # pragma: no cover - a library that cannot be restored
            log.warning("ph_code_graph: could not restore the grammar cache", exc_info=True)

    return restore


@dataclass(frozen=True, slots=True, eq=False)
class _CacheClaim:
    """The claim of one mount on the cache of the pack. Claims are compared by
    identity, so two mounts that name one directory are two claims."""

    directory: Path


def use_cache(directory: Path) -> Path:
    """Point the grammar cache of the pack at `directory`. Returns the directory
    that is in effect.

    **This is not optional.** The pack keeps every grammar that it uses in this
    directory. That includes the grammars that it calls bundled, which it
    downloads here from GitHub the first time a parser is requested. A deployment
    whose cache directory is not writable therefore fails:

        Download error: Failed to create cache directory … Permission denied

    The default is `$XDG_CACHE_HOME/tree-sitter-language-pack/<version>/libs`.
    That is correct on the laptop of a developer and wrong in a container with a
    read-only `HOME` (both measured). The row therefore gives the pack
    `$PH_CACHE`: the root that pH already uses for rebuildable artifacts, on a
    path that pH has created and verified.

    The function uses the `configure()` of the pack and does not set
    `TREE_SITTER_LANGUAGE_PACK_CACHE_DIR`. A row that changed the process
    environment would also change it for every child that `ctx.subprocess`
    starts, and this is a decision about *this* library only.

    An operator who set the environment variable deliberately keeps it. Their
    value wins, which is what keeps the variable meaningful.
    """
    global _configured

    override = os.environ.get(GRAMMAR_CACHE_ENV)
    if override:
        return Path(override)
    directory.mkdir(parents=True, exist_ok=True)
    pack = _pack()
    pack.configure(pack.PackConfig(cache_dir=str(directory)))
    _configured = directory
    return directory


def _reset() -> None:
    """Give the pack its own default back. This is the restore for a base of `None`."""
    global _configured, _fetch_refused

    pack = _pack()
    pack.configure()
    _configured = None
    _fetch_refused = False


def detect_language(path: str) -> str | None:
    """The name that the pack uses for the language of this path, or `None` for
    "not code".

    `None` is the ordinary answer for most of a repository (a `.md` file, a
    lockfile, an image), so it is a value and not an error. The pack also
    recognizes Markdown and other prose languages. The caller excludes those,
    because the caller knows that it wants code. This function only reports what
    the extension says.
    """
    try:
        found: str | None = _pack().detect_language_from_path(path)
    except Exception:
        return None
    return found


@cache
def indexable(language: str) -> bool:
    """Whether this language has a tags query, and can therefore yield an edge.

    **The answer is derived, not listed.** This function replaced a hand-written
    `PROSE` set of nine names. That set copied a property that the pack reports
    directly, and the copy was incomplete, because the detector recognizes 371
    languages. `.txt`, `.ini`, `.proto`, `.rst`, `.conf`, and a dozen others also
    have no tags query. With the default `glob: "**/*"` of the row, each of them
    reached the extractor and came back as an internal `TypeError` that was
    presented as a per-file skip reason.

    Asking the pack cannot drift, and it covers all 371 languages. It also costs
    nothing. The answer is cached, and `get_tags_query` reads a string that the
    pack already holds. It does not materialize a grammar.
    """
    try:
        return bool(_pack().get_tags_query(language))
    except Exception:
        return False


def local(language: str) -> bool:
    """Whether the grammar of this language is **already on disk**.

    This lists the cache, so it reads the disk and fetches nothing. That makes it
    the correct question for `/code-graph status`. A person who asks "is this
    ready" must not get an answer from an action that makes it ready.

    **This is deliberately not the question of the indexer.** The indexer asks
    `obtained`, which fetches a missing grammar where the network allows it. An
    unconfined run on a fresh cache therefore still indexes. Where a fetch is not
    possible (a confined command, or no network), the file is skipped with a
    reason that names `/code-graph install`.
    """
    try:
        return language in set(_pack().downloaded_languages())
    except Exception:
        return False


def parseable(language: str) -> bool:
    """Whether a parser for `language` can be obtained at all.

    This call materializes the grammar, on purpose. `get_parser` loads a grammar
    from the cache and fetches one that is not there. Measured 2026-11-11 with
    packs 1.16.2 and 1.25.0: on an empty cache with no network, no parser can be
    obtained, including the python parser. The 26 grammars that the pack calls
    bundled come as one archive from GitHub. `local` answers the readiness
    question without a fetch.
    """
    if not indexable(language):
        return False
    try:
        _pack().get_parser(language)
    except Exception:
        return False
    return True


def obtained(language: str) -> bool:
    """Whether a parser for `language` is available. This is the question of the
    indexer. A grammar on disk is loaded. A grammar that is not on disk is
    fetched, until one fetch fails in this process. A confined command cannot
    reach GitHub, and a fetch attempt for each missing language caused one
    `sandbox/denied` each. A grammar on disk that does not load does not stop
    later fetches."""
    global _fetch_refused

    if local(language):
        return parseable(language)
    if _fetch_refused:
        return False
    found = parseable(language)
    _fetch_refused = not found
    return found


def readiness(languages: Sequence[str]) -> tuple[list[str], list[str]]:
    """The `languages` that are on disk, and those that are not, in one pass.

    The function is here and not beside its caller because there were two
    versions of it. The provisioning command partitioned the list one way.
    `ensure` derived the same partition again, three functions later, with a
    `one not in ready` scan. That was the same O(n²) work that the docstring of
    the first version claimed to have replaced. With one function, the two
    callers cannot disagree about the list that the unavailable side comes from.
    """
    ready = [one for one in languages if local(one)]
    have = set(ready)
    return ready, [one for one in languages if one not in have]


def ensure(languages: Sequence[str]) -> tuple[list[str], list[str]]:
    """Fetch the grammars that are missing. Returns `(ready, unavailable)`. The
    provisioning command uses this function.

    It calls the `prefetch` of the pack, which downloads what is not on disk and
    loads each grammar. It does not call `download()`. Measured 2026-11-11 on an
    empty cache: the `download()` of pack 1.16.2 fetches nothing for the
    languages that the pack calls bundled, and `prefetch` fetches all 18 defaults
    on 1.16.2 and 1.25.0 in about 3 s. One name that the pack cannot fetch fails
    the whole batch, so the remaining names are then requested one at a time. A
    confined agent cannot reach GitHub and depends on an earlier run of this
    function.
    """
    wanted = [one for one in languages if indexable(one)]
    _, missing = readiness(wanted)
    if missing:
        try:
            _pack().prefetch(missing)
        except Exception:
            log.warning("ph_code_graph: grammar prefetch failed", exc_info=True)
            for one in readiness(missing)[1]:
                parseable(one)
    ready, _ = readiness(wanted)
    return ready, [one for one in languages if one not in set(ready)]


@cache
def _tags_query(language: str) -> Any:  # noqa: ANN401
    """The compiled tags query for one language, with its supplement when
    `SUPPLEMENTS` has one. **It is cached, and the saving is measured.**

    Compilation is work for each language, but it was done for each *file*:
    2.8 ms for python, 22 ms for ruby, and 35.7 ms for typescript. Typescript is
    the worst case, because `INHERITS` joins two queries. Over the `ph-core` tree
    of this repo (136 files), that was 44 % of all extraction time. A TypeScript
    package of a thousand files paid 35 s for it.

    `functools.cache` follows the precedent of `ph.seams.fs._compiled`, and it is
    safe for the same reason. A `Query` is read-only after it is built. `extract`
    makes a new `QueryCursor` for each call, and the cursor is the mutable part.
    """
    # Not a declared dependency: the row checks for it before this runs.
    from tree_sitter import Query  # noqa: PLC0415

    pack = _pack()
    grammar = pack.get_language(language)
    sources = [pack.get_tags_query(one) for one in INHERITS.get(language, ())]
    sources.append(pack.get_tags_query(language))
    query = "\n".join(one for one in sources if one)
    supplement = SUPPLEMENTS.get(language)
    if supplement:
        try:
            return Query(grammar, f"{query}\n{supplement}")
        except Exception:
            log.warning(
                "ph_code_graph: the %s supplement does not fit the grammar; it is skipped",
                language,
                exc_info=True,
            )
    return Query(grammar, query)


def extract(path: str, text: str, language: str) -> Extraction:
    """Parse `text` and return what it defines, what it references, and what it
    imports.

    :raises Exception: whatever the pack raises for a file that it cannot parse.
        This function does not catch it. The caller indexes a *batch*, and the
        caller decides whether one bad file stops the batch or is reported and
        skipped.
    """
    # Not a declared dependency: the row checks for it before this runs.
    from tree_sitter import QueryCursor  # noqa: PLC0415

    pack = _pack()
    config = pack.ProcessConfig(
        language=language, symbols=True, imports=True, docstrings=True, comments=True
    )
    result = pack.process(text, config)

    # +1: `Span.start_line` is 0-based, and everything downstream is 1-based. See
    # the module docstring.
    definitions = [
        Definition(
            name=symbol.name,
            kind=_kind_of(symbol.kind),
            start_line=symbol.span.start_line + 1,
            end_line=symbol.span.end_line + 1,
            # **Not `symbol.doc`.** This pack never sets it: it measured `None`
            # for every symbol in a file full of docstrings. The prose arrives on
            # two other channels, and `_documented` merges them.
            doc=None,
        )
        for symbol in (result.symbols or [])
        if symbol.name
    ]

    source = text.encode("utf-8")
    tree = pack.get_parser(language).parse(source)
    references: list[Reference] = []
    seen_definitions = {(one.name, one.start_line) for one in definitions}
    for _pattern, capture in QueryCursor(_tags_query(language)).matches(tree.root_node):
        named = capture.get("name") or []
        if not named:
            continue
        # `Node.text` is `bytes | None`. It is `None` when the tree outlived the
        # source that it was parsed from. That cannot happen here, but the
        # signature allows it, so the code handles it and does not assert.
        raw = named[0].text
        if raw is None:
            continue
        name = raw.decode("utf-8", errors="replace")
        for label, nodes in capture.items():
            if not label.startswith("definition.") and not label.startswith("reference."):
                continue
            head, _, tail = label.partition(".")
            line = nodes[0].start_point[0] + 1
            if head == "definition":
                # The tags query also finds definitions, mostly the same ones that
                # `process()` found. One is kept only when `process()` missed it,
                # which happens for a language whose intelligence layer is
                # thinner than its tags query. The match is on (name, line), so
                # the union has no duplicates to remove later.
                if (name, line) not in seen_definitions:
                    seen_definitions.add((name, line))
                    definitions.append(
                        Definition(
                            name=name,
                            kind=_kind_of(tail),
                            start_line=line,
                            end_line=nodes[0].end_point[0] + 1,
                        )
                    )
            elif tail in REFERENCE_KINDS:
                references.append(Reference(name=name, kind=tail, line=line))

    return Extraction(
        language=language,
        definitions=_documented(definitions, result),
        references=tuple(references),
        imports=tuple(one.source for one in (result.imports or []) if getattr(one, "source", None)),
        lines=text.count("\n") + 1,
    )


def _documented(
    definitions: list[Definition],
    result: object,
) -> tuple[Definition, ...]:
    """Attach the prose of each definition, from the two channels that carry it.

    **`SymbolInfo.doc` is always `None`** in this pack (measured against a file
    full of docstrings). A row that trusted it would have shipped an index with
    an empty `doc` column and a `search` mode that matched names only. The prose
    exists on two other channels:

    * **`result.docstrings`** carries `associated_item`, which is the pack's own
      answer to "which symbol does this docstring belong to". It is used as it
      is where present, because the pack knows the convention of each language
      better than a heuristic here would.
    * **`result.comments`** carries doc comments: the `///` of Rust and the block
      form of TypeScript (kind `Doc`), and the `//` of Go (kind `Line`, because
      that *is* the documented convention of Go). `associated_item` is not set
      for these. The rule is therefore positional and deliberately narrow: the
      last line of the comment must be **immediately above** the first line of a
      definition. A gap of one blank line means that the comment was about
      something else. Attaching it anyway would show the model the wrong prose.

    Prose that belongs to no definition, such as a module docstring or a comment
    inside a function body, is dropped. It is not attached to the nearest
    definition.
    """
    prose: dict[str, str] = {}
    starts: dict[int, str] = {}
    for one in definitions:
        starts.setdefault(one.start_line, one.name)

    for docstring in getattr(result, "docstrings", None) or []:
        item = getattr(docstring, "associated_item", None)
        text = (docstring.text or "").strip()
        if item and text:
            prose.setdefault(str(item), clean_prose(text))

    for _, last, text in _comment_blocks(result):
        # **Use the text of the comment, not its `span.end_line`.** The `///`
        # comment node of Rust includes the trailing newline, so its `end_line`
        # is one past the line that it occupies. With `end_line`, the
        # immediately-above rule missed the definition that it should have
        # matched *and* matched one a blank line away. A count of the lines that
        # the stripped text spans is exact for every language here.
        owner = starts.get(last + 1)
        if owner and text:
            prose.setdefault(owner, clean_prose(text))

    # Use `replace`, not a hand-written rebuild of six fields. A hand-written copy
    # could silently drop a field when `Definition` gets a new one.
    return tuple(
        replace(one, doc=prose[one.name]) if one.name in prose else one for one in definitions
    )


def _comment_blocks(result: object) -> list[tuple[int, int, str]]:
    """Doc comments as `(first line, last line, text)`, with contiguous runs
    merged.

    Both line numbers are 1-based. Runs are merged because Rust and Go write a
    paragraph as a *run* of single-line comments, each its own node. If only the
    last node were attached, the index would hold the final sentence of every doc
    comment in those languages and drop the rest.

    A blank line ends a run, which is also where a reader would end it.
    """
    blocks: list[tuple[int, int, str]] = []
    for comment in getattr(result, "comments", None) or []:
        if str(getattr(comment, "kind", "")).rsplit(".", 1)[-1] not in ("Doc", "Line"):
            continue
        text = (comment.text or "").rstrip()
        if not text.strip():
            continue
        first = comment.span.start_line + 1
        last = first + text.count("\n")
        if blocks and blocks[-1][1] + 1 == first:
            was_first, _, before = blocks[-1]
            blocks[-1] = (was_first, last, f"{before}\n{text}")
        else:
            blocks.append((first, last, text))
    return blocks


MARKERS = ('"""', "'''", "///", "/**", "*/", "//", "#")
"""The comment and docstring delimiters, with the longest first within each family."""


def clean_prose(text: str) -> str:
    """Return the prose without its delimiters, which is what an FTS index should
    hold.

    A docstring that is stored with its quotes puts punctuation into every
    neighboring search term, and shows the model syntax that it did not ask for.
    The text is stripped line by line, so a block comment also loses its leading
    `*`.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        for marker in MARKERS:
            while line.startswith(marker):
                line = line[len(marker) :].strip()
            while line.endswith(marker):
                line = line[: -len(marker)].rstrip()
        if line.startswith("*"):
            line = line[1:].strip()
        lines.append(line)
    return "\n".join(lines).strip()


def _kind_of(raw: object) -> str:
    """`SymbolKind.Function` / `"function"` / `"Function"` → `function`.

    The `kind` of the pack is an enum whose `str()` is the variant name. The tags
    query gives a plain lowercase string. One spelling reaches the index, so
    `kind: "class"` in a tool argument means the same thing for a row from either
    pass.
    """
    text = getattr(raw, "name", None) or str(raw)
    return text.rsplit(".", 1)[-1].lower()


def owners(definitions: Sequence[Definition]) -> dict[int, Definition]:
    """Return `{line: the tightest definition that contains it}`, in one pass for
    each file.

    **The tightest definition, not the first.** A nested `def` inside a 40-line
    function is the caller that a reference belongs to. With the outer function,
    the calls of every closure would be attributed to whatever contains the
    closure. Take a `claim_key` that spans lines 40-75 with an inner `release`
    on lines 69-73, as `_registry.py` had when this was written. A call on line
    70 has two candidates and exactly one correct answer.

    The table is built one time for the file. The earlier version searched for
    each reference, which was O(definitions x references). That was fine for a
    module with 30 symbols. For a generated file with 1 500 definitions and
    3 000 references, which `max_bytes` admits, it took 58 ms, as much as the
    whole parse. The widest spans are written first, so a narrower span
    overwrites a wider one, and the last writer for each line is the tightest.
    """
    table: dict[int, Definition] = {}
    for one in sorted(definitions, key=lambda d: d.start_line - d.end_line):
        for line in range(one.start_line, one.end_line + 1):
            table[line] = one
    return table
