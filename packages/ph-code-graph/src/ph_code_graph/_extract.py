"""One file in, definitions and references out — with the lines they are on.

Two passes over `tree-sitter-language-pack`, because each answers something the
other cannot and both are cheap:

* **`process()`** — the pack's own intelligence layer. Definitions with kinds,
  the file's imports, and its docstrings and comments. What it does not give is
  *references*: `SymbolInfo` says a function exists, never who calls it. It also
  does not give a symbol its own prose — `SymbolInfo.doc` is always `None`; see
  `_documented`.
* **the tags query** — tree-sitter's own `tags.scm` for the language, the same
  data GitHub's code navigation is built on. `@definition.*` and, crucially,
  `@reference.call`. That is the edge this package exists to record.

Measured over `packages/ph-core/src/ph`: 136 files and 1.58 MiB through
`process()` in 219 ms and through the tags query in 108 ms, so both passes
together index this repository's core in about a third of a second. Parsing
twice is worth more than the parse it saves.

## The two coordinate systems, reconciled here

`ProcessConfig` spans are **0-based**; tree-sitter points are 0-based too, and
every line number pH puts in front of a model is **1-based**, because that is
what `read` takes and what an editor shows. Both are normalised on the way out
of this module, once, so nothing downstream has to remember which source a
number came from. Getting this wrong is a one-line-off pointer, which is the
kind of error a model stops trusting the tool over rather than reporting.

## Query inheritance

`tree-sitter-typescript`'s grammar extends javascript's, and upstream splits the
tags queries the same way: the typescript query covers only what typescript adds
(interfaces, signatures, abstract classes), so `class`, `function` and `call`
live in the *javascript* query. A `.ts` file matched against the typescript
query alone yields two tags and no calls — measured. `INHERITS` is that fact.

@module ph_code_graph._extract
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

__all__ = [
    "INHERITS",
    "Definition",
    "Extraction",
    "Reference",
    "cache_release",
    "detect_language",
    "ensure",
    "extract",
    "indexable",
    "local",
    "owners",
    "parseable",
    "readiness",
    "use_cache",
]

log = logging.getLogger("ph_code_graph.extract")

_configured: Path | None = None
"""The cache base this process last handed the pack, or `None` for its default.

Mirrored here because the pack does not report it: `cache_dir()` answers with the
resolved leaf it derives, which cannot be fed back in. Module state because the
thing it mirrors is a module-level global in somebody else's library — recording
it anywhere narrower would be a second answer to a process-wide question.
"""

INHERITS: dict[str, tuple[str, ...]] = {
    "typescript": ("javascript",),
    "tsx": ("javascript",),
}
"""Languages whose tags query is only half the story. See the module docstring."""

REFERENCE_KINDS = ("call",)
"""Which `@reference.*` captures become edges.

`call` only, deliberately. The tags queries also emit `reference.type`,
`reference.class`, `reference.implementation` and more, and each is a real
relationship — but they answer a different question than "what runs what", and
mixing them into one edge table would make `callers` return the places that
merely *mention* a type. They are dropped rather than stored-and-filtered so the
index does not carry rows nothing reads.
"""


@dataclass(frozen=True, slots=True)
class Definition:
    """A symbol this file defines."""

    name: str
    kind: str
    """`function`, `class`, `method`, `interface`, `module`, `constant` — whatever
    the language's own tags query calls it, with the `definition.` prefix off."""
    start_line: int
    """1-based, inclusive."""
    end_line: int
    """1-based, inclusive."""
    doc: str | None = None


@dataclass(frozen=True, slots=True)
class Reference:
    """A name this file uses, and where."""

    name: str
    kind: str
    line: int
    """1-based."""


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything one file yielded."""

    language: str
    definitions: tuple[Definition, ...]
    references: tuple[Reference, ...]
    imports: tuple[str, ...]
    lines: int


def _pack() -> Any:
    import tree_sitter_language_pack

    return tree_sitter_language_pack


def cache_release(directory: Path) -> Callable[[], None]:
    """Point the pack's grammar cache at `directory`; return the restore.

    **An `enter` returning its own release**, which is the shape `ctx.effect`
    takes — so the mutation unwinds with the row that made it (§4.9, I2) instead
    of outliving it. `pack.configure` is process-global and last-writer-wins, so
    without this the library kept pointing at an unmounted row's `$PH_CACHE`:
    visible in this suite, where that path is a per-test `tmp_path` the pack went
    on holding after teardown.

    Harmless in production today — every mount in a process computes the same
    path — and wrong depth all the same, which is the whole argument for the
    rule: a global taken without a release is one nobody notices until two
    deployments in one process disagree.
    """
    # **The base we set, not `cache_dir()`.** That reports the *resolved* leaf —
    # `<base>/tree-sitter-language-pack/<version>/libs` — so feeding it back to
    # `PackConfig(cache_dir=...)` makes the pack append its own suffix a second
    # time and the restore nests instead of restoring. Caught by the test for
    # this function, which is the argument for having written one.
    previous = _configured
    use_cache(directory)

    def restore() -> None:
        # Best-effort and logged rather than raised: a disposer that throws is
        # logged and the unwind continues anyway, so failing loudly here would
        # only obscure whatever else the scope was releasing.
        try:
            if previous is None:
                _reset()
            else:
                use_cache(previous)
        except Exception:  # pragma: no cover - a library that cannot be restored
            log.warning("ph_code_graph: could not restore the grammar cache", exc_info=True)

    return restore


def use_cache(directory: Path) -> Path:
    """Point the pack's grammar cache at `directory`. Returns where it landed.

    **Not optional, and not only about the download tail.** The 26 bundled
    languages are shipped inside the wheel as an archive and *materialised into
    this directory on first use* — so a deployment whose cache directory is not
    writable does not fall back to the wheel, it fails:

        Download error: Failed to create cache directory … Permission denied

    The default is `$XDG_CACHE_HOME/tree-sitter-language-pack/<version>/libs`,
    which is fine on a developer's laptop and wrong in a container with a
    read-only `HOME` — measured both ways. So the row hands over `$PH_CACHE`
    instead: the root pH already designates for rebuildable artifacts, on a path
    it has created and verified.

    Through the pack's own `configure()` rather than by setting
    `TREE_SITTER_LANGUAGE_PACK_CACHE_DIR`: a row that mutated the process
    environment would change it for every child `ctx.subprocess` spawns too, and
    this is a decision about *this* library.

    An operator who set the environment variable deliberately keeps it — their
    spelling wins, which is what makes the variable still mean something.
    """
    import os

    global _configured

    override = os.environ.get("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR")
    if override:
        return Path(override)
    directory.mkdir(parents=True, exist_ok=True)
    pack = _pack()
    pack.configure(pack.PackConfig(cache_dir=str(directory)))
    _configured = directory
    return directory


def _reset() -> None:
    """Hand the pack back its own default. The `previous is None` restore."""
    global _configured

    pack = _pack()
    pack.configure()
    _configured = None


def detect_language(path: str) -> str | None:
    """The pack's own name for this path's language, or `None` for "not code".

    `None` is the ordinary answer for most of a repository — a `.md`, a lockfile,
    an image — so it is a value and not an error. Markdown and the other prose
    languages the pack recognises are excluded by the caller, which knows it
    wants code; this only reports what the extension says.
    """
    try:
        found: str | None = _pack().detect_language_from_path(path)
    except Exception:
        return None
    return found


@cache
def indexable(language: str) -> bool:
    """Whether this language has a tags query, and so can yield an edge at all.

    **Derived, not listed.** This replaced a hand-written `PROSE` set of nine
    names, which was a copy of a property the pack reports directly — and an
    incomplete one, because the detector claims 371 languages: `.txt`, `.ini`,
    `.proto`, `.rst`, `.conf` and a dozen others also have no tags query, and
    with the row's default `glob: "**/*"` every one of them reached the
    extractor and came back as an internal `TypeError` dressed up as a per-file
    skip reason.

    Asking the pack cannot drift and covers all 371. It also costs nothing: the
    answer is cached, and `get_tags_query` reads a string the pack already holds
    rather than materialising a grammar.
    """
    try:
        return bool(_pack().get_tags_query(language))
    except Exception:
        return False


def local(language: str) -> bool:
    """Whether this language's grammar is **already on disk**.

    A cache listing, so it reads the disk and fetches nothing — which is what
    makes it the right question for `/code-graph status`: a person asking "is
    this ready" must not be answered by an action that makes it ready.

    **Deliberately not the indexer's question.** The pack unpacks its 26 bundled
    grammars from the wheel on first use, with no network, so gating indexing on
    this would skip `.py` files on a fresh cache and tell the caller to run a
    command that had nothing to fetch. The indexer asks `parseable`, which
    admits that unpack; the residue — a language outside the bundle, which does
    reach GitHub — is what `/code-graph install` is for, and what the skip
    reason names.
    """
    try:
        return language in set(_pack().downloaded_languages())
    except Exception:
        return False


def parseable(language: str) -> bool:
    """Whether a parser for `language` can be obtained at all.

    Materialising, and that is the point: `get_parser` unpacks a bundled grammar
    from the wheel — local, and the ordinary case — and only reaches the network
    for the ~345 outside it. The docstring this replaced claimed to be "the
    bundled question" and was not, which mattered because it was also the skip
    *reason*: a network-denied deployment reported every file as an unsupported
    language rather than an unprovisioned cache.

    So the wording is now about the parser rather than about bundling, and
    `local` answers the readiness question separately.
    """
    if not indexable(language):
        return False
    try:
        _pack().get_parser(language)
    except Exception:
        return False
    return True


def readiness(languages: Sequence[str]) -> tuple[list[str], list[str]]:
    """Which of `languages` are on disk, and which are not. One pass.

    Here rather than beside its caller because there were two of these: the
    provisioning command partitioned the list one way and `ensure` re-derived
    the same partition with a `one not in ready` scan three functions along —
    the very O(n²) rederivation the other one's docstring claimed to have
    replaced. One function, so the two cannot disagree about which list the
    unavailable side is drawn from.
    """
    ready = [one for one in languages if local(one)]
    have = set(ready)
    return ready, [one for one in languages if one not in have]


def ensure(languages: Sequence[str]) -> tuple[list[str], list[str]]:
    """Fetch what is missing. `(ready, unavailable)` — the provisioning door.

    The pack's own `download()` rather than `get_parser`-for-its-side-effect, so
    the fetch lives in exactly one place: the command a person runs.
    """
    wanted = [one for one in languages if indexable(one)]
    try:
        _pack().download(list(wanted))
    except Exception:
        log.warning("ph_code_graph: grammar download failed", exc_info=True)
    ready, _ = readiness(wanted)
    return ready, [one for one in languages if one not in set(ready)]


@cache
def _tags_query(language: str) -> Any:
    """The compiled tags query for one language. **Cached, and measurably so.**

    Compiling is per-language work that was being paid per *file*: 2.8 ms for
    python, 22 ms for ruby, and 35.7 ms for typescript — the worst case, because
    `INHERITS` concatenates two queries. Over this repo's `ph-core` tree (136
    files) that was 44 % of all extraction time; a thousand-file TypeScript
    package paid 35 s of it.

    `functools.cache` is the `ph.seams.fs._compiled` precedent, and it is safe
    for the same reason: a `Query` is read-only once built, and `extract` makes
    a fresh `QueryCursor` per call — which is the mutable half.
    """
    from tree_sitter import Query

    pack = _pack()
    sources = [pack.get_tags_query(one) for one in INHERITS.get(language, ())]
    sources.append(pack.get_tags_query(language))
    return Query(pack.get_language(language), "\n".join(one for one in sources if one))


def extract(path: str, text: str, language: str) -> Extraction:
    """Parse `text` and return what it defines, what it references, what it imports.

    :raises Exception: whatever the pack raises for an unparseable file. Not
        caught here: the caller indexes a *batch*, and whether one bad file
        should stop the batch or be reported and skipped is its decision, not
        this function's.
    """
    from tree_sitter import QueryCursor

    pack = _pack()
    config = pack.ProcessConfig(
        language=language, symbols=True, imports=True, docstrings=True, comments=True
    )
    result = pack.process(text, config)

    # +1: `Span.start_line` is 0-based and everything downstream is 1-based.
    # See the module docstring.
    definitions = [
        Definition(
            name=symbol.name,
            kind=_kind_of(symbol.kind),
            start_line=symbol.span.start_line + 1,
            end_line=symbol.span.end_line + 1,
            # **Not `symbol.doc`**, which this pack never populates — measured
            # `None` for every symbol in a file full of docstrings. Prose
            # arrives on two other channels and `_documented` merges them.
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
        # `Node.text` is `bytes | None` — `None` when the tree outlived the
        # source it was parsed from, which cannot happen here but is the
        # signature's promise, so it is handled rather than asserted away.
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
                # The tags query finds definitions too, and mostly the same ones
                # `process()` did. Kept only where it found one `process()`
                # missed — a language whose intel layer is thinner than its
                # tags query — and matched on (name, line) so the union has no
                # duplicates to deduplicate later.
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


def _documented(definitions: list[Definition], result: Any) -> tuple[Definition, ...]:
    """Attach each definition's prose, from the two channels that carry it.

    **`SymbolInfo.doc` is always `None`** in this pack — measured against a file
    full of docstrings — so a row that trusted it would have shipped an index
    with an empty `doc` column and a `search` mode that matched names only. The
    prose is there, on two other channels:

    * **`result.docstrings`** carries `associated_item`, the pack's own answer to
      "which symbol is this the docstring of". Used verbatim where present,
      because the pack knows each language's convention better than a heuristic
      here would.
    * **`result.comments`** carries doc comments — Rust's `///` and
      TypeScript's block form (kind `Doc`), and Go's `//` (kind `Line`, because
      that *is* Go's documented convention). `associated_item` is not set for
      these, so the rule is positional and deliberately narrow: the comment's
      last line must be **immediately above** a definition's first. A gap of one
      blank line means it was about something else, and attaching it anyway
      would put the wrong prose in front of the model.

    Anything owned by nobody — a module docstring, a comment inside a function
    body — is dropped rather than attached to whatever happened to be nearest.
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
        # **The comment's own text, not its `span.end_line`.** Rust's `///`
        # comment node includes the trailing newline, so its `end_line` is one
        # past the line it occupies — which made the immediately-above rule miss
        # the definition it should have matched *and* match one a blank line
        # away. Counting the lines the stripped text actually spans is exact for
        # every language here, and was wrong for none.
        owner = starts.get(last + 1)
        if owner and text:
            prose.setdefault(owner, clean_prose(text))

    # `replace` rather than a hand-written six-field rebuild: the copy could
    # silently drop a field the day `Definition` grows one.
    return tuple(
        replace(one, doc=prose[one.name]) if one.name in prose else one for one in definitions
    )


def _comment_blocks(result: Any) -> list[tuple[int, int, str]]:
    """Doc comments as `(first line, last line, text)`, contiguous runs merged.

    Both 1-based. Merged because Rust and Go write a paragraph as a *run* of
    single-line comments — each its own node — so attaching only the last one
    would index the final sentence of every doc comment in those languages and
    drop the rest.

    A run breaks on a blank line, which is also how a reader would break it.
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
"""Comment and docstring delimiters, longest-first within each family."""


def clean_prose(text: str) -> str:
    """The prose without its delimiters — what an FTS index should hold.

    A docstring stored with its quotes still attached puts punctuation into
    every neighbouring search term and shows the model syntax it did not ask
    for. Stripped line by line, so a block comment loses its leading `*` too.
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


def _kind_of(raw: Any) -> str:
    """`SymbolKind.Function` / `"function"` / `"Function"` → `function`.

    The pack's `kind` is an enum whose `str()` is the variant name, while the
    tags query hands over a plain lowercase string. One spelling reaches the
    index, so `kind: "class"` in a tool argument means the same thing whichever
    pass produced the row.
    """
    text = getattr(raw, "name", None) or str(raw)
    return text.rsplit(".", 1)[-1].lower()


def owners(definitions: Sequence[Definition]) -> dict[int, Definition]:
    """`{line: the tightest definition containing it}`, in one pass per file.

    **Tightest, not first**: a nested `def` inside a 40-line function is the
    caller a reference belongs to, and taking the outer one would attribute
    every closure's calls to whatever happened to contain it. `_registry.py`'s
    `claim_key` spans 40-75 and its inner `release` spans 69-73, so a call on
    line 70 has two candidates and exactly one right answer.

    Built once for the file rather than searched per reference, which is what
    this replaced: that was O(definitions x references), fine for a 30-symbol
    module and 58 ms — as much as the whole parse — for a generated file with
    1 500 definitions and 3 000 references, which `max_bytes` happily admits.
    Widest first, so a narrower span overwrites it and the last writer per line
    is the tightest.
    """
    table: dict[int, Definition] = {}
    for one in sorted(definitions, key=lambda d: d.start_line - d.end_line):
        for line in range(one.start_line, one.end_line + 1):
            table[line] = one
    return table
