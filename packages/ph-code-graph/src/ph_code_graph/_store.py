"""The graph on disk: four tables, an FTS index, and one recursive query.

## Why the standard library and not `pyturso`

pH's session log runs on turso, so this is the odd one out and the reason is
measured. Against a real 40 MB CodeGraph database (11 396 nodes, 36 830 edges),
`pyturso` served indexed lookups, joins and aggregates correctly — and then:

* **FTS5 is absent.** A `CREATE VIRTUAL TABLE … USING fts5` is invisible to it,
  shadow tables included. That is `search`.
* **`Recursive CTEs are not yet supported`.** That is `impact`.
* **`json_extract` silently returns NULL** where SQLite returns the value —
  measured `0.95` against `None` on the same rows. A wrong answer with no error
  is worse than a missing feature, and it is what makes the first two look like
  gaps rather than the same class of problem.

Two of the five things this store does are exactly the two turso cannot do, so
it uses `sqlite3` from the standard library — SQLite 3.45, FTS5 compiled in, no
dependency to add. If turso grows both, this is one import.

## The schema, and what it deliberately is not

`files` · `symbols` · `refs` · `imports`, plus `symbols_fts`. A **name-based**
graph: a reference records the name it used, and `callers`/`callees` join on
that name. It is not a resolved graph — two `register` methods in two classes
are one name here — and every query that could be ambiguous reports how many
definitions the name has, so the caller can see it rather than being quietly
given one of them.

That is the honest ceiling of this approach, and the reason is worth stating: a
*resolved* graph is import-graph plus scope plus type inference per language, and
in the tool this package replaces that was 29 708 lines of it. Name-based
answers most of what an agent asks ("who calls this", "what does this touch")
and says so where it cannot.

## Incremental by content, not by clock

A file is re-extracted when its **sha256 changes**, never on an mtime — a
checkout, a rebase or a `touch` moves mtimes without moving content, and
re-indexing a repository because git changed some timestamps is the kind of cost
nobody attributes correctly. Re-running the indexer over an unchanged tree
therefore reads and hashes, and writes nothing.

@module ph_code_graph._store
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._extract import Extraction, owners

__all__ = ["EMPTY_STATS", "CodeGraphStore", "Hit", "IndexVersion", "SymbolRow", "digest_of"]

log = logging.getLogger("ph_code_graph.store")

SCHEMA_VERSION = 1
"""The baseline. Nothing has shipped, so there is no second shape in the world
and no migration to write — `SCHEMA` is simply what a version-1 index is.

Read and refused by `prepare()` even so, because that is the half that was
missing rather than a courtesy: `meta` was written on every open and read
nowhere, so it was state that could only ever be wrong. The first bump that
costs anybody a rebuild is the first one that will be believed."""

EMPTY_STATS: dict[str, Any] = {"files": 0, "symbols": 0, "refs": 0, "languages": []}
"""What an absent or schemaless index reports. One literal, two callers."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    language   TEXT NOT NULL,
    digest     TEXT NOT NULL,
    vcs_id     TEXT NOT NULL DEFAULT '',
    lines      INTEGER NOT NULL,
    indexed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    doc        TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path, start_line);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

CREATE TABLE IF NOT EXISTS refs (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    line        INTEGER NOT NULL,
    from_symbol INTEGER
);
CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS idx_refs_from ON refs(from_symbol);
CREATE INDEX IF NOT EXISTS idx_refs_path ON refs(path);

CREATE TABLE IF NOT EXISTS imports (
    path   TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imports_path ON imports(path);

-- `content=''` — an external-content table would have to be kept in step with
-- `symbols` by triggers, and this index is rebuilt wholesale for a file at a
-- time by the indexer anyway. Contentless is smaller and has one writer.
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name, doc, path, content=''
);
"""


class IndexVersion(RuntimeError):
    """The database was written by a schema this build does not read.

    Its own type because the only fix is a human decision — delete and rebuild
    — exactly as `ph_text_index._store.IndexMismatch` is for the same reason.
    """


def _version_of(connection: sqlite3.Connection) -> int | None:
    """The stamped schema version, or `None` for a database that has none yet."""
    try:
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
    except sqlite3.OperationalError:
        return None
    try:
        return int(row["value"]) if row is not None else None
    except (TypeError, ValueError):
        return None


def digest_of(text: str) -> str:
    """The content key. See the module docstring on incrementality."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SymbolRow:
    """One definition, as the model reads it."""

    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    doc: str | None = None

    @property
    def lines(self) -> int:
        """How many lines the definition spans.

        Derived rather than stored. It was a field fed by an `AS span` clause in
        six separate SELECTs, so a seventh query that forgot the alias failed at
        row-mapping time rather than at the query — and it was a third value
        that had to agree with two others.
        """
        return self.end_line - self.start_line + 1

    def as_value(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "doc": self.doc,
            "lines": self.lines,
        }


@dataclass(frozen=True, slots=True)
class Hit:
    """An edge: a symbol at the far end, and where the reference itself is.

    **`path` and `ref_path` are different files and both matter.** For `callers`
    they coincide — the calling symbol contains the call. For `callees` they do
    not: the symbol is the callee's *definition*, while the reference is a line
    in the caller. Rendering one against the other printed `target.py:941` for a
    line that lives in `context.py`, which is a pointer to nothing.
    """

    symbol: SymbolRow
    line: int
    """Where the reference is written — the line to open."""
    ref_path: str
    """Which file that line is in. See the class docstring."""
    via: str
    """The referenced name, which for a name-based graph is the edge's label."""

    def as_value(self) -> dict[str, Any]:
        return {
            **self.symbol.as_value(),
            "ref_line": self.line,
            "ref_path": self.ref_path,
            "via": self.via,
        }


@dataclass(slots=True)
class CodeGraphStore:
    """The index. **Blocking**; the seam calls it in a worker thread."""

    path: Path

    # ------------------------------------------------------------ lifecycle ----

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A connection with the pragmas this index wants, closed on the way out.

        Opened per call rather than held: `sqlite3` connections are not safe to
        share across threads, and every caller here arrives on whichever worker
        `to_thread` picked. WAL so a read during a write does not block, and
        `foreign_keys` off because the cascades are done in Python — a file's
        rows are deleted by path, which is one statement per table and clearer
        than a trigger nobody sees.

        **It creates nothing.** It used to take `write=True` and then replay
        `SCHEMA` plus a `meta` upsert on every call — and `put` is per file, so a
        20 000-file index ran 20 000 no-op schema scripts and 20 000 durable
        commits of a row nobody read. Measured at 1.8 ms per file, 24 % of all
        write time, and 70x on a synthetic run where WAL checkpointed per file.
        `prepare()` is the one writer of the schema, and the indexer already
        calls it once before the loop.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.row_factory = sqlite3.Row
            yield connection

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Connection]:
        """One connection with an actual transaction around it.

        **`with connection:` is not one here**, which is the trap this exists to
        close. `_open` asks for `isolation_level=None` so that a read costs no
        implicit transaction and `PRAGMA journal_mode` can take effect at all —
        and in autocommit the connection's own context manager has no
        transaction to commit, so each of a file's ~60 statements was its own
        durable commit. Measured over 137 real files: **8.19 ms per file as it
        was, 3.42 ms with this**, for the atomicity `put`'s docstring already
        claimed.

        `BEGIN IMMEDIATE` rather than a deferred begin: this only ever wraps
        writes, and taking the write lock up front turns a later
        mid-transaction contention into an honest wait at the start.

        Used by the single-statement writer too, which does not need it: what
        that one needed was to stop *reading* as though `, connection:` were a
        transaction, since that is the whole mistake.
        """
        with self._open() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def prepare(self) -> None:
        """Create the schema and stamp its version. Idempotent.

        :raises IndexVersion: when the file on disk was written by a schema this
            build does not read. Checked rather than assumed, which is the half
            that was missing: `meta` was written on every open and read nowhere,
            so it was state that could only ever be wrong. The sibling package's
            sidecar `format` is read and refused, and this is that.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode = WAL")
            found = _version_of(connection)
            if found is not None and found != SCHEMA_VERSION:
                raise IndexVersion(
                    f"{self.path} is schema {found} and this build reads {SCHEMA_VERSION}; "
                    "delete the file to rebuild"
                )
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema', ?)",
                (str(SCHEMA_VERSION),),
            )

    def exists(self) -> bool:
        return self.path.exists()

    # -------------------------------------------------------------- indexing ----

    def known(self) -> dict[str, tuple[str, str]]:
        """`{path: (digest, vcs_id)}` for everything already indexed.

        The whole map in one read, because the indexer's next act is to compare
        every candidate against it — a `SELECT` per file would be one statement
        per file to answer a question one statement answers.

        Both ids, because the indexer asks two questions per file and they have
        different answers: `vcs_id` decides whether the file need be *opened*,
        and `digest` decides whether what was opened is different.
        """
        if not self.exists():
            return {}
        with self._open() as connection:
            try:
                rows = connection.execute("SELECT path, digest, vcs_id FROM files").fetchall()
            except sqlite3.OperationalError:
                # No schema yet: an empty index and an absent one are the same
                # thing to a caller, and the indexer is about to create it.
                return {}
        return {row["path"]: (row["digest"], row["vcs_id"]) for row in rows}

    def token(self) -> str:
        """The version-control token stored at the last index. `""` if none.

        jj's working-copy id, which its one-call diff needs as a starting point.
        Per index rather than per file, because it describes the *tree*."""
        if not self.exists():
            return ""
        with self._open() as connection:
            try:
                row = connection.execute("SELECT value FROM meta WHERE key = 'vcs'").fetchone()
            except sqlite3.OperationalError:
                return ""
        return str(row["value"]) if row is not None else ""

    def remember(self, token: str) -> None:
        """Store the token for the next run to diff from."""
        with self._writing() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('vcs', ?)", (token,)
            )

    def forget(self, paths: Sequence[str]) -> int:
        """Drop every row belonging to `paths`. Returns how many files went."""
        if not paths:
            return 0
        with self._writing() as connection:
            gone = self._forget(connection, paths)
        return gone

    def _forget(self, connection: sqlite3.Connection, paths: Sequence[str]) -> int:
        gone = 0
        for path in paths:
            # The FTS rows first, by the id they were inserted under: a
            # contentless table cannot be asked which rows belong to a path, so
            # the ids come from `symbols` while it still has them.
            ids = [
                row["id"]
                for row in connection.execute("SELECT id FROM symbols WHERE path = ?", (path,))
            ]
            connection.executemany(
                "INSERT INTO symbols_fts(symbols_fts, rowid, name, doc, path) "
                "VALUES ('delete', ?, '', '', '')",
                [(one,) for one in ids],
            )
            connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
            connection.execute("DELETE FROM refs WHERE path = ?", (path,))
            connection.execute("DELETE FROM imports WHERE path = ?", (path,))
            gone += connection.execute("DELETE FROM files WHERE path = ?", (path,)).rowcount
        return gone

    def put(self, path: str, digest: str, extraction: Extraction, vcs_id: str = "") -> int:
        """Replace one file's rows with `extraction`. Returns symbols written.

        Replace, not merge: a file is the unit of extraction, so the old rows are
        deleted and the new ones written in one transaction. A crash mid-write
        therefore leaves the file absent from `files`, which the indexer treats
        as "not indexed" and does again — rather than leaving half its symbols
        behind, which nothing would ever notice.
        """
        with self._writing() as connection:
            self._forget(connection, [path])
            connection.execute(
                "INSERT INTO files(path, language, digest, vcs_id, lines, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    path,
                    extraction.language,
                    digest,
                    vcs_id,
                    extraction.lines,
                    int(time.time()),
                ),
            )
            by_line: dict[int, int] = {}
            for definition in extraction.definitions:
                cursor = connection.execute(
                    "INSERT INTO symbols(path, name, kind, start_line, end_line, doc) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        path,
                        definition.name,
                        definition.kind,
                        definition.start_line,
                        definition.end_line,
                        definition.doc,
                    ),
                )
                identifier = int(cursor.lastrowid or 0)
                by_line[definition.start_line] = identifier
                connection.execute(
                    "INSERT INTO symbols_fts(rowid, name, doc, path) VALUES (?, ?, ?, ?)",
                    (identifier, definition.name, definition.doc or "", path),
                )
            # One `{line: tightest definition}` table for the whole file, rather
            # than a scan of every definition per reference: that was
            # O(definitions x references), which a generated file with 1 500
            # definitions and 3 000 references turned into 58 ms — as much as its
            # entire parse.
            #
            # Resolved here rather than by a later pass either way: the
            # containing definition is known while the file's own definitions are
            # in hand, and a join that recomputed containment per query would be
            # a range scan over every symbol in the file.
            owner = owners(extraction.definitions)
            connection.executemany(
                "INSERT INTO refs(path, name, kind, line, from_symbol) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        path,
                        reference.name,
                        reference.kind,
                        reference.line,
                        by_line.get(found.start_line) if found is not None else None,
                    )
                    for reference in extraction.references
                    for found in (owner.get(reference.line),)
                ],
            )
            connection.executemany(
                "INSERT INTO imports(path, source) VALUES (?, ?)",
                [(path, one) for one in extraction.imports],
            )
        return len(extraction.definitions)

    # --------------------------------------------------------------- queries ----

    def search(self, query: str, limit: int) -> list[SymbolRow]:
        """Names and docstrings, by relevance. FTS5, which is why not turso."""
        with self._open() as connection:
            rows = connection.execute(
                """SELECT s.*
                   FROM symbols_fts f JOIN symbols s ON s.id = f.rowid
                   WHERE symbols_fts MATCH ?
                   ORDER BY bm25(symbols_fts, 10.0, 1.0, 0.0) LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [_row(one) for one in rows]

    def define(self, name: str, limit: int) -> list[SymbolRow]:
        """Every definition of an exact name."""
        with self._open() as connection:
            rows = connection.execute(
                """SELECT * FROM symbols
                   WHERE name = ? ORDER BY path, start_line LIMIT ?""",
                (name, limit),
            ).fetchall()
        return [_row(one) for one in rows]

    def callers(self, name: str, limit: int) -> list[Hit]:
        """Symbols containing a reference to `name`, and the line of each."""
        with self._open() as connection:
            rows = connection.execute(
                """SELECT s.*,
                          r.line AS ref_line, r.path AS ref_path
                   FROM refs r JOIN symbols s ON s.id = r.from_symbol
                   WHERE r.name = ? ORDER BY s.path, r.line LIMIT ?""",
                (name, limit),
            ).fetchall()
        return [
            Hit(symbol=_row(one), line=one["ref_line"], ref_path=one["ref_path"], via=name)
            for one in rows
        ]

    def callees(self, name: str, limit: int) -> list[Hit]:
        """Definitions of the names referenced from inside `name`."""
        with self._open() as connection:
            rows = connection.execute(
                """SELECT t.*,
                          r.line AS ref_line, r.path AS ref_path, r.name AS via
                   FROM symbols s
                   JOIN refs r ON r.from_symbol = s.id
                   JOIN symbols t ON t.name = r.name
                   WHERE s.name = ? ORDER BY r.line, t.path LIMIT ?""",
                (name, limit),
            ).fetchall()
        return [
            Hit(
                symbol=_row(one),
                line=one["ref_line"],
                ref_path=one["ref_path"],
                via=one["via"],
            )
            for one in rows
        ]

    def impact(self, name: str, distance: int, limit: int) -> list[tuple[int, SymbolRow]]:
        """Transitive callers, `1..distance` hops out, with the hop each is at.

        **The recursive CTE that turso cannot run.** One statement rather than a
        BFS of N queries per ring — the difference is not style: a Python loop
        would issue a query per frontier symbol, and this is the query an agent
        asks when it wants to know what a change breaks, over a graph where the
        interesting names have hundreds of callers.

        `MIN(depth)` because a symbol reachable at two distances is at the
        nearer one, which is what makes the rings a budget rather than a
        multiset — the same rule the old row's hand-written walk had.
        """
        with self._open() as connection:
            rows = connection.execute(
                """WITH RECURSIVE reach(id, depth) AS (
                       SELECT id, 0 FROM symbols WHERE name = ?
                     UNION
                       SELECT s.id, r.depth + 1
                       FROM reach r
                       JOIN symbols origin ON origin.id = r.id
                       JOIN refs f ON f.name = origin.name
                       JOIN symbols s ON s.id = f.from_symbol
                       WHERE r.depth < ?
                   )
                   SELECT MIN(depth) AS depth, s.*
                   FROM reach JOIN symbols s ON s.id = reach.id
                   WHERE depth > 0
                   GROUP BY s.id ORDER BY depth, s.path, s.start_line LIMIT ?""",
                (name, distance, limit),
            ).fetchall()
        return [(one["depth"], _row(one)) for one in rows]

    def entities(
        self, prefix: str | None, kind: str | None, limit: int, offset: int
    ) -> tuple[list[SymbolRow], int]:
        """The biggest definitions first — "what is worth opening" in one list."""
        where = ["1 = 1"]
        args: list[Any] = []
        if prefix:
            root = prefix.rstrip("/")
            where.append("(path = ? OR path LIKE ? || '/%')")
            args.extend([root, root])
        if kind and kind != "any":
            where.append("kind = ?")
            args.append(kind)
        clause = " AND ".join(where)
        with self._open() as connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) FROM symbols WHERE {clause}", args).fetchone()[
                    0
                ]
            )
            rows = connection.execute(
                f"""SELECT * FROM symbols
                    WHERE {clause}
                    ORDER BY (end_line - start_line + 1) DESC, path, start_line
                    LIMIT ? OFFSET ?""",
                [*args, limit, offset],
            ).fetchall()
        return [_row(one) for one in rows], total

    def file_count(self) -> int:
        """How many files are indexed. **The only number a query header needs.**

        Its own method because `stats()` also counts `symbols` and `refs` and
        groups `files` by language — three full scans a query was paying per
        call to fill one field. Measured on a 2 000-file index: 1.97 ms against
        0.22 ms, and it grows with the corpus without bound.
        """
        if not self.exists():
            return 0
        with self._open() as connection:
            try:
                return int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            except sqlite3.OperationalError:
                return 0

    def definition_count(self, name: str) -> int:
        """How many places define `name` — the ambiguity a name-based graph owes.

        `COUNT(*)` rather than `len(define(name, 100))`, which materialised up to
        a hundred rows and then reported the *cap* as the count for anything
        past it.
        """
        with self._open() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM symbols WHERE name = ?", (name,)
                ).fetchone()[0]
            )

    def stats(self) -> dict[str, Any]:
        """What `ph doctor` and every result's header report."""
        if not self.exists():
            return dict(EMPTY_STATS)
        with self._open() as connection:
            try:
                files, symbols, refs = connection.execute(
                    "SELECT (SELECT COUNT(*) FROM files), (SELECT COUNT(*) FROM symbols), "
                    "(SELECT COUNT(*) FROM refs)"
                ).fetchone()
                languages = [
                    row["language"]
                    for row in connection.execute(
                        "SELECT language FROM files GROUP BY language ORDER BY COUNT(*) DESC"
                    )
                ]
            except sqlite3.OperationalError:
                return dict(EMPTY_STATS)
        return {"files": files, "symbols": symbols, "refs": refs, "languages": languages}


def _row(row: sqlite3.Row) -> SymbolRow:
    return SymbolRow(
        name=row["name"],
        kind=row["kind"],
        path=row["path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        doc=row["doc"],
    )
