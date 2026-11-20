"""The graph on disk: four tables, a small `meta`, an FTS index, and one recursive query.

## Why the standard library and not `pyturso`

The session log of pH runs on turso, so this module is the exception, and the
reason is measured. Against a real 40 MB CodeGraph database (11 396 nodes,
36 830 edges), `pyturso` served indexed lookups, joins, and aggregates correctly.
It then failed in three ways:

* **FTS5 is absent.** `pyturso` does not see a `CREATE VIRTUAL TABLE … USING
  fts5`, or its shadow tables. `search` needs FTS5.
* **`Recursive CTEs are not yet supported`.** `impact` needs a recursive CTE.
* **`json_extract` silently returns NULL** where SQLite returns the value:
  measured `0.95` against `None` on the same rows. A wrong answer with no error
  is worse than a missing feature. It also shows that the first two are not only
  gaps but the same class of problem.

Two of the six queries that this store answers are exactly the two that turso
cannot run. The store therefore uses `sqlite3` from the standard library: SQLite
3.45, with FTS5 compiled in, and no dependency to add. If turso gets both
features, the change here is one import.

## The schema, and what it deliberately is not

The tables are `files`, `symbols`, `refs`, and `imports`, plus `symbols_fts`.
`meta` holds the two facts about the store itself: its schema version, and the
version-control token at which it was last indexed. This is a **name-based**
graph. A reference records the name that it used, and `callers` and `callees`
join on that name. It is not a resolved graph: two `register` methods in two
classes are one name here. Every query that could be ambiguous reports how many
definitions the name has. The caller can then see the ambiguity, and is not
silently given one of the definitions.

That is the real limit of this approach, and the reason is worth stating. A
*resolved* graph needs an import graph, scope, and type inference for each
language. In the tool that this package replaces, that was 29 708 lines. A
name-based graph answers most of what an agent asks ("who calls this", "what does
this touch") and says so where it cannot.

## Incremental by content, not by clock

A file is extracted again when its **sha256 changes**, never because of an mtime.
A checkout, a rebase, or a `touch` changes mtimes without changing content.
Nobody attributes the cost correctly when a repository is indexed again because
git changed some timestamps. A second run of the indexer over an unchanged tree
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
"""The baseline. Nothing has shipped, so only one shape exists and there is no
migration to write. `SCHEMA` is simply what a version-1 index is.

`prepare()` still reads the version and refuses a mismatch. That check was the
missing half, not a courtesy. `meta` was written on every open and read nowhere,
so it was state that could only be wrong. The first version bump that costs
someone a rebuild is the first one that will be believed."""

EMPTY_STATS: dict[str, Any] = {"files": 0, "symbols": 0, "refs": 0, "languages": []}
"""What an absent index, or an index with no schema, reports. One literal for
two callers."""

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
    """A schema that this build does not read wrote the database.

    This has its own type because the only fix is a human decision: delete the
    file and rebuild. `ph_text_index._store.IndexMismatch` exists for the same
    reason.
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
        """The number of lines that the definition spans.

        This is derived, not stored. It was a field that an `AS span` clause
        filled in six separate SELECTs. A seventh query that forgot the alias
        would fail when a row was mapped, not at the query. It was also a third
        value that had to agree with two others.
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
    """An edge: a symbol at the far end, and the place of the reference itself.

    **`path` and `ref_path` are different files, and both matter.** For
    `callers` they are the same file, because the calling symbol contains the
    call. For `callees` they are not: the symbol is the *definition* of the
    callee, and the reference is a line in the caller. When one was rendered
    against the other, the output was `target.py:941` for a line that is in
    `context.py`, which points to nothing.
    """

    symbol: SymbolRow
    line: int
    """The line where the reference is written. This is the line to open."""
    ref_path: str
    """The file that holds that line. See the class docstring."""
    via: str
    """The referenced name. In a name-based graph it is the label of the edge."""

    def as_value(self) -> dict[str, Any]:
        return {
            **self.symbol.as_value(),
            "ref_line": self.line,
            "ref_path": self.ref_path,
            "via": self.via,
        }


def _fts_values(rowid: int, name: str, doc: str | None, path: str) -> tuple[int, str, str, str]:
    """One `symbols_fts` row, for the insert **and** for the delete that undoes it.

    The `'delete'` command of FTS5 derives the postings to remove from the values
    that it is given. A delete whose tuple differs from the tuple of the insert,
    even by one NULL, removes nothing, and SQLite reports no error. `_forget`
    used to carry a copy of this expression, with a comment that said it mirrored
    the insert. That asked a reader to check two expressions in two methods, when
    one function makes the agreement structural.

    The essential part is `doc or ""`: `symbols.doc` is nullable, and the FTS
    column is not.
    """
    return (rowid, name, doc or "", path)


@dataclass(slots=True)
class CodeGraphStore:
    """The index. Its methods **block**, so the seam calls them in a worker thread."""

    path: Path

    # ------------------------------------------------------------ lifecycle ----

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A connection with the pragmas that this index needs, closed on exit.

        A connection is opened for each call and not held. `sqlite3` connections
        are not safe to share across threads, and each caller here arrives on
        whichever worker `to_thread` chose. WAL is on, so that a read during a
        write does not block. `foreign_keys` is off because the cascades are done
        in Python: the rows of a file are deleted by path, with one statement for
        each table, which is clearer than a trigger that nobody sees.

        **This method creates nothing.** It used to take `write=True`, and then
        ran `SCHEMA` and a `meta` upsert on every call. `put` runs for each file,
        so an index of 20 000 files ran 20 000 schema scripts that did nothing,
        and 20 000 durable commits of a row that nobody read. That measured
        1.8 ms per file, 24 % of all write time, and 70x on a synthetic run in
        which WAL checkpointed for each file. `prepare()` is the one writer of
        the schema, and the indexer already calls it one time before the loop.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.row_factory = sqlite3.Row
            yield connection

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Connection]:
        """One connection, with a real transaction around it.

        **`with connection:` is not a transaction here.** This method exists to
        close that trap. `_open` asks for `isolation_level=None`, so that a read
        costs no implicit transaction and `PRAGMA journal_mode` can take effect.
        In autocommit, the context manager of the connection has no transaction
        to commit. Each of the roughly 60 statements for a file was therefore its
        own durable commit. Measured over 137 real files: **8.19 ms per file
        before, 3.42 ms with this method**. It also gives the atomicity that the
        docstring of `put` already claimed.

        The transaction starts with `BEGIN IMMEDIATE`, not a deferred begin. This
        method only wraps writes. Taking the write lock at the start turns
        contention in the middle of a transaction into a plain wait at the start.

        The single-statement writer also uses this method, although it does not
        need a transaction. Its code must not *read* as if `, connection:` were a
        transaction, because that reading is the whole mistake.
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
        """Create the schema and stamp its version. Safe to call again.

        :raises IndexVersion: when a schema that this build does not read wrote
            the file on disk. The version is checked, not assumed. That check was
            the missing half: `meta` was written on every open and read nowhere,
            so it was state that could only be wrong. The sibling package reads
            the `format` of its sidecar and refuses a mismatch, and this is the
            same check.
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
        """`{path: (digest, vcs_id)}` for every file that is already indexed.

        The whole map comes from one read, because the indexer next compares
        every candidate with it. A `SELECT` for each file would use one statement
        per file for a question that one statement answers.

        Both ids are returned, because the indexer asks two questions for each
        file, and they have different answers. `vcs_id` decides whether the file
        must be *opened*. `digest` decides whether the opened content is
        different.
        """
        if not self.exists():
            return {}
        with self._open() as connection:
            try:
                rows = connection.execute("SELECT path, digest, vcs_id FROM files").fetchall()
            except sqlite3.OperationalError:
                # No schema yet. To a caller, an empty index and an absent index
                # are the same, and the indexer is about to create the schema.
                return {}
        return {row["path"]: (row["digest"], row["vcs_id"]) for row in rows}

    def token(self) -> str:
        """The version-control token that was stored at the last index, or `""` if
        there is none.

        This is the working-copy id of jj, which its one-call diff needs as a
        starting point. It is kept for the index and not for each file, because
        it describes the *tree*."""
        if not self.exists():
            return ""
        with self._open() as connection:
            try:
                row = connection.execute("SELECT value FROM meta WHERE key = 'vcs'").fetchone()
            except sqlite3.OperationalError:
                return ""
        return str(row["value"]) if row is not None else ""

    def remember(self, token: str) -> None:
        """Store the token, so that the next run can diff from it."""
        with self._writing() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('vcs', ?)", (token,)
            )

    def forget(self, paths: Sequence[str]) -> int:
        """Delete every row that belongs to `paths`. Returns the number of files
        removed."""
        if not paths:
            return 0
        with self._writing() as connection:
            gone = self._forget(connection, paths)
        return gone

    def _forget(self, connection: sqlite3.Connection, paths: Sequence[str]) -> int:
        gone = 0
        for path in paths:
            # Delete the FTS rows first, **with the values that they were inserted
            # with**. A contentless table cannot report the rows that belong to a
            # path, so the values come from `symbols` while it still has them
            # (X1).
            #
            # The delete passes the columns and not only the rowid, because the
            # `'delete'` command of FTS5 needs them. It derives the postings to
            # remove from the values that it is given, so three empty strings
            # remove *nothing*, and SQLite reports no error. In that state, every
            # posting of every re-indexed symbol stayed. `symbols.id` has no
            # `AUTOINCREMENT`, so the next insert took the id of the dead row,
            # and the stale posting then pointed at a different symbol. `search`
            # joins the rowid back to `symbols`, so the old name returned the new
            # symbol. That is worse than a miss, because it reads like an answer.
            #
            # Both tuples come from `_fts_values`. The agreement that the
            # paragraph above depends on is therefore structural, and does not
            # depend on a reader who checks two expressions in two methods.
            stale = [
                _fts_values(row["id"], row["name"], row["doc"], row["path"])
                for row in connection.execute(
                    "SELECT id, name, doc, path FROM symbols WHERE path = ?", (path,)
                )
            ]
            connection.executemany(
                "INSERT INTO symbols_fts(symbols_fts, rowid, name, doc, path) "
                "VALUES ('delete', ?, ?, ?, ?)",
                stale,
            )
            connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
            connection.execute("DELETE FROM refs WHERE path = ?", (path,))
            connection.execute("DELETE FROM imports WHERE path = ?", (path,))
            gone += connection.execute("DELETE FROM files WHERE path = ?", (path,)).rowcount
        return gone

    def put(self, path: str, digest: str, extraction: Extraction, vcs_id: str = "") -> int:
        """Replace the rows of one file with `extraction`. Returns the number of
        symbols written.

        This replaces and does not merge. A file is the unit of extraction, so
        the old rows are deleted and the new rows are written in one transaction.
        A crash during the write therefore leaves the file absent from `files`.
        The indexer treats that as "not indexed" and does the file again. The
        alternative, half of its symbols left behind, is something that nothing
        would ever notice.
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
                    _fts_values(identifier, definition.name, definition.doc, path),
                )
            # Build one `{line: tightest definition}` table for the whole file,
            # and do not scan every definition for each reference. The scan was
            # O(definitions x references). For a generated file with 1 500
            # definitions and 3 000 references it took 58 ms, as much as the
            # whole parse.
            #
            # The owner is resolved here, not in a later pass. The containing
            # definition is known while the definitions of the file are in hand.
            # A join that computed containment again for each query would be a
            # range scan over every symbol in the file.
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
        """Search names and docstrings by relevance. This uses FTS5, which turso
        does not have."""
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
        """The symbols that contain a reference to `name`, and the line of each
        reference."""
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
        """The definitions of the names that are referenced from inside `name`."""
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
        """The transitive callers, `1..distance` hops out, each with its hop count.

        **This is the recursive CTE that turso cannot run.** It is one statement,
        not a breadth-first search with N queries for each ring. The difference
        is not style. A Python loop would issue one query for each symbol on the
        frontier. An agent asks this query when it wants to know what a change
        breaks, over a graph in which the interesting names have hundreds of
        callers.

        `MIN(depth)` is used because a symbol that can be reached at two
        distances belongs to the nearer one. That makes the rings a budget and
        not a multiset. The hand-written walk of the old row had the same rule.
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
        """The biggest definitions first: "what is worth opening", in one list."""
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
        """The number of indexed files. **A query header needs only this number.**

        This is its own method because `stats()` also counts `symbols` and
        `refs`, and groups `files` by language. Those are three full scans, and a
        query paid for them on each call to fill one field. Measured on an index
        of 2 000 files: 1.97 ms against 0.22 ms, and the cost grows with the
        corpus without limit.
        """
        if not self.exists():
            return 0
        with self._open() as connection:
            try:
                return int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            except sqlite3.OperationalError:
                return 0

    def definition_count(self, name: str) -> int:
        """The number of places that define `name`. A name-based graph must report
        this ambiguity.

        This uses `COUNT(*)` and not `len(define(name, 100))`. That call
        materialized up to a hundred rows, and for any name past the cap it
        reported the *cap* as the count.
        """
        with self._open() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM symbols WHERE name = ?", (name,)
                ).fetchone()[0]
            )

    def stats(self) -> dict[str, Any]:
        """The counts that `phern doctor` and the header of every result report."""
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
