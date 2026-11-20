"""The index on disk: a turbovec file for the vectors, and JSON for what they mean.

There are two files, because they follow different rules. `index.tvim` is the
format of turbovec itself. It is written through `sync()`, which is incremental,
does one fsync for each call, and is crash-safe at any byte. `chunks.json` is the
sidecar: for each vector, the document that it came from, the lines, and the
passage itself.

**The sidecar holds the passage text, and that cost is deliberate.** Search could
return a path and a line range and let the caller `read` it, which would keep
this file small. But every hit would then need a second tool call before the
model learns whether the hit was any good. The answer would also come silently
from a file that has changed since it was indexed. Because the sidecar holds the
text, one call answers, and the answer is the text that was really indexed.

## What is *not* claimed

The sidecar is rewritten whole on every save, while the vectors append. That is
fine for a corpus of documents and wrong for a corpus of millions. The ceiling is
the sidecar, not turbovec. It is stated here so that nobody has to discover it
later.

The two files can also diverge. A crash between the two writes leaves vectors
with no meaning, or meanings with no vectors. `open` reconciles toward the
sidecar in both directions. It drops any chunk that the index does not hold,
because a chunk that nobody can retrieve is invisible. It removes any vector that
the sidecar does not name, because the next `add` would allocate the id of that
vector again.

**One writer at a time, across processes.** Every root and process on a
workspace shares its directory. Each holds its own copy in memory, and a save
writes that copy whole. With two writers, the sidecar therefore described
whichever writer saved last. The service opens and writes under a file lock. It
loads a copy again when another writer has changed it since (`stale`). See
`TextIndexSeam.writing`.

## Calibration

The TQ+ calibration of turbovec is worth 2.5 to 8.7 points of R@10. Its
documentation is emphatic about the one way to get it wrong. The sample must be a
uniform random, representative draw of what the index will hold, and a clustered
prefix "fits a calibration that actively destroys recall". An incremental indexer
does not have such a draw at the moment when it would have to commit one: the
first document is the most clustered prefix that exists.

The calibration is therefore committed in exactly one situation, the one that
the upstream advice describes: the index is **empty**, and the batch that is
about to be added is large enough to sample from. A random sample of that batch
*is* then a representative draw of the contents of the index, and calibrating
before the add is the documented order. Otherwise the index stays uncalibrated.
That is plain TurboQuant, which is merely good and not wrong.

@module ph_text_index._store
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from ph.json import as_int, as_seq, as_str
from ph.paths import file_stamp, write_atomic

from ._chunk import Chunk
from ._config import EmbedderRecord

if TYPE_CHECKING:
    from ._embed import Vectors

__all__ = ["CALIBRATION_SAMPLE", "IndexMismatch", "Record", "TextIndex"]

log = logging.getLogger("ph_text_index.store")

FORMAT = 1
"""The shape of the sidecar. The number goes up when a field changes meaning. An
unknown version is refused and not guessed at.

This is still the baseline. Nothing has shipped, so `vcs` and `vcs_ids` are part
of what a version-1 sidecar *is*, and not an addition to migrate to. Every field
is read back through `.get` with a default. The refusal here is therefore for a
shape from the future, not for one of our own that is missing a key."""

CALIBRATION_SAMPLE = 1_024
"""The number of rows that the guidance of turbovec calls enough for a
calibration fit."""

_Signature: TypeAlias = tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]
"""The `file_stamp` of the sidecar, and of the vectors."""


class IndexMismatch(RuntimeError):
    """A different embedder than this one built the index on disk.

    This has its own type because the only fix is a human decision: embed the
    corpus again under the new model, or put the old row back. A caller must be
    able to say that, and not report a dimension number.
    """


@dataclass(frozen=True, slots=True)
class Record:
    """One indexed passage: the id of the vector, and everything that it means."""

    id: int
    path: str
    """As the workspace names it, so that a hit is a path that the agent can pass
    to `read`."""
    start_line: int
    end_line: int
    text: str

    def as_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }


@dataclass(slots=True)
class TextIndex:
    """A turbovec index and its sidecar, as one object that opens and saves.

    Every method here **blocks**, because it does file I/O and SIMD search. The
    service that owns the index calls each method through a worker thread. The
    class is synchronous on purpose. Nobody can reason about an index whose
    consistency depends on interleaving, and the lock of the service is easier to
    see than an await inside a mutation.
    """

    root: Path
    model: str
    """Which embedder built it. Compared on load, not assumed."""
    bit_width: int = 4
    calibrate: bool = True
    _index: Any = None
    _records: dict[int, Record] = field(default_factory=dict)
    _by_path: dict[str, list[int]] = field(default_factory=dict)
    _next_id: int = 1
    _token: str = ""
    """The version-control token from the last index. See `ph.seams.changes`."""
    _vcs_ids: dict[str, str] = field(default_factory=dict)
    """Path → the content id from the backend when this path was last indexed.

    This map is beside the digests and not inside a `Record`. It holds one fact
    for each *document*, while a record is one for each passage. Stored for each
    chunk, it would be the same string repeated for every paragraph of every
    file."""
    _digests: dict[str, str] = field(default_factory=dict)
    """Path → a digest of the text of the document when this path was last
    indexed.

    The digest shows "unchanged" in a tree with no version control, and after a
    run that was stopped before it could store its token. The next run reads the
    document, finds the digest that it already has, and embeds nothing. The map
    is for each document, beside `_vcs_ids`, for the same reason as that field."""
    _seen: _Signature = (None, None)
    """The files as this copy last read or wrote them. See `stale`."""
    readonly: bool = False
    """Whether a reader opened this copy (`open`). Such a copy may hold vectors
    that no record names, so it can be searched and must never be added to."""
    _embedder: list[EmbedderRecord] = field(default_factory=list)
    """The embedder log, oldest first: what the embedder reported about itself
    each time an index run found that its answer had changed (`EmbedderRecord`).
    The log is saved with the passages, so the log and the passages that it was
    used to cut never separate."""

    # ------------------------------------------------------------- paths ----

    @property
    def _dim(self) -> int | None:
        """The vector width, read from the index and not tracked beside it.

        It used to be a field that was read from the sidecar and then immediately
        overwritten from `self._index.dim` whenever an index file existed. Those
        were two values that had to agree, and one of them could be stale in
        exactly the state that `stats()` reports to `phern doctor`.
        """
        return None if self._index is None else int(self._index.dim)

    @property
    def index_path(self) -> Path:
        return self.root / "index.tvim"

    @property
    def sidecar_path(self) -> Path:
        return self.root / "chunks.json"

    # -------------------------------------------------------------- open ----

    def open(self, *, readonly: bool = False) -> None:
        """Load what is on disk, or start empty. Idempotent.

        `readonly` is the open of a reader, made without the writer lock: a
        search, or a status. It changes no file. A writer may be part-way through
        a save, with the vectors synced and the sidecar that names them not yet
        written. Those vectors are then left where they are, because `search`
        answers only for a vector that the sidecar names. Only a writer removes
        them (`_remove_strays`), under the lock. When a reader removed them, it
        took out of the file what the writer was about to name.

        :raises IndexMismatch: when the sidecar has a different `format`, or names
            a different embedder.
        """
        self._seen = self._signature()
        self.readonly = readonly
        if not self.sidecar_path.exists():
            return
        raw = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        version = raw.get("format")
        if version != FORMAT:
            raise IndexMismatch(
                f"{self.sidecar_path} is format {version}, and this row reads {FORMAT}; "
                "delete the directory to rebuild"
            )
        stored = as_str(raw.get("model"))
        if stored != self.model:
            raise IndexMismatch(
                f"{self.root} was built with embedder {stored!r} and this deployment "
                f"has {self.model!r}; a vector from one model means nothing to the "
                "other, so delete the directory and re-index"
            )
        self._next_id = as_int(raw.get("next_id"), 1)
        self._token = as_str(raw.get("vcs"))
        self._vcs_ids = {str(path): str(one) for path, one in (raw.get("vcs_ids") or {}).items()}
        # An index that was written before documents had a digest has none. Each
        # document then reads as changed one time, and is embedded again.
        self._digests = {str(path): str(one) for path, one in (raw.get("digests") or {}).items()}
        self._embedder = _embedder_log(raw.get("embedder"), self.sidecar_path)
        if self.index_path.exists():
            self._index = _turbovec().IdMapIndex.load(str(self.index_path))
        held = {
            record["id"]: Record(
                id=as_int(record["id"]),
                path=as_str(record["path"]),
                start_line=as_int(record["start_line"]),
                end_line=as_int(record["end_line"]),
                text=as_str(record["text"]),
            )
            for record in raw.get("records", [])
        }
        # Trust the sidecar, and drop what the index cannot answer for. See the
        # module docstring on divergence.
        self._records = {
            id_: record
            for id_, record in held.items()
            if self._index is not None and self._index.contains(id_)
        }
        dropped = len(held) - len(self._records)
        if dropped:
            log.warning(
                "ph_text_index: %d sidecar chunk(s) in %s have no vector and were dropped",
                dropped,
                self.root,
            )
        if not readonly:
            self._remove_strays()
        self._reindex_paths()

    def _remove_strays(self) -> None:
        """Remove every vector that the sidecar does not name, and write the
        result to disk.

        Such vectors are what a crash between the two writes of `save` leaves:
        the vectors synced, and the sidecar that names them never written. Their
        ids start at the `next_id` of the sidecar, which is where the next `add`
        allocates. turbovec refuses an id that it already holds, so every later
        add raised until the directory was deleted.

        turbovec lists no ids, so the strays are found with a search whose `k` is
        the whole index. That is one full scan, and it runs only when the count
        shows that there is something to find. Every id that the sidecar kept is
        in the index, so the difference in size is exactly the number of strays.
        The removal is synced immediately, so the scan is made one time and not
        on every open until somebody saves. This runs only for an open under the
        writer lock, where nobody else is writing.
        """
        if self._index is None:
            return
        strays = len(self._index) - len(self._records)
        if strays <= 0:
            return
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        query = np.zeros((1, int(self._index.dim)), dtype=np.float32)
        _scores, ids = self._index.search(query, len(self._index))
        for id_ in ids[0].tolist():
            if id_ not in self._records:
                self._index.remove(id_)
        self._index.sync(str(self.index_path))
        self._seen = self._signature()
        log.warning(
            "ph_text_index: %d vector(s) in %s had no sidecar chunk and were removed",
            strays,
            self.root,
        )

    def _signature(self) -> _Signature:
        return (file_stamp(self.sidecar_path), file_stamp(self.index_path))

    def stale(self) -> bool:
        """Whether another writer changed the files since this copy read or wrote them.

        The check is one stat of each file, so it is cheap enough to make before
        every use. It covers both files, because a writer that died between its
        two writes changed only the vectors. A copy that was checked against the
        sidecar alone would sync its own changes onto a file that had changed
        under it."""
        return self._signature() != self._seen

    def _reindex_paths(self) -> None:
        self._by_path = {}
        for record in self._records.values():
            self._by_path.setdefault(record.path, []).append(record.id)

    # -------------------------------------------------------------- save ----

    def save(self) -> None:
        """Persist both files, vectors first, so that a crash never leaves text
        without a vector.

        The order is all of the crash handling. `sync` commits durably before it
        returns. A failure after it leaves vectors that the next `open` cannot
        name, and `open` drops them. A failure *before* it would leave named
        chunks with no vector. That is the divergence that appears as a hit that
        cannot be answered.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._index.sync(str(self.index_path))
        payload = {
            "format": FORMAT,
            "vcs": self._token,
            "vcs_ids": self._vcs_ids,
            "digests": self._digests,
            "embedder": [entry.to_wire() for entry in self._embedder],
            "model": self.model,
            "dim": self._dim,
            "bit_width": self.bit_width,
            "next_id": self._next_id,
            "records": [record.as_value() for record in self._records.values()],
        }
        # Replace the file, and never truncate it in place. The sidecar is the
        # only copy of what the vectors mean, and a partial write of it loses the
        # corpus.
        write_atomic(self.sidecar_path, json.dumps(payload))
        self._seen = self._signature()

    # ---------------------------------------------------------- embedder ----

    def embedder(self) -> EmbedderRecord | None:
        """What the embedder last reported about itself when an index run asked
        it: the newest entry of the log, or `None` before any run has embedded."""
        return self._embedder[-1] if self._embedder else None

    def note_embedder(self, record: EmbedderRecord) -> bool:
        """Add `record` to the log when it says something that the newest entry
        does not. Returns whether it was added. The record is saved with the next
        `save`, which the caller must then make."""
        newest = self.embedder()
        # The time when it was asked is not part of what it reported.
        if newest is not None and newest.model_dump(exclude={"asked_at_ms"}) == record.model_dump(
            exclude={"asked_at_ms"}
        ):
            return False
        self._embedder.append(record)
        return True

    # ------------------------------------------------------------ mutate ----

    @property
    def token(self) -> str:
        """The stored tree token, for the next run to diff from."""
        return self._token

    def remember(self, token: str) -> None:
        self._token = token

    def vcs_id(self, path: str) -> str:
        """The content id that the backend gave this document when it was
        indexed."""
        return self._vcs_ids.get(path, "")

    def digest(self, path: str) -> str:
        """The digest of the text of this document when it was indexed, or `""`."""
        return self._digests.get(path, "")

    def vouch(self, path: str, vcs_id: str) -> None:
        """Record the content id that the backend gives this document now, for a
        document that its digest showed to be unchanged. The backend then tells
        the next run that the document is unchanged, and the run does not have to
        read it."""
        if vcs_id:
            self._vcs_ids[path] = vcs_id

    def holds(self, path: str) -> bool:
        """Whether this index has any passages for `path`.

        **This is the guard in front of `vouches_for`**, and it must exist here
        and not only in the mind of the caller. jj answers with no per-file ids,
        so `vouches_for` proves only that a path has not *changed* since the
        token. That is equally true of a document that was never indexed. Without
        this guard, a first run over `docs/a.md` followed by a run over `docs/`
        counted every other document as unchanged, and left them out of the index
        permanently. The sibling package writes the same guard as
        `stored_digest and …`.
        """
        return path in self._by_path

    def forget(self, path: str) -> int:
        """Drop every chunk of one document. Returns how many there were."""
        self._vcs_ids.pop(path, None)
        self._digests.pop(path, None)
        ids = self._by_path.pop(path, [])
        for id_ in ids:
            self._records.pop(id_, None)
            if self._index is not None:
                self._index.remove(id_)
        return len(ids)

    def add(
        self,
        path: str,
        chunks: list[Chunk],
        vectors: Vectors,
        vcs_id: str = "",
        digest: str = "",
    ) -> int:
        """Index `chunks`, whose rows are `vectors`. Returns the number that were
        added.

        :raises IndexMismatch: when the width of the vectors is not the width of
            the index.
        """
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        if len(chunks) != int(vectors.shape[0]):
            raise ValueError(f"{len(chunks)} chunks against {vectors.shape[0]} vectors")
        if not chunks:
            return 0
        self._vcs_ids[path] = vcs_id
        self._digests[path] = digest
        rows = np.ascontiguousarray(vectors, dtype=np.float32)
        width = int(rows.shape[1])
        if self._index is None:
            self._index = _turbovec().IdMapIndex(dim=width, bit_width=self.bit_width)
            self._maybe_calibrate(rows)
        elif width != self._dim:
            raise IndexMismatch(
                f"{self.model!r} produced {width}-dimensional vectors and {self.root} "
                f"holds {self._dim}-dimensional ones; delete the directory and re-index"
            )
        ids = np.array([self._next_id + offset for offset in range(len(chunks))], dtype=np.uint64)
        self._next_id += len(chunks)
        self._index.add_with_ids(rows, ids)
        for chunk, id_ in zip(chunks, ids.tolist(), strict=True):
            record = Record(
                id=int(id_),
                path=path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
            )
            self._records[record.id] = record
            self._by_path.setdefault(path, []).append(record.id)
        return len(chunks)

    def _maybe_calibrate(self, rows: Vectors) -> None:
        """Fit TQ+ before the first add, when this batch is a fair sample of the
        index."""
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        if not self.calibrate or int(rows.shape[0]) < CALIBRATION_SAMPLE:
            return
        generator = np.random.default_rng()
        picked = generator.choice(int(rows.shape[0]), size=CALIBRATION_SAMPLE, replace=False)
        self._index.calibrate(np.ascontiguousarray(rows[picked], dtype=np.float32))

    # ------------------------------------------------------------ search ----

    def search(
        self,
        vector: Vectors,
        k: int,
        *,
        allowed: list[int] | None = None,
    ) -> list[tuple[float, Record]]:
        """The best `k` passages for one query vector, best first.

        **This takes the id set and not the paths**, so that a filtered search
        builds the set one time. It used to take `paths` and call `ids_under`
        itself, while the caller called `ids_under` separately for the count that
        it reports. Those were two O(chunks) passes for each query, measured at
        7.5 ms against a 0.22 ms unfiltered search on a 10 000-chunk index.

        That also removes a claim that this docstring used to make. turbovec does
        filter inside the SIMD kernel, but the Python-side allowlist costs
        roughly fifteen times as much. A selective filter therefore does **not**
        cost less than an unfiltered search. It costs more, and the correct
        reason to use one is precision, not speed.

        `None` means no filter. An empty list means that nothing is allowed. This
        layer must give that answer, because turbovec raises on an empty
        allowlist and cannot tell the two apart.
        """
        # Heavy, and needed only where an index is built or searched.
        import numpy as np  # noqa: PLC0415

        if self._index is None or not self._records:
            return []
        if allowed is not None and not allowed:
            return []
        query = np.ascontiguousarray(np.atleast_2d(vector), dtype=np.float32)
        allowlist = None if allowed is None else np.array(sorted(set(allowed)), dtype=np.uint64)
        scores, ids = self._index.search(query, k, allowlist=allowlist)
        hits: list[tuple[float, Record]] = []
        for score, id_ in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            record = self._records.get(int(id_))
            if record is not None:
                hits.append((float(score), record))
        return hits

    def ids_under(self, paths: Sequence[str]) -> list[int]:
        """The chunk ids of every indexed document at or under one of `paths`.

        **The prefix test runs for each document, not for each id.** It used to
        be inside the id loop, so the `any()` and its `startswith` ran one time
        for each chunk, with an FFI `contains` call beside it. That took 3.26 ms
        for 10 000 chunks, against 0.11 ms here. Selectivity gave no benefit,
        because the cost was in building the list and not in scoring it.

        The `contains` probe was removed at the same time. `open()` already
        reconciles the sidecar against the index, and `add` and `forget` maintain
        the pair. A check of an established invariant for each chunk of each
        query paid for a divergence that cannot survive a load.

        The prefix rule itself is the rule of `ph.paths.is_under`, written one
        time here and not called for each document. `is_under` takes `Path`s and
        rebuilds the normcased root string on every call. Called for each
        document, it rebuilt that string 1 500 times for a filter of one:
        4.20 ms against 0.42 ms for 1 500 documents, for the same answer. The
        comparison here is the body of `is_under`, case-folded in the same way as
        every other path comparison in the tree.
        """
        roots = [os.path.normcase(one).rstrip("/") for one in paths]
        found: list[int] = []
        for path, ids in self._by_path.items():
            here = os.path.normcase(path)
            if any(here == root or here.startswith(f"{root}/") for root in roots):
                found.extend(ids)
        return found

    # ------------------------------------------------------------- report ----

    def documents(self) -> list[str]:
        return sorted(self._by_path)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._by_path),
            "chunks": len(self._records),
            "dim": self._dim,
            "bit_width": self.bit_width,
            "calibration": (
                getattr(self._index, "calibration_state", "empty")
                if self._index is not None
                else "empty"
            ),
        }


def _embedder_log(raw: object, where: Path) -> list[EmbedderRecord]:
    """The embedder log of the sidecar, oldest first. A log that cannot be read
    counts as no log: the next run that embeds asks again and starts a new log."""
    try:
        return [EmbedderRecord.model_validate(entry) for entry in as_seq(raw)]
    except ValueError:
        log.warning("ph_text_index: the embedder log in %s will not read", where, exc_info=True)
        return []


def _turbovec() -> Any:  # noqa: ANN401
    # Upstream has no py.typed marker, and the surface used here is seven methods
    # on one class. A stub file would be more of this module than the module.
    # Heavy, and needed only where an index is built or searched.
    import turbovec  # type: ignore[import-untyped]  # noqa: PLC0415

    return turbovec
