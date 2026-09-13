"""The index on disk: a turbovec file for the vectors, JSON for what they mean.

Two files, because they answer to different rules. `index.tvim` is turbovec's
own format, written through `sync()` — incremental, one fsync per call, and
crash-safe at any byte. `chunks.json` is the sidecar: which document each vector
came from, which lines, and the passage itself.

**The sidecar holds the passage text, and that is a deliberate cost.** Search
could return a path and a line range and let the caller `read` it, which would
keep this file small. It would also make every hit a second tool call before the
model learns whether the hit was any good, and would silently answer from a file
that has changed since it was indexed. Holding the text means one call answers,
and what it answers with is what was actually indexed.

## What is *not* claimed

The sidecar is rewritten whole on every save, while the vectors append. That is
fine for a corpus of documents and wrong for a corpus of millions; the ceiling
is the sidecar, not turbovec. It is stated here rather than discovered later.

The pair can also diverge — a crash between the two writes leaves vectors with
no meaning, or meanings with no vectors. `load` reconciles by trusting the
sidecar and dropping any id the index does not hold, because a chunk nobody can
retrieve is invisible while a vector with no text would surface as a hit this
row could not describe.

## Calibration

turbovec's TQ+ calibration is worth 2.5 to 8.7 points of R@10, and its
documentation is emphatic about the one way to get it wrong: the sample must be
a uniform random, representative draw of what the index will hold, and a
clustered prefix "fits a calibration that actively destroys recall". An
incremental indexer does not have such a draw at the moment it would have to
commit one — the first document is the most clustered prefix there is.

So it is committed in exactly one situation, which is the one the upstream
advice describes: the index is **empty**, and the batch about to land is large
enough to sample from. Then a random sample of that batch *is* a representative
draw of the index's contents, and calibrating before the add is the documented
order. Otherwise the index stays uncalibrated, which is plain TurboQuant and
merely good rather than wrong.

@module ph_text_index._store
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ph.json import as_int, as_str

from ._chunk import Chunk

__all__ = ["CALIBRATION_SAMPLE", "IndexMismatch", "Record", "TextIndex"]

log = logging.getLogger("ph_text_index.store")

FORMAT = 1
"""The sidecar's shape. Bumped when a field changes meaning; an unknown version
is refused rather than guessed at.

Still the baseline: nothing has shipped, so `vcs` and `vcs_ids` are part of what
a version-1 sidecar *is* rather than an addition to migrate to. Every field is
read back through `.get` with a default, so the refusal here is for a shape from
the future, not for one of ours missing a key."""

CALIBRATION_SAMPLE = 1_024
"""Rows turbovec's own guidance calls enough for a calibration fit."""


class IndexMismatch(RuntimeError):
    """The index on disk was built by a different embedder than this one.

    Its own type because the only fix is a human decision — re-embed the corpus
    under the new model, or put the old row back — and a caller has to be able
    to say that rather than reporting a dimension number.
    """


@dataclass(frozen=True, slots=True)
class Record:
    """One indexed passage: the vector's id, and everything it means."""

    id: int
    path: str
    """As the workspace names it, so a hit is a path the agent can hand to `read`."""
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
    """A turbovec index and its sidecar, as one thing that opens and saves.

    Every method here is **blocking** — file I/O and SIMD search — and is called
    through a worker thread by the service that owns it. Written synchronously on
    purpose: an index whose consistency depends on interleaving is one nobody can
    reason about, and the service's lock is easier to see than an await inside a
    mutation.
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
    """The version-control token from the last index — see `ph.seams.changes`."""
    _vcs_ids: dict[str, str] = field(default_factory=dict)
    """Path → the backend's content id when this path was last indexed.

    Beside the digests rather than inside a `Record`, because it is one fact per
    *document* while a record is one per passage: storing it per chunk would be
    the same string repeated once for every paragraph of every file."""

    # ------------------------------------------------------------- paths ----

    @property
    def _dim(self) -> int | None:
        """The vector width, asked of the index rather than tracked beside it.

        It was a field read from the sidecar and then immediately overwritten
        from `self._index.dim` whenever an index file existed — two values that
        had to agree, one of which could be stale in exactly the state `stats()`
        reports to `ph doctor`.
        """
        return None if self._index is None else int(self._index.dim)

    @property
    def index_path(self) -> Path:
        return self.root / "index.tvim"

    @property
    def sidecar_path(self) -> Path:
        return self.root / "chunks.json"

    # -------------------------------------------------------------- open ----

    def open(self) -> None:
        """Load what is on disk, or start empty. Idempotent.

        :raises IndexMismatch: when the sidecar names a different embedder.
        """
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
        # Trusting the sidecar and dropping what the index cannot answer for; see
        # the module docstring on divergence.
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
        self._reindex_paths()

    def _reindex_paths(self) -> None:
        self._by_path = {}
        for record in self._records.values():
            self._by_path.setdefault(record.path, []).append(record.id)

    # -------------------------------------------------------------- save ----

    def save(self) -> None:
        """Persist both halves — vectors first, so a crash never orphans text.

        The order is the whole of the crash story. `sync` commits durably before
        it returns, so a failure after it leaves vectors the next `open` cannot
        name and drops; a failure *before* it would leave named chunks with no
        vector, which is the divergence that surfaces as an unanswerable hit.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._index.sync(str(self.index_path))
        payload = {
            "format": FORMAT,
            "vcs": self._token,
            "vcs_ids": self._vcs_ids,
            "model": self.model,
            "dim": self._dim,
            "bit_width": self.bit_width,
            "next_id": self._next_id,
            "records": [record.as_value() for record in self._records.values()],
        }
        # Replace, never truncate-in-place: the sidecar is the only copy of what
        # the vectors mean, and a partial write of it loses the corpus.
        scratch = self.sidecar_path.with_suffix(f".{secrets.token_hex(4)}.tmp")
        scratch.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(scratch, self.sidecar_path)

    # ------------------------------------------------------------ mutate ----

    @property
    def token(self) -> str:
        """The stored tree token, for the next run to diff from."""
        return self._token

    def remember(self, token: str) -> None:
        self._token = token

    def vcs_id(self, path: str) -> str:
        """What the backend called this document's content when it was indexed."""
        return self._vcs_ids.get(path, "")

    def holds(self, path: str) -> bool:
        """Whether this index has passages for `path` at all.

        **The guard in front of `vouches_for`**, and it has to exist here rather
        than in the caller's head. jj answers with no per-file ids, so
        `vouches_for` proves only that a path has not *changed* since the token —
        which is equally true of a document that was never indexed. Without this,
        a first run over `docs/a.md` followed by a run over `docs/` counted every
        other document as unchanged and left them out of the index permanently.
        The sibling package spells the same guard as `stored_digest and …`.
        """
        return path in self._by_path

    def forget(self, path: str) -> int:
        """Drop every chunk of one document. Returns how many there were."""
        self._vcs_ids.pop(path, None)
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
        vectors: Any,  # noqa: ANN401
        vcs_id: str = "",
    ) -> int:
        """Index `chunks`, whose rows are `vectors`. Returns how many landed.

        :raises IndexMismatch: when the vectors' width is not the index's.
        """
        import numpy as np

        if len(chunks) != int(vectors.shape[0]):
            raise ValueError(f"{len(chunks)} chunks against {vectors.shape[0]} vectors")
        if not chunks:
            return 0
        self._vcs_ids[path] = vcs_id
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

    def _maybe_calibrate(self, rows: Any) -> None:  # noqa: ANN401
        """Fit TQ+ before the first add, when this batch is a fair sample of it."""
        import numpy as np

        if not self.calibrate or int(rows.shape[0]) < CALIBRATION_SAMPLE:
            return
        generator = np.random.default_rng()
        picked = generator.choice(int(rows.shape[0]), size=CALIBRATION_SAMPLE, replace=False)
        self._index.calibrate(np.ascontiguousarray(rows[picked], dtype=np.float32))

    # ------------------------------------------------------------ search ----

    def search(
        self,
        vector: Any,  # noqa: ANN401
        k: int,
        *,
        allowed: list[int] | None = None,
    ) -> list[tuple[float, Record]]:
        """The best `k` passages for one query vector, best first.

        **Takes the id set rather than the paths**, so a filtered search builds
        it once. It used to take `paths` and call `ids_under` itself, while the
        caller called `ids_under` separately for the count it reports — two
        O(chunks) passes per query, measured at 7.5 ms against a 0.22 ms
        unfiltered search on a 10 000-chunk index.

        That also retires a claim this docstring used to make. turbovec does
        filter inside the SIMD kernel, but the Python-side allowlist dominates
        it by roughly fifteen times, so a selective filter does **not** cost
        less than an unfiltered search — it costs more, and the honest reason to
        use one is precision, not speed.

        `None` means no filter; an empty list means nothing is allowed, which is
        this layer's answer to give because turbovec raises on an empty
        allowlist and cannot tell the two apart.
        """
        import numpy as np

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
        """Chunk ids of every indexed document at or under one of `paths`.

        **The prefix test is per document, not per id.** It used to sit inside
        the id loop, so the `any()` and its `startswith` ran once per chunk and
        an FFI `contains` went with it: 3.26 ms for 10 000 chunks against 0.11 ms
        here, and selectivity bought nothing because the cost was in building
        the list rather than in scoring it.

        The `contains` probe is gone with it. `open()` already reconciles the
        sidecar against the index and `add`/`forget` maintain the pair, so
        re-verifying an established invariant once per chunk per query was
        paying for a divergence that cannot survive a load.

        The prefix itself is `ph.paths.is_under`'s, spelled once here instead of
        called per document: `is_under` takes `Path`s and rebuilds the normcased
        root string on every call, so asking it per document rebuilt it 1 500
        times for a filter of one — 4.20 ms against 0.42 ms for 1 500 documents,
        for the same answer. The comparison is its body, case-folded the way
        every other path comparison in the tree is.
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


def _turbovec() -> Any:  # noqa: ANN401
    # No py.typed marker upstream, and the surface used here is four methods on
    # one class — a stub file would be more of this module than the module.
    import turbovec  # type: ignore[import-untyped]

    return turbovec
