"""`text-index` — chunking that keeps its place, and an index that survives a reopen.

Gates: *no embedder means no tool; a reopened index answers the same question;
a chunk's line span names the lines it actually came from.*

## Why a stub embedder rather than the real model

The shipped provider is `sentence-transformers`, which downloads weights on
first use. A suite that loaded it would be a suite that fails in an airgapped
CI, takes a minute to prove that paging works, and tests the model rather than
this row. So these tests register a deterministic hashing embedder and exercise
**the real turbovec index** through it — which is where the interesting
behaviour is: quantized scores, an allowlist that raises on empty, incremental
`sync`, and a sidecar that has to agree with a binary file.

`HashingEmbedder` is a bag of words over `blake2b` buckets, so similar text
really does produce similar vectors and the retrieval assertions below are about
ranking rather than about a fixture. `blake2b` and not `hash()`, which is salted
per process — a stub whose vectors changed between runs would make a reopened
index look corrupt.

## Why the line span is asserted against the file

A chunk that does not know where it came from is the failure this row exists to
avoid, and it is a silent one: the passage still reads correctly, the score is
still fine, and the `read` the model does next lands somewhere else. So the
assertions compare the span against the document's own lines rather than against
an expected number.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ph.cordis import DEPLOYMENT
from ph.keys import AGENTS, COMMANDS, FS, SESSIONS, SKILLS, SYSTEM_PROMPT, TOOLS
from ph.llm.types import text_of
from ph.testing import FAKE_OPTIONS, MountProfile, report_section, run_tool
from ph.testing.git import git, git_repo
from ph.testing.jj import jj_repo
from ph_text_index import TEXT_INDEX
from ph_text_index._chunk import chunk_text
from ph_text_index._store import IndexMismatch, TextIndex

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "text-index", "name": "text-index"}

DOCUMENT = """# Workspaces

The workspace seam decides where an agent's writes land.

A containment tier is what a deployment asks for: advisory keeps the agent in
the person's own checkout, and a worktree branches a fresh one.

## Sandboxing

Confinement is a different question, answered by the sandbox backend — bwrap on
Linux and Seatbelt on macOS.
"""


@dataclass(slots=True)
class HashingEmbedder:
    """A deterministic bag-of-words embedder. See the module docstring."""

    dim: int = 96
    calls: int = 0

    @property
    def name(self) -> str:
        return f"stub-hashing:{self.dim}"

    def encode(self, texts: Any, *, query: bool) -> Any:
        self.calls += 1
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            for token in "".join(
                character if character.isalnum() else " " for character in text.lower()
            ).split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                rows[index, int.from_bytes(digest, "big") % self.dim] += 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(rows / norms, dtype=np.float32)


_SEQ = itertools.count()


def _agent(ctx: Any) -> Any:
    """A fresh agent, on a session id nothing else has taken.

    Counted rather than fixed: a test needing two agents would otherwise collide
    on the session id and fail as `SESSION_ALREADY_EXISTS`, several frames from
    the cause. The sibling package's helper says the same.
    """
    return ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create(f"ti-{next(_SEQ)}"), FAKE_OPTIONS
    )


async def _mounted(
    mount: MountProfile, tmp_path: Path, **config: Any
) -> tuple[Any, HashingEmbedder]:
    """The seam, plus a stub embedder claimed the way a provider row claims one.

    The `reconcile()` matters and is not ceremony: the tools are registered
    directly on the claim, but the **skill** goes through `contribute_via`, which
    waits on the `skills` key through the loader's fixpoint. During a real mount
    that settles as part of the mount; a claim made by hand afterwards has to ask
    for the same settling, which is what `test_subagent_task` spells as an
    explicit `profile/mounted` for the same reason.
    """
    settings = {"path": str(tmp_path / "index"), **config}
    ctx = await mount({**ROW, "config": settings})
    embedder = HashingEmbedder()
    ctx.require(TEXT_INDEX).register(embedder)
    await ctx.reconcile()
    return ctx, embedder


# ----------------------------------------------------------------- chunking ----


def test_a_chunk_knows_the_lines_it_came_from() -> None:
    """Asserted against the document, not against a number. See the docstring."""
    lines = DOCUMENT.splitlines()
    chunks = chunk_text(DOCUMENT, max_chars=200, overlap_chars=0)

    assert chunks
    for chunk in chunks:
        span = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph in span, f"{paragraph!r} is not within its own line span"


def test_cuts_land_between_paragraphs() -> None:
    """The reason this is not a character stride: no chunk starts mid-sentence."""
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


def test_the_overlap_never_repeats_a_whole_chunk() -> None:
    """`_tail`'s guard: a carry-over of everything would stop the walk advancing."""
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


def test_an_empty_document_yields_nothing() -> None:
    assert chunk_text("", max_chars=100, overlap_chars=0) == []
    assert chunk_text("\n\n   \n", max_chars=100, overlap_chars=0) == []


# -------------------------------------------------------------------- store ----


def _store(root: Path, **kwargs: Any) -> TextIndex:
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
    """The crash-between-two-writes case, reconciled toward the retrievable half."""
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


def test_a_filter_that_matches_nothing_returns_nothing_rather_than_raising(
    tmp_path: Path,
) -> None:
    """turbovec raises on an empty allowlist, and rightly — it cannot tell
    "nothing allowed" from "no filter". The empty answer is this layer's."""
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


async def test_no_embedder_means_no_tools(mount: MountProfile, tmp_path: Path) -> None:
    """A tool refused on every call would teach a capability nobody has."""
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})

    assert ctx.require(TOOLS).get("text_index", scope=DEPLOYMENT) is None
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is None


async def test_an_embedder_claimed_anywhere_brings_both_tools(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx, _ = await _mounted(mount, tmp_path)

    assert ctx.require(TOOLS).get("text_index", scope=DEPLOYMENT) is not None
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None


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
    # The pointer the whole row exists to hand back.
    assert 1 <= hits[0]["start_line"] <= hits[0]["end_line"] <= len(DOCUMENT.splitlines())
    assert f":{hits[0]['start_line']}-{hits[0]['end_line']}" in text_of(found.content)


async def test_a_hits_line_span_names_the_lines_in_the_real_file(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The span has to survive `glob`'s relative naming and the rejoin."""
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
    assert found.value["searched"] < ctx.require(TEXT_INDEX)._index.stats()["chunks"]


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
    """A call over a directory must not fail because it found a bundle — and the
    caller has to learn what was skipped, or a quiet corpus looks like a quiet
    failure."""
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

    A screen registered on `ctx.fs` decides what the walk may show, so a
    document it prunes must never reach the index — which it cannot if the row
    reads through the seam, and trivially would if it walked the tree itself.
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
    ctx, embedder = await _mounted(mount, tmp_path)

    assert report_section(ctx, "Text index")["embedder"] == embedder.name


async def test_doctor_says_when_no_embedder_is_registered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The one thing an operator needs told when the tools are simply absent."""
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})

    assert "none registered" in report_section(ctx, "Text index")["embedder"]


# ---------------------------------------------------- the bundle, and the RLM ----


def test_the_bundle_is_discoverable_without_importing_it() -> None:
    """The entry point is the whole coupling between a profile and this package."""
    from ph.bundles import installed_bundles, resolve_bundle
    from ph_text_index import BUNDLE

    assert "text-index" in installed_bundles()
    assert resolve_bundle("text-index") == BUNDLE


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    from ph.cordis import Profile
    from ph.cordis.loader import resolve_plugin
    from ph_text_index import BUNDLE

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        assert resolve_plugin(row["name"]) is not None, row["name"]


def test_the_bundle_ships_the_provider_beside_the_seam() -> None:
    """**The subtlety that would otherwise be a silence.**

    This seam registers no tools until an embedder is claimed, so a bundle
    carrying only `text-index` would mount a service and advertise nothing —
    and nothing would fail. The profile would compose, the row would activate,
    `ph doctor` would show the section, and the model would simply never be
    offered `text_search`. Both rows, or the bundle means nothing.
    """
    from ph.cordis import Profile
    from ph_text_index import BUNDLE

    assert [row.id for row in Profile.from_paths([BUNDLE]).enabled_rows()] == [
        "text-index",
        "text-index-local",
    ]


async def test_the_rlm_indexed_profile_layers_this_bundle() -> None:
    from ph_app.profiles import available_profiles, resolve_profile
    from ph_text_index import BUNDLE

    assert "rlm-indexed" in available_profiles()
    assert BUNDLE in resolve_profile("rlm-indexed")


async def test_code_mode_hands_the_model_both_tools_through_the_sdk(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Under Code Mode "can the RLM use this" is "are these in the SDK listing".

    Composed from Code Mode's own rows rather than by mounting `rlm`, because
    the claim is about any Code Mode deployment — and the embedder is claimed by
    hand here for the reason `_mounted` does it: the real provider row would
    load a model to prove a fact about a prompt.
    """
    ctx = await mount(
        {"id": "tools", "config": {"mode": "code"}},
        {"id": "tools-code-mode", "name": "tools-code-mode"},
        {"id": "code-runtime-stub", "name": "code-runtime-stub"},
        {**ROW, "config": {"path": str(tmp_path / "index")}},
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


async def test_no_embedder_means_the_sdk_offers_nothing_either(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The absence is complete, not partial: no tool, and no SDK line.

    A listing that named `text_search` while the seam had no provider would
    teach the model a capability every call would refuse — the whole reason the
    tools wait for the claim.
    """
    ctx = await mount(
        {"id": "tools", "config": {"mode": "code"}},
        {"id": "tools-code-mode", "name": "tools-code-mode"},
        {"id": "code-runtime-stub", "name": "code-runtime-stub"},
        {**ROW, "config": {"path": str(tmp_path / "index")}},
    )
    agent = _agent(ctx)

    prompt = await ctx.require(SYSTEM_PROMPT).assemble(agent=agent, scope=agent.ctx)
    sdk = dict(prompt.sections)["tools:sdk"]
    assert "text_search" not in sdk
    assert "text_index" not in sdk


def test_a_model_that_brings_its_own_code_is_reachable_by_config() -> None:
    """`nomic-embed-text-v1.5` and every other model with an `auto_map`.

    Its `config.json` maps `AutoModel` at `nomic-ai/nomic-bert-2048`, so
    `sentence-transformers` will not load it without `trust_remote_code=True` —
    which means that without this knob the model is simply unreachable, however
    the row is configured. Off by default, because granting it executes Python
    downloaded from a model repository in this process.

    Constructed, not loaded: the assertion is that the configuration reaches the
    embedder, and downloading half a gigabyte to prove a keyword argument was
    passed is not a test anyone should wait for.
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

    A prefix changes where a vector lands — nomic's `search_document: ` index is
    a different space from the same model's unprefixed one — so it belongs in
    the identity, and switching gets a fresh index rather than wrong neighbours.
    `trust_remote_code` changes what may *load*, so an index must not be
    invalidated because an operator granted or revoked it.
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
    """What makes trying a bigger model safe: no collision, and no re-embedding
    the old one to switch back.

    The slot is set directly rather than through `register`, because the claim
    is about how `root()` keys the directory and `register` would drag the
    provider's whole lifecycle in to say it. Registering twice without disposing
    the first is misuse — `claim_slot` refuses it — and the first draft of this
    test did exactly that, then reported a tool-name collision as though it were
    a finding about the index path.
    """
    ctx = await mount({**ROW})
    seam = ctx.require(TEXT_INDEX)

    seam.provider = HashingEmbedder(dim=96)
    small = seam.root()
    seam.provider = HashingEmbedder(dim=384)
    large = seam.root()

    assert small != large, "two embedders shared one index directory"
    assert small.parent == large.parent, "both still under the row's cache root"


# --------------------------------------------------- provisioning, and skills ----


@dataclass(slots=True)
class LoadableEmbedder(HashingEmbedder):
    """A stub that has a model to load, so the provisioning paths are exercised.

    `HashingEmbedder` deliberately has neither `load` nor `ready` — it is the
    *endpoint* shape, with nothing to download — so it cannot test the branch
    that matters here. This one counts loads and can be told to fail, which is
    the whole of what `provision` and `preload` do.
    """

    fails: str = ""
    loads: int = 0
    loaded: bool = False
    cache_folder: str = "/nowhere"
    """`LocalWeights` declares this, and `isinstance` checks it.

    The first draft of this double omitted it, so the Protocol reported it as an
    endpoint-backed embedder with nothing to download — which is exactly the
    misnamed-member case a `getattr` probe could not have caught."""

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


async def test_the_command_reports_and_then_provisions(mount: MountProfile, tmp_path: Path) -> None:
    """A person types this, and it costs no model turn — the seam's own rule.

    `status` before `install` is the point of the pair: "can I download it" is a
    question to answer *before* an agent is mid-turn, which is the whole reason
    this is not left to the first `text_index` call.
    """
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})
    embedder = LoadableEmbedder()
    ctx.require(TEXT_INDEX).register(embedder)
    agent = _agent(ctx)

    before = await ctx.require(COMMANDS).dispatch(
        "/text-index status", agent=agent, scope=agent.ctx
    )
    assert before is not None and "not loaded" in before
    assert embedder.loads == 0, "status must not download anything"

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)
    assert said is not None and "is ready (96-dimensional)" in said
    assert embedder.loads == 1

    after = await ctx.require(COMMANDS).dispatch("/text-index status", agent=agent, scope=agent.ctx)
    assert after is not None and "loaded" in after and "not loaded" not in after


async def test_a_failed_install_says_what_upstream_said(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The `einops` case, and why the message is passed through verbatim.**

    `nomic-embed-text-v1.5` downloads its weights and its remote modelling code
    successfully and then fails at import with "requires the following packages
    that were not found in your environment: einops" — a transitive dependency
    only the model's own code knows about. That sentence is the useful half, so
    the command reports it rather than rewording it, and returns a line rather
    than raising, because a person asked and is waiting for words.
    """
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})
    ctx.require(TEXT_INDEX).register(
        LoadableEmbedder(fails="requires the following packages ...: einops")
    )
    agent = _agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert said is not None and "did not load" in said
    assert said is not None and "einops" in said, "the actionable half of the message was dropped"


async def test_an_embedder_with_nothing_to_download_says_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An endpoint-backed embedder has no weights, and must not be made to pretend."""
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})
    ctx.require(TEXT_INDEX).register(HashingEmbedder())
    agent = _agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert said is not None and "needs no download" in said


async def test_the_command_is_absent_rather_than_broken_without_a_provider(
    mount: MountProfile, tmp_path: Path
) -> None:
    """It is still registered — a person may type it to find out *why* nothing works."""
    ctx = await mount({**ROW, "config": {"path": str(tmp_path / "index")}})
    agent = _agent(ctx)

    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)

    assert said is not None and "No embedder is registered" in said


async def test_preload_refuses_the_mount_rather_than_failing_mid_turn(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cookbook's rule, and the whole reason `preload` exists.

    An unattended run — a daemon, a scheduled tick, `ph -p` in CI — has nobody
    to type `/text-index install`, so without this the first thing it learns
    about an unreachable model is a failed tool call inside somebody's turn.
    `MountRefusal` is what every command that mounts a profile turns into a
    sentence and an exit code, rather than a traceback.

    The load is patched on the class rather than pointed at a bogus model id: a
    404 from the hub would make this a network test to prove a local branch.
    """
    from ph.cordis import MountRefusal
    from ph_text_index._embed import SentenceTransformerEmbedder

    def explode(self: Any) -> int:
        raise RuntimeError("requires the following packages ...: einops")

    monkeypatch.setattr(SentenceTransformerEmbedder, "load", explode)

    with pytest.raises(MountRefusal, match="asked to preload"):
        await mount(
            {**ROW, "config": {"path": str(tmp_path / "index")}},
            {
                "id": "text-index-local",
                "name": "text-index-local",
                "config": {"preload": True},
            },
        )


async def test_without_preload_the_mount_survives_a_model_that_cannot_load(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default, and why it is the default.

    A person at a TUI would rather the harness start now and pay for the model
    when they use it — and find out about a bad one from `/text-index install`,
    which is a sentence they asked for, rather than from a refused startup.
    """
    from ph_text_index._embed import SentenceTransformerEmbedder

    def explode(self: Any) -> int:
        raise RuntimeError("no network")

    monkeypatch.setattr(SentenceTransformerEmbedder, "load", explode)

    ctx = await mount(
        {**ROW, "config": {"path": str(tmp_path / "index")}},
        {"id": "text-index-local", "name": "text-index-local"},
    )
    agent = _agent(ctx)

    # Mounted, tools offered — and the command is where the truth comes out.
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None
    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)
    assert said is not None and "no network" in said


async def test_the_weights_land_under_the_cache_root(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not `~/.cache/huggingface`, which no pH root covers.

    Left to `sentence-transformers` a gigabyte of weights would sit somewhere
    `ph doctor` never mentions and `rm -rf $PH_CACHE` would not reclaim.
    """
    from ph_text_index import LocalConfig
    from ph_text_index._embed import SentenceTransformerEmbedder

    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    from ph.paths import resolve_roots

    default = LocalConfig()
    assert default.cache == "", "the row decides, so the field stays empty"
    expected = str(resolve_roots().cache / "models")

    built = SentenceTransformerEmbedder(model_name="m", cache_folder=expected)
    assert built.cache_folder == expected
    assert str(tmp_path) in built.cache_folder


# ------------------------------------------------------------------- the skill ----


async def test_the_skill_arrives_with_the_plugin(mount: MountProfile, tmp_path: Path) -> None:
    """What the RLM reads before it reaches for these tools.

    Registered on the **provider's** scope, so it is in the catalog exactly when
    the tools are — a procedure advertised for tools a deployment does not have
    is worse than no procedure, and `skills-progressive` ships an empty `paths`
    precisely so a skill is something a distribution installs on purpose (I7).
    """
    from ph.cordis import DEPLOYMENT

    ctx, _ = await _mounted(mount, tmp_path)

    names = {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}
    assert "text-search" in names

    skill = ctx.require(SKILLS).get("text-search", DEPLOYMENT)
    assert skill is not None
    assert "meaning" in skill.description
    # The tools it names are the ones this package registers.
    assert {"text_index", "text_search"} <= set(skill.allowed_tools)


async def test_only_the_description_rides_the_prompt(mount: MountProfile, tmp_path: Path) -> None:
    """G9: the catalog is in every request, the body is read when asked for."""
    from ph.cordis import DEPLOYMENT

    ctx, _ = await _mounted(mount, tmp_path)

    skill = ctx.require(SKILLS).get("text-search", DEPLOYMENT)
    assert skill is not None and len(skill.description) < 300
    body = ctx.require(SKILLS).body("text-search", DEPLOYMENT)
    assert body is not None and len(body) > 1_500, "the body is a page, and stays on disk"
    assert "text_search(query=" in body


# ------------------------------------------------- the version-control filter ----


@pytest.mark.needs_git
async def test_git_stops_an_unchanged_document_being_re_embedded(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The expensive saving.** This loop has no digest short-circuit of its
    own, so before the filter an unchanged document was re-read, re-cut and
    re-embedded on every call — measured at 20 s of MiniLM for `docs/`, or
    185 s under nomic.

    Counted on the embedder, because that is the cost: a test that watched the
    passage counts would pass against an implementation that re-embedded and
    then wrote the same vectors back.
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
    """The safe direction, and it must survive an *uncommitted* edit — which is
    `status`'s job, since git's index still holds the old blob id."""
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
    """No commit needed — jj snapshots the working copy on any command."""
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


async def test_a_document_the_index_never_held_is_not_vouched_for(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The guard in front of `vouches_for`, and why it cannot be skipped.**

    jj answers with no per-file ids, so `vouches_for` proves only that a path has
    not *changed* since the caller's token — which is equally true of a document
    that was never indexed at all. Index one file, then widen the call to the
    directory: without `store.holds`, every other document is "unchanged" on the
    strength of a token that was stored before they were ever looked at, and they
    stay out of the index permanently. Silent, and permanent, which is the pair
    that makes it worth a test of its own.
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


async def test_without_version_control_the_behaviour_is_what_it_was(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The filter is an optimisation, and losing it must cost only speed.

    This loop has no content digest, so with no backend an unchanged document is
    re-embedded exactly as it was before — stated as a test rather than left for
    someone to discover as a regression when they index a tarball.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "w.md").write_text(DOCUMENT, encoding="utf-8")
    ctx, embedder = await _mounted(mount, tmp_path)
    before = embedder.calls

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))
    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=_agent(ctx))

    assert first.value["chunks_added"] > 0
    assert again.value["unchanged"] == 0
    assert embedder.calls > before
