"""The live gate for a local OpenAI-compatible server — llama.cpp, in practice.

**Opt-in, and the opt-in is the base URL**: nothing here runs unless
`LLAMA_BASE_URL` is set, so a developer with something unrelated listening on the
port never has this suite talk to it. That is the same bargain
`test_adapters.py::test_real_api_smoke` strikes with `DEEPSEEK_API_KEY`, minus
the part where the key is what makes it live.

    ./test.sh smoke              # reads the port from llama.yaml, probes /health first
    ./test.sh smoke -k cat       # one of them
    LLAMA_BASE_URL=http://127.0.0.1:9931/v1 uv run pytest tests/test_local_server.py -q

The gate is the documented door, and not only for the short `$TMPDIR`: it takes
the base URL out of `llama.yaml` — the same document these tests mount — so the
port lives in exactly one place, and it says "nothing answered /health" in one
line rather than raising a `ConnectError` inside a fixture.

What this covers that nothing else does:

* **that the route is hooked up at all.** Every other adapter test drives a fake
  `HttpClient` and asserts request *shape*; `ph doctor` mounts the profile and
  deliberately makes no provider call. Between them, a `baseUrl` pointing at
  nothing passes the whole suite.
* **that the prefix cache is actually read, by a real one.**
  `tests/test_prompt_cache.py` asserts `cacheReadTokens > 0` against a simulated
  Anthropic cache — its own docstring says so, and a simulation cannot fail for
  a wire that never reports the field. llama.cpp reports it as
  `prompt_tokens_details.cached_tokens`, which `_to_usage` already reads, so
  this is A12's prefix stability priced by a server rather than by a fixture.
* **that the window pH budgets against fits one slot**, which is the
  configuration mistake a 6-agent deployment actually makes.
* **that a picture arrives as a picture.** Both wire renderers used to flatten a
  media-only message to empty text, and `accepts` is a *claim* a route makes
  about itself — neither failure is visible without a model on the other end
  saying what it saw.

The model is *discovered* rather than configured, because `--model` names a
`.gguf` path on one host and an alias on the next, and a gate nobody can run
without editing it is not a gate. `LLAMA_MODEL` overrides for a server holding
more than one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from ph.agent.types import AgentOptions
from ph.keys import AGENTS, LLM, SESSIONS
from ph.llm.types import text_of
from ph.session import Session
from ph.session.json import as_obj
from ph.testing import MountProfile
from ph_app.attach import ingest, prompt_message
from ph_app.profiles import resolve_profile

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not os.environ.get("LLAMA_BASE_URL"),
        reason="LLAMA_BASE_URL is not set; the live local-server gate is opt-in",
    ),
]

BASE_URL = os.environ.get("LLAMA_BASE_URL", "")
PROBE_TIMEOUT = 30.0
"""For the two metadata endpoints only. Generation gets the adapter's own
timeout, which is minutes — a 27B model on a laptop takes them."""

CAT = Path(__file__).parents[1] / "samples" / "CatPic.jpg"
"""The picture the vision test attaches — a photograph, deliberately.

A generated square would test the wire just as well and would tell nobody
whether the *model* received an image, because "what animal is this" has no
answer for a test pattern. It is checked into the repo so the gate needs no
fixture step, and skipped rather than failed when absent, since a clone that
omitted it is not a defect in pH."""


def _get(url: str) -> Any:
    """One metadata read, or a failure that names the URL.

    A failure and not a skip: the person who set `LLAMA_BASE_URL` asked whether
    the server is reachable, and answering "skipped" to that question is how a
    hookup test comes to be green against nothing at all. The one endpoint whose
    absence is genuinely inconclusive is handled by its own reader below.
    """
    try:
        reply = httpx.get(url, timeout=PROBE_TIMEOUT)
        reply.raise_for_status()
        return reply.json()
    except httpx.HTTPError as error:
        pytest.fail(f"{url} did not answer: {error}")


def _server_root() -> str:
    """The server's own root, which is *not* the OpenAI base URL.

    llama.cpp serves the compatible wire under `/v1` and its own endpoints —
    `/props`, `/slots`, `/health` — beside it rather than under it, so one
    variable cannot address both without this. Derived rather than asked for as a
    second variable: two URLs for one server is how a gate comes to probe a
    different process than it prompts.
    """
    base = BASE_URL.rstrip("/")
    return base.removesuffix("/v1").rstrip("/")


def _props() -> Any | None:
    """llama.cpp's `/props`, or `None` when this server does not publish it.

    The one inconclusive answer in the file: a 404 here means "not llama.cpp, or
    started with the endpoint off", which is a fact about the server rather than a
    defect. Anything else — a refused connection, a 500 — is still a failure,
    because `/models` just answered and something is wrong if this cannot.
    """
    url = f"{_server_root()}/props"
    try:
        reply = httpx.get(url, timeout=PROBE_TIMEOUT)
        if reply.status_code == httpx.codes.NOT_FOUND:
            return None
        reply.raise_for_status()
        return reply.json()
    except httpx.HTTPError as error:
        pytest.fail(f"{url} did not answer: {error}")


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> str:
    """The profile's credential, defaulted — and the model to ask for.

    `LLAMA_API_KEY` is a formality llama.cpp ignores unless it was started with
    `--api-key`, but the route resolves it at the request edge like any other, so
    unset means `MISSING_CREDENTIAL` rather than an anonymous request. Defaulted
    here rather than required of the person, who already said which server.
    """
    if not os.environ.get("LLAMA_API_KEY"):
        monkeypatch.setenv("LLAMA_API_KEY", "local")
    configured = os.environ.get("LLAMA_MODEL")
    if configured:
        return configured
    models = _get(f"{BASE_URL.rstrip('/')}/models").get("data") or []
    if not models:
        pytest.fail(f"{BASE_URL} has no model loaded; /models listed none")
    return str(models[0]["id"])


def _options(model: str, *, max_tokens: int = 32) -> AgentOptions:
    """Small `max_tokens` by default, because what is under test is the round trip.

    The answer's *content* is not asserted anywhere above — a local server runs
    whatever model the person loaded, and a gate that expected a particular word
    would be a test of that model's instruction following. The vision test is
    the one exception and says why, and it is also why this is a parameter: a
    model that narrates before answering needs room to reach the answer.
    """
    return AgentOptions(provider="llama", model=model, max_tokens=max_tokens)


def _usage(session: Session) -> list[dict[str, Any]]:
    return [
        dict(as_obj(event.data["usage"]))
        for event in session.events
        if event.type == "assistant/message" and event.data.get("usage")
    ]


async def test_the_local_route_completes_a_turn(mount: MountProfile, route: str) -> None:
    """One real round trip through the shipped `llama` profile.

    `resolve_profile` rather than a hand-written overlay: what is under test
    includes the document, so a `baseUrl` or an `apiKeyEnv` that is wrong in
    `llama.yaml` has to fail here.
    """
    ctx = await mount(profile=resolve_profile("llama"))
    session = ctx.require(SESSIONS).create("local-smoke")
    agent = ctx.require(AGENTS).create(session, _options(route))

    await agent.prompt("Reply with the single word: ok")

    assert as_obj(session.events[-1].data["reason"])["kind"] == "completed"
    assert any(event.type == "assistant/chunk" for event in session.events)


UNPARSED_CALL = ("<tool_call>", "<function=", "<tools>")
"""Markers of a tool call that reached pH as *prose*.

Every one of these is a template's own syntax for a call the server was supposed
to parse into `tool_calls` and did not. They are searched for in the assistant's
text because that is where a mis-set parser puts them, and the failure is
otherwise invisible: the turn completes, the JSON is valid, and the harness
simply never sees a call."""


async def test_a_tool_call_comes_back_parsed(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    """The wire a harness actually lives on, and the one nothing else here checks.

    `test_the_local_route_completes_a_turn` passes against a server whose tool
    parser does not match its model: the model emits a call, the server hands it
    back as text, pH sees an ordinary answer and the turn *completes*. Measured —
    vLLM started `--tool-call-parser hermes` in front of a model that emits
    `<function=read><parameter=path>` XML returned the call as prose, and every
    gate in this file was green while the harness could not use the server at
    all. For an agent harness that is the most important fact about a local
    route, so it is asserted rather than assumed.

    **Two assertions, because a mismatched parser fails two different ways.**
    Non-streaming, it hands the call back untouched and the markers are in the
    text — the first assertion, which names the flag to fix. Streaming, which is
    what pH does, it consumes the tags it recognises and discards what it cannot
    parse, so the answer comes back *empty* and only the missing `tool/call`
    event says anything happened. Measured, both, against the same server.

    A model that answers without calling anything fails the second too, and
    deliberately: the prompt names the tool and the file, and a model that will
    not take that instruction is not one to run an agent on.

    `tmp_path` is the mounted root (`mount` pins `project=` to it), so the file
    the model is asked for is the file the `read` tool resolves.
    """
    (tmp_path / "note.txt").write_text("the secret word is marmalade\n", encoding="utf-8")
    ctx = await mount(profile=resolve_profile("llama"))
    session = ctx.require(SESSIONS).create("local-tools")
    agent = ctx.require(AGENTS).create(session, _options(route, max_tokens=256))

    await agent.prompt("Use the read tool to read the file note.txt, then say what it contains.")

    answer = _answer(session)
    leaked = [marker for marker in UNPARSED_CALL if marker in answer]
    assert not leaked, (
        f"the server returned a tool call as text, not as `tool_calls`: {leaked} appears in "
        f"the answer. Its tool parser does not match this model's template — for a Qwen3 "
        f"emitting `<function=...><parameter=...>` that is vLLM's `--tool-call-parser "
        f"qwen3_coder`, not `hermes`. Answer: {answer[:400]!r}"
    )
    assert any(event.type == "tool/call" for event in session.events), (
        "the model was given tools and asked to use one by name, and called nothing. "
        f"Answer: {answer[:400]!r}"
    )


async def test_thinking_does_not_arrive_as_the_answer(mount: MountProfile, route: str) -> None:
    """A reasoning block is a block, not prose with a close tag in it.

    pH maps `reasoning_content` to a `ReasoningBlock` — the adapter has done so
    since DeepSeek — so a server that splits thinking out is already handled. One
    that does not sends the whole `<think>…</think>` inline, and then the tag is
    in the *transcript*: it is replayed to the model on every following request,
    it is what a person reads back, and it is what `_answer` returns.

    A failure rather than a skip, which is where this parts company with the
    vision test. `--mmproj` absent means a capability this deployment does not
    have; a thinking tag in the answer means the log is now wrong about what the
    model said, and that is not something to be quiet about.
    """
    ctx = await mount(profile=resolve_profile("llama"))
    session = ctx.require(SESSIONS).create("local-reasoning")
    agent = ctx.require(AGENTS).create(session, _options(route, max_tokens=256))

    await agent.prompt("Say ok.")

    answer = _answer(session)
    assert "</think>" not in answer, (
        "the model's thinking arrived as the answer: the server is not separating it into "
        "`reasoning_content`, so the tag is now in the transcript and will be replayed to "
        "the model. vLLM wants `--reasoning-parser qwen3` (`deepseek_r1` on older builds); "
        f"llama.cpp wants `--reasoning-format deepseek`. Answer: {answer[:400]!r}"
    )


async def test_the_second_turn_reads_the_prefix_cache(mount: MountProfile, route: str) -> None:
    """A12, priced by the server: turn two re-reads turn one's prefix.

    **The first turn is not asserted to be a miss**, which is where this parts
    company with the simulated gate. A slot's KV cache outlives the process that
    filled it, so a fresh session whose system prompt and tool list match the
    last run's starts *warm* — measured at 1401 cached tokens on a first turn
    while writing this. That is the server behaving correctly, and a
    `"cacheReadTokens" not in first` assertion would fail on the second run of
    the day for the best possible reason.

    **A server that publishes no cache figures at all is inconclusive**, and is
    skipped for `_props`' reason rather than failed: llama.cpp and OpenAI report
    `prompt_tokens_details.cached_tokens` and DeepSeek reports
    `prompt_cache_hit_tokens`, but a vLLM build answering
    `"promptTokensDetails": null` says nothing about its prefix cache either way
    — and reading that silence as "the server is reprocessing every turn" points
    the message below at a defect that is not there. What is *not* skipped is a
    server that publishes the field and reports nothing read on turn two; that
    is the regression this gate exists for, and the two are told apart by
    whether the number is missing everywhere or zero here.

    It reached this file as a `KeyError` on the subscript, which is the worst of
    both: a failure whose own sentence never printed.
    """
    ctx = await mount(profile=resolve_profile("llama"))
    session = ctx.require(SESSIONS).create("local-cache")
    agent = ctx.require(AGENTS).create(session, _options(route))

    await agent.prompt("Say ok.")
    await agent.prompt("Say ok again.")

    usage = _usage(session)
    assert len(usage) >= 2, f"expected two answered turns, got {usage}"
    if not any("cacheReadTokens" in turn for turn in usage):
        pytest.skip(f"{_server_root()} reports no cached-token count on any turn; nothing to read")
    assert usage[1].get("cacheReadTokens", 0) > 0, (
        "the second turn re-read no prefix: the server is reprocessing the whole "
        "conversation every turn. Either prompt caching is off, or something "
        "moved the prefix — see docs/dev-notes/prefix-cache-benchmark.md"
    )


async def test_the_configured_window_fits_one_slot(mount: MountProfile, route: str) -> None:
    """`contextWindow` is one slot's, and llama.cpp is the one that knows.

    The mistake this exists for: `--ctx-size` is divided by `--parallel`, so
    `-c 1572864 --parallel 6` is 262144 per slot, and a profile that copied the
    server's total would let six agents each budget against the whole KV cache —
    six overflows that pH never saw coming, because nothing above the route
    disputes the number it was given.

    Skipped rather than failed on a server with no `/props`: this file is opt-in
    for anything speaking the OpenAI wire, and only llama.cpp publishes its slot
    geometry — beside `/v1` rather than under it, which is what `_server_root`
    is for.
    """
    # `or {}` so the skip below is the only place "no /props" is spelled: an
    # absent document and a document without the field are one answer here.
    props = _props() or {}
    per_slot = (props.get("default_generation_settings") or {}).get("n_ctx")
    if not isinstance(per_slot, int):
        pytest.skip(f"{_server_root()} publishes no per-slot n_ctx; not a llama.cpp server")

    ctx = await mount(profile=resolve_profile("llama"))
    model = ctx.require(LLM).resolve_model("llama", route)

    # `resolve_model` answers with an empty `ResolvedModel` when no adapter owns
    # the provider, so an absent window means either the profile set none or the
    # row never registered — both are this assertion's business.
    assert model.context_window is not None, (
        "no llama route published a context window; the profile sets none, or "
        "`llm-llama` did not register"
    )
    assert model.context_window <= per_slot, (
        f"the profile budgets against {model.context_window} tokens but each of this "
        f"server's {props.get('total_slots')} slots holds {per_slot} — "
        "llama.cpp divides --ctx-size by --parallel, and pH compacts against what it "
        "was told here"
    )


def _answer(session: Session) -> str:
    """What the person was shown — `print_mode`'s own extraction.

    The transcript rather than the model surface, and joined across messages
    because a model that thinks out loud lands more than one.
    """
    return "\n".join(
        text_of(message.content)
        for message in session.transcript()
        if message.role == "assistant" and text_of(message.content)
    )


async def test_the_model_sees_the_cat(mount: MountProfile, route: str) -> None:
    """The media path, end to end, with the model as the instrument.

    **This is the one test here that reads what the model said**, and the reason
    is that nothing else can distinguish the two failures it covers. A
    `MediaBlock` that the renderer flattens to text and an `accepts` list that
    claims a capability the server lacks both produce a *successful* turn:
    correct JSON, a plausible answer, no error anywhere. Only an answer that
    describes the wrong thing — or describes nothing — says the pixels never
    arrived.

    So the assertion is two-layered, deliberately:

    * **no `attachment/degraded` event** — deterministic, pH's own, and the half
      that fails when `llama.yaml`'s `accepts` and the server's `--mmproj` state
      disagree. This is the assertion to trust.
    * **the word** — generous (any of three), because "cat" is not a claim about
      instruction following the way "reply with exactly `ok`" would be. A model
      that gets this wrong while the first assertion passes is a model that
      cannot see, which is worth a failure even though the fix is not pH's.

    Skipped when the server declares no vision: `--mmproj` is a launch flag, not
    a property of pH, and a red suite for a server started without it would be a
    gap nobody could close from this repo.
    """
    if not CAT.is_file():
        pytest.skip(f"{CAT} is not in this checkout")
    modalities = (_props() or {}).get("modalities") or {}
    if not modalities.get("vision"):
        vision = modalities.get("vision")
        pytest.skip(f"{_server_root()} reports vision={vision!r}; start it with --mmproj")

    ctx = await mount(profile=resolve_profile("llama"))
    session = ctx.require(SESSIONS).create("local-vision")
    # The human door (I-9), which is what `ph -p --attach` uses: `ingest` reads
    # the path with the harness's own permissions, and `prompt_message` is the
    # message `prompted` builds — text first, then the media.
    refs = await ingest(ctx, [CAT])
    agent = ctx.require(AGENTS).create(session, _options(route, max_tokens=256))
    agent.followup(prompt_message("What animal is in this photograph?", refs))
    await agent.run()

    degraded = [event for event in session.events if event.type == "attachment/degraded"]
    assert not degraded, (
        "the attachment never reached the model as media: "
        f"{[dict(event.data) for event in degraded]}"
    )
    answer = _answer(session)
    assert any(word in answer.lower() for word in ("cat", "kitten", "feline")), (
        f"nothing in the answer names the animal in {CAT.name}, so the picture "
        f"probably did not arrive as one: {answer!r}"
    )
