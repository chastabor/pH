"""P6-33 — naming one event vocabulary or the other, and one namespace in it.

pH has two vocabularies that share a `namespace/name` spelling: 36 cordis bus
events and 63 session-log types. They overlap in **zero** names, yet six roots
appear in both and the log's `tool/*` sits one letter from the bus's `tools/*`.
That near-miss is the gate here, not a footnote: it is precisely what a substring
filter gets wrong, and the reason `ph.selectors` exists rather than a
`startswith`.

**Why the scheme is in the selector and not in the stored type names.**
Namespacing the data — prefixing every stored type with `event/` — was the
alternative, and was refused on three counts: 944 literal type strings across
107 files; a session-format bump that would make every stored log unreadable;
and it does not even fix the collision it is aimed at, since `event/tool` is
still one letter from `event/tools`. Nothing anywhere holds a lone type string
and asks "bus or log?" — the ambiguity only exists in the *question* a person
types, which is where the scheme was put instead.
"""

from __future__ import annotations

import pytest

from ph.cordis import import_plugin_modules
from ph.cordis.events import events as event_registry
from ph.selectors import (
    SCHEMES,
    Selector,
    SelectorError,
    matches_any,
    parse,
    parse_all,
    unknown_namespaces,
)
from ph.session import Session
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.testing import workspace_acquired, workspace_disposed, workspace_log

# ------------------------------------------------------------ the near-miss --


def test_a_namespace_never_matches_a_longer_one_that_starts_with_it() -> None:
    """**The row's gate.** `log:tool` must not select `tools/change`.

    The session log calls them `tool/*` and the cordis bus calls them `tools/*` —
    one letter apart, in two vocabularies that share no names. A substring test
    catches both, which is the whole reason matching here compares *segments*.
    """
    log_tool = parse("log:tool")
    assert log_tool.matches("tool/call")
    assert log_tool.matches("tool/code-dispatch-start")
    assert not log_tool.matches("tools/change"), "a substring match would take this"
    assert not log_tool.matches("tools/execute")

    bus_tools = parse("bus:tools")
    assert bus_tools.matches("tools/execute")
    assert not bus_tools.matches("tool/call")


def test_the_near_miss_is_real_in_the_shipped_vocabularies() -> None:
    """Not a hypothetical: both prefixes are occupied, in different registries.

    If this ever stops being true the gate above is still correct but no longer
    load-bearing, and whoever renamed one of them should find out here.
    """
    import_plugin_modules()
    bus = set(event_registry.names())
    assert {name for name in KNOWN_SESSION_EVENT_TYPES if name.startswith("tool/")}
    assert {name for name in bus if name.startswith("tools/")}
    assert not (bus & KNOWN_SESSION_EVENT_TYPES), "the two vocabularies share no names"


def test_the_two_vocabularies_share_roots_so_a_bare_pattern_is_ambiguous() -> None:
    """**The premise of the whole module**, asserted rather than asserted-in-prose.

    Six roots were shared when selectors were written — `agent`, `approval`,
    `fs`, `harness`, `llm`, `session` — which is why a bare `workspace` or
    `agent` typed into a filter does not say which vocabulary is meant, and why
    `parse` refuses an unscoped pattern with no default instead of picking one.
    Checked as a non-empty overlap rather than as that exact set, so adding an
    event does not fail this; if it ever empties, `SelectorError`'s "write
    log:x or bus:x" refusal has stopped earning its keep.
    """
    import_plugin_modules()
    bus_roots = {name.split("/")[0] for name in event_registry.names()}
    log_roots = {name.split("/")[0] for name in KNOWN_SESSION_EVENT_TYPES}

    assert bus_roots & log_roots, "a bare pattern would be unambiguous"
    assert {"agent", "session"} <= bus_roots & log_roots


# ------------------------------------------------------------------ parsing --


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        ("log:workspace", ("workspace",)),
        ("log:workspace/*", ("workspace",)),
        ("log:workspace/acquired", ("workspace", "acquired")),
        ("log:agent/inbox/spliced", ("agent", "inbox", "spliced")),
        ("log:*", ()),
        ("  log:workspace  ", ("workspace",)),
    ],
)
def test_a_trailing_star_means_the_namespace_itself(text: str, segments: tuple[str, ...]) -> None:
    """`workspace` and `workspace/*` are the same set.

    A namespace with nothing under it could not mean anything else, so accepting
    both spellings costs nothing and refusing one would be a rule to remember.
    """
    assert parse(text) == Selector(scheme="log", segments=segments)


def test_a_bare_pattern_takes_the_surfaces_vocabulary() -> None:
    """The terse form stays terse where the surface already knows."""
    assert parse("workspace", scheme="log") == Selector("log", ("workspace",))
    assert parse("tools", scheme="bus") == Selector("bus", ("tools",))
    assert parse("bus:tools", scheme="log").scheme == "bus", "an explicit scheme wins"


@pytest.mark.parametrize(
    ("text", "because"),
    [
        ("", "empty"),
        ("   ", "whitespace only"),
        ("nope:workspace", "unknown scheme"),
        ("log:", "a vocabulary and no namespace"),
        ("log:a//b", "an empty segment"),
        ("log:work*/acquired", "a star inside a segment"),
        ("log:*/acquired", "a star that is not trailing"),
    ],
)
def test_a_malformed_selector_is_refused_with_a_reason(text: str, because: str) -> None:
    """Refused, not coerced. A selector nobody can read is a filter that silently
    answers the wrong question, and these all have an obvious intended meaning
    that the parser must not guess at."""
    with pytest.raises(SelectorError):
        parse(text)


def test_an_unscoped_pattern_with_no_default_says_what_to_write() -> None:
    """The one refusal a person meets by typing the obvious thing, so it names
    both vocabularies rather than only complaining."""
    with pytest.raises(SelectorError, match="log:workspace or bus:workspace"):
        parse("workspace")


def test_the_scheme_error_names_what_is_accepted() -> None:
    with pytest.raises(SelectorError, match="expected one of log, bus"):
        parse("session:workspace")
    assert SCHEMES == ("log", "bus")


# ------------------------------------------------------- surface resolution --


def test_a_surface_refuses_a_vocabulary_it_cannot_serve() -> None:
    """**A refusal, not an empty result**, and the distinction is the point.

    `bus:tools` asked of a session log is a person querying the wrong surface.
    Answering "nothing matched" would let them conclude the log holds no tool
    events, which is false — it holds four `tool/*` types.
    """
    with pytest.raises(SelectorError, match='vocabulary; it holds "log" events'):
        parse_all(["bus:tools"], vocabulary="log")
    with pytest.raises(SelectorError, match='vocabulary; it holds "bus" events'):
        parse_all(["log:workspace"], vocabulary="bus")


def test_a_foreign_scheme_is_reported_among_accepted_ones() -> None:
    """One refusal listing every offender, rather than failing on the first: a
    person who mistyped two selectors should fix both in one pass."""
    with pytest.raises(SelectorError, match="bus:tools, bus:agent"):
        parse_all(["workspace", "bus:tools", "bus:agent"], vocabulary="log")


def test_no_selectors_means_no_filter() -> None:
    """An empty list is "the caller asked for nothing", which is every unfiltered
    read. The opposite default would make an absent `--type` return an empty log.
    """
    assert matches_any("anything/at-all", [])
    assert parse_all([], vocabulary="log") == []


# --------------------------------------------------------------- validation --


def test_an_unknown_namespace_is_reported_separately_from_matching() -> None:
    """`matches` never consults a vocabulary; this does.

    A stored log may legitimately carry types this build does not know — that is
    what `ignorable` is for — so a matcher that refused an unrecognised namespace
    would break reading a log written by a newer harness. Reporting is the opt-in
    half, for a command that would rather say "no such namespace" than return
    nothing and let the reader conclude there is nothing there.
    """
    known = {"workspace/acquired", "turn/start"}
    selectors = parse_all(["workspace", "wrokspace", "turn/start"], vocabulary="log")

    assert unknown_namespaces(selectors, known) == ["log:wrokspace"]
    assert all(one.matches("wrokspace/whatever") for one in selectors[1:2]), (
        "matching stays mechanical: an unknown namespace still matches its own prefix"
    )


def test_a_catch_all_is_never_reported_as_unknown() -> None:
    """`log:*` names no namespace, so it cannot name a wrong one."""
    assert unknown_namespaces(parse_all(["*"], vocabulary="log"), set()) == []


def test_a_typo_report_is_deduplicated_and_keeps_the_order_typed() -> None:
    """The report is read by the person who typed it, so: once each, in order.

    `workspace` and `workspace/*` parse to the same selector, so a caller who
    spells a namespace both ways has made one typo and should be told once. The
    order is the order they typed, because that is the list they will scan.
    """
    selectors = parse_all(["wrokspace", "nope", "wrokspace/*"], vocabulary="log")

    assert unknown_namespaces(selectors, {"workspace/acquired"}) == ["log:wrokspace", "log:nope"]


# ------------------------------------------------------------ Session.select --


def _log() -> Session:
    """Through `ph.testing`'s builders, so the payload shapes are not a second
    source of truth — the argument their own docstring makes."""
    return workspace_log(
        ("turn/start", {"turn": 1}),
        workspace_acquired("a", "/t"),
        workspace_disposed("a", kept=True),
        ("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    )


def test_select_answers_a_namespace_and_a_single_type() -> None:
    """The question `last_event_of` cannot ask.

    That one takes whole type names and answers with the newest; this takes a
    prefix and answers with all of them — so "every workspace record" is not
    spelled as a list of five literals a sixth type would silently fall out of.
    """
    session = _log()

    assert [event.type for event in session.select("workspace")] == [
        "workspace/acquired",
        "workspace/disposed",
    ]
    assert [event.type for event in session.select("workspace/acquired")] == ["workspace/acquired"]
    assert len(session.select()) == len(session.events), "no patterns is no filter"


def test_select_keeps_log_order_across_namespaces() -> None:
    """Several selectors are a union, and the log's order is the answer's order —
    an auditor reading two namespaces together is reading a timeline."""
    session = _log()

    assert [event.type for event in session.select("turn", "workspace/disposed")] == [
        "turn/start",
        "workspace/disposed",
        "turn/end",
    ]


def test_select_refuses_the_other_vocabulary() -> None:
    with pytest.raises(SelectorError):
        _log().select("bus:tools")


# ------------------------------------------------- the bus side, via the CLI --


def test_the_bus_vocabulary_is_selectable_without_cordis_knowing_about_selectors() -> None:
    """The registry has no `select` of its own, deliberately.

    One was written and removed: it was the **only** upward import in the whole
    of `ph/cordis/`, which is a port of a standalone meta-framework — and it
    dragged the name `"log"`, a vocabulary cordis has no business knowing exists,
    into it. Nothing called it either; `ph events` filters the matrix with
    `matches_any` directly, which is one comprehension and keeps the dependency
    pointing the way every other module in this package points it.

    So the bus half is exercised here at the level anything else would use, and
    the near-miss still holds from this side: `bus:tools` reaches none of the
    log's four `tool/*` types.
    """
    import_plugin_modules()
    selectors = parse_all(["tools"], vocabulary="bus")
    names = [name for name in event_registry.names() if matches_any(name, selectors)]

    assert names and all(name.startswith("tools/") for name in names)
    assert not any(name.startswith("tool/") for name in names)
    assert not hasattr(event_registry, "select"), "the method is gone, not merely unused"
