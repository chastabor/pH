"""The rows of every picker, built from what the registries of pH report.

These are builders, not screens. Each one takes rows that say what exists and
turns them into `Choice` rows. The terminal asks its `FrontSession` for those
rows: `commands()` for the commands, `models()` for the models that the profile
of the session lists, `presets()` for the permission postures, and
`browse_sessions()` for the sessions. Nothing here keeps a parallel list of what
pH can do, so a plugin that registers a command or an adapter appears in the
picker without a change to this module (I1, I7).

Sessions are listed as a tree. Forks make them a tree, because the header
records `parent_session`. A flat list of ids hides which session came from
which, so children are indented under their parent. A session with no fork is a
tree with one node, and the list reads the same in both cases. The log of a
sub-agent also hangs under the agent that spawned it. It is marked as a
sub-agent, and it is shown but not offered: the mount of its root writes it, so
it is never attached (P11-08).

@module ph_app.tui.modals.pickers
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rich.filesize import decimal

from ph.seams.commands import CommandDefinition
from ph.seams.models import ModelEntry
from ph.seams.permission_presets import PresetSchema
from ph.text import one_line

from ...sessions import SessionSummary
from ..state import PromptRecord
from ..themes import ThemeCatalog, ThemeProfile
from .base import Choice

__all__ = [
    "NEW_SESSION",
    "command_choices",
    "history_choices",
    "model_choices",
    "preset_choices",
    "session_choices",
    "theme_choices",
]


def command_choices(definitions: Iterable[CommandDefinition]) -> list[Choice]:
    """Every registered slash command.

    This takes definitions and not the registry. The terminal asks its
    `FrontSession` for them. A front end in another process can therefore
    answer with a list from the wire, and needs no service that it cannot hold.
    """
    return [
        Choice(
            value=f"/{definition.name}",
            label=f"/{definition.name} {definition.argument_hint}".rstrip(),
            detail=definition.summary,
        )
        for definition in definitions
    ]


def preset_choices(presets: Sequence[PresetSchema]) -> list[Choice]:
    """The permission postures that the daemon offers, with the live one marked.

    The rows come from the projection, and not from `PRESETS` plus a name that
    is folded separately. The *mark* forced that. A client that attached after
    somebody switched had no event to fold, so it marked nothing. The rows come
    with the mark because a picker that asks one question should get one
    answer. `PRESETS` is a closed table today, so nothing here claims that a
    deployment could vary it.
    """
    return [
        Choice(
            value=preset.name,
            label=preset.name,
            detail=preset.summary,
            marked=preset.active,
        )
        for preset in presets
    ]


def theme_choices(active: str, catalog: ThemeCatalog, profile: ThemeProfile) -> list[Choice]:
    """Every theme that the catalog holds, in the order of the user.

    The `order` of the profile comes first, and the rest follow alphabetically.
    **Position is the only thing that the order says.** There is no second word
    for it. A list of seven themes does not need one, and `default` is already
    used for the thing that the profile actually records.

    Exactly one row therefore says `default`: the theme that this profile opens
    in. On a first run that is `DEFAULT_THEME`, and after a pick it is the pick.
    It is *not* the same row as the marked one. `marked` follows the cursor
    while the picker previews. During a preview the dot therefore says "this is
    what you are looking at", and `default` says "this is what you will get at
    the next launch".
    """

    def detail(name: str) -> str:
        source = "user" if name in catalog.user else "built-in"
        return f"default · {source}" if name == profile.chosen else source

    return [
        Choice(value=name, label=name, detail=detail(name), marked=name == active)
        for name in profile.ordered(catalog.names)
    ]


def history_choices(history: Sequence[PromptRecord]) -> list[Choice]:
    """The prompts that this user has sent, newest first.

    The **seq is the value**, not the text, and that lets the picker do two
    jobs. Choosing a row inserts its text. Moving the cursor onto a row reveals
    that row in the transcript. The second job needs a position in the log, and
    the text would not give one for a prompt that was sent twice.

    Rule 6 applies, because this is visible: `Choice.matches` searches the value
    together with the label. A filter of digits only can therefore match a row
    by its seq and not by anything in the prompt. That adds a row to the list
    and does not hide one, which is why it is named here and not worked around.

    `one_line` is the one answer for "the text of a person, on a row". The tool
    cards, the trajectory and `phern agents` use it. A pasted prompt of three
    hundred lines therefore renders like every other truncated row, with the
    ellipsis that says it was cut. The whole text is what is inserted.
    """
    return [
        Choice(
            value=str(record.seq),
            label=one_line(record.text, 72),
            detail=f"turn {record.turn}" if record.turn else "",
        )
        for record in history
    ]


def model_choices(
    entries: Sequence[ModelEntry], active_key: str, active_route: str
) -> list[Choice]:
    """The models that the profile of this session lists, by key, with the one in
    force marked.

    pH still keeps no catalog. The list belongs to the profile, which the user
    writes, so nothing here goes stale by itself. A key is the value because
    `/model` and `--model` take a key. The route beside the key is what the key
    means.

    A route can be in force although the profile does not list it, because a
    user named it. Such a route is a row of its own at the top, so the picker
    never marks a listed model as active when none is. The free-text entry
    names another model as `provider/model`.
    """
    rows = [
        Choice(
            value=entry.key,
            label=entry.key,
            detail=entry.route.label + (" · default" if entry.default else ""),
            marked=entry.key == active_key,
        )
        for entry in entries
    ]
    if active_route and not active_key:
        rows.insert(
            0, Choice(value=active_route, label=active_route, detail="in force", marked=True)
        )
    return rows


NEW_SESSION = ""
"""The value of the "start a new session" row, and what a dismissal answers.

It is empty on purpose. There are two ways to decline a stored session: choose
the first row, or press escape. Both arrive as the same answer, so the caller
has one branch and not two. `free_text` also cannot reach this value, because it
offers what was *typed*, and an empty value cannot be typed.
"""


def session_choices(
    sessions: Sequence[SessionSummary], *, current: str = "", offer_new: bool = False
) -> list[Choice]:
    """Stored sessions, newest first, with **branches** indented under their parent.

    Two kinds of parent link reach this list, and they must not render alike. A
    fork is a second conversation and belongs one level in. A segment is the
    *same* conversation in a new file. A session that rolled three times is
    therefore one row, and not a staircase of three rows that share one
    inherited title.

    The rule is: contract every segment edge and keep every fork edge. Each
    chain of segments collapses to its **tip**. The tip is the newest file, and
    it is the file that a resume should open. The user picks from one row for
    each conversation.

    A child whose parent is not on disk is shown at the root and not hidden. To
    lose a session from the list is worse than to show it in the wrong place.

    This function takes rows, and nothing to read them from. `sessions/browse`
    folds them on the harness: stored logs and live roots in one list, and each
    row carries the `state` that the daemon gives it. This function therefore
    only arranges. Set `offer_new` where declining is a real answer. That is
    the startup picker, where the first row starts a fresh session and is the
    row that `enter` lands on. `/sessions` leaves it off. There the question is
    which session to *switch to*, and escape already means "stay here".
    """
    summaries = {summary.session_id: summary for summary in sessions}
    under = _contract_segments(summaries)
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for session_id in summaries:
        if session_id not in under:
            continue  # a superseded segment; its tip carries the row
        parent = under[session_id]
        if parent is None:
            roots.append(session_id)
        else:
            children.setdefault(parent, []).append(session_id)

    rows: list[Choice] = []
    if offer_new:
        # This row is first, and is therefore selected already: `ChoicePicker`
        # lands on the first row when nothing is marked. `enter` therefore gives
        # the behavior that a user had before this picker existed.
        rows.append(
            Choice(
                value=NEW_SESSION,
                label="start a new session",
                detail="or pick one below to carry on",
            )
        )

    def walk(session_id: str, depth: int) -> None:
        summary = summaries[session_id]
        rows.append(_session_row(summary, depth, marked=session_id == current))
        for child in children.get(session_id, ()):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return rows


def _contract_segments(summaries: dict[str, SessionSummary]) -> dict[str, str | None]:
    """Every surviving row, mapped to the row that it branches from (`None` = a root).

    A segment chain is one conversation, so it gets one row: the tip. A
    superseded segment is marked by its absence from this map. A *branch* is
    mapped to a surviving row. A fork of an earlier segment therefore lands
    under the conversation that it came from, and not under a file that the list
    no longer shows.

    This is one map and not two. The two halves were only ever used together,
    and the caller had to compose them to ask either question.

    The branch parent is read from the **origin** of the chain. The origin is
    the only member whose parent link is a fork and not a continuation. A rolled
    session therefore stays exactly where it was before it rolled.
    """
    continues = {
        summary.parent: summary.session_id
        for summary in summaries.values()
        if summary.kind == "segment" and summary.parent in summaries
    }
    # This is hoisted. It was a linear scan of the values *inside* the loop
    # below, which made the contraction quadratic in the number of listed
    # sessions.
    continued = set(continues.values())
    represents: dict[str, str] = {}
    origin_of: dict[str, str] = {}
    for origin in summaries:
        if origin in continued:
            continue  # not a chain head; it is reached from its own origin
        chain, node = [origin], origin
        while (following := continues.get(node)) is not None and following not in chain:
            chain.append(following)
            node = following
        for member in chain:
            represents[member] = node
        origin_of[node] = origin
    # A segment *cycle* has no head, so nothing above reached it. Each member
    # then stands for itself and does not vanish from the list. The rule of this
    # view is that to lose a session is worse than to show it in the wrong
    # place. To drop the whole map because of a corrupt header is the worst
    # form of losing one.
    stranded = [one for one in summaries if one not in represents]
    for one in stranded:
        represents[one] = one
        origin_of[one] = one
    branched: dict[str, str | None] = {}
    for tip, origin in origin_of.items():
        parent = summaries[origin].parent
        # A cycle member is shown at the root. If it hung under its own partner,
        # both would be inside the children graph with nothing above them. The
        # render walk starts from roots, so the rows would not only be
        # misplaced: they would all disappear.
        branched[tip] = None if tip in stranded else represents.get(parent) if parent else None
    return branched


def _session_row(summary: SessionSummary, depth: int, *, marked: bool) -> Choice:
    child = summary.origin == "subagent"
    indent = f"{'  ' * depth}{('⤷ ' if child else '↳ ') if depth else ''}"
    # The size and mtime of a running session are whatever the last flush left,
    # so they would describe the file and not the conversation. The state is
    # what a user chooses on.
    detail = (
        "running" if summary.state == "running" else f"{summary.when} · {decimal(summary.size)}"
    )
    if summary.cwd:
        detail = f"{detail} · {summary.cwd}"
    return Choice(
        value=summary.session_id,
        label=f"{indent}{summary.title or summary.session_id}",
        detail=f"sub-agent · {detail}" if child else detail,
        marked=marked,
        # Shown, not offered. Attaching a child is refused (`NotARoot`), and a
        # picker that sent it anyway would exchange the row for an error. A
        # user can read the log of the child, and that needs no daemon.
        unavailable=(
            f"{summary.session_id} is a sub-agent of {summary.parent or 'another agent'}, "
            f"so its root's mount writes it and it cannot be opened here — read it with "
            f"`phern --mode trajectory --session {summary.session_id}`"
            if child
            else ""
        ),
    )
