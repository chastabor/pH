"""`clm-mirror` — the model's context as a file it edits with its ordinary tools.

Before each request the surface is written to a file in the agent's workspace
scratch, one `[[SECTION S<id>]]` block per section and one `[[RESULT S<id>]]` block
per tool result inside a step. The model edits it with whatever it already has —
`edit`, `sed`, Python in a cell — and when the tool call that touched the file
finishes, its changes are compiled into the editor's verbs and landed in one batch:

| what changed in a section | becomes |
|---|---|
| its block is gone, or its body emptied | a tombstone (consecutive ones, one marker) |
| a `RESULT` body, headers unchanged | that result rewritten in place |
| the reply text above the results | the reply rewritten in place, its calls kept |
| the `RESULT` headers themselves | the whole step replaced by the text left |
| a user message's or a revision's text | a replacement standing for it |
| no `SECTION` lines at all | one replacement for everything editable |

**A body holds only text the model may edit; the harness's words go in the header.**
A revision's label ("revised context, standing for S412..S431", a tombstone's "removed
from context: …") is its own block in the message (`edits.labeled`), so the file shows
it on the section's header line and the body below holds the model's text alone. A
revision edited again says what it stands for once, with no pattern to strip.

**Read back in `tools/post-execute`, before the call's result is logged**, so the
edit lands before the result that reports it: a crash between the two leaves an
edit that landed beside a call that did not finish, which is what happened. Read
back on the next step instead, the log could say a write succeeded whose edit a
restart then lost. The receipt rides on that result, so nothing else is added to
the context to say what changed.

**Each node's file form is folded once** (`MirrorService`'s `SessionFoldCache`): its
body already escaped, and its label. What a node shows depends only on its event, so
a render is a join, and read-back compares the file's raw bodies with the folded ones
and unescapes only those that changed.

**The file is a cache of the log, never a source.** Its first line names the session,
the surface's `replace_generation` and the last section it holds. While the
generation stands, the sections up to that one are exactly what the log folds to
now, so read-back rebuilds the file's base from the log rather than remembering a
render — which is also what makes it right after a restart. A file written against
an older generation is refused, and every refusal is rewritten from the log. A file
this process did not write is never applied: after a restart, whatever is on disk is
rewritten from the log, since it may be an edit a crashed call left half done. And
an edit no call made — the person in an editor, another process — is caught before
the next request and recorded as declined, never applied (decision 7).

**Private, and no longer than its session.** The `clm/` directory is 0700 and the
file 0600, since the file is a whole conversation. Both go when the session leaves the
store. A link at either path is replaced, never followed: read through one, a refusal
quoting a damaged line would put another file's contents in the model's context.

**A harness-owned file, written directly** like the spill store's, not through
`ctx.fs`: that seam gates what the *model* may touch, and routing the harness's own
render through it would ask the person's permission rules about a file the model
never chose to write, on every step.

@module ph_clm.mirror
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Literal

import anyio

from ph.agent.types import AgentHandle, RequestProposal
from ph.cordis import Context, Next, plugin
from ph.keys import SYSTEM_PROMPT, TOOLS
from ph.llm.types import LlmCallConfig, TextBlock, ToolCallBlock, ToolResultBlock
from ph.paths import write_atomic
from ph.seams.invariants import contribute_fold_cache
from ph.seams.workspace import scratch_of
from ph.session import Session, SessionFoldCache, derive_event_message
from ph.session.writers import log_writer
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ph.text import one_line, thousands
from ph.tools.definition import Accept, PostToolDecision, ToolExecution, ToolExecutionResult

from .edits import Editor, Pending, Revision, labeled, node_text, receipt
from .keys import CLM, CLM_MIRROR
from .kinds import DECLINED
from .sections import EditRefused, Section, label

_LOG = log_writer(__name__)

__all__ = ["Block", "MirrorService", "apply", "parse", "render"]

REMOVED_IN_FILE = "removed in the context file"
"""The reason a tombstone made by deleting a section from the file gives."""

OUTSIDE = (
    "the context file changed outside any tool call — in an editor, or by another "
    "process — and only an edit a tool call makes is applied"
)
"""Why an edit no call made was declined (decision 7)."""

Declined = Literal["mirror", "outside"]
"""How a declined edit came: a call's write to the file, or a change no call made."""

LABEL_CHARS = 300
"""The most of a revision's label a section header carries."""

_HEADER = re.compile(r"^\[\[(LIVE_CONTEXT|SECTION|RESULT)(?:\s+(.*?))?\]\]\s*$")
_ESCAPED = re.compile(r"^(\\*)(\[\[(?:LIVE_CONTEXT|SECTION|RESULT)\b)")
_ID = re.compile(r"^S(\d+)$")


# ------------------------------------------------------------------ the file --


@dataclass(frozen=True, slots=True)
class Shown:
    """What one node shows in the file — a function of its event, so folded once."""

    text: str
    """Its body as the file holds it: the text the model may edit, already escaped."""
    label: str
    """A revision's label, for its section's header; `""` for anything else."""
    result: bool
    answers: str
    """The call a result answers; `""` for anything else."""
    calls: tuple[tuple[str, str], ...]
    """The `(call id, tool name)` pairs a reply made."""


@dataclass(frozen=True, slots=True)
class Block:
    """One section as the file holds it, escaped: the text, then each result by its id."""

    id: int
    text: str
    results: tuple[tuple[int, str], ...] = ()

    def flattened(self) -> str:
        return "\n\n".join(part for part in (self.text, *(t for _, t in self.results)) if part)


def block_of(section: Section, shown: Mapping[int, Shown]) -> Block:
    """What a section looks like in the file: its text, and its results if a step."""
    texts: list[str] = []
    results: list[tuple[int, str]] = []
    for seq, origin in zip(section.nodes, section.origins, strict=True):
        facts = shown.get(seq)
        if facts is None:
            continue
        if facts.result:
            results.append((origin, facts.text))
        elif facts.text:
            texts.append(facts.text)
    return Block(section.id, "\n\n".join(texts), tuple(results))


def render(session: Session, sections: Sequence[Section], shown: Mapping[int, Shown]) -> str:
    """The file for these sections — deterministic, so read-back can rebuild it."""
    through = sections[-1].name if sections else "none"
    generation = session.surface.replace_generation
    lines = [f"[[LIVE_CONTEXT session={session.id} generation={generation} through={through}]]"]
    for section in sections:
        facts = [section.kind, ",".join(section.calls), f"~{thousands(section.tokens)}"]
        facts += [f"after=~{thousands(section.after)}", "protected" if section.protected else ""]
        labels = [shown[seq].label for seq in section.nodes if seq in shown and shown[seq].label]
        facts += [f"— {one_line(' · '.join(labels), LABEL_CHARS)}" if labels else ""]
        lines += ["", f"[[SECTION {section.name} {' '.join(part for part in facts if part)}]]"]
        block = block_of(section, shown)
        if block.text:
            lines.append(block.text)
        names = _result_names(section, shown)
        for result_id, text in block.results:
            name = names.get(result_id, "")
            lines += [f"[[RESULT {' '.join(part for part in (label(result_id), name) if part)}]]"]
            if text:
                lines.append(text)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class Parsed:
    header: dict[str, str]
    lead: str
    """Any text before the first `[[SECTION]]` line, as written: escaped."""
    blocks: tuple[Block, ...]
    """Each section as written: escaped, so it compares with `block_of`'s as it is."""


def parse(text: str) -> Parsed:
    """A context file read back, or a refusal naming the first line that cannot be read.

    Bodies stay escaped: compared with the folded bodies as they are, and unescaped
    only where they changed (`_unescaped`).
    """
    lines = text.lstrip("\n").splitlines()
    first = _HEADER.match(lines[0]) if lines else None
    if first is None or first.group(1) != "LIVE_CONTEXT":
        raise EditRefused(
            "the context file's first line must stay the [[LIVE_CONTEXT …]] line it was "
            "written with"
        )
    header = dict(part.split("=", 1) for part in (first.group(2) or "").split() if "=" in part)
    lead: list[str] = []
    body = lead
    sections: list[tuple[int, list[str], list[tuple[int, list[str]]]]] = []
    for line in lines[1:]:
        found = _HEADER.match(line)
        if found is None:
            body.append(line)
            continue
        kind, rest = found.group(1), (found.group(2) or "").split()
        ident = _ID.match(rest[0]) if rest else None
        if kind == "LIVE_CONTEXT" or ident is None:
            raise EditRefused(f"the context file has a damaged header: {line.strip()}")
        number = int(ident.group(1))
        if kind == "SECTION":
            if any(number == seen for seen, _, _ in sections):
                raise EditRefused(f"{label(number)} appears twice in the context file")
            body = []
            sections.append((number, body, []))
        elif not sections:
            raise EditRefused(f"a [[RESULT]] line sits outside any section: {line.strip()}")
        else:
            body = []
            sections[-1][2].append((number, body))
    return Parsed(
        header=header,
        lead=_joined(lead),
        blocks=tuple(
            Block(number, _joined(text), tuple((rid, _joined(rtext)) for rid, rtext in results))
            for number, text, results in sections
        ),
    )


def _escaped(text: str) -> str:
    """A body as the file holds it: lines rejoined with `\n`, edges trimmed, and every
    line that would read as a header given one more backslash, which `_unescaped`
    takes back off."""
    return "\n".join(_ESCAPED.sub(r"\\\1\2", line) for line in text.splitlines()).strip("\n")


def _unescaped(text: str) -> str:
    """A body as written in the file, back to the text it stands for."""
    return "\n".join(
        line[1:] if (match := _ESCAPED.match(line)) is not None and match.group(1) else line
        for line in text.splitlines()
    )


# --------------------------------------------------------------- the compile --


def compile_drafts(
    editor: Editor,
    session: Session,
    sections: Sequence[Section],
    shown: Mapping[int, Shown],
    parsed: Parsed,
) -> list[Pending]:
    """The drafts a file asks for, against the base it was written from.

    Refuses a file written against another session or an older generation, a reorder,
    a section that is not in the base, a change to a protected section, and text that
    belongs to no section.
    """
    if parsed.header.get("session") != session.id:
        raise EditRefused("this context file belongs to another session")
    if parsed.header.get("generation") != str(session.surface.replace_generation):
        raise EditRefused(
            "the context was revised after this file was written, so its sections may "
            "not be the ones you see now"
        )
    base = _base(sections, parsed.header.get("through", ""))
    if not parsed.blocks:
        return _whole(editor, session, base, _unescaped(parsed.lead))
    if parsed.lead:
        raise EditRefused("text before the first [[SECTION]] line belongs to no section")
    order = {section.id: position for position, section in enumerate(base)}
    positions = []
    for written in parsed.blocks:
        if written.id not in order:
            raise EditRefused(
                f"{label(written.id)} is not a section of the context this file holds"
            )
        positions.append(order[written.id])
    if positions != sorted(positions):
        raise EditRefused("sections cannot be reordered; a revision lands where it stood")
    blocks = {block.id: block for block in parsed.blocks}
    drafts: list[Pending] = []
    edited: set[int] = set()
    gone: list[int] = []
    for position, section in enumerate(base):
        block = blocks.get(section.id)
        was = block_of(section, shown)
        if block == was:
            continue
        if section.protected is not None:
            raise EditRefused(f"{section.name} cannot be edited: {section.protected}")
        if block is None or not block.flattened():
            gone.append(position)
            continue
        edited.add(section.id)
        drafts += _changed(editor, session, base, position, shown, was, block)
    drafts += editor.tombstoning_runs(session, base, gone, REMOVED_IN_FILE, keep=frozenset(edited))
    return drafts


def _base(sections: Sequence[Section], through: str) -> list[Section]:
    """The sections up to the one the file names last — what it was written against."""
    if through == "none":
        return []
    for position, section in enumerate(sections):
        if section.name == through:
            return list(sections[: position + 1])
    raise EditRefused(f"the context file names {through!r} as its last section, which is not one")


def _whole(editor: Editor, session: Session, base: Sequence[Section], text: str) -> list[Pending]:
    """A file with no sections left: one revision for every editable section."""
    editable = [position for position, section in enumerate(base) if section.protected is None]
    if not editable:
        raise EditRefused("there is nothing in this context file you may edit")
    start, stop = editable[0], editable[-1] + 1
    if editable != list(range(start, stop)):
        raise EditRefused(
            "protected sections sit between the ones you may edit, so the file cannot be "
            "replaced whole; edit it section by section"
        )
    if text:
        return [editor.replacing(base, start, stop, text)]
    return [editor.tombstoning(session, base, start, stop, REMOVED_IN_FILE)]


def _changed(
    editor: Editor,
    session: Session,
    base: Sequence[Section],
    position: int,
    shown: Mapping[int, Shown],
    was: Block,
    block: Block,
) -> list[Pending]:
    """The drafts for one section whose body changed: its reply and results rewritten in
    place while their structure stands, a replacement holding the text once it does not.
    Only what changed is unescaped."""
    section = base[position]
    reply = next((seq for seq in section.nodes if seq in shown and not shown[seq].result), None)
    if (
        section.kind in ("step", "assistant")
        and reply is not None
        and [rid for rid, _ in block.results] == [rid for rid, _ in was.results]
    ):
        nodes = dict(zip(section.origins, section.nodes, strict=True))
        changes = [(reply, block.text)] if block.text != was.text else []
        changes += [
            (nodes[rid], text)
            for (rid, text), (_, before) in zip(block.results, was.results, strict=True)
            if text != before
        ]
        return [
            editor.rewriting_text(session, section, seq, _unescaped(text)) for seq, text in changes
        ]
    return [editor.replacing(base, position, position + 1, _unescaped(block.flattened()))]


# --------------------------------------------------------------- the service --


@dataclass(slots=True)
class MirrorService:
    """Renders the context file before each request and reads it back after each call."""

    ctx: Context
    editor: Editor
    _shown: SessionFoldCache[dict[int, Shown]] = field(init=False)
    _written: dict[str, dict[Path, tuple[int, int]]] = field(default_factory=dict)
    """Each session's files, by session id, with the `(mtime_ns, size)` this process
    last wrote each at: a call that left a file alone costs an `lstat`, not a parse;
    a file not here is not this process's; and these are what `forget` deletes."""
    _locks: dict[Path, anyio.Lock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._shown = SessionFoldCache(self.fold, extend=self.extend)

    def fold(self, session: Session) -> dict[int, Shown]:
        return self.extend({}, session, 0)

    def extend(self, known: dict[int, Shown], session: Session, start: int) -> dict[int, Shown]:
        """`known`, with what every surface event from `start` on shows — in place."""
        for event in session.events_from(start):
            message = None if event.surface_op is None else derive_event_message(event)
            if message is None:
                continue
            split = labeled(message)
            words, body = split if split is not None else ("", node_text(message))
            answers = next(
                (b.tool_call_id for b in message.content if isinstance(b, ToolResultBlock)), ""
            )
            known[event.seq] = Shown(
                text=_escaped(body),
                label=words,
                result=event.type == "tool/result",
                answers=answers,
                calls=tuple(
                    (b.id, b.name) for b in message.content if isinstance(b, ToolCallBlock)
                ),
            )
        return known

    def stale(self, sessions: Iterable[Session]) -> list[str]:
        """The fold cache's own drift check, for its invariant row."""
        return self._shown.stale(sessions)

    def blocks(self, session: Session) -> list[Block]:
        """Every section as the file shows it now."""
        shown = self._shown.read(session)
        return [block_of(section, shown) for section in self.editor.sections(session)]

    def path(self, session: Session, agent: AgentHandle) -> Path:
        """Where this agent's context file lives: its workspace scratch, which every
        tier keeps writable and outside the tree a repository tracks."""
        return scratch_of(self.ctx, session.id, agent) / "clm" / "context.md"

    async def refresh(self, session: Session, agent: AgentHandle) -> Path:
        """Before a request: catch an edit no call made, then write the file afresh.

        Every call's own edit was read back when the call ended, so a file that moved
        since this process last wrote it was moved by no call: declined and recorded,
        and overwritten (decision 7). A file deleted outside a call is only rewritten.
        """
        path = self.path(session, agent)
        async with self._lock(path):
            written = self._written.get(session.id, {}).get(path)
            if written is not None and _signature(path) not in (written, None):
                _decline(session, "outside", None, OUTSIDE)
            return await self.render(session, agent)

    async def render(self, session: Session, agent: AgentHandle) -> Path:
        """Write the file for the surface as it stands."""
        path = self.path(session, agent)
        text = render(session, self.editor.sections(session), self._shown.read(session))
        written = await anyio.to_thread.run_sync(partial(_write, path, text))
        self._written.setdefault(session.id, {})[path] = written
        return path

    async def read_back(
        self, session: Session, agent: AgentHandle, *, call_id: str | None
    ) -> str | None:
        """Land whatever the model changed in the file, and say what happened — `None`
        when it changed nothing."""
        path = self.path(session, agent)
        async with self._lock(path):
            # Inline: a local `lstat` is a microsecond, and every tool call asks.
            found = _signature(path)
            if found is None:
                return None
            written = self._written.get(session.id, {}).get(path)
            if written is None:
                # Not this process's: a leftover from before a restart, possibly half
                # written by a call the log shows interrupted. Rewritten, never
                # credited to whichever call happens to end first.
                await self.render(session, agent)
                return None
            if found == written:
                return None
            try:
                text = await anyio.to_thread.run_sync(partial(_read, path))
                revisions = self._apply(session, text, call_id)
            except EditRefused as refusal:
                _decline(session, "mirror", call_id, str(refusal))
                await self.render(session, agent)
                return (
                    f"[context file not applied: {refusal}. It has been rewritten from your "
                    "context as it stands; edit it again.]"
                )
            if not revisions:
                self._written[session.id][path] = found
                return None
            await self.render(session, agent)
            return f"[context file applied. {receipt(revisions)}]"

    def forget(self, session_id: str) -> None:
        """A session left the store: its files go, and everything kept about them.

        Synchronous, because `session/disposed` is: an `unlink` and an `rmdir` per agent.
        A directory something else wrote into stays.
        """
        for path in self._written.pop(session_id, {}):
            self._locks.pop(path, None)
            path.unlink(missing_ok=True)
            with suppress(OSError):
                path.parent.rmdir()
        self._shown.forget(session_id)

    def _lock(self, path: Path) -> anyio.Lock:
        return self._locks.get(path) or self._locks.setdefault(path, anyio.Lock())

    def _apply(self, session: Session, text: str, call_id: str | None) -> list[Revision]:
        drafts = compile_drafts(
            self.editor,
            session,
            self.editor.sections(session),
            self._shown.read(session),
            parse(text),
        )
        return self.editor.land(session, drafts, call_id=call_id, via="mirror") if drafts else []


# ------------------------------------------------------------------- the row --


def protocol(path: Path) -> str:
    """The prompt section that tells the model the file exists and how it reads."""
    return f"""## Your context, as a file

Before each request, the conversation you can see is written to `{path}`. Edit that
file with your ordinary tools to change your own context: the edit is applied when
the tool call that made it finishes, and that call's result says what changed.

- Keep the first line, `[[LIVE_CONTEXT …]]`, exactly as it is.
- Each section starts with a `[[SECTION S<id> …]]` line giving its kind, its size and
  the tokens after it. Ids never change. A revised section's line also says what it
  stands for; the text under it is what you may edit.
- Delete a section (its line and its text) to remove it; a one-line marker stays.
- Change the text under a line to rewrite it. Inside a step, change a `[[RESULT …]]`
  body, or the reply above the results; or delete the step's `[[RESULT]]` lines and
  write a summary in their place.
- Leave no `[[SECTION]]` lines after the first line to replace everything you may
  edit with the text you wrote.
- Sections marked `protected` cannot change, and sections cannot be reordered.
- Everything after an edit is read again on the next request: one batched edit near
  the end costs less than several small early ones, and a detailed summary costs
  little.
- A line starting `\\[[` stands for one starting `[[` inside a section's text.
- Only an edit one of your tool calls makes is applied; the file is rewritten before
  every request.
- The session log keeps every original; `context_recall` and `context_diff` read
  them back."""


@plugin("clm-mirror", affects="environment", inject=[CLM, TOOLS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: None) -> None:
    """Render the context file before each request and read it back after each call."""
    service = MirrorService(ctx=ctx, editor=ctx.require(CLM))
    ctx.provide(CLM_MIRROR, service)
    contribute_fold_cache(
        ctx, id="clm-mirror-text", subject="context file text", stale=service.stale
    )

    async def before_request(
        proposal: RequestProposal, next_: Next[LlmCallConfig]
    ) -> LlmCallConfig:
        # After the rest of the chain, so a listener that revises the surface on the
        # way to the request (a paste offloaded) is in the file the model edits.
        config = await next_(proposal)
        await service.refresh(proposal.session, proposal.agent)
        return config

    async def after_call(
        execution: ToolExecution, result: ToolExecutionResult, next_: Next[PostToolDecision]
    ) -> PostToolDecision:
        decision = await next_(execution, result)
        # The call the model made, not a dispatch inside a Code Mode cell: the cell's
        # own call ends after every write the cell made, its Python's included.
        if execution.parent is not None or execution.session is None or execution.agent is None:
            return decision
        said = await service.read_back(
            execution.session, execution.agent, call_id=execution.call_id
        )
        if said is None or not isinstance(decision, Accept):
            return decision
        content = ctx.require(TOOLS).projected_content(execution, decision, result)
        if content is None:
            return decision
        return replace(decision, content=[*content, TextBlock(text=said)])

    ctx.on("agent/request", before_request)
    ctx.on("tools/post-execute", after_call)
    ctx.on("session/disposed", lambda session: service.forget(session.id))

    def section(request: AssembleContext) -> str:
        agent, session = request.agent, request.session
        if agent is None or session is None:
            return ""
        return protocol(service.path(session, agent))

    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(name="clm:mirror", order=ORDER_TOOL_GUIDANCE + 10, text=section)
    )


# ------------------------------------------------------------------ helpers --


def _write(path: Path, text: str) -> tuple[int, int]:
    """Write, and say what was written (`_signature`), in one thread hop. Not durable:
    the file is rebuilt from the log on the next request, and syncing it before every
    model call would be a disk barrier spent on a cache.

    The directory is made 0700 and the file written 0600. Anything but a directory at
    the directory's path — a link the model planted — is removed rather than followed:
    a `chmod` through a link changes whatever it names.
    """
    directory = path.parent
    with suppress(FileNotFoundError):
        if not stat.S_ISDIR(os.lstat(directory).st_mode):
            directory.unlink()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    write_atomic(path, text, durable=False, private=True)
    found = os.lstat(path)
    return found.st_mtime_ns, found.st_size


def _decline(session: Session, via: Declined, call_id: str | None, reason: str) -> None:
    """The auditor's copy of a file edit not applied (`clm/declined`)."""
    _LOG.append(session, DECLINED, {"via": via, "callId": call_id, "reason": reason})


def _signature(path: Path) -> tuple[int, int] | None:
    """`(mtime_ns, size)` of the path itself, never what a link names; `None` when
    nothing is there."""
    try:
        found = os.lstat(path)
    except FileNotFoundError:
        return None
    return found.st_mtime_ns, found.st_size


def _read(path: Path) -> str:
    """The file's text, never through a link: a link, a pipe, or anything but UTF-8
    text is a refusal — whose reason, unlike a damaged line's, quotes nothing from the
    file. Non-blocking, so a pipe put in the file's place cannot hold the thread."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise EditRefused(
            "the context file was replaced by a link or something that is not a file, "
            "which is never followed"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EditRefused("the context file was replaced by something that is not a file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read()
    finally:
        os.close(fd)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EditRefused("the context file is no longer UTF-8 text") from error


def _result_names(section: Section, shown: Mapping[int, Shown]) -> dict[int, str]:
    """Each result's tool name, by the result's id: the call it answers, as its reply
    named it."""
    facts = [
        (origin, shown[seq])
        for seq, origin in zip(section.nodes, section.origins, strict=True)
        if seq in shown
    ]
    names = dict(call for _, one in facts for call in one.calls)
    return {origin: names.get(one.answers, "") for origin, one in facts if one.result}


def _joined(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip("\n")
