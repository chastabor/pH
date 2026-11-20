"""`clm-mirror`: the context of the model as a file that it edits with its ordinary tools.

Before each request, the surface is written to a file in the scratch directory
of the agent's workspace. The file has one `[[SECTION S<id>]]` block for each
section, and one `[[RESULT S<id>]]` block for each tool result inside a step.
The model edits the file with the tools that it already has: `edit`, `sed`, or
Python in a cell. When the tool call that changed the file finishes, the changes
are compiled into the verbs of the editor and landed in one batch:

| Change in a section | Result |
|---|---|
| Its block is gone, or its body is empty | A tombstone (consecutive ones become one marker) |
| A `RESULT` body changed, with the headers unchanged | That result is rewritten in place |
| The reply text above the results changed | The reply is rewritten in place; its calls stay |
| The `RESULT` headers themselves changed | The whole step is replaced by the text that is left |
| The text of a user message or of a revision changed | A replacement that stands for it |
| No `SECTION` lines are left | One replacement for everything editable |

**A body holds only text that the model may edit. The words of the harness go in
the header.** The label of a revision is its own block in the message
(`edits.labeled`). Examples of a label are "revised context, standing for
S412..S431" and the "removed from context: …" of a tombstone. The file shows the
label on the header line of the section, and the body below holds only the text
of the model. A revision that is edited again therefore says what it stands for
one time, and there is no pattern to strip.

**The file is read back in `tools/post-execute`, before the result of the call is
logged.** The edit therefore lands before the result that reports it. A crash
between the two leaves an edit that landed beside a call that did not finish,
which is what happened. If the file were read back on the next step instead, the
log could say that a write succeeded, and a restart could then lose its edit.
The receipt is appended to that result, so nothing else is added to the context
to say what changed.

**The file form of each node is folded one time** (the `SessionFoldCache` of
`MirrorService`): its body, already escaped, and its label. What a node shows
depends only on its event, so a render is a join. Read-back compares the raw
bodies in the file with the folded bodies, and unescapes only the bodies that
changed.

**The file is a cache of the log, never a source.** Its first line names the
session, the `replace_generation` of the surface, and the last section that the
file holds. While the generation is current, the sections up to that last
section are exactly what the log folds to now. Read-back therefore rebuilds the
base of the file from the log and does not remember a render. This is also what
makes it correct after a restart. Four rules follow:

* A file that was written against an older generation is refused.
* After every refusal, the file is rewritten from the log.
* A file that this process did not write is never applied. After a restart, the
  file on disk is rewritten from the log, because it may hold an edit that a
  crashed call left half done.
* An edit that no call made (the user in an editor, or another process) is
  caught before the next request and recorded as declined. It is never applied
  (decision 7).

**The file is private, and it does not outlive its session.** The `clm/`
directory has mode 0700 and the file has mode 0600, because the file is a whole
conversation. Both are removed when the session leaves the store. A link at
either path is replaced, never followed. If the row read through a link, a
refusal that quotes a damaged line would put the contents of another file in the
context of the model.

**The harness owns the file and writes it directly**, like the file of the spill
store, and not through `ctx.fs`. That seam controls what the *model* may touch.
If the render of the harness went through it, the permission rules of the user
would be applied on every step to a file that the model never chose to write.

@module ph_clm.mirror
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Literal

import anyio

from ph.agent.types import AgentHandle, RequestProposal
from ph.cordis import Context, Next, plugin
from ph.keys import SYSTEM_PROMPT, TOOLS
from ph.llm.types import LlmCallConfig, TextBlock, ToolCallBlock, ToolResultBlock
from ph.locks import KeyedLocks
from ph.paths import write_atomic
from ph.seams.invariants import contribute_fold_cache
from ph.seams.workspace import scratch_of
from ph.session import Session, SessionFoldCache, SessionTable, derive_event_message
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
"""The reason that a tombstone gives when it was made by deleting a section from the file."""

OUTSIDE = (
    "the context file changed outside any tool call — in an editor, or by another "
    "process — and only an edit a tool call makes is applied"
)
"""Why an edit that no call made was declined (decision 7)."""

Declined = Literal["mirror", "outside"]
"""How a declined edit arrived: a write to the file by a call, or a change that
no call made."""

LABEL_CHARS = 300
"""The maximum length of a revision label on a section header."""

_HEADER = re.compile(r"^\[\[(LIVE_CONTEXT|SECTION|RESULT)(?:\s+(.*?))?\]\]\s*$")
_ESCAPED = re.compile(r"^(\\*)(\[\[(?:LIVE_CONTEXT|SECTION|RESULT)\b)")
_ID = re.compile(r"^S(\d+)$")


# ------------------------------------------------------------------ the file --


@dataclass(frozen=True, slots=True)
class Shown:
    """What one node shows in the file. It depends only on the event of the node,
    so it is folded one time."""

    text: str
    """The body as the file holds it: the text that the model may edit, already escaped."""
    label: str
    """The label of a revision, for the header of its section. `""` for anything else."""
    result: bool
    answers: str
    """The call that a result answers. `""` for anything else."""
    calls: tuple[tuple[str, str], ...]
    """The `(call id, tool name)` pairs that a reply made."""


@dataclass(frozen=True, slots=True)
class Block:
    """One section as the file holds it, escaped: the text, then each result by its id."""

    id: int
    text: str
    results: tuple[tuple[int, str], ...] = ()

    def flattened(self) -> str:
        return "\n\n".join(part for part in (self.text, *(t for _, t in self.results)) if part)


def block_of(section: Section, shown: Mapping[int, Shown]) -> Block:
    """Build the file form of a section: its text, and its results when it is a step."""
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
    """Render the file for these sections. The output is deterministic, so
    read-back can rebuild it."""
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
    """Each section as written. It is still escaped, so it compares directly with
    the output of `block_of`."""


def parse(text: str) -> Parsed:
    """Read a context file back, or raise a refusal that names the first line that
    cannot be read.

    Bodies stay escaped. They are compared with the folded bodies as they are,
    and unescaped only where they changed (`_unescaped`).
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
    """A body as the file holds it. The lines are joined again with `\n` and the
    edges are trimmed. Each line that would read as a header gets one more
    backslash, which `_unescaped` removes again."""
    return "\n".join(_ESCAPED.sub(r"\\\1\2", line) for line in text.splitlines()).strip("\n")


def _unescaped(text: str) -> str:
    """Convert a body as written in the file back to the text that it stands for."""
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
    """Compile the drafts that a file asks for, against the base that it was
    written from.

    Refuses a file that was written against another session or an older
    generation, a reorder, a section that is not in the base, a change to a
    protected section, and text that belongs to no section.
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
    """The sections up to the last one that the file names. The file was written
    against these."""
    if through == "none":
        return []
    for position, section in enumerate(sections):
        if section.name == through:
            return list(sections[: position + 1])
    raise EditRefused(f"the context file names {through!r} as its last section, which is not one")


def _whole(editor: Editor, session: Session, base: Sequence[Section], text: str) -> list[Pending]:
    """Handle a file with no sections left: one revision for every editable section."""
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
    """The drafts for one section whose body changed. While the structure of the
    section is intact, its reply and its results are rewritten in place. When the
    structure is not intact, a replacement holds the text. Only the changed text
    is unescaped."""
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
    _written: SessionTable[dict[Path, tuple[int, int]]] = field(init=False)
    """The files of each session, with the `(mtime_ns, size)` at which this
    process last wrote each file. A call that left a file unchanged then costs
    one `lstat`, not a parse. A file that is not in this table was not written by
    this process. These are the files that are removed with the session
    (`_remove_all`, when the store releases it)."""
    _locks: KeyedLocks[Path] = field(default_factory=KeyedLocks)
    """Allows one use of a context file at a time. A lock is held only while the
    file is in use."""

    def __post_init__(self) -> None:
        self._shown = SessionFoldCache(self.fold, extend=self.extend)
        self._written = SessionTable(released=_remove_all)

    def fold(self, session: Session) -> dict[int, Shown]:
        return self.extend({}, session, 0)

    def extend(self, known: dict[int, Shown], session: Session, start: int) -> dict[int, Shown]:
        """Add to `known` what every surface event from `start` onward shows.
        `known` is changed in place."""
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

    def blocks(self, session: Session) -> list[Block]:
        """Every section as the file shows it now."""
        shown = self._shown.read(session)
        return [block_of(section, shown) for section in self.editor.sections(session)]

    def path(self, session: Session, agent: AgentHandle) -> Path:
        """The location of the context file of this agent: the scratch directory of
        its workspace. Every tier keeps that directory writable, and it is outside
        the tree that a repository tracks."""
        return scratch_of(self.ctx, session.id, agent) / "clm" / "context.md"

    async def refresh(self, session: Session, agent: AgentHandle) -> Path:
        """Before a request: catch an edit that no call made, then write the file
        again.

        The edit of each call was read back when the call ended. A file that
        changed since this process last wrote it was therefore changed by no call.
        That change is declined and recorded, and the file is overwritten
        (decision 7). A file that was deleted outside a call is only rewritten.
        """
        path = self.path(session, agent)
        async with self._lock(path):
            written = self._written.get(session, {}).get(path)
            if written is not None and _signature(path) not in (written, None):
                _decline(session, "outside", None, OUTSIDE)
            return await self.render(session, agent)

    async def render(self, session: Session, agent: AgentHandle) -> Path:
        """Write the file for the surface as it is now.

        **Only for a session that the store still holds.** The write runs on a
        thread, and the store can release the session while the write runs. If the
        file were recorded after that, it would be kept under a session that
        nothing would forget again, and it would stay in the scratch directory of
        the agent. The file is removed instead (Holdings rule 4: the record and
        the file are one holding, and the session leaving the store ended it).
        """
        path = self.path(session, agent)
        text = render(session, self.editor.sections(session), self._shown.read(session))
        written = await anyio.to_thread.run_sync(partial(_write, path, text))
        if session.disposed:
            await anyio.to_thread.run_sync(partial(_remove, path))
            return path
        self._written.setdefault(session, dict)[path] = written
        return path

    async def read_back(
        self, session: Session, agent: AgentHandle, *, call_id: str | None
    ) -> str | None:
        """Land what the model changed in the file, and return a description of
        what happened. Returns `None` when the model changed nothing."""
        path = self.path(session, agent)
        async with self._lock(path):
            # Done inline, not on a thread. A local `lstat` takes about a
            # microsecond, and every tool call makes this check.
            found = _signature(path)
            if found is None:
                return None
            written = self._written.get(session, {}).get(path)
            if written is None:
                # This process did not write the file. It is left over from before
                # a restart, and a call that the log shows as interrupted may have
                # half written it. Rewrite it, and never credit it to whichever
                # call happens to end first.
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
                self._written.setdefault(session, dict)[path] = found
                return None
            await self.render(session, agent)
            return f"[context file applied. {receipt(revisions)}]"

    def _lock(self, path: Path) -> AbstractAsyncContextManager[None]:
        return self._locks.hold(path)

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
    """The prompt section that tells the model that the file exists and how to read it."""
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
  them back, and `context_restore` undoes a section's last edit."""


@plugin("clm-mirror", affects="environment", inject=[CLM, TOOLS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: None) -> None:
    """Render the context file before each request and read it back after each call."""
    service = MirrorService(ctx=ctx, editor=ctx.require(CLM))
    ctx.provide(CLM_MIRROR, service)
    contribute_fold_cache(
        ctx, id="clm-mirror-text", subject="context file text", stale=service._shown.stale
    )

    async def before_request(
        proposal: RequestProposal, next_: Next[LlmCallConfig]
    ) -> LlmCallConfig:
        # Write the file after the rest of the chain. A listener that revises the
        # surface on the way to the request is then in the file that the model
        # edits: a paste that is offloaded, or a compaction at the request itself.
        config = await next_(proposal)
        await service.refresh(proposal.session, proposal.agent)
        return config

    async def after_call(
        execution: ToolExecution, result: ToolExecutionResult, next_: Next[PostToolDecision]
    ) -> PostToolDecision:
        # Handle only the call that the model made, not a dispatch inside a Code
        # Mode cell. The call of the cell itself ends after every write that the
        # cell made, including the writes of its Python.
        if execution.parent is not None or execution.session is None or execution.agent is None:
            return await next_(execution, result)
        # Land the edit before the rest of the chain runs, so that a listener
        # that measures the context (the readouts of `clm-context`) measures it
        # after the edit.
        said = await service.read_back(
            execution.session, execution.agent, call_id=execution.call_id
        )
        decision = await next_(execution, result)
        if said is None or not isinstance(decision, Accept):
            return decision
        # Append the receipt after the rest of the chain. A row that spills a
        # large result (`tool-result-offload`) then spills only what the tool
        # printed, and the receipt stays in the context.
        content = ctx.require(TOOLS).projected_content(execution, decision, result)
        if content is None:
            return decision
        return replace(decision, content=[*content, TextBlock(text=said)])

    # Both listeners are prepended, so they run outside every row that mounted
    # before this one. Each acts after the rest of its chain, so it must be
    # outside every row that revises the surface or the result. Those rows
    # (`compaction`, `input-offload`, `tool-result-offload`) never prepend, so
    # this holds in any mount order.
    ctx.on("agent/request", before_request, prepend=True)
    ctx.on("tools/post-execute", after_call, prepend=True)

    def section(request: AssembleContext) -> str:
        agent, session = request.agent, request.session
        if agent is None or session is None:
            return ""
        return protocol(service.path(session, agent))

    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(name="clm:mirror", order=ORDER_TOOL_GUIDANCE + 10, text=section)
    )


# ------------------------------------------------------------------ helpers --


def _remove(path: Path) -> None:
    """Remove a context file, and its directory when nothing else is in it."""
    path.unlink(missing_ok=True)
    with suppress(OSError):
        path.parent.rmdir()


def _remove_all(files: dict[Path, tuple[int, int]]) -> None:
    """Remove the context files of a session, when the record of them is released.

    This function is synchronous, because the disposal of the store is
    synchronous: one `unlink` and one `rmdir` for each agent. A directory that
    something else wrote into stays.
    """
    for path in files:
        _remove(path)


def _write(path: Path, text: str) -> tuple[int, int]:
    """Write the file and return what was written (`_signature`), in one thread
    hop. The write is not durable. The file is rebuilt from the log on the next
    request, and a sync before every model call would spend a disk barrier on a
    cache.

    The directory is created with mode 0700 and the file is written with mode
    0600. If the path of the directory holds anything that is not a directory,
    such as a link that the model planted, it is removed and not followed. A
    `chmod` through a link changes the file that the link names.
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
    """Write the auditor's copy of a file edit that was not applied (`clm/declined`)."""
    _LOG.append(session, DECLINED, {"via": via, "callId": call_id, "reason": reason})


def _signature(path: Path) -> tuple[int, int] | None:
    """The `(mtime_ns, size)` of the path itself, never of what a link names.
    Returns `None` when nothing is at the path."""
    try:
        found = os.lstat(path)
    except FileNotFoundError:
        return None
    return found.st_mtime_ns, found.st_size


def _read(path: Path) -> str:
    """Read the text of the file, never through a link. A link, a pipe, or
    anything that is not UTF-8 text is refused. Unlike the reason for a damaged
    line, the reason for this refusal quotes nothing from the file. The open is
    non-blocking, so a pipe that was put in place of the file cannot hold the
    thread."""
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
    """The tool name of each result, keyed by the id of the result. The name is
    that of the call that the result answers, as the reply named it."""
    facts = [
        (origin, shown[seq])
        for seq, origin in zip(section.nodes, section.origins, strict=True)
        if seq in shown
    ]
    names = dict(call for _, one in facts for call in one.calls)
    return {origin: names.get(one.answers, "") for origin, one in facts if one.result}


def _joined(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip("\n")
