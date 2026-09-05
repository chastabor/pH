"""`tool-attach` — the model puts a file it found in front of its own eyes (P7-01).

The **model door** of the split I-9 draws, and the half that was missing: a person
attaches with `--attach`, which reads a path directly with the harness's own
permissions, while a model may attach only what `ctx.fs` already lets it read. So
this tool is a `ctx.fs.read_bytes` and a `save_bytes`, and every rule that bounds
a `read` — `permissions-fs`, the workspace tier, any `fs/read-intent` listener a
deployment adds — bounds it with nothing new to configure. A tool that reached
`save_path` instead would have been an exfiltration primitive with a friendly
name: attach a private key as a "document" and let a provider read it out.

**The media rides a context message, not the tool result.** A result's content
becomes a `tool-result` block, and both shipped wires flatten those to text —
Anthropic keeps only text blocks inside `tool_result`, OpenAI's `tool` role takes
a string — so an image returned as result content would have been dropped by the
renderer on the way out, silently, which is the exact failure this row exists to
end. `run.defer_context` is the mechanism that already exists for "this call
produced something the *conversation* should carry": the loop appends it after
`tool/result`, so call/result adjacency survives and the picture is in the next
request. What the result itself carries is the *account* — what was attached, how
big, and what it measured — which is what makes a replayed session render the
same card as the live one.

**Only what a provider ingests as content.** `MediaBlock` is deliberately not a
general file block, and this tool is where a model would otherwise discover that:
a `.zip` or a 40 MB CSV "attached" would be stored, sent, refused by the route and
degraded to a pointer — three layers of work to tell the model what this refusal
says in one sentence, and it names the tool that does read a CSV. What counts as
media is `ph.llm.media.is_attachable`, beside the accept policy rather than here:
it is a statement about what a `MediaBlock` may carry, and the other producer —
a person's `--attach` — is owed the same answer.

@module ph.tools.builtin.attach_tool
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ...llm.media import is_attachable
from ...llm.types import MediaBlock, PluginSource, create_user_message
from ...seams.attachments import OCTET_STREAM, mime_of
from ...wire import WireModel
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..errors import HarnessError
from ..presentation import simple_views

__all__ = ["MAX_ATTACH_BYTES", "apply"]

MAX_ATTACH_BYTES = 32 * 1024 * 1024
"""The default ceiling on one attached file, before any route is consulted.

Larger than either shipped route's own limit (5 MB at Anthropic, 20 MB on the
OpenAI wire) **on purpose**, because it is a different limit answering a different
question. `ResolvedModel.max_attachment_bytes` is "will this provider take it",
and `media-degrade` already answers that per request, per route, with a pointer
the model can read — a second copy of it here would refuse for the route the
session happens to be on and be wrong the moment it resumes on another. What this
bounds is the harness: the bytes are read into memory, hashed and written, so the
number that matters is the one past which a model can make the process swap by
naming a file. A stored blob over a route's ceiling is not wasted, either — the
store keeps originals (P7-03), so the same attachment is there for a route that
takes it.
"""

REACH_FOR_READ = (
    "attach is for media a model can be *shown* — an image, audio, video or a PDF. "
    "Use read for anything else; a text file, a CSV or a log is content you can read "
    "directly, and no provider takes an archive at all."
)
"""What the model is told when it attaches the wrong kind of file.

Names the tool that does work, because a refusal that only says no costs a turn:
the model's next move should be `read`, and it should not have to guess that."""


class Config(WireModel):
    """Row config for the attach tool."""

    max_bytes: int = MAX_ATTACH_BYTES
    """The largest file this tool will read into memory, in bytes.

    The harness's bound, not a route's: whether a provider will *take* the file is
    `ResolvedModel.max_attachment_bytes`, answered per request by `media-degrade`
    with a pointer the model can read. Lowering this refuses the read outright, so
    a deployment that sets it below its route's ceiling has quietly made a second
    accept policy."""


class AttachArgs(ToolModel):
    path: str = Field(
        description="Path to the media file. Relative paths resolve against the workspace."
    )


class AttachValue(ToolModel):
    path: str
    attachment_id: str
    mime: str
    bytes: int
    width: int | None = None
    height: int | None = None


def _render(args: Any, value: Any) -> Any:
    """The sentence the model reads, which has to say where the file went.

    A confirmation alone would leave the model to guess whether it is looking at
    the picture yet — it is not, quite: the media arrives in the *next* request,
    because a context message is appended after this result. Saying so is what
    stops a model attaching the same file twice, or answering about an image it
    has not been shown.
    """
    size = f"{value['mime']}, {value['bytes']} bytes"
    if value.get("width") and value.get("height"):
        size += f", {value['width']}x{value['height']}"
    return text_content(f"Attached {value['path']} ({size}). It follows this result.")


@plugin("tool-attach", config=Config, inject=["tools", "fs", "attachments"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the attach tool.

    `inject` names `attachments` as well as `fs`, so a profile that mounts no
    store never registers the tool at all — the `subagent-task` rule. A model
    offered `attach` in every prompt and told on every call that there is nowhere
    to put a file has been taught a capability the deployment does not have, and
    pays prompt tokens for the lesson on every request.
    """
    store = ctx.attachments

    async def attach(args: AttachArgs, run: ToolRunContext) -> Any:
        target = ctx.fs.resolve(args.path, agent=run.agent)
        mime = mime_of(target.name)
        if not is_attachable(mime):
            # Before the gate, and cheap: this is a statement about the argument,
            # decided from the name alone with nothing read and no policy asked —
            # the same ordering `UNKNOWN_TOOL` has ahead of policy (C6). An
            # unclassifiable extension reads as "no type", which is more useful
            # than the literal `application/octet-stream`.
            named = "no recognisable type" if mime == OCTET_STREAM else mime
            raise HarnessError(f"{args.path} is {named}. {REACH_FOR_READ}", "NOT_ATTACHABLE")
        content = await ctx.fs.read_bytes(
            args.path,
            scope=run.scope,
            agent=run.agent,
            session=run.session,
            max_bytes=config.max_bytes,
        )
        ref = await store.save_bytes(content=content, mime=mime, name=target.name)
        run.defer_context(
            create_user_message(
                content=[MediaBlock(attachment=ref)],
                # `relay`, the form a plugin uses for content it is carrying on
                # somebody's behalf, and never `{"kind": "user"}`: a person did
                # not attach this and the transcript must not say they did.
                source=PluginSource(plugin="tool-attach", form="relay"),
            )
        )
        return {
            "path": ctx.fs.named(target, agent=run.agent),
            "attachment_id": ref.attachment_id,
            "mime": ref.mime,
            "bytes": ref.bytes,
            "width": ref.width,
            "height": ref.height,
        }

    ctx.tools.register(
        define_tool(
            "attach",
            "Attach an image, audio file, video or PDF from the workspace so you can see it. "
            "The file arrives in the conversation after this call's result.",
            parameters=AttachArgs,
            output=ToolOutput(schema=AttachValue, render=_render),
            execute=attach,
            # **Not** `effects_confined_to_workspace`, and the default is right
            # rather than merely safe: the blob lands in `$PH_HOME/attachments`,
            # which a workspace restore does not undo and *must not* — the log
            # references it, so collecting it would make the session unresumable
            # (`ph attachments gc` is the only thing allowed to, and only for a
            # digest no session mentions). `/revert` listing it as uncovered is
            # therefore the true statement, not the cautious one.
            is_concurrency_safe=True,
            # It reads a file the model can already read and stores a copy where
            # only this harness looks. There is nothing here to take back, so
            # asking would be the over-asking `is_irreversible` warns about.
            is_irreversible=False,
            **simple_views("read", "Attach", "path"),
        )
    )
