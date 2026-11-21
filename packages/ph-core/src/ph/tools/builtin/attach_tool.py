"""`tool-attach`: the model attaches a file that it found, to see it (P7-01).

This tool is the **model side** of the split that I-9 draws, and it is the side
that was missing. A user attaches with `--attach`, which reads a path directly
with the permissions of the harness. A model may attach only what `ctx.fs`
already lets it read. So this tool is a `ctx.fs.read_bytes` and a `save_bytes`.
Every rule that bounds a `read` also bounds this tool, with nothing new to
configure: `permissions-fs`, the workspace tier, and any `fs/read-intent`
listener that a deployment adds. A tool that called `save_path` instead would be
an exfiltration primitive with a friendly name: attach a private key as a
"document" and let a provider read it out.

**The media goes in a context message, not in the tool result.** The content of
a result becomes a `tool-result` block, and all three shipped wires flatten
those blocks to text. Anthropic keeps only the text blocks inside `tool_result`,
the `tool` role of OpenAI takes a string, and the `functionResponse` of Google
takes text. The renderer would therefore silently drop an image that was
returned as result content. That is the exact failure that this row exists to
end. `run.defer_context` is the existing mechanism for "this call produced
something that the *conversation* should carry". The loop appends the message
after `tool/result`, so the call and the result stay adjacent and the picture is
in the next request. The result itself carries the *account*: what was attached,
its size, and its measured dimensions. With that account, a replayed session
renders the same card as the live session.

**Only what a provider ingests as content.** `MediaBlock` is deliberately not a
general file block, and this tool is where a model would otherwise find that
out. A `.zip` or a 40 MB CSV that was "attached" would be stored, sent, refused
by the route and degraded to a pointer. That is three layers of work to tell the
model what the refusal of this tool says in one sentence, and the refusal names
the tool that does read a CSV. `ph.llm.media.is_attachable` decides what counts
as media. It is beside the accept policy and not here, because it is a statement
about what a `MediaBlock` may carry. The other producer, the `--attach` of a
user, must get the same answer.

@module ph.tools.builtin.attach_tool
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ...json import JsonObject
from ...keys import ATTACHMENTS, FS, TOOLS
from ...llm.media import is_attachable
from ...llm.types import ContentBlock, MediaBlock, PluginSource, create_user_message
from ...seams.attachments import OCTET_STREAM, mime_of
from ...wire import WireModel
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..errors import HarnessError
from ..presentation import simple_views

__all__ = ["MAX_ATTACH_BYTES", "apply"]

MAX_ATTACH_BYTES = 32 * 1024 * 1024
"""The default ceiling on one attached file, before any route is consulted.

The value is larger than the limit of every shipped route (5 MB at Anthropic,
20 MB on the OpenAI wire and at Google) **on purpose**. It is a different limit
that answers a different question. `ResolvedModel.max_attachment_bytes` answers
"will this provider take it". `media-degrade` already answers that question for
each request and each route, with a pointer that the model can read. A second
copy of that limit here would refuse for the route that the session happens to
be on, and would be wrong as soon as the session resumes on another route. This
value bounds the harness. The bytes are read into memory, hashed and written, so
the number that matters is the size past which a model can make the process swap
by naming a file. A stored blob over the ceiling of a route is not wasted
either. The store keeps originals (P7-03), so the same attachment is there for a
route that takes it.
"""

REACH_FOR_READ = (
    "attach is for media a model can be *shown* — an image, audio, video or a PDF. "
    "Use read for anything else; a text file, a CSV or a log is content you can read "
    "directly, and no provider takes an archive at all."
)
"""What the model is told when it attaches the wrong kind of file.

The text names the tool that does work, because a refusal that only says no
costs a turn. The next call of the model should be `read`, and the model should
not have to guess that."""


class Config(WireModel):
    """Row config for the attach tool."""

    max_bytes: int = MAX_ATTACH_BYTES
    """The largest file that this tool reads into memory, in bytes.

    This is the bound of the harness and not of a route. Whether a provider will
    *take* the file is `ResolvedModel.max_attachment_bytes`, and `media-degrade`
    answers that for each request with a pointer that the model can read. A lower
    value here refuses the read outright. A deployment that sets it below the
    ceiling of its route has therefore made a second accept policy without saying
    so."""


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


def _render(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    """The sentence that the model reads. It must say where the file went.

    A confirmation alone would leave the model to guess whether it can see the
    picture yet. It cannot yet: the media arrives in the *next* request, because
    a context message is appended after this result. The sentence says so. That
    stops a model from attaching the same file twice, or from answering about an
    image that it has not been shown.
    """
    size = f"{value['mime']}, {value['bytes']} bytes"
    if value.get("width") and value.get("height"):
        size += f", {value['width']}x{value['height']}"
    return text_content(f"Attached {value['path']} ({size}). It follows this result.")


@plugin("tool-attach", affects="environment", config=Config, inject=[TOOLS, FS, ATTACHMENTS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the attach tool.

    `inject` names `attachments` as well as `fs`, so a profile that mounts no
    store never registers the tool. This is the `subagent-task` rule. A model
    that is offered `attach` in every prompt, and is told on every call that
    there is nowhere to put a file, has been taught a capability that the
    deployment does not have. It also pays prompt tokens for that on every
    request.
    """
    store = ctx.require(ATTACHMENTS)

    async def attach(args: AttachArgs, run: ToolRunContext) -> dict[str, Any]:
        fs = ctx.require(FS)
        target = fs.resolve(args.path, agent=run.agent)
        mime = mime_of(target.name)
        if not is_attachable(mime):
            # This check runs before the gate, and it is cheap. It is a statement
            # about the argument, decided from the name alone, with nothing read
            # and no policy asked. `UNKNOWN_TOOL` has the same position ahead of
            # policy (C6). An extension that cannot be classified is reported as
            # "no type", which is more useful than the literal
            # `application/octet-stream`.
            named = "no recognizable type" if mime == OCTET_STREAM else mime
            raise HarnessError(f"{args.path} is {named}. {REACH_FOR_READ}", "NOT_ATTACHABLE")
        content = await fs.read_bytes(
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
                # `relay` is the form that a plugin uses for content that it
                # carries on behalf of somebody else. The source is never
                # `{"kind": "user"}`: a person did not attach this, and the
                # transcript must not say that one did.
                source=PluginSource(plugin="tool-attach", form="relay"),
            )
        )
        return {
            "path": fs.named(target, agent=run.agent),
            "attachment_id": ref.attachment_id,
            "mime": ref.mime,
            "bytes": ref.bytes,
            "width": ref.width,
            "height": ref.height,
        }

    ctx.require(TOOLS).register(
        define_tool(
            "attach",
            "Attach an image, audio file, video or PDF from the workspace so you can see it. "
            "The file arrives in the conversation after this call's result.",
            parameters=AttachArgs,
            output=ToolOutput(schema=AttachValue, render=_render),
            execute=attach,
            # **Not** `effects_confined_to_workspace`. The default is right, and
            # not only safe. The blob lands in `$PH_HOME/attachments`, which a
            # workspace restore does not undo and *must not* undo. The log
            # references the blob, so collecting it would make the session
            # unresumable. Only `phern attachments gc` may collect it, and only
            # for a digest that no session mentions. So when `/revert` lists the
            # call as not covered, that statement is true and not only cautious.
            is_concurrency_safe=True,
            # The tool reads a file that the model can already read, and stores a
            # copy where only this harness looks. There is nothing to take back.
            # A request for approval here would be the over-asking that
            # `is_irreversible` warns about.
            is_irreversible=False,
            **simple_views("read", "Attach", "path"),
        )
    )
