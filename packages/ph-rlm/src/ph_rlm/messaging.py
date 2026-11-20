"""`rlm-messaging`: messages between agents, and the boundary on them (P3-12).

The `agent_message.send` of prime-agent traveled over the comm channel. The
nuclear-family check was therefore inside a handler, where nothing else could
see it. Here the send is a governed tool. The boundary can then be the strongest
mechanism that the pipeline offers, and the rate limit can be the weakest. The
two mechanisms are deliberately different.

**The family boundary is a `ctx.tools.guard` (C7).** A guard can only deny, it
runs *last*, and no later listener can turn its denial back into permission. So
there is no ordering in which a permissive row permits a send outside the family
again. The guard also uses the same resolution that `send` itself uses, from
`reachable_family`. The roster that the model reads and the rule that refuses a
send are therefore one implementation.

**The rate limit is not a guard.** A token bucket is backpressure, not a refusal.
The right response is to wait and send again. Under C3 a *denial* ends the whole
cell, so four messages in one second would cost the model its program. The
bucket therefore raises `ToolCallError` from the tool body, which the program
can handle. The deployment tunes the bucket in the row config. This is a
deliberate deviation from the plan, which says "rate limit as a
`tools/pre-execute` listener". The reason is the asymmetry that C3 creates
between a denial and a failure.

**Delivery is always steer.** A message reaches the target at its next *step*,
not its next turn, so a running agent picks it up without finishing first. A
target that is busy or already holds input reports `queued` and not `delivered`.
"It is in the queue" and "it is in the next request" are different facts, and
the sender can act on the difference.

@module ph_rlm.messaging
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
from pydantic import ValidationError

from ph.cordis import Context, plugin
from ph.json import JsonObject, as_obj, as_seq, thaw_json
from ph.keys import AGENTS, SESSION_PERSISTENCE, SESSIONS, SUBAGENTS, TOOLS
from ph.llm.types import ContentBlock, PluginSource, create_user_message, text_of
from ph.persistence import read_if_stored
from ph.seams.code_runtime import CodeBindingNamespace
from ph.seams.subagents import (
    STATUS,
    FamilyRole,
    StatusCause,
    reachable_family,
)
from ph.session import Session, SessionEvent, derive_event_message, session_written, settle_of
from ph.tools import (
    TOOL_DISPATCH_EVENT_TYPES,
    Done,
    NotDone,
    Reconciled,
    ToolExecution,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    Unknown,
    call_id_of,
    define_tool,
    recorded_arguments,
    text_content,
)
from ph.tools.code_mode import CodeBindingsRequest, ToolCallError, governed_binding
from ph.wire import WireModel

__all__ = [
    "MAX_MESSAGE_CHARS",
    "MAX_PENDING_PER_SESSION",
    "MESSAGE_NAMESPACE",
    "OBSERVE_NAMESPACE",
    "SEND_TOOL",
    "Config",
    "apply",
    "render_received",
    "replied_to_parent",
]

log = logging.getLogger("ph_rlm.messaging")

MESSAGE_NAMESPACE = "agent_message"
OBSERVE_NAMESPACE = "agent_observe"

SEND_TOOL = "agent_message_send"
LIST_TOOL = "agent_message_list_agents"
OBSERVE_LIST_TOOL = "agent_observe_list"
OBSERVE_GET_TOOL = "agent_observe_get"

MAX_MESSAGE_CHARS = 16_384
"""The `DEFAULT_AGENT_MESSAGE_MAX_CHARS` of prime-agent. A message is a message,
not a file transfer. A sibling that needs to hand over 200 KiB writes it to a
file and sends the path."""
MAX_PENDING_PER_SESSION = 20
"""How much unclaimed input one target may hold. Past this number the send
*fails*, and the program can handle the failure. The alternative is a queue that
grows and that the target will never work through."""

OUT_OF_REACH = "Agent reach is limited to parent, siblings, and children"
"""Ported verbatim. The model of prime-agent has already seen this sentence."""


class Config(WireModel):
    """Row config for `rlm-messaging`."""

    max_message_chars: int = MAX_MESSAGE_CHARS
    max_pending: int = MAX_PENDING_PER_SESSION
    rate_capacity: int = 3
    """The capacity of the token bucket for one sender and one target. The default
    is the capacity that prime-agent uses."""
    rate_refill_seconds: float = 1.0
    observe_max_messages: int = 40
    """The most messages from the transcript of another agent that one read may
    return. The bound makes the result offloadable like any other large tool
    result (C5). Without the bound, the result would arrive as unbounded cell
    output."""


class SendArgs(ToolModel):
    """`agent_message.send(...)`. Snake_case: the model types these."""

    message: str
    receiver_role: str = "parent"
    """`parent` | `child` | `sibling`. The argument is the role and not the id. A
    child then knows who to answer, although nobody told it an id."""
    receiver_name: str | None = None
    """Required when more than one family member holds the role."""


class ObserveArgs(ToolModel):
    agent_id: str
    limit: int = 20


class Receipt(WireModel):
    """What a send hands back. A receipt, not a reply."""

    delivery_status: str
    """`delivered` means that the message is in the next request of the target.
    `queued` means that the message waits behind work that the target has not
    finished. `recorded` is on a receipt that a resume rebuilt from the log of the
    receiver (L6b). That is all a log can say about a message that was sent before
    a crash."""
    receiver_id: str
    receiver_role: str
    message_id: str
    pending: int | None = None
    """How much unclaimed input the target now holds. A sender can use the number
    to slow down before it reaches the cap. `None` on a rebuilt receipt."""


@dataclass(slots=True)
class _Bucket:
    """One sender→target token bucket."""

    tokens: float
    checked_at: float


@dataclass(slots=True)
class _Limiter:
    """Token buckets per (sender, target). Backpressure, not policy."""

    capacity: int
    refill_seconds: float
    _buckets: dict[tuple[str, str], _Bucket] = field(default_factory=dict)

    def take(self, sender: str, target: str) -> bool:
        now = anyio.current_time()
        bucket = self._buckets.get((sender, target))
        if bucket is None:
            bucket = self._buckets[(sender, target)] = _Bucket(
                tokens=float(self.capacity), checked_at=now
            )
        elapsed = max(0.0, now - bucket.checked_at)
        bucket.checked_at = now
        if self.refill_seconds > 0:
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed / self.refill_seconds)
        if bucket.tokens < 1:
            return False
        bucket.tokens -= 1
        return True

    def forget(self, agent_id: str) -> None:
        """Drop every bucket that names `agent_id`, on either side. An agent that
        is gone sends and receives nothing more. Without this, a long-lived root
        kept one bucket for each child that it ever wrote to."""
        for pair in [pair for pair in self._buckets if agent_id in pair]:
            del self._buckets[pair]


def render_received(
    *, sender_label: str, sender_id: str, target_id: str, message_id: str, body: str
) -> str:
    """The text that the *target* sees. Ported from the rendering of prime-agent.

    The text is verbatim because it is the shape that a model has been trained to
    parse. The body is reproduced exactly. The framing belongs to the sender, and
    a harness that rewrote it would make the log stop saying what the target
    read.
    """
    return (
        f"[from {sender_label}]\n"
        "Agent-to-agent message received.\n"
        f"Source: agent_message\n"
        f"From: {sender_id}\n"
        f"To: {target_id}\n"
        f"Message id: {message_id}\n\n"
        f"{body}"
    )


def _label(role: FamilyRole, name: str) -> str:
    """The sender as the target sees it: `child:scout`, `parent`, `sibling:beta`."""
    return role if role in ("parent", "self") else f"{role}:{name}"


@plugin(
    "rlm-messaging",
    affects="environment",
    config=Config,
    inject=[TOOLS, SESSIONS, AGENTS, SUBAGENTS],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the send/observe tools, the family guard, and the two namespaces."""
    tools = ctx.require(TOOLS)
    limiter = _Limiter(capacity=config.rate_capacity, refill_seconds=config.rate_refill_seconds)
    # The id of an agent is the id of its session.
    ctx.on("session/disposed", lambda session: limiter.forget(session.id))

    def family(agent_id: str) -> dict[str, FamilyRole]:
        return reachable_family(ctx.require(SESSIONS).list(), agent_id)

    def resolve(sender_id: str, role: str, wanted: str | None) -> tuple[str | None, str]:
        """Return `(target_id, refusal)`. The guard and the body share this one
        resolution.

        The function returns the refusal and does not raise. The guard needs the
        reason as a string, and the body needs it as an error. A boundary and its
        error message drift apart when the decision is made twice.

        The function takes the two fields and not the payload. Both callers hold
        a validated `SendArgs`. To read the fields back out by string key was the
        one way in which the guard and the body could still disagree. A role that
        was neither a `str` nor absent read as one thing to a `str()` coercion
        and as another thing to a default.
        """
        reach = family(sender_id)
        candidates = [
            agent_id for agent_id, kind in reach.items() if kind == role and agent_id != sender_id
        ]
        if not candidates:
            offered = ", ".join(sorted({kind for kind in reach.values() if kind != "self"}))
            return None, (
                f'no family member has the role "{role}" '
                f"(reachable roles: {offered or 'none'}). {OUT_OF_REACH}."
            )
        if wanted is not None:
            named = [agent_id for agent_id in candidates if _named(ctx, agent_id, wanted)]
            if not named:
                return None, (
                    f'no {role} is named "{wanted}"; '
                    f"use agent_message.list_agents() to see who is reachable"
                )
            return named[0], ""
        if len(candidates) > 1:
            return None, (
                f'there are {len(candidates)} agents with the role "{role}"; '
                "pass receiver_name to say which"
            )
        return candidates[0], ""

    # ------------------------------------------------------------- the guard --
    #
    # C7. The guard can only deny and it runs last, so no later listener can
    # permit a send outside the family again. The boundary is not a policy that a
    # deployment may relax.

    def family_guard(execution: ToolExecution) -> str | None:
        if execution.name != SEND_TOOL:
            return None
        sender = execution.agent.id if execution.agent is not None else None
        if sender is None:
            return "an agent message needs a sending agent"
        asked = SendArgs.model_validate(thaw_json(execution.arguments))
        target, refusal = resolve(sender, asked.receiver_role, asked.receiver_name)
        if target is None:
            return refusal
        if family(sender).get(target) in (None, "self"):
            return OUT_OF_REACH
        return None

    tools.guard(family_guard)

    # ------------------------------------------------------------- the tools --

    def sender_of(tool: str, run: ToolRunContext) -> str:
        """Return who is calling. Refuse, and do not guess, when nobody is.

        Every tool here answers *relative to the caller*, because reach is
        computed from the place of the sender in the family. A call that arrives
        without an agent therefore has no question to answer. It does not have a
        default question. While `run.agent` was untyped, each tool read `.id`
        from it directly. A call without a caller was then an `AttributeError`
        raised from inside a body, and not a refusal that the program can handle.
        """
        sender = run.agent
        if sender is None:
            raise ToolCallError(tool, "this tool has to be called by an agent")
        return sender.id

    async def send(args: SendArgs, run: ToolRunContext) -> dict[str, Any]:
        sender_id = sender_of(SEND_TOOL, run)
        body = args.message.strip()
        if not body:
            raise ToolCallError(SEND_TOOL, "an agent message needs a body")
        if len(body) > config.max_message_chars:
            raise ToolCallError(
                SEND_TOOL,
                f"an agent message is at most {config.max_message_chars} characters; "
                f"this one is {len(body)}. Write the detail to a file and send the path.",
            )
        target_id, refusal = resolve(sender_id, args.receiver_role, args.receiver_name)
        if target_id is None:
            # The guard has already refused this. If control reaches here, the
            # guard was not mounted, and the message must still not be delivered.
            raise ToolCallError(SEND_TOOL, refusal)

        if not limiter.take(sender_id, target_id):
            raise ToolCallError(
                SEND_TOOL,
                f"sending to {target_id} too fast (limit {config.rate_capacity} per "
                f"{config.rate_refill_seconds:g}s). Do other work and send again — this is "
                "backpressure, not a refusal.",
            )

        target = ctx.require(AGENTS).get(target_id)
        if target is None:
            # A settled child kept its session and its log. It lost the agent
            # that holds an inbox. To wake it is the job of the seam (P3-13), and
            # to address a settled child is exactly the trigger that the plan
            # names. So a send to a finished child works, and the sender is not
            # told to go and read a transcript.
            await ctx.require(SUBAGENTS).ensure_addressable(target_id)
            target = ctx.require(AGENTS).get(target_id)
        if target is None:
            raise ToolCallError(
                SEND_TOOL,
                f"{target_id} is not running and could not be woken; its transcript is "
                "still readable with agent_observe.get()",
            )
        pending = len(target.inbox.next_turn) + len(target.inbox.next_step)
        if pending >= config.max_pending:
            raise ToolCallError(
                SEND_TOOL,
                f"{target_id} already holds {pending} unread messages (cap "
                f"{config.max_pending}); wait for it to work through them",
            )

        role = family(sender_id).get(target_id, "sibling")
        sender_role = family(target_id).get(sender_id, "sibling")
        message_id = relay_message_id(run.idempotency_key)
        target.steer(
            create_user_message(
                content=[
                    {
                        "type": "text",
                        "text": render_received(
                            sender_label=_label(sender_role, _display(ctx, sender_id)),
                            sender_id=sender_id,
                            target_id=target_id,
                            message_id=message_id,
                            body=body,
                        ),
                    }
                ],
                source=PluginSource(plugin="ph_rlm.messaging", form="relay"),
                # The id that the receipt names and the text shows, so that the
                # log of the receiver can be searched for it (`reconcile_send`,
                # L6b).
                message_id=message_id,
            )
        )
        # **On the disk of the receiver before the sender is told** (L6). `steer`
        # records the message in the log of the receiver. That record used to wait
        # for the next flush of the receiver. For an agent in a long tool call,
        # that flush comes after the flush of the sender. A crash in between left
        # the log of the sender saying `delivered` and the log of the receiver
        # saying nothing. With the receiver written first, the worst that a crash
        # leaves is the call of the sender unresolved, and repair reports that as
        # outcome unknown. The call is `written` and not `flush`, for the reason
        # that the daemon gives in `_acted`, which `_mutate` calls. The message is
        # already in the inbox of the receiver, so a write that fails must not
        # tell the sender that the message was not sent.
        if target.session is not None:
            await session_written(ctx, target.session)
        return Receipt(
            delivery_status="queued" if target.status == "running" or pending else "delivered",
            receiver_id=target_id,
            receiver_role=str(role),
            message_id=message_id,
            pending=pending + 1,
        ).to_wire()

    def list_agents(_args: object, run: ToolRunContext) -> dict[str, Any]:
        """Who this agent may address, and how each is related."""
        sender_id = sender_of(LIST_TOOL, run)
        rows = [
            {
                "agentId": agent_id,
                "role": role,
                "name": _display(ctx, agent_id),
                "running": ctx.require(AGENTS).get(agent_id) is not None,
            }
            for agent_id, role in sorted(family(sender_id).items())
            if role != "self"
        ]
        return {"agents": rows}

    def observe_list(_args: object, run: ToolRunContext) -> dict[str, Any]:
        return list_agents(_args, run)

    def observe_get(args: ObserveArgs, run: ToolRunContext) -> dict[str, Any]:
        """A bounded read of the transcript of another family member.

        The same rule limits reach as for a send. An agent that may not talk to
        another agent may not read it either. A bounded read is offloadable like
        any other large result, and does not arrive as cell output (C5).
        """
        sender_id = sender_of(OBSERVE_GET_TOOL, run)
        role = family(sender_id).get(args.agent_id)
        if role is None or role == "self":
            raise ToolCallError(OBSERVE_GET_TOOL, OUT_OF_REACH)
        session = ctx.require(SESSIONS).get(args.agent_id)
        if session is None:
            raise ToolCallError(OBSERVE_GET_TOOL, f"no session for {args.agent_id}")
        limit = max(1, min(args.limit, config.observe_max_messages))
        return {"agentId": args.agent_id, "role": role, "messages": _transcript(session, limit)}

    def stored_log(session_id: str) -> Sequence[SessionEvent] | None:
        """Return a log from the store: `()` when the store has none, and `None`
        when the log could not be read. This call blocks, because it reads and
        parses a whole log."""
        persistence = ctx.get(SESSION_PERSISTENCE)
        if persistence is None:
            return None
        try:
            stored = read_if_stored(persistence, session_id)
        except Exception:
            log.warning("ph_rlm.messaging: could not read %s to reconcile a send", session_id)
            return None
        return () if stored is None else stored[1]

    async def log_of(session_id: str) -> Sequence[SessionEvent] | None:
        """Return a log as a resume finds it: live, else stored, else empty. The
        stored read runs off the event loop, because a daemon resumes its roots
        concurrently."""
        live = ctx.require(SESSIONS).get(session_id)
        if live is not None:
            return live.events
        return await anyio.to_thread.run_sync(stored_log, session_id)

    async def candidates(sender: Session, role: str) -> tuple[list[str], bool]:
        """Return every session that a send in `role` could have reached, and
        whether that list is complete.

        This is not the resolution that the send itself uses. That resolution
        reads the live family, and after a restart most of the family is not live
        yet. Here:

        - A parent is named in the header of the sender.
        - Children are read from their own logs, live or stored
          (`load_children`). They are the children of the sender, or the children
          of the parent of the sender for a sibling. A child is filed with its
          parent, so both sets are in the family of the sender.
        - The siblings of a root are the other roots, and only the live roots
          stand for them. So a sibling send from a root can be found but never
          ruled out.

        The parent is the agent that spawned the sender (`delegating_parent`),
        not the log that a fork continues. A parent send from a forked root
        reached nobody, and the siblings of a forked root are the roots, as in
        `reachable_family`.
        """
        parent_id = sender.header.delegating_parent
        if role == "parent":
            return ([parent_id] if parent_id else []), True
        subagents = ctx.require(SUBAGENTS)
        family = sender.header.family
        if role == "child":
            children = await subagents.load_children(sender.id, family)
            return [state.session_id for state in children.values()], True
        if parent_id is None:
            live = ctx.require(SESSIONS).list()
            return [one.id for one in live if one.header.delegating_parent is None], False
        siblings = await subagents.load_children(parent_id, family)
        return [state.session_id for state in siblings.values()], True

    async def reconcile_send(
        arguments: Any,  # noqa: ANN401
        opened: SessionEvent,
        session: Session,
    ) -> Reconciled:
        """Did this send reach the log of its receiver? A resume asks this (L6b).

        The id of the message is derived from the call (`relay_message_id`). The
        log of the receiver is written before the sender is told (L6). So the
        message is in the log of a receiver exactly when the send happened.

        - Found: the send is done. The model is shown a receipt, and is not told
          to decide whether to send again.
        - Searched for in every session that the send could have reached, and
          absent from all of them: the send did not happen.
        - A log that could not be read, or a list of receivers that is not
          complete: the outcome stays unknown.
        """
        call_id = call_id_of(opened)
        try:
            asked = SendArgs.model_validate(arguments)
        except ValidationError:
            return Unknown()
        if not call_id:
            return Unknown()
        wanted = relay_message_id(f"{session.id}/{call_id}")
        targets, complete = await candidates(session, asked.receiver_role)
        for target_id in targets:
            events = await log_of(target_id)
            if events is None:
                complete = False
            elif any(_carries(event, wanted) for event in events):
                return Done(
                    Receipt(
                        delivery_status="recorded",
                        receiver_id=target_id,
                        receiver_role=asked.receiver_role,
                        message_id=wanted,
                    ).to_wire()
                )
        return NotDone() if complete else Unknown()

    tools.register(
        define_tool(
            SEND_TOOL,
            "Send a message to your parent, a sibling, or one of your children. It "
            "arrives at their next step; you keep working.",
            parameters=SendArgs,
            output=ToolOutput(schema=Receipt, render=_render_receipt),
            execute=send,
            reconcile=reconcile_send,
        )
    )
    tools.register(
        define_tool(
            LIST_TOOL,
            "The agents you may address: parent, siblings, children.",
            parameters={"type": "object", "properties": {}},
            output={"type": "object"},
            render=_render_agents,
            execute=list_agents,
            is_concurrency_safe=True,
        )
    )
    tools.register(
        define_tool(
            OBSERVE_LIST_TOOL,
            "The agents whose transcripts you may read.",
            parameters={"type": "object", "properties": {}},
            output={"type": "object"},
            render=_render_agents,
            execute=observe_list,
            is_concurrency_safe=True,
        )
    )
    tools.register(
        define_tool(
            OBSERVE_GET_TOOL,
            "The recent messages of one agent you may reach.",
            parameters=ObserveArgs,
            output={"type": "object"},
            render=_render_transcript,
            execute=observe_get,
            is_concurrency_safe=True,
        )
    )

    def message_namespace(request: CodeBindingsRequest) -> CodeBindingNamespace:
        return _namespace(
            ctx,
            request,
            MESSAGE_NAMESPACE,
            "message your parent, siblings and children; every send is governed",
            (("send", SEND_TOOL), ("list_agents", LIST_TOOL)),
        )

    def observe_namespace(request: CodeBindingsRequest) -> CodeBindingNamespace:
        return _namespace(
            ctx,
            request,
            OBSERVE_NAMESPACE,
            "read what agents you may reach have said",
            (("list", OBSERVE_LIST_TOOL), ("get", OBSERVE_GET_TOOL)),
        )

    tools.register_code_namespace(MESSAGE_NAMESPACE, message_namespace)
    tools.register_code_namespace(OBSERVE_NAMESPACE, observe_namespace)


# ----------------------------------------------------------------- helpers --


def _namespace(
    ctx: Context,
    request: CodeBindingsRequest,
    name: str,
    description: str,
    specs: tuple[tuple[str, str], ...],
) -> CodeBindingNamespace:
    view = ctx.require(TOOLS).view(request.scope)
    bindings = [
        governed_binding(request, public, definition.schema())
        for public, tool_name in specs
        # A tool that is restricted away for this agent is absent from the SDK
        # block too, so the prompt cannot offer what a cell could not call.
        if (definition := view.visible.get(tool_name)) is not None
    ]
    return CodeBindingNamespace(name=name, description=description, bindings=tuple(bindings))


def _named(ctx: Context, agent_id: str, wanted: str) -> bool:
    return _display(ctx, agent_id) == wanted or agent_id == wanted


def _display(ctx: Context, agent_id: str) -> str:
    """The name that the admission of an agent gives it, or its id when nobody
    named it.

    This is one line because the lookup is in the seam that owns the fold. The
    prompt needs the same answer. With two copies, a prompt could name one agent
    while a send delivers to another.
    """
    return ctx.require(SUBAGENTS).name_of(agent_id)


def _transcript(session: Session, limit: int) -> list[dict[str, Any]]:
    """The last `limit` model-visible messages of one session, as text."""
    rows: list[dict[str, Any]] = []
    for event in reversed(session.events):
        message = derive_event_message(event)
        if message is None:
            continue
        text = text_of(message.content).strip()
        if not text:
            continue
        rows.append({"role": message.role, "text": text})
        if len(rows) >= limit:
            break
    rows.reverse()
    return rows


def replied_to_parent(child: Session) -> bool:
    """Whether `child` sent its parent a message since it was last handed a task.
    That message is the reply that makes "completed without sending a reply"
    untrue of the child.

    **The answer is read from the log of the child** (Phase 11). There a send is
    the call that it was: a native `tool/call` and its `tool/result`, or the
    start and the settle of a Code Mode dispatch, each named for the tool. A send
    counts when it addressed the parent and settled without an error. A send that
    a crash cut off counts after its resume has reconciled it from the inbox of
    the parent (`reconcile_send`), which settles it as done. The answer used to
    be a flag on the in-memory child of the provider. A child that was readmitted
    after a restart had therefore forgotten a reply that it sent, and its parent
    was told that it never replied.

    **Since it was last handed a task**: a rehydration asks a settled child
    something new, so the start of the rehydration begins the count again. A
    resumed start continues the task that the child had, and a reply from before
    the restart still counts.
    """
    asked: set[str] = set()
    replied = False
    for event in child.events:
        if event.type == STATUS and event.data.get("cause") == _NEW_TASK:
            asked.clear()
            replied = False
        elif event.type in TOOL_DISPATCH_EVENT_TYPES:
            if event.data.get("name") == SEND_TOOL and _to_parent(event):
                asked.add(call_id_of(event))
        elif (settled := settle_of(event)) is not None:
            call_id, failed = settled
            if call_id in asked and not failed:
                replied = True
    return replied


_NEW_TASK: StatusCause = "rehydrated"
"""The start that hands a settled child a new task. The constant is typed, so a
rename of the causes of the seam fails the type check here. Without the type, the
rename would quietly end the reset."""


def _to_parent(call: SessionEvent) -> bool:
    """Whether the record of a send addressed the parent of the sender."""
    try:
        return SendArgs.model_validate(recorded_arguments(call)).receiver_role == "parent"
    except ValidationError:
        return False


def _render_receipt(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    pending = value.get("pending")
    return text_content(
        f"{value['deliveryStatus']} to {value['receiverId']} ({value['receiverRole']})"
        + ("" if pending is None else f"; {pending} pending")
    )


_RELAY_IDS = uuid.uuid5(uuid.NAMESPACE_URL, "ph_rlm.messaging")


def relay_message_id(call_key: str) -> str:
    """The id of the message that a send relays, derived from the call that sent
    it (L6b).

    `ToolRunContext.idempotency_key` is `{session}/{call id}`, and it includes the
    sub-call id of a Code Mode dispatch. The key is unique to one call and can be
    rebuilt from the record of the call. The `reconcile` of the send can
    therefore look for the message in the log of the receiver after a crash. The
    id is a UUID, the shape that `new_message_id` produces.
    """
    return str(uuid.uuid5(_RELAY_IDS, call_key))


def _carries(event: SessionEvent, message_id: str) -> bool:
    """Whether a record put this message in the inbox of its session."""
    return event.type == "agent/inbox/spliced" and any(
        as_obj(message).get("id") == message_id for message in as_seq(event.data.get("inserted"))
    )


def _render_agents(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    rows = value.get("agents") or []
    if not rows:
        return text_content("no reachable agents")
    return text_content(
        "\n".join(
            f"- {row['role']}: {row['name']} ({row['agentId']})"
            f"{'' if row.get('running') else ' [not running]'}"
            for row in rows
        )
    )


def _render_transcript(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    rows = value.get("messages") or []
    if not rows:
        return text_content(f"{value.get('agentId')} has said nothing")
    return text_content("\n".join(f"{row['role']}: {row['text']}" for row in rows))
