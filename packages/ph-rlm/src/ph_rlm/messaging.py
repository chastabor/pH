"""`rlm-messaging` — agent-to-agent messages, and the boundary on them (P3-12).

Prime Agent's `agent_message.send` traveled over the comm channel, so the
nuclear-family check lived inside a handler where nothing else could see it.
Here the send is a governed tool, which lets the boundary be the strongest thing
the pipeline offers and the rate limit be the weakest — and those are
deliberately different mechanisms:

**The family boundary is a `ctx.tools.guard` (C7).** Guards are deny-only, run
*last*, and cannot be turned back into permission by any later listener — so
there is no ordering in which a permissive row re-permits a send outside the
family. It also covers the same resolution `send` itself uses, from
`reachable_family`, so the roster the model reads and the rule that refuses it
are one implementation.

**The rate limit is not.** A token bucket is backpressure, not a refusal: the
right response is to wait and send again. Under C3 a *denial* ends the whole
cell, which would mean four messages in one second costs the model its program.
So the bucket raises `ToolCallError` from the tool body — the program's to
handle — and the deployment tunes it by row config. This is a deliberate
deviation from the plan's "rate limit as a `tools/pre-execute` listener", and the
reason is exactly the asymmetry C3 creates between a denial and a failure.

**Delivery is always steer.** A message reaches the target at its next *step*,
not its next turn, so a running agent picks it up without finishing first. A
target that is busy or already holding input reports `queued` rather than
`delivered`, because "it is in the queue" and "it is in the next request" are
different facts and the sender can act on the difference.

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
from ph.json import JsonObject, as_obj, as_seq, as_str, thaw_json
from ph.keys import AGENTS, SESSION_PERSISTENCE, SESSIONS, SUBAGENTS, TOOLS
from ph.llm.types import ContentBlock, PluginSource, create_user_message, text_of
from ph.seams.code_runtime import CodeBindingNamespace
from ph.seams.subagents import (
    FamilyRole,
    reachable_family,
    roster_of,
    subagent_roster,
)
from ph.session import Session, SessionEvent, derive_event_message, session_written
from ph.tools import (
    Done,
    NotDone,
    Reconciled,
    ToolExecution,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    Unknown,
    define_tool,
    text_content,
)
from ph.tools.code_mode import CodeBindingsRequest, ToolCallError, governed_binding
from ph.wire import WireModel

from .keys import RLM_CHILDREN

__all__ = [
    "MAX_MESSAGE_CHARS",
    "MAX_PENDING_PER_SESSION",
    "MESSAGE_NAMESPACE",
    "OBSERVE_NAMESPACE",
    "Config",
    "apply",
    "render_received",
]

log = logging.getLogger("ph_rlm.messaging")

MESSAGE_NAMESPACE = "agent_message"
OBSERVE_NAMESPACE = "agent_observe"

SEND_TOOL = "agent_message_send"
LIST_TOOL = "agent_message_list_agents"
OBSERVE_LIST_TOOL = "agent_observe_list"
OBSERVE_GET_TOOL = "agent_observe_get"

MAX_MESSAGE_CHARS = 16_384
"""Prime Agent's `DEFAULT_AGENT_MESSAGE_MAX_CHARS`. A message is a message, not a
file transfer: a sibling that needs to hand over 200 KiB writes it down and sends
the path."""
MAX_PENDING_PER_SESSION = 20
"""How much unclaimed input one target may hold. Past this the send *fails* —
the program's to handle — rather than growing a queue the target will never
work through."""

OUT_OF_REACH = "Agent reach is limited to parent, siblings, and children"
"""Ported verbatim: the sentence prime-agent's model has already seen."""


class Config(WireModel):
    """Row config for `rlm-messaging`."""

    max_message_chars: int = MAX_MESSAGE_CHARS
    max_pending: int = MAX_PENDING_PER_SESSION
    rate_capacity: int = 3
    """Token bucket per sender→target. Prime Agent's capacity."""
    rate_refill_seconds: float = 1.0
    observe_max_messages: int = 40
    """How much of another agent's transcript one read may return. Bounded so the
    result is offloadable like any other large tool result (C5) rather than
    arriving as unbounded cell output."""


class SendArgs(ToolModel):
    """`agent_message.send(...)`. Snake_case: the model types these."""

    message: str
    receiver_role: str = "parent"
    """`parent` | `child` | `sibling`. The role, not the id, because that is how a
    child knows who to answer without being told an id it never saw."""
    receiver_name: str | None = None
    """Required when more than one family member holds the role."""


class ObserveArgs(ToolModel):
    agent_id: str
    limit: int = 20


class Receipt(WireModel):
    """What a send hands back. A receipt, not a reply."""

    delivery_status: str
    """`delivered` — in the target's next request — or `queued` behind work it
    has not finished. `recorded` on a receipt a resume rebuilt from the receiver's
    log (L6b), which is all a log can say about a message sent before a crash."""
    receiver_id: str
    receiver_role: str
    message_id: str
    pending: int | None = None
    """How much unclaimed input the target now holds, so a sender can slow down
    before it hits the cap. `None` on a rebuilt receipt."""


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


def render_received(
    *, sender_label: str, sender_id: str, target_id: str, message_id: str, body: str
) -> str:
    """The text the *target* sees. Ported from prime-agent's rendering.

    Verbatim because it is the shape a model has been trained to parse, and the
    body is reproduced exactly — framing is the sender's, and a harness that
    rewrote it would make the log stop saying what the target read.
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
    """`child:scout`, `parent`, `sibling:beta` — the sender as the target sees it."""
    return role if role in ("parent", "self") else f"{role}:{name}"


@plugin("rlm-messaging", config=Config, inject=[TOOLS, SESSIONS, AGENTS, SUBAGENTS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the send/observe tools, the family guard, and the two namespaces."""
    tools = ctx.require(TOOLS)
    limiter = _Limiter(capacity=config.rate_capacity, refill_seconds=config.rate_refill_seconds)

    def family(agent_id: str) -> dict[str, FamilyRole]:
        return reachable_family(ctx.require(SESSIONS).list(), agent_id)

    def resolve(sender_id: str, role: str, wanted: str | None) -> tuple[str | None, str]:
        """`(target_id, refusal)` — the one resolution the guard and the body share.

        Returning the refusal rather than raising, because the guard needs the
        reason as a string and the body needs it as an error; deciding twice is
        how a boundary and its error message drift apart.

        Takes the two fields rather than the payload: both callers hold a
        validated `SendArgs`, so reading them back out by string key was the one
        way the guard and the body could still disagree — a role neither `str`
        nor absent read as one thing to a `str()` coercion and another to a
        default.
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
    # C7. Deny-only and last, so no later listener can re-permit a send outside
    # the family — the boundary is not a policy a deployment may relax.

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
        """Who is calling, refusing rather than guessing when nobody is.

        Every tool here answers *relative to the caller* — reach is computed from
        the sender's place in the family — so a call arriving without an agent
        has no question to answer, not a default one. While `run.agent` was
        untyped each tool read `.id` off it directly, which made the caller-less
        call an `AttributeError` raised from inside a body rather than a refusal
        the program can handle.
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
            # The guard has already refused this; reaching here means the guard
            # was not mounted, and the message must still not be delivered.
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
            # A settled child kept its session, its log and its roster row; what
            # it lost was the agent that holds an inbox. Waking it is the seam's
            # job (P3-13), and addressing one is exactly the trigger the plan
            # names — so a send to a finished child works rather than telling the
            # sender to go read a transcript.
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
                # The id the receipt names and the text shows, so the receiver's log
                # can be searched for it (`reconcile_send`, L6b).
                message_id=message_id,
            )
        )
        # **On the receiver's disk before the sender is told** (L6). `steer` records
        # the message in the receiver's log, and that record used to wait for the
        # receiver's next flush, which for an agent in a long tool call comes after
        # the sender's own. A crash in between left the sender's log saying
        # `delivered` and the receiver's saying nothing. Written first, the worst a
        # crash leaves is the sender's call unresolved, which repair reports as
        # outcome unknown. `written`, not `flush`, for `_mutate`'s reason: the
        # message is in the receiver's inbox already, so a write that fails must
        # not tell the sender it was not sent.
        if target.session is not None:
            await session_written(ctx, target.session)
        # A child answering its parent is a reply, which is what decides whether
        # the parent gets a "finished without replying" notice.
        children = ctx.get(RLM_CHILDREN)
        if children is not None and sender_role == "child":
            children.mark_replied(sender_id)
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
        """A bounded read of another family member's transcript.

        Reach-limited by the same rule as a send: an agent that may not talk to
        another may not read it either, and a bounded read is offloadable like
        any other large result rather than arriving as cell output (C5).
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
        """A log from the store, `()` when it has none, `None` when it could not be
        read. Blocking: a whole log is read and parsed."""
        persistence = ctx.get(SESSION_PERSISTENCE)
        if persistence is None:
            return None
        try:
            if not persistence.exists(session_id):
                return ()
            return persistence.read(session_id)[1]
        except Exception:
            log.warning("ph_rlm.messaging: could not read %s to reconcile a send", session_id)
            return None

    async def log_of(session_id: str) -> Sequence[SessionEvent] | None:
        """A log as a resume finds it: live, else stored, else empty. The stored read
        runs off the event loop, since a daemon resumes its roots concurrently."""
        live = ctx.require(SESSIONS).get(session_id)
        if live is not None:
            return live.events
        return await anyio.to_thread.run_sync(stored_log, session_id)

    async def candidates(sender: Session, role: str) -> tuple[list[str], bool] | None:
        """Every session a send in `role` could have reached, and whether that is all
        of them. `None` when the list itself could not be read.

        Not the send's own resolution: that reads the live family, and after a
        restart most of it is not live yet. A parent is named in the sender's header,
        children in its own roster, and a child's siblings in its parent's. A root's
        siblings are the other roots, which only the live ones stand for, so a root's
        sibling send can be found but never ruled out.
        """
        parent_id = sender.header.parent_session
        if role == "parent":
            return ([parent_id] if parent_id else []), True
        if role == "child":
            children = subagent_roster(sender).values()
            return [as_str(row.get("sessionId")) for row in children], True
        if parent_id is None:
            live = ctx.require(SESSIONS).list()
            return [one.id for one in live if one.header.parent_session is None], False
        parent_log = await log_of(parent_id)
        if parent_log is None:
            return None
        siblings = roster_of(parent_log).values()
        return [as_str(row.get("sessionId")) for row in siblings], True

    async def reconcile_send(
        arguments: Any,  # noqa: ANN401
        opened: SessionEvent,
        session: Session,
    ) -> Reconciled:
        """Did this send reach its receiver's log? Asked on resume (L6b).

        The message's id is derived from the call (`relay_message_id`), and the
        receiver's log is written before the sender is told (L6), so the message is
        in a receiver's log exactly when the send happened. Found, the send is done,
        and the model is shown a receipt rather than told to decide whether to send
        again. Looked for in every session the send could have reached and absent
        from all of them, it did not happen. Any log that could not be read, or a
        list of receivers that is not everyone, leaves it unknown.
        """
        call_id = as_str(opened.data.get("subCallId")) or as_str(opened.data.get("callId"))
        try:
            asked = SendArgs.model_validate(arguments)
        except ValidationError:
            return Unknown()
        found = await candidates(session, asked.receiver_role) if call_id else None
        if found is None:
            return Unknown()
        wanted = relay_message_id(f"{session.id}/{call_id}")
        targets, complete = found
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
        # Restricted away for this agent: absent from the SDK block too, so the
        # prompt cannot offer what a cell could not call.
        if (definition := view.visible.get(tool_name)) is not None
    ]
    return CodeBindingNamespace(name=name, description=description, bindings=tuple(bindings))


def _named(ctx: Context, agent_id: str, wanted: str) -> bool:
    return _display(ctx, agent_id) == wanted or agent_id == wanted


def _display(ctx: Context, agent_id: str) -> str:
    """The roster name of an agent, or its id when nobody named it.

    One line, because the lookup lives in the seam that owns the fold: the prompt
    needs the same answer, and two copies is how a prompt names one agent while a
    send delivers to another.
    """
    return str(ctx.require(SUBAGENTS).name_of(ctx.require(SESSIONS).list(), agent_id))


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


def _render_receipt(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    pending = value.get("pending")
    return text_content(
        f"{value['deliveryStatus']} to {value['receiverId']} ({value['receiverRole']})"
        + ("" if pending is None else f"; {pending} pending")
    )


_RELAY_IDS = uuid.uuid5(uuid.NAMESPACE_URL, "ph_rlm.messaging")


def relay_message_id(call_key: str) -> str:
    """The id of the message a send relays, derived from the call that sent it (L6b).

    `ToolRunContext.idempotency_key` — `{session}/{call id}`, a Code Mode dispatch's
    sub-call id included — is unique to one call and can be rebuilt from the call's
    record, so the send's `reconcile` can look for the message in the receiver's
    log after a crash. A UUID, the shape `new_message_id` mints.
    """
    return str(uuid.uuid5(_RELAY_IDS, call_key))


def _carries(event: SessionEvent, message_id: str) -> bool:
    """Whether a record put this message in its session's inbox."""
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
