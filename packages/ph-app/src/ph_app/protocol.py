"""The wire vocabulary both transports speak (P5-02, I-7).

pH answers on two: `--mode rpc` over stdio, for a caller that owns the process,
and `$PH_RUNTIME/daemon.sock`, for one that does not. They are the same
protocol. P5-01 shipped them as two — the daemon grew `root/*` methods and
dropped the `"jsonrpc": "2.0"` field the RPC mode sends — and the divergence was
invisible because each transport only ever tested itself.

**The names are dsh's**, deliberately: `initialize`, `session/prompt`,
`session.event`, `session.status`. dsh already ships a Python client for this
shape, and a second vocabulary would make "use the client you have" false in
exactly the deployment Phase 5 exists for.

**Envelope here, methods there.** What is genuinely transport-independent is the
request/reply/error shaping and the version; what a transport serves — one
session over a pipe, or many supervised roots over a socket — is its own. So
this module owns `respond`, `notify` and the capability block, and each server
owns its method table.

@module ph_app.protocol
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "PROTOCOL_VERSION",
    "SNAPSHOT_EVENTS",
    "DaemonError",
    "DaemonGone",
    "Dispatch",
    "Refusal",
    "capabilities",
    "cursor_of",
    "notification",
    "request",
    "respond",
    "resume_at",
]

PROTOCOL_VERSION = 1
"""One number, in one place.

It was declared twice — once per transport — which is how two servers come to
claim the same version for two different vocabularies.
"""

SNAPSHOT_EVENTS = 2048
"""How many events one `session/snapshot` reply carries.

A resumed root can hold hundreds of thousands of events, and a client that asked
for its history should not be handed a frame that trips the transport's own
`MAX_LINE` — or a reply it must buffer whole before rendering a line. The cursor
in each reply is what asks for the next page.

**A count, not a byte budget.** The first draft measured each event with its own
`dumps` to fill a 512 KiB page, which cost 8.2 ms per page — *2.2x the encode it
existed to bound*, and all of it discarded. A count needs no measuring pass, and
the transport's `MAX_LINE` is the real protection against an oversized frame.
"""

Dispatch = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""A server's method table: `(method, params) -> result`, raising to refuse."""


class Refusal(Exception):
    """A refusal that names itself, for a server to raise.

    `code` is a class attribute here because these refusals are *kinds* — an
    unknown method is one thing whatever the method was. An error that computes
    its code per instance sets `self.code` instead, which is what `ph-core`'s
    coded errors do (`HarnessError`, `SessionForkError`, `CompactionError`) and
    what `respond` reads, so both shapes reach the wire through one path.

    Declared rather than duck-typed, for the reason the seams give for their
    provider Protocols: a probe finds whatever happens to be called `code`, and
    an error whose attribute drifted would go out unnamed with nothing to say
    so.
    """

    code = ""


class DaemonError(RuntimeError):
    """A refusal the server sent back, with its name where it had one.

    The other end of `respond`'s `data.reason`. Without it a client that wanted
    to tell `session_already_active` (I-5) from a mistyped method had to match
    on the message text — which is a contract nobody wrote down and every
    rewording breaks. `reason` is `""` when the server did not name one.
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason

    @classmethod
    def of(cls, error: dict[str, Any]) -> DaemonError:
        """Rebuild the refusal from an error frame."""
        data = error.get("data")
        reason = data.get("reason", "") if isinstance(data, dict) else ""
        return cls(str(error.get("message", "the daemon refused")), str(reason))


class DaemonGone(DaemonError):
    """The connection ended without an answer. Nobody refused anything.

    A subclass, so a caller that only wants "the call did not succeed" still
    catches `DaemonError` — and a distinct type, because the two are opposite
    diagnoses and a client renders them differently. Reported as a refusal, the
    message read "the daemon refused: the daemon closed the connection", which
    is precisely the confusion `ph agents`' absent-socket / stale-socket split
    exists to prevent.
    """

    def __init__(self, message: str = "the daemon closed the connection") -> None:
        super().__init__(message, "connection_closed")


def capabilities(*names: str) -> dict[str, Any]:
    """The `initialize` reply, with whatever this transport adds.

    `sessions` and `streaming` are true of both; a transport that supervises
    adds `roots`, `attach`, `cursors`, `snapshots`, and one that does not simply
    omits them — a client reads the block rather than inferring from which
    socket it happened to open. Names rather than `**kwargs`, because every
    value is `True` by construction: a capability is present or absent.
    """
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"sessions": True, "streaming": True, **dict.fromkeys(names, True)},
    }


def notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """A frame with no id, which is what makes it a notification."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


def request(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """A frame that expects a reply. The client's half of the envelope."""
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def cursor_of(session: Any, sequence: int | None = None) -> dict[str, Any]:
    """Where a reader has got to, as `{generation, sequence}`.

    A sequence alone is only meaningful against the log that counted it, so it
    travels with the identity of that log. `generation` is the header's
    `createdAt`: durable, already on the wire, stable across a resume — which
    continues the same log — and different for anything that is not that log.

    Here rather than on the daemon's `Root`, because it is a fact about a
    *session*: the stdio transport serves the same protocol and would otherwise
    have to re-derive it.
    """
    return {
        "generation": str(session.header.created_at),
        "sequence": session.seq if sequence is None else sequence,
    }


def resume_at(session: Any, cursor: Any) -> int:
    """The index a cursor asks to resume from, or 0 when it cannot say.

    A cursor from another incarnation of the log is neither honoured nor
    refused: honouring it would skip events the client never saw, refusing it
    would strand a client that did nothing wrong. So a stale generation reads as
    "you have seen nothing of *this* log" — the only safe reading of the two,
    and the reply says where it actually started so the client is not left
    inferring it from sequence numbers.
    """
    if not isinstance(cursor, dict):
        return 0
    if str(cursor.get("generation", "")) != str(session.header.created_at):
        return 0
    seq: int = session.seq
    return max(0, min(int(cursor.get("sequence", 0)), seq))


async def respond(request_frame: dict[str, Any], dispatch: Dispatch) -> dict[str, Any] | None:
    """Run one request and shape its reply, or `None` if it wanted none.

    A failing method is *this call's* failure, not the connection's: an unknown
    method or a bad argument comes back as an error frame and the peer keeps
    talking. Framing errors are the transport's and end the stream, because
    after a bad frame there is no way to know where the next one starts.
    """
    request_id = request_frame.get("id")
    method = str(request_frame.get("method", ""))
    params = request_frame.get("params") or {}
    try:
        body: dict[str, Any] = {"result": await dispatch(method, params)}
    except Exception as error:
        failure: dict[str, Any] = {"code": -32000, "message": str(error)}
        # A refusal a client is expected to *branch* on carries a name rather
        # than making the client match message text: `session_already_active`
        # (I-5) is a thing to retry elsewhere, and telling it from a typo in a
        # method name should not mean grepping prose. JSON-RPC's own code stays
        # generic, because these are pH's vocabulary and not the transport's.
        #
        # Read off the *instance*, which is what already carries a code here —
        # `HarnessError`, `SessionForkError` and friends all set `self.code` in
        # `__init__`. The first draft read `type(error)`, so those four families
        # were structurally invisible to it and a `SessionForkError` out of
        # `sessions.create` reached the client unnamed on this very path.
        reason = getattr(error, "code", "")
        if isinstance(reason, str) and reason:
            failure["data"] = {"reason": reason}
        body = {"error": failure}
    return None if request_id is None else {"jsonrpc": "2.0", "id": request_id, **body}
