"""Decoding frames from a hostile peer (C10).

The child runs code that the model wrote, and that code has the descriptor of
the channel. It can write anything at all to fd 3: a frame with extra fields, a
reply id that is a string, a `done` for a run that never started, a number large
enough to lose precision in a downstream reader. The host therefore **rebuilds**
every inbound frame from a declared spec, and does not parse it into a model:

* Only the fields that `INBOUND` declares are copied. A forged field therefore
  cannot reach a handler that reads `frame.get(...)`.
* Every field must have its declared shape. A non-numeric id is therefore never
  *echoed*, because the frame is dropped before there is an id to echo.
* Anything malformed becomes `None` and never raises. A handler that can raise
  on a forged frame is a handler that the child can crash on demand.

The unsafe-integer rule comes from dsh. It applies **where the host reads a
number**: the `int` fields that `INBOUND` declares (`id`, `protocol`, `run`,
`dropped`). It goes through the same `_coerce` as every other shape rule. Past
2^53 a JavaScript reader loses precision silently (Q2), and an `id` that the
host would echo back is exactly the number that must survive that reader.

**The rule was a veto on the whole frame before, and that was a defect.** A
`parse_int` hook on the decoder raised on any integer past the bound *anywhere*
in the line. That included `boot-ack.limits`, which the spec declares `obj` and
does not inspect, and `done.value`, which the spec declares `any`. A payload
that the codec had explicitly declined to validate could therefore still make
the codec drop the frame, and it did. On macOS the guest reported
`RLIM_INFINITY` for a limit that the platform refused. Every `boot-ack` was
dropped as junk, and every kernel start waited for its timeout with no message
(2026-09-07). A cell that ended in `2**60` failed the same way on every
platform. The frame is now judged by its declared fields and nothing else. That
is the rule of this module, now applied to the one check that had been exempt
from it. What reaches the *log* is guarded there, by `ph.session.json`, with the
offending path in the message. That is the diagnosis that this bug never got.

@module ph_rlm.kernel.codec
"""

from __future__ import annotations

import json
from typing import Any, Final, cast

from ph.json import JSON_MAX_SAFE_INTEGER, JsonValue, loads
from ph.wire import WireModel

from .protocol import INBOUND, FieldKind, InboundFrame

__all__ = ["decode", "encode"]


class _Invalid:
    """The marker for "this field failed its declared shape".

    It is a class and not a bare `object()`, so that `_coerce` can *state* what
    it returns: a JSON value or this marker. A bare object would erase both into
    `Any`.
    """

    __slots__ = ()


_INVALID: Final = _Invalid()


def _coerce(value: JsonValue, kind: FieldKind) -> JsonValue | _Invalid:
    if kind == "any":
        return value
    if kind == "int":
        # `bool` is an `int` in Python and would pass a bare `isinstance` check.
        # An int that a JS reader cannot hold is not one that the host may echo
        # (Q2).
        if isinstance(value, bool) or not isinstance(value, int):
            return _INVALID
        return value if abs(value) <= JSON_MAX_SAFE_INTEGER else _INVALID
    if kind == "str":
        return value if isinstance(value, str) else _INVALID
    if kind == "bool":
        return value if isinstance(value, bool) else _INVALID
    if kind == "obj":
        return value if isinstance(value, dict) else _INVALID
    return value if isinstance(value, list) else _INVALID


def decode(raw: str | bytes) -> InboundFrame | None:
    """One inbound frame, rebuilt from its spec, or `None` if it is not a frame.

    The return type is the `TypedDict` that the spec was derived from, and the
    `cast` at the end is where that claim is made. The loop below copies exactly
    the keys that the type names, and coerces each one to the declared shape of
    the type. If a required key fails, the frame is refused. An optional key
    that fails is left out. What is returned therefore *is* the type. That holds
    by the construction of the spec, and not by a check that the type system
    could see.
    """
    try:
        frame = loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(frame, dict):
        return None
    kind = frame.get("type")
    if not isinstance(kind, str):
        return None
    specs = INBOUND.get(kind)
    if specs is None:
        return None
    rebuilt: dict[str, Any] = {}
    for spec in specs:
        if spec.name not in frame:
            if spec.required:
                return None
            continue
        coerced = _coerce(frame[spec.name], spec.kind)
        if coerced is _INVALID:
            if spec.required:
                return None
            continue
        rebuilt[spec.name] = coerced
    return cast("InboundFrame", rebuilt)


def encode(frame: WireModel) -> bytes:
    """One outbound frame as a line. `to_wire()` is the camelCase dump rule of pH.

    `default=repr` is the one place that handles a value that is not JSON. A
    tool result reaches `ReplyFrame.value` as whatever the tool returned, and
    this is the single point where the frame becomes bytes. It is therefore also
    the only place that must handle such a value. Normalizing earlier meant
    serializing twice.
    """
    return json.dumps(frame.to_wire(), separators=(",", ":"), default=repr).encode("utf-8") + b"\n"
