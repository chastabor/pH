"""Decoding frames from a hostile peer (C10).

The child runs model-written code, and that code has the channel's descriptor.
It can write anything at all onto fd 3 — a frame with extra fields, a reply id
that is a string, a `done` for a run that never started, a number large enough
to lose precision in a reader downstream. So the host **rebuilds** every inbound
frame from a declared spec rather than parsing it into a model:

* only the fields `INBOUND` declares are copied, so a forged field cannot ride
  along into a handler that reads `frame.get(...)`;
* every field must have its declared shape, so a non-numeric id is never
  *echoed* — the frame is dropped before there is an id to echo;
* anything malformed becomes `None`, never a raise. A handler that can raise on
  a forged frame is a handler the child can crash on demand.

The unsafe-integer rule is dsh's, and it applies **where the host reads a
number** — the `int` fields `INBOUND` declares (`id`, `protocol`) — through the
same `_coerce` every other shape rule goes through. Past 2^53 a JavaScript reader
loses precision silently (Q2), and an `id` the host would echo back is exactly the
number that must survive that reader.

**It was a whole-frame veto once, and that was a defect.** A `parse_int` hook on
the decoder raised on any integer past the bound *anywhere* in the line — inside
`boot-ack.limits`, which the spec declares `obj` and does not inspect; inside
`done.value`, declared `any`. So a payload the codec had explicitly declined to
validate could still make it drop the frame, and it did: on macOS the guest
reported `RLIM_INFINITY` for a limit the platform refused, every `boot-ack` was
dropped as junk, and every kernel start waited out its timeout in silence
(2026-09-07). A cell ending in `2**60` had the same fate on every platform. The
frame is now judged by its declared fields and nothing else — the module's own
rule, finally applied to the one check that had been exempt from it. What
reaches the *log* is guarded there, by `ph.session.json`, with the offending
path in the message, which is the diagnosis this bug never got.

@module ph_rlm.kernel.codec
"""

from __future__ import annotations

import json
from typing import Any, Final, cast

from ph.json import JSON_MAX_SAFE_INTEGER
from ph.wire import WireModel

from .protocol import INBOUND, FieldKind, InboundFrame

__all__ = ["decode", "encode"]

_INVALID: Final = object()


_DECODER: Final = json.JSONDecoder()
"""Built once, and **hookless**.

Two separate reasons, and dropping the object when the hook went would have lost
the first. *Built once*: `json.loads` re-dispatches per call — a wrapper frame plus
the str/bytes sniff — which measured **0.406 µs against 0.334 µs** for this object
on a `log` frame, the most numerous frame on the stdout path, and `decode` runs on
every frame the guest sends. *Hookless*: the `parse_int` guard that used to live
here is now `_coerce`'s, for the reason the module docstring gives.
"""


def _decode_json(raw: str | bytes) -> Any:  # noqa: ANN401
    # Decoded to `str` here rather than handing bytes to the decoder: measured
    # 0.49 µs against 0.65 µs on an 84-byte frame, because the bytes path sniffs
    # the encoding before doing exactly this.
    return _DECODER.decode(raw if isinstance(raw, str) else raw.decode("utf-8"))


def _coerce(value: Any, kind: FieldKind) -> Any:  # noqa: ANN401
    if kind == "any":
        return value
    if kind == "int":
        # `bool` is an `int` in Python and would sail through a bare isinstance;
        # and an int a JS reader cannot hold is not one the host may echo (Q2).
        numeric = isinstance(value, int) and not isinstance(value, bool)
        return value if numeric and abs(value) <= JSON_MAX_SAFE_INTEGER else _INVALID
    if kind == "str":
        return value if isinstance(value, str) else _INVALID
    if kind == "bool":
        return value if isinstance(value, bool) else _INVALID
    if kind == "obj":
        return value if isinstance(value, dict) else _INVALID
    return value if isinstance(value, list) else _INVALID


def decode(raw: str | bytes) -> InboundFrame | None:
    """One inbound frame, rebuilt from its spec, or `None` if it is not one.

    Typed as the `TypedDict` its spec was derived from, and the `cast` at the
    end is where that claim is made: the loop below copies exactly the keys the
    type names, each coerced to the type's declared shape or the frame refused,
    so what is returned *is* the type — by construction of the spec, not by a
    check the type system could see.
    """
    try:
        frame = _decode_json(raw)
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
    """One outbound frame as a line. `to_wire()` is pH's camelCase dump rule.

    `default=repr` is the one place a value that is not JSON is handled: a tool
    result reaches `ReplyFrame.value` as whatever the tool returned, and this is
    the single point at which the frame becomes bytes — so it is also the only
    place that needs to cope. Normalizing earlier meant serializing twice.
    """
    return json.dumps(frame.to_wire(), separators=(",", ":"), default=repr).encode("utf-8") + b"\n"
