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

The unsafe-integer rule is dsh's, implemented better than a text scan can
manage: `json.loads(..., parse_int=...)` sees each integer *as JSON parses it*,
so digits inside a string are not mistaken for a number and a number inside a
nested object is still checked. The rule matters because pH's log is JSON that
dsh's TypeScript tooling reads (Q2), and an integer past 2^53 silently loses
precision there.

@module ph_rlm.kernel.codec
"""

from __future__ import annotations

import json
from typing import Any, Final

from ph.session.json import JSON_MAX_SAFE_INTEGER
from ph.wire import WireModel

from .protocol import INBOUND, FieldKind

__all__ = ["UnsafeInteger", "decode", "encode", "has_unsafe_integer"]

_INVALID: Final = object()


class UnsafeInteger(ValueError):
    """A JSON integer too large to survive a JavaScript reader."""


def _guard_int(token: str) -> int:
    value = int(token)
    if abs(value) > JSON_MAX_SAFE_INTEGER:
        raise UnsafeInteger(token)
    return value


_DECODER: Final = json.JSONDecoder(parse_int=_guard_int)
"""Built once. A `parse_int` keyword bypasses `json.loads`'s cached decoder, so
every frame was constructing a fresh `JSONDecoder` and scanner — ~3 µs of the
~9 µs `decode` spent per frame."""


def has_unsafe_integer(raw: str | bytes) -> bool:
    """Whether `raw` carries an integer no JS reader could hold losslessly."""
    try:
        _decode_json(raw)
    except UnsafeInteger:
        return True
    except (ValueError, UnicodeDecodeError):
        return False
    return False


def _decode_json(raw: str | bytes) -> Any:
    return _DECODER.decode(raw if isinstance(raw, str) else raw.decode("utf-8"))


def _coerce(value: Any, kind: FieldKind) -> Any:
    if kind == "any":
        return value
    if kind == "int":
        # `bool` is an `int` in Python and would sail through a bare isinstance.
        return value if isinstance(value, int) and not isinstance(value, bool) else _INVALID
    if kind == "str":
        return value if isinstance(value, str) else _INVALID
    if kind == "bool":
        return value if isinstance(value, bool) else _INVALID
    if kind == "obj":
        return value if isinstance(value, dict) else _INVALID
    return value if isinstance(value, list) else _INVALID


def decode(raw: str | bytes) -> dict[str, Any] | None:
    """One inbound frame, rebuilt from its spec, or `None` if it is not one."""
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
    return rebuilt


def encode(frame: WireModel) -> bytes:
    """One outbound frame as a line. `to_wire()` is pH's camelCase dump rule.

    `default=repr` is the one place a value that is not JSON is handled: a tool
    result reaches `ReplyFrame.value` as whatever the tool returned, and this is
    the single point at which the frame becomes bytes — so it is also the only
    place that needs to cope. Normalizing earlier meant serializing twice.
    """
    return json.dumps(frame.to_wire(), separators=(",", ":"), default=repr).encode("utf-8") + b"\n"
