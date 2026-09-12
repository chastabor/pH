"""The narrowings `ph_runtime` needs from `ph.json` — a copy, pinned by a test.

This package ships into the guest venv with `dill` as its only dependency, so it
cannot import `ph.json`: that module lives in the ph-core wheel, and depending on
the wheel is the thing this package exists not to do.

**Copied by hand, compared character for character by
`test_protocol_mirror.py`.** What keeps the two in step is the test rather than
anyone's memory — that file opens by recording that *"a third copy that no test
compared had already drifted"*. When it fails, copy the definition across again;
when the guest comes to need a second narrowing, paste that one in and the test
picks it up by name.

The arrangement `truncation_marker` already has in `ph_runtime.protocol` — the
other thing this package copies rather than imports — though that one is held to
ph-core's by its output and this by its source, because a marker is a sentence
built from two numbers and a narrowing is its text.

Private, and re-exported by `ph_runtime.protocol` for the names the guest uses:
this is ph-core's code, not the guest's API.

@module ph_runtime._json
"""

from __future__ import annotations


def as_str(value: object, default: str = "") -> str:
    """A JSON string, or `default` — the fourth of the family, and the copied one.

    One policy, the one `as_int`'s docstring argues for: a mis-shaped field
    answers with the empty value rather than raising, because a reader of a log
    some other build wrote must lose a row and not a session.

    **Not `str(value)`**, which is what a reader writes without this and is worse
    than useless: `str(None)` is `"None"` and `str(3)` is `"3"`, so a field that
    is absent or of the wrong type comes back as a plausible-looking answer that
    no assertion catches. Narrowing says "this was not a string" by giving back
    nothing, which is the same thing `as_obj` and `as_seq` say.
    """
    return value if isinstance(value, str) else default
