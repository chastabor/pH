"""The narrowings that `ph_runtime` needs from `ph.json`: a copy, held equal by a test.

This package ships into the guest venv with `dill` as its only dependency, so it
cannot import `ph.json`. That module is in the ph-core wheel, and this package
exists to not depend on that wheel.

**The definitions are copied by hand. `test_protocol_mirror.py` compares them
character for character.** The test keeps the two copies equal, so nobody has to
remember to do it. That test file starts by recording that *"a third copy that no
test compared had already drifted"*. When the test fails, copy the definition
across again. When the guest needs a second narrowing, paste it in, and the test
finds it by name.

`truncation_marker` in `ph_runtime.protocol` has a similar arrangement. It is the
other thing that this package copies and does not import. The test compares that
function with the function of ph-core by its output, and compares this module by
its source. The reason is that a marker is a sentence built from two numbers, and
a narrowing is its text.

This module is private. `ph_runtime.protocol` re-exports the names that the guest
uses. This is the code of ph-core, not the API of the guest.

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
