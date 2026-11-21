"""How many tokens a text is, for a caller with no tokenizer to ask.

This is the careful estimate. It sizes an input for a server that refuses an
input that is too long: the passages of text-index for an embeddings endpoint,
and the state that a decision model is asked about. Careful means that the
estimate is seldom under. So it counts UTF-8 bytes and not characters. A
character that is three bytes long is about one token, and an estimate in
characters sends three times what fits in such a script.

**This is not the estimate of the meter.**
`ph.seams.token_meter.CHARS_PER_TOKEN` answers another question: about how full
a context is, for a threshold that only asks for a compaction sooner or later.
That figure is a typical case in characters, as the figure of its upstream is.
Neither should take the number of the other.

@module ph.llm.tokens
"""

from __future__ import annotations

__all__ = ["BYTES_PER_TOKEN", "bytes_of", "tokens_of"]

BYTES_PER_TOKEN = 3
"""How many bytes of text one token is taken to hold, on the careful side.
English prose runs near four characters to a token, and code nearer three. A
script with three bytes to a character is about one token to a character. So
bytes divided by three is seldom under."""


def tokens_of(text: str, bytes_per_token: float = BYTES_PER_TOKEN) -> float:
    """About how many tokens `text` is: its UTF-8 bytes divided by
    `bytes_per_token`. The result is not rounded, so the weights of the parts of
    a text add up to the weight of the whole. `bytes_per_token` is
    `BYTES_PER_TOKEN` unless the caller has learned how dense its text runs."""
    return len(text.encode("utf-8")) / bytes_per_token


def bytes_of(tokens: int) -> int:
    """How many UTF-8 bytes of text `tokens` tokens are taken to hold. It is
    `tokens_of` in the other direction, for a caller that sizes an input to a
    limit."""
    return tokens * BYTES_PER_TOKEN
